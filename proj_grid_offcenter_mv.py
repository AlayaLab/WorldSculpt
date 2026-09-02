"""proj_grid_offcenter_mv.py — off-center ProjGrid + the FINETUNED multi-view model.

Same off-center projection as proj_grid_offcenter.py (real camera pose + real
intrinsics K, arbitrary principal point), but the per-view proj features are
fused with the trained ControlNet-style ZERO-INIT aggregator instead of a plain
mean, and the denoisers are the LoRA-finetuned multi-view models (SS,
shape-1024, tex-1024). LR-512 is skipped (not finetuned): the SS occupancy is
decoded directly at grid 64 (== shape/tex coord grid), so the whole chain stays
on the finetuned MV models.

Reuses proj_grid_offcenter.enable_offcenter_projgrid (the ProjGrid.forward patch)
and mv_inference_common for building the LoRA denoisers + aggregators.
"""
from typing import List
import numpy as np
import torch

from pixal3d.modules.sparse import SparseTensor
from proj_grid_offcenter import enable_offcenter_projgrid, rgba_over_black, R_CUBE2GRID


@torch.no_grad()
def aggregate_offcenter_mv(image_cond_model, aggregator, images_rgb, poses_grid, K_norms,
                           mesh_scale=1.0, grid_resolution_override=None, device='cuda',
                           coords=None):
    """Per-view off-center proj features, fused by the trained MV aggregator.

    Mirrors proj_grid_offcenter.aggregate_offcenter but replaces the mean fusion
    with the checkpoint's aggregator (== training MV aggregation, view 0 = anchor):
      - ZeroInitMVAggregator: anchor + zero_proj(mean others), 4-arg call.
      - IBRMVAggregator: per-view features kept apart, per-voxel softmax-weighted
        residual on the mean; global token = anchor-only. When `coords` is given
        (sparse stages), each view is gathered at the active voxels INSIDE the
        loop (a dense grid-64 z_proj is ~2 GB/view; stacking K of them would OOM)
        and the returned z_proj is ALREADY sparse (N, C).
      - None: plain mean over all views.
    Returns (z_global_agg, z_proj_agg).
    """
    from pixal3d_multiview.mv_aggregator import IBRMVAggregator
    is_ibr = isinstance(aggregator, IBRMVAggregator)
    orig_grid_res = image_cond_model.grid_resolution
    if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
        image_cond_model.grid_resolution = grid_resolution_override
        image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
            grid_resolution=grid_resolution_override,
            image_resolution=image_cond_model.proj_grid.image_resolution,
        ).to(device)

    pg = image_cond_model.proj_grid
    dummy = torch.tensor([0.5], device=device)
    scale_t = torch.tensor([mesh_scale], device=device)

    # Accumulate the "others" mean incrementally (one running sum) instead of
    # stacking every view's z_proj — at grid 64 each z_proj is ~2 GB, so stacking
    # all views (e.g. --views all = 30) would OOM.
    z_global = None
    z_anchor = None
    others_sum = None
    g_others_sum = None
    n_others = 0
    per_view_feats = []               # ibr only: per-view (V, C) or coords-gathered (N, C), bf16
    if is_ibr and coords is not None:
        _gr = (grid_resolution_override if grid_resolution_override is not None
               else orig_grid_res)
        _bi = coords[:, 0].long(); _xi = coords[:, 1].long()
        _yi = coords[:, 2].long(); _zi = coords[:, 3].long()
    for i, (img, pose, K) in enumerate(zip(images_rgb, poses_grid, K_norms)):
        pg._oc_K_norm = K.to(device)
        pg._oc_pose = pose.to(device)
        z_g, z_p = image_cond_model(
            [img], camera_angle_x=dummy, distance=dummy, mesh_scale=scale_t,
            transform_matrix=None)
        if is_ibr:
            # Keep this view apart (bf16). Global: anchor-only, nothing else needed.
            if i == 0:
                z_global = z_g
            if coords is not None:
                z_p = z_p.reshape(1, _gr, _gr, _gr, -1)[_bi, _xi, _yi, _zi]  # (N, C)
            else:
                z_p = z_p[0]                                                 # (V, C)
            per_view_feats.append(z_p.to(torch.bfloat16))
            continue
        if i == 0:
            z_anchor = z_p            # view 0 = anchor (kept)
            z_global = z_g            # view 0 = anchor global token
        else:
            if others_sum is None:
                others_sum = z_p
            else:
                others_sum.add_(z_p)  # in-place: no growing list / stack
            # global token is tiny (vs the ~2GB per-voxel z_p) — accumulate it too
            # so the aggregator's trained zero_proj_global branch runs exactly as in
            # training (encode_image_proj). Skipping it would ignore the global ControlNet.
            if g_others_sum is None:
                g_others_sum = z_g.clone()
            else:
                g_others_sum.add_(z_g)
            n_others += 1
    pg._oc_K_norm = None
    pg._oc_pose = None

    if is_ibr:
        # IBR fusion (matches training encode_image_proj's ibr branch): per-voxel
        # learned view weighting; at zero-init the output == the plain masked mean.
        # Global token = anchor-only (no averaging), exactly as in training.
        feats = torch.stack(per_view_feats, dim=-2)          # (N, K, C) or (V, K, C)
        vmask = torch.ones(feats.shape[0], feats.shape[1],
                           dtype=torch.bool, device=feats.device)
        z_agg = aggregator(feats, vmask)                     # (N|V, C) fp32
        # Dense callers (SS) expect a batch dim; sparse callers get (N, C) directly.
        z_proj_agg = z_agg if coords is not None else z_agg.unsqueeze(0)
        z_global_agg = z_global
    elif aggregator is None:
        # Plain-mean fusion — matches training encode_image_proj's no-aggregator
        # branch (mean over ALL K views). Used by the `mv_lora_avg` finetunes whose
        # checkpoints have no mv_aggregator. Single view (n_others==0) -> anchor only.
        if n_others == 0:
            z_proj_agg, z_global_agg = z_anchor, z_global
        else:
            K = n_others + 1
            z_proj_agg   = (z_anchor + others_sum)   / K
            z_global_agg = (z_global + g_others_sum) / K
    else:
        # Zero-init aggregator fusion (matches training encode_image_proj):
        #   proj_agg   = z_anchor + zero_proj(mean others)
        #   global_agg = g_anchor + zero_proj_global(mean others' global tokens)
        others_mean = (others_sum / n_others) if n_others > 0 else torch.zeros_like(z_anchor)
        g_others_mean = (g_others_sum / n_others) if n_others > 0 else torch.zeros_like(z_global)
        z_proj_agg, z_global_agg = aggregator(z_anchor, others_mean, z_global, g_others_mean)

    if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
        image_cond_model.grid_resolution = orig_grid_res
        image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
            grid_resolution=orig_grid_res,
            image_resolution=image_cond_model.proj_grid.image_resolution,
        ).to(device)

    return z_global_agg, z_proj_agg


def build_offcenter_cond_ss_mv(pipeline, aggregator, images_rgb, poses_grid, K_norms, mesh_scale=1.0):
    device = pipeline.device
    icm = pipeline.image_cond_model_ss
    if pipeline.low_vram:
        icm.to(device)
    z_g, z_p = aggregate_offcenter_mv(icm, aggregator, images_rgb, poses_grid, K_norms,
                                      mesh_scale, device=device)
    if pipeline.low_vram:
        icm.cpu()
    return {
        'cond':     {'global': z_g, 'proj': z_p},
        'neg_cond': {'global': torch.zeros_like(z_g), 'proj': torch.zeros_like(z_p)},
    }


def build_offcenter_cond_shape_mv(pipeline, image_cond_model, aggregator, images_rgb,
                                  poses_grid, K_norms, coords, mesh_scale=1.0,
                                  grid_resolution_override=None):
    from pixal3d_multiview.mv_aggregator import IBRMVAggregator
    device = pipeline.device
    if pipeline.low_vram:
        image_cond_model.to(device)
    z_g, z_p = aggregate_offcenter_mv(
        image_cond_model, aggregator, images_rgb, poses_grid, K_norms, mesh_scale,
        grid_resolution_override=grid_resolution_override, device=device,
        coords=coords)

    if isinstance(aggregator, IBRMVAggregator):
        # IBR path already gathered per view at coords -> z_p is (N, C) sparse.
        z_proj_sparse = z_p
    else:
        grid_res = image_cond_model.grid_resolution
        z_proj_grid = z_p.reshape(1, grid_res, grid_res, grid_res, -1)
        b = coords[:, 0].long(); x = coords[:, 1].long(); y = coords[:, 2].long(); z = coords[:, 3].long()
        z_proj_sparse = z_proj_grid[b, x, y, z]
    z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)
    if pipeline.low_vram:
        image_cond_model.cpu()
    return {
        'cond':     {'global': z_g, 'proj': z_proj_st},
        'neg_cond': {'global': torch.zeros_like(z_g),
                     'proj':   SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords)},
    }


def _install_official_samplers(pipeline):
    """Swap in the OFFICIAL pretrained/Pixal3D/pipeline.json sampler config
    (verbatim; tuned for the BASE model): FlowEulerGuidanceIntervalSampler,
    12 steps + high CFG with rescale / guidance-interval / rescale_t per stage."""
    from pixal3d.pipelines import samplers
    cfg = {
        'ss':    dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
                      guidance_interval=(0.6, 1.0), rescale_t=5.0),
        'shape': dict(steps=12, guidance_strength=7.5, guidance_rescale=0.5,
                      guidance_interval=(0.6, 1.0), rescale_t=3.0),
        'tex':   dict(steps=12, guidance_strength=1.0, guidance_rescale=0.0,
                      guidance_interval=(0.6, 0.9), rescale_t=3.0),
    }
    pipeline.sparse_structure_sampler = samplers.FlowEulerGuidanceIntervalSampler(1e-5)
    pipeline.sparse_structure_sampler_params = dict(cfg['ss'])
    pipeline.shape_slat_sampler = samplers.FlowEulerGuidanceIntervalSampler(1e-5)
    pipeline.shape_slat_sampler_params = dict(cfg['shape'])
    pipeline.tex_slat_sampler = samplers.FlowEulerGuidanceIntervalSampler(1e-5)
    pipeline.tex_slat_sampler_params = dict(cfg['tex'])
    print(f"[OC-MV] OFFICIAL samplers (pipeline.json): SS={cfg['ss']}, "
          f"shape={cfg['shape']}, tex={cfg['tex']}")


def _install_train_samplers(pipeline):
    """Swap the pipeline's samplers for the ones the FINETUNED MV model was
    validated against (training run_snapshot): FlowEulerCfgSampler + plain CFG,
    SS=50 steps / gs=3.0, shape=12 / gs=3.0, tex=12 / gs=3.0 (mv_inference_common
    .TRAIN_PARAMS). The base pipeline.json ships FlowEulerGuidanceIntervalSampler
    (SS=12, gs=7.5, guidance-interval) which was tuned for the BASE model — wrong
    for the finetune. Called once before sampling."""
    from pixal3d.pipelines import samplers
    from mv_inference_common import TRAIN_PARAMS
    pipeline.sparse_structure_sampler = samplers.FlowEulerCfgSampler(1e-5)
    pipeline.sparse_structure_sampler_params = dict(TRAIN_PARAMS['ss'])
    pipeline.shape_slat_sampler = samplers.FlowEulerCfgSampler(1e-5)
    pipeline.shape_slat_sampler_params = dict(TRAIN_PARAMS['shape'])
    pipeline.tex_slat_sampler = samplers.FlowEulerCfgSampler(1e-5)
    pipeline.tex_slat_sampler_params = dict(TRAIN_PARAMS['tex'])
    print(f"[OC-MV] train samplers: SS={TRAIN_PARAMS['ss']}, "
          f"shape={TRAIN_PARAMS['shape']}, tex={TRAIN_PARAMS['tex']}")


@torch.no_grad()
def run_offcenter_mv_pipeline(pipeline, aggregators, frames, seed=42, resolution=1024,
                              max_num_tokens=49152, decode_tex=True, sampler='train'):
    """Off-center reconstruction with the finetuned MV model (ss64-direct, no LR cascade).

    pipeline   : Pixal3DImageTo3DPipeline whose ss / shape_1024 / tex_1024 flow models
                 have been swapped for the LoRA-finetuned MV denoisers (bf16).
    aggregators: {'ss','shape','tex'} ZeroInitMVAggregator modules. 'tex' may be omitted
                 when decode_tex is False.
    frames     : list of dicts {image (PIL RGBA), transform_matrix [4,4] canonical c2w,
                 K_norm [3,3] normalized OpenCV intrinsics}.
    decode_tex : if False, skip Stage 3 (texture) and decode geometry only — returns the
                 raw shape Mesh (vertices/faces, no texture attrs).
    """
    device = pipeline.device
    enable_offcenter_projgrid()
    # CRITICAL: sample the finetuned weights with the training-matched sampler,
    # not the base pipeline.json sampler (which is tuned for the base model).
    if sampler == 'train':
        _install_train_samplers(pipeline)
    elif sampler == 'official':
        _install_official_samplers(pipeline)
    else:
        print(f"[OC-MV] sampler=pipeline: using base pipeline.json samplers "
              f"(SS={pipeline.sparse_structure_sampler_params})")

    images_rgb = [rgba_over_black(f['image']) for f in frames]
    R = R_CUBE2GRID.to(device)
    poses_grid, K_norms = [], []
    for f in frames:
        c2w = torch.as_tensor(np.asarray(f['transform_matrix'], dtype=np.float32), device=device)
        poses_grid.append((R @ c2w).unsqueeze(0))
        K_norms.append(torch.as_tensor(np.asarray(f['K_norm'], dtype=np.float32), device=device).unsqueeze(0))

    print(f"[OC-MV] {len(frames)} view(s), off-center ProjGrid + per-stage MV fusion "
          f"(ibr / zero-init aggregator / plain mean, auto-detected per stage)")
    torch.manual_seed(seed)
    mesh_scale = 1.0
    grid_res = resolution // 16                       # 1024 -> 64

    # ---- Stage 1: Sparse Structure (MV) -> grid-64 coords (no LR cascade) ----
    print("[OC-MV] Stage 1: Sparse Structure ...")
    cond_ss = build_offcenter_cond_ss_mv(pipeline, aggregators['ss'], images_rgb,
                                         poses_grid, K_norms, mesh_scale)
    coords = pipeline.sample_sparse_structure(cond_ss, resolution=grid_res, num_samples=1,
                                              sampler_params={})
    del cond_ss; torch.cuda.empty_cache()
    print(f"[OC-MV]   coords: {coords.shape[0]} occupied voxels (grid {grid_res})")

    # ---- Stage 2: Shape HR (MV) on those coords ----
    print(f"[OC-MV] Stage 2: Shape @ {resolution} ...")
    cond_shape = build_offcenter_cond_shape_mv(
        pipeline, pipeline.image_cond_model_shape_1024, aggregators['shape'],
        images_rgb, poses_grid, K_norms, coords, mesh_scale)
    shape_slat = pipeline.sample_shape_slat(
        cond_shape, pipeline.models['shape_slat_flow_model_1024'], coords, {})
    del cond_shape; torch.cuda.empty_cache()

    # ---- Geometry-only: skip Stage 3, decode shape mesh and return ----
    if not decode_tex:
        print("[OC-MV] Skipping Stage 3 (texture); decoding geometry only ...")
        meshes, _subs = pipeline.decode_shape_slat(shape_slat, resolution)
        mesh = meshes[0]
        mesh.fill_holes()
        return mesh, resolution, poses_grid, K_norms, images_rgb, (coords, grid_res)

    # ---- Stage 3: Texture (MV) ----
    print(f"[OC-MV] Stage 3: Texture @ {resolution} ...")
    cond_tex = build_offcenter_cond_shape_mv(
        pipeline, pipeline.image_cond_model_tex_1024, aggregators['tex'],
        images_rgb, poses_grid, K_norms, shape_slat.coords, mesh_scale)
    tex_slat = pipeline.sample_tex_slat(
        cond_tex, pipeline.models['tex_slat_flow_model_1024'], shape_slat, {})
    del cond_tex; torch.cuda.empty_cache()

    # ---- Decode ----
    print("[OC-MV] Decoding mesh + texture ...")
    out_meshes = pipeline.decode_latent(shape_slat, tex_slat, resolution)
    return out_meshes[0], resolution, poses_grid, K_norms, images_rgb, (coords, grid_res)
