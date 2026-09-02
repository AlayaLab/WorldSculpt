"""
reconstruct_object.py — off-center Pixal3D with the FINETUNED MULTI-VIEW model.

Same as reconstruct_object.py (real pose + real off-center intrinsics K,
camera need not look at the cube center), but the SS / shape-1024 / tex-1024
denoisers are the LoRA-finetuned multi-view models and per-view proj features are
fused with the trained ControlNet-style zero-init aggregator (not a plain mean).
LR-512 is skipped (not finetuned): SS is decoded directly at grid 64.

Single-view (--views anchor) or multi-view (--views all / 0,7,14,...). The base
LR cascade is not used; the chain stays on the finetuned MV models.

OUTPUT (default <case_root>/_recon/<instance>/):
    mesh.pt          raw state (feeds step5_check_anchor_reproject.py --mesh_pt)
    mesh_world.glb   metric world via T_canon_to_metric

USAGE:
  python reconstruct_object.py \\
      --transforms_json assets/video/cup_and_tea/_crops/brown_teacup/transforms.json \\
      --views all
"""

import os
import sys
import json
import argparse
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault('ATTN_BACKEND', 'flash_attn')
os.environ['FLEX_GEMM_AUTOTUNE_CACHE_PATH'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Base (frozen) Pixal3D checkpoints, relative to this directory — README Step 3
# downloads them to exactly these paths.
SS_BASE    = 'pretrained/Pixal3D/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors'
SHAPE_BASE = 'pretrained/Pixal3D/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors'
TEX_BASE   = 'pretrained/Pixal3D/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors'

# The finetuned config / ckpt-dir paths have NO defaults on purpose: they must name
# the exact weights a run used, and a stale default would silently reconstruct with
# the wrong ones. Every stage's --*_config and --*_ckpt_dir is therefore required
# (texture only when Stage 3 actually runs, i.e. without --no_tex).


def remesh_geometry_glb(vertices, faces, grid_size, decimation_target=100_000,
                        band=1.0, project_back=0.0):
    """Geometry-only counterpart of `o_voxel.postprocess.to_glb(remesh=True)`.

    to_glb cannot be called with --no_tex: `attr_volume` / `coords` / `attr_layout`
    are required args and only Stage 3 (texture) produces them. So reproduce its
    GEOMETRY stages verbatim — same calls, same order, same constants —
        fill_holes(3e-2) -> remesh_narrow_band_dc(band, project_back) -> simplify
    — and skip only the UV-unwrap + texture-bake tail, which does not touch
    geometry. The exported surface is then identical to the textured path's.

    Frame: to_glb ends with an axis swap (x,y,z) -> (x, z, -y) (postprocess.py,
    "Swap Y and Z axes, invert Y"). We apply the SAME swap, so the caller's
    `rot_yup` / `R_g_inv` post-transforms stay valid for both branches.
    """
    import cumesh
    import trimesh
    V = vertices.detach().cuda().contiguous().float()
    F = faces.detach().cuda().contiguous().int()

    m = cumesh.CuMesh()
    m.init(V, F)
    m.fill_holes(max_hole_perimeter=3e-2)          # to_glb does this before both branches
    V, F = m.read()

    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        dtype=torch.float32, device=V.device)
    scale = (aabb[1] - aabb[0]).max().item()
    m2 = cumesh.CuMesh()
    m2.init(*cumesh.remeshing.remesh_narrow_band_dc(
        V, F, center=aabb.mean(dim=0),
        scale=(grid_size + 3 * band) / grid_size * scale,
        resolution=int(grid_size), band=band, project_back=project_back,
        verbose=False, bvh=cumesh.cuBVH(V, F)))
    m2.simplify(decimation_target)
    Vo, Fo = m2.read()

    v = Vo.detach().cpu().numpy().astype(np.float64)
    v[:, 1], v[:, 2] = v[:, 2].copy(), -v[:, 1].copy()     # match to_glb's axis swap
    return trimesh.Trimesh(v, Fo.detach().cpu().numpy(), process=False)


def K_image_to_norm(K_image_pix, image_size_px):
    K = np.asarray(K_image_pix, dtype=np.float64).copy()
    W = float(image_size_px)
    K[0, 0] /= W; K[0, 2] /= W
    K[1, 1] /= W; K[1, 2] /= W
    return K


def _swap_in_mv_denoiser(pipeline, model_key, config, base, ckpt_dir, step, ema, ema_rate):
    """Replace pipeline.models[model_key] with the finetuned MV (LoRA, bf16) denoiser.

    Returns its aggregator (zero-init OR ibr — the class is auto-detected from the
    ckpt's state-dict keys in mv.build_aggregator), or None when the checkpoint has
    NO aggregator (a plain-mean / `mv_lora_avg` finetune, i.e. mv_aggregator=None
    in the config) — in which case the pipeline fuses the K views by plain mean.
    Auto-detected from whether mv_aggregator_step*.pt exists, so ibr / zero-agg /
    avg stages can be mixed freely."""
    import mv_inference_common as mv
    cfg = mv.load_stage_config(config)
    t = cfg['trainer']['args']
    s = step if step >= 0 else mv.find_latest_step(ckpt_dir)
    pipeline.models[model_key] = mv.build_mv_denoiser(
        cfg['models']['denoiser'], t['lora_config'], base,
        mv.ckpt_path(ckpt_dir, 'denoiser', s, ema, ema_rate), dtype='bfloat16')
    agg = None
    agg_cfg = t.get('mv_aggregator')
    # Build the aggregator path directly (NOT via mv.ckpt_path, which raises if the
    # file is missing) so a plain-mean finetune (no mv_aggregator_*.pt) is detected,
    # not fatal. Mirrors ckpt_path's naming: {name}[_ema{rate}]_step{step:07d}.pt.
    tag = f'_ema{ema_rate}' if ema else ''
    agg_ckpt = os.path.join(ckpt_dir, f'mv_aggregator{tag}_step{s:07d}.pt')
    if agg_cfg is not None and os.path.exists(agg_ckpt):
        agg = mv.build_aggregator(agg_cfg.get('channels'), agg_ckpt)
        from pixal3d_multiview.mv_aggregator import IBRMVAggregator
        fusion = ('ibr learned aggregator (global=anchor-only)'
                  if isinstance(agg, IBRMVAggregator) else 'zero-init aggregator')
    else:
        fusion = 'plain mean (no aggregator)'
    print(f"[oc-mv] swapped '{model_key}' <- MV finetuned (step {s}); fusion: {fusion}")
    return agg


def build_models(args):
    """Load the base pipeline and swap in the finetuned MV denoisers + aggregators.
    EXPENSIVE (loads DINOv3 + 3 flow models + decoders) — call ONCE and reuse across
    instances (see reconstruct_batch.py)."""
    from inference import init_pipeline, MODEL_PATH
    pipeline = init_pipeline(MODEL_PATH, low_vram=args.low_vram)
    aggregators = {
        'ss':    _swap_in_mv_denoiser(pipeline, 'sparse_structure_flow_model',
                                      args.ss_config, SS_BASE, args.ss_ckpt_dir, args.ss_step, args.ema, args.ema_rate),
        'shape': _swap_in_mv_denoiser(pipeline, 'shape_slat_flow_model_1024',
                                      args.shape_config, SHAPE_BASE, args.shape_ckpt_dir, args.shape_step, args.ema, args.ema_rate),
    }
    if not args.no_tex:
        aggregators['tex'] = _swap_in_mv_denoiser(pipeline, 'tex_slat_flow_model_1024',
                                      args.tex_config, TEX_BASE, args.tex_ckpt_dir, args.tex_step, args.ema, args.ema_rate)
    else:
        print('[oc-mv] --no_tex: skipping Stage 3 (texture); geometry-only output.')
    return pipeline, aggregators


def _fps_views(sel, k):
    """Cap views at k by anchor-seeded farthest-point sampling on canonical
    camera directions. sel[0] (the anchor) is always kept; remaining views are
    added greedily by largest angular distance to the already-chosen set, so
    near-duplicate handheld frames are dropped first. Deterministic. Returns
    sel filtered to k entries, original order (anchor first) preserved."""
    dirs = []
    for f in sel:
        t = np.array(f['transform_matrix'], dtype=np.float64)[:3, 3]
        n = np.linalg.norm(t)
        dirs.append(t / n if n > 1e-9 else t)
    dirs = np.stack(dirs)                      # [N, 3] unit camera directions
    chosen = np.zeros(len(sel), dtype=bool)
    chosen[0] = True
    max_cos = dirs @ dirs[0]                   # per-view max cosine to chosen set
    for _ in range(k - 1):
        cand = np.where(~chosen)[0]
        nxt = int(cand[np.argmin(max_cos[cand])])  # farthest from chosen set
        chosen[nxt] = True
        max_cos = np.maximum(max_cos, dirs @ dirs[nxt])
    return [f for i, f in enumerate(sel) if chosen[i]]


def _area_views(sel, k, tj_dir):
    """Cap views at k by largest non-zero (object) area in the input crops.
    sel[0] (the anchor) is always kept; the remaining k-1 slots go to the crops
    with the most non-zero pixels (alpha > 0; falls back to non-black RGB when
    the image has no real alpha). Ties break by subsample_idx for determinism.
    Returns sel filtered to k entries, original order (anchor first) preserved."""
    def _area(f):
        arr = np.array(Image.open((tj_dir / f['file_path']).resolve()).convert('RGBA'))
        n = int((arr[..., 3] > 0).sum())
        if n == arr.shape[0] * arr.shape[1]:   # opaque image -> no real alpha
            n = int(arr[..., :3].any(axis=-1).sum())
        return n
    ranked = sorted(range(1, len(sel)),
                    key=lambda i: (-_area(sel[i]), int(sel[i]['subsample_idx'])))
    chosen = {0, *ranked[:k - 1]}
    return [f for i, f in enumerate(sel) if i in chosen]


def run_instance(pipeline, aggregators, tj_path, args, out_dir=None):
    """Reconstruct ONE instance from its transforms.json with an already-built
    pipeline+aggregators. Saves mesh.pt (+ GLB / ss-viz per flags). Returns out_dir."""
    from proj_grid_offcenter_mv import run_offcenter_mv_pipeline
    tj_path = Path(tj_path).resolve()
    tj_dir = tj_path.parent
    with open(tj_path) as f:
        meta = json.load(f)
    all_frames = meta['frames']

    # ---- Select views ----
    if args.views == 'anchor':
        if args.anchor >= 0:
            sel = [f for f in all_frames if f['subsample_idx'] == args.anchor]
            if not sel:
                raise ValueError(f'--anchor {args.anchor} not in transforms.json')
        else:
            sel = [next((f for f in all_frames if f.get('is_anchor')),
                        all_frames[len(all_frames) // 2])]
    elif args.views == 'all':
        sel = list(all_frames)
    else:
        want = set(int(s) for s in args.views.split(','))
        sel = [f for f in all_frames if f['subsample_idx'] in want]
        if not sel:
            raise ValueError(f'no frames matched --views {args.views}')

    # ---- Pick the anchor (view 0: z_global + aggregator reference) ----
    if args.anchor >= 0:
        anchor = next((f for f in sel if f['subsample_idx'] == args.anchor), None)
        if anchor is None:
            raise ValueError(f'--anchor {args.anchor} not among selected views '
                             f'{[f["subsample_idx"] for f in sel]}')
    else:
        anchor = next((f for f in sel if f.get('is_anchor')), sel[0])
    a_sub = int(anchor['subsample_idx'])
    sel = sorted(sel, key=lambda f: (int(f['subsample_idx']) != a_sub, int(f['subsample_idx'])))

    max_views = getattr(args, 'max_views', 0)
    if max_views > 0 and len(sel) > max_views:
        n_before = len(sel)
        strategy = getattr(args, 'view_select', 'area')
        if strategy == 'fps':
            sel = _fps_views(sel, max_views)
        else:
            sel = _area_views(sel, max_views, tj_dir)
        print(f'[oc-mv] max_views ({strategy}): {n_before} -> {len(sel)} views')

    # ---- T_canon_to_metric ----
    size = float(meta['scale'])
    center_world = np.array(meta['offset'], dtype=np.float64)
    R_box = np.array(meta['R_box'], dtype=np.float64)
    T_canon_to_metric = np.eye(4, dtype=np.float64)
    T_canon_to_metric[:3, :3] = size * R_box
    T_canon_to_metric[:3, 3] = center_world

    out_dir = (Path(out_dir).resolve() if out_dir
               else tj_dir.parent.parent / '_recon' / tj_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[oc-mv] transforms.json = {tj_path}')
    print(f'[oc-mv] views: {[f["subsample_idx"] for f in sel]} (anchor sub={a_sub})')
    print(f'[oc-mv] cube size = {size:.4f} m, center_world = {center_world.round(4).tolist()}')
    print(f'[oc-mv] output = {out_dir}')

    # ---- Build off-center frames ----
    frames_oc = []
    for fr in sel:
        img = Image.open((tj_dir / fr['file_path']).resolve()).convert('RGBA')
        K_norm = K_image_to_norm(fr['K_image_pix'], fr['image_size_px'])
        frames_oc.append(dict(
            image=img,
            transform_matrix=np.array(fr['transform_matrix'], dtype=np.float32),
            K_norm=K_norm.astype(np.float32),
        ))

    print(f'[oc-mv] Running off-center MV Pixal3D @ {args.resolution} ...')
    mesh, res_grid, poses_grid, K_norms, images_rgb, ss_info = run_offcenter_mv_pipeline(
        pipeline, aggregators, frames_oc, seed=args.seed, resolution=args.resolution,
        max_num_tokens=args.max_num_tokens, decode_tex=not args.no_tex,
        sampler=getattr(args, 'sampler', 'train'))
    print(f'[oc-mv] Mesh: V={int(mesh.vertices.shape[0]):,} '
          f'F={int(mesh.faces.shape[0]):,}  res_grid={res_grid}')

    v = mesh.vertices.detach().cpu().numpy()
    print(f'[oc-mv] canonical bbox: min={v.min(0).round(3).tolist()} '
          f'max={v.max(0).round(3).tolist()}')

    # ---- Save raw mesh.pt (atomic: tmp + rename, so a killed process never
    # leaves a truncated mesh.pt that the resume checks would mistake as done) ----
    pt_path = out_dir / 'mesh.pt'
    tmp_path = out_dir / 'mesh.pt.tmp'
    if args.no_tex:
        torch.save({
            'vertices':    mesh.vertices.detach().cpu(),
            'faces':       mesh.faces.detach().cpu(),
            'res_grid':    int(res_grid),
            'views':       [int(f['subsample_idx']) for f in sel],
            'anchor_subsample_idx': a_sub,
            'T_canon_to_metric':    T_canon_to_metric,
            'mode':        'offcenter_mv_geom',
        }, tmp_path)
    else:
        torch.save({
            'vertices':    mesh.vertices.detach().cpu(),
            'faces':       mesh.faces.detach().cpu(),
            'attrs':       mesh.attrs.detach().cpu(),
            'coords':      mesh.coords.detach().cpu(),
            'origin':      mesh.origin.detach().cpu(),
            'voxel_size':  float(mesh.voxel_size),
            'voxel_shape': list(mesh.voxel_shape),
            'layout':      dict(mesh.layout),
            'res_grid':    int(res_grid),
            'views':       [int(f['subsample_idx']) for f in sel],
            'anchor_subsample_idx': a_sub,
            'T_canon_to_metric':    T_canon_to_metric,
            'mode':        'offcenter_mv',
        }, tmp_path)
    os.replace(tmp_path, pt_path)
    print(f'[oc-mv] saved {pt_path}')

    # ---- GLB (world only) ----
    if args.no_glb:
        print('[oc-mv] --no_glb: skipping all GLB exports (mesh.pt only).')
    else:
        if args.no_tex:
            # Geometry-only: same remesh recipe as the textured branch below
            # (remesh=True, band=1, project_back=0), minus the texture bake — so
            # both branches export the SAME surface and land in the same frame.
            glb = remesh_geometry_glb(mesh.vertices, mesh.faces, res_grid,
                                      decimation_target=args.glb_faces,
                                      band=1, project_back=0)
        else:
            import o_voxel
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
                coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
                grid_size=res_grid, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=args.glb_faces, texture_size=4096,
                remesh=True, remesh_band=1, remesh_project=0, use_tqdm=False,
            )
        # R_g_inv undoes to_glb's trailing axis swap (x,y,z)->(x,z,-y), which
        # remesh_geometry_glb reproduces — so it is correct for BOTH branches.
        glb_world = glb
        R_g_inv = np.eye(4); R_g_inv[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        glb_world.apply_transform(T_canon_to_metric @ R_g_inv)
        try:
            glb_world.export(str(out_dir / 'mesh_world.glb'), extension_webp=False)
        except (AttributeError, TypeError):
            glb_world.export(str(out_dir / 'mesh_world.glb'))
        print(f'[oc-mv] saved {out_dir / "mesh_world.glb"}')

    # ---- Stage-1 SS voxel visualization (optional) ----
    if args.vis_ss:
        import trimesh
        ss_coords, ss_grid = ss_info
        occ = ss_coords.detach().cpu().numpy()
        occ = occ[occ[:, 0] == 0][:, 1:].astype(np.int64)
        np.savez(out_dir / 'ss_coords.npz', coords=occ.astype(np.int32), grid=int(ss_grid))
        grid_bool = np.zeros((ss_grid, ss_grid, ss_grid), dtype=bool)
        grid_bool[occ[:, 0], occ[:, 1], occ[:, 2]] = True
        s = 1.0 / ss_grid
        T_idx2canon = np.eye(4); T_idx2canon[:3, :3] *= s; T_idx2canon[:3, 3] = 0.5 * s - 0.5
        ss_cube = trimesh.voxel.VoxelGrid(grid_bool).as_boxes()
        ss_cube.apply_transform(T_idx2canon)
        try:
            from pixal3d.representations import Mesh
            from pixal3d.utils.render_utils import (
                get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics)
            from torchvision.utils import save_image
            yaw = [a - 16 / 180 * np.pi for a in (0, np.pi / 2, np.pi, 3 * np.pi / 2)]
            pitch = [20 / 180 * np.pi] * 4
            exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

            def _render_normals(verts, faces, max_faces=2_000_000):
                rep = Mesh(torch.as_tensor(np.asarray(verts), dtype=torch.float32).cuda(),
                           torch.as_tensor(np.asarray(faces), dtype=torch.int32).cuda())
                # nvdiffrast dies (CUDA error 700) on scene-scale meshes
                # (20M+ faces at grid 1024); decimate for this viz only —
                # mesh.pt keeps the full-res geometry.
                if int(rep.faces.shape[0]) > max_faces:
                    rep.simplify(max_faces)
                r = get_renderer(rep)
                cols = [r.render(rep, e, i)['normal'].clamp(0, 1) for e, i in zip(exts, ints)]
                return torch.cat([torch.cat(cols[:2], 2), torch.cat(cols[2:], 2)], 1)

            ss_img = _render_normals(ss_cube.vertices, ss_cube.faces)
            mesh_img = _render_normals(mesh.vertices.detach().cpu().numpy(),
                                       mesh.faces.detach().cpu().numpy())
            save_image(torch.cat([ss_img, mesh_img], dim=2).cpu(), str(out_dir / 'ss_render.png'))
            print(f'[oc-mv] saved {out_dir / "ss_render.png"} '
                  f'(left: SS voxels | right: mesh; 2x2 orbit normals, {occ.shape[0]} voxels)')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'\033[91m[oc-mv] SS render FAILED: {e!r}\033[0m')

        if not args.no_glb:
            ss_world = ss_cube.copy(); ss_world.apply_transform(T_canon_to_metric)
            ss_world.export(str(out_dir / 'ss_voxels_world.glb'))
            print(f'[oc-mv] saved {out_dir / "ss_voxels_world.glb"} (overlays mesh_world.glb)')

    print('[oc-mv] Done.')
    return out_dir


def main():
    ap = argparse.ArgumentParser(
        description='Off-center Pixal3D with the finetuned multi-view model.')
    ap.add_argument('--transforms_json', type=str, required=True)
    ap.add_argument('--views', type=str, default='anchor',
                    help="'anchor' (default), 'all', or comma-separated subsample idxs.")
    ap.add_argument('--max_views', type=int, default=20,
                    help='Cap the number of input views when the --views selection has more. '
                         '0 = no cap. Selection strategy: --view_select.')
    ap.add_argument('--view_select', type=str, default='area', choices=['area', 'fps'],
                    help="How to pick --max_views views (anchor always kept as view 0). "
                         "'area' (default): largest non-zero object area in the input crops. "
                         "'fps': anchor-seeded farthest-point sampling on canonical camera "
                         "directions (max angular spread, drops near-duplicates first).")
    ap.add_argument('--anchor', type=int, default=-1,
                    help="subsample_idx of the anchor frame (view 0: provides z_global + the "
                         "aggregator reference). -1 = use transforms.json is_anchor. Useful with "
                         "--views all. Does NOT change the canonical/world frame (fixed by R_box).")
    ap.add_argument('--output_dir', type=str, default=None)
    ap.add_argument('--resolution', type=int, default=1024,
                    help='Fixed at 1024 (grid 64) — the MV models were finetuned at this res.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sampler', type=str, default='train',
                    choices=['train', 'official', 'pipeline'],
                    help="Sampler config: 'train' = training-snapshot params "
                         "(FlowEulerCfgSampler, SS 50/gs3, sparse 12/gs3); "
                         "'official' = pretrained pipeline.json params "
                         "(FlowEulerGuidanceIntervalSampler, 12 steps, gs7.5 + "
                         "rescale/interval); 'pipeline' = as loaded, uninstalled.")
    ap.add_argument('--max_num_tokens', type=int, default=49152)
    ap.add_argument('--low_vram', action='store_true')
    ap.add_argument('--glb_faces', type=int, default=100_000,
                    help='Face-count upper bound for the exported GLB (per object). '
                         'The remesh stage itself is unaffected — it always emits '
                         '~2x the input faces at grid 1024; this caps what is written. '
                         'Note the cap is ALWAYS binding: remesh output is 1.6M-21M '
                         'faces, so every object lands just under this number.')
    ap.add_argument('--no_glb', action='store_true',
                    help='Skip ALL GLB exports (mesh_world.glb, ss_voxels_world.glb). '
                         'Only mesh.pt is written — enough for geometry metric eval, '
                         'and it also skips the 1024^3 remesh that builds the GLB.')
    ap.add_argument('--no_tex', action='store_true',
                    help='Skip Stage 3 (texture): run only SS + shape and output a '
                         'geometry-only mesh (no PBR attrs, geometry-only GLB). Also skips '
                         'loading the texture MV denoiser.')
    ap.add_argument('--vis_ss', action='store_true',
                    help='Also export the Stage-1 sparse-structure voxels: ss_voxels_canon.glb '
                         '(+ ss_voxels_world.glb) as a cube grid in the SAME frame as the mesh '
                         'GLBs (so they overlay), plus raw ss_coords.npz.')
    # MV checkpoint selection (-1 = latest; EMA by default).
    ap.add_argument('--ss_step', type=int, default=-1)
    ap.add_argument('--shape_step', type=int, default=-1)
    ap.add_argument('--tex_step', type=int, default=-1)
    # MV checkpoint dirs (override the hard-coded defaults). Each holds
    # denoiser_step*.pt + mv_aggregator_step*.pt (+ _ema{rate} variants).
    ap.add_argument('--ss_ckpt_dir', type=str, required=True)
    ap.add_argument('--shape_ckpt_dir', type=str, required=True)
    ap.add_argument('--tex_ckpt_dir', type=str, default=None)
    # MV stage configs (override the hard-coded defaults). Provide the
    # denoiser/lora/aggregator definition matching the ckpt dir.
    ap.add_argument('--ss_config', type=str, required=True)
    ap.add_argument('--shape_config', type=str, required=True)
    ap.add_argument('--tex_config', type=str, default=None)
    ap.add_argument('--ema', dest='ema', action='store_true', default=False)
    ap.add_argument('--no-ema', dest='ema', action='store_false')
    ap.add_argument('--ema_rate', type=float, default=0.9999)
    args = ap.parse_args()

    pipeline, aggregators = build_models(args)
    run_instance(pipeline, aggregators, args.transforms_json, args, out_dir=args.output_dir)


if __name__ == '__main__':
    main()
