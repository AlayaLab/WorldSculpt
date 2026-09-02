"""Shared helpers for multi-view (LoRA + zero-init aggregator) inference.

The three stage scripts (`inference_mv_stage{1,2,3}_*.py`) chain together:

    stage1 (SS)    : K view images            -> sparse-structure coords (grid 64)
    stage2 (Shape) : K view images + coords   -> shape SLat  (+ optional mesh)
    stage3 (Tex)   : K view images + shape    -> pbr   SLat  -> textured .glb

Everything here is a thin wrapper that *reuses the training code path*:

  * The denoiser is rebuilt exactly as `train_multiview.py` builds it
    (base ckpt -> peft LoRA wrap), then the finetuned LoRA-only checkpoint is
    loaded on top. The zero-init `mv_aggregator` is loaded as a sibling module.
  * Multi-view conditioning is produced by the *same* method the trainer uses
    at `i_sample` time: `_MultiViewProjMixin.encode_image_proj` (loop over the K
    views through the DINOv3 proj extractor, then aggregate the per-voxel proj
    features with the ControlNet-style zero-init residual). We drive it through
    the per-view projection helpers in `proj_grid_offcenter_mv.py`, which carry the
    reads, so there is no copy/paste of the aggregation logic.
  * Sampling uses the production `FlowEulerGuidanceIntervalSampler` with the
    per-stage params from `pretrained/Pixal3D/pipeline.json`.
  * Decoding / GLB export reuses the decoders shipped in
    `pretrained/Pixal3D/ckpts` and `o_voxel.postprocess.to_glb`, exactly like
    the single-view `inference.py`.

The only thing training does NOT cover is the *inter-stage plumbing*: at train
time each stage reads GT coords / shape-latents from disk; here stage N feeds
stage N+1. See `decode_ss_to_coords` and the stage scripts for that glue.
"""
import os

# These must be set before importing pixal3d / o_voxel (read at import time).
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('ATTN_BACKEND', 'flash_attn')
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('FLEX_GEMM_AUTOTUNE_CACHE_PATH', os.path.join(_HERE, 'autotune_cache.json'))

import glob
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from pixal3d import models
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVoxel
from pixal3d.pipelines import samplers

from pixal3d_multiview.lora import prepare_lora_model
from pixal3d_multiview.mv_aggregator import ZeroInitMVAggregator, IBRMVAggregator
from pixal3d_multiview.utils import relative_pose, patch_proj_grid_for_multiview
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
    project_points_to_image_batch,
)

# Make ProjGrid.forward accept a real per-view transform_matrix (same patch the
# trainer applies). Idempotent — safe to call on import.
patch_proj_grid_for_multiview()

PROJ_ROOT = _HERE
PRETRAINED_CKPTS = os.path.join(PROJ_ROOT, 'pretrained', 'Pixal3D', 'ckpts')

# The finetuned denoisers were built with the default dtype (float32) and run
# under amp autocast(bfloat16) at train time (mix_precision_mode='amp',
# mix_precision_dtype='bfloat16'). `manual_cast` is a no-op while autocast is
# enabled, so flash-attn receives bf16. We must reproduce that autocast context
# at inference, otherwise the model runs in fp32 and flash-attn errors out.
AUTOCAST_DTYPE = torch.bfloat16

# PBR channel layout of the texture decoder output (matches Pixal3DImageTo3DPipeline).
PBR_ATTR_LAYOUT = {
    'base_color': slice(0, 3),
    'metallic': slice(3, 4),
    'roughness': slice(4, 5),
    'alpha': slice(5, 6),
}

# Per-stage sampler params.
#
# 'train' == exactly what the trainers' run_snapshot() uses at i_sample time
# (FlowEulerCfgSampler, plain CFG, guidance_strength=3.0; SS=50 steps, sparse=12).
# This is what the finetuned model was validated against, so it is the DEFAULT —
# inference output then matches the training snapshots.
#
# 'pipeline' == the production params from pretrained/Pixal3D/pipeline.json
# (FlowEulerGuidanceIntervalSampler). These were tuned for the *base* model and
# tend to over-guide the LoRA-finetuned model; kept as an option.
TRAIN_PARAMS = {
    'ss':    dict(steps=50, guidance_strength=3.0),
    'shape': dict(steps=12, guidance_strength=3.0),
    'tex':   dict(steps=12, guidance_strength=3.0),
}
PIPELINE_PARAMS = {
    'ss':    dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
                  guidance_interval=(0.6, 1.0), rescale_t=5.0),
    'shape': dict(steps=12, guidance_strength=7.5, guidance_rescale=0.5,
                  guidance_interval=(0.6, 1.0), rescale_t=3.0),
    'tex':   dict(steps=12, guidance_strength=1.0, guidance_rescale=0.0,
                  guidance_interval=(0.6, 0.9), rescale_t=3.0),
}


# ============================================================================
# Checkpoint discovery
# ============================================================================

def find_latest_step(ckpt_dir: str) -> int:
    """Largest step that has a misc_step*.pt (i.e. a fully-saved checkpoint)."""
    files = glob.glob(os.path.join(ckpt_dir, 'misc_step*.pt'))
    if not files:
        # fall back to denoiser files
        files = glob.glob(os.path.join(ckpt_dir, 'denoiser_step*.pt'))
    if not files:
        raise FileNotFoundError(f'No checkpoints found in {ckpt_dir}')
    steps = [int(os.path.basename(f).split('step')[-1].split('.')[0]) for f in files]
    return max(steps)


def ckpt_path(ckpt_dir: str, name: str, step: int, ema: bool,
              ema_rate: float = 0.9999) -> str:
    """Path to `{name}[_ema{rate}]_step{step:07d}.pt`."""
    tag = f'_ema{ema_rate}' if ema else ''
    p = os.path.join(ckpt_dir, f'{name}{tag}_step{step:07d}.pt')
    if not os.path.exists(p):
        raise FileNotFoundError(f'Checkpoint not found: {p}')
    return p


# ============================================================================
# Model building (denoiser + LoRA + aggregator + image cond model)
# ============================================================================

def load_stage_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def build_mv_denoiser(
    model_cfg: dict,
    lora_cfg: dict,
    base_ckpt: str,
    ft_ckpt: str,
    device: str = 'cuda',
    dtype: str = None,
) -> torch.nn.Module:
    """Rebuild the finetuned multi-view denoiser.

    1. instantiate the base architecture from the training config args,
    2. load the pretrained base weights + wrap with peft LoRA (== training),
    3. load the finetuned LoRA-only checkpoint on top.

    The returned module is a peft model whose forward delegates to the base
    denoiser, so it is called exactly like the trainer does:
        denoiser(x_t, t, cond[, concat_cond=...]).

    dtype: optional model dtype override (e.g. 'bfloat16'). When set, the model
    runs natively in that dtype (manual_cast is a no-op without autocast), so it
    can be used like the base bf16 pipeline models WITHOUT an autocast wrapper.
    Default None keeps the training default (float32 + amp autocast at sample time).
    """
    name = model_cfg['name']
    args = dict(model_cfg['args'])
    if dtype is not None:
        args['dtype'] = dtype
    denoiser = getattr(models, name)(**args)

    # prepare_lora_model: load base ckpt into the raw module, then peft-wrap.
    denoiser = prepare_lora_model(denoiser, base_ckpt, lora_cfg)

    # Load the finetuned LoRA-only state dict. Backfill the (already-present)
    # base + per-block proj_linear keys so load_state_dict is strict-clean —
    # this mirrors how the LoRA checkpoints are loaded at training time.
    lora_sd = torch.load(ft_ckpt, map_location='cpu', weights_only=True)
    full = denoiser.state_dict()
    full.update(lora_sd)
    denoiser.load_state_dict(full)

    denoiser = denoiser.to(device).eval()
    denoiser.requires_grad_(False)
    n_lora = sum(1 for k in lora_sd if 'lora_' in k)
    print(f'[denoiser] {name}: base={os.path.basename(base_ckpt)} '
          f'+ {n_lora} LoRA tensors from {os.path.basename(ft_ckpt)}')
    return denoiser




def build_aggregator(channels: int, agg_ckpt: str, device: str = 'cuda',
                     global_channels: Optional[int] = None) -> torch.nn.Module:
    """Build the MV aggregator MATCHING the checkpoint. The class is auto-detected
    from the state-dict keys (robust even if the config lacks a 'mode' field):
      - 'feature_mlp.0.weight' -> IBRMVAggregator (GenRecon-style learned per-voxel
        view weighting; encode_image_proj routes on isinstance and uses
        anchor-only global, per-view gather + softmax-weighted residual on mean)
      - 'zero_proj.weight'     -> ZeroInitMVAggregator (anchor + zero_proj(mean others))
    """
    sd = torch.load(agg_ckpt, map_location='cpu', weights_only=True)
    if 'feature_mlp.0.weight' in sd:
        # IBR: channels recoverable from the ckpt itself; verify config agreement.
        ckpt_channels = sd['feature_mlp.0.weight'].shape[0]
        assert channels in (None, ckpt_channels), \
            f'aggregator channels mismatch: config {channels} vs ckpt {ckpt_channels}'
        agg = IBRMVAggregator(channels=ckpt_channels)
        agg.load_state_dict(sd)
        mode = f'ibr channels={ckpt_channels} (global=anchor-only)'
    else:
        # Zero-init: auto-detect the optional global-token branch (zero_proj_global)
        # from the checkpoint and build a MATCHING module. Training adds it when the
        # config's mv_aggregator.global_channels is set (e.g. 1024 for the DINOv3-L
        # global token); older checkpoints without it still load (-> None).
        # encode_image_proj feeds g_anchor/g_others_mean to the aggregator, so once
        # the branch exists, inference uses it exactly as training did.
        if global_channels is None and 'zero_proj_global.weight' in sd:
            global_channels = sd['zero_proj_global.weight'].shape[0]
        agg = ZeroInitMVAggregator(channels=channels, global_channels=global_channels)
        agg.load_state_dict(sd)
        mode = f'zero-init channels={channels} global_channels={global_channels}'
    agg = agg.to(device).eval()
    agg.requires_grad_(False)
    print(f'[aggregator] {mode} <- {os.path.basename(agg_ckpt)}')
    return agg


def build_image_cond_model(image_cond_cfg: dict, device: str = 'cuda') -> DinoV3ProjFeatureExtractor:
    """Build the DINOv3 proj feature extractor for a stage and move to device.

    `image_cond_cfg` is the trainer-style dict {'name', 'args', 'image_attn_mode'}.
    """
    assert image_cond_cfg['name'] == 'DinoV3ProjFeatureExtractor', \
        f"Unsupported image cond model: {image_cond_cfg['name']}"
    model = DinoV3ProjFeatureExtractor(**image_cond_cfg.get('args', {}))
    model.eval()
    model = model.to(device)
    if getattr(model, 'use_naf_upsample', False):
        model._load_naf()  # pull NAF weights now (frozen)
    print(f"[image_cond] DinoV3ProjFeatureExtractor "
          f"img={model.image_size} grid={model.grid_resolution} "
          f"naf={model.use_naf_upsample} proj_ch={model.proj_channels}")
    return model


# ============================================================================
# Multi-view conditioning engine (reuses the trainer's encode_image_proj)
# ============================================================================



# ============================================================================
# View / camera batch construction (mirrors the multi-view dataset)
# ============================================================================

# Canonical front-view voxel pose used by ProjGrid (front_view_transform_matrix
# with the default -2.0 replaced by -anchor_distance), matching
# the multi-view conditioning path used during finetuning.
def _anchor_T_in_voxel(anchor_distance: float) -> torch.Tensor:
    return torch.tensor([
        [1.0, 0.0,  0.0,  0.0],
        [0.0, 0.0, -1.0, -anchor_distance],
        [0.0, 1.0,  0.0,  0.0],
        [0.0, 0.0,  0.0,  1.0],
    ], dtype=torch.float32)


def _load_view_image(image_dir: str, file_path: str, image_size: int) -> torch.Tensor:
    """Load one PNG -> (3, H, W) in [0,1] with alpha composited over black bg."""
    image = Image.open(os.path.join(image_dir, file_path))
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = image.getchannel(3)
    image = image.convert('RGB')
    image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    alpha = torch.tensor(np.array(alpha)).float() / 255.0
    return image * alpha.unsqueeze(0)






# ============================================================================
# Sampling
# ============================================================================



# ============================================================================
# Latent normalization
# ============================================================================

def normalize(slat: SparseTensor, norm: dict) -> SparseTensor:
    std = torch.tensor(norm['std'])[None].to(slat.device)
    mean = torch.tensor(norm['mean'])[None].to(slat.device)
    return (slat - mean) / std




# ============================================================================
# Decoders + GLB export (reuses the pretrained decoders + o_voxel)
# ============================================================================









# nvdiffrast's CudaRaster triangleSetupKernel faults (CUDA error 700, illegal
# access) once the triangle count exceeds its internal limit (~2^24 ≈ 16.7M). A
# pathological / noise decode can emit a 20M+ triangle "soup". We DECIMATE such
# meshes (same simplifier to_glb uses) down to a render-friendly size instead of
# skipping, so even bad generations still get a (downsampled) visualization.
RASTER_FACE_CAP = 6_000_000        # above this -> decimate before rasterizing
RASTER_DECIM_FACES = 2_000_000     # decimation target (plenty for a 1024px render)


def _prep_mesh_for_raster(vertices, faces):
    """Validate + (if too dense) decimate a mesh so nvdiffrast's CudaRaster won't
    fault (CUDA 700) on triangle soups. Returns (vertices, faces) ready to render,
    or (None, None) if the mesh is truly unrenderable (empty / non-finite verts /
    out-of-range face indices)."""
    if vertices is None or faces is None or vertices.shape[0] == 0 or faces.shape[0] == 0:
        return None, None
    if not bool(torch.isfinite(vertices).all()):
        print('[render] non-finite vertices; skipping viz', flush=True)
        return None, None
    if int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]:
        print('[render] out-of-range face indices; skipping viz', flush=True)
        return None, None
    if faces.shape[0] <= RASTER_FACE_CAP:
        return vertices, faces
    try:                                                     # too dense -> decimate
        import cumesh
        cm = cumesh.CuMesh()
        cm.init(vertices.cuda().contiguous().float(), faces.cuda().contiguous().int())
        cm.simplify(RASTER_DECIM_FACES, verbose=False)
        v, f = cm.read()
        print(f'[render] decimated dense mesh {int(faces.shape[0]):,} -> {int(f.shape[0]):,} '
              f'faces for rasterization', flush=True)
        return v, f
    except Exception as e:
        print(f'[render] decimation failed ({e!r}); skipping viz', flush=True)
        return None, None








# PBR channels the renderer produces (== training make_pbr_vis_frames set).
PBR_CHANNELS = ('shaded', 'base_color', 'normal', 'metallic', 'roughness')


def _to_3ch(img: torch.Tensor) -> torch.Tensor:
    """A rendered channel -> (3,H,W). 1-channel (metallic/roughness) -> repeated."""
    if img.dim() == 2:
        img = img.unsqueeze(0)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return img




def to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) or (H,W,3) float in [0,1] -> (H,W,3) uint8."""
    t = t.detach().cpu()
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    return (t.clamp(0, 1).numpy() * 255).astype(np.uint8)


# anchor / input / other view annotation colours.
VIEW_TAG_COLORS = {'anchor': (255, 215, 0), 'input': (90, 220, 90), 'other': (140, 140, 140)}


def save_labeled_grid(rows, row_labels, col_labels, path, cell=None, col_colors=None):
    """Save a labeled image grid.

    rows: list (one per row) of lists of (H,W,3) uint8 arrays (all rows length n).
    row_labels: left-side label per row. col_labels: top label per column.
    cell: cell size in px; None -> keep each image's native size (NO resize).
    col_colors: optional per-column RGB for the column label text.
    """
    from PIL import Image as _Image, ImageDraw
    n = len(rows[0])
    nrow = len(rows)
    ch, cw = rows[0][0].shape[:2] if cell is None else (cell, cell)
    label_scale = max(1, cw // 256)                      # bigger labels for big cells
    left_w, top_h = 24 + 16 * label_scale, 14 + 10 * label_scale
    W, H = left_w + n * cw, top_h + nrow * ch
    canvas = _Image.new('RGB', (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=12 * label_scale)
    except Exception:
        font = None
    for c in range(n):
        x = left_w + c * cw
        color = col_colors[c] if col_colors else (235, 235, 235)
        draw.text((x + 6, 6), str(col_labels[c]), fill=color, font=font)
        draw.line([(x, top_h), (x, H)], fill=(60, 60, 60), width=1)
    for r in range(nrow):
        y = top_h + r * ch
        draw.text((6, y + ch // 2 - 6 * label_scale), str(row_labels[r]), fill=(235, 235, 235), font=font)
        for c in range(n):
            img = _Image.fromarray(rows[r][c])
            if cell is not None:
                img = img.resize((cw, ch), _Image.LANCZOS)
            canvas.paste(img, (left_w + c * cw, y))
    if str(path).lower().endswith(('.jpg', '.jpeg')):
        canvas.save(path, quality=95, subsampling=0)     # high-quality JPEG (no chroma subsample)
    else:
        canvas.save(path)
    return path






@torch.no_grad()
def decode_shape_to_meshes(shape_decoder, shape_slat: SparseTensor, resolution: int):
    """shape SLat (un-normalized) -> (meshes, subs). subs guide texture decode."""
    shape_decoder.set_resolution(resolution)
    return shape_decoder(shape_slat, return_subs=True)






# ============================================================================
# SparseTensor (de)serialization for inter-stage handoff
# ============================================================================





# ============================================================================
# GT latents (for the run_snapshot-style self-check)
# ============================================================================









# ============================================================================
# Per-view alignment rendering (project decoded voxels into each view)
# ============================================================================

# Same lattice->Blender rotation ProjGrid bakes into its grid_points.
ROTATION_MV = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=torch.float32)


def splat(p2d, depth, color, valid, size, radius):
    """Depth-buffered point splat -> (size, size, 3) float image in [0,1]."""
    p2d = p2d.detach().cpu().numpy()
    depth = depth.detach().cpu().numpy()
    color = color.detach().cpu().numpy()
    valid = valid.detach().cpu().numpy().astype(bool)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    keep = np.where(valid & np.isfinite(depth))[0]
    if keep.size == 0:
        return canvas
    order = keep[np.argsort(-depth[keep])]            # far first
    xs = np.round(p2d[order, 0]).astype(np.int64)
    ys = np.round(p2d[order, 1]).astype(np.int64)
    cols = np.clip(color[order], 0.0, 1.0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px = np.clip(xs + dx, 0, size - 1)
            py = np.clip(ys + dy, 0, size - 1)
            canvas[py, px] = cols                     # later (nearer) overwrites
    return canvas


