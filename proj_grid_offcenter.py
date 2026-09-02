"""
proj_grid_offcenter.py — off-center-principal ProjGrid for Pixal3D.

WHY
---
Pixal3D's ProjGrid hard-codes the principal point at the image center
(`x_pixel = f*x/(-z) + res/2`) and uses a symmetric FOV. That forces the
camera to look straight at the cube center (front view). Real video cameras
do NOT look at the cube center (e.g. blue_cup anchor is ~13deg off), and the
each crop is cube-centered with an OFF-CENTER principal point
(cx_image ≈ -54 in a 512 crop). The centered ProjGrid therefore mis-samples.

This module adds an OFF-CENTER projection that uses the full intrinsics
K = [[fx,0,cx],[0,fy,cy],[0,0,1]] (cx/cy may be anywhere, even outside the
image) plus the REAL camera pose. Numerically verified: with the real
canonical pose (R_cube2grid @ transform_matrix) and the real K, the canonical
cube projects so its center lands at the crop center and the 8 corners fill
the crop — exactly matching the cube-centered crop, no warp, no rescale.

NON-INVASIVE
------------
The original `ProjGrid.forward` is NOT edited. We monkey-patch it at runtime:
off-center is used ONLY when `proj_grid._oc_K_norm` is set on the instance;
otherwise it transparently falls back to whatever forward was there before
(the multiview patch, or the stock centered behavior). Idempotent.

OpenCV vs Blender: the crop K is OpenCV (cx,cy with +v down); ProjGrid uses a
Blender cam (looks at -Z, +y up). One can show the double flip (Y and Z)
cancels in the projection ratio, so OpenCV cx/cy plug directly into the Blender
projection formula — no conversion needed (verified numerically).
"""

from typing import List, Optional
from PIL import Image
import numpy as np
import torch

from pixal3d.modules.sparse import SparseTensor


# canonical cube frame -> ProjGrid internal grid frame (== front_T rotation)
R_CUBE2GRID = torch.tensor([
    [1.0, 0.0,  0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0,  0.0, 0.0],
    [0.0, 0.0,  0.0, 1.0],
], dtype=torch.float32)




def rgba_over_black(img: Image.Image) -> Image.Image:
    """Composite RGBA over a black background → RGB."""
    arr = np.array(img.convert('RGBA')).astype(np.float32) / 255.0
    rgb = (arr[..., :3] * arr[..., 3:4]).clip(0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))




# =============================================================================
# Off-center projection
# =============================================================================

def project_points_with_K(points_3d: torch.Tensor, c2w: torch.Tensor,
                           K_norm: torch.Tensor, resolution: int) -> torch.Tensor:
    """Project grid points with a full (possibly off-center) intrinsic.

    Args:
        points_3d : [B, N, 3] points in the ProjGrid grid frame
        c2w       : [B, 4, 4] camera-to-grid pose (Blender convention, looks -Z)
        K_norm    : [B, 3, 3] intrinsics normalized to [0,1] (fx,fy,cx,cy / image side)
        resolution: ProjGrid image resolution (pixels)

    Returns:
        image_points : [B, N, 2] pixel coordinates (x_pixel, y_pixel)
    """
    B, N, _ = points_3d.shape
    ones = torch.ones(B, N, 1, device=points_3d.device, dtype=points_3d.dtype)
    pts_h = torch.cat([points_3d, ones], dim=-1)                    # [B,N,4]
    w2c = torch.linalg.inv(c2w.float()).to(points_3d.dtype)        # [B,4,4]
    pts_cam = torch.bmm(pts_h, w2c.transpose(-2, -1))[..., :3]      # [B,N,3]
    x = pts_cam[..., 0]; y = pts_cam[..., 1]; z = pts_cam[..., 2]

    fx = (K_norm[:, 0, 0] * resolution).unsqueeze(1)               # [B,1]
    fy = (K_norm[:, 1, 1] * resolution).unsqueeze(1)
    cx = (K_norm[:, 0, 2] * resolution).unsqueeze(1)
    cy = (K_norm[:, 1, 2] * resolution).unsqueeze(1)

    # Same structure as the stock ProjGrid (Blender, depth=-z), but with full K.
    x_pixel = fx * x / (-z + 1e-8) + cx
    y_pixel = -fy * y / (-z + 1e-8) + cy

    # Points at/behind the camera plane (depth = -z <= eps) perspective-divide by a
    # non-positive denominator and mirror through the principal point, silently
    # sampling unrelated in-image pixels. Route them far out of bounds instead:
    # grid_sample(padding_mode='border') then clamps to the crop edge — the black
    # background of this same view, i.e. the training-consistent "no observation"
    # DINO feature (a zero feature vector would be out-of-distribution).
    valid = (-z) > 1e-3
    oob = torch.full_like(x_pixel, -1e6)
    x_pixel = torch.where(valid, x_pixel, oob)
    y_pixel = torch.where(valid, y_pixel, oob)
    return torch.stack([x_pixel, y_pixel], dim=-1)


# =============================================================================
# Runtime patch (non-invasive)
# =============================================================================

def enable_offcenter_projgrid():
    """Monkey-patch ProjGrid.forward to support an off-center mode.

    Off-center is active ONLY when `self._oc_K_norm` is not None. Otherwise the
    previously-installed forward (multiview patch or stock) runs unchanged.
    Idempotent.
    """
    from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj as M

    if getattr(M.ProjGrid, '_offcenter_patched', False):
        return

    sample_features = M.sample_features
    prev_forward = M.ProjGrid.forward  # may be the multiview-patched or stock forward

    def patched_forward(self, features_map, camera_angle_x, distance, mesh_scale,
                        transform_matrix=None, BHWC=True):
        K_norm = getattr(self, '_oc_K_norm', None)
        if K_norm is None:
            # not in off-center mode -> original behavior
            return prev_forward(self, features_map, camera_angle_x, distance,
                                mesh_scale, transform_matrix, BHWC)

        if BHWC:
            B, H, W, C = features_map.shape
        else:
            B, C, H, W = features_map.shape

        grid_points = self.grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2

        pose = self._oc_pose  # [B,4,4] c2w in grid frame
        image_points = project_points_with_K(
            grid_points, pose, K_norm, self.image_resolution)
        image_points_norm = (image_points + 0.5) / self.image_resolution * 2 - 1

        if BHWC:
            features_map = features_map.permute(0, 3, 1, 2)
        x = sample_features(features_map, image_points_norm)
        return x.permute(0, 2, 1)

    M.ProjGrid.forward = patched_forward
    M.ProjGrid._offcenter_patched = True


# =============================================================================
# Off-center feature aggregation (mirror of multiview aggregate_proj_features)
# =============================================================================

@torch.no_grad()
def aggregate_offcenter(image_cond_model, images_rgb, poses_grid, K_norms,
                        mesh_scale=1.0, grid_resolution_override=None,
                        device='cuda'):
    """Per-view: stash off-center (K, pose) on the proj_grid, run image_cond_model,
    aggregate z_proj by mean across views. Returns (z_global_anchor, z_proj_agg)."""
    orig_grid_res = image_cond_model.grid_resolution
    if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
        image_cond_model.grid_resolution = grid_resolution_override
        image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
            grid_resolution=grid_resolution_override,
            image_resolution=image_cond_model.proj_grid.image_resolution,
        ).to(device)

    pg = image_cond_model.proj_grid
    z_globals, z_projs = [], []
    for img, pose, K in zip(images_rgb, poses_grid, K_norms):
        pg._oc_K_norm = K.to(device)        # [1,3,3]
        pg._oc_pose = pose.to(device)       # [1,4,4]
        # camera_angle_x / distance / transform_matrix are ignored in off-center mode
        dummy = torch.tensor([0.5], device=device)
        scale_t = torch.tensor([mesh_scale], device=device)
        z_g, z_p = image_cond_model(
            [img], camera_angle_x=dummy, distance=dummy, mesh_scale=scale_t,
            transform_matrix=None)
        z_globals.append(z_g)
        z_projs.append(z_p)
    pg._oc_K_norm = None
    pg._oc_pose = None

    z_proj_agg = torch.stack(z_projs, dim=0).mean(dim=0)
    z_global = z_globals[0]

    if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
        image_cond_model.grid_resolution = orig_grid_res
        image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
            grid_resolution=orig_grid_res,
            image_resolution=image_cond_model.proj_grid.image_resolution,
        ).to(device)

    return z_global, z_proj_agg


def build_offcenter_cond_ss(pipeline, images_rgb, poses_grid, K_norms, mesh_scale=1.0):
    device = pipeline.device
    icm = pipeline.image_cond_model_ss
    if pipeline.low_vram:
        icm.to(device)
    z_g, z_p = aggregate_offcenter(icm, images_rgb, poses_grid, K_norms,
                                   mesh_scale, device=device)
    if pipeline.low_vram:
        icm.cpu()
    return {
        'cond':     {'global': z_g, 'proj': z_p},
        'neg_cond': {'global': torch.zeros_like(z_g), 'proj': torch.zeros_like(z_p)},
    }


def build_offcenter_cond_shape(pipeline, image_cond_model, images_rgb, poses_grid,
                               K_norms, coords, mesh_scale=1.0,
                               grid_resolution_override=None):
    device = pipeline.device
    if pipeline.low_vram:
        image_cond_model.to(device)
    z_g, z_p = aggregate_offcenter(
        image_cond_model, images_rgb, poses_grid, K_norms, mesh_scale,
        grid_resolution_override=grid_resolution_override, device=device)

    grid_res = image_cond_model.grid_resolution
    z_proj_grid = z_p.reshape(1, grid_res, grid_res, grid_res, -1)
    batch_idx = coords[:, 0].long()
    x = coords[:, 1].long(); y = coords[:, 2].long(); z = coords[:, 3].long()
    z_proj_sparse = z_proj_grid[batch_idx, x, y, z]
    z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)

    if pipeline.low_vram:
        image_cond_model.cpu()
    return {
        'cond':     {'global': z_g, 'proj': z_proj_st},
        'neg_cond': {'global': torch.zeros_like(z_g),
                     'proj':   SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords)},
    }


# =============================================================================
# Off-center pipeline (mirror of run_multiview_pipeline; real pose + real K)
# =============================================================================

