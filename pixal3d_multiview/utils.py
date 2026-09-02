"""Shared multi-view utilities.

These are pulled out so both training (this package) and inference
(`multiview_inference.py`) can share the same camera-frame conventions.
"""
from typing import Tuple
import numpy as np
import torch


def relative_pose(T_world_anchor: torch.Tensor, T_world_view: torch.Tensor) -> torch.Tensor:
    """Express T_world_view in T_world_anchor's frame.

    T_view_in_anchor = T_world_anchor^-1 @ T_world_view

    Args:
        T_world_anchor: (4, 4) camera-to-world of the anchor view.
        T_world_view:   (4, 4) camera-to-world of another view.

    Returns:
        (4, 4) tensor expressing `T_world_view` in anchor's coordinate frame.
        For the anchor itself this returns the identity.
    """
    inv_anchor = torch.linalg.inv(T_world_anchor.double()).float()
    return inv_anchor @ T_world_view


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """3x3 rotation from an axis (unnormalised) + angle (radians)."""
    a = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = a
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def perturb_nonanchor_cameras(transforms, cam_angles, rng, cfg):
    """Camera-pose-noise augmentation (sim2real): synthetic poses are exact, real
    poses aren't. Inject noise into the poses fed to ProjGrid so the net learns to
    tolerate pose error. Perturbs ONLY the non-anchor views (index 1..K-1) -- the
    anchor (index 0) defines the canonical frame the GT latent lives in and must
    stay exact, else input frame and supervision disagree. The IMAGES are NOT
    touched (content is at the true pose; we only corrupt the pose used to
    project), exactly mirroring real noisy-pose conditioning.

    In place on `transforms` (K,4,4) and `cam_angles` (K,) torch tensors:
      - rotation: angle ~ U(rot_deg) about a random axis
      - translation: magnitude ~ U(trans) in a random direction (voxel-frame Δ@T)
      - FOV: camera_angle_x *= 1 + U(-fov, fov)
    each non-anchor view independent, applied with prob `p_apply`."""
    K = transforms.shape[0]
    rot_lo, rot_hi = cfg.get('rot_deg', (0.0, 5.0))
    tr_lo, tr_hi = cfg.get('trans', (0.0, 0.05))
    fov_amp = float(cfg.get('fov', 0.05))
    p = float(cfg.get('p_apply', 1.0))
    for v in range(1, K):                                  # skip anchor (index 0)
        if rng.random() >= p:
            continue
        R = _rodrigues(rng.normal(size=3),
                       np.radians(rng.uniform(rot_lo, rot_hi)))
        tdir = rng.normal(size=3); tdir /= (np.linalg.norm(tdir) + 1e-9)
        delta = np.eye(4)
        delta[:3, :3] = R
        delta[:3, 3] = float(rng.uniform(tr_lo, tr_hi)) * tdir
        D = torch.from_numpy(delta).to(transforms.dtype)
        transforms[v] = D @ transforms[v]                  # Δ @ T  (voxel-frame pose error)
        if fov_amp > 0:
            cam_angles[v] = cam_angles[v] * (1.0 + float(rng.uniform(-fov_amp, fov_amp)))
    return transforms, cam_angles


def sample_num_input_views(low: int = 2, high: int = 6, generator=None) -> int:
    """Sample K ~ Uniform{low, low+1, ..., high} (inclusive).

    Paper recipe: K ∈ {2, 3, 4, 5, 6}.
    """
    import numpy as np
    if generator is not None:
        return int(generator.integers(low, high + 1))
    return int(np.random.randint(low, high + 1))


# ---------------------------------------------------------------------------
# Camera utilities for the per-view triptych snapshot
# ---------------------------------------------------------------------------
#
# These INVERT ProjGrid's projection so we can RENDER the decoded
# representation from each cond view, geometrically aligned with the input
# photos. Verified numerically to reproduce, at the anchor view, the canonical
# `gt_view` of both conventions:
#   - SparseStructureLatent.visualize_sample  (object Y-Z swapped, cam (0,-d,0) up Z)
#   - SLatShape / SLatPbr .visualize_sample   (object un-rotated,  cam (0, 0,d) up Y)
#
# ProjGrid rotates latent-frame points by `_LATENT_TO_BLENDER_R` then projects
# with `front_view_transform_matrix` (camera-to-world) via inverse(T). The
# renderer (utils3d.extrinsics_look_at) uses the OPPOSITE camera handedness
# (looks +Z, not Blender's -Z), so renderer_ext = D @ inverse(T) with
# D = diag(1, -1, -1, 1).

# latent/mesh frame -> Blender frame (matches ProjGrid.__init__ rotation_matrix)
_LATENT_TO_BLENDER_R = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])
# projection (Blender, cam looks -Z) -> renderer (utils3d, cam looks +Z)
_PROJ_TO_RENDER_FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))


def front_view_transform_matrix(distance: float) -> torch.Tensor:
    """Canonical camera-to-world front view (matches ProjGrid), camera at
    (0, -distance, 0) looking at the origin with +Z up."""
    return torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -float(distance)],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32)


def render_homog_rotation() -> torch.Tensor:
    """(4,4) `_LATENT_TO_BLENDER_R` in the rotation block. Mesh stages render
    the object UN-rotated, so post-multiply their extrinsics by this."""
    H = torch.eye(4)
    H[:3, :3] = _LATENT_TO_BLENDER_R
    return H


def anchor_relative_render_cameras(
    anchor_T_world: torch.Tensor,
    view_T_worlds: torch.Tensor,
    fovs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Renderer-ready (extrinsics, intrinsics) to render the decoded
    representation from each view, aligned with that view's input photo.

    extrinsics[v] = D @ inverse(front_view(anchor_dist) @ relative_pose(anchor, view_v))

    This is the SS / object-rotated convention. For mesh stages (shape/pbr) that
    do NOT rotate the object, post-multiply each extrinsic by
    `render_homog_rotation()`.

    Args:
        anchor_T_world: (4,4) anchor camera-to-world (from transforms.json).
        view_T_worlds:  (V,4,4) every view's camera-to-world.
        fovs:           length-V iterable of camera_angle_x (radians) per view.

    Returns:
        exts (V,4,4) world-to-camera, ints (V,3,3).
    """
    import utils3d
    anchor_T_world = anchor_T_world.float()
    anchor_distance = float(torch.norm(anchor_T_world[:3, 3]))
    front = front_view_transform_matrix(anchor_distance)
    D = _PROJ_TO_RENDER_FLIP
    exts, ints = [], []
    for Tw, fov in zip(view_T_worlds, fovs):
        rel = relative_pose(anchor_T_world, Tw.float())
        T_v = front @ rel
        ext = D @ torch.linalg.inv(T_v.double()).float()
        fov_t = torch.tensor(float(fov))
        exts.append(ext)
        ints.append(utils3d.torch.intrinsics_from_fov_xy(fov_t, fov_t))
    return torch.stack(exts), torch.stack(ints)


def patch_proj_grid_for_multiview():
    """Replace `ProjGrid.forward` with a version that accepts a real
    `transform_matrix` (instead of asserting it must be None).

    The upstream training-time ProjGrid hard-asserts `transform_matrix is None`
    because single-view latents are pre-aligned to a canonical front-view
    frame. Multi-view training breaks that assumption: each view's image
    must be projected from its own pose (expressed in anchor's frame).
    This patch mirrors what `multiview_inference.py` does at inference time.

    Idempotent — safe to call from every spawn worker.
    """
    from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj as M

    if getattr(M.ProjGrid, '_multiview_patched', False):
        return

    project_points_to_image_batch = M.project_points_to_image_batch
    sample_features = M.sample_features

    def patched_forward(self, features_map, camera_angle_x, distance, mesh_scale,
                        transform_matrix=None, BHWC=True):
        if BHWC:
            B, H, W, C = features_map.shape
        else:
            B, C, H, W = features_map.shape

        grid_points = self.grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2

        if transform_matrix is None:
            T = self.front_view_transform_matrix.expand(B, -1, -1).clone()
            T[:, 1, 3] = -distance
        else:
            T = transform_matrix

        image_points, _, _ = project_points_to_image_batch(
            grid_points, T, camera_angle_x, self.image_resolution)
        image_points_norm = (image_points + 0.5) / self.image_resolution * 2 - 1

        if BHWC:
            features_map = features_map.permute(0, 3, 1, 2)
        x = sample_features(features_map, image_points_norm)
        x = x.permute(0, 2, 1)
        return x

    M.ProjGrid.forward = patched_forward
    M.ProjGrid._multiview_patched = True
