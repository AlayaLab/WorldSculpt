"""
prepare_crops_video.py — Pixal3D multi-view fine-tune data generation.

For each instance, produces a NeRF-style transforms.json + per-frame cropped
RGBA images. **Crop center = canonical origin's projection** (NOT SAM bbox
center) so Pixal3D's K-centered ProjGrid assumption is satisfied exactly
(canonical origin always at the crop's center pixel).

Math (per instance i, per keyframe j):
    c2w_world_blender_j = (c2w_world_opencv_j with t/=scale) @ diag(1,-1,-1,1)
    R_canon = R_box.T @ c2w_world_blender_j[:3, :3]
    t_canon = R_box.T @ (c2w_world_blender_j[:3, 3] - center_world) / size
    c2w_canon_ij = [[R_canon, t_canon], [0, 1]]   # SE(3) in canonical space

Crop:
    u_proj, v_proj = projection of T_canon_to_metric @ (0,0,0,1) into frame j
    cube_8_corners projection → image bbox extent
    half_w = max(extent_u, extent_v) / 2 * (1 + pad)
    crop_bbox = (u_proj - half_w, v_proj - half_w, u_proj + half_w, v_proj + half_w)

INPUT:
  <case_root>/frames/<NNNN>.png
  <case_root>/masks/<instance>/<NNNN>.png
  <case_root>/_step1/{vipe,da3,moge_anchor}_out.npz
  <case_root>/_step2/transforms.npz

OUTPUT (default <case_root>/_crops/):
  <instance>/
    transforms.json
    0000.png ... NNNN.png   RGBA crops (alpha = SAM mask)
    alignments/view{NN}.jpg (--save_alignments)

USAGE:
  python prepare_crops_video.py --case_root assets/video/cup_and_tea
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Optional

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np
import cv2
from PIL import Image, ImageDraw


# =============================================================================
# Geometry helpers
# =============================================================================

def crop_fov_from_full_intrinsics(crop_w_pix: float, fx_full_pix: float) -> float:
    return 2.0 * math.atan(crop_w_pix / (2.0 * fx_full_pix))


def project_canonical_origin(T_canon_to_metric: np.ndarray,
                              w2c_opencv_4x4: np.ndarray,
                              K_pix: np.ndarray):
    """Project canonical (0, 0, 0) (cube center in canonical) → image pixel.
    Returns (u, v, z_cam) in OpenCV cam coords. None if behind camera."""
    center_world = T_canon_to_metric[:3, 3]
    center_h = np.append(center_world, 1.0)
    center_cam = w2c_opencv_4x4 @ center_h
    z = float(center_cam[2])
    if z <= 1e-3:
        return None
    u = float(K_pix[0, 0] * center_cam[0] / z + K_pix[0, 2])
    v = float(K_pix[1, 1] * center_cam[1] / z + K_pix[1, 2])
    return u, v, z


def project_cube_corners(T_canon_to_metric: np.ndarray,
                          w2c_opencv_4x4: np.ndarray,
                          K_pix: np.ndarray):
    """Project the canonical [-0.5, 0.5]^3 cube's 8 corners through
    T → world → cam → image. Returns (uv_8x2, valid_8)."""
    corners = np.array([
        (x, y, z) for x in (-0.5, 0.5)
                  for y in (-0.5, 0.5)
                  for z in (-0.5, 0.5)
    ], dtype=np.float64)
    c_h = np.concatenate([corners, np.ones((8, 1))], axis=1)
    c_world = (T_canon_to_metric @ c_h.T).T[:, :3]
    c_h2 = np.concatenate([c_world, np.ones((8, 1))], axis=1)
    c_cam = (w2c_opencv_4x4 @ c_h2.T).T[:, :3]
    z = c_cam[:, 2]
    valid = z > 1e-3
    safe_z = np.where(valid, z, 1.0)
    u = K_pix[0, 0] * (c_cam[:, 0] / safe_z) + K_pix[0, 2]
    v = K_pix[1, 1] * (c_cam[:, 1] / safe_z) + K_pix[1, 2]
    return np.stack([u, v], axis=1), valid


def isolate_object_crop_rgba(image_rgb: np.ndarray, mask_u8: np.ndarray,
                               bbox):
    """Crop to bbox (may be out of image bounds), pad with zeros, return RGBA
    where alpha = mask (also cropped + padded)."""
    H, W = image_rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    cw, ch = x1 - x0, y1 - y0
    rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
    alpha = np.zeros((ch, cw), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - x0, sy0 - y0
        dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
        rgb[dy0:dy1, dx0:dx1] = image_rgb[sy0:sy1, sx0:sx1]
        alpha[dy0:dy1, dx0:dx1] = mask_u8[sy0:sy1, sx0:sx1]
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return rgba


def clean_alpha_open(rgba: np.ndarray, kernel: int, iters: int) -> np.ndarray:
    """Morphological opening on the alpha channel: erode `iters`x with a
    `kernel`x`kernel` box, then dilate `iters`x back. Drops silhouette
    specks/fringe while preserving the main blob's size. Finally zero the RGB
    wherever alpha became 0. No-op if kernel<2 or iters<1."""
    if kernel < 2 or iters < 1:
        return rgba
    rgba = rgba.copy()
    k = np.ones((kernel, kernel), np.uint8)
    a = cv2.erode(rgba[..., 3], k, iterations=iters)
    a = cv2.dilate(a, k, iterations=iters)
    rgba[..., 3] = a
    rgba[a == 0, :3] = 0
    return rgba


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Step 3: Pixal3D multi-view fine-tune data generation.')
    p.add_argument('--case_root', type=str, required=True)
    p.add_argument('--step1_dir', type=str, default=None,
                   help='Default <case_root>/_step1')
    p.add_argument('--step2_dir', type=str, default=None,
                   help='Default <case_root>/_step2')
    p.add_argument('--output_dir', type=str, default=None,
                   help='Default <case_root>/_crops')
    p.add_argument('--pad', type=float, default=0.10,
                   help='Crop padding as fraction of cube projection extent. '
                        'Default 0.10 → cube fills ~91%% of crop (close to '
                        "Pixal3D's 88%% training distribution).")
    p.add_argument('--min_mask_pixels', type=int, default=500,
                   help='Skip frames where this instance mask has fewer pixels.')
    p.add_argument('--crop_resolution', type=int, default=0,
                   help='If > 0, resize each crop to this resolution (square). '
                        '0 = keep variable size (per-frame).')
    p.add_argument('--save_alignments', action='store_true',
                   help='Save per-frame [full+overlay | RGBA crop] for inspection.')
    p.add_argument('--alpha_erode_kernel', type=int, default=3,
                   help='kxk box kernel for the saved-alpha morphological opening '
                        '(erode then dilate). <2 disables.')
    p.add_argument('--alpha_erode_iters', type=int, default=1,
                   help='Erode-then-dilate iterations on the saved alpha. Removes '
                        'silhouette specks; RGB is zeroed where alpha==0. 0 disables.')
    args = p.parse_args()

    case_root = Path(args.case_root).resolve()
    step1_dir = Path(args.step1_dir).resolve() if args.step1_dir else case_root / '_step1'
    step2_dir = Path(args.step2_dir).resolve() if args.step2_dir else case_root / '_step2'
    out_dir = (Path(args.output_dir).resolve()
               if args.output_dir else case_root / '_crops')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[crops] case_root  = {case_root}')
    print(f'[crops] step1_dir  = {step1_dir}')
    print(f'[crops] step2_dir  = {step2_dir}')
    print(f'[crops] output_dir = {out_dir}')

    # ---- Load step1 outputs ----
    vipe = dict(np.load(step1_dir / 'vipe_out.npz'))
    da3 = dict(np.load(step1_dir / 'da3_out.npz'))
    moge = dict(np.load(step1_dir / 'moge_anchor.npz'))
    scale = float(moge['scale'][0])
    sub_indices = list(da3['sub_indices'].tolist())

    # ---- Load step2 outputs (schema-tolerant) ----
    # Works for both step2_cube_fit.py (shared R_box + __center_world + anchor)
    # and step2_boxer.py (per-instance oriented box: __T + __size + __center).
    # Everything the cropping stage needs is derivable from the canon->world transform __T and
    # the cube side __size:  __T[:3,:3] == size * R_box  (cube => isotropic), so
    # R_box = __T[:3,:3] / size  and  center = __T[:3, 3].
    step2 = dict(np.load(step2_dir / 'transforms.npz', allow_pickle=True))
    instances = [str(s) for s in step2['instances']]
    inst_T = {n: step2[f'{n}__T'].astype(np.float64) for n in instances}
    inst_size = {n: float(np.asarray(step2[f'{n}__size']).ravel()[0]) for n in instances}

    def _center(n):
        for k in (f'{n}__center_world', f'{n}__center'):
            if k in step2:
                return np.asarray(step2[k], dtype=np.float64)
        return inst_T[n][:3, 3].astype(np.float64)
    inst_center = {n: _center(n) for n in instances}
    # Per-instance box orientation (cube_fit shares one; boxer is per-instance).
    inst_Rbox = {n: inst_T[n][:3, :3] / inst_size[n] for n in instances}

    # Anchor frame: prefer step2, then step1 MoGe, else the middle keyframe.
    if 'anchor_full_idx' in step2:
        anchor_full_idx = int(np.asarray(step2['anchor_full_idx']).ravel()[0])
    elif 'anchor_full_idx' in moge:
        anchor_full_idx = int(np.asarray(moge['anchor_full_idx']).ravel()[0])
    else:
        anchor_full_idx = int(sub_indices[len(sub_indices) // 2])

    print(f'[crops] {len(sub_indices)} keyframes, {len(instances)} instances')
    print(f'[crops] anchor (full idx) = {anchor_full_idx}')
    print(f'[crops] MoGe scale factor = {scale:.6f}')

    # ---- Frame paths ----
    frames_dir = case_root / 'frames'
    frame_paths_all = sorted([p for p in frames_dir.iterdir()
                              if p.suffix.lower() == '.png'])
    masks_dir = case_root / 'masks'

    fid_to_idx = {int(fid): i for i, fid in enumerate(vipe['frame_ids'])}

    # ---- Per instance ----
    for inst_name in instances:
        print(f'\n--- {inst_name} ---')
        inst_out = out_dir / inst_name
        inst_out.mkdir(parents=True, exist_ok=True)
        if args.save_alignments:
            (inst_out / 'alignments').mkdir(parents=True, exist_ok=True)

        T = inst_T[inst_name]
        size = inst_size[inst_name]
        center_world = inst_center[inst_name]
        R_box_T = inst_Rbox[inst_name].T  # per-instance box orientation

        frames_meta = []
        n_skipped_mask = 0
        n_skipped_other = 0
        for sub_i, full_i in enumerate(sub_indices):
            # Mask
            mask_path = masks_dir / inst_name / (frame_paths_all[full_i].stem + '.png')
            if not mask_path.exists():
                n_skipped_mask += 1
                continue
            mask = np.array(Image.open(mask_path).convert('L'))
            if int((mask > 127).sum()) < args.min_mask_pixels:
                n_skipped_mask += 1
                continue

            # camera-to-world (OpenCV convention), scaled to metric
            if full_i not in fid_to_idx:
                n_skipped_other += 1
                continue
            c2w_opencv = vipe['c2w'][fid_to_idx[full_i]].astype(np.float64)
            c2w_opencv_metric = c2w_opencv.copy()
            c2w_opencv_metric[:3, 3] /= scale
            # OpenCV cam → Blender cam (flip cam Y, Z)
            c2w_blender = c2w_opencv_metric @ np.diag([1.0, -1.0, -1.0, 1.0])

            # ---- Construct c2w_canon as SE(3) in canonical space ----
            c2w_canon = np.eye(4, dtype=np.float64)
            c2w_canon[:3, :3] = R_box_T @ c2w_blender[:3, :3]
            c2w_canon[:3, 3]  = R_box_T @ (c2w_blender[:3, 3] - center_world) / size

            # K at full resolution (from the input cameras)
            K_full = vipe['K_pix'][fid_to_idx[full_i]].astype(np.float64).copy()
            fx_full = float(K_full[0, 0])
            fy_full = float(K_full[1, 1])
            cx_full = float(K_full[0, 2])
            cy_full = float(K_full[1, 2])

            # Load image to determine size
            img = np.array(Image.open(frame_paths_all[full_i]).convert('RGB'))
            H_full, W_full = img.shape[:2]
            if mask.shape != (H_full, W_full):
                mask = cv2.resize(mask, (W_full, H_full),
                                  interpolation=cv2.INTER_NEAREST)

            # Project canonical origin (OpenCV w2c)
            w2c_opencv_metric = np.linalg.inv(c2w_opencv_metric)
            origin_proj = project_canonical_origin(T, w2c_opencv_metric, K_full)
            if origin_proj is None:
                n_skipped_other += 1
                continue
            u_center, v_center, _ = origin_proj

            # Project cube corners → extent
            uv_corners, valid = project_cube_corners(T, w2c_opencv_metric, K_full)
            if int(valid.sum()) < 4:
                n_skipped_other += 1
                continue
            uvv = uv_corners[valid]
            u_min, v_min = uvv.min(axis=0)
            u_max, v_max = uvv.max(axis=0)
            # Square side from larger of {U-extent, V-extent} * (1 + pad).
            # CRITICAL: center the crop *exactly* on (u_center, v_center) so
            # canonical origin always projects to crop center → Pixal3D's
            # K-centered ProjGrid assumption holds across views, and resize
            # preserves multi-view alignment.
            cube_ext = max(u_max - u_min, v_max - v_min)
            side = int(round(cube_ext * (1.0 + args.pad)))
            # Center crop strictly on (u_center, v_center)
            x0 = int(round(u_center - side / 2.0))
            y0 = int(round(v_center - side / 2.0))
            x1 = x0 + side
            y1 = y0 + side
            crop_w = side

            # crop fov (camera_angle_x)
            fov_x = crop_fov_from_full_intrinsics(crop_w, fx_full)

            # radius in canonical units (camera distance to canonical origin)
            radius = float(np.linalg.norm(c2w_canon[:3, 3]))

            # Crop image (RGBA: alpha = mask)
            rgba = isolate_object_crop_rgba(img, mask, (x0, y0, x1, y1))

            # Optional resize
            if args.crop_resolution > 0 and rgba.shape[0] != args.crop_resolution:
                rgba = cv2.resize(rgba,
                                  (args.crop_resolution, args.crop_resolution),
                                  interpolation=cv2.INTER_AREA)

            # Clean saved alpha: open (erode→dilate) to drop silhouette specks,
            # then zero RGB where alpha==0. Applied at final resolution so the
            # kernel size is in saved-image pixels.
            rgba = clean_alpha_open(rgba, args.alpha_erode_kernel, args.alpha_erode_iters)

            fname = f'{sub_i:04d}.png'
            Image.fromarray(rgba).save(inst_out / fname)

            # Crop-frame K (full K minus bbox offset). fx, fy unchanged; cx, cy
            # translated so the crop's K still maps the same 3D rays correctly.
            cx_crop = cx_full - x0
            cy_crop = cy_full - y0
            # If --crop_resolution applied, scale K accordingly. K_image_pix
            # is the K matching the actually-saved PNG (post-resize).
            image_size_px = (args.crop_resolution
                             if args.crop_resolution > 0 else crop_w)
            resize_scale = image_size_px / crop_w
            fx_image = fx_full * resize_scale
            fy_image = fy_full * resize_scale
            cx_image = cx_crop * resize_scale
            cy_image = cy_crop * resize_scale

            frames_meta.append({
                'file_path': fname,
                # ---- Intrinsics (multiple representations) ----
                'camera_angle_x': float(fov_x),     # crop horizontal FOV (rad)
                'K_image_pix': [                    # K matching the saved PNG
                    [float(fx_image), 0.0, float(cx_image)],
                    [0.0, float(fy_image), float(cy_image)],
                    [0.0, 0.0, 1.0],
                ],
                'K_crop_pix': [                     # K at crop-pixel scale (pre-resize)
                    [fx_full, 0.0, float(cx_crop)],
                    [0.0, fy_full, float(cy_crop)],
                    [0.0, 0.0, 1.0],
                ],
                'K_full_pix': [                     # K at full original image scale
                    [fx_full, 0.0, cx_full],
                    [0.0, fy_full, cy_full],
                    [0.0, 0.0, 1.0],
                ],
                'image_size_px': int(image_size_px),

                # ---- Extrinsics (canonical + metric world) ----
                'transform_matrix': c2w_canon.tolist(),       # SE(3) c2w in canonical (Blender)
                'c2w_world_blender': c2w_blender.tolist(),    # SE(3) c2w in metric world (Blender)
                'w2c_world_opencv': w2c_opencv_metric.tolist(),  # w2c in metric world (OpenCV)
                'radius': radius,                              # ||c2w_canon[:3, 3]||

                # ---- Provenance (debugging / re-projection / multi-view sampling) ----
                'full_frame_idx': int(full_i),
                'subsample_idx': int(sub_i),
                'crop_bbox': [int(x0), int(y0), int(x1), int(y1)],
                'crop_center_in_full_image': [float(u_center), float(v_center)],
                'crop_size': int(crop_w),
                'full_frame_size_hw': [int(H_full), int(W_full)],
                'full_frame_path': str(frame_paths_all[full_i]),
                'is_anchor': bool(full_i == anchor_full_idx),
            })

            # Alignment viz
            if args.save_alignments:
                pil = Image.fromarray(img.copy())
                d = ImageDraw.Draw(pil)
                # Crop bbox (yellow)
                d.rectangle([(x0, y0), (x1, y1)],
                            outline=(255, 255, 0), width=3)
                # Cube edges (magenta)
                edges = [
                    (0, 4), (1, 5), (2, 6), (3, 7),
                    (0, 2), (1, 3), (4, 6), (5, 7),
                    (0, 1), (2, 3), (4, 5), (6, 7),
                ]
                for a, b in edges:
                    if valid[a] and valid[b]:
                        d.line([(uv_corners[a, 0], uv_corners[a, 1]),
                                (uv_corners[b, 0], uv_corners[b, 1])],
                               fill=(255, 0, 255), width=2)
                # Crop center (green dot = projected canonical origin)
                r = 6
                d.ellipse([(u_center - r, v_center - r),
                           (u_center + r, v_center + r)],
                          fill=(0, 255, 0))
                full_with_overlay = np.array(pil)

                # Side-by-side: full image with overlay | RGBA crop composited on black
                rgba_disp = rgba.copy()
                rgb_black = (rgba_disp[..., :3].astype(np.float32)
                             * rgba_disp[..., 3:4].astype(np.float32) / 255.0
                             ).astype(np.uint8)
                target_h = H_full // 2
                left_w = int(W_full * target_h / H_full)
                left = cv2.resize(full_with_overlay, (left_w, target_h),
                                    interpolation=cv2.INTER_AREA)
                right = cv2.resize(rgb_black, (target_h, target_h),
                                     interpolation=cv2.INTER_AREA)
                side = np.concatenate([left, right], axis=1)
                Image.fromarray(side).save(
                    inst_out / 'alignments' / f'view{sub_i:02d}.jpg', quality=95)

        # transforms.json
        transforms_meta = {
            'aabb': [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            'scale': float(size),                # cube size in metric (m)
            'offset': center_world.tolist(),     # cube center in metric world
            'frames': frames_meta,
            # Provenance
            'anchor_full_idx': anchor_full_idx,
            'moge_scale_factor': scale,
            'R_box': inst_Rbox[inst_name].tolist(),  # box axes in world (per instance)
        }
        with open(inst_out / 'transforms.json', 'w') as f:
            json.dump(transforms_meta, f, indent=2)
        n_kept = len(frames_meta)
        print(f'  saved {n_kept} frames (skipped {n_skipped_mask} no-mask, '
              f'{n_skipped_other} other) → {inst_out}/')

    print('\n[crops] Done.')


if __name__ == '__main__':
    main()
