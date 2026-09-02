"""
prepare_crops_scene.py — GT-camera / GT-bbox variant of prepare_crops_video.py for
the synthetic Toys4k multi-object scene test set.

Unlike the video pipeline (estimated cameras + estimated bbox),
the rendered scenes carry GROUND-TRUTH cameras and per-object boxes in their
own transforms.json / scene_meta.json. So we skip step1+step2 entirely and
build the *exact same* per-instance fine-tune-data contract that
reconstruct_object.py / compose_scene.py already consume.

GT scene dir (read-only test set), e.g.
  <scene_dir>/
    NNN.png                       full RGB frames (000.png ... )
    transforms.json               camera_model=OPENGL, fl_x/fl_y/cx/cy/w/h,
                                  frames[].transform_matrix (c2w, Blender world),
                                  instances[].{pass_index,aabb_world,center_world,...}
    masks/objNN/NNNN.png          per-instance binary masks (objNN <-> pass_index NN)

This script writes a self-contained, standard-layout *work* case_root (symlinks,
no copies of the big PNGs) so reconstruction/5/6 run unmodified:
  <case_root>/
    frames/NNNN.png                       -> symlink to <scene_dir>/NNN.png
    masks/objNN/NNNN.png                  -> symlink to GT mask
    _crops/objNN/
        transforms.json                   same schema as prepare_crops_video.py
        NNNN.png                          RGBA crops (alpha = GT mask)
        alignments/viewNN.jpg             (--save_alignments)

Canonical frame per instance (matches single-object Toys4k normalisation):
  R_box  = I                          (box axes aligned to world)
  center = aabb_world centre (== GT center_world)
  size   = max world-AABB extent      (longest axis spans the [-0.5,0.5] cube)
  T_canon_to_metric = [[size*R_box, center], [0,1]]

Per frame j (GT transform_matrix is c2w in Blender/OpenGL world):
  c2w_blender = transform_matrix_j
  c2w_opencv  = c2w_blender @ diag(1,-1,-1,1)
  w2c_opencv  = inv(c2w_opencv)
  c2w_canon   = [[R_box.T @ c2w_blender[:3,:3],
                 R_box.T @ (c2w_blender[:3,3]-center)/size], [0,1]]

USAGE:
  python prepare_crops_scene.py \
      --scene_dir input/<dataset>/<scene> \
      --case_root output/<scene> \
      --crop_resolution 1024 --save_alignments
"""

import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm


def _pbar(seq, desc):
    """Progress bar for the two silent full-resolution decode loops (a 4K frame
    takes ~0.3 s, so a 16-view object sits quiet for ~10 s without one).
    Auto-disabled when stderr is not a terminal, to keep batch logs clean."""
    return tqdm(seq, desc=desc, total=len(seq), unit='view', leave=False,
                disable=not sys.stderr.isatty())


def _save_png(arr, path):
    """Encode+write on a worker thread (zlib/jpeg encode releases the GIL).
    Format follows the path suffix; .jpg/.jpeg paths get quality=95."""
    kw = {'quality': 95} if str(path).lower().endswith(('.jpg', '.jpeg')) else {}
    Image.fromarray(arr).save(path, **kw)


# Reuse the exact, tested geometry / crop helpers from the video pipeline.
from prepare_crops_video import (
    crop_fov_from_full_intrinsics,
    project_canonical_origin,
    project_cube_corners,
    isolate_object_crop_rgba,
    clean_alpha_open,
)

# Blender/OpenGL cam (+X right, +Y up, -Z fwd) <-> OpenCV cam (+X right, +Y down,
# +Z fwd). The flip matrix is its own inverse.
BLENDER_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def _symlink(src: Path, dst: Path):
    """Create/refresh a dst -> src symlink, making parent dirs.

    The target is stored RELATIVE to dst's directory: with the default layout
    (scene dir and its `_case/` work tree under one root) the link becomes e.g.
    `../../000.png`, so the whole tree can be moved, copied or re-mounted
    without breaking. Absolute targets would silently dangle after any such
    relocation.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass
    rel = os.path.relpath(os.path.abspath(src), os.path.abspath(dst.parent))
    os.symlink(rel, dst)


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest proper rotation (det=+1) to R via SVD."""
    U, _, Vt = np.linalg.svd(R.astype(np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _mask_fit_scale(valid_area, mask_cache, frame_records, K_full,
                    W_full, H_full, R_box, size, center_world, pct):
    """Isotropic cube up-scale (center fixed) so that its projected 2D bbox
    contains the instance mask 2D bbox in the kept frames.

    Rationale: the point-cloud AABB from wild_to_scene can be undersized
    (DBSCAN keeps only the largest cluster; confidence gating starves
    low-texture objects). The masks are the most reliable evidence in the
    chain, so they provide a multi-view lower bound on the cube size. The
    2D-bbox criterion matches exactly how the pass-2 crop box is computed
    (projected-corner extent), so the fitted cube guarantees the crop never
    cuts the mask in the covered frames.

    Per kept frame: binary-search the minimal s_i >= 1 with
    proj-corner-bbox(size*s_i) ⊇ mask-bbox; the returned scale is the `pct`
    percentile over {s_i} (100 = strict max; lower tolerates drifted-mask
    outlier frames). Frames with corners behind the camera (indeterminate
    projection) or unsatisfiable within s<=64 are skipped.

    Returns (s_fit >= 1, s_max, n_frames_fit, n_not_covered, n_skipped).
    """
    def _contained(s, w2c, mb):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = (size * s) * R_box
        T[:3, 3] = center_world
        uv, valid = project_cube_corners(T, w2c, K_full)
        if not valid.all():
            return None                          # indeterminate (corner behind cam)
        u0, v0 = uv.min(axis=0)
        u1, v1 = uv.max(axis=0)
        return (u0 <= mb[0] and v0 <= mb[1] and u1 >= mb[2] and v1 >= mb[3])

    s_list, n_skipped = [], 0
    for full_i in sorted(valid_area):
        mu8 = mask_cache[full_i]
        ys, xs = np.nonzero(mu8 > 127)
        mh, mw = mu8.shape
        sx, sy = W_full / mw, H_full / mh        # mask may be lower-res than frames
        mb = (xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy)
        w2c = frame_records[full_i][4]

        c1 = _contained(1.0, w2c, mb)
        if c1 is None:
            n_skipped += 1
            continue
        if c1:
            s_list.append(1.0)                   # already covered; clamp at 1 anyway
            continue
        hi, ok = 1.0, False
        while hi <= 64.0:                        # find an upper bracket by doubling
            hi *= 2.0
            c = _contained(hi, w2c, mb)
            if c is None:
                break
            if c:
                ok = True
                break
        if not ok:
            n_skipped += 1
            continue
        lo = hi / 2.0
        for _ in range(24):                      # bisect to ~1e-6 relative
            mid = 0.5 * (lo + hi)
            if _contained(mid, w2c, mb):
                hi = mid
            else:
                lo = mid
        s_list.append(hi)

    if not s_list:
        return 1.0, 1.0, 0, 0, n_skipped
    s_arr = np.asarray(s_list)
    s_fit = max(1.0, float(np.percentile(s_arr, pct)))
    n_not_covered = int((s_arr > s_fit + 1e-9).sum())
    return s_fit, float(s_arr.max()), len(s_list), n_not_covered, n_skipped


def _cube_from_aabb(aabb: np.ndarray, R_box: np.ndarray):
    """Isotropic cube enclosing the world AABB, oriented by R_box.

    Returns (size, center_world). The 8 AABB corners are expressed in the box
    frame (v_box = R_box.T @ (v - center)); size = the largest per-axis extent
    there (so the cube tightly encloses the object along the box axes, for any
    R_box). For R_box=I this reduces to size = max(aabb extent), center = AABB
    centre — identical to the original world-aligned behaviour.
    """
    mn, mx = aabb[0], aabb[1]
    center = (mn + mx) / 2.0
    corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                        for y in (mn[1], mx[1])
                        for z in (mn[2], mx[2])], dtype=np.float64)
    corners_box = (corners - center) @ R_box      # == (R_box.T @ (corner-center))
    ext = corners_box.max(axis=0) - corners_box.min(axis=0)
    return float(ext.max()), center


def _mask_px_in_crop(mask_u8: np.ndarray, box, full_wh) -> int:
    """Mask pixels the crop box actually covers. Box and full_wh in frame px.

    A box is allowed to hang off the frame — isolate_object_crop_rgba zero-pads
    it. What is not acceptable is a box that covers none of the object: it
    becomes an all-black, all-transparent "view" that no stage downstream flags.

    That happens whenever the cube CENTRE projects outside the frame while the
    corner extent still looks healthy — the two come from different quantities
    (centre from the canonical origin, side from the corner spread), so they
    degenerate independently. A camera sitting inside the object's cube is the
    common trigger: four corners fall behind it and are dropped by
    project_cube_corners, the four survivors share one depth and hand back a
    tidy square, while the centre sits a few cm in front of the camera and the
    perspective divide throws it far off-frame.
    """
    W_full, H_full = full_wh
    h, w = mask_u8.shape[:2]
    x0, y0, x1, y1 = box
    if (w, h) != (W_full, H_full):        # masks may be stored at another scale
        sx, sy = w / float(W_full), h / float(H_full)
        x0, x1 = int(np.floor(x0 * sx)), int(np.ceil(x1 * sx))
        y0, y1 = int(np.floor(y0 * sy)), int(np.ceil(y1 * sy))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((mask_u8[y0:y1, x0:x1] > 127).sum())


def main():
    p = argparse.ArgumentParser(
        description='GT-camera/GT-bbox fine-tune data generation for scene test set.')
    p.add_argument('--scene_dir', type=str, required=True,
                   help='GT scene dir (contains transforms.json, NNN.png, masks/).')
    p.add_argument('--case_root', type=str, required=True,
                   help='Output work case_root (standard layout; symlinks GT frames/masks).')
    p.add_argument('--pad', type=float, default=0.005,
                   help='Crop padding as fraction of cube projection extent.')
    p.add_argument('--min_mask_pixels', type=int, default=500,
                   help='Skip frames where this instance mask has fewer pixels.')
    p.add_argument('--min_mask_ratio', type=float, default=0.0,
                   help='Skip frames where the mask covers less than this fraction '
                        'of the image (e.g. 0.005 = 0.5%%). Quality filter for '
                        'edge-clipped/occluded views; applied on top of '
                        '--min_mask_pixels. NOT a substitute for --max_crop_px: '
                        'crop blow-up is driven by cube-corner projection, not '
                        'mask size (a 2.8%%-mask frame can still yield a 900k-px crop).')
    p.add_argument('--crop_resolution', type=int, default=1024,
                   help='If > 0, resize each crop to this square resolution.')
    p.add_argument('--max_crop_ratio', type=float, default=4.0,
                   help='Skip frames whose square crop side exceeds this multiple of '
                        'the larger full-image dimension. Real-video cameras can '
                        'pass right next to an object; its cube corners then project '
                        'at extreme off-axis angles and the crop side explodes '
                        '(100k+ px -> tens of GB of allocation). Such views are '
                        'degenerate for training anyway, so they are dropped.')
    p.add_argument('--save_alignments', action='store_true',
                   help='Save per-frame [full+overlay | RGBA crop] for inspection.')
    p.add_argument('--alpha_erode_kernel', type=int, default=3)
    p.add_argument('--alpha_erode_iters', type=int, default=3)
    p.add_argument('--mask_fit_scale', action='store_true',
                   help='Enlarge each canonical cube (center fixed) until its 2D '
                        'projection contains the mask bbox in the kept frames '
                        '(percentile over per-frame minimal scales, >=1, no cap; '
                        'warn above 3). Safety net for undersized point-cloud AABBs.')
    p.add_argument('--mask_fit_pct', type=float, default=95.0,
                   help='Percentile over per-frame minimal scales (100 = strict max).')
    p.add_argument('--anchor_frame', choices=['world', 'anchor'], default='anchor',
                   help="Canonical box orientation R_box. 'world' (default): axis-aligned "
                        "to world (R_box=I) -> anchor lands at an arbitrary angle in the grid. "
                        "'anchor': align R_box to the ANCHOR camera's orientation, so the anchor "
                        "view sits at the canonical front-view pose (matches MV training, where "
                        "the object is rotated so the anchor is the front view). Cube size is then "
                        "the tight extent of the world AABB measured along the box axes.")
    args = p.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    case_root = Path(args.case_root).resolve()
    out_dir = case_root / '_crops'
    frames_link_dir = case_root / 'frames'
    masks_link_root = case_root / 'masks'
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(scene_dir / 'transforms.json') as f:
        T_meta = json.load(f)

    fl_x = float(T_meta['fl_x']); fl_y = float(T_meta['fl_y'])
    cx = float(T_meta['cx']); cy = float(T_meta['cy'])
    K_full = np.array([[fl_x, 0.0, cx],
                       [0.0, fl_y, cy],
                       [0.0, 0.0, 1.0]], dtype=np.float64)

    gt_frames = T_meta['frames']
    instances = T_meta['instances']
    n_frames = len(gt_frames)

    print(f'[gt] scene_dir  = {scene_dir}')
    print(f'[gt] case_root  = {case_root}')
    print(f'[gt] {n_frames} frames, {len(instances)} instances, '
          f'full res {T_meta["w"]}x{T_meta["h"]}, fl_x={fl_x:.2f}')

    # ---- Pre-compute per-frame extrinsics + symlink frames into the work tree ----
    # full_frame_idx i == position in transforms.json frames list == sorted order.
    frame_records = []  # (full_i, gt_png_path, work_png_path, c2w_blender, w2c_opencv)
    for i, fr in enumerate(gt_frames):
        gt_png = scene_dir / fr['file_path']
        work_png = frames_link_dir / f'{i:04d}.png'
        _symlink(gt_png, work_png)
        c2w_blender = np.array(fr['transform_matrix'], dtype=np.float64)
        c2w_opencv = c2w_blender @ BLENDER_OPENCV
        w2c_opencv = np.linalg.inv(c2w_opencv)
        frame_records.append((i, gt_png, work_png, c2w_blender, w2c_opencv))

    # ---- Perf: full-frame decode cache + threaded PNG writes -----------------
    # Frames are shared across instances; decoding a 4K PNG costs ~0.25s, so cache
    # the decoded RGB per frame index (~25 MB/frame at 4K; ~100-frame scenes fit
    # comfortably). PNG *encoding* dominates the per-crop cost — offload every
    # save to a small thread pool (PIL's zlib encode releases the GIL). Futures
    # are drained per instance so pending arrays stay bounded.
    frame_rgb_cache = {}

    def _frame_rgb(full_i, gt_png):
        if full_i not in frame_rgb_cache:
            frame_rgb_cache[full_i] = np.array(Image.open(gt_png).convert('RGB'))
        return frame_rgb_cache[full_i]

    saver = ThreadPoolExecutor(max_workers=8)
    save_futures = []

    # ---- Per instance ----
    scene_anchors = {}   # inst_name -> anchor info (also mirrored in each transforms.json)
    for inst in instances:
        pidx = int(inst['pass_index'])
        inst_name = f'obj{pidx:02d}'
        print(f'\n--- {inst_name}  ({inst.get("file_identifier", "?")}) ---')

        # Wild scenes: instances whose point-cloud box estimation failed upstream
        # (wild_to_scene "box omitted") carry no aabb_world — nothing to build a
        # canonical cube from, skip the instance instead of crashing.
        if not inst.get('aabb_world'):
            print(f'  SKIP: no aabb_world (box estimation failed upstream)')
            continue

        aabb = np.array(inst['aabb_world'], dtype=np.float64)  # [[min],[max]]
        # Degenerate box (point cloud collapsed to ~a point): cube size would be
        # ~0 -> zero-size crops / divide-by-zero downstream. Same fate as no-box.
        if float((aabb[1] - aabb[0]).max()) < 1e-3:
            print(f'  SKIP: degenerate aabb_world '
                  f'(max extent {float((aabb[1] - aabb[0]).max()):.4f} m)')
            continue

        # Tight oriented box (object canonical frame), when the scene provides it
        # (HouseCat6D conversions: gt_scales extents + obj_pose_world R,t). The
        # canonical ±0.5 cube through T_obb == the metric OBB, so it projects via
        # the same project_cube_corners helper. Toys4k scenes lack these -> None.
        T_obb = None
        if 'gt_scales' in inst and 'obj_pose_world' in inst:
            P_obj = np.array(inst['obj_pose_world'], dtype=np.float64)
            T_obb = P_obj.copy()
            T_obb[:3, :3] = P_obj[:3, :3] @ np.diag(np.asarray(inst['gt_scales'],
                                                               dtype=np.float64))

        gt_mask_dir = scene_dir / 'masks' / inst_name
        work_mask_dir = masks_link_root / inst_name

        # ---- Pass 1: per-frame mask area -> pick anchor (largest visible mask). ----
        # Anchor must be known BEFORE R_box when --anchor_frame=anchor.
        valid_area = {}  # full_i -> mask pixel count (mask exists & >= min_mask_pixels)
        mask_cache = {}  # full_i -> raw uint8 L mask (reused verbatim in pass 2)
        for (full_i, gt_png, work_png, c2w_blender, w2c_opencv) in _pbar(
                frame_records, f'  {inst_name} masks'):
            gt_mask = gt_mask_dir / f'{full_i:04d}.png'
            if not gt_mask.exists():
                continue
            mu8 = np.array(Image.open(gt_mask).convert('L'))
            m = mu8 > 127
            a = int(m.sum())
            if a >= max(args.min_mask_pixels, args.min_mask_ratio * m.size):
                valid_area[full_i] = a
                mask_cache[full_i] = mu8
        # ---- Canonical box orientation (R_box) / size / centre ----
        # Depends on the anchor when --anchor_frame=anchor, so it is a helper:
        # anchor selection below re-evaluates it per candidate.
        def _canon_for(anchor_idx):
            if args.anchor_frame == 'anchor' and anchor_idx >= 0:
                # Align canonical axes to the anchor camera so the anchor sits at
                # the canonical front-view pose (matches MV training).
                c2w_anchor = frame_records[anchor_idx][3]   # c2w_blender of anchor
                R = _orthonormalize(c2w_anchor[:3, :3])
            else:
                R = np.eye(3, dtype=np.float64)              # world-axis-aligned
            sz, ctr = _cube_from_aabb(aabb, R)
            if T_obb is not None:
                # Tight cube: enclose the OBB corners directly in R axes. Going
                # through the world AABB first (above) inflates twice for rotated
                # objects; the OBB centre equals the AABB centre, so only size changes.
                corners = np.array([(x, y, z) for x in (-0.5, 0.5)
                                    for y in (-0.5, 0.5)
                                    for z in (-0.5, 0.5)], dtype=np.float64)
                cw = corners @ T_obb[:3, :3].T + T_obb[:3, 3]
                ctr = T_obb[:3, 3].copy()
                ext = (cw - ctr) @ R
                sz = float((ext.max(0) - ext.min(0)).max())
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = sz * R
            T[:3, 3] = ctr
            return R, sz, ctr, T

        max_side = (args.max_crop_ratio * max(int(T_meta['w']), int(T_meta['h']))
                    if args.max_crop_ratio > 0 else None)
        # Same "is there enough object here" bar pass 1 used to accept the frame,
        # re-applied to what the crop box actually covers.
        full_wh = (int(T_meta['w']), int(T_meta['h']))
        min_crop_px = int(max(args.min_mask_pixels,
                              args.min_mask_ratio * full_wh[0] * full_wh[1]))

        def _crop_box(full_i, T_canon):
            """Crop box (x0, y0, x1, y1, side) in frame px; None if unprojectable.

            Same geometry pass 2 applies: centre from the canonical origin, side
            from the spread of the cube corners that are in front of the camera.
            """
            w2c = frame_records[full_i][4]
            origin = project_canonical_origin(T_canon, w2c, K_full)
            if origin is None:
                return None
            uvc, val = project_cube_corners(T_canon, w2c, K_full)
            if int(val.sum()) < 4:
                return None
            uvv = uvc[val]
            ext = max(uvv[:, 0].max() - uvv[:, 0].min(),
                      uvv[:, 1].max() - uvv[:, 1].min())
            side = int(round(ext * (1.0 + args.pad)))
            x0 = int(round(origin[0] - side / 2.0))
            y0 = int(round(origin[1] - side / 2.0))
            return x0, y0, x0 + side, y0 + side, side

        # Anchor = largest visible mask whose own crop is usable. A close-up frame
        # can have a huge mask AND a degenerate crop; such a frame must not define
        # R_box — otherwise it gets dropped in pass 2 and the re-picked anchor no
        # longer sits at the canonical front-view pose. Two ways to degenerate:
        #   * cube corners near the camera plane explode the projected extent
        #     (caught by --max_crop_ratio), and
        #   * the cube centre projects off-frame while the extent stays healthy,
        #     so the box lands on no object at all (see _mask_px_in_crop). The
        #     ordering walks straight into this one: largest mask == closest
        #     camera == most likely to be inside the cube.
        anchor_full_idx = -1
        for cand in sorted(valid_area, key=valid_area.get, reverse=True):
            box = _crop_box(cand, _canon_for(cand)[3])
            if box is None:
                print(f'  [anchor] frame {cand:04d} rejected: unprojectable')
                continue
            if max_side is not None and box[4] > max_side:
                print(f'  [anchor] frame {cand:04d} rejected: '
                      f'crop side {box[4]} (limit {max_side:.0f})')
                continue
            px = _mask_px_in_crop(mask_cache[cand], box[:4], full_wh)
            if px < min_crop_px:
                print(f'  [anchor] frame {cand:04d} rejected: crop box covers '
                      f'{px} mask px (limit {min_crop_px}) — off-frame centre')
                continue
            anchor_full_idx = cand
            break

        R_box, size, center_world, T_canon_to_metric = _canon_for(anchor_full_idx)
        R_box_T = R_box.T
        if T_obb is not None:
            size_world, _ = _cube_from_aabb(aabb, R_box)
            print(f'  cube size: tight-OBB {size:.4f} m '
                  f'(world-AABB route would give {size_world:.4f} m)')

        # ---- Mask-fit cube up-scale (center fixed; see _mask_fit_scale) ----
        if args.mask_fit_scale and valid_area:
            s_fit, s_max, n_fit, n_unc, n_skip = _mask_fit_scale(
                valid_area, mask_cache, frame_records, K_full,
                float(T_meta['w']), float(T_meta['h']),
                R_box, size, center_world, args.mask_fit_pct)
            if s_fit > 1.0 + 1e-6:
                size *= s_fit
                T_canon_to_metric = np.eye(4, dtype=np.float64)
                T_canon_to_metric[:3, :3] = size * R_box
                T_canon_to_metric[:3, 3] = center_world
            print(f'  [mask-fit] s={s_fit:.3f} (p{args.mask_fit_pct:g} of {n_fit} frames; '
                  f'max {s_max:.3f}; {n_unc} not fully covered; {n_skip} skipped) '
                  f'-> cube {size:.4f} m')
            if s_fit > 3.0:
                print(f'  [mask-fit] WARNING: scale {s_fit:.2f} > 3 — the point-cloud '
                      f'box is badly undersized OR some masks drifted; check '
                      f'{scene_dir}/vis and masks/{inst_name}.')

        inst_out = out_dir / inst_name
        inst_out.mkdir(parents=True, exist_ok=True)
        if args.save_alignments:
            (inst_out / 'alignments').mkdir(parents=True, exist_ok=True)

        frames_meta = []
        n_skip_mask = 0
        n_skip_crop = 0
        n_skip_empty = 0

        # ---- Pass 2: crop / project / save ----
        for (full_i, gt_png, work_png, c2w_blender, w2c_opencv) in _pbar(
                frame_records, f'  {inst_name} crops'):
            if full_i not in valid_area:
                n_skip_mask += 1
                continue
            # GT mask is 4-digit (NNNN.png). Symlink it under the 4-digit work stem
            # so reconstruct_object.py / compose_scene.py (masks_dir/(frame_stem+'.png')) resolve directly.
            gt_mask = gt_mask_dir / f'{full_i:04d}.png'
            _symlink(gt_mask, work_mask_dir / f'{full_i:04d}.png')
            mask = mask_cache[full_i]           # raw uint8, loaded once in pass 1

            img = _frame_rgb(full_i, gt_png)    # cached across instances (read-only!)
            H_full, W_full = img.shape[:2]
            if mask.shape != (H_full, W_full):
                mask = cv2.resize(mask, (W_full, H_full),
                                  interpolation=cv2.INTER_NEAREST)

            # Project canonical origin (crop centre) + cube corners (extent).
            origin_proj = project_canonical_origin(T_canon_to_metric, w2c_opencv, K_full)
            if origin_proj is None:
                n_skip_mask += 1
                continue
            u_center, v_center, _ = origin_proj
            uv_corners, valid = project_cube_corners(T_canon_to_metric, w2c_opencv, K_full)
            if int(valid.sum()) < 4:
                n_skip_mask += 1
                continue
            uvv = uv_corners[valid]
            u_min, v_min = uvv.min(axis=0)
            u_max, v_max = uvv.max(axis=0)
            cube_ext = max(u_max - u_min, v_max - v_min)
            side = int(round(cube_ext * (1.0 + args.pad)))
            if side > args.max_crop_ratio * max(W_full, H_full):
                n_skip_crop += 1
                continue
            x0 = int(round(u_center - side / 2.0))
            y0 = int(round(v_center - side / 2.0))
            x1, y1 = x0 + side, y0 + side
            # Hanging off the frame is fine (zero-padded); covering none of the
            # object is not — that silently yields an all-black crop.
            if _mask_px_in_crop(mask, (x0, y0, x1, y1),
                                (W_full, H_full)) < min_crop_px:
                n_skip_empty += 1
                continue
            crop_w = side

            fov_x = crop_fov_from_full_intrinsics(crop_w, fl_x)

            # c2w in canonical (Blender) frame.
            c2w_canon = np.eye(4, dtype=np.float64)
            c2w_canon[:3, :3] = R_box_T @ c2w_blender[:3, :3]
            c2w_canon[:3, 3] = R_box_T @ (c2w_blender[:3, 3] - center_world) / size
            radius = float(np.linalg.norm(c2w_canon[:3, 3]))

            rgba = isolate_object_crop_rgba(img, mask, (x0, y0, x1, y1))
            if args.crop_resolution > 0 and rgba.shape[0] != args.crop_resolution:
                rgba = cv2.resize(rgba, (args.crop_resolution, args.crop_resolution),
                                  interpolation=cv2.INTER_AREA)
            rgba = clean_alpha_open(rgba, args.alpha_erode_kernel, args.alpha_erode_iters)

            fname = f'{full_i:04d}.png'
            save_futures.append(saver.submit(_save_png, rgba, inst_out / fname))

            cx_crop = cx - x0
            cy_crop = cy - y0
            image_size_px = args.crop_resolution if args.crop_resolution > 0 else crop_w
            rs = image_size_px / crop_w
            fx_image, fy_image = fl_x * rs, fl_y * rs
            cx_image, cy_image = cx_crop * rs, cy_crop * rs

            frames_meta.append({
                'file_path': fname,
                'camera_angle_x': float(fov_x),
                'K_image_pix': [[float(fx_image), 0.0, float(cx_image)],
                                [0.0, float(fy_image), float(cy_image)],
                                [0.0, 0.0, 1.0]],
                'K_crop_pix': [[fl_x, 0.0, float(cx_crop)],
                               [0.0, fl_y, float(cy_crop)],
                               [0.0, 0.0, 1.0]],
                'K_full_pix': [[fl_x, 0.0, cx],
                               [0.0, fl_y, cy],
                               [0.0, 0.0, 1.0]],
                'image_size_px': int(image_size_px),
                'transform_matrix': c2w_canon.tolist(),
                'c2w_world_blender': c2w_blender.tolist(),
                'w2c_world_opencv': w2c_opencv.tolist(),
                'radius': radius,
                'full_frame_idx': int(full_i),
                'subsample_idx': int(full_i),
                'crop_bbox': [int(x0), int(y0), int(x1), int(y1)],
                'crop_center_in_full_image': [float(u_center), float(v_center)],
                'crop_size': int(crop_w),
                'full_frame_size_hw': [int(H_full), int(W_full)],
                'full_frame_path': str(work_png),
                'is_anchor': bool(full_i == anchor_full_idx),
            })

            if args.save_alignments:
                pil = Image.fromarray(img.copy()); d = ImageDraw.Draw(pil)
                d.rectangle([(x0, y0), (x1, y1)], outline=(255, 255, 0), width=3)
                edges = [(0, 4), (1, 5), (2, 6), (3, 7), (0, 2), (1, 3),
                         (4, 6), (5, 7), (0, 1), (2, 3), (4, 5), (6, 7)]
                for a, b in edges:
                    if valid[a] and valid[b]:
                        d.line([(uv_corners[a, 0], uv_corners[a, 1]),
                                (uv_corners[b, 0], uv_corners[b, 1])],
                               fill=(255, 0, 255), width=2)
                if T_obb is not None:
                    uv_obb, valid_obb = project_cube_corners(T_obb, w2c_opencv, K_full)
                    for a, b in edges:
                        if valid_obb[a] and valid_obb[b]:
                            d.line([(uv_obb[a, 0], uv_obb[a, 1]),
                                    (uv_obb[b, 0], uv_obb[b, 1])],
                                   fill=(0, 255, 255), width=2)
                r = 6
                d.ellipse([(u_center - r, v_center - r), (u_center + r, v_center + r)],
                          fill=(0, 255, 0))
                full_overlay = np.array(pil)
                rgb_black = (rgba[..., :3].astype(np.float32)
                             * rgba[..., 3:4].astype(np.float32) / 255.0).astype(np.uint8)
                th = H_full // 2
                lw = int(W_full * th / H_full)
                left = cv2.resize(full_overlay, (lw, th), interpolation=cv2.INTER_AREA)
                right = cv2.resize(rgb_black, (th, th), interpolation=cv2.INTER_AREA)
                save_futures.append(saver.submit(
                    _save_png, np.concatenate([left, right], axis=1),
                    inst_out / 'alignments' / f'view{full_i:02d}.jpg'))

        # Drain pending PNG writes for this instance (bounds queued arrays and
        # surfaces any write error before transforms.json is written); free the
        # per-instance mask cache.
        for fut in save_futures:
            fut.result()
        save_futures.clear()
        mask_cache.clear()

        # The anchor (largest mask == closest camera) may itself have been
        # dropped by a pass-2 crop guard; re-pick it among saved frames so
        # anchor_full_idx / is_anchor never reference a missing crop. R_box
        # stays aligned to the original anchor camera — still a valid
        # orientation choice, just no longer an existing view's exact pose.
        saved_idxs = {f['full_frame_idx'] for f in frames_meta}
        if anchor_full_idx >= 0 and anchor_full_idx not in saved_idxs and frames_meta:
            new_anchor = max(saved_idxs, key=lambda i: valid_area[i])
            print(f'  NOTE: anchor {anchor_full_idx} was dropped (bad crop); '
                  f're-picked anchor {new_anchor}')
            anchor_full_idx = new_anchor
            for fm in frames_meta:
                fm['is_anchor'] = bool(fm['full_frame_idx'] == anchor_full_idx)

        transforms_meta = {
            'aabb': [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            'scale': float(size),
            'offset': center_world.tolist(),
            'frames': frames_meta,
            'anchor_full_idx': int(anchor_full_idx),
            'anchor_frame': args.anchor_frame,
            'moge_scale_factor': 1.0,        # GT is already metric; kept for schema parity
            'R_box': R_box.tolist(),
            'gt_source': str(scene_dir),
            'pass_index': pidx,
            'file_identifier': inst.get('file_identifier'),
            'sha256': inst.get('sha256'),          # for GT latent lookup in metric eval
            'aabb_world': aabb.tolist(),
        }
        with open(inst_out / 'transforms.json', 'w') as f:
            json.dump(transforms_meta, f, indent=2)

        anchor_file = (f'{anchor_full_idx:04d}.png' if anchor_full_idx >= 0 else None)
        scene_anchors[inst_name] = {
            'anchor_full_idx': int(anchor_full_idx),
            'anchor_file': anchor_file,                 # crop filename in this inst dir
            'anchor_scene_frame': anchor_file,          # == GT scene frame stem (4-digit)
            'file_identifier': inst.get('file_identifier'),
            'pass_index': pidx,
            'n_views': len(frames_meta),
        }
        print(f'  saved {len(frames_meta)} frames (skipped {n_skip_mask} mask, '
              f'{n_skip_crop} oversized-crop, {n_skip_empty} empty-crop), '
              f'anchor full_idx={anchor_full_idx}, frame={args.anchor_frame}, '
              f'cube size={size:.4f} m -> {inst_out}/')

    # ---- Scene-level anchor summary (one glance: which view is each object's anchor) ----
    summary = {
        'scene': scene_dir.name,
        'anchor_frame_mode': args.anchor_frame,
        'note': 'anchor = view with the largest visible mask; same index in world/anchor modes.',
        'anchors': scene_anchors,
    }
    saver.shutdown(wait=True)
    with open(out_dir / 'anchor_views.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n[gt] anchor summary -> {out_dir / "anchor_views.json"}')
    print('[gt] Done.')


if __name__ == '__main__':
    main()
