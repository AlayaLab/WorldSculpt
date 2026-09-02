"""
visualize_pointcloud.py — 3D visualization of the whole scene (all instances)
in metric world, with the camera trajectory + anchor.

Two separate figures (each with a 3D view + a top-down X-Z view):
  (A) MESH cloud  : every instance's Pixal3D mesh vertices mapped to world
                    via its own T_canon_to_metric.
  (B) BACKPROJ cloud: every instance's upstream fused point cloud (already world).

Each instance is a distinct color; cameras are the shared input poses (blue),
the anchor is a green star, short arrows = look-at direction.

OUTPUT (default <case_root>/_scene/):
  mesh_scene_3d.png      mesh_scene_topdown.png
  cloud_scene_3d.png     cloud_scene_topdown.png

USAGE:
  python visualize_pointcloud.py --case_root assets/video/cup_and_tea
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


_INSTANCE_COLORS = [
    (0.20, 0.45, 1.00),   # blue
    (0.85, 0.35, 0.10),   # orange/brown
    (0.15, 0.70, 0.20),   # green
    (0.80, 0.15, 0.80),   # magenta
    (0.10, 0.70, 0.70),   # teal
]

# canonical AABB [-0.5,0.5]^3 corners + 12 edges (index = i*4 + j*2 + k)
_CANON_CORNERS = np.array([(x, y, z) for x in (-.5, .5)
                                     for y in (-.5, .5)
                                     for z in (-.5, .5)], dtype=np.float64)
_CUBE_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7),   # X
               (0, 2), (1, 3), (4, 6), (5, 7),   # Y
               (0, 1), (2, 3), (4, 5), (6, 7)]   # Z


def load_mesh_world(mesh_pt_path, max_pts, rng):
    """Return (world_points, T_canon_to_metric)."""
    pt = torch.load(mesh_pt_path, map_location='cpu', weights_only=False)
    v = pt['vertices'].numpy().astype(np.float64)
    T = np.asarray(pt['T_canon_to_metric'], dtype=np.float64)
    vh = np.concatenate([v, np.ones((len(v), 1))], axis=1)
    vw = (T @ vh.T).T[:, :3]
    if len(vw) > max_pts:
        vw = vw[rng.choice(len(vw), max_pts, replace=False)]
    return vw, T


def cube_world_corners(T):
    """AABB [-0.5,0.5]^3 -> 8 world corners via T_canon_to_metric."""
    ch = np.concatenate([_CANON_CORNERS, np.ones((8, 1))], axis=1)
    return (T @ ch.T).T[:, :3]


def load_ply_world(ply_path, max_pts, rng):
    import trimesh
    pc = trimesh.load(str(ply_path), process=False)
    v = np.asarray(pc.vertices, dtype=np.float64)
    if len(v) > max_pts:
        v = v[rng.choice(len(v), max_pts, replace=False)]
    return v


def gather_cameras(transforms_json):
    """Return (cams[N,3], look_dirs[N,3], sub_ids, is_anchor[N])."""
    meta = json.load(open(transforms_json))
    cams, looks, subs, anch = [], [], [], []
    for fr in meta['frames']:
        c2w = np.array(fr['c2w_world_blender'], dtype=np.float64)
        cams.append(c2w[:3, 3])
        looks.append(-c2w[:3, 2])              # Blender looks at -Z
        subs.append(int(fr['subsample_idx']))
        anch.append(bool(fr.get('is_anchor')))
    return (np.array(cams), np.array(looks), subs, np.array(anch, dtype=bool))


def draw_scene(inst_clouds, colors, cams, looks, subs, is_anchor,
               title, out_3d, out_top, bounds=None, cubes=None):
    anchor_idx = int(np.where(is_anchor)[0][0]) if is_anchor.any() else -1
    if bounds is not None:
        mn, mx = bounds
    else:
        all_xyz = np.concatenate(list(inst_clouds.values()) + [cams], axis=0)
        mn = all_xyz.min(0) - 0.05
        mx = all_xyz.max(0) + 0.05
    scene_extent = float(np.linalg.norm(mx - mn))
    arrow = max(0.03, scene_extent * 0.025)

    # ---- 3D ----
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for (name, pts), col in zip(inst_clouds.items(), colors):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[col], s=2, alpha=0.35,
                   label=f'{name} ({len(pts)} pts)')
        if cubes is not None and name in cubes:
            cw = cubes[name]
            for a, b in _CUBE_EDGES:
                ax.plot([cw[a, 0], cw[b, 0]], [cw[a, 1], cw[b, 1]],
                        [cw[a, 2], cw[b, 2]], c=col, linewidth=1.6, alpha=0.95)
    if (~is_anchor).any():
        ax.scatter(cams[~is_anchor, 0], cams[~is_anchor, 1], cams[~is_anchor, 2],
                   c='royalblue', s=28, depthshade=False,
                   label=f'cams ({int((~is_anchor).sum())})')
    if anchor_idx >= 0:
        a = cams[anchor_idx]
        ax.scatter([a[0]], [a[1]], [a[2]], c='lime', marker='*', s=320,
                   depthshade=False, edgecolors='black', linewidths=1.2,
                   label=f'anchor (sub={subs[anchor_idx]:02d})')
    for i, (c, l) in enumerate(zip(cams, looks)):
        col = 'lime' if is_anchor[i] else 'cornflowerblue'
        ax.quiver(c[0], c[1], c[2], l[0]*arrow, l[1]*arrow, l[2]*arrow,
                  color=col, alpha=0.9 if is_anchor[i] else 0.45,
                  linewidth=1.4 if is_anchor[i] else 0.8)
    for i in range(len(cams)):
        if i % max(1, len(cams)//10) == 0 or is_anchor[i]:
            ax.text(cams[i, 0], cams[i, 1], cams[i, 2], f' {subs[i]:02d}',
                    fontsize=7, color='black')
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
    ax.set_xlabel('X (m, world)'); ax.set_ylabel('Y (m, world)'); ax.set_zlabel('Z (m, world)')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    try:
        ax.set_box_aspect((mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]))
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(out_3d, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'[viz] {out_3d}')

    # ---- top-down X-Z ----
    fig2, ax2 = plt.subplots(figsize=(11, 9))
    for (name, pts), col in zip(inst_clouds.items(), colors):
        ax2.scatter(pts[:, 0], pts[:, 2], c=[col], s=2, alpha=0.35, label=name)
        if cubes is not None and name in cubes:
            cw = cubes[name]
            for a, b in _CUBE_EDGES:
                ax2.plot([cw[a, 0], cw[b, 0]], [cw[a, 2], cw[b, 2]],
                         c=col, linewidth=1.3, alpha=0.95)
    if (~is_anchor).any():
        ax2.scatter(cams[~is_anchor, 0], cams[~is_anchor, 2], c='royalblue', s=28,
                    label='cams')
    if anchor_idx >= 0:
        ax2.scatter([cams[anchor_idx, 0]], [cams[anchor_idx, 2]], c='lime',
                    marker='*', s=320, edgecolors='black', linewidths=1.2,
                    label='anchor')
    a2 = max(0.03, scene_extent * 0.04)
    for i, (c, l) in enumerate(zip(cams, looks)):
        col = 'lime' if is_anchor[i] else 'cornflowerblue'
        ax2.arrow(c[0], c[2], l[0]*a2, l[2]*a2, head_width=a2*0.25,
                  head_length=a2*0.3, fc=col, ec=col, alpha=0.7)
        if i % max(1, len(cams)//10) == 0 or is_anchor[i]:
            ax2.annotate(f'{subs[i]:02d}', (cams[i, 0], cams[i, 2]),
                         textcoords='offset points', xytext=(4, 4), fontsize=7)
    ax2.set_xlim(mn[0], mx[0]); ax2.set_ylim(mn[2], mx[2])
    ax2.set_xlabel('X (m, world)'); ax2.set_ylabel('Z (m, world) — looking down from +Y')
    ax2.set_title(title + '  [top-down X-Z]')
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3); ax2.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_top, dpi=150, bbox_inches='tight'); plt.close(fig2)
    print(f'[viz] {out_top}')


def main():
    ap = argparse.ArgumentParser(description='Scene point-cloud viz (mesh + backproj) with cameras.')
    ap.add_argument('--case_root', type=str, required=True)
    ap.add_argument('--recon_dir', type=str, default=None,
                    help='Default <case_root>/_reconstruct_object')
    ap.add_argument('--instances', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Default <case_root>/_scene')
    ap.add_argument('--max_pts', type=int, default=8000,
                    help='Points shown per instance per figure.')
    args = ap.parse_args()

    case_root = Path(args.case_root).resolve()
    recon_dir = Path(args.recon_dir).resolve() if args.recon_dir else case_root / '_reconstruct_object'
    step1_pc = case_root / '_step1' / 'point_clouds'
    crops_root = case_root / '_crops'
    out_dir = Path(args.output_dir).resolve() if args.output_dir else case_root / '_scene'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.instances:
        instances = [s.strip() for s in args.instances.split(',')]
    else:
        instances = sorted([d.name for d in recon_dir.iterdir()
                            if d.is_dir() and (d / 'mesh.pt').exists()])
    print(f'[viz] instances = {instances}')
    colors = [_INSTANCE_COLORS[i % len(_INSTANCE_COLORS)] for i in range(len(instances))]

    # cameras (shared input poses; use first instance's transforms.json)
    cams = looks = subs = is_anchor = None
    for inst in instances:
        tj = crops_root / inst / 'transforms.json'
        if tj.exists():
            cams, looks, subs, is_anchor = gather_cameras(tj)
            break
    if cams is None:
        raise FileNotFoundError('No transforms.json found for cameras')
    print(f'[viz] {len(cams)} cameras, anchor sub={subs[int(np.where(is_anchor)[0][0])] if is_anchor.any() else "?"}')

    rng = np.random.default_rng(0)

    # ---- Load both clouds + each instance's world AABB cube ----
    mesh_clouds, cubes = {}, {}
    for inst in instances:
        mp = recon_dir / inst / 'mesh.pt'
        if mp.exists():
            mesh_clouds[inst], T = load_mesh_world(mp, args.max_pts, rng)
            cubes[inst] = cube_world_corners(T)
            print(f'[viz] mesh  {inst}: {len(mesh_clouds[inst])} pts')
    bp_clouds = {}
    for inst in instances:
        pp = step1_pc / f'{inst}.ply'
        if pp.exists():
            bp_clouds[inst] = load_ply_world(pp, args.max_pts, rng)
            print(f'[viz] cloud {inst}: {len(bp_clouds[inst])} pts')

    # ---- Shared bounds over mesh + cloud + cubes + cameras (so both figures match) ----
    all_pts = (list(mesh_clouds.values()) + list(bp_clouds.values())
               + list(cubes.values()) + [cams])
    allxyz = np.concatenate(all_pts, axis=0)
    bounds = (allxyz.min(0) - 0.05, allxyz.max(0) + 0.05)
    print(f'[viz] shared bounds: min={bounds[0].round(3).tolist()} max={bounds[1].round(3).tolist()}')

    # (A) mesh cloud
    if mesh_clouds:
        draw_scene(mesh_clouds, colors, cams, looks, subs, is_anchor,
                   'Scene: Pixal3D MESH vertices + AABB (metric world) + cameras',
                   out_dir / 'mesh_scene_3d.png', out_dir / 'mesh_scene_topdown.png',
                   bounds=bounds, cubes=cubes)
    # (B) backprojected cloud (upstream fused PLY)
    if bp_clouds:
        draw_scene(bp_clouds, colors, cams, looks, subs, is_anchor,
                   'Scene: back-projected cloud + AABB (metric world) + cameras',
                   out_dir / 'cloud_scene_3d.png', out_dir / 'cloud_scene_topdown.png',
                   bounds=bounds, cubes=cubes)

    print('[viz] Done.')


if __name__ == '__main__':
    main()
