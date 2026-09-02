"""
compose_scene.py — compose all per-instance off-center reconstructions into
one metric-world scene, render it through every keyframe camera (depth-composited
occlusion) next to the original frame, and export a single scene GLB.

Per view, only the instances actually visible in that frame are rendered — i.e.
those with a (non-empty) mask crop there, read from each instance's crop
transforms.json `frames` list. This skips compositing the whole scene into every
frame (the old, slow behavior) and scales with per-frame object count, not scene
size.

Each instance's mesh.pt (from reconstruct_object) is in canonical [-0.5,0.5]^3
and carries its own `T_canon_to_metric`. Mapping every instance to metric world
puts them all in the SAME metric world frame (they already align with the video),
so the scene is just their union.

INPUT (default):
  <case_root>/_reconstruct_object/<instance>/mesh.pt        (voxel state + T)
  <case_root>/_crops/<instance>/transforms.json    (per-instance cameras)

OUTPUT (default <case_root>/_scene/):
  scene.glb              multi-object scene (per-object PBR materials), metric world
  scene_mesh.glb         merged single mesh (one geometry), metric world
  renders/view{NN}.jpg   [ orig | scene base_color | scene over orig ]
                         (--normal: [ orig | normal | per-instance color | overlay ],
                          instance colors shaded by camera-space normal + name legend)
  renders/grid.png

The GLBs are rebuilt from each instance's mesh.pt via o_voxel.to_glb (full voxel
state, self-contained), NOT from reconstruct_object.py's mesh_world.glb.

USAGE:
  python compose_scene.py --case_root assets/video/cup_and_tea
"""

import os
import sys
import json
import argparse
import colorsys
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault('ATTN_BACKEND', 'flash_attn')
os.environ['FLEX_GEMM_AUTOTUNE_CACHE_PATH'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _hwc3(t):
    if t.dim() == 3 and t.shape[0] == 3:
        return t.permute(1, 2, 0)
    if t.dim() == 2:
        return t.unsqueeze(-1).expand(-1, -1, 3)
    return t


def load_mesh_with_voxel(mesh_pt_path, device, face_budget):
    from pixal3d.representations import MeshWithVoxel
    pt = torch.load(mesh_pt_path, map_location='cpu', weights_only=False)
    origin = pt['origin']
    origin = origin.tolist() if hasattr(origin, 'tolist') else list(origin)
    mesh = MeshWithVoxel(
        vertices=pt['vertices'], faces=pt['faces'], origin=origin,
        voxel_size=float(pt['voxel_size']), coords=pt['coords'],
        attrs=pt['attrs'], voxel_shape=pt['voxel_shape'],
        layout=dict(pt['layout']),
    ).to(device)
    if int(mesh.faces.shape[0]) > face_budget:
        mesh.simplify(face_budget)
    T = np.asarray(pt['T_canon_to_metric'], dtype=np.float64) if 'T_canon_to_metric' in pt \
        else None
    return mesh, T


def main():
    ap = argparse.ArgumentParser(description='Compose per-instance recons into one scene + render.')
    ap.add_argument('--case_root', type=str, required=True)
    ap.add_argument('--recon_dir', type=str, default=None,
                    help='Default <case_root>/_reconstruct_object')
    ap.add_argument('--instances', type=str, default=None,
                    help='Comma-separated; default = all under recon_dir.')
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Default <case_root>/_scene')
    ap.add_argument('--render_frames', type=str, default='all',
                    help="'all' or comma-separated subsample indices.")
    ap.add_argument('--preview_long', type=int, default=1280)
    ap.add_argument('--full_res', action='store_true',
                    help='Render each column at the full RGB resolution (long side = '
                         'max(W,H) of the frame) instead of --preview_long.')
    ap.add_argument('--ssaa', type=int, default=2)
    ap.add_argument('--peel_layers', type=int, default=4)
    ap.add_argument('--face_budget', type=int, default=3_000_000,
                    help='Decimate each instance mesh above this many faces (render only).')
    ap.add_argument('--glb_decimation', type=int, default=1_000_000,
                    help='Face budget per instance when baking GLB from mesh.pt.')
    ap.add_argument('--texture_size', type=int, default=4096,
                    help='PBR texture size when baking GLB from mesh.pt.')
    ap.add_argument('--no_glb', action='store_true', help='Skip scene GLB export.')
    ap.add_argument('--no_render', action='store_true', help='Skip per-view renders.')
    ap.add_argument('--white_bg', action='store_true',
                    help='Render panels on a WHITE background (uses the depth hit '
                         'mask, so no black anti-alias fringe).')
    ap.add_argument('--no_video', action='store_true',
                    help='Skip scene.mp4 (use when sharding frames across jobs).')
    ap.add_argument('--shard', type=str, default='',
                    help="'i,n': render only cam_list[i::n] (intra-scene parallelism; "
                         'shards share the output dir, filenames never collide).')
    ap.add_argument('--video_fps', type=int, default=10,
                    help='FPS of the per-view render video (scene.mp4).')
    ap.add_argument('--video_crf', type=int, default=28,
                    help='libx264 CRF for scene.mp4: higher = smaller file '
                         '(23 ~ default quality, 28 ~ noticeably smaller).')
    ap.add_argument('--raw_geometry', action='store_true',
                    help='with --normal, export the raw marching-cubes surface '
                         '(no fill_holes, no remesh) — only decimated when it is '
                         'above --glb_decimation. Faster, but ragged.')
    ap.add_argument('--normal', action='store_true',
                    help='GEOMETRY mode: render per-view NORMALS (not textured base_color) and '
                         'bake geometry-only scene GLBs. Works with texture-free mesh.pt '
                         '(reconstruction --no_tex, mode=offcenter_mv_geom) which has no PBR attrs.')
    ap.add_argument('--gt_world_dir', default=None,
                    help='With --normal: also render the Blender-dumped scene-world GT meshes '
                         '(<gt_world_dir>/<scene>/objNN.npz from dump_scene_gt_world.py) as normals, '
                         'so each view panel is [orig | GT-normal | pred-normal].')
    args = ap.parse_args()

    case_root = Path(args.case_root).resolve()
    recon_dir = Path(args.recon_dir).resolve() if args.recon_dir else case_root / '_reconstruct_object'
    crops_root = case_root / '_crops'
    out_dir = Path(args.output_dir).resolve() if args.output_dir else case_root / '_scene'
    render_dir = out_dir / 'renders'
    out_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    # ---- Discover instances ----
    if args.instances:
        instances = [s.strip() for s in args.instances.split(',')]
    else:
        instances = sorted([d.name for d in recon_dir.iterdir()
                            if d.is_dir() and (d / 'mesh.pt').exists()])
    if not instances:
        raise FileNotFoundError(f'No instance mesh.pt found under {recon_dir}')
    print(f'[scene] case_root  = {case_root}')
    print(f'[scene] recon_dir  = {recon_dir}')
    print(f'[scene] instances  = {instances}')
    print(f'[scene] output_dir = {out_dir}')

    # ---- Camera list (merge keyframes across instances; cameras are shared) ----
    cams = {}  # full_frame_idx -> dict(sub, w2c, K_full, path, insts)
    for inst in instances:
        tj = crops_root / inst / 'transforms.json'
        if not tj.exists():
            continue
        meta = json.load(open(tj))
        for fr in meta['frames']:
            fi = int(fr['full_frame_idx'])
            if fi not in cams:
                cams[fi] = dict(
                    sub=int(fr['subsample_idx']),
                    w2c=np.array(fr['w2c_world_opencv'], dtype=np.float64),
                    K_full=np.array(fr['K_full_pix'], dtype=np.float64),
                    path=fr['full_frame_path'],
                    insts=set(),
                )
            # this instance has a (non-empty) mask crop in this frame -> it is the
            # set of meshes actually visible here; the render only composites these
            # instead of the whole scene (huge speedup on scenes with many objects).
            cams[fi]['insts'].add(inst)
    cam_list = [cams[k] for k in sorted(cams)]
    if args.render_frames != 'all':
        want = set(int(s) for s in args.render_frames.split(','))
        cam_list = [c for c in cam_list if c['sub'] in want]
    print(f'[scene] {len(cam_list)} keyframe cameras')
    if args.shard:
        si, sn = (int(x) for x in args.shard.split(','))
        cam_list = cam_list[si::sn]
        print(f'[scene] shard {si}/{sn}: {len(cam_list)} cameras in this job')

    # Remeshed canonical geometry, filled in by the GLB stage and reused by the
    # renderer so both show the same surface. Declared out here because the render
    # path reads it even when --no_glb skipped the stage that fills it.
    remeshed, remeshed_f = {}, {}

    # ---- Export scene GLB(s) — rebuilt from each instance's mesh.pt (full voxel
    #      state, self-contained; NOT the reconstruction mesh_world.glb) ----
    if not args.no_glb:
        import trimesh
        import o_voxel
        scene = trimesh.Scene()
        baked_geoms = []   # world-baked Trimesh per geometry (for the merged mesh)
        n_added = 0
        for inst in instances:
            mp = recon_dir / inst / 'mesh.pt'
            if not mp.exists():
                print(f'[scene] WARN: {mp} missing, skip in GLB')
                continue
            pt = torch.load(mp, map_location='cuda', weights_only=False)
            if 'T_canon_to_metric' not in pt:
                print(f'[scene] WARN: {inst} mesh.pt has no T_canon_to_metric, skip')
                continue
            T = np.asarray(pt['T_canon_to_metric'], dtype=np.float64)
            if args.normal and not args.raw_geometry:
                # Geometry-only GLB *with the same post-processing as the textured
                # path*: reconstruct_object.remesh_geometry_glb reproduces to_glb's
                # geometry stages verbatim (fill_holes(3e-2) -> remesh_narrow_band_dc
                # -> simplify). Without it the exported surface is the raw
                # marching-cubes output, which is where the ragged edges come from:
                # to_glb(remesh=True) below cannot be used with --no_tex because
                # attr_volume/coords only exist when Stage 3 (texture) ran.
                from reconstruct_object import remesh_geometry_glb
                glb = remesh_geometry_glb(pt['vertices'], pt['faces'],
                                          int(pt['res_grid']),
                                          decimation_target=args.glb_decimation,
                                          band=1.0, project_back=0.0)
                # remesh_geometry_glb applies to_glb's (x,y,z)->(x,z,-y) swap, so it
                # needs the same R_g_inv undo the textured branch does below.
                R_g_inv = np.eye(4); R_g_inv[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
                glb.apply_transform(T @ R_g_inv)
                # Hand the SAME surface to the renderer. It reloads mesh.pt and would
                # otherwise draw the raw marching-cubes surface while the GLB shows the
                # cleaned one. Cached in canonical space (undo T) because the render
                # loop applies T itself; remeshing twice is not an option, it is the
                # most expensive step here.
                remeshed[inst] = np.linalg.solve(
                    T, np.concatenate([np.asarray(glb.vertices),
                                       np.ones((len(glb.vertices), 1))], axis=1).T).T[:, :3]
                remeshed_f[inst] = np.asarray(glb.faces)
            elif args.normal:
                # Raw fallback (--raw_geometry): mesh.pt verts @ T, no hole filling
                # and no remesh. Decimate per object (cumesh, same as to_glb) — raw
                # marching-cubes meshes are ~5M faces each; 55 undecimated objects
                # overflow the GLB uint32 size field (4 GiB) -> corrupt file.
                V, F = pt['vertices'].cuda(), pt['faces'].cuda()
                if args.glb_decimation and V.shape[0] > args.glb_decimation:
                    import cumesh
                    cm = cumesh.CuMesh()
                    cm.init(V, F)
                    cm.simplify(args.glb_decimation)
                    cm.remove_duplicate_faces()
                    V, F = cm.read()
                Vloc = np.asarray(V.cpu(), dtype=np.float64)
                Vw = (T @ np.concatenate([Vloc, np.ones((len(Vloc), 1))], axis=1).T).T[:, :3]
                glb = trimesh.Trimesh(vertices=Vw, faces=np.asarray(F.cpu()),
                                      process=False)
            else:
                # canonical [-0.5,0.5]^3 textured GLB straight from the voxel state
                glb = o_voxel.postprocess.to_glb(
                    vertices=pt['vertices'].cuda(), faces=pt['faces'].cuda(),
                    attr_volume=pt['attrs'].cuda(), coords=pt['coords'].cuda(),
                    attr_layout=pt['layout'], grid_size=int(pt['res_grid']),
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=args.glb_decimation, texture_size=args.texture_size,
                    remesh=True, remesh_band=1, remesh_project=0, use_tqdm=False)
                # to_glb internally rotates the mesh by ~R_x(-90deg): (x,y,z)->(x,z,-y).
                # Undo it (R_g^-1) BEFORE placing into world, so the GLB matches the
                # renderer's placement (which uses mesh.pt verts @ T directly). Verified:
                # T @ R_g_inv @ to_glb_out  ==  T @ mesh.pt_verts.
                R_g_inv = np.eye(4); R_g_inv[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
                glb.apply_transform(T @ R_g_inv)   # canonical(to_glb frame) -> metric world
            # collect geometry (bake any residual node transform), add to scene
            if isinstance(glb, trimesh.Scene):
                for node_name in glb.graph.nodes_geometry:
                    T_node, gname = glb.graph[node_name]
                    geom = glb.geometry[gname].copy()
                    if not np.allclose(np.asarray(T_node), np.eye(4)):
                        geom.apply_transform(np.asarray(T_node))
                    scene.add_geometry(geom, geom_name=f'{inst}__{gname}')
                    baked_geoms.append(geom)
            else:
                scene.add_geometry(glb, geom_name=inst)
                baked_geoms.append(glb)
            n_added += 1
            del pt
            torch.cuda.empty_cache()
            print(f'[scene] baked {inst} from mesh.pt')

        # GLB stores lengths as uint32 — a >4GiB buffer silently wraps around and
        # the exported file is corrupt (unreadable in Blender). Warn early.
        est = sum(len(g.vertices) * 12 + len(g.faces) * 12 for g in baked_geoms)
        if est >= 2**32:
            print(f'[scene] WARN: estimated GLB buffer {est / 2**30:.1f} GiB exceeds the '
                  f'4 GiB GLB limit — export will be CORRUPT. Lower --glb_decimation.')

        # (1) multi-object scene (keeps per-object PBR materials)
        scene_glb = out_dir / 'scene.glb'
        try:
            scene.export(str(scene_glb))
            print(f'[scene] saved {scene_glb} ({n_added} instances, per-object materials)')
        except Exception as e:
            print(f'[scene] WARN: scene GLB export failed: {type(e).__name__}: {e}')

        # (2) merged single mesh (one geometry; handy for geometry processing)
        if baked_geoms:
            try:
                merged = trimesh.util.concatenate(baked_geoms)
                mesh_glb = out_dir / 'scene_mesh.glb'
                merged.export(str(mesh_glb))
                print(f'[scene] saved {mesh_glb} (merged single mesh: '
                      f'V={len(merged.vertices):,} F={len(merged.faces):,})')
            except Exception as e:
                print(f'[scene] WARN: merged scene mesh export failed: {type(e).__name__}: {e}')

    if args.no_render:
        print('[scene] Done (no render).')
        return

    # ---- Load meshes for rendering ----
    device = 'cuda'
    if args.normal:
        from pixal3d.renderers import MeshRenderer
        from pixal3d.representations import Mesh
    else:
        from pixal3d.renderers import PbrMeshRenderer, EnvMap
        envmap = {'_': EnvMap(torch.zeros((2, 4, 3), dtype=torch.float32, device=device))}

    def _load_geom(mp):
        pt = torch.load(mp, map_location='cpu', weights_only=False)
        T = pt.get('T_canon_to_metric')
        if T is None:
            return None, None
        inst_name = mp.parent.name
        if inst_name in remeshed:      # cleaned surface from the GLB stage
            m = Mesh(torch.as_tensor(remeshed[inst_name], dtype=torch.float32,
                                     device=device),
                     torch.as_tensor(remeshed_f[inst_name], dtype=torch.int32,
                                     device=device))
        else:
            m = Mesh(pt['vertices'].float().to(device), pt['faces'].int().to(device))
        # Honor --face_budget here too (it previously applied only to the
        # textured load path): reconstruction meshes reach 5-12M faces each, and per-view
        # depth-peeled rendering over the undecimated scene is prohibitively slow.
        if args.face_budget and int(m.faces.shape[0]) > args.face_budget:
            n0 = int(m.faces.shape[0])
            m.simplify(args.face_budget)
            print(f'[scene] decimated {mp.parent.name}: F={n0:,} -> {int(m.faces.shape[0]):,}')
        return m, np.asarray(T, dtype=np.float64)

    meshes = []   # (inst, mesh, T_torch, v_world_h)
    for inst in instances:
        if args.normal:
            mesh, T = _load_geom(recon_dir / inst / 'mesh.pt')
        else:
            mesh, T = load_mesh_with_voxel(recon_dir / inst / 'mesh.pt', device, args.face_budget)
        if T is None:
            print(f'[scene] WARN: {inst} mesh.pt has no T_canon_to_metric, skip render')
            continue
        v = mesh.vertices.detach().cpu().numpy().astype(np.float64)
        v_h = np.concatenate([v, np.ones((len(v), 1))], axis=1)
        v_world = (T @ v_h.T).T[:, :3]
        v_world_h = np.concatenate([v_world, np.ones((len(v_world), 1))], axis=1)
        meshes.append((inst, mesh, torch.from_numpy(T.astype(np.float32)).to(device), v_world_h))
        print(f'[scene] loaded {inst}: V={int(mesh.vertices.shape[0]):,} '
              f'F={int(mesh.faces.shape[0]):,}')

    # distinct per-instance colors for the instance panel (--normal mode):
    # golden-ratio hue walk keeps neighbors in the load order far apart in hue.
    inst_colors = {m[0]: colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.8, 1.0)
                   for i, m in enumerate(meshes)}
    if args.normal:
        for inst, c in inst_colors.items():
            print(f'[scene] color {inst}: rgb=({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})')

    # ---- (optional) GT world meshes, rendered as normals for a fair comparison ----
    gt_meshes = []   # (inst, Mesh_world, eye4_torch, v_world_h)
    if args.normal and args.gt_world_dir:
        eye4 = torch.eye(4, dtype=torch.float32, device=device)
        gdir = Path(args.gt_world_dir) / case_root.name
        for inst in instances:
            p = gdir / f'{inst}.npz'
            if not p.exists():
                print(f'[scene] WARN: GT world mesh {p} missing, skip')
                continue
            d = np.load(p)
            Vw = d['vertices'].astype(np.float64)
            gm = Mesh(torch.from_numpy(d['vertices']).float().to(device),
                      torch.from_numpy(d['faces']).int().to(device))
            vh = np.concatenate([Vw, np.ones((len(Vw), 1))], axis=1)
            gt_meshes.append((inst, gm, eye4, vh))
        print(f'[scene] GT world meshes: {len(gt_meshes)} loaded from {gdir}')

    def _composite(mesh_list, extr, intr, near, far, canvas, inst_colors=None):
        """Depth-composite a list of (inst, mesh, T_torch, *) into one image
        (normal in --normal mode, else base_color). With inst_colors (--normal
        mode only), the SAME render pass also composites a second image where
        each instance gets its own flat color modulated by camera-space normal
        shading, so objects are distinguishable but orientation stays visible.
        Returns (img, inst_img_or_None, hit_mask), imgs (canvas,canvas,3) uint8,
        hit_mask (canvas,canvas) bool = pixels covered by any mesh."""
        out = np.zeros((canvas, canvas, 3), dtype=np.uint8)
        out_c = np.zeros_like(out) if inst_colors is not None else None
        zbuf = np.full((canvas, canvas), 1e10, dtype=np.float64)
        for entry in mesh_list:
            inst, mesh, T_torch = entry[0], entry[1], entry[2]
            if args.normal:
                r = MeshRenderer()
                r.rendering_options.resolution = canvas
                r.rendering_options.near = near; r.rendering_options.far = far
                r.rendering_options.ssaa = args.ssaa
            else:
                r = PbrMeshRenderer()
                r.rendering_options.resolution = canvas
                r.rendering_options.near = near; r.rendering_options.far = far
                r.rendering_options.ssaa = args.ssaa
                r.rendering_options.peel_layers = args.peel_layers
            try:
                if args.normal:
                    res = r.render(mesh, extr, intr, return_types=['normal', 'depth'],
                                   transformation=T_torch)
                else:
                    res = r.render(mesh, extr, intr, envmap=envmap, transformation=T_torch)
            except Exception as e:
                print(f'    {inst}: render failed: {type(e).__name__}: {e}')
                torch.cuda.empty_cache(); continue
            key = 'normal' if args.normal else 'base_color'
            bc = (_hwc3(res[key]).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            dep = res['depth'].squeeze().cpu().numpy().astype(np.float64)
            hit = (dep > 1e-4) & (dep < 1e9) & (dep < zbuf)
            out[hit] = bc[hit]; zbuf[hit] = dep[hit]
            if out_c is not None:
                # camera-space normal back from the RGB encoding; |n_z| is the
                # view-facing term -> headlight shading over the flat inst color
                n_cam = bc.astype(np.float32) / 255.0 * 2.0 - 1.0
                shade = 0.35 + 0.65 * np.clip(np.abs(n_cam[..., 2:3]), 0.0, 1.0)
                col = np.asarray(inst_colors[inst], np.float32) * 255.0 * shade
                out_c[hit] = col.astype(np.uint8)[hit]
            torch.cuda.empty_cache()
        return out, out_c, zbuf < 1e10

    def _legend(img):
        """Stamp instance names in their palette colors onto a copy of the
        per-instance color panel."""
        pil = Image.fromarray(img); dr_ = ImageDraw.Draw(pil)
        y = 4
        for m in meshes:
            c = tuple(int(v * 255) for v in inst_colors[m[0]])
            try:
                dr_.text((5, y), m[0], fill=c, stroke_width=1, stroke_fill=(0, 0, 0))
            except TypeError:  # Pillow < 6.2: no stroke support
                dr_.text((5, y), m[0], fill=c)
            y += 13
            if y > img.shape[0] - 13:
                dr_.text((5, y), '...', fill=(255, 255, 255))
                break
        return np.array(pil)

    # ---- Per-frame depth-composited scene render ----
    panels = []
    for c in cam_list:
        sub = c['sub']
        if not os.path.exists(c['path']):
            print(f'  sub={sub:02d}: frame missing ({c["path"]}), skip')
            continue
        full_img = np.array(Image.open(c['path']).convert('RGB'))
        H_full, W_full = full_img.shape[:2]
        w2c = c['w2c']
        K_norm = c['K_full'].copy()
        K_norm[0, 0] /= W_full; K_norm[0, 2] /= W_full
        K_norm[1, 1] /= H_full; K_norm[1, 2] /= H_full
        extr = torch.from_numpy(w2c.astype(np.float32)).to(device)
        intr = torch.from_numpy(K_norm.astype(np.float32)).to(device)

        # only the meshes actually visible (masked) in THIS frame get rendered
        vis = c.get('insts')
        view_meshes = meshes if vis is None else [m for m in meshes if m[0] in vis]
        view_gt = gt_meshes if vis is None else [g for g in gt_meshes if g[0] in vis]

        # scene near/far over the visible instances (pred + GT)
        zs = []
        for _, _, _, vwh in view_meshes + view_gt:
            z = (vwh @ w2c.T)[:, 2]
            zs.append(z[z > 0])
        zs = np.concatenate(zs) if zs else np.array([1.0])
        near = max(0.01, float(zs.min()) - 0.1) if len(zs) else 0.01
        far = float(zs.max()) + 1.0 if len(zs) else 10.0

        canvas = max(W_full, H_full) if args.full_res else args.preview_long
        aspect = W_full / H_full
        if aspect >= 1:
            pw, ph = canvas, max(1, int(round(canvas / aspect)))
        else:
            pw, ph = max(1, int(round(canvas * aspect))), canvas

        def _fit(img):
            return img if (canvas, canvas) == (ph, pw) else cv2.resize(
                img, (pw, ph), interpolation=cv2.INTER_AREA)

        pred_img, pred_inst, hit = _composite(view_meshes, extr, intr, near, far, canvas,
                                              inst_colors if args.normal else None)
        if args.white_bg:
            pred_img[~hit] = 255
            if pred_inst is not None:
                pred_inst[~hit] = 255
        pred_img = _fit(pred_img)
        if pred_inst is not None:
            pred_inst = _fit(pred_inst)
        hit_fit = hit if (canvas, canvas) == (ph, pw) else cv2.resize(
            hit.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
        full_pre = np.array(Image.fromarray(full_img).resize((pw, ph), Image.LANCZOS))

        if gt_meshes:
            gt_img, _, ghit = _composite(view_gt, extr, intr, near, far, canvas)
            if args.white_bg:
                gt_img[~ghit] = 255
            gt_img = _fit(gt_img)
            cols = [full_pre, gt_img, pred_img]
            text = f'SCENE sub={sub:02d}  orig | GT normal | pred normal'
            if pred_inst is not None:
                cols.append(pred_inst)
                text += ' | instances'
            panel = np.concatenate(cols, axis=1)
        else:
            base = pred_inst if pred_inst is not None else pred_img
            sil = hit_fit           # exact coverage mask (works on any background)
            comp = full_pre.copy(); comp[sil] = base[sil]
            if pred_inst is not None:
                panel = np.concatenate([full_pre, pred_img, pred_inst, comp], axis=1)
                text = f'SCENE sub={sub:02d}  ({len(view_meshes)} objs)  orig | normal | instances | overlay'
            else:
                panel = np.concatenate([full_pre, pred_img, comp], axis=1)
                text = f'SCENE sub={sub:02d}  ({len(view_meshes)} objs)  orig | render | overlay'

        labeled = panel   # no top-left caption banner, no per-object name legend
        Image.fromarray(labeled).save(render_dir / f'view{sub:02d}.jpg', quality=95)
        panels.append(labeled)
        print(f'  sub={sub:02d}: rendered scene ({len(view_meshes)}/{len(meshes)} pred visible'
              f'{f", {len(view_gt)} GT" if gt_meshes else ""})')

    if len(panels) > 1 and not args.no_video:
        # One video frame per rendered view (replaces the old stacked grid.png).
        # Frames must share one size: normalize widths, bottom-pad heights with
        # black (panels can differ in column count), crop to even dims for h264.
        w_min = min(p.shape[1] for p in panels)
        normed = [p if p.shape[1] == w_min else cv2.resize(
                    p, (w_min, int(round(p.shape[0] * w_min / p.shape[1]))),
                    interpolation=cv2.INTER_AREA) for p in panels]
        h_max = max(p.shape[0] for p in normed)
        frames = []
        for p in normed:
            if p.shape[0] != h_max:
                p = np.concatenate(
                    [p, np.zeros((h_max - p.shape[0], w_min, 3), np.uint8)], axis=0)
            frames.append(p[:h_max - h_max % 2, :w_min - w_min % 2])
        out_mp4 = render_dir / 'scene.mp4'
        import imageio
        imageio.mimsave(out_mp4, frames, fps=args.video_fps, codec='libx264',
                        quality=None,
                        output_params=['-crf', str(args.video_crf), '-pix_fmt', 'yuv420p'])
        print(f'[scene] video: {out_mp4} ({len(frames)} frames @ {args.video_fps} fps, '
              f'crf={args.video_crf})')

    print('[scene] Done.')


if __name__ == '__main__':
    main()
