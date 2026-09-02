"""
reconstruct_batch.py — batch version of reconstruct_object.py.

Loads the pipeline + finetuned MV denoisers/aggregators ONCE, then loops over
every instance in a case_root's _crops/objNN/transforms.json,
reconstructing each (mesh.pt + optional GLB / ss-viz). This avoids the per-
instance model reload (the dominant cost when launching reconstruction once per object).

Resumable: instances whose mesh.pt already exists are skipped (unless --overwrite).

USAGE (geometry-only, matches step_video_gt_scene.sh GEOM=1):
  python reconstruct_batch.py \
      --case_root output/<scene> \
      --views all --no-ema --no_tex --no_glb --vis_ss \
      --ss_config results/ss_ft64_mv_lora_avg_texverse/config.json \
      --ss_ckpt_dir results/ss_ft64_mv_lora_avg_texverse/ckpts --ss_step 15000 \
      --shape_config results/shape_ft1024_mv_lora_avg_texverse_fixedmem05/config.json \
      --shape_ckpt_dir results/shape_ft1024_mv_lora_avg_texverse_fixedmem05/ckpts --shape_step 10000
"""
import os
import sys
import glob
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reconstruct_object as S4


def main():
    ap = argparse.ArgumentParser(
        description='Batch off-center MV Pixal3D: build models once, loop all instances.')
    ap.add_argument('--case_root', type=str, required=True,
                    help='Scene work dir (holds _crops/objNN/transforms.json).')
    ap.add_argument('--crops_subdir', type=str, default='_crops')
    ap.add_argument('--recon_subdir', type=str, default='_recon')
    ap.add_argument('--instances', type=str, default=None,
                    help='Comma-separated objNN to restrict to (default: all).')
    ap.add_argument('--overwrite', action='store_true',
                    help='Re-run instances whose mesh.pt already exists.')
    # ---- per-instance recon knobs (mirror reconstruction) ----
    ap.add_argument('--views', type=str, default='all')
    ap.add_argument('--max_views', type=int, default=20,
                    help='Cap the number of input views. 0 = no cap. See --view_select.')
    ap.add_argument('--view_select', type=str, default='area', choices=['area', 'fps'],
                    help="How to pick --max_views views: 'area' (default, largest non-zero "
                         "object area in crops) or 'fps' (max angular spread).")
    ap.add_argument('--anchor', type=int, default=-1)
    ap.add_argument('--resolution', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sampler', type=str, default='train',
                    choices=['train', 'official', 'pipeline'],
                    help="Sampler config: 'train' = training-snapshot params; "
                         "'official' = pretrained pipeline.json params "
                         "(FlowEulerGuidanceIntervalSampler); 'pipeline' = as loaded.")
    ap.add_argument('--max_num_tokens', type=int, default=49152)
    ap.add_argument('--low_vram', action='store_true')
    ap.add_argument('--glb_faces', type=int, default=100_000)
    ap.add_argument('--no_glb', action='store_true')
    ap.add_argument('--no_tex', action='store_true')
    ap.add_argument('--vis_ss', action='store_true')
    # ---- MV checkpoint selection (same defaults as reconstruction) ----
    ap.add_argument('--ss_step', type=int, default=-1)
    ap.add_argument('--shape_step', type=int, default=-1)
    ap.add_argument('--tex_step', type=int, default=-1)
    ap.add_argument('--ss_ckpt_dir', type=str, required=True)
    ap.add_argument('--shape_ckpt_dir', type=str, required=True)
    ap.add_argument('--tex_ckpt_dir', type=str, default=None)
    ap.add_argument('--ss_config', type=str, required=True)
    ap.add_argument('--shape_config', type=str, required=True)
    ap.add_argument('--tex_config', type=str, default=None)
    ap.add_argument('--ema', dest='ema', action='store_true', default=False)
    ap.add_argument('--no-ema', dest='ema', action='store_false')
    ap.add_argument('--ema_rate', type=float, default=0.9999)
    args = ap.parse_args()

    case_root = Path(args.case_root).resolve()
    crops_root = case_root / args.crops_subdir
    recon_root = case_root / args.recon_subdir

    # ---- collect instances (skip ones with 0 valid views) ----
    if args.instances:
        want = [s.strip() for s in args.instances.split(',') if s.strip()]
        tjs = [crops_root / w / 'transforms.json' for w in want]
    else:
        tjs = [Path(p) for p in sorted(glob.glob(str(crops_root / '*' / 'transforms.json')))]
    insts = []
    for tj in tjs:
        inst = tj.parent.name
        if not tj.exists():
            print(f'[batch] {inst}: no transforms.json, skip'); continue
        try:
            nfr = len(json.load(open(tj))['frames'])
        except Exception:
            nfr = 0
        if nfr == 0:
            print(f'[batch] {inst}: 0 valid views, skip'); continue
        insts.append((inst, tj))
    if not insts:
        raise SystemExit(f'[batch] no instances under {crops_root}')
    print(f'[batch] {len(insts)} instance(s) in {case_root.name}: '
          f'{[i for i, _ in insts]}')

    # ---- early exit BEFORE the (1-2 min) model load when everything is done ----
    if not args.overwrite:
        pending = [i for i, _ in insts if not (recon_root / i / 'mesh.pt').exists()]
        if not pending:
            print(f'[batch] Done {case_root.name}: all {len(insts)} instance(s) '
                  f'already have mesh.pt — skipping model load.')
            return
        print(f'[batch] {len(pending)} instance(s) pending: {pending}')

    # ---- build models ONCE ----
    pipeline, aggregators = S4.build_models(args)

    n_done = n_skip = n_fail = 0
    for i, (inst, tj) in enumerate(insts):
        out_dir = recon_root / inst
        if (not args.overwrite) and (out_dir / 'mesh.pt').exists():
            print(f'[batch] [{i+1}/{len(insts)}] {inst}: mesh.pt exists, skip')
            n_skip += 1
            continue
        print(f'\n[batch] ============ [{i+1}/{len(insts)}] {inst} ============')
        try:
            S4.run_instance(pipeline, aggregators, tj, args, out_dir=str(out_dir))
            n_done += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'[batch] !!!! {inst} FAILED: {type(e).__name__}: {e}')
            n_fail += 1

    print(f'\n[batch] Done {case_root.name}: {n_done} reconstructed, '
          f'{n_skip} skipped (existing), {n_fail} failed.')


if __name__ == '__main__':
    main()
