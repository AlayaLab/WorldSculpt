
set -euo pipefail
trap 'echo "[pipeline] ERROR: command failed (exit $?) at line $LINENO: $BASH_COMMAND" >&2' ERR
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$SCRIPT_DIR"

# ---- Every knob must be passed explicitly; nothing is defaulted ----
# The single exception is GPU: it describes the machine, not the experiment, so a
# default there cannot silently change a result. Everything else does -- a wrong
# implicit SS_STEP or SAMPLER produces a plausible-looking scene from the wrong
# weights, which is exactly the failure that is hardest to notice afterwards.
GPU=${GPU:-0}
REQUIRED=(INPUT_ROOT OUTPUT_ROOT CKPT_ROOT SAMPLER SS_STEP SHAPE_STEP
          RENDER FACE_BUDGET)
missing=()
for v in "${REQUIRED[@]}"; do [ -n "${!v:-}" ] || missing+=("$v"); done
if [ ${#missing[@]} -gt 0 ] || [ $# -eq 0 ]; then
    {
        [ $# -eq 0 ] && echo "[pipeline] ERROR: no scene given"
        [ ${#missing[@]} -gt 0 ] && echo "[pipeline] ERROR: unset: ${missing[*]}"
        cat <<'USAGE'

usage:  <VAR=... ...> ./inference.sh <scene> [scene ...]

  INPUT_ROOT    directory holding the scene folders (<INPUT_ROOT>/<scene>/transforms.json)
  OUTPUT_ROOT   where results are written; must differ from INPUT_ROOT
  CKPT_ROOT     directory holding the finetuned LoRA weights (README Step 3)
  SAMPLER       official | train   sampling parameters
  SS_STEP       stage-1 ckpt iteration, e.g. 15000
  SHAPE_STEP    stage-2 ckpt iteration, e.g. 15000
  RENDER        1 | 0              per-view renders during composition (dominates runtime)
  FACE_BUDGET   per-instance face cap for composition, e.g. 1000000
  GPU           CUDA device index (optional, default 2)

example:
  INPUT_ROOT=./input/Marble OUTPUT_ROOT=./output CKPT_ROOT=./pretrained \
  SAMPLER=official SS_STEP=15000 SHAPE_STEP=15000 \
  RENDER=1 FACE_BUDGET=1000000 GPU=0 \
  ./inference.sh marble_serene_living_room_countryside_view
USAGE
    } >&2
    exit 1
fi
IN=$INPUT_ROOT
OUT=$OUTPUT_ROOT
echo "[data] in=$IN"
echo "[data] out=$OUT"
# See the _symlink note in the header: sharing one dir would let the cropping stage delete the
# GT masks it is trying to link.
if [ "$(cd "$IN" 2>/dev/null && pwd || echo "$IN")" = "$(cd "$OUT" 2>/dev/null && pwd || echo "$OUT")" ]; then
    echo "[pipeline] ERROR: INPUT_ROOT and OUTPUT_ROOT must differ (both = $IN)" >&2
    exit 1
fi

SCENES=("$@")

for scene in "${SCENES[@]}"; do
    echo ""
    echo "==== [$scene] ===="
    # ---- derived (per scene) ----
    # Finetuned LoRA weights live where README Step 3 downloads them
    # (hf download AlayaLab/WorldSculpt --local-dir ./pretrained).
    SS_DIR=$CKPT_ROOT/ss_ft64_mv_lora_ibr_texverse
    SHAPE_DIR=$CKPT_ROOT/shape_ft1024_mv_lora_ibr_texverse_fixedmem05
    # Fail here, with the path in the message, instead of on a FileNotFoundError
    # from deep inside the denoiser build.
    for spec in "ss:$SS_DIR:$SS_STEP" "shape:$SHAPE_DIR:$SHAPE_STEP"; do
        tag=${spec%%:*}; rest=${spec#*:}; dir=${rest%%:*}; step=${rest##*:}
        if [ ! -f "$dir/config.json" ]; then
            echo "[pipeline] ERROR: no $tag config at $dir/config.json" >&2
            echo "  Did Step 3 run?  hf download AlayaLab/WorldSculpt --local-dir ./pretrained" >&2
            exit 1
        fi
        if ! ls "$dir/ckpts"/*_step"$(printf '%07d' "$step")".pt >/dev/null 2>&1; then
            echo "[pipeline] ERROR: $tag step $step not in $dir/ckpts" >&2
            echo "  available: $(ls "$dir/ckpts" | grep -oE 'step[0-9]+' | sed 's/step0*//' \
                                 | sort -un | tr '\n' ' ')" >&2
            exit 1
        fi
    done
    RECON_SUBDIR=_recon
    render_flag="--no_render"; [ "$RENDER" = "1" ] && render_flag=""

    SCENE_DIR=$IN/$scene               # pre-built gt_scene (read-only)
    CASE_ROOT=$OUT/$scene              # all products land here
    COMPOSE_DIR=$CASE_ROOT/_scene
    echo "[args] sampler=$SAMPLER ss=$SS_STEP shape=$SHAPE_STEP" \
         "render=$RENDER face_budget=$FACE_BUDGET  (geometry-only, all views)"

    # ---- Entry gate: the scene must already be built ----
    if [ ! -f "$SCENE_DIR/transforms.json" ]; then
        {
            echo "[pipeline] ERROR: no scene at $SCENE_DIR/transforms.json"
            echo "  Expected a pre-built gt_scene directory containing:"
            echo "      transforms.json   NNN.png (frames)   masks/   [depth_gt/]"
            echo "  Scenes currently under $IN:"
            ls -d "$IN"/*/ 2>/dev/null | sed 's|^|      |' || echo "      (none)"
        } >&2
        exit 1
    fi
    mkdir -p "$CASE_ROOT"
    echo "---- [$scene] scene found, starting from the crop stage ----"
    echo "       in  $SCENE_DIR"
    echo "       out $CASE_ROOT"

    # Per-instance fine-tune data (crops + transforms.json for reconstruction/composition).
    # Skipped when every instance in the scene already has its transforms.json
    # under _crops (the cropping stage is slow: full 4K decode/encode per crop).
    # Force a re-run with FORCE_CROPS=1 (needed after changing the cropping stage args or
    # regenerating output/$scene).
    # Count only instances the cropping stage will actually build: needs an aabb_world with
    # non-degenerate extent (>1mm) — others are skipped and never get a
    # transforms.json (keep in sync with prepare_crops_scene.py's skip rules).
    n_inst=$(python -c "
import json, numpy as np
insts = json.load(open('$SCENE_DIR/transforms.json'))['instances']
n = 0
for i in insts:
    b = i.get('aabb_world')
    if b and float((np.array(b[1]) - np.array(b[0])).max()) >= 1e-3:
        n += 1
print(n)")
    # NOTE: the `|| true` is required — with pipefail, an unmatched glob makes ls
    # fail and the whole pipeline would silently kill the script under set -e.
    n_done=$( (ls $CASE_ROOT/_crops/obj*/transforms.json 2>/dev/null || true) | wc -l)
    if [ "${FORCE_CROPS:-0}" != "1" ] && [ "$n_done" -ge "$n_inst" ]; then
        echo "---- [$scene] crops done ($n_done/$n_inst objects), skip (FORCE_CROPS=1 to redo) ----"
    else
        python prepare_crops_scene.py \
            --scene_dir $SCENE_DIR \
            --case_root $CASE_ROOT \
            --crop_resolution 1024 \
            --save_alignments \
            --alpha_erode_kernel 0 \
            --alpha_erode_iters 0 \
            --min_mask_ratio 0.001 \
            --max_crop_ratio 3.0 \
            --mask_fit_scale
    fi

    # Geometry only: --no_tex skips Stage 3 (texture) so the tex denoiser/aggregator
    # are never built; --no_glb skips reconstruction's per-object GLB (composition composes instead).
    CUDA_VISIBLE_DEVICES=$GPU python reconstruct_batch.py \
        --case_root $CASE_ROOT --views all --recon_subdir $RECON_SUBDIR \
        --no-ema --vis_ss --sampler $SAMPLER \
        --no_tex --no_glb \
        --ss_config    $SS_DIR/config.json    --ss_ckpt_dir    $SS_DIR/ckpts    --ss_step    $SS_STEP \
        --shape_config $SHAPE_DIR/config.json --shape_ckpt_dir $SHAPE_DIR/ckpts --shape_step $SHAPE_STEP

    if [ -f "$COMPOSE_DIR/scene.glb" ]; then
        echo "---- [$scene] scene.glb exists, skip compose ----"
    else
        CUDA_VISIBLE_DEVICES=$GPU python compose_scene.py \
            --case_root $CASE_ROOT \
            --recon_dir $CASE_ROOT/$RECON_SUBDIR \
            --face_budget $FACE_BUDGET \
            --output_dir $COMPOSE_DIR --normal $render_flag

        CUDA_VISIBLE_DEVICES=$GPU python visualize_pointcloud.py \
            --case_root $CASE_ROOT \
            --recon_dir $CASE_ROOT/$RECON_SUBDIR \
            --output_dir $COMPOSE_DIR
    fi
done

echo ""
echo "==== pipeline done ===="
