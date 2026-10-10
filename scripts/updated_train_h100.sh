#!/usr/bin/env bash
# Train on one H100 (no Slurm) with EXACTLY the training recipe of the DCC ReplayVLA run
# (slurm_scripts/train_replayvla.sbatch on 2x H200), so the results are directly comparable:
#   effective batch 16 (here 8 per step x 2 gradient-accumulation steps; on DCC 2 GPUs x 8), constant lr 5e-4,
#   LoRA r=32 / dropout 0, image augmentation on, shuffle buffer 20,000, 50,000 steps, data libero_10_no_noops.
# Only the memory differs: USE_MEMORY=False (default) trains the vanilla OpenVLA baseline, True trains ReplayVLA.
#
#   bash scripts/updated_train_h100.sh                        # vanilla baseline to 50k steps (resumes automatically)
#   bash scripts/updated_train_h100.sh --max_steps 75000      # continue the same run to 75k
#   PER_STEP_BATCH=16 GRAD_ACCUM=1 bash scripts/updated_train_h100.sh   # same batch of 16, faster if it fits
#   USE_MEMORY=True PER_STEP_BATCH=4 GRAD_ACCUM=4 bash scripts/updated_train_h100.sh   # ReplayVLA (needs more memory)
#
# Stop cleanly: touch runs/STOP-<RUN_NOTE>   (run the script again to resume from the last checkpoint)
# Evaluate:     python vla-scripts/merge_replayvla.py --run_dir runs/<run>, then the LIBERO eval with
#               --model_family openvla for the baseline (replayvla for USE_MEMORY=True), 50 trials per task.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPLAYVLA_VENV:-$REPO_DIR/../HardCode_VLAs_WAMs/.venv}"
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/../libero-gemm-run/libero_10/data}"
RUN_ROOT="${RUN_ROOT:-$REPO_DIR/runs}"
ADAPTER_TMP="${ADAPTER_TMP:-$REPO_DIR/adapter-tmp}"
USE_MEMORY="${USE_MEMORY:-False}"
if [[ "$USE_MEMORY" == "True" ]]; then DEFAULT_NOTE=replayvla-h100; else DEFAULT_NOTE=baseline-h100; fi
RUN_NOTE="${RUN_NOTE:-$DEFAULT_NOTE}"
STOP_FILE="$RUN_ROOT/STOP-$RUN_NOTE"

# Effective batch = PER_STEP_BATCH x GRAD_ACCUM (one GPU) and must be 16 to match the DCC run
PER_STEP_BATCH="${PER_STEP_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
if [[ "$USE_MEMORY" != "True" && "$USE_MEMORY" != "False" ]]; then
    echo "USE_MEMORY must be True or False (got '$USE_MEMORY')" >&2
    exit 1
fi
if (( PER_STEP_BATCH * GRAD_ACCUM != 16 )) && [[ "${ALLOW_OTHER_BATCH:-0}" != 1 ]]; then
    echo "Effective batch is $((PER_STEP_BATCH * GRAD_ACCUM)), not 16: not comparable with the DCC ReplayVLA run." >&2
    echo "Use PER_STEP_BATCH x GRAD_ACCUM = 16 (e.g. 8 x 2, 16 x 1, 4 x 4), or set ALLOW_OTHER_BATCH=1 on purpose." >&2
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/torchrun" ]]; then
    echo "Missing torchrun: $VENV_DIR/bin/torchrun (set REPLAYVLA_VENV)" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT/libero_10_no_noops" ]]; then
    echo "Missing LIBERO RLDS dataset: $DATA_ROOT/libero_10_no_noops (set DATA_ROOT)" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT" "$ADAPTER_TMP"
exec 9>"$RUN_ROOT/.train-$RUN_NOTE.lock"
if ! flock -n 9; then
    echo "Another training launcher is already using RUN_NOTE=$RUN_NOTE" >&2
    exit 1
fi
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"   # limits host-RAM growth over long runs (as on DCC)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

echo "[$(date)] use_memory=$USE_MEMORY run_note=$RUN_NOTE batch ${PER_STEP_BATCH}x${GRAD_ACCUM}=16 code $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')"
if (( $# > 0 )); then
    echo "Extra arguments (these override the recipe below, so keep them to e.g. --max_steps): $*"
fi

cd "$REPO_DIR"
# The recipe is spelled out (not left to defaults) so it cannot drift if a default changes.
exec "$VENV_DIR/bin/torchrun" --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/train_replayvla.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name libero_10_no_noops \
    --run_root_dir "$RUN_ROOT" \
    --adapter_tmp_dir "$ADAPTER_TMP" \
    --run_id_note "$RUN_NOTE" \
    --stop_file "$STOP_FILE" \
    --use_memory "$USE_MEMORY" \
    --batch_size "$PER_STEP_BATCH" \
    --grad_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate 5e-4 \
    --lora_rank 32 \
    --lora_dropout 0.0 \
    --image_aug True \
    --shuffle_buffer_size 20000 \
    --max_steps 50000 \
    --save_steps 5000 \
    --checkpoint_interval_minutes 30 \
    --merge_on_save False \
    --wandb_mode offline \
    "$@"
