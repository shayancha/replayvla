#!/usr/bin/env bash
# Train or resume ReplayVLA on this single-H100 machine, without Slurm.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPLAYVLA_VENV:-$REPO_DIR/../HardCode_VLAs_WAMs/.venv}"
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/../libero-gemm-run/libero_10/data}"
RUN_ROOT="${RUN_ROOT:-$REPO_DIR/runs}"
ADAPTER_TMP="${ADAPTER_TMP:-$REPO_DIR/adapter-tmp}"
RUN_NOTE="${RUN_NOTE:-smoke-20261009}"
STOP_FILE="$RUN_ROOT/STOP-$RUN_NOTE"

if [[ ! -x "$VENV_DIR/bin/torchrun" ]]; then
    echo "Missing torchrun: $VENV_DIR/bin/torchrun" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT/libero_10_no_noops" ]]; then
    echo "Missing LIBERO RLDS dataset: $DATA_ROOT/libero_10_no_noops" >&2
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
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

cd "$REPO_DIR"
exec "$VENV_DIR/bin/torchrun" --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/train_replayvla.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name libero_10_no_noops \
    --run_root_dir "$RUN_ROOT" \
    --adapter_tmp_dir "$ADAPTER_TMP" \
    --run_id_note "$RUN_NOTE" \
    --stop_file "$STOP_FILE" \
    --batch_size 1 \
    --shuffle_buffer_size 64 \
    --checkpoint_interval_minutes 5 \
    --merge_on_save False \
    --wandb_mode offline \
    "$@"
