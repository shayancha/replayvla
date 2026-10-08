#!/bin/bash
#SBATCH --job-name=replayvla-smoke
#SBATCH --account=scavenger-h200
#SBATCH --partition=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/work/sc1081/logs/%x-%j.out
#
# 1x H200 smoke test of the full pipeline on real data (openvla-7b + LIBERO-10):
#   phase 1: 30 steps from scratch (LLM grad checkpointing), resumable checkpoint at the end   -> step time, GPU memory
#   phase 2: resume and run to 45 steps                                                         -> resume works on GPU
#   phase 3: resume again, touch the STOP file after ~2 min                                     -> clean stop + checkpoint
# Usage: sbatch slurm_scripts/smoke_test.sbatch [BATCH_SIZE (default 8)]

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-/work/sc1081/replayvla}/slurm_scripts/common.sh"
export TMPDIR=/tmp   # node-local: temp files on /work (NFS) hit .nfs* "Directory not empty" races

BS=${1:-8}
NOTE=smoke-bs$BS
STOP_FILE=$RUN_ROOT/STOP-$NOTE
COMMON=(--vla_path openvla/openvla-7b --data_root_dir "$DATA_ROOT" --dataset_name libero_10_no_noops
        --run_root_dir "$RUN_ROOT" --adapter_tmp_dir "$ADAPTER_TMP" --run_id_note "$NOTE"
        --batch_size "$BS" --shuffle_buffer_size 2000 --save_steps 100000 --merge_on_save False
        --stop_file "$STOP_FILE" --wandb_mode offline)
run() { torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train_replayvla.py "${COMMON[@]}" "$@"; }

nvidia-smi --query-gpu=name,memory.total --format=csv
rm -rf "$RUN_ROOT"/replayvla+*--"$NOTE"*

echo "=== phase 1: 30 steps from scratch"; run --max_steps 30
echo "=== phase 2: resume to 45";          run --max_steps 45
echo "=== phase 3: resume, stop via STOP file after 120 s"
( sleep 120; touch "$STOP_FILE"; echo "[$(date)] touched STOP" ) &
run --max_steps 100000
echo "=== smoke test done"; ls -la "$RUN_ROOT"/replayvla+*--"$NOTE"*/resume
