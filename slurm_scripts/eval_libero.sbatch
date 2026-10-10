#!/bin/bash
#SBATCH --job-name=replayvla-eval
#SBATCH --account=scavenger-h200
#SBATCH --partition=scavenger-h200
#SBATCH --gres=gpu:h200:1              # 1 GPU per eval job; 2 jobs (TASK_IDS split) use the 2-GPU allowance
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue                      # preemption = cancel + requeue; the eval resumes from its per-episode results
#SBATCH --open-mode=append
#SBATCH --output=/work/%u/logs/%x-%j.out
#
# LIBERO simulation eval of a merged ReplayVLA checkpoint (see vla-scripts/merge_replayvla.py).
#   sbatch slurm_scripts/eval_libero.sbatch <CHECKPOINT_DIR> [TRIALS_PER_TASK=50] [TASK_SUITE=libero_10] [extra args...]
#   Split across 2 GPUs:  TASK_IDS=0-4 sbatch ... ;  TASK_IDS=5-9 sbatch ...   (same checkpoint + trials -> same results)
#   Baseline (plain OpenVLA) checkpoints:  MODEL_FAMILY=openvla sbatch slurm_scripts/eval_libero.sbatch <CHECKPOINT_DIR>
#   Non-H200 GPU instead:  sbatch --account=<your account> --partition=gpu-common --gres=gpu:5000_ada:1 ...
# Results: /work/$USER/logs/libero_eval/RESULTS-<suite>-<family>--<run>--<checkpoint>[-<trials>trials]/SUMMARY.txt
#   (per-episode task*.jsonl next to it; a rerun with the same arguments skips finished episodes)

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm_scripts/common.sh"
export TMPDIR=/tmp

CKPT=${1:?usage: sbatch slurm_scripts/eval_libero.sbatch <CHECKPOINT_DIR> [TRIALS] [SUITE]}
TRIALS=${2:-50}
SUITE=${3:-libero_10}
shift $(( $# < 3 ? $# : 3 ))

TASK_ARGS=()
[[ -n "${TASK_IDS:-}" ]] && TASK_ARGS=(--task_ids "$TASK_IDS")
CKPT=$(realpath "$CKPT")
NOTE="$(basename "$(dirname "$CKPT")")--$(basename "$CKPT")"   # run + checkpoint: unique results dir per model
[[ "$TRIALS" != 50 ]] && NOTE="$NOTE-${TRIALS}trials"   # a quick 10-trial run gets its own results

echo "[$(date)] eval of $CKPT: suite $SUITE, $TRIALS trials/task, tasks ${TASK_IDS:-all} (restart count ${SLURM_RESTART_COUNT:-0})"
nvidia-smi --query-gpu=name,memory.total --format=csv
python experiments/robot/libero/run_libero_eval.py \
    --model_family "${MODEL_FAMILY:-replayvla}" \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$SUITE" \
    --num_trials_per_task "$TRIALS" \
    --center_crop True \
    --local_log_dir "$LOG_DIR/libero_eval" \
    --run_id_note "$NOTE" \
    "${TASK_ARGS[@]}" \
    "$@"
