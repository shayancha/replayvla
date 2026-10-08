#!/bin/bash
#SBATCH --job-name=replayvla-eval
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --gres=gpu:5000_ada:1          # >=24 GB is enough for 7B bf16 inference; keeps the H200s for training
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/sc1081/logs/%x-%j.out
#
# LIBERO simulation eval of a merged ReplayVLA checkpoint (see vla-scripts/merge_replayvla.py).
#   sbatch slurm_scripts/eval_libero.sbatch <CHECKPOINT_DIR> [TRIALS_PER_TASK=50] [TASK_SUITE=libero_10] [extra args...]
# Results: /work/sc1081/logs/libero_eval/EVAL-*.txt (+ rollout videos under ./rollouts)

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-/work/sc1081/replayvla}/slurm_scripts/common.sh"
export TMPDIR=/tmp

CKPT=${1:?usage: sbatch slurm_scripts/eval_libero.sbatch <CHECKPOINT_DIR> [TRIALS] [SUITE]}
TRIALS=${2:-50}
SUITE=${3:-libero_10}
shift $(( $# < 3 ? $# : 3 ))

nvidia-smi --query-gpu=name,memory.total --format=csv
python experiments/robot/libero/run_libero_eval.py \
    --model_family replayvla \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$SUITE" \
    --num_trials_per_task "$TRIALS" \
    --center_crop True \
    --local_log_dir "$LOG_DIR/libero_eval" \
    --run_id_note "$(basename "$CKPT")" \
    "$@"
