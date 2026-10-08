#!/bin/bash
#SBATCH --job-name=replayvla-train
#SBATCH --account=scavenger-h200
#SBATCH --partition=scavenger-h200
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --requeue                       # preemption = cancel + requeue: the requeued job resumes from the checkpoint
#SBATCH --signal=B:USR1@900             # 15 min before the time limit, SIGUSR1 to this script -> checkpoint + exit
#SBATCH --open-mode=append
#SBATCH --output=/work/sc1081/logs/%x-%j.out
#
# Long ReplayVLA training on 2x H200 that survives the 24 h limit and preemption:
#   - time limit: SIGUSR1 (15 min early) -> STOP file -> all ranks checkpoint at the next step and exit;
#                 then, if AUTO_RESUBMIT=1 and training isn't DONE, this script resubmits itself (one job at a time)
#   - preemption: SIGTERM -> STOP file + workers' own SIGTERM handler -> checkpoint (~30 s window); Slurm requeues
#   - always:     a resumable checkpoint every CHECKPOINT_MIN minutes (safety net if a kill gives no warning)
# Submit:   sbatch slurm_scripts/train_replayvla.sbatch      (extra train args: sbatch ... -- --lora_rank 16 ...)
# Env:      AUTO_RESUBMIT=1 to chain jobs until max_steps; MAX_RESUBMITS (default 10); RUN_NOTE for the run name

set -uo pipefail
source "$(dirname "$0")/common.sh" 2>/dev/null || source /work/sc1081/replayvla/slurm_scripts/common.sh

NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}
CHECKPOINT_MIN=${CHECKPOINT_MIN:-30}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-10}
RUN_NOTE=${RUN_NOTE:-main}
STOP_FILE=$RUN_ROOT/STOP-${RUN_NOTE}
TIME_LIMIT_HIT=0

echo "[$(date)] job $SLURM_JOB_ID (restart count ${SLURM_RESTART_COUNT:-0}, resubmit $RESUBMIT_COUNT) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

on_usr1() { echo "[$(date)] SIGUSR1: time limit approaching -> requesting checkpoint"; TIME_LIMIT_HIT=1; touch "$STOP_FILE"; }
on_term() { echo "[$(date)] SIGTERM: preempted/cancelled -> requesting checkpoint"; touch "$STOP_FILE"; }
trap on_usr1 USR1
trap on_term TERM

rm -f "$STOP_FILE"
torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/train_replayvla.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name libero_10_no_noops \
    --run_root_dir "$RUN_ROOT" \
    --adapter_tmp_dir "$ADAPTER_TMP" \
    --run_id_note "$RUN_NOTE" \
    --stop_file "$STOP_FILE" \
    --checkpoint_interval_minutes "$CHECKPOINT_MIN" \
    "$@" &
TRAIN_PID=$!

# `wait` returns early whenever a trapped signal arrives; keep waiting until training has actually exited
while kill -0 "$TRAIN_PID" 2>/dev/null; do wait "$TRAIN_PID"; done
wait "$TRAIN_PID"; STATUS=$?
echo "[$(date)] training exited with status $STATUS"

RUN_DIR=$(ls -d "$RUN_ROOT"/replayvla+*--"$RUN_NOTE"* 2>/dev/null | head -1)
if [[ -n "$RUN_DIR" && -f "$RUN_DIR/DONE" ]]; then
    echo "Training finished: $(cat "$RUN_DIR/DONE")"
elif [[ "$TIME_LIMIT_HIT" == 1 && "${AUTO_RESUBMIT:-0}" == 1 && "$RESUBMIT_COUNT" -lt "$MAX_RESUBMITS" ]]; then
    echo "Not finished: resubmitting (resubmit $((RESUBMIT_COUNT + 1)) of $MAX_RESUBMITS)"
    sbatch --export=ALL,RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)),AUTO_RESUBMIT=1,RUN_NOTE="$RUN_NOTE" "$0" "$@"
fi
exit $STATUS
