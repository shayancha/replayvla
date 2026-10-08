#!/bin/bash
#SBATCH --job-name=replayvla-train
#SBATCH --account=scavenger-h200
#SBATCH --partition=scavenger-h200
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=48                # 24 per GPU (H200 nodes: 192 CPUs / 8 GPUs); data loading is CPU-heavy
#SBATCH --mem=256G
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
source "${SLURM_SUBMIT_DIR:-/work/sc1081/replayvla}/slurm_scripts/common.sh"
export TMPDIR=/tmp   # node-local: temp files on /work (NFS) hit .nfs* "Directory not empty" races
export TF_CPP_MIN_LOG_LEVEL=3   # hide TF C++ startup noise ("Unable to register cuDNN factory", ...)
# TF's graph optimizer also prints harmless "op_level_cost_estimator ... Error in PredictCost()" warnings via a logger
# that TF_CPP_MIN_LOG_LEVEL does not control; drop just those lines from stderr (see torchrun below)
TF_NOISE="op_level_cost_estimator|PredictCost"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME:-replayvla}" ]] || ! command -v torchrun >/dev/null; then
  echo "ERROR: conda env ${CONDA_ENV_NAME:-replayvla} not active (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-none}); see slurm_scripts/common.sh"
  exit 1
fi

NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}
# Each rank runs its own TF data pipeline + PyTorch. Without caps, both size their thread pools for the whole node
# (192 cores) and fight over the job's CPUs: job 58128414 ran at 15 s/step with one GPU idle. Cap per rank.
CPUS_PER_RANK=$(( ${SLURM_CPUS_PER_TASK:-16} / NUM_GPUS ))
export OMP_NUM_THREADS=$(( CPUS_PER_RANK / 4 > 1 ? CPUS_PER_RANK / 4 : 1 ))
export TF_NUM_INTRAOP_THREADS=$CPUS_PER_RANK
export TF_NUM_INTEROP_THREADS=$(( CPUS_PER_RANK / 4 > 1 ? CPUS_PER_RANK / 4 : 1 ))
echo "[$(date)] $NUM_GPUS GPUs, ${SLURM_CPUS_PER_TASK:-?} CPUs: $CPUS_PER_RANK per rank (OMP=$OMP_NUM_THREADS, TF intra=$TF_NUM_INTRAOP_THREADS inter=$TF_NUM_INTEROP_THREADS)"
CHECKPOINT_MIN=${CHECKPOINT_MIN:-30}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-10}
RUN_NOTE=${RUN_NOTE:-main}
STOP_FILE=$RUN_ROOT/STOP-${RUN_NOTE}
# Log to wandb online only if this account is logged in on DCC (run `wandb login` once); otherwise offline (sync later)
if [[ -z "${WANDB_MODE:-}" ]]; then
  if grep -q api.wandb.ai ~/.netrc 2>/dev/null; then WANDB_MODE=online; else WANDB_MODE=offline; fi
fi
TIME_LIMIT_HIT=0

echo "[$(date)] wandb mode: $WANDB_MODE"
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
    --wandb_mode "$WANDB_MODE" \
    "$@" 2> >(grep --line-buffered -v -E "$TF_NOISE" >&2) &
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
