#!/bin/bash
# Shared environment for ReplayVLA on the Duke Compute Cluster (sourced by setup + sbatch scripts).
# sbatch jobs do not inherit ~/.bashrc, so caches are redirected here too (home quota is 25 GB).
# Same conventions as /work/sc1081/ReWAM. NOTE: /work purges files older than 75 days -- keep code on GitHub and
# copy checkpoints worth keeping elsewhere.

export WORK_DIR="${WORK_DIR:-/work/sc1081}"
export REPO_DIR="${REPO_DIR:-$WORK_DIR/replayvla}"
export HF_HOME="${HF_HOME:-$WORK_DIR/.cache/huggingface}"           # openvla/openvla-7b weights
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_DIR/.cache/pip}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$WORK_DIR/.conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WORK_DIR/.conda/pkgs}"
export TMPDIR="${TMPDIR:-$WORK_DIR/.tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK_DIR/.cache}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-replayvla}"

export DATA_ROOT="${DATA_ROOT:-$WORK_DIR/data/modified_libero_rlds}"   # openvla/modified_libero_rlds
export RUN_ROOT="${RUN_ROOT:-$WORK_DIR/runs}"
export ADAPTER_TMP="${ADAPTER_TMP:-$WORK_DIR/adapter-tmp}"
export LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$TMPDIR" "$RUN_ROOT" "$ADAPTER_TMP" "$LOG_DIR"

export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

module load Anaconda3/2024.02
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
  conda activate "$CONDA_ENV_NAME"
fi
cd "$REPO_DIR"
