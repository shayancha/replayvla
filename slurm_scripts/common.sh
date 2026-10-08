#!/bin/bash
# Shared paths/environment for ReplayVLA jobs on the Duke Compute Cluster. Sourced by the sbatch scripts.
# Everything lives under /work/sc1081 (fast; NOTE: /work purges files older than 75 days, so keep the code on GitHub
# and copy final checkpoints you want to keep elsewhere).

export WORK_ROOT=/work/sc1081
export REPO_DIR=$WORK_ROOT/replayvla                            # git clone of shayancha/replayvla (branch replayvla-memory)
export DATA_ROOT=$WORK_ROOT/data/modified_libero_rlds          # openvla/modified_libero_rlds (RLDS)
export RUN_ROOT=$WORK_ROOT/runs
export ADAPTER_TMP=$WORK_ROOT/adapter-tmp
export LOG_DIR=$WORK_ROOT/logs
export HF_HOME=$WORK_ROOT/hf_cache                             # openvla/openvla-7b weights live here
export CONDA_ROOT=$WORK_ROOT/miniconda3
export CONDA_ENV=replayvla

export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO_DIR"
mkdir -p "$RUN_ROOT" "$ADAPTER_TMP" "$LOG_DIR"
