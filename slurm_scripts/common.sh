#!/bin/bash
# Shared environment for ReplayVLA on the Duke Compute Cluster (sourced by setup + sbatch scripts).
# sbatch jobs do not inherit ~/.bashrc, so caches are redirected here too (home quota is 25 GB).
# NOTE: /work purges files older than 75 days -- keep code on GitHub and
# copy checkpoints worth keeping elsewhere.

export WORK_DIR="${WORK_DIR:-/work/$USER}"
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
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$WORK_DIR/.libero}"   # LIBERO sim config (eval)
export MUJOCO_GL="${MUJOCO_GL:-egl}"                                    # headless rendering on GPU nodes
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$TMPDIR" "$RUN_ROOT" "$ADAPTER_TMP" "$LOG_DIR"

export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# The `module` command only exists if the submitting shell exported it (login shells do; non-login shells, e.g.
# jobs submitted from an agent/ssh session, do not), so initialize Environment Modules explicitly.
if ! type module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash
fi
case ":${MODULEPATH:-}:" in   # DCC's modulefiles (login shells get this from /etc/profile.d/00-modules.sh)
  *:/opt/apps/modulefiles:*) ;;
  *) module use /opt/apps/modulefiles ;;
esac
module load Anaconda3/2024.02
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
  conda activate "$CONDA_ENV_NAME"
fi
# LIBERO (eval) has no top-level libero/__init__.py, so its editable install is not importable with current setuptools;
# put the repo on PYTHONPATH instead. Must come AFTER `module load Anaconda3`, which overwrites PYTHONPATH.
export PYTHONPATH="$WORK_DIR/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"
