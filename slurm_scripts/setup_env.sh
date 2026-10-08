#!/bin/bash
# One-time environment setup on DCC (run on a compute node, e.g. via `ssh dcc-agent`, not on a login node):
#   conda env (python 3.10) + `pip install -e .` with the repo pins + the two pins the editable install needs.
# Re-runnable: skips the env creation if it already exists.
set -euo pipefail
source "$(dirname "$0")/common.sh"

if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
  conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi
conda activate "$CONDA_ENV_NAME"

python -m pip install --upgrade pip
python -m pip install -e "$REPO_DIR"
# `pip install -e .` pulls the newest tensorflow-metadata, which needs protobuf>=5 and breaks TF 2.15 / wandb imports
python -m pip install "tensorflow-metadata==1.17.1" "protobuf==4.21.12"
python -m pip check
python -c "import torch, transformers, timm, tensorflow, peft, dlimp, wandb; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"| transformers\", transformers.__version__, \"| tf\", tensorflow.__version__)"
echo "Environment ready: conda activate $CONDA_ENV_NAME"
