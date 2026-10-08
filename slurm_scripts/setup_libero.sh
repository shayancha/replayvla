#!/bin/bash
# One-time: install the LIBERO simulator (for eval) into the replayvla env, as in OpenVLA's README.
#   LIBERO repo -> /work/sc1081/LIBERO ; config -> $LIBERO_CONFIG_PATH (pre-written so the first import never prompts)
set -euo pipefail
source "$(dirname "$0")/common.sh"
export TMPDIR=/tmp   # node-local: pip wheel builds in /work (NFS) fail with .nfs* "Directory not empty"
LIBERO_DIR="$WORK_DIR/LIBERO"
[[ -d "$LIBERO_DIR" ]] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
python -m pip install -e "$LIBERO_DIR"
# bddl must be 1.0.1 (LIBERO's own pin); unpinned it resolves to an unrelated 3.x release
python -m pip install -r "$REPO_DIR/experiments/robot/libero/libero_requirements.txt" "bddl==1.0.1"
# Same pins as setup_env.sh, in case the installs above moved them
# recent opencv-python (incl. late 4.x) needs numpy>=2; TF 2.15 needs <2: pin 4.10
python -m pip install "tensorflow-metadata==1.17.1" "protobuf==4.21.12" "numpy==1.26.4" "opencv-python==4.10.0.84"

# LIBERO asks interactive questions on first import unless its config exists
mkdir -p "$LIBERO_CONFIG_PATH"
python - <<PY
import os, yaml
root = os.path.join(os.environ["WORK_DIR"], "LIBERO", "libero", "libero")
cfg = {"benchmark_root": root, "bddl_files": os.path.join(root, "bddl_files"), "init_states": os.path.join(root, "init_files"),
       "datasets": os.path.join(root, "..", "datasets"), "assets": os.path.join(root, "assets")}
with open(os.path.join(os.environ["LIBERO_CONFIG_PATH"], "config.yaml"), "w") as f:
    yaml.safe_dump(cfg, f)
print("wrote LIBERO config:", cfg)
PY
python -m pip check || true
python -c "from libero.libero import benchmark; s = benchmark.get_benchmark_dict()[\"libero_10\"](); print(\"libero_10 tasks:\", s.n_tasks)"
echo "LIBERO ready"
