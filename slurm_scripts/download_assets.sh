#!/bin/bash
# Download what training needs into /work/$USER (run on a compute node; needs internet, which DCC compute nodes have):
#   - openvla/openvla-7b                       -> $HF_HOME (Hugging Face cache, ~15 GB)
#   - openvla/modified_libero_rlds: libero_10_no_noops only (LIBERO-Long)   -> $DATA_ROOT
set -euo pipefail
source "$(dirname "$0")/common.sh"
python - <<PY
import os
from huggingface_hub import snapshot_download
print("model ->", snapshot_download("openvla/openvla-7b"))
print("data  ->", snapshot_download("openvla/modified_libero_rlds", repo_type="dataset",
                                    allow_patterns=["libero_10_no_noops/*"], local_dir=os.environ["DATA_ROOT"]))
PY
du -sh "$DATA_ROOT"/libero_10_no_noops "$HF_HOME"/hub/models--openvla--openvla-7b
echo "Downloads complete"
