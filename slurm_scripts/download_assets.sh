#!/bin/bash
# Download what training needs into /work/$USER (run on a compute node; needs internet, which DCC compute nodes have):
#   - openvla/openvla-7b                       -> $HF_HOME (Hugging Face cache, ~15 GB)
#   - openvla/modified_libero_rlds: $LIBERO_DATASETS (default libero_10_no_noops = LIBERO-Long; e.g. LIBERO_DATASETS=libero_spatial_no_noops) -> $DATA_ROOT
set -euo pipefail
source "$(dirname "$0")/common.sh"
python - <<PY
import os
from huggingface_hub import snapshot_download
print("model ->", snapshot_download("openvla/openvla-7b"))
suites = os.environ.get("LIBERO_DATASETS", "libero_10_no_noops").split()   # e.g. "libero_10_no_noops libero_spatial_no_noops"
print("data  ->", snapshot_download("openvla/modified_libero_rlds", repo_type="dataset",
                                    allow_patterns=[f"{s}/*" for s in suites], local_dir=os.environ["DATA_ROOT"]))
PY
for d in ${LIBERO_DATASETS:-libero_10_no_noops}; do du -sh "$DATA_ROOT/$d"; done
du -sh "$HF_HOME"/hub/models--openvla--openvla-7b
echo "Downloads complete"
