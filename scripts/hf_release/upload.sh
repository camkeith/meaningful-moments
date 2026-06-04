#!/usr/bin/env bash
# Upload the staged Meaningful Moments release to HF (task 3.2).
# Prereq: hf auth login (write token) — run interactively first.
#
#   bash scripts/hf_release/upload.sh <namespace>            # create + upload (private)
#   bash scripts/hf_release/upload.sh <namespace> resume     # resume interrupted upload
set -euo pipefail

NS="${1:?usage: upload.sh <namespace> [resume]}"
REPO="$NS/meaningful-moments"
MM_ROOT="${MM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
STAGING="$MM_ROOT/hf_release_staging/v1.0"
VENV="$MM_ROOT/oracle/venv/qwen"

source "$VENV/bin/activate"

# the card + croissant carry the repo id — patch placeholder, then re-finalize
# hashes (README changed), then regenerate croissant from the final hashes
# (croissant.json itself is excluded from sha256sums.txt — no circularity)
if grep -q "<namespace>" "$STAGING/README.md"; then
    sed -i "s|<namespace>|$NS|g" "$STAGING/README.md"
    ( cd scripts/hf_release && \
      python stage.py --finalize-sha && \
      python make_croissant.py --repo-id "$REPO" )
    echo "Patched namespace; sha256sums + croissant finalized for $REPO"
fi

if [ "${2:-}" != "resume" ]; then
    hf repo create "$REPO" --repo-type dataset --private || true
fi

hf upload-large-folder "$REPO" "$STAGING" --repo-type dataset
echo "Upload complete. Next: python scripts/hf_release/verify_remote.py --repo-id $REPO"
