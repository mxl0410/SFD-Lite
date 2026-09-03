#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
print_identity
refuse_file "$GATE_DIR/package_verify.json"
refuse_file "$GATE_DIR/environment.json"
"$PYTHON" "$SCRIPT_DIR/package_manifest.py" verify \
  --manifest "$DOC_ROOT/CANONICAL_FILE_MANIFEST.json" \
  --receipt "$GATE_DIR/package_verify.json"
"$PYTHON" "$SCRIPT_DIR/check_environment.py" --require-cuda --output "$GATE_DIR/environment.json"
