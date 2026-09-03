#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
RAW_ROOT="${VISDRONE_RAW_ROOT:?Set VISDRONE_RAW_ROOT to the official VisDrone2019-DET parent directory}"
print_identity
refuse_file "$GATE_DIR/raw_data_audit.json"
"$PYTHON" "$SCRIPT_DIR/canonical_data.py" audit --raw-root "$RAW_ROOT" --output "$GATE_DIR/raw_data_audit.json"
