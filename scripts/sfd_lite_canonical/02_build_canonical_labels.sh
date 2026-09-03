#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
RAW_ROOT="${VISDRONE_RAW_ROOT:?Set VISDRONE_RAW_ROOT}"
DATA_ROOT="${VISDRONE_CANONICAL_ROOT:?Set VISDRONE_CANONICAL_ROOT to a nonexistent output directory}"
print_identity
[[ ! -e "$DATA_ROOT" ]] || { echo "Refusing existing canonical dataset: $DATA_ROOT" >&2; exit 73; }
refuse_file "$GATE_DIR/dataset_verify.json"
"$PYTHON" "$SCRIPT_DIR/canonical_data.py" build --raw-root "$RAW_ROOT" --output "$DATA_ROOT"
"$PYTHON" "$SCRIPT_DIR/canonical_data.py" verify --dataset-root "$DATA_ROOT" --receipt "$GATE_DIR/dataset_verify.json"
