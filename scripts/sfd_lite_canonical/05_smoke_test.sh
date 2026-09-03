#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
print_identity
refuse_file "$GATE_DIR/smoke.json"
"$PYTHON" "$SCRIPT_DIR/protocol_tests.py" --output "$GATE_DIR/smoke.json"
