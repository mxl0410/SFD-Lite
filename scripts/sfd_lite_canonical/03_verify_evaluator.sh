#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
print_identity
refuse_file "$GATE_DIR/evaluator.json"
"$PYTHON" "$SCRIPT_DIR/canonical_evaluator.py" sanity \
  --expected-source-root "$PACKAGE_ROOT/source" --output "$GATE_DIR/evaluator.json"
