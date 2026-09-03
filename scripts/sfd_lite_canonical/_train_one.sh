#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
ID="${1:?Experiment ID required}"
DATA_ROOT="${VISDRONE_CANONICAL_ROOT:?Set VISDRONE_CANONICAL_ROOT}"
RUN_ROOT="${SFD_CANONICAL_RUN_ROOT:?Set SFD_CANONICAL_RUN_ROOT}"
RUN_NAME="${ID}_seed0"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
LOG_IN_PROGRESS="$RUN_ROOT/.${RUN_NAME}.training.log.in_progress"
LOG_FAILED="$RUN_ROOT/${RUN_NAME}.training.failed.log"
mkdir -p "$RUN_ROOT"
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing existing run directory: $RUN_DIR" >&2; exit 73; }
refuse_file "$LOG_IN_PROGRESS"
refuse_file "$LOG_FAILED"
print_identity
echo "EXPERIMENT_ID=$ID"
set +e
"$PYTHON" "$SCRIPT_DIR/train_canonical.py" \
  --id "$ID" --gate-dir "$GATE_DIR" --dataset-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" --device "${CANONICAL_DEVICE:-0}" \
  2>&1 | tee "$LOG_IN_PROGRESS"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e
if [[ -d "$RUN_DIR" ]]; then
  mv "$LOG_IN_PROGRESS" "$RUN_DIR/training.log"
  sha256sum "$RUN_DIR/training.log" > "$RUN_DIR/training.log.sha256"
else
  mv "$LOG_IN_PROGRESS" "$LOG_FAILED"
fi
exit "$TRAIN_STATUS"
