#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PACKAGE_ROOT="$REPO_ROOT/experiments/sfd_lite_canonical"
DOC_ROOT="$REPO_ROOT/docs/paper_audit/canonical"
PYTHON="${CANONICAL_PYTHON:?Set CANONICAL_PYTHON to the frozen Python executable}"
GATE_DIR="${CANONICAL_GATE_DIR:?Set CANONICAL_GATE_DIR to a new Phase-3 gate directory}"
export PYTHONPATH="$PACKAGE_ROOT/source:$SCRIPT_DIR"
export YOLO_CONFIG_DIR="${CANONICAL_YOLO_CONFIG_DIR:?Set CANONICAL_YOLO_CONFIG_DIR to a writable isolated directory}"

[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 66; }
mkdir -p "$GATE_DIR" "$YOLO_CONFIG_DIR"

refuse_file() {
  [[ ! -e "$1" ]] || { echo "Refusing overwrite: $1" >&2; exit 73; }
}

print_identity() {
  echo "CANONICAL_VERSION=SFD-Lite-Canonical-v1"
  echo "GIT_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
  echo "TRAIN_CONFIG_SHA256=$(sha256sum "$PACKAGE_ROOT/configs/canonical_training.yaml" | awk '{print $1}')"
  echo "MATRIX_SHA256=$(sha256sum "$PACKAGE_ROOT/configs/canonical_matrix.yaml" | awk '{print $1}')"
  echo "LOSS_SHA256=$(sha256sum "$PACKAGE_ROOT/source/ultralytics/utils/loss.py" | awk '{print $1}')"
  echo "METRICS_SHA256=$(sha256sum "$PACKAGE_ROOT/source/ultralytics/utils/metrics.py" | awk '{print $1}')"
  if [[ -n "${VISDRONE_CANONICAL_ROOT:-}" && -f "$VISDRONE_CANONICAL_ROOT/FILE_MANIFEST.json" ]]; then
    echo "DATA_MANIFEST_SHA256=$(sha256sum "$VISDRONE_CANONICAL_ROOT/FILE_MANIFEST.json" | awk '{print $1}')"
  else
    echo "DATA_MANIFEST_SHA256=NOT_AVAILABLE"
  fi
}
