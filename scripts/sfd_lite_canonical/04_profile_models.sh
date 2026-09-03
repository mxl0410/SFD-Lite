#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
print_identity
refuse_file "$GATE_DIR/complexity.json"
"$PYTHON" "$SCRIPT_DIR/profile_models.py" \
  --model B0 "$PACKAGE_ROOT/models/B0_yolo11n_ciou.yaml" \
  --model C0 "$PACKAGE_ROOT/models/C0_sfd_lite_canonical.yaml" \
  --model C1 "$PACKAGE_ROOT/models/C1_no_sa_spd_lite.yaml" \
  --model C2 "$PACKAGE_ROOT/models/C2_sfd_lite_ciou.yaml" \
  --model C3 "$PACKAGE_ROOT/models/C3_no_sf_dca_lite.yaml" \
  --imgsz 640 --output "$GATE_DIR/complexity.json"
