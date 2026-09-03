# SFD-Lite

Official research-code release for **SFD-Lite**, a parameter-compact detector
for aerial small-object detection built on Ultralytics YOLO11.

This repository publishes the frozen `SFD-Lite-Canonical-v1` implementation
used for controlled experiments. It contains source code, model definitions,
training/evaluation gates, and local complexity metadata. It intentionally does
not contain datasets, trained checkpoints, private experiment logs, or
manuscript drafts.

## Main components

- **SA-SPD-Lite**: subposition-aware space-to-depth downsampling.
- **SA-SPD**: spatial-attention-guided space-to-depth downsampling.
- **SF-DCA-Lite**: context-guided residual-detail modulation in a reduced
  feature space.
- **DyNWD**: a training-time quality-aware combination of CIoU and normalized
  Wasserstein distance losses.

The canonical experiment matrix contains a YOLO11n baseline (`B0`), the full
SFD-Lite model (`C0`), and controlled variants (`C1`--`C3`).

## Repository layout

```text
experiments/sfd_lite_canonical/
  configs/      Frozen training protocol and experiment matrix
  models/       B0 and C0--C3 model definitions
  source/       Modified Ultralytics 8.4.60 source snapshot
  static/       Canonical local complexity metadata
scripts/sfd_lite_canonical/
  00--05        Environment, data, evaluator, complexity and smoke-test gates
  10--14        Training launchers for B0 and C0--C3
docs/paper_audit/canonical/  Protocol and model-identity documentation
```

## Installation

Python 3.12 is recommended. Install a CUDA-compatible PyTorch build for your
machine first, then install the frozen source snapshot in editable mode:

```bash
git clone https://github.com/mxl0410/SFD-Lite.git
cd SFD-Lite
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./experiments/sfd_lite_canonical/source
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Smoke test

The smoke test checks model construction, tensor shapes, CIoU/DyNWD numerical
behavior, and gradients for the canonical matrix:

```bash
export PYTHONPATH="$PWD/experiments/sfd_lite_canonical/source:$PWD/scripts/sfd_lite_canonical"
python scripts/sfd_lite_canonical/protocol_tests.py --output gate/smoke.json
```

The output path must be new because the canonical scripts deliberately refuse
to overwrite evidence artifacts.

## Dataset

The dataset is not redistributed. Download VisDrone from its official source
and prepare it according to
[`VISDRONE_CANONICAL_DATA_PROTOCOL.md`](docs/paper_audit/canonical/VISDRONE_CANONICAL_DATA_PROTOCOL.md).
Dataset paths must be supplied locally and must not be committed.

## Canonical training

Training is guarded by six receipts: package, environment, dataset, evaluator,
complexity, and smoke test. Review the scripts before execution and set the
required environment variables:

```bash
export CANONICAL_PYTHON="$(command -v python)"
export CANONICAL_GATE_DIR="$PWD/gate"
export CANONICAL_YOLO_CONFIG_DIR="$PWD/.yolo-config"
export VISDRONE_CANONICAL_ROOT="/absolute/path/to/verified/visdrone"
export CANONICAL_RUN_ROOT="/absolute/path/to/new/run/root"

bash scripts/sfd_lite_canonical/00_check_environment.sh
bash scripts/sfd_lite_canonical/01_verify_dataset.sh
bash scripts/sfd_lite_canonical/02_build_canonical_labels.sh
bash scripts/sfd_lite_canonical/03_verify_evaluator.sh
bash scripts/sfd_lite_canonical/04_profile_models.sh
bash scripts/sfd_lite_canonical/05_smoke_test.sh
bash scripts/sfd_lite_canonical/11_train_C0_seed0.sh
```

Do not treat model names, incomplete runs, or unmatched checkpoints as formal
paper evidence. Accuracy comparisons require the same data split, seed,
training protocol, evaluator, and best-checkpoint rule.

## Reproducibility status

- Source identity: Ultralytics `8.4.60` with the frozen SFD-Lite changes.
- Nominal input: `1 x 3 x 640 x 640`.
- The repository includes no claimed final accuracy result or pretrained
  checkpoint.
- Parameter count and operation counts do not by themselves establish runtime
  speed or real-time deployment performance.

## License and acknowledgement

This repository contains a modified Ultralytics source snapshot and is released
under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Upstream Ultralytics copyrights and
license notices are retained.

## Citation

Citation metadata will be updated after the associated manuscript receives its
final bibliographic record. Until then, cite this repository URL and the exact
commit used in your experiment.
