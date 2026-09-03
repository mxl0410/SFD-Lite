"""Single training entry point for B0/C0/C1/C2/C3.

This file is generated in Phase 3 but is not executed locally. It refuses to
train unless all six canonical gate receipts pass and the run directory is new.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

from canonical_common import (
    VERSION, file_record, git_state, load_yaml, package_root, require_gate,
    require_new_directory, repo_root, sha256, write_json_exclusive,
)


GATES = {
    "package": ("package_verify.json", "CANONICAL_PACKAGE_VERIFY"),
    "environment": ("environment.json", "CANONICAL_ENVIRONMENT"),
    "dataset": ("dataset_verify.json", "VISDRONE_CANONICAL_DATA_PROTOCOL"),
    "evaluator": ("evaluator.json", "CANONICAL_EVALUATOR"),
    "complexity": ("complexity.json", "CANONICAL_COMPLEXITY"),
    "smoke": ("smoke.json", "CANONICAL_SMOKE_TEST"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", choices=["B0", "C0", "C1", "C2", "C3"], required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    gate_records = {
        label: require_gate(args.gate_dir / filename, key)
        for label, (filename, key) in GATES.items()
    }
    dataset_protocol = args.dataset_root / "PROTOCOL_REPORT.json"
    data_yaml = args.dataset_root / "visdrone_canonical.yaml"
    coco_gt = args.dataset_root / "annotations" / "instances_val.json"
    for path in (dataset_protocol, data_yaml, coco_gt):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(dataset_protocol.read_text(encoding="utf-8"))
    if protocol.get("VISDRONE_CANONICAL_DATA_PROTOCOL") != "PASS":
        raise RuntimeError("Canonical dataset protocol is not PASS")

    matrix_path = package_root() / "configs" / "canonical_matrix.yaml"
    training_path = package_root() / "configs" / "canonical_training.yaml"
    matrix = load_yaml(matrix_path)
    training = load_yaml(training_path)
    entry: dict[str, Any] = matrix["experiments"][args.id]
    if matrix.get("initialization") != "scratch" or training.get("pretrained") is not False:
        raise RuntimeError("Canonical v1 requires scratch initialization for every model")
    model_path = package_root() / entry["model"]
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    regression = str(entry["regression"])
    os.environ["SFD_CANONICAL_REGRESSION"] = regression

    run_name = f"{args.id}_seed0"
    args.run_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_root / run_name
    require_new_directory(run_dir)
    prelaunch = {
        "version": VERSION, "experiment_id": args.id, "run_name": run_name,
        "training_started": False, "model": file_record(model_path),
        "regression": regression,
        "initialization": {
            "policy": "scratch", "pretrained_checkpoint": None,
            "transferred_parameters": 0, "missing_keys": "not_applicable",
            "unexpected_keys": "not_applicable", "seed": 0,
        },
        "training_config": file_record(training_path), "matrix": file_record(matrix_path),
        "data_yaml": file_record(data_yaml), "data_manifest": protocol["file_manifest"],
        "gate_receipts": {
            label: file_record(args.gate_dir / GATES[label][0]) for label in GATES
        },
        "git": git_state(), "python": platform.python_version(),
        "source_root": str((package_root() / "source").resolve()),
    }
    write_json_exclusive(run_dir / "PRELAUNCH_MANIFEST.json", prelaunch)

    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    from canonical_evaluator import evaluate

    train_args = {
        key: value for key, value in training.items()
        if key not in {"version", "data", "project", "name", "task", "mode"}
    }
    train_args.update(
        {
            "data": str(data_yaml.resolve()), "project": str(args.run_root.resolve()),
            "name": run_name, "exist_ok": True, "device": args.device,
            "seed": 0, "pretrained": False,
        }
    )
    prelaunch["training_started"] = True
    (run_dir / "PRELAUNCH_MANIFEST.json").write_text(
        json.dumps(prelaunch, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    init_seeds(0, deterministic=True)
    model = YOLO(str(model_path.resolve()))
    results = model.train(**train_args)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    for path in (best, last, results_csv, run_dir / "args.yaml"):
        if not path.is_file():
            raise FileNotFoundError(f"Required training artifact missing: {path}")
    evaluation_dir = run_dir / "canonical_final_validation"
    evaluate(best, data_yaml, coco_gt, evaluation_dir, args.device)
    final_manifest = {
        "version": VERSION, "experiment_id": args.id, "regression": regression,
        "best": file_record(best), "last": file_record(last),
        "args": file_record(run_dir / "args.yaml"), "results_csv": file_record(results_csv),
        "validation": file_record(evaluation_dir / "canonical_validation.json"),
        "source_loss": file_record(package_root() / "source" / "ultralytics" / "utils" / "loss.py"),
        "source_metrics": file_record(package_root() / "source" / "ultralytics" / "utils" / "metrics.py"),
        "model_yaml_sha256": sha256(model_path), "training_return_type": type(results).__name__,
    }
    write_json_exclusive(run_dir / "FINAL_RUN_MANIFEST.json", final_manifest)
    print(f"CANONICAL_TRAINING_COMPLETE={args.id}\nRUN={run_dir.resolve()}")


if __name__ == "__main__":
    main()
