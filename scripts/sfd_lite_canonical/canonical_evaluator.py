"""Frozen canonical evaluator and evaluator sanity gate."""

from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import io
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from canonical_common import EXPECTED_CLASSES, VERSION, file_record, require_new_directory, sha256, write_json_exclusive


MAIN_ARGS = {
    "split": "val", "imgsz": 640, "batch": 16, "workers": 4, "half": False,
    "conf": 0.001, "iou": 0.7, "max_det": 300, "save_json": True, "plots": True,
}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_sources() -> dict[str, Path]:
    import ultralytics.models.yolo.detect.val as detect_val
    import ultralytics.utils.loss as loss
    import ultralytics.utils.metrics as metrics

    return {
        "detect_validator": Path(inspect.getfile(detect_val)).resolve(),
        "metrics": Path(inspect.getfile(metrics)).resolve(),
        "loss": Path(inspect.getfile(loss)).resolve(),
        "canonical_evaluator": Path(__file__).resolve(),
    }


def sanity(output: Path, expected_source_root: Path) -> None:
    os.environ["SFD_CANONICAL_REGRESSION"] = "ciou"
    import ultralytics
    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.metrics import bbox_iou

    sources = canonical_sources()
    issues = []
    expected = expected_source_root.resolve()
    for label, path in sources.items():
        if label != "canonical_evaluator" and expected not in path.parents:
            issues.append(f"{label} imported outside canonical source root: {path}")
    metrics_source = sources["metrics"].read_text(encoding="utf-8")
    bbox_source = inspect.getsource(bbox_iou)
    if "nwd" in bbox_source.lower() or "wasserstein" in bbox_source.lower():
        issues.append("metrics.bbox_iou is contaminated by NWD/Wasserstein logic")
    box_a = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 4.0, 8.0]], requires_grad=True)
    box_b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 2.0, 5.0, 10.0]])
    ciou = bbox_iou(box_a, box_b, xywh=False, CIoU=True).squeeze(-1)
    official = BboxLoss(reg_max=8)._official_ciou_loss(box_a, box_b)
    if not torch.allclose(official, 1.0 - ciou, atol=1e-7, rtol=1e-6):
        issues.append("official CIoU loss differs from restored bbox_iou")
    official.sum().backward()
    if box_a.grad is None or not torch.isfinite(box_a.grad).all():
        issues.append("official CIoU gradient sanity failed")
    try:
        import pycocotools
        from pycocotools.cocoeval import COCOeval

        coco_eval = COCOeval(iouType="bbox")
        area_ranges = coco_eval.params.areaRng
        area_labels = coco_eval.params.areaRngLbl
        pycoco_version = getattr(pycocotools, "__version__", "UNKNOWN")
    except Exception as error:
        issues.append(f"pycocotools unavailable: {error}")
        area_ranges, area_labels, pycoco_version = [], [], "UNKNOWN"
    payload = {
        "version": VERSION,
        "CANONICAL_EVALUATOR": "PASS" if not issues else "FAIL",
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "ultralytics": ultralytics.__version__, "pycocotools": pycoco_version,
        },
        "main": {
            "identity": "Ultralytics YOLO-label detection evaluator",
            "arguments": MAIN_ARGS,
            "iou_thresholds": "0.50:0.05:0.95 for mAP50-95",
            "ignore_handling": (
                "Raw ignored regions/category 11 are absent from labels; predictions over omitted regions are not masked."
            ),
        },
        "supplemental": {
            "identity": "pycocotools COCOeval bbox",
            "area_labels": area_labels, "area_ranges": area_ranges,
            "max_detections": [1, 10, 100],
            "paper_label": "COCO-style area-bin supplementary analysis",
        },
        "sources": {label: file_record(path) for label, path in sources.items()},
        "metrics_source_sha256": sha256(sources["metrics"]),
        "metrics_source_contains_nwd_string": "nwd" in metrics_source.lower(),
        "issues": issues,
    }
    write_json_exclusive(output, payload)
    print(f"CANONICAL_EVALUATOR={payload['CANONICAL_EVALUATOR']}")
    if issues:
        raise SystemExit(2)


def per_class_rows(metrics: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    box = metrics.box
    indices = np.asarray(box.ap_class_index, dtype=int)
    all_ap = np.asarray(box.all_ap)
    return [
        {
            "class_id": int(class_id), "class_name": names[int(class_id)],
            "precision": float(np.asarray(box.p)[row]), "recall": float(np.asarray(box.r)[row]),
            "ap50": float(np.asarray(box.ap50)[row]),
            "ap75": float(all_ap[row, 5]) if all_ap.shape[1] > 5 else None,
            "map50_95": float(np.asarray(box.ap)[row]),
        }
        for row, class_id in enumerate(indices)
    ]


def coco_eval(gt_path: Path, predictions: Path) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(gt_path))
    categories = sorted(gt.loadCats(gt.getCatIds()), key=lambda item: item["id"])
    if [item["id"] for item in categories] != list(range(1, 11)) or [item["name"] for item in categories] != EXPECTED_CLASSES:
        raise RuntimeError("Canonical COCO categories do not match frozen 1..10 mapping")
    dt = gt.loadRes(str(predictions))
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        evaluator.summarize()
    return {
        "identity": "COCO-style area-bin supplementary analysis",
        "ap": float(evaluator.stats[0]), "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]), "ap_small": float(evaluator.stats[3]),
        "ap_medium": float(evaluator.stats[4]), "ap_large": float(evaluator.stats[5]),
        "summary": capture.getvalue(), "gt_sha256": sha256(gt_path),
        "predictions_sha256": sha256(predictions),
    }


def evaluate(weight: Path, data: Path, coco_gt: Path, output: Path, device: str) -> None:
    for path in (weight, data, coco_gt):
        if not path.is_file():
            raise FileNotFoundError(path)
    require_new_directory(output)
    from ultralytics import YOLO

    model = YOLO(str(weight.resolve()))
    val_project = output / "ultralytics_val"
    metrics = model.val(
        data=str(data.resolve()), device=device, project=str(val_project), name="canonical_val", exist_ok=False,
        **MAIN_ARGS,
    )
    names = {int(key): str(value) for key, value in model.model.names.items()}
    if [names[index] for index in sorted(names)] != EXPECTED_CLASSES:
        raise RuntimeError(f"Model class mapping mismatch: {names}")
    rows = per_class_rows(metrics, names)
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    predictions = list(val_project.rglob("predictions.json"))
    if len(predictions) != 1:
        raise RuntimeError(f"Expected exactly one predictions.json, found {predictions}")
    supplemental = coco_eval(coco_gt, predictions[0])
    (output / "coco_summary.txt").write_text(supplemental["summary"], encoding="utf-8")
    payload = {
        "version": VERSION,
        "CANONICAL_VALIDATION": "PASS",
        "weight": file_record(weight), "data": file_record(data), "coco_gt": file_record(coco_gt),
        "main_identity": "Ultralytics YOLO-label detection evaluator",
        "main_arguments": {**MAIN_ARGS, "device": device},
        "main_metrics": jsonable(metrics.results_dict),
        "per_class": rows, "supplemental": supplemental,
        "sources": {label: file_record(path) for label, path in canonical_sources().items()},
    }
    write_json_exclusive(output / "canonical_validation.json", payload)
    print(f"CANONICAL_VALIDATION=PASS\nOUTPUT={output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    sanity_parser = subparsers.add_parser("sanity")
    sanity_parser.add_argument("--output", type=Path, required=True)
    sanity_parser.add_argument("--expected-source-root", type=Path, required=True)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--weight", type=Path, required=True)
    eval_parser.add_argument("--data", type=Path, required=True)
    eval_parser.add_argument("--coco-gt", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.command == "sanity":
        sanity(args.output, args.expected_source_root)
    else:
        evaluate(args.weight, args.data, args.coco_gt, args.output, args.device)


if __name__ == "__main__":
    main()
