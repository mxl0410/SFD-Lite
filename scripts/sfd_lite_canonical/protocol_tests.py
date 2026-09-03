"""Loss, architecture and gradient sanity tests for the canonical gate."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from canonical_common import VERSION, file_record, package_root, write_json_exclusive


def structural_diffs(left: Any, right: Any, path: str = "root") -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [{"path": path, "left": left, "right": right}]
    if isinstance(left, dict):
        rows = []
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                rows.append({"path": f"{path}.{key}", "left": left.get(key), "right": right.get(key)})
            else:
                rows.extend(structural_diffs(left[key], right[key], f"{path}.{key}"))
        return rows
    if isinstance(left, list):
        if len(left) != len(right):
            return [{"path": path, "left": left, "right": right}]
        rows = []
        for index, value in enumerate(left):
            rows.extend(structural_diffs(value, right[index], f"{path}[{index}]"))
        return rows
    return [] if left == right else [{"path": path, "left": left, "right": right}]


def load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def run(output: Path) -> None:
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.metrics import bbox_iou

    issues = []
    results: dict[str, Any] = {}
    box_a = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 4.0, 8.0], [0.0, 0.0, 2.0, 2.0]],
        dtype=torch.float32, requires_grad=True,
    )
    box_b = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 2.0, 5.0, 10.0], [20.0, 20.0, 22.0, 22.0]],
        dtype=torch.float32,
    )
    losses = {}
    for mode in ("ciou", "dynwd"):
        os.environ["SFD_CANONICAL_REGRESSION"] = mode
        criterion = BboxLoss(reg_max=8)
        values = (
            criterion._official_ciou_loss(box_a, box_b)
            if mode == "ciou" else criterion._dynamic_ciou_nwd_loss(box_a, box_b)
        )
        losses[mode] = values.detach().tolist()
        if values.shape != (3,) or not torch.isfinite(values).all():
            issues.append(f"{mode}: invalid loss shape or non-finite value")
        gradient = torch.autograd.grad(values.sum(), box_a, retain_graph=True)[0]
        if not torch.isfinite(gradient).all():
            issues.append(f"{mode}: non-finite gradient")
    official_reference = 1.0 - bbox_iou(box_a, box_b, xywh=False, CIoU=True).squeeze(-1)
    if not torch.allclose(torch.tensor(losses["ciou"]), official_reference.detach(), atol=1e-7, rtol=1e-6):
        issues.append("CIoU branch does not match restored bbox_iou reference")
    if abs(losses["ciou"][0]) > 1e-6 or abs(losses["dynwd"][0]) > 1e-3:
        issues.append("identical-box loss sanity failed")
    if torch.allclose(torch.tensor(losses["ciou"]), torch.tensor(losses["dynwd"]), atol=1e-8, rtol=1e-8):
        issues.append("CIoU and DyNWD branches are unexpectedly identical")
    if "nwd" in inspect.getsource(bbox_iou).lower() or "wasserstein" in inspect.getsource(bbox_iou).lower():
        issues.append("assignment-visible bbox_iou contains NWD logic")
    results["loss"] = {"values": losses, "official_reference": official_reference.detach().tolist()}

    models_dir = package_root() / "models"
    paths = {model_id: models_dir / filename for model_id, filename in {
        "B0": "B0_yolo11n_ciou.yaml", "C0": "C0_sfd_lite_canonical.yaml",
        "C1": "C1_no_sa_spd_lite.yaml", "C2": "C2_sfd_lite_ciou.yaml",
        "C3": "C3_no_sf_dca_lite.yaml",
    }.items()}
    yamls = {model_id: load(path) for model_id, path in paths.items()}
    expected_diffs = {
        "C1": [{"path": "root.backbone[1][2]", "left": "SA_SPD_Lite", "right": "Stride2ConvBaseline"}],
        "C3": [{"path": "root.head[12][2]", "left": "SF_DCA_Lite", "right": "nn.Identity"},
               {"path": "root.head[12][3]", "left": [256, 4], "right": []}],
    }
    if structural_diffs(yamls["C0"], yamls["C2"]):
        issues.append("C2 model YAML is not structurally identical to C0")
    for model_id in ("C1", "C3"):
        actual = structural_diffs(yamls["C0"], yamls[model_id])
        if actual != expected_diffs[model_id]:
            issues.append(f"{model_id}: unexpected structural diff {actual}")
        results[f"{model_id}_diff"] = actual

    model_rows = []
    torch.manual_seed(0)
    for model_id, path in paths.items():
        model = DetectionModel(str(path.resolve()), ch=3, verbose=False).eval().cpu().float()
        layer_shapes = {}
        handles = []
        inspected_indices = (1, 23) if model_id != "B0" else (1,)
        for index in inspected_indices:
            if index < len(model.model):
                handles.append(
                    model.model[index].register_forward_hook(
                        lambda module, inputs, value, idx=index: layer_shapes.update({str(idx): list(value.shape)})
                    )
                )
        with torch.no_grad():
            output_value = model(torch.zeros(1, 3, 640, 640))
        for handle in handles:
            handle.remove()
        model_rows.append(
            {
                "id": model_id, "yaml": file_record(path),
                "params": sum(parameter.numel() for parameter in model.parameters()),
                "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                "layer_shapes": layer_shapes, "output_type": type(output_value).__name__,
            }
        )
    by_id = {row["id"]: row for row in model_rows}
    if by_id["C0"]["layer_shapes"].get("1") != by_id["C1"]["layer_shapes"].get("1"):
        issues.append("C1 layer-1 output shape differs from C0")
    if by_id["C0"]["layer_shapes"].get("23") != by_id["C3"]["layer_shapes"].get("23"):
        issues.append("C3 identity output shape differs from C0 SF-DCA-Lite output")
    results["models"] = model_rows
    payload = {
        "version": VERSION,
        "CANONICAL_SMOKE_TEST": "PASS" if not issues else "FAIL",
        "results": results,
        "issues": issues,
    }
    write_json_exclusive(output, payload)
    print(f"CANONICAL_SMOKE_TEST={payload['CANONICAL_SMOKE_TEST']}")
    if issues:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
