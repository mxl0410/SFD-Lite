"""Canonical, read-only complexity benchmark for SFD-Lite Phase 2.5.

The script profiles every YAML with the same source tree, input tensor and
unfused state. It reports THOP MACs and direct torch.profiler FLOPs separately;
it never manufactures FLOPs by multiplying MACs by two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch

from canonical_common import VERSION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def thop_macs(model: torch.nn.Module, image: torch.Tensor) -> tuple[int, int]:
    import thop

    macs, params = thop.profile(model, inputs=(image,), verbose=False)
    return int(macs), int(params)


def torch_profiler_flops(model: torch.nn.Module, image: torch.Tensor) -> tuple[int, list[dict[str, Any]]]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.no_grad(), torch.profiler.profile(activities=activities, with_flops=True) as profiler:
        model(image)
    events = profiler.key_averages()
    total = int(sum(int(getattr(event, "flops", 0) or 0) for event in events))
    rows = [
        {
            "operator": event.key,
            "flops": int(getattr(event, "flops", 0) or 0),
            "calls": int(event.count),
        }
        for event in events
        if int(getattr(event, "flops", 0) or 0) > 0
    ]
    rows.sort(key=lambda row: row["flops"], reverse=True)
    return total, rows


def profile_one(label: str, yaml_path: Path, imgsz: int) -> dict[str, Any]:
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(0)
    model = DetectionModel(str(yaml_path.resolve()), ch=3, verbose=False).eval().cpu().float()
    image = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)
    with torch.no_grad():
        model(image)  # materialize Detect strides/caches before profiling
    macs, thop_params = thop_macs(model, image)
    flops, operator_rows = torch_profiler_flops(model, image)
    return {
        "label": label,
        "yaml": str(yaml_path.resolve()),
        "yaml_sha256": sha256(yaml_path),
        "state": "unfused_eval_fp32_cpu",
        "input": [1, 3, imgsz, imgsz],
        "params_direct": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params_direct": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "thop_params": thop_params,
        "thop_macs": macs,
        "thop_gmacs": macs / 1e9,
        "torch_profiler_flops": flops,
        "torch_profiler_gflops": flops / 1e9,
        "torch_profiler_nonzero_operators": operator_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", nargs=2, metavar=("LABEL", "YAML"), required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    models = [(label, Path(path)) for label, path in args.model]
    for _, path in models:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import thop
    import ultralytics

    report = {
        "version": VERSION,
        "CANONICAL_COMPLEXITY": "PASS",
        "paper_policy": {
            "primary_params": "direct unfused parameter count",
            "macs": "THOP direct multiply-accumulate count",
            "flops": "torch.profiler direct operator FLOP attribution",
            "prohibited": "Do not derive or publish FLOPs as 2 x MACs without explicit labeling.",
            "coverage_warning": (
                "Both profilers may omit unsupported custom or functional operators. Compare models only within "
                "this exact tool/source/input/state protocol and retain the operator audit."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "thop": getattr(thop, "__version__", "UNKNOWN"),
            "platform": platform.platform(),
        },
        "models": [profile_one(label, path, args.imgsz) for label, path in models],
    }
    args.output.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"COMPLEXITY_OUTPUT={args.output.resolve()}")
    for item in report["models"]:
        print(
            f"{item['label']}: params={item['params_direct']} "
            f"gmacs={item['thop_gmacs']:.9f} profiler_gflops={item['torch_profiler_gflops']:.9f}"
        )


if __name__ == "__main__":
    main()
