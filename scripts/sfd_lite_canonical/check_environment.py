"""Fail-fast environment gate for authenticated server execution."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

from canonical_common import VERSION, git_state, package_root, write_json_exclusive


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    import ultralytics

    environment = {
        "python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count(),
        "ultralytics": ultralytics.__version__, "thop": version("ultralytics-thop"),
        "pycocotools": version("pycocotools"), "numpy": version("numpy"),
        "pyyaml": version("PyYAML"), "pillow": version("Pillow"), "os": platform.platform(),
        "ultralytics_source": str(Path(ultralytics.__file__).resolve()),
    }
    issues = []
    if not environment["python"].startswith("3.12."):
        issues.append(f"Python must be 3.12.x, got {environment['python']}")
    if not str(environment["torch"]).startswith("2.13.0"):
        issues.append(f"Torch must be 2.13.0 build, got {environment['torch']}")
    if environment["ultralytics"] != "8.4.60":
        issues.append(f"Ultralytics must be 8.4.60, got {environment['ultralytics']}")
    if environment["thop"] != "2.0.20":
        issues.append(f"ultralytics-thop must be 2.0.20, got {environment['thop']}")
    expected_packages = {"pycocotools": "2.0.10", "numpy": "2.4.4", "pyyaml": "6.0.3", "pillow": "12.2.0"}
    for name, expected_version in expected_packages.items():
        if environment[name] != expected_version:
            issues.append(f"{name} must be {expected_version}, got {environment[name]}")
    source_root = package_root() / "source"
    if source_root.resolve() not in Path(ultralytics.__file__).resolve().parents:
        issues.append("Ultralytics import is not from the canonical source snapshot")
    if args.require_cuda and not torch.cuda.is_available():
        issues.append("CUDA GPU is required but torch.cuda.is_available() is false")
    payload = {
        "version": VERSION,
        "CANONICAL_ENVIRONMENT": "PASS" if not issues else "FAIL",
        "environment": environment,
        "git": git_state(),
        "dirty_policy": (
            "Outer worktree dirtiness is recorded and is not silently promoted to identity; the canonical package "
            "file manifest must independently verify before any run."
        ),
        "issues": issues,
    }
    write_json_exclusive(args.output, payload)
    print(f"CANONICAL_ENVIRONMENT={payload['CANONICAL_ENVIRONMENT']}")
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
