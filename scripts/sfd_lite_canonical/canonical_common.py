"""Shared helpers for SFD-Lite Canonical v1."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


VERSION = "SFD-Lite-Canonical-v1"
EXPECTED_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
    "awning-tricycle", "bus", "motor",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_root() -> Path:
    return repo_root() / "experiments" / "sfd_lite_canonical"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(relative_to.resolve()).as_posix() if relative_to else str(resolved),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def write_json_exclusive(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing existing output directory: {path}")
    path.mkdir(parents=True)


def git_bytes(args: list[str]) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root().as_posix()}", *args],
            cwd=repo_root(), check=True, capture_output=True, text=False, timeout=30,
        )
        return result.stdout
    except Exception:
        return None


def git_value(args: list[str], empty: str = "UNKNOWN") -> str:
    payload = git_bytes(args)
    if payload is None:
        return "UNKNOWN"
    return payload.decode("utf-8", errors="replace").strip() or empty


def git_state() -> dict[str, Any]:
    # Remote URLs are deliberately excluded because repository configuration may contain credentials.
    status = git_value(["status", "--porcelain=v1"], empty="CLEAN")
    diff = git_bytes(["diff", "--binary"])
    return {
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["branch", "--show-current"]),
        "status": status,
        "dirty": status != "CLEAN",
        "tracked_diff_sha256": hashlib.sha256(diff if diff is not None else b"UNKNOWN").hexdigest(),
        "remote_urls_recorded": False,
    }


def require_gate(path: Path, expected_key: str, expected_value: str = "PASS") -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing gate receipt: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get(expected_key) != expected_value:
        raise RuntimeError(f"Gate failed: {path} has {expected_key}={payload.get(expected_key)!r}")
    return payload


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is unset: {name}")
    return value
