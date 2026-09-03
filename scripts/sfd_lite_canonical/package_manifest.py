"""Generate or verify the canonical package file manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from canonical_common import VERSION, file_record, git_state, repo_root, sha256, write_json_exclusive


SCOPES = (
    "experiments/sfd_lite_canonical",
    "scripts/sfd_lite_canonical",
    "docs/paper_audit/canonical",
)
MANIFEST = "docs/paper_audit/canonical/CANONICAL_FILE_MANIFEST.json"


def current_files() -> list[Path]:
    root = repo_root()
    files = []
    for scope in SCOPES:
        scope_path = root / scope
        if not scope_path.exists():
            continue
        files.extend(
            path for path in scope_path.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path != root / MANIFEST
        )
    return sorted(set(files))


def generate(output: Path) -> None:
    root = repo_root()
    payload = {
        "version": VERSION,
        "CANONICAL_PACKAGE": "FROZEN",
        "git": git_state(),
        "scopes": list(SCOPES),
        "files": [file_record(path, root) for path in current_files()],
        "note": "Remote URLs are excluded; source identity is this manifest plus the recorded commit/branch.",
    }
    write_json_exclusive(output, payload)
    print(f"CANONICAL_PACKAGE=FROZEN\nFILES={len(payload['files'])}")


def verify(manifest: Path, receipt: Path | None) -> None:
    root = repo_root()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in payload["files"]}
    actual_paths = {path.resolve().relative_to(root.resolve()).as_posix(): path for path in current_files()}
    issues = []
    for relative, item in expected.items():
        path = root / relative
        if not path.is_file():
            issues.append(f"missing: {relative}")
        elif path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            issues.append(f"hash/size mismatch: {relative}")
    extras = sorted(set(actual_paths) - set(expected))
    if extras:
        issues.append(f"unmanifested canonical files: {extras}")
    result = {
        "version": VERSION,
        "CANONICAL_PACKAGE_VERIFY": "PASS" if not issues else "FAIL",
        "manifest": str(manifest.resolve()), "manifest_sha256": sha256(manifest),
        "file_count": len(expected), "issues": issues,
    }
    if receipt:
        write_json_exclusive(receipt, result)
    print(f"CANONICAL_PACKAGE_VERIFY={result['CANONICAL_PACKAGE_VERIFY']}")
    if issues:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.output)
    else:
        verify(args.manifest, args.receipt)


if __name__ == "__main__":
    main()
