"""Audit and deterministically convert official VisDrone2019-DET data.

Raw data are never modified. ``build`` refuses an existing output directory.
Canonical YOLO labels and 10-class COCO GT are derived from the same parsed
records so their category and box mappings cannot drift.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from canonical_common import EXPECTED_CLASSES, VERSION, file_record, require_new_directory, sha256, write_json_exclusive


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = {
    "train": ("VisDrone2019-DET-train", 6471),
    "val": ("VisDrone2019-DET-val", 548),
}


def image_files(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def parse_annotation(path: Path, width: int, height: int) -> tuple[list[dict[str, Any]], Counter[str], list[str]]:
    kept: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    issues: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = tuple(part.strip() for part in line.split(","))
        counts["raw_annotations"] += 1
        if fields in seen:
            counts["duplicate_annotations"] += 1
            issues.append(f"{path.name}:{line_number}: duplicate raw annotation")
        seen.add(fields)
        if len(fields) < 8:
            issues.append(f"{path.name}:{line_number}: expected >=8 fields, got {len(fields)}")
            continue
        try:
            x, y, box_w, box_h = map(float, fields[:4])
            score, category, truncation, occlusion = (int(float(value)) for value in fields[4:8])
        except ValueError:
            issues.append(f"{path.name}:{line_number}: non-numeric field")
            continue
        if not all(math.isfinite(value) for value in (x, y, box_w, box_h)) or box_w <= 0 or box_h <= 0:
            issues.append(f"{path.name}:{line_number}: invalid bbox {(x, y, box_w, box_h)}")
            continue
        if score not in {0, 1}:
            issues.append(f"{path.name}:{line_number}: unexpected score {score}")
        if truncation not in {0, 1, 2} or occlusion not in {0, 1, 2}:
            issues.append(
                f"{path.name}:{line_number}: truncation/occlusion outside 0..2: {truncation}/{occlusion}"
            )
        if score == 0 or category == 0:
            counts["ignored_region_or_score0"] += 1
            continue
        if category == 11:
            counts["ignored_others_category11"] += 1
            continue
        if not 1 <= category <= 10:
            counts["invalid_category"] += 1
            issues.append(f"{path.name}:{line_number}: category outside 0..11: {category}")
            continue
        if x < 0 or y < 0 or x + box_w > width + 1e-6 or y + box_h > height + 1e-6:
            issues.append(
                f"{path.name}:{line_number}: retained bbox outside image {width}x{height}: {(x, y, box_w, box_h)}"
            )
            continue
        counts["retained_annotations"] += 1
        counts[f"retained_category_{category}"] += 1
        kept.append(
            {
                "category_id": category,
                "bbox": [x, y, box_w, box_h],
                "truncation": truncation,
                "occlusion": occlusion,
            }
        )
    return kept, counts, issues


def audit_split(raw_root: Path, split: str, expected_count: int) -> dict[str, Any]:
    folder = raw_root / SPLITS[split][0]
    images_dir = folder / "images"
    annotations_dir = folder / "annotations"
    issues: list[str] = []
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        return {"split": split, "issues": [f"missing official directories under {folder}"]}
    images = image_files(images_dir)
    annotations = sorted(annotations_dir.glob("*.txt"))
    image_by_stem = {item.stem: item for item in images}
    annotation_by_stem = {item.stem: item for item in annotations}
    if len(image_by_stem) != len(images):
        issues.append("duplicate image stems")
    if len(annotation_by_stem) != len(annotations):
        issues.append("duplicate annotation stems")
    if len(images) != expected_count:
        issues.append(f"image count {len(images)} != expected {expected_count}")
    missing_annotations = sorted(set(image_by_stem) - set(annotation_by_stem))
    missing_images = sorted(set(annotation_by_stem) - set(image_by_stem))
    if missing_annotations or missing_images:
        issues.append(
            f"file pairing mismatch: missing_annotations={len(missing_annotations)}, missing_images={len(missing_images)}"
        )

    aggregate: Counter[str] = Counter()
    image_manifest: list[dict[str, Any]] = []
    annotation_manifest: list[dict[str, Any]] = []
    parsed: dict[str, list[dict[str, Any]]] = {}
    image_ids: set[int] = set()
    empty_images = 0
    for stem in sorted(set(image_by_stem) & set(annotation_by_stem)):
        image_path = image_by_stem[stem]
        annotation_path = annotation_by_stem[stem]
        if not stem.isnumeric():
            issues.append(f"non-numeric VisDrone stem cannot use canonical image_id rule: {stem}")
            continue
        image_id = int(stem)
        if image_id in image_ids:
            issues.append(f"duplicate numeric image_id after removing zero padding: {image_id}")
        image_ids.add(image_id)
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        kept, counts, parse_issues = parse_annotation(annotation_path, width, height)
        aggregate.update(counts)
        issues.extend(parse_issues)
        if not kept:
            empty_images += 1
        parsed[stem] = kept
        image_manifest.append(
            {**file_record(image_path), "stem": stem, "image_id": image_id, "width": width, "height": height}
        )
        annotation_manifest.append({**file_record(annotation_path), "stem": stem})
    return {
        "split": split,
        "official_folder": str(folder.resolve()),
        "expected_image_count": expected_count,
        "image_count": len(images),
        "annotation_file_count": len(annotations),
        "paired_count": len(parsed),
        "empty_retained_annotation_images": empty_images,
        "record_counts": dict(sorted(aggregate.items())),
        "images": image_manifest,
        "raw_annotations": annotation_manifest,
        "parsed": parsed,
        "issues": issues,
    }


def audit_raw(raw_root: Path) -> dict[str, Any]:
    splits = [audit_split(raw_root, split, expected) for split, (_, expected) in SPLITS.items()]
    issues = [f"{item['split']}: {issue}" for item in splits for issue in item.get("issues", [])]
    return {
        "version": VERSION,
        "RAW_VISDRONE_AUDIT": "PASS" if not issues else "FAIL",
        "raw_root": str(raw_root.resolve()),
        "class_mapping": {index + 1: name for index, name in enumerate(EXPECTED_CLASSES)},
        "rules": {
            "retained": "score=1 and category_id in 1..10",
            "ignored_regions": "score=0 or category_id=0; omitted from YOLO and 10-class COCO GT",
            "ignored_others": "category_id=11; omitted from YOLO and 10-class COCO GT",
            "truncation_occlusion": "preserved as COCO custom fields; not used as training targets or evaluator ignore flags",
            "bbox": "raw xywh preserved; retained boxes must be inside the image and have positive width/height",
            "duplicate": "exact duplicate raw rows are a hard failure",
            "image_id": "integer value of the numeric filename stem",
            "category_id": "raw 1..10 in COCO; raw minus one in YOLO",
            "empty_images": "retained with an empty YOLO label and a COCO image record",
        },
        "splits": splits,
        "issues": issues,
    }


def yolo_line(record: dict[str, Any], width: int, height: int) -> str:
    x, y, box_w, box_h = record["bbox"]
    cx, cy = (x + box_w / 2) / width, (y + box_h / 2) / height
    return f"{record['category_id'] - 1} {cx:.8f} {cy:.8f} {box_w / width:.8f} {box_h / height:.8f}"


def build(raw_root: Path, output: Path, audit_output: Path | None) -> None:
    audit = audit_raw(raw_root)
    if audit_output:
        compact = {key: value for key, value in audit.items() if key != "splits"}
        compact["splits"] = [
            {key: value for key, value in item.items() if key != "parsed"} for item in audit["splits"]
        ]
        write_json_exclusive(audit_output, compact)
    if audit["RAW_VISDRONE_AUDIT"] != "PASS":
        raise RuntimeError(f"Raw VisDrone audit failed with {len(audit['issues'])} issue(s)")
    require_new_directory(output)
    (output / "images").mkdir()
    (output / "labels").mkdir()
    (output / "annotations").mkdir()

    generated: list[dict[str, Any]] = []
    split_summaries = []
    for split_report in audit["splits"]:
        split = split_report["split"]
        source_images = Path(split_report["official_folder"]) / "images"
        image_link = output / "images" / split
        os.symlink(source_images, image_link, target_is_directory=True)
        labels_dir = output / "labels" / split
        labels_dir.mkdir()
        image_meta = {item["stem"]: item for item in split_report["images"]}
        coco_images: list[dict[str, Any]] = []
        coco_annotations: list[dict[str, Any]] = []
        annotation_id = 1
        for stem, records in sorted(split_report["parsed"].items()):
            meta = image_meta[stem]
            label_path = labels_dir / f"{stem}.txt"
            text = "\n".join(yolo_line(record, meta["width"], meta["height"]) for record in records)
            label_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
            generated.append(file_record(label_path, output))
            image_path = Path(meta["path"])
            coco_images.append(
                {
                    "id": meta["image_id"], "file_name": image_path.name,
                    "width": meta["width"], "height": meta["height"],
                }
            )
            for record in records:
                x, y, box_w, box_h = record["bbox"]
                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": meta["image_id"],
                        "category_id": record["category_id"],
                        "bbox": [x, y, box_w, box_h],
                        "area": box_w * box_h,
                        "iscrowd": 0,
                        "truncation": record["truncation"],
                        "occlusion": record["occlusion"],
                    }
                )
                annotation_id += 1
        coco = {
            "info": {"description": f"{VERSION} VisDrone {split}; 10 retained classes"},
            "licenses": [],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [
                {"id": index + 1, "name": name, "supercategory": "object"}
                for index, name in enumerate(EXPECTED_CLASSES)
            ],
        }
        coco_path = output / "annotations" / f"instances_{split}.json"
        coco_path.write_text(json.dumps(coco, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        generated.append(file_record(coco_path, output))
        split_summaries.append(
            {
                "split": split,
                "images": len(coco_images),
                "retained_annotations": len(coco_annotations),
                "empty_images": split_report["empty_retained_annotation_images"],
                "yolo_labels": len(list(labels_dir.glob("*.txt"))),
                "coco_gt": str(coco_path.resolve()),
            }
        )

    data_yaml = output / "visdrone_canonical.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(output.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(EXPECTED_CLASSES)},
            },
            sort_keys=False, allow_unicode=True,
        ),
        encoding="utf-8",
    )
    generated.append(file_record(data_yaml, output))
    raw_files = [
        {key: value for key, value in record.items() if key != "path"} | {"path": record["path"]}
        for split_report in audit["splits"]
        for record in split_report["images"] + split_report["raw_annotations"]
    ]
    manifest = {
        "version": VERSION,
        "raw_root": str(raw_root.resolve()),
        "raw_files": raw_files,
        "generated_files": generated,
        "symlinks": {
            split: {"path": str((output / "images" / split).resolve()), "source": str((raw_root / folder / "images").resolve())}
            for split, (folder, _) in SPLITS.items()
        },
    }
    manifest_path = output / "FILE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol = {
        "version": VERSION,
        "VISDRONE_CANONICAL_DATA_PROTOCOL": "PASS",
        "raw_audit": "PASS",
        "data_yaml": file_record(data_yaml),
        "file_manifest": file_record(manifest_path),
        "splits": split_summaries,
        "rules": audit["rules"],
        "documented_difference": (
            "Ignored raw regions and category 11 are omitted. The main Ultralytics YOLO-label evaluator does not "
            "mask predictions overlapping those omitted regions; this must be disclosed and is not VisDrone official toolkit evaluation."
        ),
    }
    (output / "PROTOCOL_REPORT.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"VISDRONE_CANONICAL_DATA_PROTOCOL=PASS\nOUTPUT={output.resolve()}")


def verify(dataset_root: Path, receipt: Path) -> None:
    manifest_path = dataset_root / "FILE_MANIFEST.json"
    protocol_path = dataset_root / "PROTOCOL_REPORT.json"
    if not manifest_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("Canonical manifest/protocol report missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues = []
    for section in ("raw_files", "generated_files"):
        for item in manifest[section]:
            path = Path(item["path"])
            if section == "generated_files":
                path = dataset_root / path
            if not path.is_file():
                issues.append(f"missing: {path}")
            elif path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                issues.append(f"hash/size mismatch: {path}")
    for split, record in manifest["symlinks"].items():
        link = dataset_root / "images" / split
        if not link.is_dir() or link.resolve() != Path(record["source"]).resolve():
            issues.append(f"image symlink mismatch: {link}")
    payload = {
        "version": VERSION,
        "VISDRONE_CANONICAL_DATA_PROTOCOL": "PASS" if not issues else "FAIL",
        "dataset_root": str(dataset_root.resolve()),
        "protocol_report_sha256": sha256(protocol_path),
        "file_manifest_sha256": sha256(manifest_path),
        "issues": issues,
    }
    write_json_exclusive(receipt, payload)
    print(f"VISDRONE_CANONICAL_DATA_PROTOCOL={payload['VISDRONE_CANONICAL_DATA_PROTOCOL']}")
    if issues:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--raw-root", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--raw-root", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--audit-output", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--dataset-root", type=Path, required=True)
    verify_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        report = audit_raw(args.raw_root)
        for split in report["splits"]:
            split.pop("parsed", None)
        write_json_exclusive(args.output, report)
        print(f"RAW_VISDRONE_AUDIT={report['RAW_VISDRONE_AUDIT']}")
        if report["RAW_VISDRONE_AUDIT"] != "PASS":
            raise SystemExit(2)
    elif args.command == "build":
        build(args.raw_root, args.output, args.audit_output)
    else:
        verify(args.dataset_root, args.receipt)


if __name__ == "__main__":
    main()
