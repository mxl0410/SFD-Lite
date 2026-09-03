# VisDrone Canonical Data Protocol

## Source and split

Only untouched official `VisDrone2019-DET-train` and `VisDrone2019-DET-val` directories are accepted. Expected image counts are 6,471 train and 548 official-val. Historical YOLO labels or COCO JSON are never accepted as inputs.

## Deterministic mapping

The single parser in `canonical_data.py` produces both YOLO labels and COCO GT:

- raw category 1–10 → YOLO class 0–9;
- raw category 1–10 → COCO category 1–10;
- score=0 or category=0 → ignored region, omitted from both targets;
- category=11 (`others`) → omitted from both targets;
- truncation and occlusion must be integers 0–2, are retained as supplemental COCO fields, and do not alter the training target;
- raw xywh is preserved; retained boxes must have positive extent and lie inside the image;
- image ID is the integer value of the numeric filename stem;
- empty-target images remain in both formats with an empty YOLO label;
- exact duplicate annotations, missing images/annotations, duplicate stems/IDs, corrupt images, invalid classes and out-of-bounds retained boxes are hard failures.

## Manifests

The build records per-file size and SHA256 for every raw image and annotation, every generated YOLO label, both COCO GT files and the runtime data YAML. Images are linked read-only into the new canonical dataset namespace; source data are not modified.

## Ignore handling boundary

The main Ultralytics YOLO-label evaluator does not mask predictions overlapping omitted ignore regions. This is a deliberate documented difference from a VisDrone-specific official toolkit with ignore-region filtering. The paper must name the split “VisDrone2019-DET official val split” but must not claim the main metrics came from the official VisDrone evaluation toolkit.

## Gate

`01_verify_dataset.sh` audits the raw tree. `02_build_canonical_labels.sh` creates a new output directory and then rehashes all inputs/outputs. Formal training requires `dataset_verify.json` with:

```text
VISDRONE_CANONICAL_DATA_PROTOCOL=PASS
```

The design is complete locally; PASS can only be issued where the official raw data are present.
