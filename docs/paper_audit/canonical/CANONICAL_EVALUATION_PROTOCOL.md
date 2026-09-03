# Canonical Evaluation Protocol

## A. Main detection metrics

Identity: Ultralytics 8.4.60 YOLO-label detection evaluator.

- split=`val`, imgsz=640, batch=16, FP32;
- conf=0.001, NMS IoU=0.7, max_det=300;
- IoU thresholds 0.50:0.05:0.95;
- outputs: Precision, Recall, mAP50 and mAP50-95 plus per-class metrics;
- category mapping must exactly equal the frozen 10-class order;
- ignored raw regions/category 11 are absent from labels and predictions over them are not specially masked.

## B. Supplemental small-object analysis

Identity: `pycocotools` 2.0.10 `COCOeval`, bbox mode, canonical 10-class COCO GT.

- outputs: AP, AP50, AP75, AP_small, AP_medium and AP_large;
- COCO default area ranges and max detections are recorded by the sanity receipt;
- paper label must be **“COCO-style area-bin supplementary analysis”**;
- it must not be called “VisDrone official AP_small”.

## Frozen source

- detect validator SHA256: `25740E8C8C060949345AEA27DC45E514C98D7187E8AE1F700B435C6567C85D0E`;
- restored official CIoU metrics SHA256: `54FDF20B38D354FAFF0EA944E4775D8B374D6C8772B00E4B420BEE480B9E742F`;
- evaluator wrapper SHA256: `33037EC693615325B37C5A9C39B14EFDD8E71E57E2DB93D2F3DB7213DE48B922`.

`03_verify_evaluator.sh` checks import provenance, versions, area ranges, loss/CIoU numerics and gradients. Local source sanity passed except that pycocotools is absent locally; the authenticated server must install the frozen dependency and produce `CANONICAL_EVALUATOR=PASS` before training.
