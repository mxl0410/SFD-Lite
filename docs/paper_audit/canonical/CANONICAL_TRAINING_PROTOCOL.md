# Canonical Training Protocol

The only configuration source is `experiments/sfd_lite_canonical/configs/canonical_training.yaml`. Per-run shell scripts may set only experiment ID, verified data root, new run root and device.

## Frozen core

- imgsz=640; epochs=200; patience=50; batch=16;
- AdamW, lr0=0.0002, lrf=0.01, weight_decay=0.0005;
- warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1;
- seed=0, deterministic=true, AMP=false, workers=8;
- pretrained=false, resume=false, cache=false;
- cos_lr=false, close_mosaic=10, nbs=64;
- box=7.5, cls=0.5, dfl=1.5;
- HSV=0.015/0.7/0.4; translate=0.1; scale=0.5; fliplr=0.5;
- degrees/shear/perspective/flipud/mixup/cutmix/copy_paste=0;
- mosaic=1.0, val every epoch, best checkpoint selected by the same Ultralytics fitness rule;
- final canonical evaluation is run once on each best.pt with the frozen evaluator.

The launcher creates the run directory itself, refuses any existing path, writes a prelaunch manifest, and requires all package/environment/data/evaluator/complexity/smoke receipts to be PASS.
