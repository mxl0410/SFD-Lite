# Canonical Regression Loss Protocol

## LOSS-A: official CIoU

`1-CIoU` using the restored official geometric `bbox_iou(..., CIoU=True)`. It is used by B0 and C2.

## LOSS-B: DyNWD

Used by C0/C1/C3. For aligned xyxy boxes:

- compute CIoU;
- `q = clip((detach(CIoU)+1)/2, 0, 1)`;
- `omega = (1-q)^2`;
- NWD normalization constant `tau=8`;
- loss=`(1-omega)(1-CIoU)+omega(1-NWD)`.

It is quality-aware only. It does not read object-size bins, epoch, training stage or schedule; therefore “scale-aware”, “epoch-aware” and “training-stage-aware” are prohibited descriptions.

## Critical source correction

The historical source also injected NWD into `metrics.bbox_iou(CIoU=True)`, which affected TaskAlignedAssigner and made a loss-only comparison impossible. Canonical v1 restores official CIoU in `metrics.py` and confines DyNWD to `BboxLoss`. Both modes live in the same loss source and are selected only by the explicit recorded variable `SFD_CANONICAL_REGRESSION=ciou|dynwd`.

Local unit tests passed identical-box, non-overlap, finite-value, numerical CIoU equivalence, branch difference and gradient checks. Canonical loss SHA256 is `52DB69F9B3B00C9A06B3F2038739D0CED3F0426BE0AA0F564B24AB6DC3335397`.
