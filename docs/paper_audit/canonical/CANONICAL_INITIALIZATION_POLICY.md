# Canonical Initialization Policy

## Decision

```text
INITIALIZATION_POLICY=SCRATCH_FOR_ALL_MODELS
PRETRAINED=false
SEED=0
```

B0/C0/C1/C2/C3 all use the same Ultralytics default random initialization algorithm, the same canonical source/runtime and seed 0. No pretrained checkpoint is loaded; transferred parameter count is exactly zero and missing/unexpected transfer keys are not applicable.

This avoids the confound in which official YOLO11n receives a nearly complete pretrained transfer while the structurally different SFD-Lite receives only a small partial transfer. The policy does not assert identical initial tensors across non-isomorphic architectures; it asserts an identical, deterministic initialization rule.

Every prelaunch manifest must record `pretrained_checkpoint=null`, `transferred_parameters=0`, source/model hashes and seed. Any later pretrained study is a separate experiment and cannot replace this v1 matrix silently.
