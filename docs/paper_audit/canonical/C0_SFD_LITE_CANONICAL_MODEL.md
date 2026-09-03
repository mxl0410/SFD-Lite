# C0 SFD-Lite-Canonical Model

C0 replaces the historical “final F4” identity. It is a candidate final model until canonical training and validation finish.

## Static identity

- YAML: `experiments/sfd_lite_canonical/models/C0_sfd_lite_canonical.yaml`
- SHA256: `05E54B7473683B777454CCDEBDD47A6611982F64A6C14A9C065D0BBC41221ACC`
- Params: 687,572; Trainable Params: 687,564
- reg_max: 8
- Detect inputs: layers 19/23/26; strides P2/P3/P4 = 4/8/16
- layer 1: SA-SPD-Lite
- layer 3: original SA-SPD
- layer 23: one SF-DCA-Lite at P3, reduction=4, runtime channels 56 and hidden channels 14
- P5 participates in fusion but is not a Detect output.

## Canonical complexity

Unfused FP32, input `1×3×640×640`, THOP 2.0.20 and Torch 2.13 profiler:

- THOP direct: 3.303546188 GMACs;
- Torch-profiler direct: 6.313165990 GFLOPs.

The obsolete 6.6091744 value is forbidden. Params/MACs/FLOPs do not establish speed, real-time performance or deployment efficiency.
