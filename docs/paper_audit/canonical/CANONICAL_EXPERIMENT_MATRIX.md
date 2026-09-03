# Canonical Five-Model Matrix

First stage uses seed 0 only.

| ID | Model | Regression | Single research question |
|---|---|---|---|
| B0 | official YOLO11n (`nc=10`) | official CIoU | complete canonical method versus fair baseline |
| C0 | complete SFD-Lite-Canonical | DyNWD | final candidate anchor |
| C1 | C0 with layer-1 SA-SPD-Lite → 3×3 s2 Conv-BN-SiLU | DyNWD | task-level contribution of the complete SA-SPD-Lite module |
| C2 | architecture structurally identical to C0 | official CIoU | independent regression contribution of DyNWD |
| C3 | C0 with layer-23 SF-DCA-Lite → `nn.Identity` | DyNWD | independent task-level contribution of SF-DCA-Lite |

Static checks prove C1 differs from C0 only at `backbone[1]` module identity and preserves `[1,16,320,320]`; C2 YAML is structurally identical to C0; C3 differs only at layer 23 and preserves `[1,56,80,80]`. Historical F0 design logic/results are not reused.

| ID | Params | Trainable | THOP GMACs | Torch-profiler GFLOPs |
|---|---:|---:|---:|---:|
| B0 | 2,591,790 | 2,591,774 | 3.225139200 | 6.383440000 |
| C0 | 687,572 | 687,564 | 3.303546188 | 6.313165990 |
| C1 | 688,528 | 688,520 | 3.398573388 | 6.509773990 |
| C2 | 687,572 | 687,564 | 3.303546188 | 6.313165990 |
| C3 | 685,108 | 685,100 | 3.290553600 | 6.289420800 |

C1 cannot separately prove attention, space-to-depth or synergy. C3 tests the complete SF-DCA-Lite block against identity, not individual internal mechanisms. Single-seed differences cannot support “stable” or “significant”.
