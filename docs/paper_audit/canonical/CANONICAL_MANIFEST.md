# SFD-Lite Canonical v1 Manifest

Date: 2026-08-15  
Identity: `SFD-Lite-Canonical-v1`

## Repository state

- Base commit: `0798253837f1ed456cde6e924fba36b2f9394aec`
- Branch at package creation: `feature/sfd-lite-prototype`
- Outer worktree: **DIRTY**. Existing user changes and numerous untracked historical files were not modified, deleted, or silently promoted to canonical identity.
- Canonical identity is defined by `CANONICAL_FILE_MANIFEST.json`, which hashes every file under the three canonical namespaces. Git remote URLs are deliberately excluded because repository configuration may contain credentials.

## Local construction environment

- OS: Windows 11 (`10.0.26200`)
- Python: 3.12.13
- PyTorch: 2.13.0+cpu
- CUDA: not available in the local construction environment; server build must be recorded by gate 00.
- Ultralytics source version: 8.4.60
- ultralytics-thop: 2.0.20
- numpy 2.4.4, PyYAML 6.0.3, Pillow 12.2.0 and ultralytics-thop 2.0.20.
- pycocotools 2.0.10 is frozen but not installed locally; gate 00 checks it on the execution server.

## Frozen high-value hashes

| Object | SHA256 |
|---|---|
| custom modules `sfd_lite.py` | `0863771AE6E301D9593FE97207B913C1AABB016D636CFB166BD5904B6F056847` |
| parser `nn/tasks.py` | `73F8D45E7DDB9855EFD7A171A3931AB5B15893E538A7080D945573F48075A9C4` |
| module exports | `420420BD47A1183D098B45D4EC9E06C1CE1AD8351D60E579AC7ADA8A561EEB5C` |
| canonical loss | `52DB69F9B3B00C9A06B3F2038739D0CED3F0426BE0AA0F564B24AB6DC3335397` |
| restored metrics/CIoU | `54FDF20B38D354FAFF0EA944E4775D8B374D6C8772B00E4B420BEE480B9E742F` |
| detect validator | `25740E8C8C060949345AEA27DC45E514C98D7187E8AE1F700B435C6567C85D0E` |
| data converter/auditor | `C5F04C1FAE263A114D7B49249E4E589E0651A04422E31664B79F044A64E73416` |
| evaluator wrapper | `33037EC693615325B37C5A9C39B14EFDD8E71E57E2DB93D2F3DB7213DE48B922` |
| training config | `3BF8BB6A4B401513E3A5492DE902F4674F6F6575903048148BE0462E75434A11` |
| experiment matrix | `F6A8B57836519DFB0535B7AEFF314285FF6AA92CE1718114715218FC8417BD2D` |

The complete machine-readable file list and hashes supersede this abbreviated table.

## Evidence boundary

Historical F4/F0/SFD-X/SFD-YOLO/UAVDT run04 are exploratory references only. Quantitative final-paper claims require runs produced by this package after gates 00–05 pass.
