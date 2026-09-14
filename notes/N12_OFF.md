# N12 off-contract

Not a 2 Gvox/s claim. Not 3090 Ti. Always grade, never default on FAIL.

| ID | wall | VOI / note | task_legal |
|---|---|---|---|
| ε=0.5 T=0.3 | 461 ms | split 0.4345 merge **0.2712** | FAIL |
| ε=0.8 | 155 ms | 0.4323 / **0.3058** | FAIL |
| ε=1.0 | 143 ms | 0.4225 / **0.3760** | FAIL |
| ε=2.0 | 130 ms | 0.4490 / **0.8735** | FAIL |
| Kruskal frozen SDSL | 6092 ms vs ParHAC val 202 | split 2.94, 0 merges | FAIL |
| mutex hop (m16b) | 273 s (script); M16b wall 146 s | split 2.13 | FAIL (expected) |
| PRUF3D sm_120 | GPU **101.9 ms** (wall 3.75 s IO) vs WS val 224-344 ms | grayscale Meyer, not waterz fragments | n/a |
| RAMA | cmake: tree not on goblin |: | not built |
| cuSLINK | not executed; Kruskal is the frozen-MST analog |: | FAIL analog |
| hist-q GPU FIFO | skipped | N11 BinQueue dead |: |

No off-contract probe is a `segment_d` default.
