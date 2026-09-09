# B — CUB SortPairs tmp: in/out vs DoubleBuffer

Date: 2026-09-06. Not a 2 Gvox/s claim. No 2.16 allocation.
Idle 5090, parked val. `data/cache/b_sort_peak.json`.

Behavior unchanged: dry DoubleBuffer query only. nfrag 2175400,
`wz_fragments.npy` array_equal=True.

| quantity | bytes | GiB |
|---|---|---|
| nC | 61 035 574 | 0.227 / uint32 |
| in/out `tmp_bytes` | 495 366 143 | **0.461** |
| DoubleBuffer `tmp_bytes` | 7 081 471 | **0.007** |
| extra N (in/out − dbl) | 488 284 672 | 0.455 = 2 × nC × 4 |
| tracked peak | 2 172 426 888 | **2.023** |

`ws_peak_by_line` at the peak: flag 720e6 + vcount 720e6 + three nC arrays
(corners_in, keys_in, vc_in). That snapshot is **before** sort tmp.

WSMEM: `divide/pre-sort` cur=1.137, `divide/sort-tmp` cur=1.598, peak stayed
2.023. Sort + in/out tmp is **below** the flag+vcount+3 nC overlap.

DoubleBuffer alone therefore does **not** move the val peak. It would drop
tmp 0.461 → 0.007 after the overlap is gone. `dbl_only_fits=false`:
fused-dbl-only pred 28.88 GiB.

Lookback/bins really are ~7 MB at this nC (docs `O(P)` as “not N-payload”,
not a SM count). Hemstad CCCL #1148: do not treat the big-O as a budget;
these are nullptr-query bytes on this CUB + 5090.
