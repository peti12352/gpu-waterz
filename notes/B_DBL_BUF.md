# B — DoubleBuffer + one-sort gather + park

Date: 2026-09-06. Not a 2 Gvox/s claim. No 2.16 allocation.
Idle 5090, parked val. `data/cache/b_dbl_buf.json`.

nfrag 2175400, `wz_fragments.npy` array_equal=True, leaked=0.

| | val tracked | 2.16 fused pred |
|---|---|---|
| before (B_SORT_PEAK) | 2.023 GiB | 34.34 |
| DoubleBuffer + gather + park | **0.916 GiB** | **21.05** |

`fits: true` vs 23 usable. Binding stage is still WS, but under the cap.
Sort peak is 4 × nC + DoubleBuffer tmp (0.007 GiB). `tmp_inout` still
0.461 GiB if queried; we no longer allocate it.

What landed (identity-gated on val):

- CUB DoubleBuffer `SortPairs` on keys+indices; consume `.Current()`
- one sort; host permute of parked corners/vc (no second SortPairs)
- vcount delayed until after UF; `k_count_v2` does not need the flag array
- flag freed before keys; corners/vc host-parked across count+sort
- chunked D2H of vc so full nC vc never overlaps nvox vcount
- in-place run-start scan; no `plat_root` / `plat_nseed` (BFS derives both)
- BFS cap from `qoff` + `qtot`; `qsz` freed before the queue

bits_d was not parked. Not required after the lifetime cuts.
