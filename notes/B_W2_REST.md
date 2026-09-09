# B — remaining W2 levers, measured one at a time

Date: 2026-09-06. Not a 2 Gvox/s claim. Parked device path (after B_DEV_AFF).
Baseline tracked peak after aff-park: **2.195 GiB**.

| lever | nfrag | oracle | val peak | delta | 2.16 pred tracked | owns peak? |
|---|---|---|---|---|---|---|
| q32 (uint32 BFS q) | 2175400 | True | 2.195 GiB | 0 | — | no (BFS after peak) |
| vcount compact before sort | 2175400 | True | 2.023 GiB | −0.172 | 24.28 GiB | yes |
| e9c in-place scan | 2175400 | True | 2.023 GiB | 0 | — | no (e9c after peak) |
| WATERZ_WS_W3=1 | 2175400 | True | 2.023 GiB | 0 | — | no (e9c after peak) |

qtot unchanged at 63 943 344 after vcount compact (same queue sizing).

Phase ownership matches `w2_mem.py`: streaming aff and e9c/W3 do not hit
the divide-sort peak. uint32 q does not either. Only compacting vcount
before the radix sort moved `ws_mem_peak`.

Do not run 2.16 Gvox. Tracked 24.28 GiB is scratch only; fused still adds
seg (8.05) and not aff (parked). Fused pred ≈ 24.28 + 8.05 = 32.33 GiB
before RAG/AGG. Still over 23 usable.
