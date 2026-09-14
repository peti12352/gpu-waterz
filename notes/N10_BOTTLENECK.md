# N10: what we are actually failing, and what is not proved

Not a 2 Gvox/s number. Not a 3090 Ti number. No invented fill-ins.

N8/N9 treated "2.16 is linear, WS > 4 s after W5" as a stop. That is a
measurement of **this stack**, not a theorem that 2 Gvox/s is impossible.
Abboud ICALP 2024 / ParHAC Thm 1.2 say exact average-linkage is P-complete.
TASK allows approximate order if VOI holds. BinQueue MEAN already PASSed
T=0.3 (split 0.455129 / merge 0.241600). Hardness does not close the bounty.

## The gap (measured)

Target: 1080 ms e2e on a **3090 Ti** (1008 GB/s). Idle 5090 (1792 GB/s):

| | N8 algo-0 | N9 W5+face-clear |
|---|---|---|
| 2.16 e2e | 13518 ms / 0.160 Gvox/s | 9213 ms / 0.234 Gvox/s |
| WS | 10832 | **6536** |
| RAG | 433 | 424 |
| agg ε=0.40 | 2224 | 2222 |

Need ~15x on the 5090 to leave room for the slower graded card
(1080 x 1792/1008 ≈ 603 ms). Linear scaling val->2.16 is 12x voxels / 12x
fragments / 12x edges. That is make_big 3x2x2, not a law of watershed.

## Where N9 aimed at the wrong owner

N9 Type D timed W5 kernels **2117 ms** and BFS **14 ms** of WS 6536 ms.
The leftover **~4400 ms** was never split. We then killed E4 / path-halving /
Playne because "A/B/C cannot close 12x". Those tracks only attack the 2117 ms
slice. The 4400 ms is plateau **grouping**, not UF and not BFS.

e9b after tile-local UF (one 720 Mvox slab, nC = 244 142 296):

1. Pageable D2H of 244 M corner indices (~976 MiB).
2. `vcount` over 720 M voxels.
3. **~233** `k_gather_vcount` + 4 MiB D2H copies (`CHUNK = 1<<20`).
4. Device radix sort of 244 M keys.
5. Pageable H2D of corners and vcounts, then `k_gather_u32`.
6. Exclusive scans, then BFS **4.7 ms**.

The BFS that the sort exists for is 4.7 ms. The infrastructure around it is
the unmeasured majority. That is a Type E miss: we optimized the 5 ms kernel
and the 289+414 ms UF, and parked 244 M integers through the CPU to save
~1 GiB of a 32 GiB card.

`segment_d` also D2H/H2D the **6.03 GiB** affinity inside the timed window
(`WATERZ_AFF_PARK` default on). TASK speed is "affinity already in VRAM";
that copy is not free and is not required by the listing.

## Bottlenecks we have not overcome (nuance)

**1. Host park is a fit hack sitting in the speed path.**
B_FIT parked corners so fused 2.16 pred stayed under 24 GB. We then slabbed
(Z=125), which already fits. The park stayed. Pageable memcpy + 233 syncs
per slab x 3 slabs is O(nC) latency, not bandwidth.

**2. Two full-volume UFs still cost 2117 ms.**
W5 list is Chen's face domain (0.58 / 0.40 of a slab). E4 (unique `p1` roots)
is still legal and untried. It cannot be the closer alone; it is a real
contraction of that 2.1 s.

**3. Agglomeration is 22.2 M merges on 2.16, 2222 ms.**
ParHAC ε=0.40 is the speed path and is already linear in tiles. Exact MEAN
BinQueue is VOI-legal at T=0.3 and does the same 1.85 M / 22.2 M merge count.
We never built a GPU FIFO-bin. Serial 22 M pops at 30 ns is 660 ms: inside
the 1080 ms budget if WS and RAG shrink. X1 (union-all-in-band) failed VOI;
the FIFO-inside-bin is the one that passed.

**4. RAG is 424 ms and was left alone.**
3-dir contact atomic-hash. 12x val 43 ms. Need ~50 ms on 2.16. Not hash-table
PDFs (dirty multiplicity 1.07). A streaming sort+reduce of faces may be.

**5. Memory forces z-slabs, not 12 serial tiles.**
Fused WS pred 21 GiB + aff 6 + labels 8.6 does not fit 24 GB. Three Z=125
slabs do. 8-tile *serial* was 8.3x and is dead. Parallel tiles on one GPU
need one tile's scratch: that is the slab we already run. The win is making
one 720 Mvox slab cheap, not cutting it into val crops.

**6. Accuracy locks S1-equivalent fragments.**
Mutex / frozen-CC / hist-q / X1 failed VOI. Plateau BFS cannot be dropped
(G2 extra closed-plateau CCs). The grouping around BFS can.

## What would actually close it (order)

1. **Measure the 4400 ms** (`n10_ws_split.py`): park / vcount-chunk / sort /
   unpark / aff D2H. Numbers, not a model.
2. **`WATERZ_HOST_PARK=0`**: corners + vcount stay on device. One gather
   kernel. Identity vs `wz_fragments.npy`. Slab peak +~2 GiB; 5090 has it.
   3090 Ti: re-check 24 GB after the time drop.
3. **`WATERZ_AFF_PARK=0` on the speed path.** Aff stays in VRAM. Listing
   already assumed that.
4. **E4 unique-root stitch** on the remaining UF 2.1 s. Val identity first.
5. **GPU MEAN BinQueue** (stock FIFO, N=256). CPU PASS is the licence.
   Kill if visits do not become wall (N7).
6. **RAG sort+reduce** only after 1-5; 424 ms is not the first 15x.

Honest composition on the 5090 if 1-5 all pay: WS ~1 s, RAG ~0.1 s,
agg ~0.5 s -> ~1.6 s / 1.3 Gvox/s. Still short of 2.0 on a slower card.
That is a remaining factor, not a proof of impossibility. The next proof
is the phase split, not another paper.

## Val measurement (idle 5090, W5, identity True)

`n10_ws_split.json`. Median of 3 `run_split` (WS only, not e2e).

| | host park on | host park off |
|---|---|---|
| WS | 597.67 ms | **344.25 ms** |
| park D2H | 109.87 | 0.00 |
| vcount (+ chunks) | 113.59 | 15.35 |
| radix sort | 2.91 | 2.90 |
| unpark H2D+gather | 44.03 | 2.97 |
| exclusive scan | 0.53 | 0.53 |
| BFS | 1.18 | 1.19 |

**1.74x** on val WS. The sort that we built DoubleBuffer for is 3 ms. The
park path is 268 ms of pageable copies. 12x that tax is the 2.16 leftover.

Default `WATERZ_UF_ALGO` stays 0 until a track passes identity and its
speed gate. `WATERZ_HOST_PARK=0` is identity-true on val; 2.16 retime
follows.

## 2.16 retime (idle 5090, W5, HOST_PARK=0, AFF_PARK=0)

Same `n8_run216.py` path as N9. nfrag 26104800, nlab 3860788.

| | N9 (park on) | this run |
|---|---|---|
| e2e | 9213 ms / 0.234 Gvox/s | **4918 ms / 0.439 Gvox/s** |
| WS | 6536 | **2577** |
| RAG | 424 | **83** |
| agg | 2222 | 2238 |

RAG 424 was mostly the 6 GiB aff H2D after park (that copy sat in
`STAGE_MS["rag"]`). Real RAG is 83 ms. WS 6536 was UF 2117 + ~4 s of
pageable corner/vcount/aff traffic. With parks off, e9b phases on three
slabs sum to vcount 163 + sort 34 + unpark 33 + scan 5 + BFS 14. W5
kernels still ~2117 ms. Agg is now the e2e owner (2238 / 4918).

Not 2 Gvox/s. Not 3090 Ti. The N9 CUDA-stop is void: we were timing PCIe.
