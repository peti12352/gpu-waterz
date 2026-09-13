# Open questions

Plain-language status (best timed stack, what is still open, what is closed
and why): [WHERE_WE_ARE.md](WHERE_WE_ARE.md).

See [ATLAS.md](ATLAS.md), [N20_WIN.md](N20_WIN.md), [N21_WIN.md](N21_WIN.md),
and [N22_WIN.md](N22_WIN.md) for the measured campaign. This file states what
is still open. Not a 2 Gvox/s number; not 3090 Ti.

## Problem A -- faster contact-mean agglomeration under VOI

On the CREMI-A contact-mean RAG (7.5M edges @ val, ~90M @ 2.16 Gvox), build a
GPU agglomerator such that:

1. four-threshold VOI still PASSes the shipped waterz gate (aff 0.2/0.3/0.4/0.5,
   both halves within +0.02), and
2. agglomeration wall on the 2.16 Gvox volume at aff 0.3 drops well below the
   current ~1680 ms floor (target of interest: <=1000 ms; ambitious: <=400 ms),

**or** give a structural argument that (1+eps) matching + S3-contract on this
graph cannot go much below that floor.

N20 closed the "new HAC class" branch of (2). Remaining agg win is engineering
of the existing ParHAC matching + S3 contract. Unique-kernel floors from
`data/cache/N21_D0_OWNERS.json` (do not add NVTX `:hash_rewrite` 286.9 on top
of `k_rewrite_dirty_fuse` 273.0 — they overlap):

- rebuild+fuse 551.4 ms vanish -> leftover agg **~1128 ms**
- plus insert+emit (four-kernel 886.1) vanish -> leftover **~794 ms**
- plus pack_amask -> leftover **~694 ms**

The old ~840 ms leftover mixed NVTX with the fuse kernel and is invalid.
N21 listed insert/rebuild (A1/A2/A4) identity+four PASS on val; cold cuts
35/51/54 ms vs D0 456.8 ms; none cleared the 100 ms 2.16 gate. N22 occupied
emit (A3/A5) identity+four PASS; 18 ms solo / 62 ms stacked with listed;
insert-time slot atomicAdd ate the ntab-scan save. Rewrite remains
O(nscan). Exact / RNN / StarMerge / complete / WPGMA / spatial Lu
Alg 2 are not GPU closers (see Ruled out).

Current stage breakdown (idle RTX 5090, pin `data/cache/N19_I0_REPRO.json`):

| stage | ms |
|---|---|
| watershed | ~1309 |
| RAG | ~85 |
| agglomeration | ~1680 |
| extract | ~18 |
| end-to-end | ~3093 |

Dual-eps: 0.08 four-T, 0.40 single T=0.3. eps in (0.41, 0.49) and >=0.5 dead.

## Problem B -- watershed list-compress

`k_w5_compress_list` is ~508 ms of the watershed stage. Improving it requires
preserving S1 plateau/basin semantics (four-T VOI + identity diagnostic).
Hook / vcount micro-opts below ~200 ms of e2e were not worth the iteration
cost (N19 W1/W3). N21 D0: nlist 71-105M, hop_max=8-9. List-only J=4 was
-6 ms on val WS (N21_W2). Compress-subset no-go (N21_W1). Freeze WS
(N21_W3). Optimistic WS after compress_list->0 is still ~800 ms, which
already does not reach a 1080 ms e2e budget by itself.

## Ruled out (do not reopen without new evidence)

| Attack | Outcome | pin |
|---|---|---|
| eps > 0.40 on aff-0.3 path | merge VOI FAIL | N18 B2 |
| eps >= 0.5 | merge VOI FAIL | atlas |
| parallel / serial BinQueue as GPU closer | hang, slow, or VOI FAIL | n19_dead |
| NNG / reciprocal-NN edge filter | split VOI ~2.15 | N19_H2 |
| simplified chunk freeze (not paper Lu Alg 2) | parents != ParHAC | N19_H4 |
| mutex / GASP AbsMax / Average / Kruskal / FH / SRM / Soille / RAMA | VOI FAIL or timeout | atlas |
| fuse_pack / dirty_unmark as "closers" | noise | N19 |
| grayscale intensity PRUF as waterz S1 | wrong fragment class | N19_X1 |
| StarMerge clustered-graph rewrite | EV-dead: layer-0 is 71.3% of merges (1.32M/1.85M over 64 outers) at eps=0.08; eps=0.40 denser | N20_D1, N20_STAR |
| GPU NN-chain / RNN-HAC | dendrogram height 12539; RNN hit cap 5000 all T; four-T FAIL; parents != heap | N20_D2, N20_D3, N20_RNN |
| complete-link (min-of-means) | four-T under-merge (split 1.30-1.48) | N20_X4 |
| WPGMA 1/2+1/2 | four-T under-merge (split 0.77-1.13) | N20_X5 |
| spatial Lu Alg 2 (paper B^C) | four-T PASS, parents != heap, residual ~3.86M of 7.5M edges, 6645 s CPU, no_speed_path | N20_LU2 |
| Ward / centroid / Chamfer / cuML / GPU-UPGMA / GSHAC / cardinality UPGMA | wrong statistic or serial/dense | N20_REFUSE, papers/SOURCES.md S43 |

| listed insert/rebuild (holes ∪ alist) | identity+four PASS; val cuts 35-54 ms; no 2.16 | N21_A1, N21_A2, N21_A4 |
| compress-subset / list-only J-jump | D0 no-go / val WS -6 ms; fat list structural | N21_W1, N21_W2, N21_W3 |
| occupied-slot hash emit | identity+four PASS; val 18 ms / stack 62 ms; no 2.16 | N22_A3, N22_A5 |

Do not reopen IDs in `data/cache/n19_dead.jsonl`. N20 stamps live in
`data/cache/n20_dead.jsonl`. N21 stamps live in `data/cache/n21_dead.jsonl`.
N22 stamps live in `data/cache/n22_dead.jsonl`.
Do not launch `csrc/n20_rnn_gpu_prep.cu`.
Product default stays E6s. Do not CONT `AGG_E6t`. Do not default
`WATERZ_LISTED_INSERT` / `WATERZ_LISTED_REBUILD` / `WATERZ_LIST_JUMP` /
`WATERZ_SLOT_EMIT`.

## Remaining (structural)

1. `k_rewrite_dirty_fuse` is still dense nscan (273 ms @ 2.16). Holes are
   its output. First inner ndirty 5.6M vs nact 2.8M (half below-TL live
   dirty, not on alist). No dirty-edge CSR on the legal path; `adj_off`
   is E6t StarMerge-only (do not launch). Stamp `N22_REWRITE_NEEDS_CSR`.
2. `k_hash_emit_holes` stays O(ntab) on the product path. Occupied-slot
   emit (`WATERZ_SLOT_EMIT`) is parent-identical but 18 ms val / 62 ms
   stacked (N22_A3/A5); CUB DeviceSelect over ntab is still O(ntab).
   `k_copy_sz_list` 114.5 ms is already listed over roots and required
   for freeze eps (`N22_COPY_SZ_NECESSARY`).
3. `k_w5_compress_list` 508 ms is a 71-105M-entry list, hop_max=9. Hook list
   must keep faces+phase-1 roots. Freeze WS (N21_W3).

There is no unused GPU mean-HAC class left to port (S43). GPU prep for RNN
and StarMerge must stay unlaunched (`N20_GPU_PREP.json`: `ran_gpu=false`).
Listed / slot-emit kernels stay env-gated. Cite [N21_WIN.md](N21_WIN.md)
and [N22_WIN.md](N22_WIN.md).

## Why this matters beyond one benchmark

- Production mean agglomeration on large EM volumes still leans on CPU waterz.
- GPU HAC claims that assume frozen weights, Ward in R^k, or small-merge
  StarMerge do not transfer to this contact-mean RAG.
- A clean VOI atlas of failures is reusable when comparing new linkages.
