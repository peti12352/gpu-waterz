# Lab notes

Library docs: [README](../README.md), [usage](../docs/usage.md),
[decode](../docs/decode.md), [porting](../docs/porting.md),
[citations](../docs/citations.md).

Leftover work and closed attacks.

## Pin (idle RTX 5090)

`data/cache/N19_I0_REPRO.json`, parks off, `[3,375,2400,2400]` aff 0.3:

| stage | ms |
|---|---|
| watershed | 1309.3 |
| RAG | 85.4 |
| agglomeration | 1679.9 |
| extract | 18.3 |
| end-to-end | 3093.4 (~0.70 Gvox/s) |

Four-T PASS. Two full runs, identical labels. Gold: nfrag=2175400,
bg=506568. T=0.3 VOI split 0.440823 / merge 0.254275, inner=538,
merges=1853545, nseg=321855.

Dual-eps: **0.08** four-T, **0.40** single T=0.3. eps in (0.41, 0.49)
and >=0.5 fail merge VOI at 0.3.

```
bash scripts/build_cuda.sh
bash scripts/eval.sh          # four-T + identity on CREMI-A val
bash scripts/eval.sh --216    # plus 2.16 timing if the volume exists
```

Product default is paper-style ParHAC (E6s). Listed / slot-emit / CSR
rewrite flags stay off (CSR is identity-true and slower).

Affinity thresholds are **affinity** (merge while contact-mean > T).
Stock waterz heaps on **score** `1 - affinity`:

| affinity (this API) | waterz score |
|---|---|
| 0.2 | 0.8 |
| 0.3 | 0.7 |
| 0.4 | 0.6 |
| 0.5 | 0.5 |

## Remaining

Same partition quality in less wall time, or a written argument that
this watershed plus contact-mean ParHAC cannot get much below the pin.

1. **A coalesced dirty-edge rewrite, not a linked-list CSR.** Dense
   `k_rewrite_dirty_fuse` is still 273 ms on 2.16. Contact-mean incidence
   lists (`head`/`nxt`, splice, listed fuse) match the CPU scan every
   inner, then lose on the GPU: identity+four PASS, 2.16 agg **23240 ms**,
   `k_csr_gather` pointer-chase. Rebuilding CSR from full nscan each inner
   is the slow path.
2. **A written ceiling.** If the heavy agg kernels vanished, about
   690-1130 ms of agglomeration would still remain, plus hundreds of ms
   of watershed. Optional A5 (`LISTED_INSERT`+`LISTED_REBUILD`+`SLOT_EMIT`)
   is identity+four PASS and a **56 ms** 2.16 cut (1622-1624 vs 1679.9);
   flags stay off. That is still the 1600 ms class.
3. **Another GPU.** Unmeasured. [porting.md](../docs/porting.md).

Unique leftover owners on the pin: `k_w5_compress_list` 508,
`k_rebuild_active` 278, `k_rewrite_dirty_fuse` 273, `k_hash_insert` 174,
`k_hash_emit_holes` 161. Compress-list is 71-105 million entries;
dropping plateau faces changes fragment identity.

## Closed (do not reopen)

Every kill used four-T VOI, parent-identity against the current ParHAC
stack, or a structural count that makes a GPU port pointless. Timing vs
1679.9 ms only on an idle card.

| Attempt | Evidence |
|---|---|
| A different merge rule (Kruskal, mutex, GASP, NNG, complete-link, WPGMA, Ward, ...) | Different partition; VOI fails or the statistic is not contact-mean |
| Exact mean heap / RNN / NN-chain as a 1680 ms closer | Dendrogram height 12,539; RNN hit a 5,000-round cap on every threshold; parents != heap; four-T fail |
| StarMerge / E6t clustered rewrite | First layer is 71% of all merges (1.32M / 1.85M) at eps=0.08; denser at 0.40. Small-merge assumption is false here |
| Spatial Lu Alg 2 | Four-T passes, parents != heap, ~3.86M of 7.5M edges left, 6645 s on CPU |
| Looser merge allowance (eps 0.41-0.49 or >=0.5) | Merge VOI fails on affinity 0.3 |
| Bin-queue agglomeration | Hang, or 2253 ms and VOI fail |
| Playne / list-halving / jump-flatten / mapped-host atomics / grayscale PRUF as S1 | Wrong basins, ~1e6 ms, or a different watershed |
| Compact more / insert-only / fuse-pack / cuda graphs / sticky size | Noise, illegal identity, or slower on 2.16 |
| Listed insert / rebuild as a 2.16 closer | Parents match; four-T pass; 35 / 51 / 54 ms on val |
| Occupied-slot hash emit as a 2.16 closer | Solo 18 ms val / stack 62 ms val; stacked A5 does 2.16 at 1622-1624; keep_default false |
| Contact-mean linked-list CSR rewrite | CPU splice equals scan; GPU identity+four PASS; 2.16 agg 23240 ms |
| Skip empty hash slots with CUB DeviceSelect | Still reads the whole table |
| Compress only union-find roots / jump pointers in the watershed list | Roots are 4-55% of the list; jump-4 was 6 ms slower on val WS |
| Grind `k_copy_sz_list` | Already walks only live roots; sticky sz0 fails VOI |

Mutex/Kruskal/RNN/StarMerge/Lu freeze and the rest of the VOI atlas:
[data/cache/voi_atlas.csv](../data/cache/voi_atlas.csv).

## CPU replicas

No GPU required:

```
uv run python scripts/e1_synthetic.py     # tiny S1-S4 asserts
uv run python scripts/w0_ws_ref.py --sweep
uv run python scripts/g0_agg_ref.py --help  # CPU ParHAC; needs a cached RAG
uv run python scripts/nvcheck.py            # clang -fsyntax-only on the .cu files
```

Vendored waterz: `funkey/waterz` `a0184d2` at `src/waterz-upstream/`.
