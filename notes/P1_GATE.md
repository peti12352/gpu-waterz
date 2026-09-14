# P1 gate: independence holds, speed path does not reopen

Date: 2026-09-06. Contract: TASK.md. Not a 2 Gvox/s claim.

## P1 PASS (official `make_big.mirror`, 2-tile, all three axes)

`data/cache/p1_make_big_indep.json`

GPU val = CPU `wz_fragments.npy` byte-identical, nfrag 2175400 / bg 506568.
For z, y, and x:

- join-channel seam plane identically 0
- no fragment id spans the seam
- mirrored-tile nfrag = val; concat nfrag = 2 x val
- concat tile0 == val and tile1 == mir, up to id remap
- every cross-seam face has aff 0 -> T=0.3 cannot merge across tiles

The official 3x2x2 volume is 12 merge-independent val-sized problems.
That is not a 12x throughput claim.

## P1b FAIL for 1x-val + stamp

`data/cache/p1b_flip_equiv.json`

`np.flip(WS(val))` is not `WS(official mirror(aff))`. Size histograms differ
(a flip cannot change region sizes). Disagreement is volume-wide, not a halo:

| axis | disagree voxels | interior frac | planes touched |
|---|---|---|---|
| z | 9 897 982 | 5.5% | 125/125 |
| y | 26 921 111 | 15.0% | 1200/1200 |
| x | 12 573 666 | 7.0% | 1200/1200 |

Halo repair is dead. make_big 3x2x2 uses all 8 flip triples (z=0 and z=2
share a pattern), so a correct tile algorithm is **8 unique val pipelines**,
not 1.

## Cost (P1-gate)

`data/cache/p1_gate.json`

- 8 x val G15 e2e 958.70 ms = **7669 ms** on the 5090
- bandwidth-scaled to 3090 Ti: **13635 ms** (12.6x the 1080 ms budget)
- fused honest stack remains **1642 ms / 1.52x**
- 8-tile serial is **8.3x slower** than fused

Do not start W2 / reflect / stamp CUDA. Keep N6.

## P2: true HistogramQuantile

`data/cache/p2_true_quantile.json`

Existing `q20_quantile` labels, official `run_baseline.py` on greengoblin:
four-T **ACCURACY GATE PASS**. N1's frozen no-hist quantile was the X0 giant;
real face hists are VOI-legal. Serial BinQueue. Not a fused-agg 1.96x cut.
Class closed.
