# Remaining problem (precise)

**Not a 2 Gvox/s claim. Not a 3090 Ti claim.** See [ATLAS.md](ATLAS.md).

## Precision gate

Construct a GPU agglomerator for the CREMI-A **contact-mean** RAG
(7 505 458 edges @ val / 90 323 139 @ official 2.16 Gvox) such that:

1. four-T VOI PASSes the shipped waterz gate at aff 0.2/0.3/0.4/0.5
   (`voi_split <= base+0.02` **and** `voi_merge <= base+0.02` at every T;
   grader: `baseline/run_baseline.py --candidate ...`), **and**
2. on an idle RTX 5090, CUDA-event timed `segment_d` agglomeration wall on
   official `make_big.py` `[3,375,2400,2400]` @ affinity 0.3 is
   **<= 1000 ms** (preferably <= 400 ms),

**or** prove a structural reason that (1+eps) matching + S3-contract on this
graph cannot go below ~1680 ms agg (N17/N18/N19 measured floor).

Completion is decidable: VOI table + stage JSON, or a theorem with stated
scope. "Make it faster" is not a problem.

## Current measured floor (not the solution)

Pin: `data/cache/N19_I0_REPRO.json`, [N19_SUMMARY.md](N19_SUMMARY.md).

| stage | ms @ 2.16 T=0.3 (5090) |
|---|---|
| WS | ~1309 |
| RAG | ~85 |
| agg | ~1680 |
| extract | ~18 |
| e2e | ~3093 (~0.70 Gvox/s) |

WS alone exceeds the TASK e2e budget (1080 ms). Optimistic zero of
`k_w5_compress_list` (508 ms) + hash/rebuild/rewrite (~838 ms) still leaves
~1745 ms e2e on the **faster** card ([ATLAS.md](ATLAS.md) section 3).

## Ruled-out attacks (do not reopen)

Pinned in `data/cache/n19_dead.jsonl` and [voi_atlas.csv](../data/cache/voi_atlas.csv):

| Attack | Why dead |
|---|---|
| eps bump above 0.40 on T=0.3 | N18 B2: all eps in {0.41..0.49} merge FAIL |
| eps >= 0.5 | N12_OFF: merge FAIL |
| parallel BinQueue | N18 B1: wall 2253 ms + VOI FAIL |
| serial BinQueue | N14 hang; N11 1-thread void |
| NNG / reciprocal-NN filter | N19 H2: split 2.15 |
| simplified Lu chunk freeze | N19 H4: parents != ParHAC |
| GASP Average / AbsMax / mutex | VOI giant or under-merge |
| RAMA signed multicut | 60-600 s timeout |
| Kruskal / SDSL / FH / SRM / Soille | VOI FAIL (atlas) |
| fuse_pack / dirty_unmark as closers | noise |
| Playne hook / privatized vcount micro-opt | owner <200 ms (N19 W1/W3) |
| grayscale PRUF as S1 | Meyer != waterz fragments |
| COMPACT_EVERY=8 | slower than N17 |

## Remaining honest attacks

1. **Work-efficient clustered-graph update** (ParHAC section 2.3 / StarMerge;
   SOURCES S32, S37). E6t was VOI-legal but slower and nondeterministic --
   fix det, then measure visits to wall.
2. **`k_w5_compress_list` algorithm** (508 ms owner). Not Playne hooks
   (84 ms). Needs four-T + identity plan before any claim.
3. **True Lu-Zlateski Algorithm 2** (SOURCES S26). N19 H4 was a simplified
   freeze, not the paper; only claim speed if val parents `array_equal`
   ParHAC.

## Cascade (why this matters)

- FlyWire-scale / production mean agglomeration that currently bottles on CPU waterz.
- Any GPU HAC paper that claims waterz-quality partitions on contact-mean RAGs.
- Connectomics pipelines that confuse frozen-weight MST / mutex with mean affinity.

## Do not

- N20 micro-opt on hook/vcount/sort
- Another eps grid
- Claim 2 Gvox/s from 5090 arithmetic
- Grind compress_list without four-T identity
- Publish "we almost hit a bounty" without the atlas
