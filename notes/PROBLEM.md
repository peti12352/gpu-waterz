# Open questions

See [ATLAS.md](ATLAS.md) for the measured campaign. This file states the
remaining research problems without a hardware-grade narrative.

## Problem A -- faster contact-mean agglomeration under VOI

On the CREMI-A contact-mean RAG (7.5M edges @ val, ~90M @ 2.16 Gvox), build a
GPU agglomerator such that:

1. four-threshold VOI still PASSes the shipped waterz gate (aff 0.2/0.3/0.4/0.5,
   both halves within +0.02), and
2. agglomeration wall on the 2.16 Gvox volume at aff 0.3 drops well below the
   current ~1680 ms floor (target of interest: <=1000 ms; ambitious: <=400 ms),

**or** give a structural argument that (1+eps) matching + S3-contract on this
graph cannot go much below that floor.

Current stage breakdown (idle RTX 5090, pin `data/cache/N19_I0_REPRO.json`):

| stage | ms |
|---|---|
| watershed | ~1309 |
| RAG | ~85 |
| agglomeration | ~1680 |
| extract | ~18 |
| end-to-end | ~3093 |

## Problem B -- watershed list-compress

`k_w5_compress_list` is ~508 ms of the watershed stage. Improving it requires
preserving S1 plateau/basin semantics (four-T VOI + identity diagnostic).
Hook / vcount micro-opts below ~200 ms of e2e were not worth the iteration
cost (N19 W1/W3).

## Ruled out (do not reopen without new evidence)

| Attack | Outcome |
|---|---|
| eps > 0.40 on aff-0.3 path | merge VOI FAIL (N18 B2 grid) |
| eps >= 0.5 | merge VOI FAIL |
| parallel / serial BinQueue as GPU closer | hang, slow, or VOI FAIL |
| NNG / reciprocal-NN edge filter | split VOI ~2.15 |
| simplified chunk freeze (not paper Lu Alg 2) | parents != ParHAC |
| mutex / GASP AbsMax / Average / Kruskal / FH / SRM / Soille / RAMA | VOI FAIL or timeout (atlas) |
| fuse_pack / dirty_unmark as "closers" | noise |
| grayscale intensity PRUF as waterz S1 | wrong fragment class |

## Plausible next attacks

1. Work-efficient clustered-graph update (ParHAC section 2.3 / StarMerge).
   Earlier E6t was VOI-capable but slower and nondeterministic.
2. True Lu-Zlateski Algorithm 2 boundary freeze, with parent identity vs
   ParHAC before any speed claim.
3. Algorithmic redesign of list compress under fixed S1 semantics.

## Why this matters beyond one benchmark

- Production mean agglomeration on large EM volumes still leans on CPU waterz.
- GPU HAC claims that assume frozen weights do not transfer to contact-mean.
- A clean VOI atlas of failures is reusable when comparing new linkages.
