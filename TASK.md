# TASK.md: GPU waterz bounty

Library docs: [README.md](README.md). This file is the original listing.

**This file is the contract.** Gates, numbers, API, dataset, and out-of-scope
are copied from the bounty.tech listing as received 2026-08-30. Do not
paraphrase this file. `PLAN.md` may interpret; it may not relax a number
here. If PLAN and TASK disagree, TASK wins.

Source: bounty.tech listing
`GPU watershed + mean-affinity agglomeration matching waterz segmentation quality`
400 krajcár escrowed | status open | proposed by sz3cheny1 | tags: cuda, pytorch, gpu

Dataset and src path:
https://drive.google.com/file/d/1zbGpyr9M5Pvhgfy96V9erQwAeRZo23hW/view?usp=drive_link

---

## Listing text (verbatim)

We predict 3D affinities (3 channels, offsets -z/-y/-x, uint8 [3,Z,Y,X]) on GPU, fast.
Turning them into a segmentation runs on CPU with `waterz` (github.com/funkey/waterz),
and that step is now the pipeline bottleneck by orders of magnitude: stock waterz does
the whole affinity->labels pipeline at **6.7 Mvox/s, single-threaded**.

Build a GPU implementation (PyTorch + Triton/CUDA, or CUDA with a Python binding) of the
**whole** affinity->labels step: watershed fragments AND the agglomeration on top of them.
A bare over-segmentation is not the deliverable: the output is the final segmentation,
graded against ground truth.

Reference = stock `waterz.agglomerate(affs, thresholds)` with defaults: scoring
`OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>`, `aff_threshold_low=0.0001`,
`aff_threshold_high=0.9999`, `discretize_queue=0` (an exact `std::priority_queue`
min-heap: waterz's merge order is exact, not bucketed).

**The affinity graph.** Three channels are stored: `aff[c][z,y,x]` is the affinity of the
edge between voxel `(z,y,x)` and its neighbour one step **back** along axis `c`
(c=0 -> z, 1 -> y, 2 -> x). Each voxel nonetheless participates in **6 edges**, because
the forward edges are the backward edges of its neighbours: voxel `v` reads its `+z`
affinity as `aff[0][z+1][y][x]`. Three channels stored, 6-connected graph. Nothing is
6-channel.

What waterz does:

1. **Watershed fragments**: for each voxel take the 6 incident edge affinities
   (`aff[0..2]` at the voxel for -z/-y/-x, `aff[0..2]` at the forward neighbours for
   +z/+y/+x; out-of-volume substitutes `low`). Let `m` be their max. If `m <= low` the
   voxel is background (label 0). Otherwise mark every direction `d` with
   `aff_d == m || aff_d >= high` as a flow direction. Plateaus (voxels mutually pointing
   at each other) are resolved by BFS from plateau corners, rewriting each plateau voxel
   to the single direction that leads off the plateau. Connected flow basins become
   fragments, IDs from 1.
2. **Region graph**: one edge per adjacent fragment pair, weight = mean affinity over
   the contact faces. Built by scanning only the three negative directions per voxel,
   which visits each adjacent pair exactly once. Fragment-to-background edges are
   discarded; background never agglomerates.
3. **Agglomeration**: pop the globally cheapest edge, `score = 1 - mean_affinity`; stop
   when `score >= threshold`, i.e. merge strictly while `mean_affinity > affinity
   threshold`. When merging `b` into `a`, neighbours exclusive to `b` have their edge
   moved to `a`; neighbours shared with `a` have the two edges combined by
   contact-area-weighted mean into the **cheaper** of the two, the other deleted, the
   survivor marked stale and rescored when next popped. Scores are monotone
   non-decreasing.
4. **Extract**: path-compress the union-find once, relabel every voxel by one lookup.
   Emit uint32 labels.

Replicate the *result*, not the code. Change the algorithm freely: parallel or blockwise
watershed, mutex watershed, seeded / priority-flood variants, GPU union-find or connected
components, bucketed or sorted-edge agglomeration, approximate merge order: as long as
the accuracy gate holds. Do not aim for bit-identity with waterz: it is not even
bit-identical to itself. Two identical `agglomerate()` calls return different labels
(raw fragment counts vary by ~20 in 824 000, VOI in the 5th decimal), because plateau
tie-breaking is order-dependent. Measured spread at the graded threshold is 1.7e-05 on
split and 1.5e-05 on merge: 1000x under the gate. Aim for statistical equivalence.

Dataset, baseline labels, a PASS/FAIL grading script and the benchmark-volume generator
ship as a 532 MB tarball. Threshold convention: waterz thresholds are *scores*
(`score = 1 - mean_affinity`); we quote **affinity** thresholds throughout.

## Done looks like
- Python API returning **final labels**, not fragments:
  `segment(aff, thresholds, aff_low=1e-4, aff_high=0.9999) -> uint32 [Z,Y,X]` per
  threshold, accepting uint8 or float32 `[3,Z,Y,X]` numpy or torch CUDA tensors, plus a
  CLI reading the provided hdf5.
- **Accuracy** on the provided CREMI-A validation volume ([3,125,1200,1200] affinity from
  a published checkpoint + uint32 GT), at affinity thresholds **0.2, 0.3, 0.4, 0.5**,
  both halves within +0.02 of stock waterz:

  | aff_thr | baseline VOI split | baseline VOI merge |
  |---|---|---|
  | 0.2 | 0.3779 | 0.3325 |
  | 0.3 | 0.4538 | 0.2411 |
  | 0.4 | 0.5178 | 0.2181 |
  | 0.5 | 0.6109 | 0.2093 |

  aff_thr 0.3 is the optimum (total VOI 0.6949). PASS requires
  `voi_split <= base + 0.02` AND `voi_merge <= base + 0.02` at **every** one of the four
: trading split against merge is not a pass. No post-processing beyond a min-size
  filter. Metric is `waterz.evaluate(labels.uint64, gt.uint64)`:
  `voi_split = H(seg|gt)`, `voi_merge = H(gt|seg)`, in bits. The shipped script prints
  the verdict.
- **Speed: >= 2 Gvox/s end-to-end on a single NVIDIA RTX 3090 Ti.**
 - The benchmark volume is exactly **`[3, 375, 2400, 2400]` = 2.16 Gvox**, produced by
    `python make_big.py` with default settings from the shipped CREMI affinity (it
    mirror-tiles it 3x2x2; ~3 min, 3.05 GB on disk). Not a crop, not a different tiling
: that volume. At the target rate one run is ~1.1 s, long enough for a stable
    measurement.
 - End-to-end = affinity already in VRAM -> final uint32 labels in VRAM, at one
    threshold (0.3). Host<->device transfer and disk I/O excluded; watershed, region
    graph, agglomeration and relabel all included.
 - Report **median, min and max of 5 timed runs** after a warm-up, timed with CUDA
    events, plus a per-stage breakdown. Median is graded.
 - Also report the same measurement on the 1.44 Gvox volume
    (`make_big.py --factors 2 2 2`) so scaling is visible. Not graded.
 - It fits: 6.5 GB uint8 affinity + 8.6 GB uint32 labels = 15.1 GB, leaving ~9 GB of the
    3090 Ti's 24 GB for fragment and graph structures. Chunk internally if your design
    needs more, but the timing covers the whole volume.
 - Develop on any GPU; the reported number must come from a 3090 Ti. State the exact
    card and driver.
- **Deterministic**: same input -> byte-identical labels, run to run. (waterz itself is
  not; you control your own tie-breaking, so you can be.)
- One-command benchmark + eval script that regenerates the VOI table and the timing table.
- Short README: algorithm, where it diverges from waterz and why, memory scaling, known
  failure modes.

## Out of scope
- Training or changing the affinity network.
- Meshing, skeletonization, proofreading, multi-node distributed stitching.
- CPU fallback, ONNX/TensorRT packaging, multi-GPU.
- Matching our production tiled agglomeration stack: the gate is stock waterz.
- Long-range / mutex affinity channels as *input*: the input is and stays 3 channels
  (-z/-y/-x). Deriving extra offsets internally is fine.

## Constraints
- Python 3.10+, PyTorch. Triton and/or CUDA C++ fine. No proprietary or
  non-redistributable dependencies.
- Input affinities are uint8, scale by /255. Never scale to uint8 in bf16/fp16:
  `a*255+0.5` wraps at >= 0.999 and silently inverts saturated affinity. Cast to fp32
  first.
- Output uint32, background 0. Label IDs need not match waterz: only the partition is
  graded.
- Must handle anisotropic volumes (4x4x40 nm, so the z channel is much weaker than y/x)
  and tens of millions of fragments. Measured with stock waterz: **2 175 400 fragments /
  7 505 458 region-graph edges** at 180 Mvox, and **26 023 852 fragments / 90 323 139
  edges** on the 2.16 Gvox benchmark volume.
- Must run within the 3090 Ti's 24 GB on that volume; document any internal chunking.

## How to verify
Provided tarball (532 MB):
  cremiA_val/{affinity.h5, gt.h5, raw.h5}   [3,125,1200,1200] uint8 + [125,1200,1200] uint32 GT
  baseline/{labels_thr*.h5, voi.csv, run_baseline.py}
  make_big.py                               builds the 2.16 Gvox benchmark volume
  viz/                                      reference renders of affinity + baseline segmentation
  README.md                                 exact semantics, verified line-by-line against the waterz source

1. `pip install waterz h5py numpy`. `python baseline/run_baseline.py` reproduces voi.csv
   from the shipped affinity: the table above, to within waterz's own ~1e-05 run-to-run
   jitter. Reference cost: 26.9 s for 180 Mvox (watershed 3.8 s, region graph ~14 s,
   merge 0.5-4 s, relabel ~1 s), single-threaded.
2. Run your `segment()` on the same affinities, thresholds 0.2/0.3/0.4/0.5, save labels
   as `mine_thr<t>.h5` (dataset "labels", uint32).
3. `python baseline/run_baseline.py --candidate mine_thr0.2.h5 mine_thr0.3.h5
   mine_thr0.4.h5 mine_thr0.5.h5` -> must print `ACCURACY GATE: PASS`.
4. `python make_big.py` -> [3,375,2400,2400] = 2.16 Gvox. Time `segment()` on it at
   threshold 0.3 on a 3090 Ti. Median of 5 must be >= 2 Gvox/s. For scale, stock waterz
   needs 145 s on that volume for the watershed and region graph alone.
   (`make_big.py --verify` shows why mirroring, not `np.flip`, is used: affinity is an
   edge quantity, and mirrored crops reproduce the original watershed fragment count
   exactly: 54936 vs 54936: while naive flipping is off by 8-83%.)
5. Run twice, `np.array_equal` the two label volumes -> True.

---

## Hard numbers (do not round)

| Item | Value |
|---|---|
| Stock waterz throughput | 6.7 Mvox/s, single-threaded |
| Scoring | `OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>` |
| `aff_threshold_low` | 0.0001 |
| `aff_threshold_high` | 0.9999 |
| `discretize_queue` (reference) | 0 (exact min-heap) |
| Affinity layout | uint8 or float32 `[3,Z,Y,X]`; c=0/1/2 -> -z/-y/-x |
| Graph | 6-connected; 3 stored channels |
| Val volume | `[3,125,1200,1200]` = 180e6 voxels |
| Val GT | `[125,1200,1200]` uint32 |
| Graded aff thresholds | 0.2, 0.3, 0.4, 0.5 |
| VOI split baseline | 0.3779, 0.4538, 0.5178, 0.6109 |
| VOI merge baseline | 0.3325, 0.2411, 0.2181, 0.2093 |
| VOI slack | +0.02 on **each** half, **every** threshold |
| Optimum quoted | aff_thr 0.3, total VOI 0.6949 |
| Metric | `waterz.evaluate(labels.uint64, gt.uint64)`; split=`H(seg\|gt)`; merge=`H(gt\|seg)` bits |
| Allowed extra post | min-size filter only |
| Speed card | single NVIDIA RTX 3090 Ti |
| Speed volume | `[3,375,2400,2400]` = 2.16e9 voxels |
| Speed threshold | affinity 0.3 |
| Speed target | median of 5 >= 2 Gvox/s (~1.08 s) |
| Timing method | CUDA events; warmup excluded; H2D/D2H/disk excluded |
| Also report (not graded) | `[3,?,?]` from `make_big.py --factors 2 2 2` = 1.44 Gvox |
| VRAM fit | 6.5 + 8.6 = 15.1 GB aff+labels; ~9 GB leftover of 24 GB |
| Fragments @ 180 Mvox | 2 175 400 |
| RAG edges @ 180 Mvox | 7 505 458 |
| Fragments @ 2.16 Gvox | 26 023 852 |
| RAG edges @ 2.16 Gvox | 90 323 139 |
| Output | uint32, background 0 |
| Determinism | byte-identical labels, run to run |
| Python | 3.10+ |
| Framework | PyTorch; Triton and/or CUDA C++ fine |
| uint8 scale | `/255`; never `a*255+0.5` in bf16/fp16 |
| waterz self-jitter | ~20 fragments in 824 000; VOI 1.7e-05 split / 1.5e-05 merge |
| Tarball | 532 MB |
| `make_big.py` default | mirror-tile 3x2x2; ~3 min; 3.05 GB on disk |
| waterz val cost (quoted) | 26.9 s / 180 Mvox (WS 3.8, RAG ~14, merge 0.5-4, relabel ~1) |
| waterz big-vol WS+RAG (quoted) | 145 s |

## API that must exist

```
segment(aff, thresholds, aff_low=1e-4, aff_high=0.9999) -> uint32 [Z,Y,X]
```

per threshold. `aff` is uint8 or float32 `[3,Z,Y,X]`, numpy or torch CUDA.
CLI reads the provided hdf5.

Candidate dumps: `mine_thr<t>.h5` dataset `"labels"` uint32.

Grader:

```
python baseline/run_baseline.py --candidate mine_thr0.2.h5 mine_thr0.3.h5 mine_thr0.4.h5 mine_thr0.5.h5
```

must print `ACCURACY GATE: PASS`.
