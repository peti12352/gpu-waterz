# GPU waterz: affinity watershed + mean agglomeration that still matches CPU waterz VOI

Putting Funke-style waterz on the GPU without changing the partition.
Numbers: idle RTX 5090, CUDA events, affinity already in VRAM. Pin `data/cache/N19_I0_REPRO.json`.

---

## Problem

Electron microscopy of neuropil is a 3D grayscale volume. Neurons are packed, look alike, and snake through thousands of slices. A 3D CNN emits **affinities**: for each voxel, three numbers in `[0, 1]` for "this voxel and its +z / +y / +x neighbor are the same object." High affinity means stay together. Low affinity means membrane / cut.

Funke et al. (MALA / waterz) made that the default decode in a lot of connectomics code. The network runs on GPU. Turning affinities into labels -- one integer per voxel, background 0 -- is still often a CPU call to [waterz](https://github.com/funkey/waterz). In LSD/daisy that call sits in a block worker: affinities in, fragments, agglomeration, labels or a region graph out ([lsd agglomerate worker](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/workers/agglomerate_worker.py)).

1. **Predict affinities** (learning).
2. **Cluster the affinity graph** (this repo).

We did (2) on GPU, matching stock waterz **Variation of Information** on CREMI-A.

## Affinities, then two clusterings

The volume is a 6-neighbor grid graph. Each undirected voxel-voxel edge has weight = predicted affinity. A segmentation is a partition of the voxels: cut where the network said "different neuron," keep together where it said "same."

Thresholding ("cut every edge below 0.3, take connected components") is brittle. Thin errors in the affinity map punch holes through a neurite or glue two axons. Waterz therefore **over-segments, then merges**:

**Watershed (fragments).** Grow many small, conservative pieces that almost never cross a true membrane. Better 100 fragments of one axon than glue two axons here: later agglomeration can merge and cannot (in waterz) unmerge.

**Agglomeration.** Fragments become nodes of a **region adjacency graph** (RAG). An edge exists when two fragments touch. The weight is a statistic of the affinities on the contact surface -- in waterz, by default, the **contact-area-weighted mean**. Repeatedly merge the current most-attractive pair until remaining contacts are weaker than a threshold. That partition is what people grade.

The two-stage shape is older than deep learning. Waterz specifies affinity-flow watershed (plateaus, background floor) and mean merge updated after every merge.

## Affinity-flow watershed

Classic watershed floods a height map from local minima. Distance-transform watershed (Fiji, ilastik DT-WS) does that on a boundary probability map. Waterz is a **flow** on the affinity graph.

Each voxel looks at its six neighbors and flows toward the locally strongest affinity, with two clamps: below `aff_low` you are background; above `aff_high` you are definitely connected. Where several voxels share the same max (a plateau), they must become **one** basin, not one basin per pixel. Stock waterz's plateau rule is load-bearing. Extra closed-plateau components -- a tempting GPU simplification -- fail VOI against the CPU reference. The BFS that rewrites plateaus is cheap (~14 ms on 2.16 Gvox). The expensive part is compressing the union-find after basin assignment (`k_w5_compress_list`, ~508 ms). Changing plateau semantics to "accelerate watershed" speeds up a different algorithm.

Fragments are an over-segmentation you are allowed to glue.

## The RAG and contact-mean

After watershed, CREMI-A val has about 2.18 million fragments and 7.5 million contact edges. Each RAG edge stores `(sum, count)` of the uint8 affinity bytes on the shared face. Mean = sum/count.

When fragments A and B merge into C, every neighbor of A or B now touches C. The mean on C--D is not `min(mean(A,D), mean(B,D))`. It is the pooled `(sum_AD + sum_BD) / (count_AD + count_BD)`. That is **average linkage** / UPGMA on the contact graph, not Kruskal on frozen weights.

Kruskal (and mutex / single-linkage / "always merge the current max frozen edge") treats the original voxel-face affinities as fixed. Mean agglomeration re-asks after every glue: now that these two lumps are one object, how strongly does the combined lump stick to its neighbors? A weak sliver of contact can be diluted by a large strong contact, or a large weak contact can drag a previously strong pair below threshold. A GPU MST, mutex watershed, or GASP AbsMax on the same voxels is a different clustering.

We ran those classes on the same cached CREMI-A RAG with the same VOI grader (`data/cache/voi_atlas.csv`):

- Frozen connected components / waterfall: giant object. Merge VOI around 7.5 (the volume glued together).
- Mutex / GASP AbsMax: under-merge. Split VOI around 0.9 to 2.1 (axon left in pieces).
- Kruskal SDSL: essentially no merges at the cuts we grade.

NPP "GPU watershed" and OpenCV watershed already disagree on ordinary 2D images; the predicate is the product. Mutex-on-affinities in a PyTorch thread is the same story: useful, other clustering.

## Thresholds: affinity vs score

Waterz's default C++ scoring string is `OneMinus<MeanAffinity>`. The heap pops on **score** ~ `1 - mean_affinity`. Merge while `score < T_score`, i.e. while `mean_affinity > T_aff`.

If 0.3 is a reasonable membrane, you want `T_aff = 0.3`, which is `T_score = 0.7`. The waterz README example `[0, 100, 200]` is in score-ish units. Our API is affinity unless you pass `threshold_mode="score"`. Mixing them cuts the dendrogram at the wrong height.

## VOI

Variation of Information splits into **split** (one true neuron became many labels) and **merge** (two true neurons became one label). Lower is better. We require both halves, at affinity 0.2, 0.3, 0.4, 0.5, on both spatial halves of CREMI-A val, within +0.02 of stock waterz. That is partition quality, not matching label IDs. Stock waterz is not run-to-run identical on plateaus. We are.

## Why a GPU heap is the wrong closer

Exact average-linkage HAC has no friendly poly-log parallel algorithm under standard complexity assumptions (ParHAC 2022; Abboud et al., [arXiv:2404.14730](https://arxiv.org/abs/2404.14730)). The serial waterz heap is global min edge, merge, reweight, repeat. Mapping that to CUDA as "one pop per kernel" keeps the serial chain.

Approximate the **order of merges**; keep the **mean statistic**. ParHAC does (1+eps)-heavy matching: many disjoint merges in a round if they are close enough to locally heaviest, then contract the graph, repeat. On this RAG `eps` is a phase boundary, not an ML "quality vs speed" knob. Four-threshold VOI needs `eps = 0.08`. The single-cut aff=0.3 path allows `eps = 0.40`. Every 0.01 step from 0.41 to 0.49 fails merge VOI. Hence a dual-eps default.

Lu, Zlateski, and Seung ([arXiv:2106.10795](https://arxiv.org/abs/2106.10795)) distribute exact mean clustering over chunks by freezing anything that touches a fake boundary until the object fits in one chunk. A freeze that is not their Algorithm 2 produces different parents than ParHAC. We stopped.

## Experiments

Each attack is a hypothesis about which constraint is allowed to move.

| Hypothesis | Why it was reasonable | What happened |
|---|---|---|
| Frozen-weight MST / mutex / GASP | GPU-friendly; lots of literature | Wrong partition class. Atlas. |
| Exact heap on GPU | Bit-identical to waterz | Serial; not a throughput plan. |
| BinQueue (waterz's discretized priority queue) | Already in waterz; maybe parallelize bins | Hang, slow, or VOI fail. |
| Filter to reciprocal nearest neighbors | Fewer edges, maybe same dendrogram | Split VOI ~2.15: merge order changed. |
| "Lu freeze" without Alg 2 | Distributed mean, simpler | Parents != ParHAC. |
| Speed up BFS / Playne | Watershed looks like BFS | BFS is 14 ms; list-compress is 508 ms. |
| Park affinity on the host between kernels | Peak VRAM | Timed PCIe. Parks off roughly doubled e2e with the same labels. |
| Bigger eps | (1+eps) theory says more parallelism | Merge VOI dies past 0.40 on T=0.3. |

[funkey/waterz PR 24](https://github.com/funkey/waterz/pull/24) took RAG extract 52s -> 18s and agglomeration 71s -> 28s on 1024x1024x512, and the witty JIT 30s -> 4.3s. That is the CPU class we replace.

What survived: GPU affinity-flow with stock plateau semantics, GPU contact-mean RAG, device ParHAC, dual-eps. Four-threshold VOI pass. Labels byte-identical across two full 2.16 Gvox runs.

Idle 5090, `[3,375,2400,2400]` = 2.16 Gvox, aff 0.3:

| stage | ms |
|---|---|
| watershed | ~1309 |
| RAG | ~85 |
| agglomeration | ~1680 |
| extract | ~18 |
| e2e | ~3093 (~0.70 Gvox/s) |

Agglomeration is the largest slice. List-compress is the largest watershed slice.
Best pin, remaining paths, and closed doors: `notes/WHERE_WE_ARE.md`.
Open problem: `notes/PROBLEM.md`. A co-tenant process on the GPU makes these
numbers meaningless; the harness refuses if the card is busy.

## How to call it

Affinities are `[3, Z, Y, X]`, float32 in `[0, 1]` or uint8. Torch CUDA tensors go in through the CUDA Array Interface (same protocol Numba/CuPy use). Thresholds are affinity.

```
uv sync
bash scripts/build_cuda.sh
uv run python examples/torch_to_labels.py
uv run python examples/zarr_block.py
```

```python
import gpu_waterz as wz
labs = wz.segment(aff, [0.2, 0.3, 0.4, 0.5])
# waterz score list:
labs = wz.segment(aff, [0.8, 0.7, 0.6, 0.5], threshold_mode="score")
```

`examples/zarr_block.py` is the daisy-shaped loop: one affinity block in, one label block out. FlyWire-style proofreading (ChunkedGraph) sits after an agglomerated supervoxel graph. Query models like AGQ train a different decoder.
