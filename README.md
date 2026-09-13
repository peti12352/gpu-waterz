# gpu-waterz

GPU affinity-flow watershed + contact-mean agglomeration. Matches stock
[`waterz`](https://github.com/funkey/waterz) partition quality on CREMI-A
affinities; call it from numpy or torch CUDA tensors.

Docs: [pipeline](docs/pipeline.md), [CUDA build](docs/build.md),
[where we are](notes/WHERE_WE_ARE.md), [campaign atlas](notes/ATLAS.md),
[VOI atlas CSV](data/cache/voi_atlas.csv), [open questions](notes/PROBLEM.md),
[engineering note](docs/hn.md).

---

## Install / quickstart

```bash
uv sync
bash scripts/build_cuda.sh   # needs nvcc; writes src/lib*.so
uv run python examples/torch_to_labels.py
```

```python
import numpy as np
import gpu_waterz as wz

aff = np.random.rand(3, 32, 64, 64).astype(np.float32)  # [3,Z,Y,X]
labs = wz.segment(aff, [0.2, 0.3, 0.4, 0.5])  # list of uint32 [Z,Y,X]
# or: wz.agglomerate(...)  # same e2e; stock-waterz name
```

Torch CUDA (zero-copy via ``__cuda_array_interface__`` when libs are built):

```python
import torch, gpu_waterz as wz
aff = torch.rand(3, 32, 64, 64, device="cuda")
labs = wz.segment_d(aff, [0.3], return_device=True)
lab_t = wz.to_torch(labs[0])
```

See ``examples/torch_to_labels.py``.

## Threshold units (read this)

**API thresholds are affinity** (merge while contact-mean affinity > thr).

Stock waterz default scoring is ``OneMinus<MeanAffinity>``, so many call sites
pass **scores** ``1 - aff``. Convert explicitly:

```python
wz.scores_to_affinity([0.8, 0.7, 0.6, 0.5])  # -> [0.2, 0.3, 0.4, 0.5]
```

Details: [notes/THRESHOLD.md](notes/THRESHOLD.md).

## API

| Function | Role |
|---|---|
| ``segment(aff, thresholds, *, threshold_mode="affinity", eps=None, ...)`` | e2e labels |
| ``agglomerate(...)`` | alias of ``segment`` (list, not waterz generator) |
| ``segment_d(..., return_device=True)`` | device-resident path; torch CUDA in -> torch out |
| ``fragments(aff)`` | watershed only |
| ``region_graph(aff, frag)`` | contact-mean RAG arrays |
| ``scores_to_affinity`` / ``affinity_to_scores`` | unit conversion |
| ``from_torch`` / ``to_torch`` | torch helpers |
| ``cuda_libs_ready()`` | whether ``src/lib*.so`` exist |

``eps`` overrides ParHAC (1+eps). Else ``WATERZ_AGG_EPS`` or dual-eps defaults
(0.08 multi-T / 0.40 single T=0.3). Lab harnesses may still use ``WATERZ_*``
env vars.

Naming: stock ``waterz.agglomerate`` = full pipeline (our ``segment``). LSD
``agglomerate_worker`` = merge-from-fragments (our ParHAC stage after
``fragments``). See [docs/pipeline.md](docs/pipeline.md).

## Where this sits

```
gunpowder/torch train -> daisy predict -> [gpu_waterz decode] -> zarr
                                         optional: CAVE / ChunkedGraph
```

Speeds the waterz-class decode. Does not replace proofreading stacks or
end-to-end query models (e.g. AGQ).

## Problem

In connectomics pipelines the expensive trained step is usually affinity
prediction. The remaining decode -- watershed fragments, region graph, and
hierarchical agglomeration -- is still often a CPU ``waterz`` call inside
daisy/LSD-style workers. That call is hard to accelerate without changing the
partition: contact-mean scores reweight after every merge, so naive Kruskal /
mutex / frozen-edge GPU tricks are a different algorithm. Thresholds are easy
to misuse (stock waterz scores vs affinity), and production stacks want stage
hooks (fragments / RAG / merge) rather than only a monolith. Exact average-
linkage HAC has no friendly parallel exact algorithm; any GPU path must be
approximate and still match waterz VOI.

CNNs predict 3D affinities on GPU; CPU ``waterz`` decode is often the slow
step (~6.7 Mvox/s single-threaded on this workload). This repo runs:

1. affinity-flow watershed fragments (waterz S1 semantics)
2. contact-mean region adjacency graph
3. hierarchical agglomeration under the waterz mean statistic
4. final uint32 labels (background 0)

Reference scoring string: ``OneMinus<MeanAffinity<...>>``. Graded object is
the **partition after agglomeration**, not fragment over-segmentation alone.

## Why contact-mean

Mean affinity updates after every merge. Not Kruskal on frozen weights, not
mutex / AbsMax, not single-linkage MST. On the CREMI-A contact-mean RAG those
classes produce systematically different partitions. Measured atlas:
[data/cache/voi_atlas.csv](data/cache/voi_atlas.csv).

Exact average-linkage HAC is hard; practical GPU path uses **(1+eps)-approximate**
ParHAC on the same statistic and checks VOI (not bit-identity with the serial heap).

## Quality bar

CREMI-A val affinities ``[3,125,1200,1200]`` with GT:

- VOI split and VOI merge each within +0.02 of stock waterz at affinity
  thresholds **0.2, 0.3, 0.4, 0.5** (both halves, every threshold)
- Run-to-run byte-identical labels

| aff | VOI split | limit | VOI merge | limit |
|-----|-----------|-------|-----------|-------|
| 0.2 | 0.3707 | 0.3979 | 0.3350 | 0.3525 |
| 0.3 | 0.4512 | 0.4738 | 0.2505 | 0.2611 |
| 0.4 | 0.5162 | 0.5378 | 0.2268 | 0.2381 |
| 0.5 | 0.6129 | 0.6309 | 0.2184 | 0.2293 |

## Algorithm (current stack)

Vendored waterz pin: ``funkey/waterz`` ``a0184d2``.

1. **Flow (GPU).** 6-neighbour affinities, OOB = ``low``, bits where
   ``aff == m || aff >= high``. Background iff ``m <= low``.
2. **Plateau + basins (GPU).** Corner BFS rewrite, then list union-find.
3. **RAG (GPU).** Three negative dirs; atomic ``(sum, count)``; mean = sum/count.
4. **Agglomeration (GPU).** Paper-style ParHAC; dual-eps as above.

Recommended env for the measured VOI stack:

```
WATERZ_UF_ALGO=3 WATERZ_HOST_PARK=0 WATERZ_AFF_PARK=0 WATERZ_AGG_LEVERS=15
WATERZ_FOLD_FLATTEN=1 WATERZ_SHARE_OFF=1 WATERZ_HOOK_ROOT=1
WATERZ_FUSE_DIRTY=1 WATERZ_NLIVE_ARITH=1 WATERZ_EMIT_HOLES=1
```

## Measured performance

Idle RTX 5090, CUDA events, affinity in VRAM, mirror-tiled
``[3,375,2400,2400]`` = 2.16 Gvox at affinity 0.3
(``data/cache/N19_I0_REPRO.json``):

| stage | ms |
|---|---|
| watershed | ~1309 |
| RAG | ~85 |
| agglomeration | ~1680 |
| extract | ~18 |
| end-to-end | ~3093 (~0.70 Gvox/s) |

Four-threshold VOI: PASS. Labels byte-identical across two full runs.
Where the campaign stands (best pin, remaining paths, closed doors with
proof): [notes/WHERE_WE_ARE.md](notes/WHERE_WE_ARE.md).
Campaign report: [notes/ATLAS.md](notes/ATLAS.md).

```
bash scripts/legal_eval.sh
python src/segment.py cremiA_val/affinity.h5 --out-dir . --thresholds 0.2 0.3 0.4 0.5
```

## Findings

1. Mean affinity is not Kruskal (atlas).
2. (1+eps) contact-mean ParHAC cleared the four-T VOI gate; eps boundary near 0.40 on aff-0.3.
3. Timing must exclude host parking of affinity/corners.
4. Plateau semantics matter more than BFS wall time.
5. New merge classes and leftover kernel tricks did not move the 2.16 pin
   ([WHERE_WE_ARE.md](notes/WHERE_WE_ARE.md)).

Open questions: [notes/PROBLEM.md](notes/PROBLEM.md).

## Divergence from stock waterz

- Deterministic plateau/basin ties (fixed dir + min index).
- Fragment IDs need not match; the partition is what is compared.
- Agglomeration is (1+eps) ParHAC, not the serial exact heap.

## Known failure modes

- Mutex / frozen CC / Kruskal / FH / SRM / Soille / GASP Average / NNG: atlas.
- Raising eps past the measured boundary fails merge VOI.
- Co-tenant GPU use makes e2e times meaningless.

Dev measurements: RTX 5090, driver 580, nvcc 12.8, ``sm_120``.
