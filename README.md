# gpu-waterz

GPU affinity-flow watershed and contact-mean agglomeration. Same partition
class as stock [`waterz`](https://github.com/funkey/waterz) on CREMI-A;
call it from numpy or torch CUDA.

```
EM volume -> CNN affinities [3,Z,Y,X] -> gpu_waterz.segment -> uint32 labels
```

This is the decode after affinity prediction, not training, meshing, or
proofreading.

## Install

```bash
uv sync
bash scripts/build_cuda.sh   # nvcc; writes src/libws_gpu.so librag_gpu.so libparhac_d.so
uv run python examples/torch_to_labels.py
```

`uv sync` is the Python package. The `.so` files are a separate nvcc step.
Missing libs raise `RuntimeError` pointing at `scripts/build_cuda.sh`.
Override the compiler with `WATERZ_NVCC`. Fatbin is `sm_86` and `sm_120`.

```python
import numpy as np
import gpu_waterz as wz

aff = np.random.rand(3, 32, 64, 64).astype(np.float32)
labs = wz.segment(aff, [0.2, 0.3, 0.4, 0.5])  # list of uint32 [Z,Y,X]
```

```python
import torch, gpu_waterz as wz
aff = torch.rand(3, 32, 64, 64, device="cuda")
labs = wz.segment_d(aff, [0.3], return_device=True)  # CUDA torch tensors
```

## API

Thresholds are **affinity** (merge while contact-mean > T). Stock waterz
heaps on **score** `~ 1 - affinity`. Convert explicitly:

```python
wz.segment(aff, [0.8, 0.7, 0.6, 0.5], threshold_mode="score")
# same as affinity 0.2, 0.3, 0.4, 0.5
```

| Function | Role |
|---|---|
| `segment(aff, thresholds, *, threshold_mode="affinity", eps=None, ...)` | affinities to labels |
| `agglomerate(...)` | same as `segment` (list, not a waterz generator) |
| `labels_from_fragments(aff, frag, thresholds, ...)` | merge existing fragments (LSD agglomerate-worker shape) |
| `segment_d(..., return_device=True)` | device-resident; torch CUDA in stays on device |
| `fragments(aff)` | watershed only |
| `region_graph(aff, frag)` | contact-mean RAG arrays |
| `scores_to_affinity` / `affinity_to_scores` | unit conversion |
| `from_torch` / `to_torch` | torch helpers |
| `cuda_libs_ready()` | whether `src/lib*.so` exist |

`eps` sets ParHAC `(1+eps)`. `None` uses `WATERZ_AGG_EPS` or dual-eps:
**0.08** for several cuts, **0.40** for a single threshold of 0.3.
Raising past 0.40 fails merge VOI at 0.3.

Stock `waterz.agglomerate` is the full pipeline (our `segment`). LSD's
agglomerate worker is merge-from-fragments (`labels_from_fragments`).

## Quality (CREMI-A val `[3,125,1200,1200]`)

VOI split **and** VOI merge each within +0.02 of stock waterz at affinity
0.2, 0.3, 0.4, 0.5. Run-to-run labels are byte-identical. Fragment IDs
need not match waterz; the partition is what is graded.

| aff | VOI split | limit | VOI merge | limit |
|-----|-----------|-------|-----------|-------|
| 0.2 | 0.3707 | 0.3979 | 0.3350 | 0.3525 |
| 0.3 | 0.4512 | 0.4738 | 0.2505 | 0.2611 |
| 0.4 | 0.5162 | 0.5378 | 0.2268 | 0.2381 |
| 0.5 | 0.6129 | 0.6309 | 0.2184 | 0.2293 |

Those two VOI numbers are what older notes call "both halves" — not two
spatial crops. Mutex, Kruskal, and frozen-edge MST produce different
partitions on the same RAG: [data/cache/voi_atlas.csv](data/cache/voi_atlas.csv).

## Speed (idle RTX 5090)

CUDA events, affinity already in VRAM, `[3,375,2400,2400]` = 2.16 Gvox,
affinity 0.3. Pin `data/cache/N19_I0_REPRO.json`.

| stage | ms |
|---|---|
| watershed | 1309 |
| RAG | 85 |
| agglomeration | 1680 |
| extract | 18 |
| end-to-end | 3093 (~0.70 Gvox/s) |

Four-threshold VOI: PASS. Two full runs, identical labels. A busy GPU
makes these times meaningless (one co-tenant run moved 1634–5018 ms).

Env used for that pin:

```
WATERZ_UF_ALGO=3 WATERZ_HOST_PARK=0 WATERZ_AFF_PARK=0 WATERZ_AGG_LEVERS=15
WATERZ_FOLD_FLATTEN=1 WATERZ_SHARE_OFF=1 WATERZ_HOOK_ROOT=1
WATERZ_FUSE_DIRTY=1 WATERZ_NLIVE_ARITH=1 WATERZ_EMIT_HOLES=1
```

Do not turn on `WATERZ_LISTED_INSERT`, `WATERZ_LISTED_REBUILD`,
`WATERZ_SLOT_EMIT`, or `WATERZ_LIST_JUMP` by default.

```
bash scripts/legal_eval.sh
```

## Docs

| File | What it is |
|---|---|
| [docs/usage.md](docs/usage.md) | pipeline call sites, stages, build |
| [docs/decode.md](docs/decode.md) | why this algorithm, VOI, knobs vs other clustering |
| [docs/porting.md](docs/porting.md) | other GPUs: build, VRAM, how to time |
| [notes/README.md](notes/README.md) | lab log (current pin, closed attacks, stamps) |

Vendored waterz: `funkey/waterz` `a0184d2`. Measurements: driver 580, nvcc 12.8.
