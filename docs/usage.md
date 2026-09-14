# Using gpu-waterz

Install and API table: [README](../README.md). What the algorithm is, and
what the CREMI-A numbers mean: [decode.md](decode.md). Other GPUs:
[porting.md](porting.md).

## vs `pip install waterz`

| | stock waterz | gpu_waterz |
|---|---|---|
| Entry | `waterz.agglomerate` (generator) | `gpu_waterz.segment` / `agglomerate` (list) |
| Device | CPU C++ | CUDA watershed + RAG + ParHAC |
| Thresholds | often **scores** under `OneMinus<MeanAffinity>` | **affinity** unless `threshold_mode="score"` |
| Stages | mostly monolith | `fragments`, `region_graph`, `labels_from_fragments`, e2e |
| Merge order | exact serial heap | (1+eps) contact-mean ParHAC; VOI-matched on CREMI-A |

Do not infer units from the numeric range. Stock waterz is called with
`score = 1 - aff` (affinity 0.2/0.3/0.4/0.5 -> scores 0.8/0.7/0.6/0.5).

Upstream: https://github.com/funkey/waterz
Region-graph helpers: https://github.com/PytorchConnectomics/waterz

## Build

```bash
uv sync
bash scripts/build_cuda.sh
# WATERZ_NVCC=/path/to/nvcc bash scripts/build_cuda.sh
```

Writes `src/libws_gpu.so` (`csrc/ws.cu`), `src/librag_gpu.so` (`csrc/rag.cu`),
`src/libparhac_d.so` (`csrc/parhac_d.cu`). Check:

```bash
uv run python -c "import gpu_waterz as w; print(w.cuda_libs_ready())"
```

## Torch

```python
import torch
import gpu_waterz as wz

aff = torch.rand(3, 32, 64, 64, device="cuda", dtype=torch.float32)
labs = wz.segment_d(aff, [0.3], return_device=True)
# labs[0] is already a CUDA torch tensor
```

`examples/torch_to_labels.py` is a synthetic cube (no CREMI download).

## Daisy / LSD-style workers

`examples/zarr_block.py` is the block loop: affinity `[3,Z,Y,X]` in, uint32
labels out. No Mongo.

1. Read an affinity block from zarr.
2. `segment(aff, [thr])` for e2e, or `fragments` then `labels_from_fragments`
   if fragment and merge workers are split. `region_graph` if you only need
   the RAG arrays.
3. Write labels.

[LSD agglomerate worker](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/workers/agglomerate_worker.py)
is merge-from-fragments. Our `agglomerate` name matches stock waterz (full
pipeline). `labels_from_fragments` is the host numpy path, not `segment_d`.
It is still contact-mean ParHAC. CREMI-A four-T numbers assume *our*
watershed fragments on that volume.

## Knobs that stay in this clustering class

- Any threshold list (not hardcoded to 0.2/0.3/0.4/0.5)
- `threshold_mode="affinity"` or `"score"`
- `eps=` or `WATERZ_AGG_EPS` (dual-eps default otherwise)
- `aff_low` / `aff_high` (background / sure-link clamps)
- numpy host or torch CUDA

Not in the public API yet: `min_size` (allowed by the original gate, pin
uses 0), merge from a precomputed RAG without re-scanning affinities.

## Out of scope

- gunpowder `BatchFilter`
- Lu distributed freeze / production chunk-stitch ([arXiv:2106.10795](https://arxiv.org/abs/2106.10795))
- ChunkedGraph / CAVE
- mutex, Kruskal, or long-range affinity as this library's clustering
- end-to-end SOTA vs query models (e.g. AGQ)
