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

Stock waterz is called with `score = 1 - aff` (affinity 0.2/0.3/0.4/0.5
maps to scores 0.8/0.7/0.6/0.5).

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
labels out.

1. Read an affinity block from zarr.
2. `segment(aff, [thr])` for e2e, or `fragments` then `labels_from_fragments`
   if fragment and merge workers are split. `region_graph` if you only need
   the RAG arrays.
3. Write labels.

[LSD agglomerate worker](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/workers/agglomerate_worker.py)
is merge-from-fragments. Our `agglomerate` name matches stock waterz (full
pipeline). `labels_from_fragments` is the host numpy path, not `segment_d`.
CREMI-A four-T numbers assume our watershed fragments on that volume.
`min_size` and merge-from-a-precomputed-RAG are not in the public API yet.

## Eval volumes

`scripts/legal_eval.sh` looks for CREMI-A val at
`data/cremiA_val/affinity.h5` and `data/cremiA_val/gt.h5` (or
`WATERZ_VAL_DIR`). Optional `--216` times `[3,375,2400,2400]` from
`data/cremiA_216/affinity.h5`, `data/cache/big_216.h5`, or `WATERZ_AFF_216`.
