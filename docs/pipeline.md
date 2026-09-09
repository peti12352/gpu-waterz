# Using gpu-waterz in an EM/ML pipeline

## vs ``pip install waterz``

| | stock waterz | gpu_waterz |
|---|---|---|
| Entry | ``waterz.agglomerate`` (generator) | ``gpu_waterz.segment`` / ``agglomerate`` (list) |
| Device | CPU C++ | CUDA WS + RAG + ParHAC |
| Thresholds | often **scores** under ``OneMinus<MeanAffinity>`` | **affinity** (merge while mean_aff > thr) |
| Stages | mostly monolith; optional fragments arg | ``fragments``, ``region_graph``, e2e |
| Quality | exact serial heap | (1+eps) contact-mean ParHAC; VOI-matched on CREMI-A |

Convert stock score lists with ``gpu_waterz.scores_to_affinity``.

Pinned stock API: https://github.com/funkey/waterz  
Pinned PC fork (region-graph helpers): https://github.com/PytorchConnectomics/waterz

## Quick path (torch CUDA)

```python
import torch
import gpu_waterz as wz

aff = torch.rand(3, 32, 64, 64, device="cuda", dtype=torch.float32)
labs = wz.segment_d(aff, [0.3], return_device=True)
lab_t = wz.to_torch(labs[0])  # stays on GPU via CAI when possible
```

Install: ``pip install -e .`` then ``bash scripts/build_cuda.sh`` (needs nvcc).

## Daisy / LSD worker sketch (no Mongo)

1. Read affinity block from zarr (layout ``[3,Z,Y,X]``).
2. ``gpu_waterz.segment(aff, [thr])`` or stage ``fragments`` then ``region_graph``.
3. Write uint32 labels to zarr.

LSD separates fragment and agglomerate workers
(https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/workers/agglomerate_worker.py).
Our e2e ``agglomerate`` matches stock waterz naming; LSD "agglomerate" means
merge-from-fragments (closer to our ParHAC stage after ``fragments``).

## What we do not do

- No gunpowder ``BatchFilter`` in-tree
- No Lu distributed freeze / exact chunk-stitch product (arXiv:2106.10795)
- No ChunkedGraph / CAVE client
- No claim of end-to-end SOTA vs query models (e.g. AGQ); this is a faster
  waterz-class decode for affinities you already have

## Build

See [build.md](build.md).
