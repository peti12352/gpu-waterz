# GPU waterz: affinity watershed + mean agglomeration that still matches CPU waterz VOI

Draft for a regular HN post, not a Show HN. Show HN wants something a stranger can try without barriers (https://news.ycombinator.com/showhn.html). This needs CUDA `.so` files from `nvcc`. Paste later if you open the repo.

Numbers below are idle RTX 5090, CUDA events, affinity already in VRAM, parks off. Pin: `data/cache/N19_I0_REPRO.json`. Do not treat them as portable.

---

The CNN that predicts affinities already runs on the GPU. The step that turns those affinities into neuron IDs often does not. In FunkeLab pipelines that is still `waterz.agglomerate` inside a daisy/LSD worker: read a zarr block, watershed fragments, agglomerate, write labels or a region graph (https://github.com/funkelab/lsd , worker https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/workers/agglomerate_worker.py). Stock waterz is a small C++ library with a Python generator API (https://github.com/funkey/waterz). It is also a compile-time hobby: missing PyPI wheels (issue 23), NumPy 2 breaks (issue 18), JIT Cython in `~/.cython/inline` until `agglomerate` is missing (issue 17), and the README example thresholds are `[0, 100, 200]` while affinities live in `[0, 1]`. People file "why is this a generator?" (issue 8). None of that is the algorithm. The algorithm is worse.

## What "watershed" means here

Not Fiji distance-transform watershed. Not NPP `nppiSegmentWatershed`. NVIDIA's GPU watershed and OpenCV's watershed already disagree on a tray of apples (https://forums.developer.nvidia.com/t/opencv-and-npp-watershed-return-completely-different-outputs/202490). If you "port watershed to CUDA" and change the predicate, you have a different segmentation.

Connectomics waterz is **affinity-flow**:

1. Each voxel looks at 6-neighbor affinities. Flow toward the locally strongest link, with explicit plateau bits and a background floor.
2. Plateaus get one basin. Extra closed-plateau components fail VOI against stock waterz. The BFS rewrite that fills plateaus is **not** the timed owner (~14 ms of a 2.16 Gvox watershed). List-compress of the union-find is (~508 ms).
3. Fragments are an over-segmentation. The product is the **partition after agglomeration**.

Then you build a region adjacency graph: contact faces, `(sum, count)` of raw affinity bytes, mean = sum/count. Then you merge fragments while that mean stays above a threshold. Image.sc people who have never heard of CREMI still reach for a RAG and hierarchical merge after watershed (https://forum.image.sc/t/algorithm-for-aggregating-blobs-into-adjacent-labels/78865). Same shape, smaller voxels.

Stock waterz default scoring is `OneMinus<MeanAffinity>`. Their heap thresholds are **scores** (~`1 - mean_aff`). Our API thresholds are **affinity**. Mix them and you silently decode the wrong cut. `threshold_mode="score"` exists because we got tired of writing that sentence.

## Subproblem: mean is not Kruskal

After two fragments merge, every shared face with a third fragment is a new contact. The mean on that edge is not the min of the old means. Kruskal on frozen weights, mutex / AbsMax, single-linkage MST, Felzenszwalb, SRM, Soille alpha-omega, GASP Average: we ran them on the **same** CREMI-A contact-mean RAG (7.5M edges) with the **same** official VOI grader. They fail in different directions. Frozen-CC grows a giant (merge VOI ~7.5). Mutex under-merges (split ~0.9 to 2.1). Kruskal SDSL does zero merges at the cuts we care about. The table is `data/cache/voi_atlas.csv`. This is not "we didn't tune k."

PyTorch forum threads about mutex watershed on predicted affinities are post-processing, not a PyTorch op (https://discuss.pytorch.org/t/instance-segmentation-using-mutex-watershed/156083). PytorchConnectomics even added an affogato mutex decoder (PR 213) as an alternate decode path. Fine. It is a different partition class than waterz mean. We needed the waterz class.

## Subproblem: exact average-linkage hates GPUs

Exact average-linkage HAC is hard in the theory sense people actually use: no friendly poly-log parallel algorithm under standard assumptions (ParHAC 2022; Abboud et al., https://arxiv.org/abs/2404.14730). So a GPU "heap" that pops global-min edges in serial order is not a speed plan. You approximate the **matching schedule** and you keep the **statistic**. We use paper-style ParHAC: (1+eps)-heavy matching, contract, repeat. Quality gate is VOI vs stock waterz, not bit-identity of labels (stock waterz is not even self-identical across plateau ties).

Distributed exact mean on trillion-edge affinity graphs is a different paper (Lu, Zlateski, Seung, https://arxiv.org/abs/2106.10795): freeze anything that touches a chunk boundary until the chunk contains the whole object. We did **not** implement that. A simplified freeze that is not Algorithm 2 produces different parents than ParHAC. We killed it.

Flood-filling networks skip the fragment graph entirely (https://arxiv.org/abs/1611.00421). AGQ (ICLR 2025, https://openreview.net/forum?id=Y0QqruhqIa, https://github.com/chenhang98/AGQ) is a query decoder that mostly deletes watershed. Both are full systems you train. This repo is the decode box when you already paid for affinities.

## What we tried and killed

One line each. Do not reopen without new evidence.

- **eps > 0.40** on the single-threshold aff=0.3 path: merge VOI fails. Every 0.01 step from 0.41 to 0.49. eps >= 0.5 also fails. The dual-eps schedule is not a hyperparameter: 0.08 for four thresholds, 0.40 for T=0.3.
- **BinQueue** (serial or "parallel") as the GPU closer: hang, slow, or VOI fail. The waterz CPU discretized queue is not a hidden 2 Gvox/s.
- **NNG / reciprocal-NN** edge filter: split VOI ~2.15. You changed the merge order.
- **Simplified chunk freeze**: parents != ParHAC. Do not claim Lu until you match Algorithm 2.
- **mutex / GASP AbsMax / Average / Kruskal / FH / SRM / Soille / RAMA**: VOI fail or timeout. Atlas.
- **Host parking of affinity or corner lists inside the timed window**: PCIe, not watershed. Turning parks off roughly doubled e2e without touching the partition.
- **Playne / BFS rewrites** aimed at the 14 ms BFS while `k_w5_compress_list` owns 508 ms.

Stock CPU waterz agglomeration is also slow when you look: PR 24 on funkey/waterz (https://github.com/funkey/waterz/pull/24) took a 1024x1024x512 volume from 52s to 18s on RAG extract and 71s to 28s on agglomeration, and 30s to 4.3s on the witty JIT. That is the CPU baseline class. We are not competing with a straw man.

## What survived

GPU affinity-flow watershed with S1 plateau semantics. GPU contact-mean RAG. Device ParHAC on that RAG. Dual-eps. Four-threshold VOI within +0.02 of stock waterz on CREMI-A val `[3,125,1200,1200]` at aff 0.2/0.3/0.4/0.5, both split and merge, both volume halves. Labels byte-identical run to run on the 2.16 Gvox volume.

Measured on idle 5090, `[3,375,2400,2400]` = 2.16 Gvox, aff 0.3:

| stage | ms |
|---|---|
| watershed | ~1309 |
| RAG | ~85 |
| agglomeration | ~1680 |
| extract | ~18 |
| e2e | ~3093 (~0.70 Gvox/s) |

Agglomeration is still the largest piece. List-compress is the largest watershed piece. That is the remaining research problem (`notes/PROBLEM.md`), not a marketing gap.

Timing that copies the affinity to the host inside the window is a lie. `card_busy` refuse exists because a co-tenant LLM turns 3s into 15s.

## What this unlocks

A drop-in for the waterz **call site**. Affinities in `[3,Z,Y,X]`, numpy or torch CUDA via `__cuda_array_interface__` (https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html). Labels out. `examples/zarr_block.py` is the daisy-shaped loop without Mongo. Torch CAI in with `return_device=True` returns a CUDA tensor; wrapping a buffer with `torch.as_tensor` without `device="cuda"` can CPU-copy (https://github.com/pytorch/pytorch/issues/54139). We do not do that.

It does not replace FlyWire proofreading (https://news.ycombinator.com/item?id=36568609, https://pmc.ncbi.nlm.nih.gov/articles/PMC8903166/, https://github.com/CAVEconnectome/PyChunkedGraph). It does not replace AGQ. It does not ship CUDA wheels from a box without `nvcc`. It does not squat the `waterz` name on PyPI.

## How to run

```
uv sync
bash scripts/build_cuda.sh
uv run python examples/torch_to_labels.py
uv run python examples/zarr_block.py
```

```python
import gpu_waterz as wz
labs = wz.segment(aff, [0.2, 0.3, 0.4, 0.5])  # affinity
# stock scores:
labs = wz.segment(aff, [0.8, 0.7, 0.6, 0.5], threshold_mode="score")
```

Missing `src/libws_gpu.so` / `librag_gpu.so` / `libparhac_d.so` raises a RuntimeError that says to run the build script, not a ctypes traceback.

If you already have a waterz worker, the swap is the function call and the threshold unit. If you wanted a new megastack, you are in the wrong repository.
