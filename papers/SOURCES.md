# Pinned sources

Every claim in `PLAN.md` that is not a TASK.md number or a measurement
we took on greengoblin is pinned here. Quote + location. No paper that
failed to download is cited.

waterz-upstream pin: `funkey/waterz` commit
`a0184d2af2ab3ed044721fb92822fc6ea9cee665` (2025-09-18, PR #21 witty).
Vendored at `src/waterz-upstream/` (`.git` stripped).

---

## S1 — waterz watershed (`basic_watershed.hpp`)

6-neighbour read, OOB → `low`, flow iff `aff == m || aff >= high`,
background iff `m` is not `> low` (i.e. `m <= low`):

```
F negz = (z>0) ? aff[0][z][y][x] : low;
F negy = (y>0) ? aff[1][z][y][x] : low;
F negx = (x>0) ? aff[2][z][y][x] : low;
F posz = (z<(zdim-1)) ? aff[0][z+1][y][x] : low;
F posy = (y<(ydim-1)) ? aff[1][z][y+1][x] : low;
F posx = (x<(xdim-1)) ? aff[2][z][y][x+1] : low;
F m = std::max({negx,negy,negz,posx,posy,posz});
if ( m > low ) {
    if ( negz == m || negz >= high ) { id |= 0x01; }
    ...
}
```

Plateau corners = flow out to a neighbour that does not flow back.
BFS from those corners rewrites each plateau voxel to **one** exit dir
(`to_set`). Scan order is linear index 0..size-1, dir 0..5
(`-z,-y,-x,+z,+y,+x`). This is the order-dependence TASK.md names.

Basin fill is a second BFS along remaining flow bits. IDs start at 1.
`high_bit` marks settled voxels (`types.hpp`: uint32 `0x80000000`,
uint64 `0x8000000000000000`). Final store: `seg_raw[idx] &= traits::mask`.

Matches TASK.md step 1.

## S2 — region graph (`region_graph.hpp`)

Only the three **negative** directions. `id1 != id2` collected;
edges emitted only for `id1 = 1 .. max_segid`, so any pair whose
`minmax` first component is 0 (background) is never inserted.

```
for (int d = 0; d < 3; d++) {
    if (p[d] == 0) continue;
    ID id2 = seg[p[0]-(d==0)][p[1]-(d==1)][p[2]-(d==2)];
    if (id1 != id2) {
        auto mm = std::minmax(id1, id2);
        affinities[mm.first][mm.second].push_back(aff[d][p[0]][p[1]][p[2]]);
    }
}
for (ID id1 = 1; id1 <= max_segid; ++id1)
    for (const auto& p: affinities[id1]) { ... addEdge + addAffinity }
```

File-header comment says "maximum affinity". That comment is **stale**.
Default scoring uses `MeanAffinityProvider::addAffinity`, incremental mean.

Matches TASK.md step 2.

## S3 — mean affinity (`MeanAffinityProvider.hpp`)

```
_meanAffinities[e] = (affinity + mean*n)/(n+1);   // voxel face
_meanAffinities[to] = (fromMean*fromN + toMean*toN)/(fromN + toN);  // edge merge
return true;   // score changed
```

Area = contact-face count. Matches TASK.md "contact-area-weighted mean".

## S4 — agglomeration (`IterativeRegionMerging.hpp`)

Min-heap of scores. Stop when `score >= threshold`. Stale edges rescore
and are asserted non-decreasing (`assert(newScore >= score)`). Merge `b`
into `a`: exclusive neighbour edges `moveEdge` to `a`; shared neighbours
keep the **cheaper** edge, `notifyEdgeMerge` the other into it, delete
the expensive one, mark survivor stale.

```
if (score >= threshold) break;
...
if (_edgeScores[neighborEdge] > _edgeScores[aNeighborEdge]) {
    notifyEdgeMerge(neighborEdge, aNeighborEdge);
    removeEdge(neighborEdge); _deleted[neighborEdge] = true;
    if (edgeStatisticChanged) _stale[aNeighborEdge] = true;
} else { /* swap: keep neighborEdge, delete aNeighborEdge */ }
```

`OneMinus` (`Operators.hpp`): `return 1.0 - x`.
Default scoring string (`_agglomerate.py`):
`OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>`.
`discretize_queue == 0` → `PriorityQueue` = `std::priority_queue` with
`std::greater` (min-heap). Nonzero → `BinQueue<N>`.

**Threshold units.** `mergeUntil` compares the **score** (`1 - mean`) to
`threshold`. TASK.md: "waterz thresholds are *scores*; we quote
**affinity** thresholds throughout." So aff_thr `0.3` means merge while
`mean > 0.3`, i.e. waterz `thresholds=[0.7]`. Confirm against the
shipped `baseline/run_baseline.py` in E0 before any VOI run. Do not
guess past that.

## S5 — VOI (`evaluate.hpp`)

Ignore `gt == 0`. `s` = segmentation, `t` = GT.

```
voi_split = H_st - H_t;   // H(s|t) = H(seg|gt)
voi_merge = H_st - H_s;   // H(t|s) = H(gt|seg)
```

Matches TASK.md. `waterz.evaluate` takes uint64 volumes (`evaluate.pyx`).

## S6 — layout wrap (`frontend_agglomerate.cpp` / `agglomerate.pyx`)

Python `affs.shape = (3, Z, Y, X)` is passed as
`initialize(affs.shape[1], affs.shape[2], affs.shape[3], ...)`
and wrapped `boost::extents[3][width][height][depth]` with
`width=Z, height=Y, depth=X`. C++ names `zdim=aff.shape()[1]` etc.
agree with TASK.md `[3,Z,Y,X]`.

## S7 — Wolf et al., Mutex Watershed (arXiv:1904.12654; TPAMI 2020
DOI 10.1109/TPAMI.2020.2980827). PDF: `mutex_watershed_wolf2020.pdf`.
ECCV 2018 short form: arXiv:1705.08369, `mutex_watershed_wolf2018.pdf`.

> "sort all edges E, attractive or repulsive, by their absolute weight
> in descending order into a priority queue."

> "an algorithm that is empirically no more expensive than a MST
> computation"

> "the time complexity of merge(i, j) and connected(i, j) is O(α(V))
> … total runtime complexity is dominated by the initial sorting of
> the edges O(E log E)"

Mutex is **frozen-weight Kruskal plus repulsive constraints**. It is
not mean-affinity agglomeration (means do not update). TASK.md allows
it as an algorithm *if* the VOI gate holds. Input stays 3 channels;
"deriving extra offsets internally is fine." Using mutex as the
*primary* agglomerator is a VOI bet, not the first design.

## S8 — Yeghiazaryan, Gabrielyan, Voiculescu 2024.
"Parallel Watershed Partitioning: GPU-Based Hierarchical Image
Segmentation." arXiv:2410.08946. PDF: `parallel_watershed_gpu_2024.pdf`.

> "Our watershed algorithms attain competitive execution times in both
> 2D and 3D, processing an 800 megavoxel image in less than 1.4 sec."

800e6 / 1.4 = 0.57 Gvox/s for an **intensity** watershed + waterfall,
not affinity-RAG + mean-affinity agglomeration. Plateau resolution is
named as the expensive step:

> "The most expensive step of PRUF … is the resolution of non-minimal
> plateaux."

Use as a GPU watershed *pattern* (path compression / union-find), not
as a substitute for TASK.md agglomeration.

## S9 — Funke et al. 2017 / J Neurosci 2019? PDF downloaded as
`funke_structured_loss_2017.pdf` from arXiv:1709.02974
("Large Scale Image Segmentation with Structured Loss based Deep
Learning for Connectome Reconstruction"). This is the pipeline
context: CNN affinities → waterz-style agglomeration. Not an algorithm
we implement. Do not treat it as a speed or VOI number.

## S10 — Nunez-Iglesias, Kennedy, Parag, Shi, Chklovskii 2013.
"Machine learning of hierarchical clustering to segment 2D and 3D
images." PLOS ONE. arXiv:1303.5942. PDF: `nunez_iglesias_gala_2013.pdf`.
GALA: learned agglomeration on a RAG, VOI as the connectomics metric.
Confirms VOI-split / VOI-merge as the field's grading language.
Does not give us a GPU algorithm.

## S11 — not obtained (do not cite as read)

| Attempt | Status |
|---|---|
| Meilă VOI (`compare-colt.pdf`) | curl still running / not confirmed on disk |
| Zlateski MIT thesis | dspace hung; waterz README cites bitbucket.org/poozh/watershed and TuragaLab/zwatershed — we have the **forked source**, not the thesis |
| Turaga 2010 affinity nets | not confirmed downloaded |
| Tarball `README.md` (532 MB, "verified line-by-line") | **not yet downloaded**. E0. Until then, S1–S5 + TASK.md are the spec. |

## S12 — greengoblin, 2026-08-30 15:26 local (measured, not guessed)

```
HOST=greengoblin
GPU=NVIDIA GeForce RTX 5090
memory.total=32607 MiB
compute_cap=12.0
driver=580.159.03
nvidia-smi CUDA=13.0
nvcc=/usr/local/cuda/bin/nvcc  12.8.61
gcc=13.3.0
python3=/usr/bin/python3  3.12.3
uv=/home/v/.local/bin/uv
RAM=125 GiB
disk=/  3.6T  1.4T free
path=/home/v/proj/petya/waterz   (empty at inspect)
```

GPU was **busy**: PID 1880797 `python3 lego/verify/gate3_pass_at_k.py`,
~8850 MiB, ~69% util, 3h31m elapsed. Free VRAM ~23200 MiB. Do not
kill that job. Val volume fits in the free slice; 2.16 Gvox bench
needs the card idle or we wait.

This card is **not** the graded card. TASK.md: "the reported number
must come from a 3090 Ti."
