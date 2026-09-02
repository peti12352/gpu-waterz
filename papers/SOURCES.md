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

## S10 — GALA / Nunez-Iglesias 2013: **not read**

PLOS ONE 8(8):e71741 is the real GALA paper. The file that landed as
`nunez_iglesias_gala_2013.pdf` was arXiv:1303.5942 (Brassard et al.,
unrelated). Deleted. Do not cite GALA. VOI definition we use is S5
+ TASK.md, not that paper.

## S11 — not obtained (do not cite as read)

| Attempt | Status |
|---|---|
| Meilă VOI | not on disk |
| Zlateski MIT thesis | dspace hung; we have the **forked waterz source**, not the thesis |
| Turaga 2010 affinity nets | not on disk. A file briefly named `turaga_affinity_2010.pdf` was Tschopp ETH TR arXiv:1509.03371 — deleted, not cited |
| Nunez-Iglesias GALA | see S10 |
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

---

## S13 — TeraHAC (Dhulipala, Lee, Łącki, Mirrokni, arXiv:2308.03578)

Saved: `papers/terahac_dhulipala2023.txt`. Official SM:
`papers/repos/graph-mining/in_memory/clustering/hac/{terahac.cc,terahac_internal.h,subgraph/}`.

Reducible (paper Def. 1 / Benzécri):
`f(x, y∪z) ≤ max(f(x,y), f(x,z))`. Contact-mean S3 is a convex combination
(S3), hence reducible. Lemma 3 uses this for `wmax(v) ≤ max(wmax(v1),wmax(v2))`.

Good merge (paper Def. 2):
`max(wmax(u),wmax(v)) / min(M(u),M(v),w(uv)) ≤ 1+ε`
with `M(singleton)=∞` and `M(u∪v)=min(M(u),M(v),w(uv))`.

`terahac.cc` (read): `ClusteredGraph<AverageLinkageWeight>` (UPGMA — not our
accuracy path; E13 killed UPGMA), default `ε=0.1`,
`pruning_threshold = linkage_threshold/(1+ε)`, loop =
`SizeConstrainedAffinity` then `ApproximateSubgraphHacWrapper`,
`size_constraint = max(n/100, 1e6)`. Partitioning is arbitrary for
correctness (paper §4).

## S14 — GASP (Bailoni et al., arXiv:1906.11713)

Saved: `papers/gasp_bailoni2019.txt`. Repo: `papers/repos/GASP`.
AbsMax linkage ≡ Mutex Watershed (Wolf). Empirical `O(N log N)`.
AbsMax does not update neighbor weights after merge.

## S15 — Mutex Watershed Alg. 2 (Wolf TPAMI 2020)

Already `papers/mutex_watershed_wolf2020.txt`. Repo clone:
`papers/repos/mutex-watershed` (sciai-lab). Sort `|w|` desc; attractive
merge if not mutex-blocked; repulsive adds a mutex. Long-range cues in
the paper are CNN; TASK allows deriving offsets internally. Voxel MWS
on 2.16 Gvox does not fit leftover VRAM — mutex stays on the RAG.

## S16 — Funke et al. TPAMI 2019 (10.1109/TPAMI.2018.2835450)

Saved: `papers/funke_tpami2019.txt` (IEEE/HTML extract; arXiv:1709.02974 is the
2017 structured-loss preprint, not this agglomeration writeup).

> "an efficient O(n) agglomeration scheme based on quantiles of predicted affinities"

> "We initialize the edge scores f(e) for e in E0 with one minus the maximum
> affinity between the fragments linked by e and update them using a quantile
> value of scores of the initial edges under e."

> "We discretize the initial scores f0 into k bins, evenly spaced in [0,1].
> First, a bucket priority queue for sorting edge scores can be used, providing
> constant time insert and pop operations. … overall worst-case complexity of
> O(n). With k = 256 bins, we noticed no sacrifice of accuracy in comparison
> to the non-discretized variant."

> "the new edge score after merging … is always greater than or equal to its
> previous score. We can therefore mark e as stale …"

This is quantile + BinQueue, not MeanAffinity + heap. O(n) = one serial pop
per merge. Not a G6 path (H0 heap 17.7 s).

waterz `BinQueue.hpp`: FIFO per bin. `HistogramQuantileProvider.hpp`:
`InitWithMax=true` then Q-th percentile after `notifyEdgeMerge` adds hists.

## S17 — It's Hard to HAC (Abboud et al., arXiv:2404.14730, ICALP 2024)

Saved: `papers/hard_hac_2024.txt`.

> "we prove that average linkage HAC likely cannot be parallelized even on
> simple graphs by showing that it is CC-hard on trees of diameter 4"

> "If average linkage HAC can be solved by a combinatorial algorithm in
> O(n^{3/2-ε}) time for any ε>0, then the Combinatorial BMM Conjecture is false."

Possibility side: near-linear when dendrogram height is small; NC on paths.
Forbids exact UPGMA as a parallel G6 plan. Irrelevant once we stay (1+ε) or
change the statistic.

## S18 — Bateni et al. Affinity Clustering (NeurIPS 2017) — skip as product

Saved: `papers/bateni_affinity2017.txt` (PDF).

> "we propose affinity, a novel hierarchical clustering based on Borůvka’s
> MST algorithm"

Borůvka MST hierarchy ≡ single-linkage. X0-class on this RAG (frozen CC
giant-componented). Not an accuracy path. Not cloned as product code.

## S19 — cuSLINK (Nolet et al., arXiv:2306.16354) — skip as product

Saved: `papers/cuslink_2023.txt`.

> "computing an MST using a variant of Borůvka’s classic parallel algorithm"

Exact single-linkage on GPU. Same kill as X0 / GASP-max. Not cloned.

## S20 — affogato + nifty (GASP mean path)

Cloned: `papers/repos/affogato` (`compute_mws_clustering` —
AbsMax/MWS, already M16 FAIL).
Cloned: `papers/repos/nifty` `get_GASP_policy` default `linkage_criteria='mean'`,
`edgeSizes` weighting. Unsigned mean ≡ waterz heap. S21 uses signed mean
with repulsive only on `mean<0.5`, not AbsMax.

## S21 — Zlateski & Seung 2015 (arXiv:1505.00249)

Saved: `papers/zlateski_2015.txt`.

> "we visit all the edges of the watershed basin graph in non-decreasing
> order and merge the corresponding clusters based on the introduced predicate."

> "Λ(C1,C2) = true if d_{C1,C2} ≥ τ(min{S(C1),S(C2)}) false otherwise"

> "The value of τ(s) represents the maximal saliency allowed between a
> cluster of size s and any adjacent cluster."

> "For example, when ω is constant the algorithm will tend to aggressively
> merge segments smaller than the given constant."

Their `d` is min-edge saliency (single-linkage) on the basin graph, not
contact-mean. We run the same control flow on contact-mean (waterz S3)
and skip max-face (no per-face list on `rag.npz`). Product shape is
batched Kruskal, not their serial pass.

## S22 — Felzenszwalb–Huttenlocher / LV (quoted via Baltaxe et al. arXiv:1504.06507)

Saved: `papers/fh_lv_baltaxe_2015.txt`. Original IJCV 2004 paywalled;
this restates the MInt predicate.

> "Int(C_i) = max_{e ∈ MST(C_i)} w(e)"

> "MInt(C_i,C_j) = min_{x∈{i,j}} (Int(C_x) + T(C_x))"

> "T(C_i) = K / |C_i|"

> "if (w(e_q) ≤ MInt(C_i,C_j)) ∧ (C_i ≠ C_j) then merge"

We add a hard cut `w ≤ 1-T` so the graded affinity threshold still
exists. Zlateski (S21) claim SDSL beats FH on EM; we still grade FH.

## S23 — Nock & Nielsen SRM (TPAMI 2004)

Saved: `papers/srm_nock_2004.txt` (Sony CSL PDF extract).

> "it is enough to give a merging predicate and an order to test region
> mergings, to completely define our segmentation algorithm."

> "|(R − R') − E(R − R')| ≤ g √( (1/(2Q)) (1/|R| + 1/|R'|) ln(1/δ) )"

Scale Q is the only knob. On the RAG we take a node attribute R = max
incident contact-mean, size-weighted after merge, and still require
mean > T.

## S24 — Soille α–ω / HIGRA constrained connectivity (TPAMI 2008)

Saved: `papers/soille_higra.txt` (HIGRA docs, which quote Soille 2008).

> "a set of vertices X is α-connected, if for any two vertices i and j
> in X, there exists a path from i to j in X composed of edges of
> weights lower than or equal to α."

> "the α−ω-connected components of the graph are the maximal α′-connected
> sets of vertices with a range lower than or equal to ω, with α′≤α."

> "Let X be a set of vertices, the range of X is the maximal weight of
> the edges linking two vertices inside X." (strong connection)

S26 implements the strong-connection Kruskal: union if `w=1-mean ≤ 1-T`
and the component edge-weight range stays ≤ ω. HIGRA is not cloned
unless S26 is reached; we use a 150-line UF.

## S25 — Lu, Zlateski, Seung 2021 (arXiv:2106.10795) — skip as product

Saved: `papers/lu_zlateski_2021.txt`.

> "The trick is to delay merge decisions for regions that touch chunk
> boundaries, and only complete them in a later round after the regions
> are fully contained within a chunk."

> "In (Zlateski and Seung 2015) the max affinity linkage criterion was
> augmented by size thresholds."

> Algorithm 1 is a generic heap agglomerator: pop max affinity, stop
> when A < T, update the RAG.

Exact distributed S3. Same sequential dependence as waterz. Y1 on a
single 3090 Ti. We take only the size-threshold *idea* (Z25/C28), not
the distributed heap.

## S26 — Lu 2021 Algorithm 2 (chunk freeze; implementable, not the product)

Same file as S25 (`papers/lu_zlateski_2021.txt`). S25 skips the
*distributed heap product*. This pin is Algorithm 2 itself: interior
merges in a chunk are exact if boundary-touching supernodes are frozen;
residuals go to the parent chunk. Result ≡ global heap.

> "The modified algorithm suited for chunked input based on this idea
> is presented in Algorithm 2."

Algorithm 2 `AGGLOMERATE CHUNK(G_C, B_C, T)` (quoted steps):

> "H ← Priority queue from G_C"
> "F ← B_C"
> "if u ∈ F or v ∈ F then  Freeze nodes and edges
>  F ← {u} ∪ F; F ← {v} ∪ F; add edge {u,v} into G_F; continue"

> "B_C … contains all the boundary supervoxels touching the artificial
> chunk boundaries. These supervoxels may be split among chunks, and
> the edges associated with them in G_C may be incomplete."

> "We call these segments “frozen segments” and keep track of them in
> a set F. At the beginning, F = B_C."

> "if neither u nor v is in F, u and v are mutual nearest neighbors
> and we can agglomerate them immediately. Otherwise it means whether
> we can agglomerate u and v depends on agglomeration decisions
> involving segments in F. We cannot merge u and v before we resolve
> those pending agglomerations."

> "In the end, we return the partial dendrogram, and the frozen edges
> for later steps."

> "Algorithm 2 is the corner stone of our distributed clustering"

B34 uses this freeze + double-tile residual, on one GPU. Depth =
levels (≤8), not intra-block pops, if each tile is one sort+UF.

## S27 — lsd / daisy / Volara: skip as product

`funkelab/lsd` tutorial `03_agglomerate_blockwise.py` +
`workers/agglomerate_worker.py` (fetched 2026-09-01). Production
blockwise waterz: serial `waterz` per daisy block, edges into Mongo.

> daisy.run_blockwise(total_roi, read_roi, write_roi, …)
> read_roi = block_size grown by context; write_roi = block_size

> lsd.agglomerate_in_block(affs, fragments, rag_provider, block,
>     merge_function=waterz_merge_function, threshold=1.0)

> MongoDbGraphProvider(…, edges_collection='edges_' + merge_function)

merge_function includes `'mean': 'OneMinus<MeanAffinity<…>>'` — stock
waterz S3 inside each block. Context exists so the block sees
overlapping fragments; it does **not** replace the heap with a
shallow GPU kernel. Same class as Lu-as-distributed-product (S25):
Y1 serial waterz × blocks + a DB. Do not run daisy. Block/context
numbers are experiment knobs, not a G6 algorithm.

## S28 — relative-contact (our definition; not a paper)

Not claimed as published. Zlateski/Seung size thresholds (S25 quote)
and the FlyWire-style dumbbell veto are *absolute* caps
(`min(S)<S0` / `area≥a`). Those were measured: A27 `area≥16` is still
a 75% giant at T=0.2; Z25 `min(S)<S0` blocks the giant **and** the
large–large merges waterz needs.

Untried predicate, one UF, sizes update:

```
merge iff mean > T  AND  area ≥ γ · min(S1, S2)^α
```

α = 2/3 is surface scaling (contact vs body surface); α = 1/2 is the
other sweep point. Thin tunnels between large bodies fail the
area test; large shared faces pass. Batched frozen-size regrade is
the G6 shape (B≤32). No repo cloned.

## S29 — ParHAC depth vs E6 10 ms (budget correction)

ParHAC (Dhulipala, Łącki, Lee, Mirrokni, arXiv:2206.11654):

> "we provide a (1+ε)-approximation algorithm for this problem on m
> edge graphs using Õ(m) work and poly-logarithmic depth."

> "obtaining an average-linkage exact HAC algorithm with
> poly-logarithmic depth is not possible under standard
> complexity-theory assumptions (i.e., it is a P-complete problem)."

E6 SKIP in [scripts/e6_persistent.py](../scripts/e6_persistent.py) used a
**10 ms** floor (`939 × 20 µs`) and `inners > 400`. That 10 ms is not a
TASK.md number. TASK speed is 2 Gvox/s on 2.16 Gvox ≈ 1.08 s e2e; val
proxy 180 Mvox ≈ 90 ms e2e; device WS+RAG+extract ≈ 40 ms → **val AGG
budget ≈ 50 ms**. P0a: `layer_mean=40647`, `layer_max=2367715`, CPU
inner p50=181 µs (the 20 µs GPU floor was a guess). A fused device
ContractLayer of the locked ε=0.08 is remaining TASK work, not a new
agglomerator.

## S30 — It's Hard to HAC height escape

Abboud et al., arXiv:2404.14730 (already S17). Extra pin for Track B:

> "we prove that average linkage HAC likely cannot be parallelized even
> on simple graphs by showing that it is CC-hard on trees of diameter 4."

> "There is an implementation of the nearest-neighbor chain algorithm
> for average linkage HAC that runs in O(m · h log n) time where h is
> the height of the output dendrogram."

P0w measures h / parallel-chain rounds on the val RAG. P0i height=9 was
a 50k-edge sample — not the full graph. No ParChain clone.

## S31 — leftover after relative-contact (not L33)

L33 leftover is SDSL large–large `mean>T` (measured: that leftover *is*
the giant). R32 leftover is the complement of S28:

```
mean > T  AND  area < γ · min(S1, S2)^α
```

Thin contacts. R32 γ=0.10 α=0.67: 0.2 merge 0.3451 PASS, split 0.427
FAIL (0.007 merge slack, 0.029 split gap). P0v counts that residual.
Not another γ=0.10 serial grade. γ=0.05 α=0.67 was a P0t all-four
candidate and was never VOI-graded.

## S32 — ParHAC clustered-graph MultiMerge (impl, not a new agglomerator)

`papers/parhac_dhulipala2022.txt` (arXiv:2206.11654) §2.3 and appendix:

> "many of these rounds only merge a small number of vertices, and leave
> the majority of the edges unaffected, and so updating the entire graph
> each round can be highly wasteful."

> "update the underlying similarity graph in work proportional to the
> number of merged vertices and their incident neighbors, rather than
> proportional to the total number of edges in the graph."

> "new implementations using the compressed clustered graph, which only
> recompute the weights of edges incident to a merge, are between 7–11x
> faster across the graphs we evaluate."

Official API already cloned:
`papers/repos/graph-mining/in_memory/clustering/parallel_clustered_graph.h`
`StarMerge` / `SubgraphMerges`. E6r does the opposite (full
`compact_and_combine` every inner). E6s ports this contract onto the
locked ε=0.08 device ContractLayer. Not a new agglomerator.

## S33 — per-red prefix accept (ParHAC Alg. 1)

`papers/parhac_dhulipala2022.txt` Algorithm 1 lines 10–11:

> "Let Tr be triples from T with first component equal to r."
> "For each r ∈ R, select the first prefix of Tr, in which the total
> size of blue vertices exceeds ε|r|"

Reds are independent. E6r `k_accept_serial<<<1,1>>>` is a sequential
scan of all proposals. E6s-a is one thread (or block) per red, same
`take=k+1` as `parhac_paper_cpu`.

## S34 — Hornet blocked adjacency (idea only)

Busato, Green, Bombieri, Bader, *Hornet: An Efficient Data Structure
for Dynamic Sparse Graphs and Matrices on GPUs* (2018). Abstract:

> "Hornet can grow to very large sizes without requiring any data
> re-allocation or re-initialization during the whole dynamic evolution
> of data."

Power-of-two adjacency blocks. Steal the idea (~20 lines). Do not add
Hornet / cuSTINGER / Gunrock / cuGraph as a dependency.

## S35 — Tseng–Dhulipala–Shun SPAA 2022 (exact-dynamic HAC)

Saved: `papers/tseng_spaa2022.txt` (arXiv:2205.04956 HTML).

> "assuming the strong exponential time hypothesis, dynamic graph HAC
> requires Ω(n^{1-o(1)}) work per update or query on a graph with n
> vertices for complete linkage, weighted average linkage, and average
> linkage."

> "For average linkage, the bound weakens to Ω(n^{1/2-o(1)}) for
> incremental and decremental algorithms, and the bounds still hold
> when allowing n^{o(1)}-approximation."

Forbids incremental *exact* UPGMA as a G6 plan. Irrelevant once we
stay (1+ε) paper-ParHAC.

## S36 — Affinity / SCC ≡ E5 Borůvka (do not retry)

Already S18. ParHAC §3 (`papers/parhac_dhulipala2022.txt`):

> "In each round of Affinity clustering, each vertex selects its
> heaviest incident edge, and all connected components induced by the
> chosen edges are merged to form new clusters."

> "The SCC algorithm [54] is closely related to the Affinity algorithm,
> and can be viewed as running Affinity with different weight
> thresholds."

≡ E5 one-sided Borůvka. `ACCURACY GATE: FAIL`. Not an E6s path.

Closed this sweep (inspect only; no VOI-fish): Filter-Kruskal /
ECL-MST / data-parallel Kruskal = X0/S19; Quantile / MALA /
HistogramQuantile / MeanMaxK = S16/Q20/Y1; TeraHAC / Lu / daisy =
distributed product; cuSLINK / Mahalanobis GPU HAC = S19 / point-set;
PATHFINDER / RagNet = training, TASK out of scope; R32 leftover /
NN-chain / matching = L36/R36/P0w/B18.

## S37 — CPU `Graph.adj` / `unite_keep` is StarMerge

`src/rac_agg.cpp` `Graph::unite_keep` (also `unite`). Host paper-ε
`paper_unite` calls this. Walks **only** `adj[drop]`, `InsertOrUpdate`
via `lookup`, then `adj[drop].clear()`:

```
for (int eid : adj[drop]) {
    ...
    other = find(other);
    if (other == keep) { e.alive = false; continue; }
    auto it = lookup.find(pair_key(lo, hi));
    if (it == lookup.end()) { e.u = lo; e.v = hi; adj[keep].push_back(eid); ... }
    else { keep_e.sum += e.sum; keep_e.n += e.n; e.alive = false; }
}
adj[drop].clear();
parent[drop] = keep;
```

Official twin: `papers/repos/graph-mining/in_memory/clustering/parallel_clustered_graph.h`
`StarMergeImplementation` maps satellites, copies `2 * neighbors.size()`
incident edges, sorts triples, `MergeWeights`, `InsertTriples` →
`InsertOrUpdate`. Work ∝ merged vertices × degree, not |E|.
E6s never ported this; it full-scans COO every inner.

## S38 — ParHAC Alg. 1 lines 13–14 (G and Gc)

`papers/parhac_dhulipala2022.txt` Algorithm 1:

> "13: Merge vertices in G and Gc based on the pairs from M, updating
> edge weights in Gc."
> "14: Remove edges of Gc that have two red endpoints or weight below TL"

E6s-c did line 14 on a snapshot and skipped line 13 (G stale) →
T=0.2 merge 4.01. Legal leftover is line 13 then 14: StarMerge G
every inner, then filter Gc from the **updated** G.

Dead approximations (do not retry): hash-full-graph + empty-outer-abort;
skip-combine / keep-ratio-without-merge; Gc-extract-without-updating-G.

