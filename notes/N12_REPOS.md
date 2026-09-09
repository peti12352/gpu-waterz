# N12 cloned trees

Source, not abstracts. `papers/repos/`. GPU? from `find -name '*.cu'`.

No public `.cu` for ParHAC / TeraHAC / DynHAC / SubgraphHAC. Closest GPU HAC is cuVS single-linkage MST. Do not email Funke or Wolf. Dhulipala/Yu 8-liner only if this still holds (it does).

## graph-mining

- path: `papers/repos/graph-mining`
- license: Apache-2.0
- GPU? no (zero `.cu`)

Quoted loop (`in_memory/clustering/hac/parhac.cc` bucket `W_max/(1+ε)`):

```
if (max_weight == 0 || max_weight < linkage_threshold) break;
auto [num_merged, inner_rounds] = ProcessHacBucketRandomized(
    clustered_graph, max_weight / (1 + epsilon), epsilon);
```

Quoted loop (`parallel_clustered_graph.h` `StarMergeImplementation` — CPU compact analog: sort **merge triples**, not the live edge list):

```
parlay::integer_sort_inplace(make_slice(merge_seq), get_key);
```

Quoted loop (`hac/subgraph/approximate_subgraph_hac.cc` `GetGoodEdge` — sequential, **not** GPU):

```
std::tie(node_a, node_b, goodness_ab) = subgraph->GetGoodEdge();
if (goodness_ab == ApproximateSubgraphHacGraph::kDefaultGoodness) break;
```

## ParAlg/ParHAC

- path: `papers/repos/ParHAC`
- license: no LICENSE at repo root; bundled parlaylib Apache-2.0
- GPU? no (zero `.cu`)

Quoted loop (`examples/parhac/parhac/ParHac.h` per-blue `parlay::sort_inplace(neighbors)` — **not** global live-edge radix):

```
parlay::sort_inplace(neighbors);
for (size_t j = 0; j < neighbors.size(); ++j) {
  auto[wgh, v] = neighbors[j];
```

`UniteMergeBatched` then `parlay::sort_inplace(make_slice(merge_seq))`.

## DynHAC (`yushangdi/dynamic-hac`)

- path: `papers/repos/dynamic-hac`
- license: Apache-2.0
- GPU? no (zero `.cu`)

CPU dynamic HAC. No unpublished GPU compact in this tree.

## PRUF-watershed

- path: `papers/repos/PRUF-watershed`
- license: Apache-2.0
- GPU? **yes** (`cuda/PRUF3D.cu` and siblings). 6-conn intensity, plateau path-reduction, not affinity RAG.

Quoted loop (`cuda/PRUF3D.cu` `resolve_plateaux` + Find):

```
__device__ unsigned Find(const int *s_buf, unsigned n) {
    while (s_buf[n] != n) {
        n = s_buf[n];
    }
    return n;
}
```

Makefile ships `-arch=sm_89`. N12 off-contract rebuilds `sm_120`.

## cuVS

- path: `papers/repos/cuvs` (sparse checkout `cpp/src/cluster`)
- license: Apache-2.0
- GPU? **yes**, but **frozen MST / single-linkage**, wrong statistic for mean-HAC.

Quoted loop (`cpp/src/cluster/detail/single_linkage.cuh`):

```
detail::build_sorted_mst<value_idx, value_t>(handle,
                                             X.data_handle(),
                                             ...
```

## RAMA

- path: `papers/repos/RAMA`
- license: BSD-3-Clause
- GPU? **yes** (`src/rama_py.cu`, thrust device graphs). Signed GAEC; stop at negative costs. Contraction is thrust gather/sort/`reduce_by_key`, not mean-HAC.

Quoted loop (`include/graph.h` `Graph::contract`):

```
thrust::gather(fwd_tails.begin(), fwd_tails.end(), node_mapping.begin(), mapped_tails.begin());
thrust::sort_by_key(edge_begin, edge_end, fwd_costs.begin());
```

Broken gitlinks under `external/cudaMST/` (empty submodule paths). Ignore.

## Email (unsent)

Public trees still have zero `.cu` for ParHAC / TeraHAC / DynHAC / SubgraphHAC.
Closest public GPU HAC is cuVS/cuSLINK **single-linkage MST** (wrong statistic).
Do not email Funke (vendored waterz `a0184d2`) or Wolf (mutex FAIL on 3-ch aff).

### Recipients (verified 2026-09-07)

Send **To Dhulipala, Cc Yu**. One message. Do not mass-mail the author list.

| Role | Person | Address | Why |
|---|---|---|---|
| **To** | Laxman Dhulipala | `laxman@umd.edu` | UMD faculty page + TeraHAC author line. ParHAC/TeraHAC academic owner. Also Google Graph Mining. |
| **Cc** | Shangdi Yu | `shangdiy@mit.edu` | CV + DynHAC first author. Student researcher on Graph Mining with Łącki. |
| Hold (2nd wave if no reply) | Jakub Łącki | `jlacki@google.com` (papers); `j.lacki@mimuw.edu.pl` (personal page) | Graph Mining HAC owner, Yu’s host. Do **not** Cc on first send: looks like asking Google for unpublished GPU code. |
| Do not | Jessica Shi | `jeshi@mit.edu` (stale) | ParHAC coauthor; now D.E. Shaw. |
| Do not | Julian Shun | `jshun@mit.edu` | Advisor, not the HAC GPU implementer. |
| Do not | Vahab Mirrokni, Jason Lee | Google | Senior / theory. Will not know an unpublished compact kernel. |
| Do not | `laxmandhulipala@gmail.com` | CV personal | Use the faculty address. |

### Copy (send as-is; fill From)

```
From: <your name> <<your email>>
To: Laxman Dhulipala <laxman@umd.edu>
Cc: Shangdi Yu <shangdiy@mit.edu>
Subject: GPU compact for ParHAC / DynHAC — unpublished pointer?

Prof. Dhulipala, Shangdi,

We have a GPU (1+ε)-ParHAC on a 3D contact-mean RAG (connectomics waterz
replacement). Public trees we cloned — google/graph-mining, ParAlg/ParHAC,
yushangdi/dynamic-hac — contain no .cu. StarMerge in
parallel_clustered_graph.h sorts merge triples and inserts into per-center
neighbor tables; it does not radix-sort the live edge list each inner round.
Our bottleneck is the dual of that (COO dirty-scan / rewrite), not live-edge
radix.

Is there an unpublished GPU compact for ParHAC/TeraHAC, or a DynHAC GPU
port, that avoids scanning the live edge list every inner? A pointer is
enough. We are not asking you to run anything or look at our code.

Thank you,
<name>
```

No Funke. No Wolf. Do not attach a repo. Do not mention the bounty $ or 2 Gvox/s.
