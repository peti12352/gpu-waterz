# B: 24 GB fit, first measured lever

Date: 2026-09-06. Not a 2 Gvox/s claim.

## Papers (read the sections, not the abstracts)

- ParHAC arXiv:2206.11654 **p.6 §2.3** and **p.22-23 MultiMerge**, **p.23 Affinity/SCCsim**, **D.3 p.30-31 CPAM vs HT**. PDF: `papers/parhac_dhulipala2022.pdf`.
- DynHAC arXiv:2501.07745 **§1-§3**, Def. 2 good merge. PDF: `papers/dynhac_yu2025.pdf`.
- TeraHAC SubgraphHAC good-merge + arbitrary partition correctness (already in `papers/terahac_dhulipala2023.txt`).
- d1 stage maxima: `scripts/d1_mem.py` lines 176-181. SOURCES S32 (corrected), S39, S40.

The 7-11x quote is Affinity/SCCsim GBBS vs clustered-graph. It is **not** a ParHAC work factor. E2 CSR already is MultiMerge (3.86x, parent-identical). Layer 0 is ~20 660 merges/outer: not the ε=0.01 "few vertices" regime. GPU StarMerge v2 is not started. Keep N6 on agglomeration work.

## W2 lever 1 (measured)

`cudaFree(aff)` immediately after `k_flow` in `e9c_watershed` (host path). `watershed_gpu_e9_d` unchanged: caller aff is still live for RAG.

`data/cache/b_w2_free_aff.json`

- nfrag 2175400, `wz_fragments.npy` **array_equal=True**, leaked 0
- val peak **3.536 -> 3.033 GiB** (saved 0.503 GiB = val aff)
- linear pred at 2.16 Gvox **36.40 GiB** (was 42.22)

Still over 23 GiB usable. Remaining sound levers (unmeasured): uint32 BFS queue, vcount-per-root, in-place e9c scan. After those, overall peak is still `max(WS', RAG 21.74, AGG 23.98)`.

## Work axis

`data/cache/a_clustered_cpu.json`: keep_n6_on_work=True, start_gpu_starmarge_v2=False.
