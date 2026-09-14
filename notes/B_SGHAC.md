# B: static SubgraphHAC good-merge (CPU, rag.npz, T=0.3)

Date: 2026-09-06. No CUDA.

TeraHAC Def. 1 / DynHAC Def. 2: merge while
`mean >= max(best[u], best[v]) / (1+ε)`, ε=0.10.

- 14 rounds, 1 865 946 merges, nseg 309 454, 3.8 s
- VOI 0.1550 / **7.5519** FAIL (merge giant; split limit 0.4738, merge 0.2611)
- work_cut vs locked E6s T=0.3 ≈ 2.01x, but VOI failed
- `close_class=True`, `start_cuda=False`

Saturated CAD RAG (N3): good-merge is a giant. Class closed.
