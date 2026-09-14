# N19 LW: Lance-Williams family (paper parser)

not a 2 Gvox/s number; not 3090 Ti.

Full quotes + HTTP ledger: [papers/lw/REPORT.md](../papers/lw/REPORT.md). PDFs in `papers/lw/`.

**VOI-legal among the 7:** only contact-count UPGMA = waterz `MeanAffinityProvider` (size = face count, missing contacts omitted). Cardinality UPGMA already E13 FAIL (split 2.31-2.44).

| method | atlas | legal? |
|---|---|---|
| single | Kruskal / frozen CC | no |
| complete (CLINK) | **not tested as CLINK** | no (min-of-means != pooled mean) |
| UPGMA \|A\|·\|B\| | E13 | no |
| UPGMA contact n | S4 / ParHAC | **yes** |
| WPGMA = McQuitty | no | no (½+½) |
| centroid / UPGMC | no | no meaning on graph RAG |
| median / WPGMC | no | no meaning; inversions |
| Ward | no | no meaning (needs μ in R^k) |

RNN-HAC (Bruynooghe 1977 NUMDAM, HTTP 200) is an algorithm for *reducible* linkages, not a linkage. S3 is reducible (P0b). Exact RNN-on-S3 would preserve the dendrogram; N19 H3 still unimplemented. No GPU RNN-HAC paper 2020-2026 found (arXiv NN-chain∩GPU∩clustering = 0 hits). ParChain/RAC are CPU. 2507.20047 Ward/centroid NC is Euclidean point-set; does not transfer to this RAG.
