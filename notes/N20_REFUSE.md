# N20 refuse stamps (quote-backed, not coded)

N20 atlas hole / skip-class. Not a throughput claim. No 2.16.

| ID | Reason | Quote / pin |
|---|---|---|
| N20_WARD | no_meaning_on_rag | 2507.20047 defines Ward/centroid via mu(X) in R^k. Fragments have no feature vectors. |
| N20_CENTROID | no_meaning_on_rag | same paper; spatial bbox centroid would cluster geometry, not contact-mean |
| N20_MEDIAN | no_meaning_on_rag | Lance-Williams median is not S3 |
| N20_CHAMFER | illegal statistic | "None of the variants of Chamfer-linkage satisfy reducibility." `papers/fetched/chamfer_2602.10444.pdf` |
| N20_CUML | Kruskal-class | cuML 26.08 linkage={"single"} only |
| N20_PANDORA | Kruskal-class | GPU MST dendrogram 2401.06089 |
| N20_CUSLINK | Kruskal-class | cuSLINK 2306.16354 |
| N20_GPUUPGMA | memory refuse | N in 1000..10000 dense matrix; val has 2.17M fragments |
| N20_SMELKO | wrong domain | Mahalanobis-average on cytometry points; Euro-Par chapter not obtained |
| N20_GSHAC | serial exact heap class | workstation fastcluster on sparse geo graph |
| N20_CARD_UPGMA | do not reopen E13 | cardinality UPGMA already split 2.31-2.44 |
| N20_FORUM | zero UPGMA | t24259 document clustering; t8876 MST; t191439 HDBSCAN |

JSON: `data/cache/N20_REFUSE.json`.
