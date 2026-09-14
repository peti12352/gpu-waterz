# Citations

Papers and code this repo used, and what we concluded. Local PDFs and
clones live under `papers/` on the machine that fetched them (not in git).

Vendored waterz: `funkey/waterz` commit
`a0184d2af2ab3ed044721fb92822fc6ea9cee665` (2025-09-18), at
`src/waterz-upstream/`. Watershed / RAG / mean / heap / VOI semantics
are that source plus the affinity-vs-score table in [notes/lab.md](../notes/lab.md).

## Used in the product

Funke, J., et al. Large scale image segmentation with structured loss
based deep learning for connectome reconstruction. *IEEE TPAMI*, 2019.
DOI 10.1109/TPAMI.2018.2835450. Also arXiv:1709.02974.
CNN affinities, then waterz-style agglomeration.

Dhulipala, L., Blelloch, G. E., Shun, J. Hierarchical agglomerative
graph clustering in poly-logarithmic depth. *NeurIPS*, 2022.
arXiv:2206.11654. Code: https://github.com/ParAlg/ParHAC
(1+eps)-heavy matching plus contract. Official impl is CPU (CPAM), not
CUDA. Product agglomeration is this order approximation on the
contact-mean RAG. Dual-eps here: 0.08 for four thresholds, 0.40 for a
single T=0.3 cut. Clustered-graph "small merge" rounds are false on
this RAG at those eps (layer 0 is most of the merges).

Abboud, A., Cohen-Addad, V., Lee, E., Schwiegelshohn, C. It's hard to
HAC. *ICALP*, 2024. arXiv:2404.14730.
Exact average-linkage has no friendly poly-log parallel algorithm
under standard assumptions. Why we approximate merge *order* and keep
the mean statistic.

Liu, D., et al. Cross-dimension affinity distillation for 3D EM neuron
segmentation. *CVPR*, 2024.
CREMI-A val affinities in the bounty tarball (CAD checkpoint).

## Same RAG, different clustering (measured, rejected)

Same cached CREMI-A contact-mean RAG, same VOI grader
(`data/cache/voi_atlas.csv`):

Wolf, S., et al. The mutex watershed and its superpixelation of images.
*IEEE TPAMI*, 2020. DOI 10.1109/TPAMI.2020.2980827. arXiv:1904.12654.
ECCV 2018 short: arXiv:1705.08369.
Frozen-weight Kruskal plus repulsive constraints. Means do not update.
Under-merge on this RAG (split VOI ~0.9-2.1).

Bailoni, A., et al. A generalized framework for agglomerative clustering
of signed graphs applied to instance segmentation. arXiv:1906.11713.
GASP. AbsMax path is mutex-class. Mean path in nifty/affogato is unsigned
mean.

Zlateski, A., Seung, H. S. Image segmentation by size-dependent single
linkage clustering of a watershed basin graph. arXiv:1505.00249.
Size-capped single-linkage. Blocks waterz large-large merges.

Felzenszwalb, P. F., Huttenlocher, D. P. Efficient graph-based image
segmentation. *IJCV*, 2004. (We used Baltaxe et al. arXiv:1504.06507
as the readable quote.)
Cannot hit both VOI halves on this RAG.

Nock, R., Nielsen, F. Statistical region merging. *IEEE TPAMI*, 2004.
Giants and under-merge.

Soille, P. Constrained connectivity for hierarchical image partitioning
and simplification. *IEEE TPAMI*, 2008. (HIGRA docs quote the rule.)
Not contact-mean.

Lu, R., Zlateski, A., Seung, H. S. A connectomic reconstruction method
for EM. arXiv:2106.10795.
Exact mean over chunks by freezing anything that touches a fake
boundary (Algorithm 2). A freeze that is not Alg 2 produces different
parents than ParHAC. Spatial Alg 2 four-T PASSes and is still ~6645 s
CPU with millions of residual edges.

Yeghiazaryan, V., Gabrielyan, G., Voiculescu, I. Parallel watershed
partitioning. arXiv:2410.08946.
GPU *intensity* watershed + waterfall, ~0.57 Gvox/s on 800 Mvox. Not
affinity-flow plus mean RAG. Plateau resolution is their expensive
step; ours is list-compress after basins, not BFS.

Dhulipala, L., et al. TeraHAC. arXiv:2308.03578.
SubgraphHAC / good-merge on a partitioned graph. Control-flow is not
our ParHAC matching on one RAG.

Yu, S., Dhulipala, L., Łącki, J., Parotsidis, N. DynHAC. arXiv:2501.07745.
Dynamic TeraHAC.

## Average-linkage relatives

Murtagh, F., Contreras, P. Algorithms for hierarchical clustering: an
overview. *WIREs Data Mining*, 2012.
Of the classical Lance-Williams list, only *contact-count* UPGMA
matches waterz. Complete, WPGMA/McQuitty, centroid, median, and Ward
are different statistics. Cardinality UPGMA (ParHAC
`AverageLinkageWeight`) already failed VOI here.

Bruynooghe, M. Méthodes nouvelles en classification automatique.
NUMDAM, 1977/78. Reciprocal-nearest-neighbour HAC is a *schedule* for
reducible linkages, not an 8th linkage. Exact RNN on this S3 RAG hit a
5000-round cap and failed four-T.

Garg, V., et al. Reciprocal agglomerative clustering. arXiv:2105.11653.
Same RNN class.

Dhulipala et al. SeqHAC. arXiv:2106.05610. Complete-link under-merges
here (split ~1.3-1.5).

Nolet, C., et al. cuSLINK. arXiv:2306.16354. GPU single-linkage MST.
Kruskal-class.

Tseng, T., Dhulipala, L., Shun, J. Parallel batch-dynamic hierarchical
clustering. *SPAA*, 2022. arXiv:2205.04956. Exact-dynamic HAC hardness.
Not a CUDA closer for contact-mean.

Bateni, M., et al. Affinity clustering. *NeurIPS*, 2017. And later
"parallel HAC for low-height dendrograms" (arXiv:2507.20047): Ward /
centroid in R^k, not ParHAC clustered-graph.

Chamfer linkage. arXiv:2602.10444. None of the variants are reducible.

GSHAC. arXiv:2604.11656. Serial exact heap on a sparse geographic graph.

ParChain. arXiv:2106.04727. CPU point-set HAC.

GPU-UPGMA (Chang et al., *Concurrency Computat. Pract. Exper.*, 2015,
DOI 10.1002/cpe.3355). Dense N x N, N ~ 1e3-1e4. Val has 2.17e6
fragments.

RAPIDS cuML 26.08 `AgglomerativeClustering`: `linkage={"single"}` only.

## Software we cloned locally (not vendored)

https://github.com/ParAlg/ParHAC
https://github.com/google/graph-mining (TeraHAC / clustered-graph)
https://github.com/constantinpape/nifty and affogato (GASP mean path)
https://github.com/sciai-lab/mutex-watershed
https://github.com/funkelab/lsd (daisy agglomerate worker shape)
https://github.com/rapidsai/cuvs (single-linkage, not this RAG)

GALA (Nunez-Iglesias et al., *PLOS ONE* 2013, e71741) was not read.
An arXiv:1303.5942 PDF that landed under that name is a different paper.

Meilă's VOI paper, Zlateski's MIT thesis, and Turaga 2010 were not
obtained. VOI is waterz `evaluate.hpp`: split `H(seg|gt)`, merge
`H(gt|seg)`, skip `gt==0`.
