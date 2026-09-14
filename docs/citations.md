# Citations

Papers and code this repo used, and what we concluded.

- Vendored waterz: [`funkey/waterz`](https://github.com/funkey/waterz) commit
  [`a0184d2`](https://github.com/funkey/waterz/commit/a0184d2af2ab3ed044721fb92822fc6ea9cee665)
  (2025-09-18), at [`src/waterz-upstream/`](../src/waterz-upstream/).
- Watershed / RAG / mean / heap / VOI semantics are that source plus the
  affinity-vs-score table in [notes/lab.md](../notes/lab.md).
- Same-RAG VOI table: [`data/cache/voi_atlas.csv`](../data/cache/voi_atlas.csv).

## Used in the product

- **Large scale image segmentation with structured loss based deep learning
  for connectome reconstruction.** Funke et al. *IEEE TPAMI*, 2019.
  - [DOI](https://doi.org/10.1109/TPAMI.2018.2835450) ;
    [arXiv:1709.02974](https://arxiv.org/abs/1709.02974)
  - CNN affinities, then waterz-style agglomeration.

- **Hierarchical agglomerative graph clustering in poly-logarithmic depth.**
  Dhulipala, Eisenstat, Lacki, Mirrokni, Shi. *NeurIPS*, 2022. (ParHAC)
  - [arXiv:2206.11654](https://arxiv.org/abs/2206.11654) ;
    code: [ParAlg/ParHAC](https://github.com/ParAlg/ParHAC)
  - (1+eps)-heavy matching plus contract. Official impl is CPU (CPAM), not
    CUDA.
  - Product agglomeration is this order approximation on the contact-mean
    RAG. Dual-eps here: 0.08 for four thresholds, 0.40 for a single T=0.3
    cut.
  - Clustered-graph "small merge" rounds are false on this RAG at those
    eps (layer 0 is most of the merges).

- **It's Hard to HAC Average Linkage!** Bateni, Dhulipala, Gowda,
  Hershkowitz, Jayaram, Lacki. *ICALP*, 2024.
  - [arXiv:2404.14730](https://arxiv.org/abs/2404.14730) ;
    [LIPIcs](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ICALP.2024.18)
  - Exact average-linkage has no friendly poly-log parallel algorithm
    under standard assumptions. Why we approximate merge *order* and keep
    the mean statistic.

- **Cross-dimension affinity distillation for 3D EM neuron segmentation.**
  Xiaoyu Liu et al. *CVPR*, 2024.
  - [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Cross-Dimension_Affinity_Distillation_for_3D_EM_Neuron_Segmentation_CVPR_2024_paper.html) ;
    [DOI](https://doi.org/10.1109/CVPR52733.2024.01056) ;
    code: [liuxy1103/CAD](https://github.com/liuxy1103/CAD)
  - CREMI-A val affinities from the CAD checkpoint.

## Same RAG, different clustering

Same cached CREMI-A contact-mean RAG, same VOI grader
([`voi_atlas.csv`](../data/cache/voi_atlas.csv)):

- **The mutex watershed and its superpixelation of images.** Wolf et al.
  *IEEE TPAMI*, 2020.
  - [DOI](https://doi.org/10.1109/TPAMI.2020.2980827) ;
    [arXiv:1904.12654](https://arxiv.org/abs/1904.12654)
  - ECCV 2018 short: [arXiv:1705.08369](https://arxiv.org/abs/1705.08369)
  - Frozen-weight Kruskal plus repulsive constraints. Means do not update.
  - Under-merge on this RAG (split VOI ~0.9-2.1).

- **A generalized framework for agglomerative clustering of signed graphs
  applied to instance segmentation.** Bailoni et al. (GASP)
  - [arXiv:1906.11713](https://arxiv.org/abs/1906.11713)
  - AbsMax path is mutex-class. Mean path in nifty/affogato is unsigned
    mean.

- **Image segmentation by size-dependent single linkage clustering of a
  watershed basin graph.** Zlateski, Seung.
  - [arXiv:1505.00249](https://arxiv.org/abs/1505.00249)
  - Size-capped single-linkage. Blocks waterz large-large merges.

- **Efficient graph-based image segmentation.** Felzenszwalb, Huttenlocher.
  *IJCV*, 2004.
  - [DOI](https://doi.org/10.1023/B:VISI.0000022288.19776.77)
  - Readable quote used here: Baltaxe et al.
    [arXiv:1504.06507](https://arxiv.org/abs/1504.06507)
  - Cannot hit both VOI halves on this RAG.

- **Statistical region merging.** Nock, Nielsen. *IEEE TPAMI*, 2004.
  - [DOI](https://doi.org/10.1109/TPAMI.2004.110)
  - Giants and under-merge.

- **Constrained connectivity for hierarchical image partitioning and
  simplification.** Soille. *IEEE TPAMI*, 2008.
  - [DOI](https://doi.org/10.1109/TPAMI.2007.70817) ;
    [HIGRA docs](https://higra.readthedocs.io/en/stable/python/constrained_connectivity_hierarchy.html)
  - Not contact-mean.

- **A connectomic reconstruction method for EM.** Lu, Zlateski, Seung.
  - [arXiv:2106.10795](https://arxiv.org/abs/2106.10795)
  - Exact mean over chunks by freezing anything that touches a fake
    boundary (Algorithm 2).
  - A freeze that is not Alg 2 produces different parents than ParHAC.
  - Spatial Alg 2 four-T PASSes and is still ~6645 s CPU with millions of
    residual edges.

- **Parallel watershed partitioning.** Yeghiazaryan, Gabrielyan,
  Voiculescu.
  - [arXiv:2410.08946](https://arxiv.org/abs/2410.08946)
  - GPU intensity watershed + waterfall, ~0.57 Gvox/s on 800 Mvox.
  - Not affinity-flow plus mean RAG. Plateau resolution is their
    expensive step; ours is list-compress after basins, not BFS.

- **TeraHAC: Hierarchical agglomerative clustering of trillion-edge
  graphs.** Dhulipala, Lee, Lacki, Mirrokni.
  - [arXiv:2308.03578](https://arxiv.org/abs/2308.03578)
  - SubgraphHAC / good-merge on a partitioned graph. Control-flow is not
    our ParHAC matching on one RAG.

- **DynHAC: Fully dynamic approximate hierarchical agglomerative
  clustering.** Yu, Dhulipala, Lacki, Parotsidis.
  - [arXiv:2501.07745](https://arxiv.org/abs/2501.07745)
  - Dynamic TeraHAC.

## Average-linkage relatives

- **Algorithms for hierarchical clustering: an overview.** Murtagh,
  Contreras. *WIREs Data Mining*, 2012.
  - [DOI](https://doi.org/10.1002/widm.53)
  - Of the classical Lance-Williams list, only *contact-count* UPGMA
    matches waterz.
  - Complete, WPGMA/McQuitty, centroid, median, and Ward are different
    statistics.
  - Cardinality UPGMA (ParHAC `AverageLinkageWeight`) already failed VOI
    here.

- **Methodes nouvelles en classification automatique de donnees
  taxinomiques nombreuses.** Bruynooghe. *Statistique et analyse des
  donnees*, 1977.
  - [NUMDAM](https://www.numdam.org/item/SAD_1977__2_3_24_0/)
  - Reciprocal-nearest-neighbour HAC is a *schedule* for reducible
    linkages, not an 8th linkage.
  - Exact RNN on this S3 RAG hit a 5000-round cap and failed four-T.

- **Scaling hierarchical agglomerative clustering to billion-sized
  datasets.** Sumengen et al. (RAC)
  - [arXiv:2105.11653](https://arxiv.org/abs/2105.11653)
  - Same RNN class.

- **Hierarchical agglomerative graph clustering in nearly-linear time.**
  Dhulipala, Eisenstat, Lacki, Mirrokni, Shi. (SeqHAC)
  - [arXiv:2106.05610](https://arxiv.org/abs/2106.05610)
  - Complete-link under-merges here (split ~1.3-1.5).

- **cuSLINK: Single-linkage agglomerative clustering on the GPU.** Nolet
  et al.
  - [arXiv:2306.16354](https://arxiv.org/abs/2306.16354)
  - GPU single-linkage MST. Kruskal-class.

- **Parallel batch-dynamic hierarchical clustering.** Tseng, Dhulipala,
  Shun. *SPAA*, 2022.
  - [arXiv:2205.04956](https://arxiv.org/abs/2205.04956)
  - Exact-dynamic HAC hardness. Not a CUDA closer for contact-mean.

- **Affinity clustering: hierarchical clustering at scale.** Bateni et al.
  *NeurIPS*, 2017.
  - [NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/2e1b24a664f5e9c18f407b2f9c73e821-Abstract.html)
  - Later, same line: **Parallel hierarchical agglomerative clustering in
    low dimensions.** Bateni, Dhulipala, Fletcher, Gowda, Hershkowitz,
    Jayaram, Lacki. [arXiv:2507.20047](https://arxiv.org/abs/2507.20047)
  - Ward / centroid in R^k, not ParHAC clustered-graph.

- **Chamfer-linkage for hierarchical agglomerative clustering.** Gowda,
  Fletcher, Bateni, Dhulipala, Hershkowitz, Jayaram, Lacki.
  - [arXiv:2602.10444](https://arxiv.org/abs/2602.10444)
  - None of the variants are reducible.

- **Scalable exact hierarchical agglomerative clustering via sparse
  geographic distance graphs.** Maus, Borin. (GSHAC)
  - [arXiv:2604.11656](https://arxiv.org/abs/2604.11656)
  - Serial exact heap on a sparse geographic graph.

- **ParChain: A framework for parallel hierarchical agglomerative
  clustering using nearest-neighbor chain.** Yu, Wang, Gu, Dhulipala,
  Shun.
  - [arXiv:2106.04727](https://arxiv.org/abs/2106.04727)
  - CPU point-set HAC.

- **GPU-UPGMA: high-performance computing for UPGMA algorithm based on
  graphics processing units.** Lin, Lin, Hung, Chung, Lee. *Concurrency
  Computat. Pract. Exper.*, 2015.
  - [DOI](https://doi.org/10.1002/cpe.3355)
  - Dense N x N, N ~ 1e3-1e4. Val has 2.17e6 fragments.

- **RAPIDS cuML** `AgglomerativeClustering`: `linkage={"single"}` only.
  - [docs](https://docs.nvidia.com/cuml/latest/api/generated/cuml.cluster.AgglomerativeClustering/)

## Related software

- [ParAlg/ParHAC](https://github.com/ParAlg/ParHAC)
- [google/graph-mining](https://github.com/google/graph-mining) (TeraHAC / clustered-graph)
- [constantinpape/nifty](https://github.com/constantinpape/nifty)
- [constantinpape/affogato](https://github.com/constantinpape/affogato) (GASP mean path)
- [sciai-lab/mutex-watershed](https://github.com/sciai-lab/mutex-watershed)
- [funkelab/lsd](https://github.com/funkelab/lsd) ;
  [agglomerate worker](https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/workers/agglomerate_worker.py)
- [rapidsai/cuvs](https://github.com/rapidsai/cuvs) (single-linkage, not this RAG)
- [PytorchConnectomics/waterz](https://github.com/PytorchConnectomics/waterz) (region-graph helpers)

## Not used here

- **GALA.** Nunez-Iglesias et al. *PLOS ONE*, 2013, e71715.
  - [DOI](https://doi.org/10.1371/journal.pone.0071715) ;
    [arXiv:1303.6163](https://arxiv.org/abs/1303.6163)
  - [arXiv:1303.5942](https://arxiv.org/abs/1303.5942) is a different paper
    (Brassard, Devroye, Gravel, *Exact simulation of the GHZ distribution*).

- Meila, **Comparing clusterings by the variation of information.**
  *COLT*, 2003. [DOI](https://doi.org/10.1007/978-3-540-45167-9_14).
  VOI here is waterz
  [`evaluate.hpp`](../src/waterz-upstream/src/waterz/backend/evaluate.hpp):
  split `H(seg|gt)`, merge `H(gt|seg)`, skip `gt==0`.

- Zlateski, **A design and implementation of an efficient, parallel
  watershed algorithm for affinity graphs.** MIT M.Eng., 2011.
  [DSpace](https://dspace.mit.edu/handle/1721.1/66820).

- Turaga et al., **Convolutional networks can learn to generate affinity
  graphs for image segmentation.** *Neural Computation*, 2010.
  [DSpace](https://dspace.mit.edu/handle/1721.1/60924).
