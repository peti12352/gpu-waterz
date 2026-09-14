# N9 Type D measurements

Not a 2 Gvox/s number. Not a 3090 Ti number. Idle 5090 only.
Numbers below are from this run. No invented fill-ins.

GPU: `NVIDIA GeForce RTX 5090, 526 MiB, 31584 MiB, 0 %`

## Val W5 (`WATERZ_UF_ALGO=3`)

- identity vs `wz_fragments.npy`: True
- nfrag=2175400 bg=506568
- STAGE_MS ws=557.2412637993693 rag=42.572562117129564 agg=201.87185797840357
- w5_ms=175.96 (calls=2) bfs_ms=1.18 (calls=1)
- nlist_last=71249612 / nvox=180000000 frac=0.39583117777777777
- stitch rounds_sum=10
- BFS / WS = 0.0021242934265740814 (PRUF dead if <0.01)

## Component sizes

- basin (fragments `wz_fragments.npy`): ncomp=2175400 max=1863297 max/nvox=0.010352 giant>10%=False
- plateau (`e9a_plateau_hist`): ncomp=6165057 max=1408692 max/nvox=0.007826 giant>10%=False
- ConnectIt/Afforest: SKIP (no component >10% nvox)

## Dirty-key multiplicity (`hash_combine_dirty`, val G15)

- calls=315 dirty_e_sum=69650023 unique_sum=64983359
- mean multiplicity=1.0718132160573601 (max dirty_e=5644981 max unique=4439124)
- mean_mult < warp 32: True (hash-table PDFs stay dead if true)

## 2.16 W5 + z-slab face-clear (`WATERZ_UF_ALGO=3`)

- e2e_ms=9212.7 gvox_s_5090=0.234 (not TASK-grade; wrong card)
- STAGE_MS ws=6536.229460965842 rag=424.4484193623066 agg=2222.334673628211
- nlab=3860788 (T=0.3 labels). WS nfrag=26104800 from `E9 z-slab n=3` (12x val 2175400)
- w5_ms=2116.74 bfs_ms=14.08 nlist_max=418817490 nlist_last=285029332 / 720000000
- WS > 4000 ms: True STOP CUDA tracks A/B/C

## Verdicts from this run

- PRUF: dead (BFS <1% WS)
- ConnectIt/Afforest: do not implement
- hash-table library swap: do not implement (multiplicity << warp)
- CUDA leftover tracks: STOP (WS still >4s after W5)
