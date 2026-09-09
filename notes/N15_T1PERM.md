# N15 exp1 T1-PERM

Not a 2 Gvox/s claim. Not 3090 Ti. No 2.16. WATERZ_FOLD_FLATTEN=1 vs wz_fragments.npy.

- identity_byte=False array_equal=False nfrag=2175400 bg=506568 (gold 2175400/506568)
- run2_array_equal=False bg_mask_eq=True
- same_partition=False canon_eq=False fp_eq=False
- ndiff_raw=161028442 ndiff_canon=161028442 vox=180000000
- pairs n=3018980 gold_split=826033 pred_merge=63 max_gold_fanout=6 max_pred_fanout=822582
- verdict=wrong_basins revive_t1=False kill_t1_as_speed=True
- nfrag/bg match is not a partition: size histogram also differs (fp_eq=False)
- 89.5% voxels disagree after min-index canon. One pred id covers 822582 gold frags.
- run2_array_equal=False: fold is not deterministic. Matches D2 `uf_find_ro` walking
  `parent==seg` while other threads store labels.
- keep_default=False (exp1 is a diagnosis; no product-default change)
- next: share-off+fold find (N15 T2). Do not min-index relabel.
