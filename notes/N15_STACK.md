# N15 legal stack T2+T5+T6

Not a 2 Gvox/s claim. Not 3090 Ti. FOLD_FLATTEN=1 SHARE_OFF=1 HOOK_ROOT=1 FUSE_DIRTY=1. T8 excluded (four-T FAIL).

- ident=True run2=True nfrag=2175400 peak=1928284592
- voi ok=True run2=True
- 2.16 WS=1313.1 (base 2579.0 cut 1.964) agg=2081.2 (base 2234.1 cut 1.073) e2e=3499.3 (base 4915.3 cut 1.405) nlab=3860788 rc=0
- keep_default=False (need WS<=2149 AND agg<=1862; T6 agg alone misses 1.2x)
- C++ defaults stay off: share_off is +4 B/vox; 3090 24GB peak not measured
