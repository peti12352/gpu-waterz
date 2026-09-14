# N16 legal stack

Not a 2 Gvox/s claim. Not 3090 Ti.

Legal env (all process-cached, default 0): FOLD_FLATTEN + SHARE_OFF + HOOK_ROOT + FUSE_DIRTY + NLIVE_ARITH.
Killed this round: STITCH_ARENA (leak, 1.00x), PIN_CHANGED (1029 s WS), SORT_PACK (slower), FUSE_PACK (1.008x), T8 sticky (N15 four-T FAIL), T10 graph (D2H), T13 drop vcount (BFS qsz).

N15 T2T5+T6 e2e=3499.3 agg=2081.2. N16 adds T7 arith: deep e2e=3345.2 agg=1929.0. Still keep_default=False.

C++ defaults unchanged: UF=3, parks off, ε=0.40 T=0.3 / 0.08 four-T, compact k=0, fold/share_off/hook_root/fuse_dirty/nlive_arith/arena/pin/sort_pack/fuse_pack all getenv 0.
