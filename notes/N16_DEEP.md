# N16 deep stack verify

Not a 2 Gvox/s claim. Not 3090 Ti. Idle RTX 5090. Parks off.
Env: `FOLD_FLATTEN=1 SHARE_OFF=1 HOOK_ROOT=1 FUSE_DIRTY=1 NLIVE_ARITH=1`.
C++ product defaults stay 0. T8 sticky excluded (N15 four-T FAIL). Arena/pin/sort-pack/fuse-pack off.

## Identity (val, `wz_fragments.npy`)

identity=True array_equal=True run2_array_equal=True
nfrag=2175400 bg=506568 ndiff_raw=0 same_partition=True
fp=fff9037cab341692be0c9bf3c577d4ff
val peak=1928284592 leaked_bytes=0

## T=0.3 VOI (ε=0.40, 2-run)

ok=True run2_array_equal=True
split=0.44082269408145347 merge=0.25427477829766865 nseg=321855
identical to N13. Limits 0.4738 / 0.2611.

## four-T ε=0.08

`n15_four.py` wrote `data/ws_bounty/n16_deep_four/mine_thr{0.2,0.3,0.4,0.5}.h5`.
Independent official `run_baseline.py --candidate` regrade: **ACCURACY GATE: PASS**

| aff | split | merge | limit_s | limit_m |
|-----|-------|-------|---------|---------|
| 0.2 | 0.370698 | 0.335006 | 0.3979 | 0.3525 |
| 0.3 | 0.451219 | 0.250531 | 0.4738 | 0.2611 |
| 0.4 | 0.516214 | 0.226759 | 0.5378 | 0.2381 |
| 0.5 | 0.61294 | 0.218363 | 0.6309 | 0.2293 |

Byte-identical table vs T7 and T15 regrades.

## 2.16 two-run labels

Official `[3,375,2400,2400]` = 2.16e9 vx. Labels sha256 of full 8 GiB uint32:

| run | e2e ms | WS | rag | agg | extract | nlab | sha256 | ws_peak |
|-----|--------|-----|-----|-----|---------|------|--------|---------|
| a | 3345.18 | 1312.02 | 84.97 | 1929.00 | 18.40 | 3860788 | c53d430e5ba7f2dbbb8b011d93b8de6a3e98f34276ab5f8e5afcd0c23eced937 | 14193138368 |
| b | 3323.68 | 1310.32 | 82.79 | 1911.37 | 19.03 | 3860788 | c53d430e5ba7f2dbbb8b011d93b8de6a3e98f34276ab5f8e5afcd0c23eced937 | 14193138368 |

det_216=True (sha equal). nlab=3860788 matches N13/N15 stack. 5090 e2e ≈ 0.646 Gvox/s. Not a 2 Gvox/s number. Not 3090 Ti.

ws_peak=13.22 GiB (`ws_mem_peak`). 3090 Ti 24 GB **not measured**. SHARE_OFF is +4 B/vox on the parent; z-slab is the 3090 fit path. Do not ship from this peak.

## keep_default

False. Need four-T + 2.16 det + WS≤2149 **and** agg≤1861.7. WS 1312 passes 1.2× vs N13 (cut 1.966); agg 1929 cut 1.158 misses by 67 ms.

`data/cache/n16_deep.json`
