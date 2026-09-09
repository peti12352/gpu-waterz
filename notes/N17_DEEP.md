# N17 T2 two-run 2.16 labels

Not a 2 Gvox/s claim. Not 3090 Ti. Idle RTX 5090.
`EMIT_HOLES=1` + FOLD+SHARE_OFF+HOOK_ROOT+FUSE_DIRTY+NLIVE_ARITH.

Ident / T=0.3 VOI / four-T already gated in T2. This is 2-run sha256 of the
full 8 GiB uint32 labels on official `[3,375,2400,2400]`.

| run | e2e ms | WS | rag | agg | nlab | sha256 | ws_peak |
|-----|--------|-----|-----|-----|------|--------|---------|
| a | 3110.83 | 1314.96 | 85.53 | 1690.41 | 3860788 | c53d430e5ba7f2dbbb8b011d93b8de6a3e98f34276ab5f8e5afcd0c23eced937 | 13.22 GiB |
| b | 3083.07 | 1310.99 | 82.95 | 1669.08 | 3860788 | c53d430e5ba7f2dbbb8b011d93b8de6a3e98f34276ab5f8e5afcd0c23eced937 | 13.22 GiB |

det_216=True. **sha256 is byte-identical to N16 deep** (without EMIT_HOLES).
Same partition, 240 ms less agg. 5090 ≈ 0.694 Gvox/s. Not 2 Gvox/s. Not 3090 Ti.

keep_default=True (env). C++ EMIT_HOLES stays 0: 3090 24 GB peak not measured.

`data/cache/n17_deep.json`
