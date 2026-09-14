# B: 8 official mirror flip triples

Date: 2026-09-06. Not a speed path.

Official `make_big.mirror` on all 8 `(z,y,x)` flip triples. GPU WS, val tile.

- every triple: nfrag **2175400**, bg **506568** (same count as identity)
- **8 distinct size-histogram fingerprints** (P1b: `WS(mirror)` is not `flip(WS)`)
- 12 x identity nfrag = 26 104 800
- official fused nfrag = 26 023 852 (80 948 fewer: waterz tie-break on the
  fused volume, not a different per-tile nfrag)

The 8 unique pipelines are different partitions with the same fragment
*count*. Cannot stamp. Already 8.3x slower than fused honest (P1_GATE).
