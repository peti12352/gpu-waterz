# B — measured 24 GB fit table

Date: 2026-09-06. Not a 2 Gvox/s claim. No 2.16 Gvox allocation.

Val `segment_d` (parked aff, DoubleBuffer + park, EDGES_PER_VOX=0.043,
thinned edges, dead cu/cv gone), idle 5090. `data/cache/b_fit.json`.

| stage | val tracked | val fused | 2.16 pred fused |
|---|---|---|---|
| WS | 0.916 GiB | 1.754 GiB | 21.05 GiB |
| RAG | 1.250 GiB | 2.871 GiB | **21.16 GiB** |
| AGG | 0.788 GiB | 1.627 GiB | 19.55 GiB |

Worst 21.16 (RAG). All three stages under 23 usable. `fits: true`.
Do not run 2.16. This is not a 2 Gvox/s number.
