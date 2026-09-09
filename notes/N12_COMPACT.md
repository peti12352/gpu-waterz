# N12 compact-every

Not a 2 Gvox/s claim. Not 3090 Ti. nsys: `compact_radix` 1.4% NVTX, `k_hash_insert` 3.4% kernels — hash/radix do not own 2229 ms.

| k | T=0.3 ms | split | merge | T=0.3 | four-T ε=0.08 | vs cold k=0 |
|---|---|---|---|---|---|---|
| 0 | 477.6 (cold) | 0.4408 | 0.2543 | PASS | PASS | 1.00× |
| 2 | 248.6 | 0.4421 | 0.2537 | PASS | PASS | 1.92× vs cold |
| 4 | 231.6 | 0.4460 | **0.2653** | FAIL | PASS | — |
| 8 | 219.7 | 0.4408 | 0.2543 | PASS | PASS | same trajectory as k=0 |

Warm nsys val agg was **211 ms**. k=2/8 are not a 1.2× cut vs that. Do **not** default `WATERZ_COMPACT_EVERY`. Keep the env lever. k=4 T=0.3 merge over limit.

`WATERZ_HASH_INSERT_ONLY` not defaulted (`k_hash_insert` << 40% of 2229).
