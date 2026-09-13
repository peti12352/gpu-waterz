# N21 D0 owners

N21 diagnostic; not a throughput claim; not a 2 Gvox/s number; not 3090 Ti.

Source: `data/cache/n19_owners.json` (I0 2.16 nsys). Pin agg 1679.9 ms is `N19_I0_REPRO.json` CUDA events, not the nsys stage 1693.7 ms.

## Double count

NVTX `:hash_rewrite` 286.9 ms wraps `k_rewrite_dirty_fuse` 273.0 ms plus two 4-byte D2H copies. Do not subtract both.

## Unique floors vs 1679.9 ms

| If these go to 0 | leftover agg ms |
|---|---|
| rebuild + fuse (551.4) | ~1128 |
| + insert + emit (886.1 four-kernel) | ~794 |
| + pack_amask (986.3) | ~694 |

Old ~840 ms floor is invalid.

## Already listed

`k_propose_listed` 146.8 ms. Do not reopen as a dense-scan fix.

Hook 166.6 combined Recip, vcount 148.1: owner &lt;200 ms, stay skipped.
