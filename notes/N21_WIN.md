# N21 finish: listed contract + list-compress

Campaign closed on greengoblin (idle RTX 5090). Not a 2 Gvox/s number;
not 3090 Ti. Product default still E6s. Dual-eps 0.08 four-T / 0.40 T=0.3.
No 2.16: no ID cleared the val-cut >=100 ms gate after four-T PASS.

## Best timed product (unchanged pin)

Idle RTX 5090, parks off, pin `data/cache/N19_I0_REPRO.json`:

WS ~1309 ms, RAG ~85, agg **1679.9 ms**, extract ~18, e2e ~3093 ms on
`[3,375,2400,2400]` at aff 0.3.

## True owners (unique kernels; no NVTX double count)

From `data/cache/N21_D0_OWNERS.json` / I0 nsys. Old leftover ~840 ms added
NVTX `:hash_rewrite` 286.9 on top of `k_rewrite_dirty_fuse` 273.0 (overlap).

| kernel | 2.16 ms | launches | domain |
|---|---|---|---|
| `k_w5_compress_list` | 508.0 | 6 | WS list UF |
| `k_rebuild_active` | 278.4 | 438 | dense nscan |
| `k_rewrite_dirty_fuse` | 273.0 | 438 | dense nscan |
| `k_hash_insert` | 173.6 | 438 | dense nscan, keep predicate |
| `k_hash_emit_holes` | 161.1 | 438 | scan ntab |
| `k_pack_amask` | 100.2 | 443 | dense nscan |
| `k_propose_listed` | 146.8 | | already listed |

Floors vs 1679.9 if kernels vanished (optimistic): rebuild+fuse leftover
**~1128**; four-kernel **~794**; +pack **~694**.

## D0 gates (val T=0.3, 315 dirty records, hops n=2)

- ndirty/nscan p50=0.0185 p99=0.408 -> listed insert GO
- nact/nscan p50=0.0034 -> listed rebuild GO
- early inners are not sparse (first: ndirty/nscan=0.75, nact/nscan=0.38)
- e9b: nlist=104.7M mean_hops=0.90 root_frac=0.55 hop_max=8
- e9c: nlist=71.2M mean_hops=2.17 root_frac=0.038 hop_max=9
- compress-subset NO-GO (p50 root_frac=0.29); list-jump GO by p50 hops=1.53
  but hop_max=9 means the 508 ms owner is **nlist size**, not chain height
- val T=0.3 ParHAC pin 456.8 ms is rag.npz, not the 2.16 1679.9 pin
- WS identity True (nfrag=2175400, bg=506568)

## IDs vs evidence

| ID | status | one-line |
|---|---|---|
| N21_D0_OWNERS | done | unique kernels; 840 invalid |
| N21_D0_DIRTY | done | listed insert/rebuild GO |
| N21_D0_HOPS | done | fat list; hop_max=8-9 |
| N21_A1 | identity+four PASS, not a closer | listed insert over holes; cold val 422 vs D0 457 = **35 ms**; no HASH_INSERT_ONLY |
| N21_A2 | identity+four PASS, no 2.16 | listed rebuild holes ∪ old alist; vacated amask in rewrite; cold val 406 = **51 ms** (noise-edge vs 50 ms; <100 ms 2.16 gate) |
| N21_A4 | identity+four PASS, no 2.16 | A1+A2 stack cold val 403 = **54 ms**; levers overlap, do not add |
| N21_A3 | skip | ntab already next_pow2(ndirty*2+1024); not a broken domain |
| N21_W1 | D0 no-go | p50 root_frac 0.29; e9c 3.8% roots; packing a subset is the same nlist tax |
| N21_W2 | ident+four PASS, noise | list-only J=4; val WS 436 vs 430 = **-6 ms**; gold fp fff9037c... |
| N21_W3 | freeze WS | nlist 71-105M is structural; not LIST_HALVING / JUMP_FLATTEN / Playne |

All identity checks: inner=538 merges=1853545 nseg=321855, parents
array_equal vs E6s control, four-T PASS. `timing_usable_vs_1679=true`,
hogs empty. `keep_default=false`. `ran_216=false`.

## Bottlenecks that remain

1. **Rewrite is still O(nscan).** Holes are the *output* of
   `k_rewrite_dirty_fuse`. Listed insert/rebuild do not delete the 273 ms
   fuse scan. Early-layer dirty ratio 0.75 so listed insert cannot 10x
   `k_hash_insert` either.
2. **Emit is O(ntab)** with a 0.5 load-factor table. Not a listing bug.
3. **WS compress is a fat list.** Six stitch invocations, nlist 0.40-0.58
   nvox, hop_max 9. J-jump and root-skip do not change the bandwidth of
   walking 70-100M uncoalesced indices. Hook list must keep faces+roots
   (stitch `atomicMin` writes into phase-1 roots).
4. Val ~50 ms agg cuts are inside/at the 50 ms noise band taught by N18 B3
   (2.16 can go backwards). They were **not** promoted to 2.16.

Env-gated only: `WATERZ_LISTED_INSERT`, `WATERZ_LISTED_REBUILD`,
`WATERZ_LIST_JUMP`. Do not default. Do not reopen n19_dead IDs.

Stamps: `data/cache/n21_dead.jsonl`. Pins: `N21_D0.json`, `N21_A1.json`,
`N21_A2.json`, `N21_A4.json`, `N21_W2.json`.
