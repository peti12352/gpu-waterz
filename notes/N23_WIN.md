# N23 finish: A5 2.16 + contact-mean CSR rewrite

Campaign closed on greengoblin (idle RTX 5090). Not a 2 Gvox/s number.
Product default still E6s. Dual-eps 0.08 four-T / 0.40 T=0.3.
Flags stay default off: `WATERZ_SLOT_EMIT`, `WATERZ_LISTED_INSERT`,
`WATERZ_LISTED_REBUILD`, `WATERZ_CSR_REWRITE`. Do not call `e6t_rebuild`.

## N23_A0: the A5 2.16 hole

N22_A5 had identity+four PASS and a 62 ms val cut, then skipped 2.16
under the N18 B3 100 ms val gate (`COMPACT_EVERY=8` looked fine on val
and was slower on 2.16). A0 forces 2.16 after identity+four anyway.

Idle 5090, parks off, `timing_usable_vs_1679=true`:

| | ms |
|---|---|
| val T=0.3 cold | 439.6 (cut 17 vs D0 456.8) |
| four-T | 459, PASS |
| 2.16 agg run1 | 1623.8 |
| 2.16 agg run2 | 1622.2 |
| 2.16 e2e | 3055.9 |
| pin agg / e2e | 1679.9 / 3093.4 |

Identity: inner=538 merges=1853545 nseg=321855, two-run parents
array_equal, equal E6s control, VOI split=0.440823 merge=0.254275.
nlab_216=3860788. `keep_default=false`. Stamp `216_cut`.

Both 2.16 aggs are >=50 ms under 1679.9. That is a closer on the
optional A5 env, not a product-default move.

## N23_D0: CPU splice equals scan

Full `data/cache/rag.npz`. `g0_agg_ref --mode both --csr --lemma`.

| eps | pass | fast csr_lookup | fast csr_splice | base compact_edge | fast compact_edge |
|---|---|---|---|---|---|
| 0.08 | PASS | 349,299,110 | 40,300,661 | 1,539,103,036 | 197,347,864 |
| 0.40 | PASS | 263,661,473 | 41,525,448 | 573,688,003 | 84,867,120 |

No AssertionError. Dirty set from incidence lists equals the endpoint
scan every inner. `go_gpu_csr=true`. The `--csr` work counter charges
walk visits to both `dirty_scan_visits` and `csr_lookup_visits`, so
their ratio is 1.0; the predicted rewrite-domain shrink is compact_edge
base/fast (~7.8x at 0.08, ~6.8x at 0.40). Pointer-chase visits
(`csr_lookup`) stay 3x the unique dirty edges because dead slots stay
on the lists.

## N23_A1: GPU CSR is identity-true and slower

`WATERZ_CSR_REWRITE=1`, A5 flags off, `FUSE_DIRTY` required. Layer-entry
`head`/`nxt` prepend-build, listed fuse, in-place hash commit (CAS-win
slot, not prefix-emit into `holes[0:m]`; that permute broke splice),
then splice.

Prefix-emit into foreign slots failed identity (inners 563-570).
In-place commit: identity+four PASS, same VOI, same counts.

| | ms |
|---|---|
| val T=0.3 cold | 5617.8 (5161 slower than D0) |
| four-T | 11150, PASS |
| 2.16 agg | 23240 |
| 2.16 e2e | 24657 |
| pin agg | 1679.9 |

Val nsys (not the 273 ms 2.16 pin; kernel mix):

| kernel | nsys share | launches | avg us |
|---|---|---|---|
| k_csr_gather | 94.9% | 315 | 15896 |
| k_csr_splice | 2.9% | 315 | 479 |
| k_rewrite_dirty_fuse_listed | 0.1% | 315 | 24 |
| k_csr_scatter | ~0 | 5 | 105 |

Listed fuse is cheap. The linked-list gather is the new owner:
uncoalesced `nxt` walks, including dead slots. Dense `k_rewrite_dirty_fuse`
at 273 ms on 2.16 stays the legal rewrite. Stamp `216_slower`.
`keep_default=false`. Do not rebuild CSR from full nscan each inner
(same cost class as today's rewrite). Do not default the flag.

## Product pin

Unchanged: `data/cache/N19_I0_REPRO.json`, agg **1679.9 ms**, e2e
**3093.4 ms**. Optional A5 env is a measured ~56 ms 2.16 cut with
identity+four; turning it on is a product choice, not this campaign.
CSR stays off.

Dead stamps: `data/cache/n23_dead.jsonl` only (do not append n19-n22).
