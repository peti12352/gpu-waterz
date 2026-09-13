# N22 finish -- occupied-slot emit + leftover inventory

Campaign closed on greengoblin (idle RTX 5090). Not a 2 Gvox/s number;
not 3090 Ti. Product default still E6s. Dual-eps 0.08 four-T / 0.40 T=0.3.
No 2.16: no ID cleared the val-cut >=100 ms gate after four-T PASS.

## Why N21_A3 was the wrong skip

N21_A3 skipped emit because `ntab_use = next_pow2(ndirty*2+1024)` already
(load ~0.26 on the first inner). nsys disagrees on *empty-slot scan*:
`k_hash_emit_holes` 161.1 ms, 438 launches, avg 368 us vs med 42 us
(avg/med=8.7), max 6.8 ms. First inner ntab=16,777,216 vs nuniq=4,439,124
(empty_frac=0.735). CUB DeviceSelect over ntab is still O(ntab). Occupied
indices recorded at insert CAS-win is the unrun trial (`WATERZ_SLOT_EMIT`).

## GPU IDs (idle 5090, parks off, `timing_usable_vs_1679=true`, hogs empty)

Val T=0.3 pin from N21_D0: **456.8 ms** on rag.npz (not the 2.16 1679.9 pin).
Cold first-run wall vs D0. Warm run2 is not the cut.

| ID | status | one-line |
|---|---|---|
| N22_D0_EMIT | go | nsys avg/med 8.7x; occupied emit is the GPU trial |
| N22_A3 | identity+four PASS, noise | SLOT_EMIT=1; cold val **438 vs 457 = 18 ms** |
| N22_A5 | identity+four PASS, no 2.16 | A3+A1+A2 stack cold val **395 = 62 ms**; vs N21_A4 54 ms, emit adds ~8 ms (not additive with A3's 18) |
| N22_COPY_SZ_NECESSARY | stamp, no kernel | `k_copy_sz_list` 114.5 ms, 246 x ~465 us uniform, already listed over roots; STICKY_SZ0 VOI-dead |
| N22_REWRITE_NEEDS_CSR | stamp, no kernel | first inner ndirty 5.6M vs nact 2.8M; holes are rewrite *output*; do not CONT E6t |
| N22_PROPOSE_ALREADY_LISTED | skip | 146.8 ms; avg/med 25x is atomicMax on large nact, not empty-slot scan |
| N22_RAG_FACES_BELOW_GATE | skip | `k_hash_faces` 48 ms of RAG 85 |
| N22_COUNT_V2_WS_SKIP | skip | WS; N21_W3 freeze |
| N22_COMPACT_ROOTS_LISTED | skip | already listed over live roots |
| N22_PACK_AMASK_A2 | skip | N21 A2 already attacked this domain |

All identity checks: inner=538 merges=1853545 nseg=321855, parents
array_equal vs E6s control, four-T PASS, VOI split=0.440823 merge=0.254275.
`keep_default=false`. `ran_216=false`. Insert-time `atomicAdd` onto the
slot list ate most of the ntab-scan save: occupied emit is parent-identical
and not a closer.

## Structural leftovers (do not grind)

1. **Rewrite is still O(nscan).** Half of first-inner dirty edges are
   below-TL live (not on alist). No dirty-edge CSR on the legal path.
2. **Emit stays O(ntab) on the product path.** SLOT_EMIT is env-gated,
   default off, 18 ms val / 62 ms stacked, both <100 ms 2.16 gate.
3. **WS compress is a fat list.** Frozen N21_W3.
4. **`k_copy_sz_list` is necessary work** for freeze eps, already listed.

Env-gated only: `WATERZ_SLOT_EMIT` (plus the N21 listed/jump levers).
Do not default. Do not reopen n19_dead / n20_dead / n21_dead IDs.
Do not launch `csrc/n20_rnn_gpu_prep.cu`. Do not CONT `AGG_E6t`.

Stamps: `data/cache/n22_dead.jsonl`. Pins: `N22_D0.json`, `N22_A3.json`,
`N22_A5.json`.
