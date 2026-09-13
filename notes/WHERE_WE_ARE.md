# Where we are

Idle RTX 5090 pin for the current stack. What the algorithm is:
[docs/decode.md](../docs/decode.md). Other GPUs:
[docs/porting.md](../docs/porting.md). Lab index: [README.md](README.md).

Quality gate: four affinity thresholds (0.2 / 0.3 / 0.4 / 0.5) must still
match stock waterz VOI (split and merge each +0.02). Speed on 2.16 Gvox is
compared only after that gate, on an idle card. A val-set improvement
smaller than 100 ms is treated as noise.

Product agglomeration stays paper-style ParHAC (E6s). Merge allowance is
**(1+0.08)** when grading four thresholds, **(1+0.40)** on the single
affinity-0.3 speed path. Larger allowances fail merge VOI.

Detail and raw stamps: [ATLAS.md](ATLAS.md), [PROBLEM.md](PROBLEM.md),
[N20_WIN.md](N20_WIN.md), [N21_WIN.md](N21_WIN.md), [N22_WIN.md](N22_WIN.md).
Dead lists: `data/cache/n19_dead.jsonl`, `n20_dead.jsonl`, `n21_dead.jsonl`,
`n22_dead.jsonl`.

---

## Absolute best timed result

Pin: `data/cache/N19_I0_REPRO.json`. Volume `[3,375,2400,2400]` (2.16 Gvox),
affinity 0.3, CUDA events, affinity already on the GPU, parks off.

| stage | milliseconds |
|---|---|
| watershed | 1309.3 |
| region graph | 85.4 |
| agglomeration | 1679.9 |
| extract labels | 18.3 |
| **end-to-end** | **3093.4 (~0.70 Gvox/s)** |

Four-threshold VOI: PASS. Two full runs produce the same labels.
Watershed identity: 2,175,400 fragments, fingerprint `fff9037c…`.
Agglomeration at affinity 0.3: VOI split 0.4408 / merge 0.2543
(limits 0.4738 / 0.2611), 538 inner iterations, 1,853,545 merges.

The same agglomerator on the small cached region graph (not 2.16) takes
**456.8 ms**. That val number is a diagnostic, not the product pin.

---

## Remaining paths

The open problem is still: **same partition quality, less wall time** — or
a proof that this watershed + contact-mean ParHAC split cannot get there.

What is actually left:

1. **A contact-mean rewrite that is not a full-array scan.** After each
   matching round the code must relabel live edges. That pass is still 273 ms
   on 2.16 because it walks every scanned edge. Early rounds are dense (first
   round: 5.6 million dirty edges, only 2.8 million of them on the "active"
   list). Listing the active set cannot delete this scan: the hole list is
   the *output* of rewrite. Doing this well needs an adjacency structure that
   still updates contact-mean the waterz way. The StarMerge clustered graph
   is the wrong structure (see closed). Do not turn on the E6t code path.

2. **A written ceiling, not more micro-kernels.** If the heavy agglomeration
   kernels vanished, about 690-1130 ms of agglomeration would still remain,
   plus ~800 ms of watershed even in the fantasy where list-compress is free.
   The honest remaining write-up is this floor, not another 20 ms env flag.

3. **Another GPU.** Unmeasured. Do not scale 5090 times by HBM.
   [porting.md](../docs/porting.md).

4. **Shipping the identity-true env flags.** Listed insert/rebuild and
   occupied-slot emit match the current parents and pass four-threshold VOI.
   They save 18-62 ms on the small graph and were not taken to 2.16. Turning
   them on by default is a product choice, not a path off the 1680 ms agg pin.

Shrinking the watershed compress list (508 ms, 71–105 million entries) is
**not** an open speed path unless fragment identity is allowed to change.
The hook list has to keep both plateau faces and phase-1 roots.

---

## Fully closed (do not reopen)

Constraints used for every kill: four-threshold VOI, or parent-identity
against the current ParHAC stack, or a structural count that makes a GPU
port pointless. Timing vs 1679.9 ms only when the card is idle.

| Attempt | What the evidence says | Where |
|---|---|---|
| A different merge rule (Kruskal, mutex, GASP, NNG, complete-link, WPGMA, Ward, …) | Different partition; VOI fails or the statistic is not contact-mean | atlas, N19 H2/X*, N20 X4/X5/REFUSE |
| Exact mean heap / RNN / NN-chain as a 1680 ms closer | Dendrogram height 12,539; RNN hit a 5,000-round cap on every threshold; parents ≠ heap; four-T fail | N20 D2, D3, RNN |
| StarMerge / E6t clustered rewrite | First layer is **71%** of all merges (1.32M / 1.85M) at eps=0.08; denser at 0.40. Small-merge assumption is false here | N20 D1, STAR |
| Spatial Lu Alg 2 | Four-T passes, but parents ≠ heap, ~3.86M of 7.5M edges left, **6645 s** on CPU | N20 LU2 |
| Looser merge allowance (eps 0.41–0.49 or ≥0.5) | Merge VOI fails on affinity 0.3 | N18 B2 |
| Bin-queue agglomeration | Hang, or 2253 ms and VOI fail | N18 B1, n19_dead |
| Playne / list-halving / jump-flatten / mapped-host atomics / grayscale PRUF as S1 | Wrong basins, ~1e6 ms, or a different watershed | n19_dead, N12 |
| "Just compact more / insert-only / fuse-pack / cuda graphs / sticky size" | Noise, illegal identity, or **slower on 2.16** (compact-every-8: 1765 vs 1690) | N18 B3, n19_dead |
| Listed insert / rebuild as a 2.16 closer | Parents match; four-T pass; **35 / 51 / 54 ms** on val; below the 100 ms gate | N21 A1, A2, A4 |
| Occupied-slot hash emit as a 2.16 closer | Parents match; four-T pass; **18 ms** alone, **62 ms** stacked; recording slots at insert ate the empty-slot win | N22 A3, A5 |
| Skip empty hash slots with CUB DeviceSelect | Still reads the whole table | N22 D0 |
| Compress only union-find roots / jump pointers in the watershed list | Roots are 4–55% of the list; jump-4 was **6 ms slower** on val watershed. Cost is list length, not hop height | N21 W1, W2, W3 |
| Grind `k_copy_sz_list` | Already walks only live roots; needed for the freeze test; skipping it (sticky sz0) fails VOI | N22 COPY_SZ |

Do not launch `csrc/n20_rnn_gpu_prep.cu`. Do not default
`WATERZ_LISTED_INSERT`, `WATERZ_LISTED_REBUILD`, `WATERZ_LIST_JUMP`,
`WATERZ_SLOT_EMIT`. Do not flip C++ defaults until peak VRAM is measured
on the target card.

---

## Last two campaigns (N21, N22)

**N21.** No new agglomeration class. Timed the real kernels (an older "~840 ms
leftover" had double-counted a range marker with the rewrite kernel). Walking
only dirty / active items is legal on typical rounds and still dense on the
first rounds. Identity-true; val cuts too small for 2.16. Watershed: the
508 ms owner is a 70–100 million entry list.

**N22.** The skipped emit idea was "empty slots in a load-0.5 table." nsys:
average emit 368 µs vs median 42 µs; first round 16.8M table slots vs 4.4M
keys. Recording occupied slots at insert matches parents and does not clear
the 100 ms gate. Rewrite still needs an adjacency structure; copy-size is
necessary work, not a closer.

The 1679.9 ms / 3093 ms product pin did not move.
