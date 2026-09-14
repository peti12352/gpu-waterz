# N20 finish: where we are

Campaign closed on greengoblin 2026-09-12T00:23:44+02:00. Host idle at
audit 2026-09-13T15:24:37+02:00 (load 0.35, GPU 0% / 16 MiB, no n20 procs).
Not a 2 Gvox/s number; not 3090 Ti. No kernel launch. No 2.16 on FAIL.

Oracle for exact mean is the waterz/s4 heap (`N20_D2_parents.npy`), not ParHAC.

## Best timed product (unchanged)

Idle RTX 5090, parks off, pin `data/cache/N19_I0_REPRO.json`:

WS ~1309 ms, RAG ~85, agg **1679.9 ms**, extract ~18, e2e ~3093 ms on
`[3,375,2400,2400]` at aff 0.3. Dual-eps: 0.08 four-T, 0.40 single T=0.3.

## Plan IDs vs evidence

| ID | status | one-line |
|---|---|---|
| S43 archive | done | `papers/SOURCES.md` S43; `papers/n20_http.json` |
| N20_UNIT | PASS | 12/12 3-node RAG |
| N20_D1 | EV-dead | layer-0 1.32M/1.85M = 71.3% over 64 outers at eps=0.08 |
| N20_STAR | gated kill | do not rewrite CUDA; do not CONT E6t |
| N20_D2 | height kill | h_max=12539 at T=0.2 (P0i sample was 9). Wall ~2005 s **contaminated**; `timing_usable_vs_1679=false`; no speed redo |
| N20_X4 | VOI FAIL | complete-link under-merge; split 1.296-1.482; wall 1642 s idle |
| N20_X5 | VOI FAIL | WPGMA under-merge; split 0.767-1.130; wall 1870 s; VOI valid (end-of-job loadavg lag is not a hog) |
| N20_D3 | cap kill | rounds=5000 all T, cap=5000, `go_gpu_rnn=false`; wall 1116 s idle |
| N20_RNN | kill closer | parents != heap; four-T FAIL (incomplete at cap); `go_gpu=false` |
| N20_LU2 | VOI PASS, no speed | spatial B^C four-T PASS; parents != heap; nres 3.76-3.87M of 7.5M; lu_ms 6645 s; `no_speed_path` |
| N20_GPU_PREP | prep only | `ran_gpu=false`; nvcc empty; skeleton exists; do not set `N20_GPU_RUN` |
| N20_REFUSE | closed | Ward/cuML/Chamfer/GSHAC/GPU-UPGMA/forums |
| atlas + PROBLEM | done | this file; `voi_atlas.csv` 113 rows; `n20_dead.jsonl` |

## Win vs 1679.9 ms

No new HAC class beats contact-mean ParHAC under four-T VOI.

Structural cap on the **existing** algorithm (`N21_D0_OWNERS.json`; unique
kernels, no NVTX+kernel double count):

- rebuild 278.4 + fuse 273.0 = 551.4 ms; leftover if those go to 0: **~1128 ms**
- plus insert 173.6 + emit_holes 161.1 = 886.1 ms four-kernel; leftover **~794 ms**
- plus pack_amask 100.2; leftover **~694 ms**
- old `hash_rewrite` 286.9 + rebuild + fuse = 838.3 leftover **~840 ms** is invalid
  (`:hash_rewrite` NVTX wraps the fuse kernel)

Both floors are optimistic (kernels do not vanish). Exact heap / RNN / Lu CPU
walls are 1116-6645 s: `correct_but_serial` or incomplete, not GPU closers.

## Timing hygiene

Compare a wall to 1679.9 only if `timing_usable_vs_1679=true` and hogs empty.
D2 wall is contaminated (LeoCAD / `repair_at_scale`); height is still valid.
X4, X5, D3, RNN, LU2 stamped idle (`hogs=[]`). Decaying loadavg after a 100%
heap is not a hog.

## GPU

Skeleton `csrc/n20_rnn_gpu_prep.cu` is inventory. `N20_GPU_PREP.json` requires
D3 `go_gpu_rnn`, RNN four-T PASS, user lift, and `N20_GPU_RUN=1` before any
launch. All four are false. Do not nvcc-run. Do not nsys. Product stays E6s.
