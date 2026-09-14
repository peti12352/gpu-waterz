# N8: 2.16 is linear. Stop CUDA.

Not a 2 Gvox/s number. Not a 3090 Ti number. One official `make_big.py` volume
`[3,375,2400,2400]` = 2.16 Gvox, one `segment_d` at T=0.3, CUDA events, idle
RTX 5090 32 GB. Aff already in VRAM.

## The fact

| | val (0.18 Gvox) | 2.16 Gvox | x |
|---|---|---|---|
| e2e | 1154 ms | **13518 ms** | 11.7 |
| WS | 913 ms | 10832 ms | 11.9 |
| RAG | 41 ms | 433 ms | 10.6 |
| agg (ε=0.40) | 197 ms | 2224 ms | 11.3 |
| Gvox/s | 0.156 | **0.160** |: |

`n8_216.json`. nfrag after 3 z-slabs = **26 104 800** = 12 x 2 175 400 (val).
Labels at T=0.3: 3 860 788.

Target 2 Gvox/s is 1080 ms. This run is **12.5x** that, on a faster card than
the graded 3090 Ti. Time ∝ voxels. There is no sublinear 2.16.

## What 1-4 did (and did not)

- **Unpark.** Host permute of 61e6 corners is gone. Device `k_gather_u32` after
  a raw park of corners/vc (no CPU permute). Val WS 1272 -> 913 ms. Identity
  vs `wz_fragments.npy` holds. Fused 2.16 pred still 26.4 GiB OVER.
- **Z-slab.** Official make_big z-seams are aff=0 (P1). N=3 slabs of Z=125,
  one scratch, label offset. Face ±z bits cleared so BFS cannot walk out of
  the slab. Slab pred ~15.5 GiB. Fitted; ran.
- **W5.** Val identity, **1.63x** vs algo 0 (967 -> 594 ms). Default left at 0:
  first 2.16 W5 attempt overflowed BFS on slab 1 before the face-bit clear.
  Not the 227 ms 2.16 model. Three slabs of ~900 ms UF is 2.7 s of WS alone.
- **Dirty hash.** Parent-identical vs G15. Warp+SM **1.31x slower** (158 vs
  120 ms). Full-dirty CUB RBK **4.2x slower** (499 ms). G15 CAS stays. Hash
  ≤40 ms did not happen.
- **Dual-ε.** Speed path (single T=0.3) defaults to ε=0.40 (N2 PASS). Four-T
  stays 0.08. `WATERZ_AGG_EPS` overrides. README documents it.

## Why this is the stop

Val 959-1154 ms is 0.16-0.19 Gvox/s. 2 Gvox/s is a 10-12x if time ∝ voxels.
The only escape was sublinear 2.16 time (W5/E4 byte model: WS 227-281 ms).
Step 5 measured **WS 10832 ms**. The model was hope. The run is the proof.

No further agglomerator. No 8-tile serial. No cuco. No CUDA graphs on coarsen.
No host-stream aff. Dataset stays on greengoblin.
