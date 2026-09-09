# N7 — post-fit remesure

Date: 2026-09-06. Contract: TASK.md. Not a 2 Gvox/s claim. No 2.16 allocation.

Idle 5090, parked aff + DoubleBuffer + `WATERZ_AGG_LEVERS=15` + `WATERZ_STAGE_MS=1`.
`data/cache/n7_postfit_stages.json`.

## Median of 5 (CUDA-event e2e)

| | pre-fit N0 | post-fit N7 |
|---|---|---|
| e2e | 958.70 ms | **1669.48 ms** |
| ws | 536 (hardcoded) | **1272.39 ms** |
| rag | 9 | 41.50 |
| agg | 317.61 (p0aa) | 351.72 (STAGE_MS) |
| extract | 3.434 (device) | 1.68 |
| leftover | 96.08 | **2.19** |
| parked | — | True |
| backend | gpu_dev | gpu_dev |

Val throughput: 180e6 / 1.669 s = **0.108 Gvox/s**. Pre-fit G15 was 0.188 Gvox/s.

The leftover vanished: STAGE_MS now accounts for the park + host permute that N0 dumped into a 96 ms hole. That work landed in **WS**, not extract.

## What the park cost

e9b host-parks 61 035 574 corners + chunked vcount, then permutes on the host.
`segment_d` also parks 540 MB affinity between `ws_flow_d` and `ws_label_d`.
WS wall +736 ms vs the old 536 line. Divide-only was 1.19 ms; the rest is host memcpy + gather.

## Honest 2.16 line

N0 `SCALE=21.333` applied to this WS wall prints 29 s / 0.074 Gvox/s. That overstates the device part (UF ~224 ms) and is the wrong tool for a host-park dominated stage. Do not quote 29 s as a TASK number.

The val number is enough: this path is **18.5×** under 2 Gvox/s on the 5090 at 180 Mvox, before 3090 Ti or 2.16 Gvox.
