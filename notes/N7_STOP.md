# N7: stop memo (successor to N6)

Date: 2026-09-06. Contract: TASK.md. Not a 2 Gvox/s claim.

## Verdict

Do not start CUDA for CSR splice / E2 GPU / E6t v2 / a new agglomerator.

24 GB fit is real (`B_FIT.md`: RAG 21.16 GiB worst, fits 23 usable).
The post-fit timed path is slower, not faster. Compact split shows the
1150 ms honest-stack agg line was an **E2 IOU**: hash owns compact, not
the O(n) dirty scan. Splice cannot deliver the credited 3.86x.

## Evidence

### Post-fit stages: `n7_postfit_stages.json`

Idle 5090 val `segment_d` (park + DoubleBuffer + G15):

- median e2e **1669 ms** (was 959)
- WS **1272 ms** (was 536 hardcoded)
- leftover **2 ms** (was 96; park is now inside WS)
- 0.108 Gvox/s on 180 Mvox

### Compact IOU: `n7_compact_iou.json`

G15 p0aa compact **227.4 ms**:

| bucket | ms | share |
|---|---|---|
| hash emit/insert/fill | **119.7** | 53% |
| rewrite_scan (O(n)) | 43.0 | 19% |
| other (mark/memset) | 57.0 | 25% |
| layer compact_radix | 7.7 | 3% |

`owner=hash`, `scan_frac=0.189`, `scan_dominates=false`.
N0 credited compact -> `227 x 0.259 = 59 ms`. Measured GPU compact stays 227.
Without that credit, scaled agg is ~2350 ms, not 1150.

Even if compact went to zero, post-fit WS alone is already over the
val-scaled budget.

## TASK.md / dataset

TASK.md matches the listing on every graded gate. The "~9 GB leftover"
sentence is listing math (24-15.1), not the measured 21 GiB peak.
Dataset stays on greengoblin (`/home/v/proj/petya/waterz/data/ws_bounty/`).
No 3090 Ti here; no graded speed number.

## Hard limits (unchanged)

- Abboud ICALP 2024 Thm 3: exact average-linkage CC-hard on diameter-4 trees.
- ParHAC Thm 1.2: exact average-linkage P-complete; (1+ε) is the parallel class.
- Tseng SPAA 2022: dynamic HAC SETH-hard (we are static).
- No public GPU ParHAC. PCBS 2024 is CPU. G-kway: graphs do not help coarsen.
- MultiMerge / StarMerge / G2 already spent (E2 3.86x visits, G2 1.15x wall,
  E6t slower + non-deterministic).

## N7 gate

need_x without N4 was 1.96 on a 1150 ms IOU. That IOU is false.
Best legal leftover (ε=0.40) is still 1.04x.
CSR splice is closed by measurement, not by taste.

**No CUDA.**
