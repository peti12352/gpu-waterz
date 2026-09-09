# N6 — stop memo

Date: 2026-09-06. Contract: TASK.md. Not a 2 Gvox/s claim.

## Verdict

Do not start CUDA for E2 / E4 / W5 / a new agglomerator.

After N0–N4, no identified lever cuts the remaining agglomeration residual
by ≥1.95× (the honest-stack gate without a proven make_big reflect). The
only extra T=0.3-legal V is ε=0.40 (1.04× on top of E1’s ε=0.32). That
is not enough. The missed experiment class is closed: every shallow
four-T-dead agglomerator also fails T=0.3 VOI.

## Honest budget (N0)

`data/cache/e6_recal_honest.json`

Idle 5090 val leftover = 96.08 ms (e2e 958.70 − ws 536 − rag 9 − agg 317.61).
Real extract = 3.434 ms. E6-as-written dumped leftover into extract → 2049 ms.

| attribution | total | Gvox/s | vs 1080 ms |
|---|---|---|---|
| leftover → WS (honest) | 1642 ms | 1.315 | 1.52× |
| leftover → alloc (old E6) | 3692 ms | 0.585 | 3.42× |
| leftover → sync | 1738 ms | 1.242 | 1.61× |

Honest stack 3: ws 227 + rag 192 + agg 1150 + extract 73 = 1642 ms.
Agg must be ≤ 587 ms → **another 1.96× on 1150**.
Even WS+RAG reflect (N4, unproven on real aff) leaves agg ≤ 1025 ms → 1.12×.

## N1 — T=0.3 regrade of shallow agglomerators

`data/cache/n1_t3_regrade.json`  keep=0

| tag | split | merge | T=0.3 |
|---|---|---|---|
| locked ε=0.08 (control) | 0.4512 | 0.2505 | PASS, not shallow |
| no-asym ε=0.08 | 0.4570 | 0.2421 | PASS, not shallow |
| X0 frozen CC | 0.1550 | 7.5519 | FAIL giant |
| HistogramQuantile p50/p85 frozen | ≡ X0 | ≡ X0 | FAIL (degenerate hist) |
| Zlateski S0=256 | 1.0633 | 0.2201 | FAIL split |
| Zlateski S0=1024 | 0.7951 | 0.2373 | FAIL split |
| rel-contact γ=0.05 α=2/3 | 0.4210 | 0.3458 | FAIL merge |
| rel-contact γ=0.10 α=2/3 | 0.4628 | 0.2795 | FAIL merge (closest) |
| waterfall full | 0.1550 | 7.5519 | FAIL giant |
| waterfall 1-pass | 1.9041 | 0.1936 | FAIL split |
| waterfall 2-pass | 0.9554 | 0.2405 | FAIL split |
| mutex / AbsMax | 0.9097 | 0.2033 | FAIL split |

Four-T death is T=0.3 death for every shallow class. The E1 pattern does
not generalize beyond ParHAC-ε.

## N2 — higher ε at T=0.3 only

`data/cache/n2_higher_eps.json`

| ε | split | merge | sum_nlive | vs locked 2.691e9 | T=0.3 |
|---|---|---|---|---|---|
| 0.32 (E1) | 0.4451 | 0.2509 | 933245065 | 2.88× | PASS |
| **0.40** | 0.4408 | 0.2543 | 894217121 | **3.01×** | **PASS** |
| 0.48 | 0.4422 | 0.2892 | 723337741 | 3.72× | FAIL merge 0.2892>0.2611 |

Best legal V is now ε=0.40 size_asym on, T=0.3 only. Extra vs E1: 1.04×.
Replacing 1150 ms with 1150×2.88/3.01 = 1100 ms still 1.87× the 587 ms
agg budget.

## N3 — graph structure

`data/cache/n3_rag_structure.json`

- nedge 7 505 458; mean==1: 65 256; mean>0.9: 6 762 417; mean>0.3: 6 909 860
- prefilter edge cut 1.09× (97% of edges are already above T=0.3)
- mean degree 6.90; locked visits/merge ~1152 vs essential ~7 (167× slack
  is the G2 scan, already counted in E2’s 3.9× — not a new 167×)
- prefilter mean≤T drop: fingerprint ≠ locked; sum_nlive cut 1.19×;
  T=0.3 VOI PASS (0.4527 / 0.2513). Not 1.95×. Not stackable on E2
  identity (partition moved).
- saturated mean==1 prefix: leftover 49 586 edges, giant, merge VOI 7.87 FAIL

## N4 — mirror / seam

`data/cache/n4_mirror_identity.json`

Reconstructed make_big (flip + shift + zero seam): synthetic seams are
identically 0. Fragment 2-tile fg–fg differ counts are large (z 1.41M).
Zero seam aff ⇒ those faces cannot be mean>T RAG edges, so tiles are
merge-independent **if** the official generator matches the README.

This is not a 12× wall-clock claim. Time is still val e2e × (1792/1008).
make_big.py and affinity.h5 were not on disk; no 2-tile RAG identity on
the real generator output. N5 therefore does **not** take the 1.18× gate.

## Hard limits (unchanged, now with N1–N3)

- Abboud et al. ICALP 2024: exact average-linkage HAC is CC-hard on
  diameter-4 trees; combinatorial O(n^{3/2-ε}) refutes Combinatorial BMM.
- Dhulipala et al. NeurIPS 2022: exact average-linkage is P-complete;
  (1+ε) is necessary for poly-log depth. We already live in (1+ε).
- Tseng–Dhulipala–Shun SPAA 2022: exact dynamic HAC is SETH-hard.
- Uniform ε>0.08 is four-T illegal. T=0.3-only ε dies at 0.48.
- Frozen / single-linkage / mutex / waterfall / Zlateski / relative-contact
  fail T=0.3 (N1). cuSLINK / PANDORA / cuML are that class.
- Yeghiazaryan PRUF: 0.57 Gvox/s intensity WS — not a 2 Gvox/s precedent.
- 2.16 Gvox WS peak 42.20 GiB. No timing until a measured peak fits 24 GB.

## N5 gate

need_x_without_n4 = 1.96. Best new factor = 1.04 (ε=0.40) or 1.19
(prefilter, different partition). Neither ≥ 1.95.

**No CUDA.**

The scientifically honest bounty outcome is: the TASK 2 Gvox/s gate is
not reachable from the locked (1+ε) mean-affinity ParHAC design plus
every legal lever that passed identity or T=0.3 VOI, and no remaining
shallow agglomerator passes T=0.3.
