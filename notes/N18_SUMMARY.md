# N18 summary — VOI-first max-EV sprint

Not a 2 Gvox/s claim. Not 3090 Ti. Parks off. card_busy refuse.

## Verdict

- **A1 unlock PASS:** `voi_only` harness on N17 stack; identity diagnostic only; four-T PASS; 2.16 **e2e=3104 ms / ~0.70 Gvox/s** (ws=1315, agg=1683).
- **A-track STOP:** A2–A5 no WS≤900 with four-T + timed 2.16. Freeze WS.
- **B-track:** parallel BinQueue dead; ε∈(0.40,0.5) all VOI fail; COMPACT_EVERY=8 slower. **ParHAC local maximum** at ~1690 ms agg.
- **C:** script ready; **REFUSE** on 5090 until real 3090 Ti; no C++ default flip.

## Remaining gap (honest)

Need ~2.9× e2e on this 5090 (~5× if bw-scaled to 3090 Ti). Agg alone cannot hit TASK while WS stays ~1315.
