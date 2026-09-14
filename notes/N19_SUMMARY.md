# N19 summary: literature-driven dual-ceiling

not a 2 Gvox/s number; not 3090 Ti. Parks off.

Community capture: [ATLAS.md](ATLAS.md) · [PROBLEM.md](PROBLEM.md) · [voi_atlas.csv](../data/cache/voi_atlas.csv) · [legal_eval.sh](../scripts/legal_eval.sh)

## I0
- nsys force-export OK -> `n19_owners.json` (40 kernels)
- A1 reproduce: e2e=3093.4 ms (vs 3104; within ±2%), four-T PASS, keep_default=True

## W (WS)
- W0: compress_list~508 ms; hook/vcount/sort below 200 ms kill thresh
- W1 Playne hook: SKIP owner 84.5 ms
- W2 Allegretti: SKIP no flag/scan >=200 ms
- W3 vcount_priv: SKIP owner 148 ms
- **W_STOP**: no WS≤900 with four-T (no legal micro-opt cleared owner gate)

## H (MEAN)
- H0: hash/rebuild/rewrite owners >=200 ms
- H1 bin-ladder: wall 5841 ms, VOI FAIL -> dead
- H2 NNG: VOI/four FAIL -> dead
- H3 batch RNN: not implemented -> stamped
- H4 Lu chunk: parents != ParHAC -> stamped
- **H_STOP**: MEAN-exact local max; agg still ~1680 ms

## X
- X1 PRUF affinity: no integration; stamp
- X2 GASP Average: VOI FAIL expected
- X3 RAMA: 60 s timeout stamp

## C
- REFUSE on 5090; no C++ default flip

## Honest ceiling
- Best e2e ~3093-3115 ms / ~0.70 Gvox/s on 5090
- Need ~2.9x to 1080 ms; WS~1310 alone exceeds TASK budget
- Highest remaining device owners: w5_compress_list + ParHAC hash/rebuild path
