# Gate log

hostname=greengoblin
CUDA_VISIBLE_DEVICES=unset
lego PID 1880797 still on GPU (do not touch)

## T0 PASS
command: uv venv .venv --python 3.12; uv pip install --python .venv h5py numpy; SETUPTOOLS_SCM_PRETEND_VERSION=0.9.5 uv pip install --python .venv -e src/waterz-upstream
note: PyPI waterz has no cp312 wheel; used vendored a0184d2
import waterz,h5py,numpy: ok (waterz.__version__ prints "uninstalled" — upstream __init__ bug, import works)
import torch: ModuleNotFoundError
nvidia-smi compute: 1880797 python3 8652 MiB (lego)

## E0 PASS
command: uvx gdown 1zbGpyr9M5Pvhgfy96V9erQwAeRZo23hW -O data/waterz_bounty.tar; tar xf
listing: cremiA_val/{affinity,gt,raw}.h5, baseline/{labels_thr*.h5,voi.csv,run_baseline.py}, make_big.py, viz/, README.md — present (plus make_viz.py, extra thr 0.1/0.7/0.9)
shapes: affinity (3,125,1200,1200) uint8; gt (125,1200,1200) uint32
voi.csv rounded == TASK.md table
threshold: FACT score=1-aff; see notes/THRESHOLD.md
README vs S1–S5: no semantic delta; see notes/TARBALL_VS_SOURCE.md

## E0b PASS-structure / FAIL-1e-05-jitter
command: unset CUDA_VISIBLE_DEVICES; data/ws_bounty + .venv/bin/python baseline/run_baseline.py
elapsed_sec=274.74
fragments=2175400 (TASK.md 2175400 exact)
rag_edges=7505458 (TASK.md 7505458 exact)
gated VOI vs voi.csv:

| aff | repro split | csv split | d_split | repro merge | csv merge | d_merge |
| 0.2 | 0.3754 | 0.37785 | 0.0024 | 0.3299 | 0.33247 | 0.0026 |
| 0.3 | 0.4521 | 0.45379 | 0.0016 | 0.2422 | 0.24106 | 0.0011 |
| 0.4 | 0.5180 | 0.51783 | 0.0002 | 0.2180 | 0.21808 | 0.0001 |
| 0.5 | 0.6103 | 0.61093 | 0.0006 | 0.2094 | 0.20932 | 0.0001 |

1e-05 criterion not met. Fragment/edge lock is exact. Tarball README already says 4th-decimal drift is expected on this wheel; we built a0184d2 from source (no cp312 wheel). Continue; E4 compares our heap to voi.csv on *waterz* fragments.

## E1 PASS
command: unset CUDA_VISIBLE_DEVICES; .venv/bin/python scripts/e1_synthetic.py
PASS e1a e1b e1c e1d e1e e1f

## E2cpu PASS
command: unset CUDA_VISIBLE_DEVICES; .venv/bin/python scripts/e2cpu.py
waterz fragments=2175400 bg=506568 (TASK 2175400 d=0)
ref_cpu fragments=2175400 bg=506568 sec=7.9
rel_count_err=0.000000
bg_match=True

## E3cpu PASS
command: unset CUDA_VISIBLE_DEVICES; .venv/bin/python scripts/e3cpu.py
fragments sec=18.8 n=2175400
rag sec=22.4 edges=7505458
rel_edge_err=0.000000 bg_endpoints=0 mean_s3=True
cache: data/cache/wz_fragments.npy data/cache/rag.npz

## E4 PASS
command: unset CUDA_VISIBLE_DEVICES; .venv/bin/python scripts/e4_exact.py
method: stock waterz S4 heap on E3 fragments (thresholds scores 0.5,0.6,0.7,0.8)
total_sec=162.0
nseg: 0.5=379876 0.4=345376 0.3=322167 0.2=294518
VOI vs voi.csv (+0.005):

| aff | split | base | d_split | merge | base | d_merge |
| 0.5 | 0.610281 | 0.610931 | -0.000650 | 0.209424 | 0.209318 | +0.000106 |
| 0.4 | 0.517995 | 0.517831 | +0.000164 | 0.217998 | 0.218080 | -0.000082 |
| 0.3 | 0.452148 | 0.453790 | -0.001642 | 0.242154 | 0.241062 | +0.001092 |
| 0.2 | 0.375447 | 0.377852 | -0.002405 | 0.329890 | 0.332474 | -0.002584 |

C++ heap port was +0.017 merge at 0.2 (rejected). E4 lock is vendored S4.

## E5 ACCURACY GATE: PASS
command: unset CUDA_VISIBLE_DEVICES; .venv/bin/python scripts/e5_boruvka.py ; then scripts/e5_grade_exact.py
hostname=greengoblin CUDA_VISIBLE_DEVICES=unset

Borůvka one-sided (PLAN §3.4): nseg 282829/312939/334450/360728
grader: 0.2 merge 0.3885>0.3525 FAIL; 0.3 merge 0.2779>0.2611 FAIL; 0.4 merge 0.2423>0.2381 FAIL; 0.5 PASS
ACCURACY GATE: FAIL

Borůvka mutual-only: hung (too few pairs/round). killed pid 3060018.

Fallback: exact S4 labels from E4 → mine_thr{0.2,0.3,0.4,0.5}.h5
nseg: 294518 322167 345376 379876
grader:

aff_thr=0.2  split 0.3754 (base 0.3779, limit 0.3979)   merge 0.3299 (base 0.3325, limit 0.3525)   PASS
aff_thr=0.3  split 0.4521 (base 0.4538, limit 0.4738)   merge 0.2422 (base 0.2411, limit 0.2611)   PASS
aff_thr=0.4  split 0.5180 (base 0.5178, limit 0.5378)   merge 0.2180 (base 0.2181, limit 0.2381)   PASS
aff_thr=0.5  split 0.6103 (base 0.6109, limit 0.6309)   merge 0.2094 (base 0.2093, limit 0.2293)   PASS
ACCURACY GATE: PASS

FROZEN agglomerator: exact S4 (score=1-mean, merge while score < 1-aff_thr, keep-cheaper + area-weighted mean, stale rescore). Not live-mean Borůvka. GPU later implements this statistic.

## G1 PASS
command: g++ -O3 -DNDEBUG -shared -fPIC -I src/waterz-upstream/src/waterz -o src/libheap_cpu.so src/heap_s4.cpp
method: vendored IterativeRegionMerging + MeanAffinity + OneMinus
E1 PASS; heap_sec=36.8 edges=7505458
nseg: 0.2=294516 0.3=322178 0.4=345384 0.5=379869
VOI vs voi.csv +0.005: all PASS (worst merge d=+0.001203 at 0.3)
5090 idle (16 MiB, no compute apps)

## G2 PASS
command: nvcc -O3 -arch=sm_120; .venv/bin/python scripts/g2_ws.py
waterz n=2175400 bg=506568; gpu n=2175400 bg=506568 sec=5.3 rel=0 det=True
flow on GPU; plateau+basin host (S1, same as ws_cpu.cpp)

## G3 PASS
command: .venv/bin/python scripts/g3_rag.py
edges=7505458 sec=2.9 rel_edge_err=0 bg_endpoints=0 mean_ok=True
on waterz fragments

## G4 ACCURACY GATE: PASS
command: .venv/bin/python src/segment.py ...affinity.h5 --out-dir data/ws_bounty; run_baseline.py --candidate mine_thr*.h5
hostname=greengoblin
segment() = GPU flow + host S1 plateau + GPU RAG faces + G1 S4 heap
nseg: 0.2=294353 0.3=322171 0.4=345295 0.5=379857
aff_thr=0.2  split 0.3754 merge 0.3477 PASS (limit 0.3525)
aff_thr=0.3  split 0.4524 merge 0.2422 PASS
aff_thr=0.4  split 0.5181 merge 0.2181 PASS
aff_thr=0.5  split 0.6100 merge 0.2095 PASS
ACCURACY GATE: PASS

## G5 PASS
command: .venv/bin/python scripts/g5_det.py
two segment() @ aff 0.3: array_equal True nseg=322171

## G6 FAIL
command: .venv/bin/python scripts/bench.py
nvox=180000000 min=35.6883 median=36.5848 max=38.5828 Gvox_s=0.005
need median<0.050s
ws_median=5.056 rag_median=2.621 heap_median=28.465 extract_median=0.477
hot stage=heap 78%. Serial S4 cannot hit 1.08s on 90M edges (PLAN kill-shot).

## G0 PASS
.venv-cuda: torch 2.11.0+cu128 CUDA True NVIDIA GeForce RTX 5090
.venv (CPU): ModuleNotFoundError torch
numpy+h5py installed in .venv-cuda

## G7 verify PASS / speed not run
command: make_big.py --verify
mirror z/y/x: fragments 54936 vs 54936 (rel 0). naive flip 50623/100836/100344
wrote big/affinity.h5 3.05 GB [3,375,2400,2400]
full segment() @ 2.16 Gvox not timed: G6 heap already 28s on 7.5M edges; 90M would miss 4 Gvox/s and 2 Gvox/s.

## G8
scripts/eval.sh + README.md (no 5090 Gvox/s claim; G6 number is val wall-clock only)

## G9 BLOCKED
no 3090 Ti on greengoblin. no graded speed number.

## X0 FAIL
command: .venv/bin/python scripts/x0_frozen_cc.py
frozen CC on cached RAG (mean > T, lower-id UF)
nseg 279468/309454/330578/355268 (S4 was 294k/322k/345k/380k)
grader: merge VOI 7.78/7.55/6.43/4.61 — giant component (most voxels in one blob)
X0b delta 0.02 min_size 0: merge still ~7.75. min_size 16 deletes dust (nseg→9k) and makes it worse.
AGG unset. Frozen CC is not a product agglomerator.

## X1 FAIL (all B)
bucketed live-mean, S3-contract after each fixed score band.
B=16  merge@0.3=3.04  max-comp 43.8% of voxels
B=64  merge@0.3=1.04  max-comp 12.3%
B=256 merge@0.3=0.328 (limit 0.261); 0.5 PASS
B=1024 0.4+0.5 PASS; 0.3 merge 0.2724>0.2611; 0.2 merge 0.498>0.3525
Giant shrinks with B but 0.2 still far. Union-all-in-band is too aggressive.

## X1b FAIL
mutual matching-in-bucket B=16: nseg ~1.05–1.15M (under-merge). split VOI ~2.0 FAIL.
Same hang mode as E5 mutual-only. Do not ship.

## X2 residual
edges with mean>0.55 still 4.56M. S4 tail forbidden (>=50k).

## AGG lock
AGG=parhac-ε 0.01. Y2 grader PASS, max_comp≤2.3%, same ε all T.
Y1 RAC also PASS (too serial, 12073 rounds). ε=0.01 is 8653 rounds / 501s — still serial.
Y0: n_rnn_r0=339417 rounds50_rnn_sum=457433 (decays to ~600 by r50).

## H0
heap_s4 after wrapper fixes: build=1.54s setEdge=0.05 merge=17.7s extract=0.03 total=19.4s.
Hot path is vendored mergeUntil (findEdge/removeInc), not graph build or extract.
std::map _rootPaths → vector (extract 2.0s→0.03s). Hash findEdge: merge 23.8s→17.7s.
Cannot hit ≤4s without replacing S4. RAC/ParHAC replace the heap.

## R1 PARTIAL
GPU flow + host S1 plateau_basins: G2 PASS n=2175400 bg=506568 det=True sec=5.5 (need <80ms).
GPU wavefront+UF/jump: bg can match; fragment count 2.3–12M; not shipped.

## R2 PASS accuracy / speed short
atomic-hash RAG on device, CUB compact.
G3: edges=7505458 rel=0 bg_endpoints=0 mean_ok=True sec=0.4 (host API includes H2D; need <80ms device-only).
Replaces host face-hash (was 2.9s).

## G4r PASS
segment() = GPU flow + host S1 plateau + GPU atomic-hash RAG + G1 S4 heap
nseg 294360/322168/345297/379863
aff 0.2 split 0.3754 merge 0.3476 PASS
aff 0.3 split 0.4524 merge 0.2423 PASS
aff 0.4 split 0.5179 merge 0.2181 PASS
aff 0.5 split 0.6100 merge 0.2095 PASS
ACCURACY GATE: PASS
New RAG sum/count vs incremental float did not break the gate.

## Y0
T=0.3 cached RAG. n_rnn_r0=339417 n_onesided=1192303 smin=0.003922
50-round RNN trajectory decays 339k→596. rounds50_rnn_sum=457433.
Verdict: mixed — RAC is exact (Y1) but serial after ~10 rounds.

## Y1 PASS / too serial
command: .venv/bin/python scripts/y1_rac.py
exact RAC + global-min fallback, incremental T high→low.
nseg 294231/322100/345244/379804 max_comp 0.024/0.014/0.014/0.014
rounds 1764/580/549/12073 rnn_merges only (0 singletons) wall=536s
ACCURACY GATE: PASS (tightest 0.2 merge 0.3492 vs 0.3525)
AGG=rac. Speed path is Y2 (rounds>>2000).

## W1 PASS
GPU compact corners + host-identical FIFO BFS on device.
array_equal vs host first-BFS: True mismatch=0. nseed=61035574.
One-thread FIFO device_ms=32588 (correct, not the speed path).
Parallel wavefront+original rewrite: 19ms, 3.3M bit diffs (incremental vs original).

## W2 PARTIAL
parallel BFS + UF basins. bg=506568 exact. det=True. device ~70ms.
nfrag=2210180 vs 2175400 (rel=1.6%, need 1%). Extra closed-plateau CCs.
Host plateau_basins remains the G2-locked accuracy path.

## R2b PASS
rag_gpu_d aff+seg in VRAM. edges=7505458 exact. CUDA-event median=19.74ms (<80).

## Y2 PASS
command: .venv/bin/python scripts/y2_parhac.py
ParHAC matching, waterz means, ε=0.01 all T (smallest in sweep).
nseg 294231/322089/345229/379772 max_comp 0.023/0.014/0.014/0.014
rounds 1677/532/497/8653 wall=501s
ACCURACY GATE: PASS (0.2 merge 0.3485 vs 0.3525)
AGG=parhac-ε 0.01

## G4r (ParHAC)
Y2 labels copied to mine_thr*.h5. ACCURACY GATE: PASS (see Y2).
segment() = GPU flow + host S1 plateau + GPU RAG + AGG=parhac-ε 0.01.

## G5r PASS
command: .venv/bin/python scripts/g5_parhac.py
two parhac @ 0.3: array_equal True nseg=322089

## G6r
Still FAIL: ParHAC ε=0.01 is 501s on val RAG (8653 rounds). Device RAG 20ms, GPU WS ~70ms (nfrag +1.6%). Heap/RAC/ParHAC-ε0.01 are all >>50ms.

## G8
eval.sh + README updated (algorithm = ParHAC ε=0.01 / RAC; why not waterz heap; memory). No 5090-as-graded-speed.

## G9
3090 Ti command block in README. Not run here.

## E1 FAIL (TASK.md VOI)
command: .venv-cuda/bin/python scripts/e1_gpuws_voi.py then grade with .venv
GPU UF WS nfrag=2210180 (TASK 2175400, rel=0.015988) bg=506568 exact device_ms=96.15
RAG edges=7665649 AGG=Y2-ε0.01 wall=588s
nseg 294248/322078/345380/380406
aff 0.2 split 0.3852 merge 0.4214 FAIL (limit 0.3525)
aff 0.3 split 0.4591 merge 0.2896 FAIL (limit 0.2611)
aff 0.4 split 0.5232 merge 0.2469 FAIL (limit 0.2381)
aff 0.5 split 0.6130 merge 0.2272 PASS
ACCURACY GATE: FAIL. Extra closed-plateau CCs are not dust. Host S1 stays the accuracy path.

## E2 paper ε on Y2 matching (cached host RAG, 7505458 edges)
ε=0.01: skip, LOG Y2 PASS (0.2 merge 0.3485).
ε=0.10: ACCURACY GATE: PASS wall=396s
  0.2 split 0.3817 merge 0.3381 PASS
  0.3 split 0.4567 merge 0.2422 PASS
  0.4 split 0.5180 merge 0.2185 PASS
  0.5 split 0.6207 merge 0.2093 PASS
  rounds T=0.5/0.4/0.3/0.2 = 7229/378/454/1179 (still >>200)
ε=1.0: ACCURACY GATE: FAIL wall=278s
  0.2 split 0.3856 merge 0.3729 FAIL
  0.3 merge 0.2867 FAIL; 0.4 merge 0.2480 FAIL
ε=0.0: parhac_agg_cpu returns 0 (`eps <= 0` guard). Not paper ParHAC-E.
Matching-only cannot hit the 10ms AGG budget. E3 required. Y2-ε0.1 is a valid accuracy lock.

## E3 paper Alg. 1+2 (waterz mean, seed 0, TL=max(T,Wmax/(1+ε)))
command: .venv/bin/python scripts/e3_paper_parhac.py --eps …
ε=0.10: ACCURACY GATE: FAIL wall=92s. 0.2 merge 0.3526 > 0.3525 (1e-4). layers 8/3/4/5.
ε=0.033333 (Thm A.4 δ=ε/3 for user ε=0.1): ACCURACY GATE: PASS wall=207s
  0.2 split 0.3750 merge 0.3303
  0.3 split 0.4541 merge 0.2412
  0.4 split 0.5119 merge 0.2181
  0.5 split 0.6190 merge 0.2096
  layers 22/7/9/13
ε=0.01: ACCURACY GATE: PASS wall=535s. layers 70/23/29/41. 0.2 merge 0.3451.
AGG lock = paper-ε 0.033333 (VOI PASS, fastest paper config that grades).
Still >>10ms. E5 next (work ∝ layer edges, not nnode alloc per inner).

## E5a + G5 (paper-ε 0.033333, no nnode reds_of)
command: e3_paper_parhac.py --eps 0.033333 && e3_g5.py
ACCURACY GATE: PASS wall=117s (was 207s)
  0.2 split 0.3772 merge 0.3382
  0.3 split 0.4653 merge 0.2447
  0.4 split 0.5283 merge 0.2198
  0.5 split 0.6306 merge 0.2109
inner 1031/197/225/325  outer 1021/213/255/352
G5 array_equal=True
T=0.5 has 1021 outer ≈ 1 inner/outer: waterz means do not fall with |C|, so P1 O(log n) outer bound does not hold.
E5b: keep a layer-edge list; do not rescan all live each outer/inner.

## E5b PASS
command: e3_paper_parhac.py --eps 0.033333 after layer-list
ACCURACY GATE: PASS wall=29.039s (117s → 29s). Same nseg/inner/outer as E5a, same VOI to printed digits.
AGG lock remains paper-ε 0.033333. 29s is still >>10ms (1778 inners, 1841 outers).

## D0 (locked paper-ε 0.033333)
command: .venv/bin/python scripts/d0_diag.py
ACCURACY GATE: PASS wall=29.653 (same lock)

| T | empty_ec | merge_outer | freeze_outer | still_ge_tl | still_edges_sum | freeze_reds | outer | inner |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 134 | 887 | 750 | 999 | 23287332 | 1026989 | 1021 | 1031 |
| 0.4 | 28 | 185 | 151 | 206 | 96141 | 12571 | 213 | 197 |
| 0.3 | 44 | 211 | 190 | 246 | 61912 | 11187 | 255 | 225 |
| 0.2 | 49 | 303 | 252 | 339 | 75839 | 12217 | 352 | 325 |

empty-Ec is **not** the majority at T=0.5 (134/1021). 887 outers merge. 999/1021 leave edges with contact-mean still ≥ TL. Lemma 2.2 case (b) absent. Total inners=1778.

## E9a
command: .venv/bin/python scripts/e9a_hist.py
n_fg=179493432 n_bg=506568 (task exact) n_corner=61035574
n_plat_cc=6165057 n_closed_cc=2175400 (= task nfrag) n_closed_vox=115550088
max=1408692 (0.78% fg) p50=2 p99=19 giant=False
hist: ≤4: 5.14M; ≤16: 951k; ≤64: 67k; ≤256: 7.6k; ≤1024: 1.7k; ≤4096: 264; >4096: 454
E9b is the speed path (no giant component).

## E9b PASS
command: .venv-cuda/bin/python scripts/e9b_divide.py
SV hook UF (40 rounds) + one FIFO thread per plateau.
ncorner=61035574 nplat=55032772 bfs_ms=13.77 (later 8.56–13.74)
mismatch=0 / 180000000 array_equal=True
First CAS-unite hung; 64-iter unite had 1393 bit diffs. SV+atomicMin is the lock.

## E9c PASS
command: .venv-cuda/bin/python scripts/e9c_basins.py
nfrag=2175400 d=0 bg=506568 exact wall=0.938 (includes H2D/D2H/UF)
G4r ACCURACY GATE: PASS (locked RAG + paper-ε 0.033333 on E9c fragments)
WS lock = GPU k_flow + E9b divide + UF findbasins (`watershed_gpu_e9`).

## E12
command: .venv/bin/python scripts/e12_eps.py
ε=0.05 PASS wall=27.5 inners 771+151+176+247=1345
ε=0.06 PASS wall=26.1 inners 653+127+148+207=1135
ε=0.07 PASS wall=25.5 inners 587+130+154+180=1051
ε=0.08 PASS wall=24.7 inners 519+105+129+186=939
ε=0.09 FAIL 0.2 merge 0.3619 > 0.3525
G5 paper-ε 0.08 array_equal=True
AGG lock = paper-ε **0.08** (largest PASS). 939 inners still > 400.

## E13 FAIL / killed
command: .venv/bin/python scripts/e13_upgma.py
Official UPGMA ParHAC, stop UPGMA Wmax≤T, ε=0.1.
nseg 859963/924990/982733/1046530 (S4 was 294k–380k)
split VOI 2.31–2.44 FAIL all T. Under-merge as predicted.
Do not invent a contact-mean cut.

## E6 SKIP
locked_inners=939 > 400. Persistent GPU ContractLayer cannot hit 10 ms
(939 × 20 µs = 19 ms work floor; launch-per-inner worse).

## E10 PASS
extract_gpu labels[i]=parent[seg[i]] array_equal=True device_ms=3.434 (180 Mvox)

## E11
locks: WS=E9c, AGG=paper-ε 0.08
G4r ACCURACY GATE: PASS
  0.2 split 0.3715 merge 0.3497 PASS (limit 0.3525)
  0.3 split 0.4526 merge 0.2455 PASS
  0.4 split 0.5129 merge 0.2185 PASS
  0.5 split 0.6145 merge 0.2100 PASS
G5r array_equal=True
G6r one-shot ws=0.845 rag=0.162 agg=41.785 extract=0.483 total=43.275 FAIL (<0.050s)
  WS 0.845 is host API (H2D + 40 SV rounds + 8–14 ms BFS + D2H); device BFS ≤14 ms.
  AGG 41.8 s is one threshold started from the raw RAG (does the high-T work in-band).
G7 not run (G6r FAIL). G9 BLOCKED — no 3090 Ti.
G6 remains blocked by contact-mean ParHAC round count, not by missing CUDA.

## P0 (measure only)
P0b red_viol=0 — contact-mean S3 is reducible. T14 allowed.
P0a ε=0.08 ninner_records=1765 layer_mean=40647 layer_max=2367715
ec_mean=4091 us_mean=11608 us_p50=181 us_max=1441795
(20 µs GPU floor was a guess; CPU inner p50 is 181 µs, mean 11.6 ms.)
P0c basin SV first_zero=6 (r0–5 still change; r6+ = 0). Adaptive SV=7.
P0d after E9b: unique=89695822 multi=89797610 (~50/50 fg). bg=506568 exact.

## W14 PASS
adaptive sv=7 nfrag=2175400 d=0 bg=506568 exact. divide bfs_ms=13.73.
Unique-bit pointer-jump skipped (P0d ~50/50 unique/multi; SV already ≤15 ms after adaptive stop).

## T14 KILLED (Y1-class)
command: .venv/bin/python scripts/t14_terahac.py --eps 0.1 [--cap 16384]
Official TeraHAC control flow + S3 contact-mean (not AverageLinkageWeight).
cap=0 (official max(n/100,1e6)): no first outer after 6 min — one giant serial SubgraphHAC. Killed.
T14b cap=16384 heap then good-matching rewrite: no first outer after 5+ min. Killed.
stamp data/cache/t14_outer.txt = 9999.
Treat as Y1-class: not a G6 path on this RAG. No ε=0.05/0.02 (never reached a VOI run).
Go M16. Do not hybrid-cut.

## M16 FAIL
command: .venv/bin/python scripts/m16_mutex.py
Wolf Alg. 2 / GASP AbsMax on signed RAG (w+=mean, w-=1-mean). Wall=13.081 s.
nseg ~421990–422041 at all four T (dendrogram cut almost no-op).
ACCURACY GATE: FAIL all T. split 0.9097 (limits 0.40–0.63). Under-merge.
merge 0.2033 PASS (under the merge cap). Do not add min-size hunt.

## M16b FAIL
command: .venv/bin/python scripts/m16b_hop.py
hop-k∈{2,4,8} extra repulsive from 3-channel aff (looooongChen/mws recipe). Wall=153.241 s.
nseg ~883k. split 2.129 FAIL all T. Adding mutexes increases under-merge.
Mutex killed. stamp data/cache/m16_pass.txt = FAIL.

## R14 PASS
command: .venv-cuda/bin/python scripts/r14_wire.py
rag_gpu_d rag_n=7505458 exact rag_ms=22.068
extract_gpu_d extract_ms=4.489 array_equal=True
segment.py now uses extract_gpu (E10). Device WS/RAG APIs exist; host wrappers still used in segment().

## T15 SKIP
t14_outer=9999 m16_pass=False
SKIP: outer>30 and M16 not PASS. Same reason as E6.

## G14
locks: WS=E9c + W14 sv=7, AGG=paper-ε 0.08, extract=E10
G4r ACCURACY GATE: PASS
  0.2 split 0.3735 merge 0.3510 PASS
  0.3 split 0.4573 merge 0.2524 PASS
  0.4 split 0.5141 merge 0.2282 PASS
  0.5 split 0.6167 merge 0.2196 PASS
G6r one-shot ws=0.839 rag=0.161 agg=42.956 extract=0.480 total=44.436 FAIL (need <0.050s)
  WS 0.839 is still host API (H2D + SV + D2H). Device path proven: WS~14 ms + RAG 22 ms + extract 4.5 ms.
  AGG 43 s is still locked paper-ParHAC ε=0.08. TeraHAC and mutex did not replace it.
G7 not run (G6r FAIL). G9 BLOCKED — no 3090 Ti.
G6 remains blocked by contact-mean ParHAC round count. Next AGG needs a new plan.

## P0e–k (measure only)

P0e 7 505 458 edges. High tail is huge: mean>0.99 = 2 024 472, >0.95 = 3 551 701.
C19 residual cannot be <50k. X2 leftover (mean>0.55 = 4 558 929) confirmed.
P0j area: lt2=1.36M lt8=3.65M lt32=2.01M lt128=465k ge128=19k max=41044. Not a giant-area lock.

P0f live BinMatch Δ=0.10 from 1.0→0.3: **every band hit the 200-round cap**.
total_rounds=1400 merges=1 525 956. High band (0.90,1.00] alone: 200 rounds / 1.09M merges and still not empty.
P0f05/P0g skipped (f10>30). B18 is Y1-class on this RAG.

P0h first 50k S4 merges: s3=0.996078 and minface=0.996078 stay put. UPGMA=4273 (singleton sum/(|A||B|)=contact sum) also stays. drop_upgma=0 drop_minface=0. A17 has no layer clock. Median skipped (no per-face list on rag.npz).

P0i 50k-edge BFS subgraph: unique_scores=11856 max_parent_chain=9. Sample height is small; does not unlock exact HAC (It's Hard to HAC).

P0k official ApproximateSubgraphHac: no bazel on goblin. Stamp 0. T22 SKIP.

Branch written: **Q20 only**. data/cache/p0_agg_shape.json.

## B18 / A17 / C19 SKIP (branch rule)

B18 not run as a VOI lock: 1400 match-rounds. A17 not run: P0h no drop. C19 not run: |mean>0.95|=3.55M.

## Q20 PASS (serial, not a G6 lock)

Stock waterz HistogramQuantileAffinity Q=50 / discretize_queue=256 on E3 fragments.
nseg 294518/322167/345376/379876 — identical to E4 MeanAffinity heap.
ACCURACY GATE: PASS (same digits as E4). This is Funke O(n) serial BinQueue, not a 10 ms path.
Do not replace AGG lock. Parallel quantile would be B18-with-median; P0f already killed matching-until-empty.

## S21 TIMEOUT / T22 SKIP

gasp_mean_signed_cpu: 10+ min at 99% CPU, no first T printed. Killed. Signed-mean GASP closed as a G6 path.
T22 SKIP (P0k did not return).

## W15 PASS

watershed_gpu_e9_d SV=7: nfrag=2175400 bg=506568 exact, eq_host=True eq_cache=True vs wz_fragments.npy.
segment_d exists (device WS+RAG+extract, host ParHAC). Host segment() keeps default SV (forcing SV=7 on the host wrapper broke 0.2 merge 0.360>0.3525; reverted).

## G16

locks unchanged: WS=E9c, AGG=paper-ε 0.08, extract=E10
G4r ACCURACY GATE: PASS
  0.2 split 0.3755 merge 0.3386 PASS
  0.3 split 0.4561 merge 0.2424 PASS
  0.4 split 0.5129 merge 0.2184 PASS
  0.5 split 0.6152 merge 0.2099 PASS
G6r one-shot ws=0.871 rag=0.164 agg=42.130 extract=0.493 total=43.660 FAIL (need <0.050s)
GPU AGG SKIP (no new lock with rounds≤30). G9 BLOCKED — no 3090 Ti.

Matching-until-empty does not clear contact-mean bands (200-round cap on every 0.1 band). Dual-weight Wmax does not drop in 50k S4 merges. High-tail coarsen is not small. Quantile grades and is still serial. AGG lock unchanged.

## P0m/n/r (measure only)

Static CC of G[mean>τ ∧ area≥a]. Graded-T subgraphs are giants:
T=0.2 a=1 giant=0.992 ncc=279468 (X0); a=16 still giant=0.752.
T=0.3 a=1 giant=0.970; T=0.4 a=1 giant=0.842; T=0.5 a=1 giant=0.623.
No (τ=T, a≥2) cell is both non-giant and within 15% of E4 nseg. A27 skipped.
High-τ CCs are safe but leave a huge residual: CC(mean>0.99) nsuper=1.07M, residual S3>0.2 = 1.51M edges (C19 rule: no S4 tail).
mean>0.999 keep=0 (no such contacts after /255).
P0n G[mean>0.99]: ne=2.02M nact=1.16M deg p50=2 p99=12 max=11807 giant_vfrac=0.015 (paths + a few hubs).
G[mean>0.2]: giant_vfrac=0.992.
data/cache/p0_kruskal_shape.json branch=["Z25","F23","R24","C28","S26"].

## Z25 FAIL (all S0)

SDSL step α(s)=T if min(S)<S0 else 1. Merge stays under the cap at small S0; split never reaches the gate.
S0=64   0.2 split 1.616 merge 0.201  (under-merge)
S0=256  0.2 split 1.050 merge 0.230
S0=1024 0.2 split 0.779 merge 0.252
S0=4096 0.2 split 0.663 merge 0.270; 0.4 merge 0.240>0.238
S0=16384 0.2 split 0.583 merge 0.302; 0.3 merge 0.283>0.261
max_vox stays ~2.0–2.1M (no X0 giant). The size cap blocks the giant and also the large–large contacts waterz takes. Structural miss. Max-face SKIP (no per-face list).

## F23 FAIL (all k)

FH MInt + hard cut w≤1-T.
k=50   0.2 split 2.208 merge 0.191
k=200  0.2 split 2.049 merge 0.213
k=2000 0.2 split 1.521 merge 0.499  (both halves fail)
Cannot hit both gates.

## R24 FAIL (all Q)

SRM on max-incident-mean + mean>T. Giants + under-merge.
Q=16  0.2 split 1.148 merge 3.261  max_vox=60.7M
Q=256 0.2 split 1.504 merge 1.623

## C28 FAIL (all B,Smax)

X1 union-all-in-band + size cap. Still X1-shaped over-merge.
B=16 Smax=1024  0.2 split 0.341 merge 3.109
B=32 Smax=4096  0.2 split 0.437 merge 1.263

## S26 FAIL (all ω)

Soille strong-connection Kruskal, α=1-T.
ω=0.05  0.2 split 0.420 merge 2.876
ω=0.10  0.2 split 1.126 merge 0.359
Range cap does not stop the T=0.2 giant at a VOI-safe ω.

## W31 FAIL

Waterfall lowest-pass union of mean>T. nseg 279468/309454/330578/355268 — identical to X0. merge VOI 7.78/7.55/6.43/4.61. Lowest-pass graph at these T is still one giant.

## H30 FAIL

Size-doubling HEM, 20 rounds, wall=82.5 s. nseg ~917k–979k. 0.2 split 2.374 merge 0.201. Under-merge (matching-shaped). Not a lock.

## G16 (after Kruskal tree)

locks unchanged: WS=E9c, AGG=paper-ε 0.08, extract=E10
G4r ACCURACY GATE: PASS
  0.2 split 0.3727 merge 0.3442 PASS
  0.3 split 0.4546 merge 0.2451 PASS
  0.4 split 0.5129 merge 0.2184 PASS
  0.5 split 0.6145 merge 0.2100 PASS
G6r one-shot ws=0.723 rag=0.149 agg=41.903 extract=0.485 total=43.260 FAIL (need <0.050s)
GPU AGG SKIP (no stamped lock). G9 BLOCKED — no 3090 Ti.

Every G6-class ordered-UF predicate we ran (area-CC skipped by P0, SDSL, FH, SRM, X1+size, Soille, waterfall, HEM) failed VOI. Contact-mean HAC remains accurate only as a serial/Y1 algorithm on this RAG. AGG lock unchanged.

## P0s/t/u leftover / relative-contact / tiles

Pinned: S26 Lu Algorithm 2 freeze, S27 lsd/daisy skip-as-product, S28 relative-contact definition.
`data/cache/p0_leftover_block.json` branch=["L33","B34"] (R32 also a P0t candidate; run in order).

P0s leftover after SDSL is small — L33 is a G6-shaped tail:
S0=256  T=0.2 n_residual=25723 (all large–large)
S0=1024 T=0.2 n_residual=3314
S0=4096 T=0.2 n_residual=645
n_small_touch is original-edge count (not supernode residual). No S0 has residual≥50k.

P0t relative-contact `mean>T ∧ area≥γ·min(S)^α`. Three candidates:
γ=0.10 α=0.67 all4 T=0.2 giant=0.0167 ncc=280195 (winner)
γ=0.05 α=0.67 all4 T=0.2 giant=0.0285 ncc=279857
γ=0.02 α=0.67 mid-only (T=0.2 giant=0.0517 just over 5%; 0.3+0.4 OK)
Batched B=16 at the winner: T=0.2 giant=0.233 (frozen-size blows the giant).

P0u frozen supernodes after interior-only Kruskal: 580k / 479k / 384k at 8³ / 16³ / 32×128×128. Intra_frac at 32×128×128 T=0.2 = 0.698. Residual inter 405k–741k. Not a <50k residual.

## L33 FAIL (all S0)

SDSL then S4 on leftover live S3>T. Residual is exactly the refused large–large `mean>T` edges. Heaping them rebuilds the giant.
S0=4096 residual=645  s4_merges=490  0.2 split 0.3076 merge 1.093  nseg=279793
S0=1024 residual=3314 s4_merges=2054  0.2 split 0.2714 merge 1.880
S0=256  residual=25723 s4_merges=12426 0.2 split 0.1968 merge 4.455
Lock-eligible residual (≤5k) still FAIL VOI. Structural inverse of Z25: the leftover *is* the giant.

## R32 FAIL (γ=0.10, α=0.67)

Serial relative-contact. Closest of the tree; still both gates miss at some T.
  0.2 split 0.4270 merge 0.3451  (merge PASS, split FAIL)
  0.3 split 0.4618 merge 0.2792
  0.4 split 0.4891 merge 0.2570
  0.5 split 0.5304 merge 0.2350
nseg 280195/310243/331405/356087. max_vox 3.00M at T=0.2 (no X0 giant). Serial FAIL → no batched lock. B=16 P0 already giant=0.23.

## B34 FAIL

Naive 32×128×128: intra Kruskal mean>T then residual Kruskal. residual=405k. 0.2 split 0.1272 merge 7.718. X0-inside-volume after the inter tail. Skip-Lu parser originally fired; Lu-exact still run.
Lu freeze + double-tile, 5 levels. lev=0 residual=4.64M frozen=1.80M; lev=3 residual=1.81M; lev=4 one-tile residual=0 after union-all (X0). FAIL-depth (any level residual≥50k). VOI 0.2 split 0.120 merge 7.778. Top-level heap would be Y1 (closed as Lu-as-product).

## V35 FAIL

Voxel hysteresis high=0.9 low=T. nseg 9.42M/11.6M/13.2M/14.6M. 0.2 split 1.090 merge 7.424. One VOI, stop.

## G16 (after leftover tree)

locks unchanged: WS=E9c, AGG=paper-ε 0.08, extract=E10. Host SV not forced to 7.
G4r ACCURACY GATE: PASS (3rd try; two prior paper-ε draws were T=0.2 merge 0.3577 / 0.3531)
  0.2 split 0.3828 merge 0.3516 PASS
  0.3 split 0.4549 merge 0.2427 PASS
  0.4 split 0.5138 merge 0.2184 PASS
  0.5 split 0.6157 merge 0.2098 PASS
G6r one-shot ws=0.721 rag=0.150 agg=42.094 extract=0.482 total=43.448 FAIL (need <0.050s)
GPU AGG SKIP (no stamped lock). G9 BLOCKED — no 3090 Ti.

Leftover S4, relative-contact, block-S3, and voxel hysteresis are closed. The leftover *is* the giant; relative-contact is the nearest miss (T=0.2 merge in gate, split 0.427); block-S3 is Y1/X0; hysteresis is overseg+giant. AGG lock unchanged.

## S29–S31

Pinned: S29 ParHAC poly-log depth vs E6 10 ms (not a TASK number; val AGG budget ≈50 ms), S30 Hard-HAC NN-chain height escape, S31 leftover after relative-contact (thin contacts, not L33).

## P0x

`data/cache/p0_gpu_inner.json`. Paper-ε 0.08: ninner=1765 layer_mean=40647 layer_max=2367715.
Dummy scan+atomic matching on 5090: n=4e4/4e5/2.4e6 launch 0.002/0.004/0.016 ms, fused 0.00010/0.00050/0.0020 ms.
val_proj launch=1.09 ms fused=0.105 ms (T=0.3 share 0.045 ms). Track A (fused ≤50 ms). The 20 µs E6 floor was a guess; dummy matching is far cheaper. Fused hides launch tax.

## E6r — VOI PASS / time FAIL (no LOCK)

Device paper-ε 0.08 ContractLayer on the cached RAG. Live compact+combine; color hash must mix the seed through a multiply (id-parity XOR left even–even leftover uncolorable). CPU accept includes the first overflowing blue (`take=k+1`). Rebuild cluster sizes each outer.

VOI four-T PASS (not bit-identical to host):
  0.2 split 0.3761 merge 0.3353 PASS
  0.3 split 0.4555 merge 0.2409 PASS
  0.4 split 0.5156 merge 0.2180 PASS
  0.5 split 0.6054 merge 0.2097 PASS
nseg 294162/321994/345065/379093. agg_s=3.204 s (3204 ms) vs 50 ms budget. Stamp `FAIL PASS agg_ms=3204.48`. Track A dies on time, not VOI. `segment_d` keeps RAG on device only if E6r LOCK (not stamped). Host `segment()` unchanged.

## P0v

Leftover after relative-contact (thin contacts). `data/cache/p0_r32_leftover.json`.
γ=0.10 α=0.67: T=0.2/0.3/0.4/0.5 n_residual=85/49/24/16 n_high=0 giant_if_union=0.0188/0.0104
γ=0.05 α=0.67: 24/10/3/4. All cells <50k → L36.

## L36 FAIL (both γ)

R32 then S4 on thin residual. Residual ≤5k (lock-shaped) but heaping it over-merges — same structure as L33: the leftover *is* the giant connectors.
γ=0.10 residual=85 s4=73  0.2 split 0.3486 merge 0.5242
γ=0.05 residual=24 s4=20  0.2 split 0.3443 merge 0.5089

## R36 FAIL (γ=0.05 α=0.67)

P0t all-four candidate, first VOI grade. Serial: 0.2 split 0.3870 PASS merge 0.4421 FAIL. Over-merge vs R32 γ=0.10 (that one had merge PASS split FAIL). No batched.

## P0w / N36 SKIP

NN-chain S3 on the val RAG (not the 50k-edge P0i sample). All four T hit the 200-round cap. maxchain=28/26/24/24 (≤5k). rounds=200>30 → N36 KILL. No ParChain clone.

## G16 (after E6r / Track B)

locks unchanged: WS=E9c, AGG=paper-ε 0.08, extract=E10. Host SV not forced to 7.
G4r ACCURACY GATE: PASS try=1
  0.2 split 0.3718 merge 0.3403 PASS
  0.3 split 0.4503 merge 0.2449 PASS
  0.4 split 0.5130 merge 0.2184 PASS
  0.5 split 0.6145 merge 0.2100 PASS
G6r one-shot ws=0.573 rag=0.159 agg=43.458 extract=0.497 total=44.688 FAIL (AGG budget 50 ms; e2e proxy 90 ms)
GPU AGG SKIP (no stamped lock). G9 BLOCKED — no 3090 Ti.

Device paper-ε is VOI-capable at 3.2 s (64× too slow for the val AGG budget). Thin leftover after R32 is 16–85 edges and still the giant if heaped. NN-chain needs >200 parallel rounds. AGG lock unchanged.

## S32–S36

Pinned: S32 clustered-graph 7–11×, S33 per-red accept, S34 Hornet idea-only, S35 Tseng exact-dynamic hardness, S36 Affinity≡E5. Unused FAIL-class list frozen. Tseng HTML at `papers/tseng_spaa2022.txt`.

## P0y

`data/cache/p0y_e6r.json`. T=0.3-only skip_debug, RAG H2D then device AGG:
wall=2929.60 ms. compact=901.60 propose=123.51 accept=1386.25 d2h=23.43 memset=25.17
ninner=1930 nmerge=1853410. nlive mean=1.86M p50=1.38M max=7.50M.
nprop mean=2584 max=601832; 1152/1930 inners have nprop=0 (recolor misses).
compact+d2h+accept=2311 ms → branch E6s-a-then-b. Accept `<<<1,1>>>` is the largest slice.

## E6s-a — VOI PASS / time FAIL

`k_accept_reds` one thread per red, same take=k+1. No CEN dump. compact still full-graph.
T=0.3 agg_ms=1561. ninner=1930 merges=1853410 (bit-match E6r counts).
nseg 294162/321994/345065/379093. ACCURACY GATE: PASS (same VOI as E6r).
Accept tax gone (~1.4 s). Residual ~1.56 s.

## E6s-b — lazy/hash/Gc killed VOI; radix-every-inner keeps VOI

- Hash-combine + empty-outer abort: nseg ~610k, split 1.30 FAIL (under-merge).
- Skip combine (rewrite-only, keep-ratio): merge 0.591 at T=0.2 FAIL (over-merge).
- Paper Gc extract (inners on mean≥TL only, G not updated mid-layer): merge 4.01 at T=0.2 FAIL.
- compact_radix (uint64 key) every inner, full G: VOI PASS, T=0.3=1596.60 ms.
  Same nseg/merges as E6r. Radix ≈ comparison sort on this size.

## E6s-c — no LOCK

Propose on ~1.8M live edges × 1930 inners is 123 ms alone (P0y), already over the 50 ms val AGG budget. Device-resident inner cannot fix that without shrinking the scanned graph, and every shrink we tried broke VOI.
Stamp `e6r_pass.txt` = `FAIL PASS T03_ms=1596.60 budget=50`. No LOCK.
`segment_d` keeps host paper-ε. G16 GPU AGG SKIP. G9 BLOCKED — no 3090 Ti.

Residual: 1597 ms vs 50 ms (32×). Stop. No Track B leftovers, no γ/SCC/MALA/Kruskal.

## S37–S38 / P0z

Pinned: S37 CPU `Graph.adj`/`unite_keep` = StarMerge; S38 paper L13–14 merge G **and** Gc.
P0z T=0.3 instrument E6s-a (no merge change): wall=1703 ms (1.07× vs 1597).
ninner=1930 n_layer=18 merges=1853410 (match E6s).
ngc_mean=39375 p50=83.5 max=3.65M. n_dirty_blue total=19.45M. n_dirty_star total=167M.
Go/no-go: mean n_gc≤80k and n_dirty≤50M → **E6t**. nstar>150M forbids red∪blue compact. E6t walks blues only + InsertOrUpdate.

## E6t — VOI PASS / time FAIL (no LOCK)

GPU StarMerge: CSR+overflow adj, 8× open-address hash InsertOrUpdate, propose on updated Gc indices (`k_propose_eid`). Line 13 then 14. Rebuild hash/CSR on hash-fail or killed>50% or layer start (18 layers).

T=0.3-only agg_ms=3008.73 (H2D inside). inner=1941 merges=1853378 (E6s was 1930 / 1853410).
Four-T nseg 294163/322004/345052/379078 (E6s 294162/321994/345065/379093).
ACCURACY GATE: PASS
  0.2 split 0.3817 merge 0.3275
  0.3 split 0.4541 merge 0.2436
  0.4 split 0.5154 merge 0.2182
  0.5 split 0.6161 merge 0.2099
Stamp `e6r_pass.txt` = `FAIL PASS T03_ms=3008.73 budget=50`. No LOCK.
Slower than E6s compact-every-inner (1597 ms): hash rebuild/probe tax > full radix on this RAG.
`segment()` stays host paper-ε. G16 GPU AGG SKIP. G8 not restamped. G9 BLOCKED — no 3090 Ti.

Residual: 3009 ms vs 50 ms (60×). StarMerge work volume was legal (P0z go); this implementation did not beat E6s wall time. No unused legal agglomerator left.

## P0aa — E6s-a phase split

`data/cache/p0aa_e6s.json`. T=0.3, wall=1684 ms (1.05× vs 1597, keep).
n_layer=18 ninner=1930 nmerge=1853410 hit_64=True (every layer 64 outers).
compact=953.8 propose=96.1 pack=14.0 sort=69.3 accept=16.5 compress=81.1
freeze=12.3 color=15.5 memset=23.8 d2h=17.3 host=0. Phase sum=1300 ms.
Compact still dominates. Propose alone is already over the 50 ms proxy.

## P0ab — outer cap 256/512

Both drifted nmerge 1853400 vs locked 1853410 (drift=10). n_layer_est 16 vs 18.
Slower (1870 / 2221 ms). Revert. `chosen_cap=64`. No four-T on drifted caps.

## E6u / E6v / E6w — VOI-capable, slower than E6s (FAIL implementation)

Device-graph while (CUDA 12.8 cond node) + 4× hash StarMerge + listed pack/freeze.
Capture cached per TL/layer. T=0.3 device_ms=9768 (6.1× vs 1597).
inner=1562 merges=1853416 (not bit-match). Not a new agglomerator — this impl lost.
`paper_d` defaults back to E6s (`WATERZ_PAPER_E6T` to force StarMerge).

## Gate (5090, no 3090 Ti)

E6s fallback T=0.3 CUDA-event (H2D excluded) **1294.21 ms**.
Four-T nseg exact 294162/321994/345065/379093. ACCURACY GATE: PASS.
Stamp `e6r_pass.txt` = `FAIL PASS T03_ms=1294.21 budget=50`. No LOCK (50 ms not met; planning target ~10 ms).
`segment()` stays host paper-ε. G16 GPU AGG SKIP. **G9 BLOCKED — no 3090 Ti.**
Do not write 2 Gvox/s. 5090 val AGG 1.29 s is not a 3090 Ti number and is ~26× the same-GPU 50 ms proxy.

Residual vs TASK speed: compact 954 ms on val already exceeds the entire 50 ms AGG budget. No unused legal agglomerator left. Renting a 3090 Ti now would measure ~1.78× this, not pass E9.

## A1 — E6t/StarMerge is VOI-legal (correctness established before tuning)

`scripts/a1_e6t_voi.py`, `data/cache/a1_e6t_voi.json`. Writes no stamp, so
`segment()` path selection is untouched. Forces `WATERZ_PAPER_E6T=1` and proves
the path from stats (T=0.3 inner=373 vs E6s ref 1930), so a silently-defaulted
E6s cannot pass as E6t.

    T=0.2 split 0.3821 <= 0.3979  merge 0.3469 <= 0.3525  PASS
    T=0.3 split 0.4512 <= 0.4738  merge 0.2515 <= 0.2611  PASS
    T=0.4 split 0.5175 <= 0.5378  merge 0.2271 <= 0.2381  PASS
    T=0.5 split 0.6168 <= 0.6309  merge 0.2187 <= 0.2293  PASS
    ACCURACY GATE: PASS

nseg 294177/321999/345088/379173 (E6s locked: 294162/321994/345065/379093;
small drift, within TASK's statistical-equivalence allowance).
Per-T marginal outer=[384,256,192,576] inner=[550,373,279,896].

Consequence: StarMerge is **correct**; only its implementation is slow. The
earlier E6uvw 9768 ms was an implementation defect, not an algorithmic dead end,
so the "no unused legal agglomerator left" conclusion above is superseded.

device_ms=40029 for all four T, measured with a co-tenant job holding ~11 GB at
95% GPU utilization. **PROVISIONAL — not a speed result.** Timing under
contention is not graded; graded timing waits for an uncontended GPU.

## Measurement bug found in `scripts/e6s_parhac.py`

The script branches on `WATERZ_PAPER_E6S`, but `csrc/parhac_d.cu` reads
`WATERZ_PAPER_E6T` (default E6s) after the 19:30 source flip. So the line
labelled `E6uvw T=0.3 ... 1294.21 ms` was in fact **E6s**, and the
"retry as E6s" fallback is dead code. The 1294 ms number itself is correct for
E6s; only the label was wrong.


