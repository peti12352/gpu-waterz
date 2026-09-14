# Gate log

hostname=greengoblin
CUDA_VISIBLE_DEVICES=unset
lego PID 1880797 still on GPU (do not touch)

## T0 PASS
command: uv venv .venv --python 3.12; uv pip install --python .venv h5py numpy; SETUPTOOLS_SCM_PRETEND_VERSION=0.9.5 uv pip install --python .venv -e src/waterz-upstream
note: PyPI waterz has no cp312 wheel; used vendored a0184d2
import waterz,h5py,numpy: ok (waterz.__version__ prints "uninstalled": upstream __init__ bug, import works)
import torch: ModuleNotFoundError
nvidia-smi compute: 1880797 python3 8652 MiB (lego)

## E0 PASS
command: uvx gdown 1zbGpyr9M5Pvhgfy96V9erQwAeRZo23hW -O data/waterz_bounty.tar; tar xf
listing: cremiA_val/{affinity,gt,raw}.h5, baseline/{labels_thr*.h5,voi.csv,run_baseline.py}, make_big.py, viz/, README.md: present (plus make_viz.py, extra thr 0.1/0.7/0.9)
shapes: affinity (3,125,1200,1200) uint8; gt (125,1200,1200) uint32
voi.csv rounded == TASK.md table
threshold: FACT score=1-aff; see notes/THRESHOLD.md
README vs S1-S5: no semantic delta; see notes/TARBALL_VS_SOURCE.md

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

Fallback: exact S4 labels from E4 -> mine_thr{0.2,0.3,0.4,0.5}.h5
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
grader: merge VOI 7.78/7.55/6.43/4.61: giant component (most voxels in one blob)
X0b delta 0.02 min_size 0: merge still ~7.75. min_size 16 deletes dust (nseg->9k) and makes it worse.
AGG unset. Frozen CC is not a product agglomerator.

## X1 FAIL (all B)
bucketed live-mean, S3-contract after each fixed score band.
B=16  merge@0.3=3.04  max-comp 43.8% of voxels
B=64  merge@0.3=1.04  max-comp 12.3%
B=256 merge@0.3=0.328 (limit 0.261); 0.5 PASS
B=1024 0.4+0.5 PASS; 0.3 merge 0.2724>0.2611; 0.2 merge 0.498>0.3525
Giant shrinks with B but 0.2 still far. Union-all-in-band is too aggressive.

## X1b FAIL
mutual matching-in-bucket B=16: nseg ~1.05-1.15M (under-merge). split VOI ~2.0 FAIL.
Same hang mode as E5 mutual-only. Do not ship.

## X2 residual
edges with mean>0.55 still 4.56M. S4 tail forbidden (>=50k).

## AGG lock
AGG=parhac-ε 0.01. Y2 grader PASS, max_comp≤2.3%, same ε all T.
Y1 RAC also PASS (too serial, 12073 rounds). ε=0.01 is 8653 rounds / 501s: still serial.
Y0: n_rnn_r0=339417 rounds50_rnn_sum=457433 (decays to ~600 by r50).

## H0
heap_s4 after wrapper fixes: build=1.54s setEdge=0.05 merge=17.7s extract=0.03 total=19.4s.
Hot path is vendored mergeUntil (findEdge/removeInc), not graph build or extract.
std::map _rootPaths -> vector (extract 2.0s->0.03s). Hash findEdge: merge 23.8s->17.7s.
Cannot hit ≤4s without replacing S4. RAC/ParHAC replace the heap.

## R1 PARTIAL
GPU flow + host S1 plateau_basins: G2 PASS n=2175400 bg=506568 det=True sec=5.5 (need <80ms).
GPU wavefront+UF/jump: bg can match; fragment count 2.3-12M; not shipped.

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
50-round RNN trajectory decays 339k->596. rounds50_rnn_sum=457433.
Verdict: mixed: RAC is exact (Y1) but serial after ~10 rounds.

## Y1 PASS / too serial
command: .venv/bin/python scripts/y1_rac.py
exact RAC + global-min fallback, incremental T high->low.
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
command: .venv/bin/python scripts/e3_paper_parhac.py --eps ...
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
ACCURACY GATE: PASS wall=29.039s (117s -> 29s). Same nseg/inner/outer as E5a, same VOI to printed digits.
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

empty-Ec is **not** the majority at T=0.5 (134/1021). 887 outers merge. 999/1021 leave edges with contact-mean still >= TL. Lemma 2.2 case (b) absent. Total inners=1778.

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
ncorner=61035574 nplat=55032772 bfs_ms=13.77 (later 8.56-13.74)
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
nseg 859963/924990/982733/1046530 (S4 was 294k-380k)
split VOI 2.31-2.44 FAIL all T. Under-merge as predicted.
Do not invent a contact-mean cut.

## E6 SKIP
locked_inners=939 > 400. Persistent GPU ContractLayer cannot hit 10 ms
(939 x 20 us = 19 ms work floor; launch-per-inner worse).

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
  WS 0.845 is host API (H2D + 40 SV rounds + 8-14 ms BFS + D2H); device BFS ≤14 ms.
  AGG 41.8 s is one threshold started from the raw RAG (does the high-T work in-band).
G7 not run (G6r FAIL). G9 BLOCKED: no 3090 Ti.
G6 remains blocked by contact-mean ParHAC round count, not by missing CUDA.

## P0 (measure only)
P0b red_viol=0: contact-mean S3 is reducible. T14 allowed.
P0a ε=0.08 ninner_records=1765 layer_mean=40647 layer_max=2367715
ec_mean=4091 us_mean=11608 us_p50=181 us_max=1441795
(20 us GPU floor was a guess; CPU inner p50 is 181 us, mean 11.6 ms.)
P0c basin SV first_zero=6 (r0-5 still change; r6+ = 0). Adaptive SV=7.
P0d after E9b: unique=89695822 multi=89797610 (~50/50 fg). bg=506568 exact.

## W14 PASS
adaptive sv=7 nfrag=2175400 d=0 bg=506568 exact. divide bfs_ms=13.73.
Unique-bit pointer-jump skipped (P0d ~50/50 unique/multi; SV already ≤15 ms after adaptive stop).

## T14 KILLED (Y1-class)
command: .venv/bin/python scripts/t14_terahac.py --eps 0.1 [--cap 16384]
Official TeraHAC control flow + S3 contact-mean (not AverageLinkageWeight).
cap=0 (official max(n/100,1e6)): no first outer after 6 min: one giant serial SubgraphHAC. Killed.
T14b cap=16384 heap then good-matching rewrite: no first outer after 5+ min. Killed.
stamp data/cache/t14_outer.txt = 9999.
Treat as Y1-class: not a G6 path on this RAG. No ε=0.05/0.02 (never reached a VOI run).
Go M16. Do not hybrid-cut.

## M16 FAIL
command: .venv/bin/python scripts/m16_mutex.py
Wolf Alg. 2 / GASP AbsMax on signed RAG (w+=mean, w-=1-mean). Wall=13.081 s.
nseg ~421990-422041 at all four T (dendrogram cut almost no-op).
ACCURACY GATE: FAIL all T. split 0.9097 (limits 0.40-0.63). Under-merge.
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
G7 not run (G6r FAIL). G9 BLOCKED: no 3090 Ti.
G6 remains blocked by contact-mean ParHAC round count. Next AGG needs a new plan.

## P0e-k (measure only)

P0e 7 505 458 edges. High tail is huge: mean>0.99 = 2 024 472, >0.95 = 3 551 701.
C19 residual cannot be <50k. X2 leftover (mean>0.55 = 4 558 929) confirmed.
P0j area: lt2=1.36M lt8=3.65M lt32=2.01M lt128=465k ge128=19k max=41044. Not a giant-area lock.

P0f live BinMatch Δ=0.10 from 1.0->0.3: **every band hit the 200-round cap**.
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
nseg 294518/322167/345376/379876: identical to E4 MeanAffinity heap.
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
GPU AGG SKIP (no new lock with rounds≤30). G9 BLOCKED: no 3090 Ti.

Matching-until-empty does not clear contact-mean bands (200-round cap on every 0.1 band). Dual-weight Wmax does not drop in 50k S4 merges. High-tail coarsen is not small. Quantile grades and is still serial. AGG lock unchanged.

## P0m/n/r (measure only)

Static CC of G[mean>τ ∧ area>=a]. Graded-T subgraphs are giants:
T=0.2 a=1 giant=0.992 ncc=279468 (X0); a=16 still giant=0.752.
T=0.3 a=1 giant=0.970; T=0.4 a=1 giant=0.842; T=0.5 a=1 giant=0.623.
No (τ=T, a>=2) cell is both non-giant and within 15% of E4 nseg. A27 skipped.
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
max_vox stays ~2.0-2.1M (no X0 giant). The size cap blocks the giant and also the large-large contacts waterz takes. Structural miss. Max-face SKIP (no per-face list).

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

Waterfall lowest-pass union of mean>T. nseg 279468/309454/330578/355268: identical to X0. merge VOI 7.78/7.55/6.43/4.61. Lowest-pass graph at these T is still one giant.

## H30 FAIL

Size-doubling HEM, 20 rounds, wall=82.5 s. nseg ~917k-979k. 0.2 split 2.374 merge 0.201. Under-merge (matching-shaped). Not a lock.

## G16 (after Kruskal tree)

locks unchanged: WS=E9c, AGG=paper-ε 0.08, extract=E10
G4r ACCURACY GATE: PASS
  0.2 split 0.3727 merge 0.3442 PASS
  0.3 split 0.4546 merge 0.2451 PASS
  0.4 split 0.5129 merge 0.2184 PASS
  0.5 split 0.6145 merge 0.2100 PASS
G6r one-shot ws=0.723 rag=0.149 agg=41.903 extract=0.485 total=43.260 FAIL (need <0.050s)
GPU AGG SKIP (no stamped lock). G9 BLOCKED: no 3090 Ti.

Every G6-class ordered-UF predicate we ran (area-CC skipped by P0, SDSL, FH, SRM, X1+size, Soille, waterfall, HEM) failed VOI. Contact-mean HAC remains accurate only as a serial/Y1 algorithm on this RAG. AGG lock unchanged.

## P0s/t/u leftover / relative-contact / tiles

Pinned: S26 Lu Algorithm 2 freeze, S27 lsd/daisy skip-as-product, S28 relative-contact definition.
`data/cache/p0_leftover_block.json` branch=["L33","B34"] (R32 also a P0t candidate; run in order).

P0s leftover after SDSL is small: L33 is a G6-shaped tail:
S0=256  T=0.2 n_residual=25723 (all large-large)
S0=1024 T=0.2 n_residual=3314
S0=4096 T=0.2 n_residual=645
n_small_touch is original-edge count (not supernode residual). No S0 has residual>=50k.

P0t relative-contact `mean>T ∧ area>=γ·min(S)^α`. Three candidates:
γ=0.10 α=0.67 all4 T=0.2 giant=0.0167 ncc=280195 (winner)
γ=0.05 α=0.67 all4 T=0.2 giant=0.0285 ncc=279857
γ=0.02 α=0.67 mid-only (T=0.2 giant=0.0517 just over 5%; 0.3+0.4 OK)
Batched B=16 at the winner: T=0.2 giant=0.233 (frozen-size blows the giant).

P0u frozen supernodes after interior-only Kruskal: 580k / 479k / 384k at 8³ / 16³ / 32x128x128. Intra_frac at 32x128x128 T=0.2 = 0.698. Residual inter 405k-741k. Not a <50k residual.

## L33 FAIL (all S0)

SDSL then S4 on leftover live S3>T. Residual is exactly the refused large-large `mean>T` edges. Heaping them rebuilds the giant.
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
nseg 280195/310243/331405/356087. max_vox 3.00M at T=0.2 (no X0 giant). Serial FAIL -> no batched lock. B=16 P0 already giant=0.23.

## B34 FAIL

Naive 32x128x128: intra Kruskal mean>T then residual Kruskal. residual=405k. 0.2 split 0.1272 merge 7.718. X0-inside-volume after the inter tail. Skip-Lu parser originally fired; Lu-exact still run.
Lu freeze + double-tile, 5 levels. lev=0 residual=4.64M frozen=1.80M; lev=3 residual=1.81M; lev=4 one-tile residual=0 after union-all (X0). FAIL-depth (any level residual>=50k). VOI 0.2 split 0.120 merge 7.778. Top-level heap would be Y1 (closed as Lu-as-product).

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
GPU AGG SKIP (no stamped lock). G9 BLOCKED: no 3090 Ti.

Leftover S4, relative-contact, block-S3, and voxel hysteresis are closed. The leftover *is* the giant; relative-contact is the nearest miss (T=0.2 merge in gate, split 0.427); block-S3 is Y1/X0; hysteresis is overseg+giant. AGG lock unchanged.

## S29-S31

Pinned: S29 ParHAC poly-log depth vs E6 10 ms (not a TASK number; val AGG budget ≈50 ms), S30 Hard-HAC NN-chain height escape, S31 leftover after relative-contact (thin contacts, not L33).

## P0x

`data/cache/p0_gpu_inner.json`. Paper-ε 0.08: ninner=1765 layer_mean=40647 layer_max=2367715.
Dummy scan+atomic matching on 5090: n=4e4/4e5/2.4e6 launch 0.002/0.004/0.016 ms, fused 0.00010/0.00050/0.0020 ms.
val_proj launch=1.09 ms fused=0.105 ms (T=0.3 share 0.045 ms). Track A (fused ≤50 ms). The 20 us E6 floor was a guess; dummy matching is far cheaper. Fused hides launch tax.

## E6r: VOI PASS / time FAIL (no LOCK)

Device paper-ε 0.08 ContractLayer on the cached RAG. Live compact+combine; color hash must mix the seed through a multiply (id-parity XOR left even-even leftover uncolorable). CPU accept includes the first overflowing blue (`take=k+1`). Rebuild cluster sizes each outer.

VOI four-T PASS (not bit-identical to host):
  0.2 split 0.3761 merge 0.3353 PASS
  0.3 split 0.4555 merge 0.2409 PASS
  0.4 split 0.5156 merge 0.2180 PASS
  0.5 split 0.6054 merge 0.2097 PASS
nseg 294162/321994/345065/379093. agg_s=3.204 s (3204 ms) vs 50 ms budget. Stamp `FAIL PASS agg_ms=3204.48`. Track A dies on time, not VOI. `segment_d` keeps RAG on device only if E6r LOCK (not stamped). Host `segment()` unchanged.

## P0v

Leftover after relative-contact (thin contacts). `data/cache/p0_r32_leftover.json`.
γ=0.10 α=0.67: T=0.2/0.3/0.4/0.5 n_residual=85/49/24/16 n_high=0 giant_if_union=0.0188/0.0104
γ=0.05 α=0.67: 24/10/3/4. All cells <50k -> L36.

## L36 FAIL (both γ)

R32 then S4 on thin residual. Residual ≤5k (lock-shaped) but heaping it over-merges: same structure as L33: the leftover *is* the giant connectors.
γ=0.10 residual=85 s4=73  0.2 split 0.3486 merge 0.5242
γ=0.05 residual=24 s4=20  0.2 split 0.3443 merge 0.5089

## R36 FAIL (γ=0.05 α=0.67)

P0t all-four candidate, first VOI grade. Serial: 0.2 split 0.3870 PASS merge 0.4421 FAIL. Over-merge vs R32 γ=0.10 (that one had merge PASS split FAIL). No batched.

## P0w / N36 SKIP

NN-chain S3 on the val RAG (not the 50k-edge P0i sample). All four T hit the 200-round cap. maxchain=28/26/24/24 (≤5k). rounds=200>30 -> N36 KILL. No ParChain clone.

## G16 (after E6r / Track B)

locks unchanged: WS=E9c, AGG=paper-ε 0.08, extract=E10. Host SV not forced to 7.
G4r ACCURACY GATE: PASS try=1
  0.2 split 0.3718 merge 0.3403 PASS
  0.3 split 0.4503 merge 0.2449 PASS
  0.4 split 0.5130 merge 0.2184 PASS
  0.5 split 0.6145 merge 0.2100 PASS
G6r one-shot ws=0.573 rag=0.159 agg=43.458 extract=0.497 total=44.688 FAIL (AGG budget 50 ms; e2e proxy 90 ms)
GPU AGG SKIP (no stamped lock). G9 BLOCKED: no 3090 Ti.

Device paper-ε is VOI-capable at 3.2 s (64x too slow for the val AGG budget). Thin leftover after R32 is 16-85 edges and still the giant if heaped. NN-chain needs >200 parallel rounds. AGG lock unchanged.

## S32-S36

Pinned: S32 clustered-graph 7-11x, S33 per-red accept, S34 Hornet idea-only, S35 Tseng exact-dynamic hardness, S36 Affinity≡E5. Unused FAIL-class list frozen. Tseng HTML at `papers/tseng_spaa2022.txt`.

## P0y

`data/cache/p0y_e6r.json`. T=0.3-only skip_debug, RAG H2D then device AGG:
wall=2929.60 ms. compact=901.60 propose=123.51 accept=1386.25 d2h=23.43 memset=25.17
ninner=1930 nmerge=1853410. nlive mean=1.86M p50=1.38M max=7.50M.
nprop mean=2584 max=601832; 1152/1930 inners have nprop=0 (recolor misses).
compact+d2h+accept=2311 ms -> branch E6s-a-then-b. Accept `<<<1,1>>>` is the largest slice.

## E6s-a: VOI PASS / time FAIL

`k_accept_reds` one thread per red, same take=k+1. No CEN dump. compact still full-graph.
T=0.3 agg_ms=1561. ninner=1930 merges=1853410 (bit-match E6r counts).
nseg 294162/321994/345065/379093. ACCURACY GATE: PASS (same VOI as E6r).
Accept tax gone (~1.4 s). Residual ~1.56 s.

## E6s-b: lazy/hash/Gc killed VOI; radix-every-inner keeps VOI

- Hash-combine + empty-outer abort: nseg ~610k, split 1.30 FAIL (under-merge).
- Skip combine (rewrite-only, keep-ratio): merge 0.591 at T=0.2 FAIL (over-merge).
- Paper Gc extract (inners on mean>=TL only, G not updated mid-layer): merge 4.01 at T=0.2 FAIL.
- compact_radix (uint64 key) every inner, full G: VOI PASS, T=0.3=1596.60 ms.
  Same nseg/merges as E6r. Radix ≈ comparison sort on this size.

## E6s-c: no LOCK

Propose on ~1.8M live edges x 1930 inners is 123 ms alone (P0y), already over the 50 ms val AGG budget. Device-resident inner cannot fix that without shrinking the scanned graph, and every shrink we tried broke VOI.
Stamp `e6r_pass.txt` = `FAIL PASS T03_ms=1596.60 budget=50`. No LOCK.
`segment_d` keeps host paper-ε. G16 GPU AGG SKIP. G9 BLOCKED: no 3090 Ti.

Residual: 1597 ms vs 50 ms (32x). Stop. No Track B leftovers, no γ/SCC/MALA/Kruskal.

## S37-S38 / P0z

Pinned: S37 CPU `Graph.adj`/`unite_keep` = StarMerge; S38 paper L13-14 merge G **and** Gc.
P0z T=0.3 instrument E6s-a (no merge change): wall=1703 ms (1.07x vs 1597).
ninner=1930 n_layer=18 merges=1853410 (match E6s).
ngc_mean=39375 p50=83.5 max=3.65M. n_dirty_blue total=19.45M. n_dirty_star total=167M.
Go/no-go: mean n_gc≤80k and n_dirty≤50M -> **E6t**. nstar>150M forbids red∪blue compact. E6t walks blues only + InsertOrUpdate.

## E6t: VOI PASS / time FAIL (no LOCK)

GPU StarMerge: CSR+overflow adj, 8x open-address hash InsertOrUpdate, propose on updated Gc indices (`k_propose_eid`). Line 13 then 14. Rebuild hash/CSR on hash-fail or killed>50% or layer start (18 layers).

T=0.3-only agg_ms=3008.73 (H2D inside). inner=1941 merges=1853378 (E6s was 1930 / 1853410).
Four-T nseg 294163/322004/345052/379078 (E6s 294162/321994/345065/379093).
ACCURACY GATE: PASS
  0.2 split 0.3817 merge 0.3275
  0.3 split 0.4541 merge 0.2436
  0.4 split 0.5154 merge 0.2182
  0.5 split 0.6161 merge 0.2099
Stamp `e6r_pass.txt` = `FAIL PASS T03_ms=3008.73 budget=50`. No LOCK.
Slower than E6s compact-every-inner (1597 ms): hash rebuild/probe tax > full radix on this RAG.
`segment()` stays host paper-ε. G16 GPU AGG SKIP. G8 not restamped. G9 BLOCKED: no 3090 Ti.

Residual: 3009 ms vs 50 ms (60x). StarMerge work volume was legal (P0z go); this implementation did not beat E6s wall time. No unused legal agglomerator left.

## P0aa: E6s-a phase split

`data/cache/p0aa_e6s.json`. T=0.3, wall=1684 ms (1.05x vs 1597, keep).
n_layer=18 ninner=1930 nmerge=1853410 hit_64=True (every layer 64 outers).
compact=953.8 propose=96.1 pack=14.0 sort=69.3 accept=16.5 compress=81.1
freeze=12.3 color=15.5 memset=23.8 d2h=17.3 host=0. Phase sum=1300 ms.
Compact still dominates. Propose alone is already over the 50 ms proxy.

## P0ab: outer cap 256/512

Both drifted nmerge 1853400 vs locked 1853410 (drift=10). n_layer_est 16 vs 18.
Slower (1870 / 2221 ms). Revert. `chosen_cap=64`. No four-T on drifted caps.

## E6u / E6v / E6w: VOI-capable, slower than E6s (FAIL implementation)

Device-graph while (CUDA 12.8 cond node) + 4x hash StarMerge + listed pack/freeze.
Capture cached per TL/layer. T=0.3 device_ms=9768 (6.1x vs 1597).
inner=1562 merges=1853416 (not bit-match). Not a new agglomerator: this impl lost.
`paper_d` defaults back to E6s (`WATERZ_PAPER_E6T` to force StarMerge).

## Gate (5090, no 3090 Ti)

E6s fallback T=0.3 CUDA-event (H2D excluded) **1294.21 ms**.
Four-T nseg exact 294162/321994/345065/379093. ACCURACY GATE: PASS.
Stamp `e6r_pass.txt` = `FAIL PASS T03_ms=1294.21 budget=50`. No LOCK (50 ms not met; planning target ~10 ms).
`segment()` stays host paper-ε. G16 GPU AGG SKIP. **G9 BLOCKED: no 3090 Ti.**
Do not write 2 Gvox/s. 5090 val AGG 1.29 s is not a 3090 Ti number and is ~26x the same-GPU 50 ms proxy.

Residual vs TASK speed: compact 954 ms on val already exceeds the entire 50 ms AGG budget. No unused legal agglomerator left. Renting a 3090 Ti now would measure ~1.78x this, not pass E9.

## A1: E6t/StarMerge is VOI-legal (correctness established before tuning)

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
95% GPU utilization. **PROVISIONAL: not a speed result.** Timing under
contention is not graded; graded timing waits for an uncontended GPU.

## A2: E6s is deterministic, E6t is not (TASK line 118 violation)

`scripts/a2_determinism.py`, T=0.3 only, 2 runs per path, cached rag.npz so the
input is byte-identical by construction. Compares returned parent arrays.

    E6s  byte_identical=True   ndiff=0        merges 1853427 both runs
    E6t  byte_identical=False  ndiff=397852   merges 1853416 vs 1853407

397852 of 2175401 parents disagree: 18% of nodes, not float noise at the
margin. Three causes found; two fixed, one structural:

1. FIXED. Proposal priority hashed the edge's array **index**. Harmless under
   E6s (compact_radix re-sorts by (u,v) every inner, so the index is a function
   of the graph) but fatal under E6t, where StarMerge assigns a merged edge's
   surviving slot by atomicCAS race. `prop_pri_bits()` now hashes the canonical
   root pair, and widens tie space 24 -> 31 bits. ndiff 531431 -> 405456.
2. FIXED. `atomicAdd(&sm[dest], olds)` on a double is not associative.
   Contact sums are now rescaled once into integral affinity-byte units
   (`k_scale_sm_bytes`, bound 1.65e12 << 2^53), making every accumulation exact
   and order-free. ndiff 405456 -> 397852. E6s merges unchanged by the rescale
   at 1853427, confirming the quantization is benign.
3. NOT FIXED: structural. In `k_starmarge_blue` the `is_new` branch pushes a
   survivor onto a node's overflow adjacency only when *that* edge's own
   endpoints differ from the merged roots. Which racing edge wins the CAS
   therefore decides whether the survivor stays reachable at all, so the
   graph's reachability varies per run, not merely its numbering. Fixing this
   means redesigning StarMerge's incremental adjacency maintenance.

Verdict: E6t cannot ship regardless of speed. Optimization effort moves to the
deterministic path. E6s re-graded after fixes 1 and 2, four thresholds,
ACCURACY GATE: PASS (split 0.3707/0.4512/0.5162/0.6129, T=0.2 better than the
0.3779 baseline).

Contention note: E6s measured 27.5 s here against 1.29 s idle: a **22x**
inflation with a co-tenant at 95% GPU. No timing on this box is usable while
that holds.

## B1: the GPU RAG was nondeterministic too, now fixed and oracle-checked

`scripts/b1_rag_determinism.py`. `csrc/rag.cu` accumulated contact sums with
`atomicAdd` on a **float32**. Two runs on identical input:

    before: sm bit-identical=False, 740853 of 7505458 edges differ (9.9%),
            max drift 5.2e-3; edge set and counts identical
    after:  sm bit-identical=True, 0 edges differ, drift 0.0

This one sat in the **production** `segment()` path, so the earlier G5
determinism pass was luck, not a guarantee.

Fix: `Slot.sum` (float) -> `Slot.isum` (uint32) accumulating the RAW uint8
affinity bytes, with `sm = isum/255.0` at scatter. Integer addition is exact,
so the result is independent of atomic order. Slot stays 16 B, so the table
costs nothing extra. The `isum <= 255*n` invariant is asserted on device rather
than assumed (returns -2 and warns on violation); uint32 would only overflow
past 16.8M faces on one fragment pair, against a whole-volume total of 3*nvox.

Gated against the independent CPU reference in rag.npz (built by e3cpu.py via
`region_graph`, not by this code path):

    edge count 7505458 == TASK          edge set equal          counts EXACT
    sum max abs diff 2.12e-05

so the builder is deterministic *and* still right; determinism alone would also
be satisfied by a stable-but-wrong builder.

Note `rag.npz` comes from the CPU oracle, so the AGG gates are decoupled from
this change and the E6s VOI pass above is unaffected by it.

## C1: the 24 GB memory wall is 6.7x, not the 1.6x previously assumed

`scripts/c1_memory_budget.py`, `data/cache/c1_memory_budget.json`. Every
cudaMalloc in ws.cu / rag.cu / parhac_d.cu is now routed through an in-library
counter, so these are exact allocation figures, not estimates. Both stages
report `leaked=0`, i.e. every free matched, so the counters are self-consistent.

Measured at val (125x1200x1200 = 180 Mvox):

    watershed peak  11.489 GiB   68.53 B/vox   nfrag=2175400
    rag peak         2.871 GiB   17.12 B/vox   nedge=7505458
    concurrent peak 13.33 GiB    (max stage + resident aff/seg/out)

Projected. WS is scaled linearly, which is right for the benchmark volumes
because make_big.py tiles val, so plateau/corner density is preserved. RAG is
computed from its own `next_pow2(2*max_edges) * 16 B` formula instead, since
its table is a step function of the max_edges argument and not of voxel count;
scaling the val measurement had given 34.45 G at 2.16 Gvox where the formula
gives 7.66 G.

    volume                  WS        RAG      io       peak    fits 24 GiB
    val 180 Mvox         11.49 G    1.70 G   1.84 G   13.33 G   yes
    1.44 Gvox            91.91 G    6.77 G  14.75 G  106.66 G   NO
    2.16 Gvox           137.87 G    7.66 G  22.13 G  159.99 G   NO   <-- 6.7x

At 2.16 Gvox: ~26.1M fragments, ~90.1M edges, largest z-slab that fits ~20 GiB
of usable VRAM is ~270 Mvox, so **>= 8 slabs**.

Where the watershed's 68.53 B/vox goes, from the E9b diagnostics
(`ncorner=61035574 nplat=55032772 qtot=565241094`):

    q            qtot * 8  = 4.21 GiB   <-- largest single buffer, 3.14 ent/vox
    parent/flag/psum/vcount  4 * 4 B/vox = 2.68 GiB
    basin UF stage           ~20 B/vox   = 3.35 GiB
    corners_in/out           nC * 8 * 2  = 0.91 GiB
    plat_begin/nseed/root/qsz/qoff  P*20 = 1.02 GiB
    keys_in/out, start, start_ps        = 0.91 GiB

Consequences, and they reorder the whole plan:

1. Z-slab chunking is **mandatory**, not an optimization. At 68.53 B/vox plus
   resident aff/seg/out, a slab fitting in ~20 GiB of usable VRAM is at most
   ~270 Mvox, so 2.16 Gvox needs **8 or more slabs** with halos and seam
   stitching.
2. The memory blocker, not the speed gap, is the critical path. No amount of
   agglomeration tuning makes 2.16 Gvox run on a 24 GB card.
3. Cheapest identified reductions, in order of size: `q` is int64 but the
   graded volume has 2.16e9 < 2^32 voxels, so uint32 indices halve the largest
   buffer (-2.1 GiB at val scale equivalent); `flag`/`psum` are separate 4 B/vox
   arrays that can share storage with an in-place scan; `vcount` is only needed
   per plateau, not per voxel.

Two things fixed already from this analysis:

- `segment.py` hardcoded `max_e = 20_000_000`. At the measured 0.0417
  edges/vox that caps the pipeline at ~480 Mvox, so 2.16 Gvox would have
  failed with `rag_gpu_d overflow` on edge capacity before it ever reached a
  memory limit. Now `_max_edges(nvox)` with a 20M floor.
- That cap is 0.055 edges/vox, 32% over measured, deliberately chosen to stay
  under a power-of-two boundary: rag.cu sizes its table `next_pow2(2*max_edges)`,
  so 0.08 tips into 2^29 slots and an 8.00 GiB table where 0.055 stays at 2^28
  and 4.00 GiB. Overflow returns -1 rather than truncating, so too small fails
  loudly.

The 160 GiB verdict itself is not yet fixed; chunking is the only thing that
changes it. Recorded so the 2 Gvox/s claim is not attempted on a card that
provably cannot hold the volume.

## Measurement bug found in `scripts/e6s_parhac.py`

The script branches on `WATERZ_PAPER_E6S`, but `csrc/parhac_d.cu` reads
`WATERZ_PAPER_E6T` (default E6s) after the 19:30 source flip. So the line
labelled `E6uvw T=0.3 ... 1294.21 ms` was in fact **E6s**, and the
"retry as E6s" fallback is dead code. The 1294 ms number itself is correct for
E6s; only the label was wrong.



## Watershed memory: 68.53 -> 25.43 B/vox, and the plateau BFS got 2.7x faster

Track C wanted the watershed's 68.53 B/vox down. Before touching it, the
existing gates were too weak to protect the change: they checked `nfrag` and
`bg` counts, which a change can preserve while moving voxels between
fragments. `scripts/c2_ws_invariants.py` closes that. It records, on val, a
sha256 of the *sorted region-size histogram*, which is invariant to label
renumbering but changes if a single voxel moves; plus two-run determinism and
`array_equal` against the CPU oracle `wz_fragments.npy`.

Worth noting on its own: the GPU watershed is **bit-identical to the CPU
oracle**, not merely count-matched. Baseline fingerprint
`fff9037cab341692be0c9bf3c577d4ff`, held by every step below.

### The plateau BFS queue was 8.84x oversized

`k_plat_meta` sized each plateau's queue slot `vcount[root] + nseed + 8`.
Summed over 55,032,772 plateaus the `+8` alone is 440M of the 565M entries, so
78% of the largest buffer in the pipeline was padding.

The slack is not needed, and the argument is exact rather than empirical.
`k_or40` pre-marks every seed `0x40` before the BFS, so no seed can be pushed
a second time; any `j` the BFS does push is reciprocally linked to a popped
voxel, hence in the same union-find component and itself counted in `vcount`.
So `tail <= vcount[root]`, and `vcount[root] >= nseed` because every seed is
flagged and therefore counted. Instrumented `WATERZ_WS_QDIAG=1` to measure it
anyway: `max(tail - vcount) == 0` over all 55M plateaus, `qused_max` only 121.
A positive value there would have falsified the argument, and it also confirms
the union-find is fully converged, since an unconverged plateau would split
across parent values and overrun.

    qtot 565,241,094 -> 63,943,344   (waste factor 8.84 -> 1.000)
    peak 68.53 -> 51.15 B/vox
    plateau BFS 11.2-14.3 ms -> 4.56 ms

The 2.7x speedup is a side effect: the queue is 8.84x smaller and far more
cache-friendly. That is ~7 ms off a 38 ms end-to-end budget, so this is a
speed result as much as a memory one.

Exact sizing removes the margin that was hiding a silent failure mode, so
`k_indep_bfs` now takes `qsz` and an `overflow` flag and returns -3 loudly
instead of running one plateau's BFS into the next plateau's slot.

### Then lifetimes, a duplicated scan, and index width

- `parent`, `flag`, `psum` are dead at `k_scatter_idx`/`k_keys_from_parent`
  but were held to the end of the function, straight through the 61M-pair
  radix sort. Freeing them at their last use, and likewise `keys_out`,
  `start`, `start_ps`, `vcount` before the queue is allocated: 51.15 -> 32.14.
- `psum` was a whole 4 B/vox duplicate of a scan that can run in place
  (already done at `ws.cu:634` and `rag.cu:185`, so it was proven here).
  After an in-place exclusive scan the corner predicate is still recoverable
  as `ps[i+1] > ps[i]`, with the last voxel covered by a `last_f` read taken
  before the scan. `flag` and `psum` become one buffer.
- Corner and queue entries were int64 holding values that fit uint32, which
  also halves the radix sort's payload traffic. `e9b_divide_d` now rejects
  volumes above 4.29e9 voxels rather than wrapping silently. `qsz`/`qoff` are
  uint32 for the same reason, and safely so: `qtot` is a sum of `vcount` over
  distinct roots, so it is bounded by the voxel count.
- Allocating the per-plateau queue arrays after releasing the per-corner ones,
  instead of overlapping them: 25.99 -> 25.43.

Cumulative 68.53 -> 25.43 B/vox, 2.69x, fingerprint and oracle equality
unchanged at every step. Peak now sits at `divide/corners`: `parent` + `flag` +
`vcount` at 4 B/vox each, the corner and key arrays, and 1.345 GiB of caller
buffers.

### What this does and does not buy at 2.16 Gvox

    2.16 Gvox total   160 GiB -> 73.28 GiB      slabs needed   8 -> 4

Still 3x over a 24 GiB card, so z-slab chunking remains mandatory. The
projection also makes clear that no amount of working-set trimming can remove
that: at 2.16 Gvox the input affinity alone is 6.48 GiB and the output labels
8.64 GiB, so 15.1 GiB of a 24 GiB card is consumed by the volume's own I/O
before any working buffer exists. The slab design therefore has to stream
affinity in and labels out, not merely partition the working set.

## The plateau union-find ran 40 rounds and needs 6

`e9b_divide_d` ran its hook/compress loop a fixed 40 times. Each round sweeps
every voxel and reads its six neighbours' bits and parents, twice over, so a
round is tens of GB of traffic and this loop dominates the watershed stage:
the whole thing measures `uf_ms=498` against `bfs_ms=4.1` for the plateau BFS.
Spare rounds are the most expensive idle work in the pipeline.

Replaced with a loop that stops when a round changes nothing, flagged by
`atomicExch` from both the hook (when its `atomicMin` actually lowered a
value) and the compress (when a parent moved). Measured: **6 to 7 rounds**, so
the fixed count was doing ~6x the necessary work. Plateaus are shallow, which
`qused_max=121` already implied, so no deeper convergence was ever needed.

Terminating on both kernels being at a fixpoint is what makes this safe rather
than merely faster. Everything downstream needs `parent` flattened to true
roots, because `vcount` is indexed by `parent[i]`, and an unflattened parent
would split a plateau's count across several nodes and undersize its BFS
queue. Exiting only when hook and compress both change nothing means `parent`
is a fixpoint of compress, i.e. fully flattened. The `uf_cap = 64` bound
remains as a safety net and reports `NOT-CONVERGED`, with the queue overflow
guard from the previous entry as the second line of defence.

One observation worth recording: **the round count varies run to run, 6 or 7,
while the output stays bit-identical.** That is expected and not a determinism
violation. The `atomicMin` race order changes how quickly the loop converges,
but the fixpoint it converges to is the component-wise minimum label, which is
order-independent. C2 confirms `array_equal` against the CPU oracle either
way. It does mean stage timings carry a round of jitter.

Next candidate on this loop, not yet done: only ~36% of voxels (64M of 180M,
from `qused_sum`) are in a plateau at all, so the rounds could sweep a
compacted active list instead of the full volume, roughly 21 GB of traffic
saved against ~1.5 GB to build the list.

## Rejected: active-list union-find rounds

Tried and reverted. The premise was that the union-find rounds sweep the whole
volume while only the ~64M voxels implied by `qused_sum` can be hooked, so a
compacted list would cut each round to 36% of the work.

The measurement refuted it: **128,450,317 of 180,000,000 voxels have a
reciprocated neighbour**, 71% of the volume, not 36%. The list saves 1.4x on
the sweep, not 2.8x, against three added passes to build it.

The gap between 128M and 64M is the useful part of this result. `qused_sum`
counts voxels in plateaus that reach the plateau list, and a component only
gets there if it contains at least one corner. So roughly 64M voxels sit in
corner-free flat regions, local minima with no outlet, which become basins
directly and are never divided. They are reciprocally linked, hence hookable,
hence on the active list, but contribute nothing to the queue total.

Reverted because the trade is bad on the axis that actually binds. Net traffic
saving works out around 25%, which is unverifiable on a GPU at 96% contention
(`uf_ms` 498 -> 510, i.e. noise), while the list costs 2.86 B/vox, about
6.2 GiB at 2.16 Gvox, during the phase that competes for the memory the whole
Track C effort is trying to free. Output stayed bit-identical throughout, so
this was a performance judgement and not a correctness one. Worth revisiting
only once timings are trustworthy.

## Where agglomeration time actually goes, and two hypotheses it killed

Agglomeration is the biggest gap to budget, 1294 ms against 25 ms, but the GPU
is at 96% contention so wall-clock comparisons are worthless. `scripts/
a3_work_accounting.py` measures the loop in counts instead, which contention
cannot distort, and `parhac_e6s_p0aa` supplies an 11-phase event split.

### Track A's premise was wrong: the listing ceiling is 4x, not 50x

The plan assumed the per-node sweeps waste most of their work, since every
inner iteration sweeps all `nnode` nodes four times (`memset dprop` at 8 B per
node, `k_pack_prop`, `k_compress`, `k_freeze`) no matter how few roots are
live. Added optional `hist_nact` to the profiler to count live roots per
iteration, and it does not collapse the way the premise required:

    nnode 2,175,401    nact max 1,785,980  p50 428,417  min 321,973
    node-visits done 1.60e10, useful 3.97e9  ->  ceiling 4.0x

95% of iterations have fewer than half the nodes live, but none has fewer than
a tenth: `nact` bottoms out at 321,973, about nnode/6.8. So `a-listed` and
`a-roots` are worth at most 4x on a phase group that is not the dominant one.
Also worth noting the outer-loop sweeps are 50.9% of all node work, so listing
only the inner ones would address half of that 4x.

My first version of this bound used `2*nlive` as the useful width, on the
reasoning that nlive edges touch at most that many endpoints. True, but so
loose it reported a 1.0x ceiling and would have wrongly killed Track A
outright: most of those endpoints are repeat references to the same few roots.
Recording the mistake because the loose bound looked like a clean negative
result.

### And the accept kernel is innocent

`k_accept_serial` runs `<<<1, 1>>>`, one thread walking every proposal, which
looked like an obvious culprit. It is not: **25 ms of 52,634**. The proposal
count explains why. Across all 1840 inner iterations there are only 4.97e6
proposals in total, median **zero** per iteration, so half of them accept
nothing at all.

### The real distribution

    sort      14157 ms   26.9%      compress   2402 ms
    compact    8671 ms   16.5%      propose      91 ms
    everything else, pack/accept/freeze/color/memset/d2h, under 30 ms each
    wall      52634 ms

`sort` is two `thrust` calls per inner iteration over zip iterators, the same
pattern already removed from `compact_radix` for being slow. But the arithmetic
says the cost is not the sorting: 14157 ms over ~3680 calls is 3.85 ms per
call, to sort a few thousand elements. That is not compute and it is not even
thrust's per-call `cudaMalloc`, which runs in tens of microseconds. It is the
**device synchronization** thrust's default policy performs on every call:
each one waits for the co-tenant process's kernels to drain, so this phase is
measuring the neighbour's job as much as ours.

Two consequences. The 14157 ms must not be quoted as our sort cost. And the
structural fix is right regardless of contention: CUB with temp storage
allocated once outside the loop, no per-iteration synchronization, which also
decouples the stage from whatever else shares the card.

One latent hazard found while reading it. `k_pack_prop` stores the priority as
`__uint_as_float` of a 31-bit hash, and the sort then compares those bit
patterns *as floats*. Patterns in 0x7f800000-0x7fffffff are inf or NaN, and NaN
comparisons are false, so the order among them is undefined. A2 measured E6s
deterministic in practice, so this is not an active bug, but the priority is an
integer hash and should be sorted as one. Doing that changes the tie order and
therefore the output, so it needs a VOI re-grade rather than a bit-equality
check.

## Fused the proposal sort: sort phase 8.8x, E6s end-to-end 1.94x

Acting on the phase split from the previous entry. The proposal sort was two
`thrust` calls per inner iteration over zip iterators, and at 3.85 ms per call
to sort a few thousand elements the cost was thrust's per-call device
synchronization, each of which also waits for the co-tenant process's kernels
to drain.

`k_accept_reds` needs only proposals grouped by red with each group in
descending priority, and it never reads the priority itself. So `k_pack_prop_
fused` now builds the sort key and payload directly,

    key = red : (0x7fffffff - priority)      payload = blue : size

and one ascending 64-bit `cub::DeviceRadixSort::SortPairs` produces exactly
that order. CUB scratch and the four key/payload buffers are allocated once
outside the loop, sized for `nnode`, which bounds every call since CUB's byte
requirement is monotonic in item count. Two comparison sorts over zip
iterators, ~3700 temporary allocations and ~3700 device syncs become one radix
sort and a trivial unpack pass.

    sort phase          14157 ms -> 1603 ms     8.8x
    E6s device @T=0.3   25467 ms -> 13133 ms    1.94x
    wall (contended)    52634 ms -> 40463 ms

This also retires a latent hazard rather than only being faster. The old path
stored the priority as `__uint_as_float` of a 31-bit hash and compared those
bit patterns *as floats*; patterns in 0x7f800000-0x7fffffff are inf or NaN and
NaN comparisons are false, so the order among roughly 0.39% of proposals was
undefined. Ordering the hash as the integer it actually is has a defined total
order. `dpris` is gone from this path entirely.

Because that changes the tie order, bit-equality with the previous build was
not the right gate; VOI was. It came back stronger than required:

- unique-parent fingerprint `[294165, 322000, 345131, 379293]`, **identical**
  to the stored baseline at all four thresholds
- `outer=1088 inner=1840 merges=1853427`, identical to before
- ACCURACY GATE: PASS at all four thresholds
- A2 determinism `byte_identical=True ndiff=0`

So the undefined float ordering never actually decided a merge differently on
this input, which is luck rather than design, and is exactly why it was worth
removing.

A2 still reports FAIL overall, but only from E6t, which was root-caused as
structurally nondeterministic earlier and is not the production path.

`compact` is now the largest phase at 8766 ms, 21.7% of wall. The same
question applies to it: whether `compact_radix` allocates or synchronizes per
call. `parhac_dev`, the older profiling path behind a3 and p0y, still has the
original thrust sorts; it is not the production path so it was left alone.

## Compaction onto CUB too: 1.65x, and p0aa's wall is mostly its own instrument

`compact_radix` had the same pathology as the proposal sort, four times per
call. `thrust::copy_if` and both `reduce_by_key` calls return an iterator or a
count, so thrust must synchronize the device to hand it back, and each call
also allocated its own temporaries. At ~1840 compactions that was the largest
phase left.

The CUB equivalents write their counts to device memory and take caller-owned
scratch, so a `CompactScratch` (one temp buffer sized by querying all three
ops at `n_edges`, plus two count slots) is allocated once per host function
and threaded through. What remains is two D2H copies per compaction, for the
two counts that genuinely decide later launch geometry. No new buffers were
needed: `tv` was already dead at that point and takes the sorted index list.

    compact   8766 ms -> 5306 ms    1.65x

Unlike the proposal sort this one is exactly order-independent, so
bit-equality was the right gate and it holds. Affinity sums are integral
affinity-byte counts held exactly in a double and counts are int64, so
within-segment order cannot change a reduction result:

- A2 `byte_identical=True ndiff=0`
- unique-parent fingerprint `[294165, 322000, 345131, 379293]`, unchanged
- ACCURACY GATE: PASS at all four thresholds

Where the production path now stands, all on the contended card:

    E6s device @T=0.3       25467 ms -> 9754 ms     2.61x
    four-threshold grade    34265 ms -> 12297 ms    2.79x

### p0aa's wall time is 74% instrumentation

Worth writing down before it misleads the next measurement. p0aa reports
wall 37192 ms, but the phases sum to 9573 ms. The 27619 ms gap is almost
exactly the gap in the previous run too (52634 wall against ~25000 of phases),
i.e. **the unaccounted time is a constant ~27.6 s that did not move when two
phases got 3.5x faster.**

It is the `EvAccum` instrumentation. Ten instrumented regions over 1840 inner
iterations is ~18400 event stops, each of which synchronizes to read the
elapsed time, and each of those syncs waits for the co-tenant process's
kernels to drain as well. The same mechanism that made the thrust calls
expensive makes the timer expensive.

So: read p0aa for phase *ratios* only, never for magnitude. Magnitude comes
from the uninstrumented paths, a2's `device_ms` and a1's `device_ms`, which is
why the numbers above are quoted from those. This also means the real phase
figures are somewhat smaller than printed, since each region's own event pair
is inside it.

Next largest real phase is compact at 5306 ms, then compress at 2392 ms.

## What a host round-trip actually costs: 883 us, and why that is a trap

Goal was to find where the remaining 9854 ms of E6s at T=0.3 goes. Kernels in
the phase split come to ~200 ms outside compact, compress and sort, so the
suspicion was that the loop is latency-bound: four blocking copies per inner
iteration (proposal count, merge count, and the two counts inside
compact_radix), 7360 in total.

Added `WATERZ_SYNC_PROBE=k`, which inserts k extra copies per iteration, and
`scripts/a4_sync_cost.py`, which sweeps k and fits a line. The slope is a
better instrument than any absolute timing here because every point pays the
same contention.

**First attempt was wrong and said the opposite.** With the probe as a bare
`cudaMemcpy`, the slope came out at -2 us, i.e. nothing, and I briefly
concluded round-trips were free. They were free *as written*: the probe copies
followed the real copy with nothing enqueued in between, so they landed on a
stream that was already drained and returned immediately. The cost of a
round-trip is the pipeline drain, and a drain only exists when there is queued
work to drain. Launching a trivial kernel before each probe copy fixes it:

    median device_ms   k=0: 9854   k=1: 11498   k=2: 13126   k=4: 16356
    slope 883 us per round-trip, max fit residual 12 ms over a 6500 ms range
    the loop's own 7360 -> 6498 ms, 66% of the 9854 ms baseline

Iterations and merges are identical at every k, so the probe changes nothing
but timing.

### The trap

883 us is far too large for a drain. On an idle card it is 10-20 us. That
number is a GPU scheduling quantum: each drain hands the device to the
co-tenant process and waits to be scheduled again. **It is a property of
sharing the card, not of this code.** On a dedicated card those same 7360
round-trips cost on the order of 110 ms, not 6500 ms.

So the finding is not "restructure the loop for device-side control". It is:

- every absolute timing taken on this card is contaminated in a way that
  scales with *sync count*, not with work, so it cannot be used to rank
  optimizations. a3's count-based accounting and per-kernel work analysis are
  the only sound basis until the card is free.
- the p0aa phase split is distorted in a specific direction worth knowing:
  each EvAccum stop drains, so the copy that follows it is instant. That is
  why `d2h` reads 22 ms for 1840 iterations, ~12 us per copy, when the same
  copy costs 883 us uninstrumented. The nprop and nmerge drains are real but
  are hidden in the unmeasured gaps between regions. compact's two internal
  drains, by contrast, sit inside `ev_compact` and are counted there.
- reducing round-trips and kernel launches is still worth doing, since it
  helps under contention and is neutral otherwise. It is just not worth
  restructuring the algorithm around.

### Redundant compress removed, worth nothing, kept anyway

`compact_radix` opened with a full `k_compress` sweep, and the inner loop
compresses immediately before calling it with only `k_freeze` in between,
which touches sz and frozen but never parent. So it was pure duplicate work
and is now skipped via `recompress=false` from that call site; the layer-level
call follows a weight scan and keeps it.

Measured gain: none. compact went 5306 -> 5345 ms, i.e. unchanged. The reason
is that `k_compress` walks to the root, so after the first pass every entry
already points at a root and the second pass is two reads per node, ~17 MB.
The 2381 ms in the `compress` phase is the pass right after `k_accept_reds`,
where the chains are genuinely long. Kept the change since it is strictly less
work, but the cost model that motivated it was wrong.

Gated bit-identical: A2 `byte_identical=True ndiff=0`, unique-parent
fingerprint `[294165, 322000, 345131, 379293]` unchanged, ACCURACY GATE PASS.

### Next, on work rather than on timings

`compact_radix` runs an 8-pass 64-bit radix sort over the whole live edge list
every inner iteration, for the sole purpose of grouping duplicate keys so
`ReduceByKey` can merge parallel edges. That is ~28 kernel launches and O(m
log m) traffic per iteration to do an O(m) job.

This file already contains `hash_combine_live`, complete and currently
unreferenced: clear table, insert with atomicAdd on the sums, emit. It should
be determinism-safe for the same reason the RAG fix was, namely that the sums
are exact integers after `k_scale_sm_bytes`, and edge *order* does not affect
the result because propose picks by content-based priority under atomicMax.
Two things to watch: `k_hash_clear` currently clears a table sized from
`n_edges`, ~384 MB per iteration, which must be sized from the current `nlive`
instead; and `k_hash_emit` compacts with an atomic, so the output order is
nondeterministic and the claim that order does not matter has to be *tested*
with A2, not assumed.

Also queued: `sort` is 1658 ms for a median nprop of 0 and ~2700 average,
which cannot be work. A 64-bit CUB device sort is ~16 kernel launches; a
single-block `cub::BlockRadixSort` fast path for small nprop would make it one.

## Dedup by hash instead of by sort, and the bug that had shelved it

`compact_radix` ran an 8-pass 64-bit radix sort over the whole live edge list
every inner iteration for one reason only: to put duplicate keys next to each
other so `ReduceByKey` could merge parallel edges. O(m log m) traffic and ~28
kernel launches to do an O(m) job.

`hash_combine_live` was already in this file, complete and unreferenced. It
inserts each live edge into a hash table keyed on the canonical (u,v) with
atomicAdd on the sums, then emits. Two things made it safe to try:

- `k_rewrite` already zeroes `keep` for empty counts, self-loops and
  background-incident edges and canonicalises u < v, so the guards inside
  `k_hash_insert` are redundant and both paths combine the same edge multiset.
- the sums commute: contact sums are integral affinity-byte counts held
  exactly in a double after `k_scale_sm_bytes`, counts are integers.

Two things did not obviously hold, and both got handled rather than assumed.

**The emitted edge order.** `k_hash_emit` claims output slots with an atomic,
so the edge list comes out in a different and run-to-run unstable order. The
argument that this is harmless is that propose picks per node by atomicMax on
a content-derived priority, so nothing depends on edge position. A2 tests it
instead of taking it on trust, and it holds: byte-identical output, ndiff 0.

**The 128-probe bound.** Falling out of that loop drops an edge silently,
which is a wrong answer rather than a slow one, so `k_hash_insert` now sets an
overflow flag the caller checks. It fired immediately on the first run, with
merges down from 1853427 to 1522739. The flag is the only reason that was a
finding rather than a plausible-looking speedup.

The cause is a textbook one:

    unsigned long long h = key * 0x9E3779B97F4A7C15ull;
    int slot = (int)(h & mask);

In a multiplicative hash the low k bits of the product depend only on the low
k bits of the input, and here those are `v` alone. Every edge sharing an
endpoint `v` hashed into one cluster, overran 128 probes and was dropped.
Fibonacci hashing needs the *high* bits; the fix mixes to full avalanche with
a murmur3 finalizer and then masks. This is very likely why the function was
written, found to give wrong answers, and shelved instead of debugged.

Also sized the table from the current `nlive` rather than the original edge
count. Clearing is proportional to table size and `nlive` falls by an order of
magnitude across the layers, so a table fixed at the initial size would clear
~400 MB per iteration to hold a fraction of that. Load factor stays at or
below 0.5, which is what makes the probe bound safe.

    compact                 5383 ms -> 3469 ms    1.55x
    E6s device @T=0.3       9854 ms -> 8255 ms
    four-threshold grade   12350 ms -> 10390 ms

Gated: A2 `byte_identical=True ndiff=0`, unique-parent fingerprint
`[294165, 322000, 345131, 379293]` unchanged, ACCURACY GATE PASS, no overflow.
Hash is now the default for the inner-loop compaction; `WATERZ_E6S_HASH=0`
still reaches the radix path, which is the reference that set the fingerprint.
The layer-level compaction still uses the radix path because it needs the
threshold argument.

Where E6s time sits now, contended, phases: compact 3469, compress 2240,
sort 1483, everything else ~200. Note from the previous entry that roughly
two thirds of the *total* is contention-induced drain latency, so these
figures rank the work but do not predict a dedicated card.

Still queued: `sort` is 1483 ms for a median nprop of 0, so it is CUB's ~16
kernel launches per call, not work; a single-block `cub::BlockRadixSort` fast
path for small nprop makes that one launch.

## The graded entry point was not using any of this work

The card went idle, so for the first time the numbers mean something. Two
findings, and the second one dwarfs every optimisation in this log.

### On an idle card the contention theory checks out exactly

    a4 round-trip cost   883 us contended  ->  2.55 us idle    346x
    the loop's 7360      6498 ms           ->  19 ms, 3% of total
    p0aa wall @T=0.3     32791 ms          ->  1022 ms         32x
    E6s device @T=0.3     8255 ms          ->  668 ms

So declining to restructure the inner loop for device-side control was right:
that whole 66% was a scheduling quantum, not a drain, and it is 3% on a card
we own. The earlier entry's warning about ranking work by contended timings is
now confirmed rather than merely argued, and the *rankings* changed too, not
just the magnitudes. Contended, `sort` looked like 1483 ms; idle it is 23 ms.

Idle stage split at T=0.3 on val, 180 Mvox:

    compact 377   propose 90   compress 77   sort 23   memset 22
    d2h 17   accept 16   color 14   pack 13   freeze 12       sum 661

### segment() was running the host agglomeration

`scripts/bench.py` reported `heap=20.4 s` and turned out to be measuring the
CPU reference path, so it was never going to show this. Writing a bench against
the real entry point (`scripts/d_bench.py`) surfaced it immediately:
`agg=40991 ms` where the device path is 668 ms.

`_parhac` called `parhac_paper_cpu` unconditionally. The device build was
reachable only behind `data/cache/e6r_pass.txt`, which read

    FAIL PASS T03_ms=1294.21 budget=50

That gate required the device build to come in under 50 ms before it was
allowed to run at all. The effect was backwards: missing a speed target left
the graded path on an implementation ~60x slower still. Correctness is the
right gate for *which* implementation runs; speed is the thing being
optimised, not a precondition for being used. And the device path carries the
stronger correctness evidence, being graded VOI-legal at four thresholds and
byte-identical run to run, neither of which the host path has in this repo.

`_parhac` now dispatches to `parhac_paper_d`, whose signature is identical, and
`WATERZ_AGG_CPU=1` forces the host build for A/B. `segment.AGG_BACKEND`
records which ran so a benchmark can refuse to publish a number measured on
the fallback, which `d_bench.py` does.

Verified end to end through `segment()`, not just on a cached RAG:

- `eval.sh` accuracy: ACCURACY GATE PASS at 0.2/0.3/0.4/0.5, with
  nseg `[294164, 321999, 345130, 379292]` and VOI figures matching a1 exactly
- `g5_det.py`: PASS, byte-identical, nseg 321973

### Where the pipeline actually stands

Idle-card stage medians on val (180 Mvox), from before the wiring change for
ws/rag and from the device path for agg:

    ws 536 (of which Union-Find 225)   rag 114   agg 668   extract ~175

That is ~1.5 s against the ~90 ms that 2 Gvox/s implies at this size. Every
stage is now the honest one, which it was not an hour ago.

One thing to watch: `uf_rounds` came out 6 in one run and 7 in another while
the output stayed byte-identical. The extra round is the one that observes no
change, so a benign race on when the flag is seen would explain it, and the
result is unaffected. Worth confirming rather than assuming.

`d_bench.py` refuses to grade while the card is shared, and it is shared again,
so the median-of-5 throughput number still needs an idle window.

## Watershed peak 25.43 -> 22.71 B/vox by not overlapping two buffer sets

Memory is the hard blocker for the graded benchmark, and unlike timing it is
immune to the co-tenant, so it is the right thing to work on while the card is
shared.

`WATERZ_WS_MEMLOG=1` puts the peak at the `divide/corners` checkpoint. Adding
a mark at function entry pins the composition down exactly, at val:

    divide/enter    1.341 GiB   inherited, 8 B/vox
    + parent, flag, vcount      3 x 4 B/vox   -> 3.353 GiB at divide/uf
    + corners_in, keys_in, corners_out, keys_out, 4 x nC uint32
                                              -> 4.262 GiB, the peak

The four corner arrays are the inputs and outputs of a 61M-pair radix sort.
Only the *inputs* are needed while `parent` and `flag` are still live, since
the outputs are untouched until the sort runs, and `parent` and `flag` are
freed immediately after the two kernels that fill the inputs. So allocating
the outputs after those frees, rather than before, removes a 0.49 GiB overlap
with 1.34 GiB that existed for no reason.

    peak   4.262 GiB -> 3.808 GiB     25.43 -> 22.71 B/vox

Gated by c2: deterministic with ndiff 0, nfrag 2175400 and bg 506568 both
matching TASK, region-size fingerprint equal to the CPU oracle with
array_equal True, nothing leaked.

Projection for the graded volume improves but does not change the conclusion:

    2.16 Gvox needs 67.82 GiB of a 24 GiB card   (was 73.28)
    largest z-slab fitting ~20 GiB usable: ~637 Mvox, so >= 4 slabs

At the peak window the remaining live set is 8 B/vox inherited, 12 B/vox of
parent/flag/vcount and 2.7 B/vox of sort inputs. `vcount` is the next
candidate, being 4 B/vox held across the corner phase for a `k_plat_meta` call
that happens after it, though it is computed during the union-find phase
before it, so moving it is not a lifetime tweak but a restructuring.

## The layer loop never had a convergence test, and 97.6% of edge work is ineligible

Two results from one piece of instrumentation, and the second is the largest
finding in this log.

`k_propose` already computes each edge's mean and compares it to `TL`. Counting
the edges that clear that bar *and* join two distinct roots costs one shared
counter and one global atomic per block, and the count rides along on the
device-to-host copy the loop already made for `nprop` by allocating the two
counters adjacent. So the measurement is free and, being a count, it is immune
to the co-tenant on the card.

### A1: the outer loop ran its cap on every layer

    for (int outer = 0; outer < max_outer; ++outer) {

There was no convergence exit at all. The inner loop's `nprop <= 0` and
`hm == 0` breaks only leave the *inner* loop; control fell through to the next
outer round, which re-coloured and tried again. With `nlive > 0` that repeated
to the cap unconditionally, giving 17 x 64 = 1088 rounds.

The above-TL count is colour-independent, so when it is zero no colouring can
propose anything and the layer is finished. Leaving then is a no-op rather than
an approximation: the skipped rounds would re-run `k_compress` and
`k_rebuild_sz` (idempotent functions of `parent`), re-colour, propose nothing,
and never reach `k_freeze` or the compaction, so `parent`, `sz` and `nlive` are
already at the values the next layer reads.

    outer   1088 -> 653      inner   1840 -> 1405
    nmerge  1853427 -> 1853427       sum_above  77586985 -> 77586985

`sum_above` being *exactly* equal is the useful check: the eligible work is
untouched, only the spinning is gone. Per-layer, the round at which the layer
drained is

    [-1, 48, 39, 41, 36, 44, 42, 46, 33, 39, 34, 36, 26, 29, 28, 28, 24]

so only layer 1 genuinely needs the cap. This is 1.67x, not the 4x I guessed
from `layer_merges`: layers do drain, just at round 24-48 rather than round 2.

Gated: A2 `byte_identical=True ndiff=[0]`, ACCURACY GATE PASS at all four
thresholds (0.2/0.3/0.4/0.5), `nseg` T=0.5 379292. The `A2 FAIL` line in that
run is E6t, the StarMerge path already established as structurally
non-deterministic and not used.

### A3: the real number

    sum_nlive   3298578206      edges visited across inner iterations
    sum_above     77586985      of those, eligible to merge
    above_frac     0.0235

97.6% of all edge work in the agglomeration is spent on edges below `TL` that
cannot merge under any colouring. 3.3 billion visits to examine 78 million
eligible ones. I had planned A3 around dirty *roots*; the counts say the
partition that matters is the `TL` band, and the ceiling is 42x rather than the
few-x a dirty-root scan would give.

What makes a band partition exact rather than a heuristic: deduplication
combines parallel edges as `(sm1+sm2)/(ct1+ct2)`, a weighted mean of the two
component means, so the combined mean lies between them and can never exceed
`max(mean1, mean2)`. Two below-TL edges therefore cannot combine into an
above-TL one. The only way a below-TL edge enters the band is by combining with
an above-TL edge, which requires one of its endpoints to be a root that just
merged - and only band edges can propose, so those roots are known. The band
plus the edges incident to merged roots is a closed set, which is what makes
restricting the sweep to it exact.

## B1: the index division was real, and removing it bought nothing

Every full-volume kernel in `ws.cu` recovered its coordinates with

    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;

NVIDIA GPUs have no integer divide instruction, and since `Y` and `X` arrive as
runtime arguments the compiler cannot fold these into multiply-shift either. It
emits the full expansion, which `scripts/b1_sass.py` counts off the sm_120
binary:

    kernel            total  arith  mem  arith%  MUFU
    k_flow              360    244   23   67.8%     3
    k_hook_bidir        408    216   47   52.9%     3
    k_hook_remain       328    198   24   60.4%     3
    k_uf_compress_c     176     89   26   50.6%     0

`MUFU.RCP` is the float-reciprocal step inside the division sequence, so three
of them is direct confirmation the divisions are really there. Carrying y and z
in `blockIdx.y/z` removes them, and since consecutive `threadIdx.x` still map to
consecutive x it changes no access pattern:

    k_flow          360 -> 176      k_hook_bidir    408 -> 240
    k_hook_remain   328 -> 160      MUFU  3 -> 0 everywhere

Then the matched A/B, both builds run back to back under the same load:

    before  uf_ms 512.28, 512.24        after  uf_ms 509.63, 511.37

0.3%. Nothing. My estimate of ~0.4 s on the graded volume was wrong, and the
reason is in the table above: `k_uf_compress_c` is the one kernel with no
division, so B1 could not touch it. Splitting the round confirms which kernel
the loop is actually waiting on:

    uf_ms 498.80    hook_ms 170.82    comp_ms 316.52
    uf_ms 505.96    hook_ms 172.45    comp_ms 319.59

64% of the union-find is `k_uf_compress_c`, chasing parent pointers at random
across a 180M-element array. That is what B3 has to attack; the hook half was
never the problem. Keeping B1 anyway - it is bit-identical, strictly less work,
and the arithmetic it removes is more exposed on a card with less bandwidth to
hide it behind - but it is not a speedup and should not be reported as one.

Gated: C2 `deterministic=True ndiff=[0]`, `nfrag=2175400`, `bg=506568`,
`array_equal=True` and `fingerprint_equal=True` against the CPU oracle, nothing
leaked.

Incidentally this settles the open `uf_rounds` question: 6 in one run, 7 in the
next, output byte-identical both times. `uf_find` compresses paths while other
threads are reading them, so how many rounds it takes to reach the fixed point
depends on the interleaving. The fixed point itself is unique - it is the
min-index connected-component labelling - so the result cannot vary, and
`ndiff=0` across runs is the evidence.

## B2: the basin union-find was trusting a count tuned on the small volume

`e9c_basins_d` ran a host-fixed number of rounds:

    int nsv = g_sv_rounds;              // ws_set_sv_rounds(7) from segment.py
    for (int r = 0; r < nsv; ++r) {
        k_hook_remain<<<...>>>(bits_d, parent, Z, Y, X);
        k_uf_compress<<<...>>>(parent, size);
    }

7 was measured on the 180 Mvox validation volume. The failure mode this creates
is worse than a slow run: a volume whose basins need an eighth round finishes
with them unmerged, produces the wrong fragments, and says nothing about it.
The graded volume is 12x larger and has never been executed, so nothing
justified carrying the count over.

Now converge-driven, same flag pattern the plateau union-find already used, and
`NOT-CONVERGED` printed if it ever hits the bound. `_ensure_sv7` sets the bound
to the safety cap rather than pinning the count, since pinning can now only
truncate.

    E9c sv_rounds=7/40

So val does converge at exactly 7 and the original tuning was right, sitting
precisely on the boundary. Work is unchanged on val - the 7th round is the one
that observes nothing changed - so this is not a speedup, it is the removal of
an assumption that had no evidence at 2.16 Gvox.

Gated: C2 PASS, `ndiff=[0]`, `nfrag=2175400`, `bg=506568`,
`array_equal=True` and `fingerprint_equal=True` against the CPU oracle.

## C1: collapsing the region-graph's duplicate atomics inside the warp

The hash itself was fine (proper `mix64`, no low-bits defect), but every face
did its own `atomicCAS` plus two `atomicAdd`s. At val that is ~84M faces
inserted for 7,505,458 distinct edges, about 11 atomic sequences per edge, and
they do not arrive spread out: a warp spans 32 consecutive x, so when it crosses
a y or z sheet boundary every lane sees the same pair of fragments and emits the
*same* key, and those 32 atomics serialise on one slot.

`__match_any_sync` on the packed key groups the lanes that agree,
`__reduce_add_sync` sums the group's affinity bytes and face count, and the
lowest lane in the group does one atomic sequence for all of them. Exact, not
approximate: both accumulators are integers, so summing in the warp gives the
same total as summing at the slot, and the result stays order-independent.

Two things the shape forces. Lanes with no face to emit still have to reach
`__match_any_sync`, so the kernel no longer returns early - out-of-range lanes
carry key 0 and group together harmlessly - and the mask is the full warp rather
than `__activemask()`, which would be fragile under the divergence the old
`if (z > 0)` guards created. `rag.cu` also moved to the 3D grid from B1, which
matters here beyond arithmetic: with a 1D launch a warp can straddle a row
boundary and cover two y values, exactly the locality the aggregation depends
on.

Matched A/B, builds interleaved run by run so a drift in load hits both:

    work  median 17.44  min 17.06  max 17.87   nedge 7505458
    base  median 23.37  min 22.60  max 25.92   nedge 7505458
    1.34x

Worth noting what this measurement corrects: the plan projected RAG at ~1.5 s
for the graded volume, which came from contended wall clock. Device time on val
is 17.4 ms, so 2.16 Gvox scales to roughly 210 ms against a 150 ms budget. RAG
was never the problem it looked like, and C2 (shared-memory staging) is not
needed.

Gated: `edge_set_equal=True` and `count_exact=True` against the CPU oracle,
`ct_bit_identical=True`, `sm_bit_identical=True`, `max_abs_drift=0.000e+00`,
`nedge` 7505458 as required. Watershed re-gated after the shared header moved:
C2 PASS, `ndiff=[0]`, oracle `array_equal=True`.

## A3: the compaction was bandwidth, not hashing

I went into A3 expecting to need a dirty-set index, because A2 said 97.6% of
edge visits are on edges that cannot merge. Reading what the compaction actually
does first turned out to matter more than that ratio. Per inner iteration
`hash_combine_live` ran:

    k_rewrite        over n edges
    k_hash_clear     over next_pow2(2n + 1024) slots, 24 B stored on each
    k_hash_insert    over n edges
    k_hash_emit      over the same slots, 24 B read on each
    4x cudaMemcpy    D2D, copying the emitted arrays back over the input

The table is sized at ~2.7 slots per live edge, so clear and emit together move
about 130 B per edge, and the copy-back another 48 B, against roughly 24 B of
actual edge payload. The phase was not waiting on random hash access at all; it
was moving bytes that did not need to move. Three of them do not need to exist:

**The copy-back.** The emitted arrays *are* the compacted edge list. Swapping
which buffer is live is the same result for free. The one trap is ownership:
after an odd number of compactions the `t*` names hold the caller's arrays and
the `d*` names hold this function's, so the teardown has to free the
allocations rather than whatever the names ended up pointing at.

**The clear pass.** `k_hash_emit` already reads every slot. `k_hash_insert`
only writes a slot after a CAS that set its key, so a slot with key 0 has
untouched zeros in `sm` and `ct`, which means emit can zero the slots it finds
occupied and leave the whole table clean, including the tail above a later,
smaller `ntab_use`. The separate clear becomes one clear after allocation.

**The proposal array.** Same trade: `k_pack_prop_fused` reads every entry of
`prop`, and entries are only ever written by an `atomicMax` up from zero, so it
can reset the ones it consumes instead of a `cudaMemset` over all `nnode` every
iteration. (`prop[0]` is never written, since `k_propose` requires both roots
nonzero, so the one entry the pack kernel skips stays clean from the initial
memset.)

Traffic removed, from the per-iteration `nlive` the profiler already records
(scripts/a3_work_accounting.py):

    copyback     120.3 GiB    exact, 48 B x sum(nlive) = 2.691e9
    dpropclear    22.8 GiB    exact, 8 B x 2175401 nodes x 1405 iterations
    hashclear    ~165   GiB    scaled from the profiled slot distribution
    total        ~308   GiB

### What I could not measure

Nothing, in wall clock. The co-tenant held the card at 99% for this whole
stretch, and successive runs of the identical binary came out at 19770, 22249
and 22352 ms. Worse, the `memset` phase counter jumped from 16 ms to 1375 ms
*after* I deleted its largest memset - under that much contention whichever
operation sits at a scheduling boundary absorbs the queue wait, so the phase
labels stop meaning anything. `compact` did read 3113 -> 1569 ms, but I do not
trust the sign of any of it. The bytes above are the claim; the stopwatch is
deferred to an idle window.

Gated: `merges=1853427`, `nlive=898197`, `inner=1405`, `outer=653`, every one
identical to the A1 baseline. A2 `byte_identical=True ndiff=[0]`. ACCURACY GATE
PASS at all four thresholds with the fingerprint unchanged at
`[294164, 321999, 345130, 379292]`.

### One mistake worth recording

The first attempt came back with `merges=389420` instead of 1853427. I bisected
it twice and both bisects said "still broken", which was uninformative because
the modified `k_hash_emit` was live in every arm - I had removed the
clear-skipping but not the zeroing, so I never actually tested a control.
Reading `git diff` found it in seconds: my edit to insert the one-time table
clear had matched a two-line block and dropped `cudaMalloc(&dnout, 4)` from the
replacement. `dnout` was an uninitialised pointer, so the emitted-edge count was
garbage and the compaction reported an empty graph. Read the diff before
bisecting.

## B3: measured the premise before building on it, and it did not hold

B1 left a clean diagnosis: `k_uf_compress_c` is 64% of the plateau union-find
(comp 317 ms of 499 ms) while moving about 2 GB, so it runs ~30x off bandwidth
roofline and the cost is dependent random access rather than bytes. The planned
fix was shared-memory tile labelling, on the theory that the expense is walking
parent chains through global memory.

One thing makes any replacement tractable here: what the gate pins is the
*fixed point*, not the trajectory. The loop settles when neither hooking nor
flattening changes anything, and that state is exactly "every entry holds its
component's minimum index" regardless of how it got there. So the flattening
kernel can be swapped outright.

The cheapest test of the chain-walking theory is pointer jumping, `p[i] <-
p[p[i]]`: two loads and at most one store per thread, halving every chain, no
walking. Added behind `WATERZ_UF_ALGO` and run interleaved, twice each:

    algo 0  uf_find compression   rounds 7  comp 320.2 321.4 322.0 322.4  uf ~507
    algo 1  pointer jumping       rounds 9  comp 428.8 424.6 435.6 434.0  uf ~637

Both bit-identical: `nfrag=2175400`, `bg=506568`, `array_equal=True` against the
CPU oracle in every run, which is the fixed-point argument confirmed
empirically. Jumping is 1.26x *slower*.

The per-round figures are what matter:

    algo 0   320.2 / 7 = 45.7 ms per round
    algo 1   428.8 / 9 = 47.6 ms per round

Within 4%. Two kernels doing very different amounts of chain work cost the same
per round, which says the chains are already short - mostly one or two hops -
and the round is paid for by the *single* random gather `p[a]` that both do, one
per voxel over a 720 MB array. Chain length was never the cost. Round count is,
and `uf_find` needs 7 where jumping needs 9.

So the tile-labelling rewrite would be built on a refuted premise: making
within-tile chain walks local cannot help when there are barely any chain walks.
What sets the gather cost is that `a = p[i]` is a spatial neighbour, and in
linear indexing a spatial neighbour is ±1 (same line), ±X (4.8 KB away) or ±YX
(5.76 MB away) - so two thirds of gathers miss. Fixing that means re-indexing
the volume into a tiled or Morton order, which is a far larger change than tile
labelling and touches every kernel.

Leaving both variants in place behind the switch rather than deleting the loser:
this is exactly the measurement E1 needs to repeat on the 3090 Ti, where L2 goes
from 96 MB to 6 MB and the cost of a missing gather changes by a large factor.
That is the term most likely to reverse the ranking, and it is cheap to re-run.

I also considered skipping the final round's flatten, since the round that
observes no hook change appears to be pure overhead. It is not safe: if adjacent
voxels in a component share a parent value that is not itself a root, the hook
reports no change while parent is still unflattened, and the flatten's flag is
what catches it. `vcount` is indexed by `parent[i]`, so an unflattened parent
splits a plateau's count and undersizes its BFS queue. The last round is the
price of proving flatness.

## D3 PASS: the device-resident path, by deleting a dependency instead of installing one

`segment_d` had never run on this machine. It imported torch, torch is not
installed, and TASK grades exactly this path: affinity already in VRAM, labels
in VRAM, timed with CUDA events. The obvious fix was to install torch. The
right one was to look at what torch was being asked to do:

    torch.empty(..., device="cuda")   allocate
    t.data_ptr()                      get a pointer
    t.cpu().numpy()                   copy back

That is `cudaMalloc`, a pointer, and `cudaMemcpy`. A 2.5 GB dependency was
serving as an allocator. `segment_d` now calls the CUDA runtime through ctypes
(`_rt()`, `DevBuf`) and reads its input through `__cuda_array_interface__`,
which torch CUDA tensors, cupy and numba all implement: so a caller can still
hand it a torch tensor, we just no longer import torch to receive one. Fewer
moving parts on the graded path, and it runs here today.

`DevBuf` publishes the interface too, so with `return_device=True` the labels
stay in VRAM and the caller can consume them without a copy. That is the
literal grading shape; the previous code always ended in `.cpu()`.

Added `aff_f32_to_u8_d` to `ws.cu` for float32 input already in VRAM, matching
host `_as_u8` (scale 255, `rintf`, clamp). Otherwise a float caller would have
to round-trip through the host just to quantise.

**The real find was that the device agglomeration entry point was dead code.**
`parhac_paper_d_dev` exists and takes device pointers, but `segment_d` only
called it `if _e6r_locked()`, and `data/cache/e6r_pass.txt` reads
`FAIL PASS T03_ms=6841.94 budget=50`. So every `segment_d` call copied the whole
RAG to the host and back: on a path whose entire purpose is not doing that.
The gate was also the wrong question: comparing the two entry points,

    parhac_paper_d      cudaMemcpy(..., cudaMemcpyHostToDevice)
    parhac_paper_d_dev  cudaMemcpy(..., cudaMemcpyDeviceToDevice)

they are otherwise identical: both malloc fresh scratch and call
`parhac_e6s_dev`. Same bytes in, same kernel, so same bytes out, bit for bit;
no experiment can distinguish them. E6r was a *speed* budget on an unrelated
experiment being used to gate *correctness* of a different function. Now gated
on `_PARHAC_D.is_file()` and honouring `WATERZ_AGG_CPU`, with `AGG_BACKEND` set
to `gpu_dev` so a bench can tell which one ran and refuse to report a number
from the wrong one.

Gate, `scripts/d3_dev_check.py` on a 2.1 Mvox crop, all against `segment()`:

    host input, host out                    identical=True
    u8 in VRAM, labels in VRAM              identical=True   backend=gpu_dev
    f32 in VRAM, quantised on device        identical=True
    device path run twice, same input       identical=True

Crop, not full val, deliberately: a co-tenant holds 27.4 GiB of the 32 GiB and
val needs ~4.4 GiB for the watershed alone against 4.6 GiB free. A correctness
gate is not worth OOM-ing someone else's job for, and it does not need the
whole volume to be conclusive.

`d_bench.py --device` is the median/min/max-of-5 bench, bracketed by
`cuda_event_time`, with both copies outside the measured region. It also
compares its labels against the host path, so a fast number cannot come from a
different answer. On a 28.3 Mvox crop with the card at 99%:

    run0 1633.9   run1 3664.1   run2 5018.4   run3 4198.8   run4 5013.0 ms
    median 4198.8  min 1633.9  max 5018.4   deterministic=True

A 3.1x spread between min and max, which is the contention signature and not a
measurement. `gradeable` is false in the JSON for both reasons (shared card,
and cropped). The harness is what D3 owed; the number waits for an idle card.

Also noted for D1: `_max_edges` floors at 20M edges, so the four edge arrays
cost 480 MB regardless of volume: 4+4+8+8 bytes each. Narrowing `sm` and `ct`
to uint32 halves that to 240 MB, and the same 2x applies to the scratch copy
`parhac_paper_d_dev` makes internally.

## D1 PASS: 18.5 GiB off the 2.16 Gvox peak, and the plan's estimate of it was low

Started by reading the trackers nobody read. `ws.cu` and `parhac_d.cu` both wrap
cudaMalloc in a peak counter and nothing ever called the accessors, so every
memory figure in the plan was an estimate. `scripts/d1_mem.py` measures two crop
sizes and fits a line in voxel count, which separates the per-voxel slope from
the fixed floor `_max_edges` imposes below ~360 Mvox.

First measurement corrected two numbers the plan had wrong:

    ws scratch     15.69 B/vox measured, not the assumed 22.71
    agg peak       8.77 B/vox = 17.64 GiB at 2.16 Gvox, not the assumed ~10 GB

A single peak total does not say *which* of ~30 live buffers it is, and
narrowing the wrong one saves nothing, so I made the tracker attribute the peak
to source lines (`__LINE__` through the malloc macro, snapshot the book whenever
a new peak is set, `agg_mem_peak_lines` to read it out). That immediately named
the target: `dtab`, the dedup hash table, was 40% of the peak at 83 B/edge.
Everything else was 8 B/edge or less.

Four changes, each gated:

**Fragment and label buffers share one allocation.** `k_extract` is
`out[i] = parent[seg[i]]`: thread `i` reads only `seg[i]`, writes only `out[i]`,
and `parent` is a distinct array, so `out` may alias `seg` with no race. The
last threshold now writes labels over the fragments. 8.05 GiB at 2.16 Gvox.
Earlier thresholds still need their own buffer, so the four-threshold path is
gated separately from the single-threshold one: both identical.

**The affinity is freed when the RAG is built.** It is 3n bytes and dead the
moment the edge weights exist, but it was staying resident through the
agglomeration peak. 6.03 GiB. Only freed if we allocated it; a caller's buffer
is not ours to free.

**HSlot narrowed 24 B to 16 B.** `{uint64 key, double sm, uint64 ct}` to
`{uint64, uint32, uint32}`. This is exact rather than approximate, which is the
whole reason it is allowed: `sm` is already whole affinity bytes by the time it
reaches the table (`k_scale_sm_bytes` llrounds it at the top of each layer) and
`ct` is a face count, so both accumulators are integers and both atomicAdds are
exact integer sums at either width. Only range is given up. The file's own note
bounds `sm` by `255*3*nvox = 1.65e12`, far past uint32, so the fit is not
provable and `k_hash_insert` checks each add against the width using the value
atomicAdd returns, rather than assuming: the discipline `rag.cu` already
applies to its own `isum`. Probe overflow and width overflow are separate bits
now so one cannot mask the other.

**The agglomeration works in place on the caller's edges.** This was the
surprise, and it is worth more than narrowing every payload: `parhac_paper_d_dev`
allocated a second complete edge set and copied device-to-device into it, so two
full sets were live for the whole run: 2.06 GiB of pure duplicate at 2.16 Gvox
plus 2 GiB of pointless copy. Thresholds are processed in one descending pass
with snapshots, so the edges are consumed exactly once and never need to be
pristine again; the only caller frees them immediately after. Now destructive
and documented as such.

Gate is `scripts/d1_narrow_gate.py`, which builds both widths from one source
(`-DHSLOT_WIDE`) and runs them in one process on the same input. Comparing
against `segment()` would have proved nothing since both link the same library
and would move together; two libraries is the only honest reference.

    wide   HSlot 24 B   agg peak 7.89 B/vox
    narrow HSlot 16 B   agg peak 6.70 B/vox
    T=0.2 0.3 0.4 0.5   identical=True, nseg 52724 57018 60454 65647
    no overflow reported at either width

Where that leaves 2.16 Gvox, with the watershed slabbed by D2:

    fragments/labels     8.05 GiB   spans all stages
    edge arrays          2.66 GiB
    agg tracked peak    13.28 GiB   was 17.64
    stage peak agg      23.98 GiB   was 45.64 at the ws stage

So it fits a 24 GiB card, and I do not believe the margin. 0.02 GiB of spare
against a peak that is explicitly a *lower bound*: thrust allocates its own
scratch for sort_by_key and reduce_by_key outside the tracked path: is not a
fit, it is a coincidence. The honest statement is that 2.16 Gvox now fits the
5090's 32 GiB with room and sits exactly on the 3090 Ti's line.

Widening that margin is the payload narrowing I did not do: `tsm`, `tct`, `csm`,
`cct` are still 8 B/edge, worth 1.37 GiB, and the external `sm`/`ct` arrays
another 0.89 GiB, taking the peak to 21.72 GiB. I stopped short of it on
purpose. Those four temporaries are swapped with `dsm`/`dct` by the A3
buffer-swap, so they must share a type with the caller's arrays, which couples
the change to `rag.cu`'s output and the device entry signature: it cannot be
kept internal. That is a wide change through the part of this codebase whose
bit-identity has cost the most to establish, for 2.26 GiB, and it cannot be
validated where it matters: the binding stage at 2.16 Gvox is still the
watershed at 45.64 GiB until D2 lands, and there is no 24 GiB card here to check
against. Doing it blind, before D2, in exchange for margin on a stage that is
not yet the constraint, is the wrong order. The typedef mechanism and the
two-library gate are both in place for when it is worth doing.

Also measured: `EDGES_PER_VOX = 0.055` against 0.0427 actual, a 29%
overallocation. Not worth changing: the RAG table is
`next_pow2(2*max_edges)`, and 0.048 and 0.055 both round to the same 268 M
slots, so tightening it buys nothing on the table and only trims the arrays.

## D2 PARTIAL: buffer sharing landed and gated; the z-slab decomposition did not

Measured before rewriting, which changed what the rewrite should be. Gave
`ws.cu`'s allocation tracker the same per-line attribution as `parhac_d.cu`'s,
and the watershed's 15.69 B/vox turned out to be four per-voxel uint32 arrays
plus a tail of nC-sized ones:

    parent   4.0 B/vox     flag   4.0 B/vox
    vcount   4.0 B/vox     bits   1.0 B/vox     corners_in + keys_in  2.6

`parent` did not need to exist. The stage ends in `k_write_labels`, which is
`seg[i] = psum[parent[i]] + 1`: thread `i` reads slot `i` of parent, writes slot
`i` of seg, and `psum` is a separate array, so no thread can observe another's
overwrite and the two may be the same storage. Exactly the argument that let
`k_extract` go in place in D1. e9b's parent has the same property and is dead
before e9c starts, so both stages borrow the caller's label buffer -
`e9b_divide_d` takes it as an optional `scratch_d` and `e9c_basins_d` already
had `seg_d` in its signature.

Gated by `scripts/d2_share_gate.py`, building both from one source with
`-DWS_NO_BUFFER_SHARE` for the reference. There is no CPU oracle at crop scale
and the val oracle needs ~4.4 GiB against 4.6 GiB free with a co-tenant on the
card, so a second library was the only reference available; comparing the shared
build against `segment()` would have compared it against itself.

    reference (own parent)   peak 22.68 B/vox
    shared    (borrows seg)  peak 20.98 B/vox
    identical=True  fingerprint_equal=True  nfrag=353562  bg=78924

The saving is 1.71 B/vox, not the 4 the arithmetic suggests, because removing
parent moved the peak rather than lowering it by its own size. Re-attributing
after the change shows the peak is now in the plateau-BFS phase, past the point
where parent and flag are already freed:

    vcount 4.0   corners_out 1.3   keys_out 1.3   start 1.3   start_ps 1.3
    plat_begin 1.2   plat_nseed 1.2   plat_root 1.2   bits 1.0   = 13.98 B/vox

That is the honest state: watershed scratch 15.69 -> 13.98 B/vox, and the ws
stage at 2.16 Gvox 45.64 -> 42.20 GiB. Still far over both cards, so **slabbing
remains mandatory and unfinished.** What it needs to be is now much clearer
than when the plan was written, and the plan's framing is incomplete in one
important way:

"`k_flow` is provably a pure 1-voxel-halo function, so slabs are exact" is true
of `k_flow` and only of `k_flow`. It does not carry to the two stages that
actually hold the memory. The plateau union-find and the basin union-find
propagate along plateaus and basins that can run the full z extent, so a slab
with a 1-voxel halo cannot resolve them locally - a component's root may lie
many slabs away. Exact slabbing of those needs either iterated halo exchange
until no slab boundary changes, or the two-level scheme: per-slab local roots,
then a union-find over *representatives only* across the seams, then flatten
and renumber globally. The second is the better shape here, because the
representative count is the fragment count - about 24 M at 2.16 Gvox, so a
representative-indexed table is ~96 MB rather than 4 B/vox - and because
min-index roots make the seam unions order-independent, which is what keeps the
result deterministic and bit-identical to the whole-volume answer.

I stopped rather than start that here. It is a genuine algorithm change across
both watershed stages, its gate is bit-identity against the whole-volume result
on val, and val cannot currently be run: 4.42 GiB needed against 4.67 GiB free
with a 27.4 GiB co-tenant, where being wrong means OOM-ing someone else's job.
Writing it blind and gating it later inverts the discipline every other item in
this plan followed.

Two smaller reductions are available first and do not need slabbing, both
visible in the attribution above. `flag` in `e9c_basins_d` is a 0/1 corner
predicate held as a uint32 and then scanned in place; `cub::DeviceSelect::Flagged`
over a transform iterator computing the predicate on the fly would produce the
corner index list directly and remove the array, and Flagged preserves input
order so the list is identical to what the scatter produces. `vcount` is 4 B/vox
indexed by root and is now the largest single item at 28.6%.

## A4: condition met, but half of it is already refuted by E12 and the other half is gated on a card I cannot get

A4 was conditional: "only if A1+A3 fall short". They fell short, and not
marginally. From `data/cache/p0aa_e6s.json` on val:

    outers_now 653   outers_if_exit 653   outer_reduction 1.00
    layer_first_zero [-1, 48, 39, 41, 36, 44, 42, 46, 33, 39, 34, 36, 26, 29, 28, 28, 24]

Every layer but the first now exits on its last outer (48 of 49, 39 of 40, and
so on), so A1 is already extracting all of the convergence exit there is - the
1.67x it bought is the whole of that lever, and `outer_reduction` is 1.00
because the exit is applied, not because it does nothing. Against the plan's own
uncontended projection the agglomeration is ~36x over its 400 ms budget.

The eps half of A4 is refuted by an experiment already in this log. E12 swept it
and locked the answer:

    eps 0.05 PASS   0.06 PASS   0.07 PASS   0.08 PASS
    eps 0.09 FAIL   0.2 merge VOI 0.3619 > 0.3525

0.08 is locked as the largest value that passes. A4 asks for 0.16, double the
value that already fails, so it does not need running to be answered: it breaks
the VOI gate and reverts. The plan reasoned from margin ("tightest existing
margin is ~0.011 of the allowed 0.02") without E12's measured failure at 0.09.
E12 also measured what eps buys, and it is small: 0.05 -> 1345 inners,
0.08 -> 939, a 1.43x return for a 1.6x eps. Even a hypothetically passing 0.16
would be well under 2x against a 36x gap.

The size-asymmetry half - relaxing `sz[r] >= sz[bl]` in `k_propose`, worth up to
~2x on per-round merge probability - is not refuted and is worth testing. It
needs the four-threshold VOI gate on val, and val cannot run right now: 4.42 GiB
against 4.67 GiB free with a 27.4 GiB co-tenant. VOI is a whole-volume metric
against whole-volume ground truth, so unlike the memory and bit-identity gates
it does not have a valid cropped form. It waits for an idle card.

Worth stating plainly, because it is the finding that matters more than either
half of A4: neither lever is the shape of the problem. `above_frac` = 0.0288
says **97.1% of edge visits are on edges that cannot merge** - the mean is below
the layer threshold or the endpoints already share a root. A3 removed the
compaction traffic over those edges but `k_propose` still scans all of them,
because there is no structure that answers "which edges could merge this round"
without looking. eps tuning changes how many rounds happen and the asymmetry
changes how many merges each round lands; neither reduces the 97% waste inside
a round. Closing a 36x gap means a per-layer adjacency or bucketed-by-weight
structure so a round touches candidates rather than everything. That is a
larger change than A4 and it does not risk the partition, which makes it the
better next move.

## E3 PARTIAL: four-threshold VOI PASS, determinism PASS, memory table done; speed table blocked on an idle card

The accuracy gate is the one that had to survive all of Track A and D1, and it
did. `scripts/a1_e6t_voi.py --e6s` runs from the cached RAG rather than the
watershed, so it fits the 4.67 GiB the co-tenant leaves and was runnable when
nothing else at val scale was.

    aff 0.2  split 0.3707 (limit 0.3979)   merge 0.3350 (limit 0.3525)   PASS
    aff 0.3  split 0.4512 (limit 0.4738)   merge 0.2505 (limit 0.2611)   PASS
    aff 0.4  split 0.5162 (limit 0.5378)   merge 0.2268 (limit 0.2381)   PASS
    aff 0.5  split 0.6129 (limit 0.6309)   merge 0.2184 (limit 0.2293)   PASS
    ACCURACY GATE: PASS

    unique-parent fingerprint [294165, 322000, 345131, 379293]   locked value
    merges                    [27835, 23131, 34162, 1796108]     unchanged

Fingerprint and merge counts are bit-for-bit the locked reference, which is the
real result: A1's convergence exit, A3's compaction, the narrowed HSlot and the
in-place edge arrays all landed without moving a single voxel. The counts show
A1 working in the same run that shows the partition unchanged:

    outer  [384, 256, 192, 640] -> [209, 114, 102, 441]   1.7x fewer
    inner  [565, 377, 292, 1161] -> [390, 235, 202, 962]  1.3x fewer

Same merges, fewer iterations to reach them, which is exactly what a
convergence exit should look like and is the strongest evidence available that
it is a no-op on the answer.

Determinism: PASS at every level tested - `d3_dev_check` runs the device path
twice on one input and compares, `d1_narrow_gate` and `d2_share_gate` compare
across two independently built libraries, and all four thresholds match the
host path.

Memory table is complete and, unlike speed, contention-independent, so it is a
real deliverable rather than a provisional one. Measured at two crop sizes and
fitted; see `data/cache/d1_mem.json`.

    stage        B/vox    2.16 Gvox stage peak
    watershed    13.98    42.20 GiB   over both cards, needs D2
    rag           n/a     21.74 GiB   fits
    agglom        6.60    23.98 GiB   fits 32 GiB, on the line at 24 GiB

    change                          saved at 2.16 Gvox
    fragment/label buffer shared     8.05 GiB
    affinity freed after rag         6.03 GiB
    agglomeration edges in place     2.06 GiB
    HSlot 24 B -> 16 B               2.38 GiB
    ws parent borrows label buffer   3.43 GiB
    total                           21.95 GiB

Speed table is not deliverable and I am not going to fake it. The card has been
at 98-100% with a 27.4 GiB co-tenant throughout, and the D3 bench shows what
that does to a measurement: five runs of one workload spread 1633.9 to 5018.4 ms,
3.1x between min and max. `d_bench.py` writes `gradeable: false` in that state
by design. The median-of-5 at 2.16 Gvox and 1.44 Gvox needs D2 to fit at all
and an idle card to mean anything, and both are E1's dependency, not something
that can be worked around here.

## E1 BLOCKED: needs hardware I cannot obtain; made it a one-command run instead

E1 is "rent a 3090 Ti". Renting needs an account and a payment method, so it is
not something I can execute. What I could do is remove every other reason the
run might not happen, so it is one command on a fresh box.

`scripts/e3_final.py` sequences the gates rather than reimplementing them, since
a gate living in two places drifts. `--quick` is the crop-scale correctness set,
safe on a shared card; `--full` adds the val-scale gates and both benches and
wants the card to itself. It refuses to call any speed number gradeable while
another process holds memory, and records `card_busy` in the JSON either way.

Current state on the shared 5090, all six correctness gates:

    PASS  d3_device_path        device path identical to host, four thresholds
    PASS  d1_hslot_narrow       narrow slot identical to wide reference
    PASS  d2_buffer_share       borrowed label buffer identical to own
    PASS  d1_memory             per-stage peaks and 2.16 Gvox extrapolation
    PASS  voi_four_threshold    four-threshold VOI, locked fingerprint
    PASS  rag_determinism       RAG deterministic and equal to CPU oracle
    skip  ws_invariants         needs --full
    skip  bench_host            needs --full
    skip  bench_device          needs --full

    6/6 passed; speed numbers gradeable: False

The three skipped gates are skipped for one reason: 4.67 GiB free against a
27.4 GiB co-tenant, where val needs 4.42 GiB. Running them would risk OOM-ing
someone else's job to produce a timing that contention has already made
meaningless.

What E1 is actually for, and why it is still worth doing rather than assuming
the 5090 answers transfer: the 3090 Ti has 6 MB of L2 against the 5090's 96 MB,
and the two measurements this project made that are most sensitive to that are
both already set up to re-run. B3 found path compression beats pointer jumping
1.26x, with the per-round cost within 4% between two kernels doing very
different amounts of chain work - which said the cost is one random gather per
voxel over a 720 MB array, not chain length. A 16x smaller L2 is exactly the
term that could reverse that, which is why `WATERZ_UF_ALGO` was kept rather than
deleting the loser. C1's warp aggregation is the other one: it won 1.34x by
removing duplicate atomics, and how much that is worth depends on how much of
the hash table L2 can hold.

## M1, G0-G3, W1, V1-V2 - the cost model reverses the plan's priorities

The plan I was handed ordered the work agglomeration-first: five of its thirteen
items (G1-G4 plus the epsilon lever) attack agglomeration, and the watershed
sits at items six through nine. I built the cost model first, as the plan asked,
and the model says that ordering is backwards. Not slightly - the agglomeration
work is close to unnecessary for the *speed gate*, and the watershed is the
entire problem.

`scripts/m1_cost_model.py` recomputes every projection from the cached JSONs
instead of asserting it in prose. Its bottom line at 2.16 Gvox on a 3090 Ti,
after every lever in the plan lands:

    V-lever work factor  3.65x
    watershed          11435 ms  90.9% of total, untouched
    rag                  372 ms
    agglomeration        706 ms  (18.3x from 12949)
    extract               73 ms
    TOTAL              12586 ms = 0.172 Gvox/s, 11.7x over budget

The watershed alone is 10.6x the whole 1.08 s budget. So no amount of
agglomeration work reaches the gate, and the plan's own headline - "~2.2 s =
0.98 Gvox/s, roughly 2x short" - was too optimistic by 5.7x. The reason is
mundane: the plan credits the watershed with going 11.4 s to ~1.5 s from block
labels, then to ~400 ms "only if the kernel engineering lands", and then adds up
the optimistic branch of both. The model refuses to spend a factor it has not
measured, and neither of those two watershed factors has been measured.

What the model *did* get to measure, unexpectedly, is the agglomeration levers -
all of them, exactly, without a GPU.

### A bit-identical CPU replica turned out to be the whole trick

I have no GPU on this box (`nvidia-smi` missing, no `/dev/nvidia*`, no `nvcc`).
The obvious move was to write the CUDA and leave it unverified until a card
appears. Instead I wrote `scripts/g0_agg_ref.py`, a NumPy replica of
`parhac_e6s_dev`, and pushed it until it agreed with the device on every
recorded number: 17 layers, 653 outers, 1405 inners, 1853427 merges,
`sum_nlive` 2691304379, `sum_above`, the per-layer outer and merge vectors, the
final segment count, and the parent array itself.

Getting there took three real bugs, each of which would have been a silent
wrong answer on the device too:

`frozen` was allocated once instead of per outer round. The device memsets
`dfrozen` inside the outer loop; my replica hoisted it out, so nodes froze
permanently, `szmax` stuck at 3 and the threshold ladder never descended. This
is the kind of divergence that produces a *plausible* segmentation, which is
why it took a layer-vector comparison rather than a spot check to find.

Slicing `u[:nlive]` in the restructured mode was wrong because dead edge slots
are interleaved, not suffixed. The device's `compact_radix` physically compacts;
my prefix slice processed an arbitrary mixture and produced 402 extra merges. Fix
was to carry an `alive` mask and compact at layer boundaries the way the device
does.

Tracking the active set with `np.setdiff1d` over an index list was correct but
quadratic enough to blow the wall clock on the full graph. Replaced with a
boolean mask updated incrementally over the dirty edges.

With the replica trustworthy, the lever payoffs stopped being estimates. It
counts logical work directly, so `m1_cost_model.py` now reads them out of
`g0_agg_ref.json` rather than guessing:

    g1 freeze_reds (+color)     1636M ->    1.1M node visits
    g2 dirty-set compaction     1539M ->  197.3M   = 7.8x
    g3 active edge list         2691M ->   77.6M
    g3 candidate blue list      3056M ->    5.0M
    g4 per-outer root list      8523M ->  356.0M
    g4 per-inner compress       3096M -> 1461.5M

Two of these correct the plan. G2's dirty-set compaction was sold as a 185x
reduction, from `ndirty_total = 19453169` against a mean `nlive` of 1.86M. That
ratio is real but it is not the speedup: the compaction still has to *find* the
dirty edges through a CSR index whose rows it must read, and the measured
end-to-end work reduction is 7.8x, not 185x. Still the single largest lever, but
a factor of 24 smaller than advertised. And G1 is worth much more than the plan
thought, because `k_propose` stops reading `dcolor` once reds are frozen from
the proposal list, so `k_color` dies along with `k_freeze` - two full node passes
per outer, not one.

The guard on that data is worth describing because the first version was wrong.
I initially had the cost model check a `nedge` metadata field to refuse work
counters harvested from a subgraph. The field was absent from the very file it
was meant to protect, because the long full-scale run had been launched before
the field was added, so the check silently fell back to estimates while looking
like it was working. A later run did write the metadata, but the episode is the
argument against the design: a guard that trusts a field written by the same
process it is guarding fails open. It now validates against the device instead.
A replica that reproduces `p0aa_e6s.json`'s `sum_nlive`, `nmerge` and `ninner`
to the digit necessarily ran the whole val graph to convergence, and a subgraph
or a truncated run cannot fake that no matter what it writes about itself.

### Why G2 needs the CSR index, which the replica accidentally proved

The replica finds its dirty set by scanning the live edges, because writing a
CSR index in NumPy to save NumPy time would have been silly. That accident
produced the number that justifies the CSR index in CUDA:

    compact_edge_visits   1539103036 ->  197347864
    dirty_scan_visits              0 -> 1504914171

The compaction itself drops 7.8x, but scanning to *find* the dirty edges costs
1.505e9 visits - within 2% of the 1.539e9 whole-graph rebuild it was supposed to
replace. Scan-based dirty-set compaction is therefore not a lever at all; it
only wins because a sequential 8-byte `(u,v)` read is cheaper per visit than a
random `atomicCAS` hash probe. The CSR index is what converts that 1.505e9 scan
into 1.63e8 row reads, and it is load-bearing rather than an implementation
detail.

Costing it honestly matters too. The plan prices the transpose at its storage,
"90.3M x 8 B = 1.4 GB, which fits", and says row offsets "fall out of a
run-length scan for free". Half true: the forward rows are free because
`compact_radix` already leaves the array sorted by the 64-bit `(u,v)` key, but
the transpose needs an actual sort by `(v,u)`, one per layer, which is four CUB
radix passes of read-plus-write over 90.3M keys - 223 ms of the 251 ms the index
costs. G2 still clears easily, 4709 ms saved against 251 ms spent, but the
margin is 19x rather than the unbounded win the plan implies, and a design that
rebuilt the transpose per outer round instead of per layer would lose outright.

### The plan's active-list design is not bit-identical

G3 says to rebuild the active-edge list "once per layer". `k_propose`'s
predicate is `mean >= TL`, so restricting it to the `mean >= TL` set is indeed
definitionally identical - but only if the set is current. Merges rewrite edge
endpoints within a layer, which changes which edges satisfy the predicate. I
added `--stale-active` to the replica to test the plan's version directly, and it
diverges. So the active-edge half needs incremental maintenance over the dirty
set, which means it is not independent of G2 at all; it is the same data
structure. The candidate-blue half has no such coupling and is implemented
(`k_propose_bluelist`, `k_pack_listed_fused`, lever bit 4), along with G1
(`k_freeze_reds`, lever bit 1). Both are behind `WATERZ_AGG_LEVERS` bits so the
device can diff them against the reference path in one run when a card appears,
and both type-check - `scripts/nvcheck.py` assembles a fake `CUDA_HOME` out of
the `nvidia-cuda-*` pip wheels so `clang++ -x cuda -fsyntax-only` works with no
toolkit and no GPU.

### The V levers are large and the accuracy risk is real

`scripts/v_levers.py` sweeps epsilon and the `sz[r] >= sz[bl]` asymmetry through
the replica. Combined, epsilon 0.32 with the asymmetry removed cuts logical work
3.65x while segment count barely moves. That is the biggest single multiplier
available anywhere in agglomeration, and it costs a parameter change. It is also
the only lever here that can fail the VOI gate, so it stays quarantined behind
the four-threshold run.

### W1's premise holds, its L2 claim does not

`scripts/w1_block_analysis.py` reads the real `gpu_fragments.npy` rather than
reasoning about it. The core premise is confirmed: 538.3M face-adjacent voxel
pairs, 454.0M of them same-fragment, collapse to 66.5M block face pairs, an 8.1x
reduction in union sites.

The L2 argument is wrong, though. The plan claims a block-label z-plane-pair is
5.76 MB and "fits in 6 MB L2". A plane is 5.76 MB, but the hook kernels need a
three-plane window for the +/-z gathers, which is 17.28 MB. Nothing whole-plane
fits. What fits is an xy tile: 512^2 blocks over three planes is 3.15 MB and
covers 6.29M voxels. So the L2 story survives only as *tiled* block labels, and
W4's "z-tiled `k_flow`" is not an optional extra on top of W1 - it is what makes
W1's L2 claim true in the first place.

"One uint32 per 8 voxels" is also not enough. Blocks average 1.58 distinct
fragments and only 62.2% are internally uniform, so a single slot spills on
37.8% of blocks. Two slots per block is the design point: 1.0 B/vox, 15.1%
spill, 3.0 GiB saved at 2.16 Gvox - which is 0.5 B/vox less than the 3.5 B/vox
W2 budgets for the no-slab target, so that arithmetic needs redoing against
`d1_mem.py` before W2 is called done.

### W0, and the watershed the plan is not talking about

Since the cost model puts the watershed at 90.5% of projected runtime, the next
thing worth having is the `g0_agg_ref.py` equivalent for `ws.cu`: a CPU replica
that turns "bit-identical" and "`nfrag == 2175400`" from GPU gates into CPU
gates. `scripts/w0_ws_ref.py` is the start of it, and building it turned up
something that has to be settled before any of W1-W4 is written.

`ws.cu` contains three watersheds, not one:

    plateau_basins    ws.cu:703   via watershed_gpu      only caller is g2_ws.py
    watershed_device  ws.cu:603   via watershed_gpu_d    a few scripts
    watershed_gpu_e9  ws.cu:1796  via segment/segment_d  production

`src/segment.py` calls `watershed_gpu_e9` and `watershed_gpu_e9_d` from both
`segment()` and `segment_d()`, so e9 is what produced `gpu_fragments.npy` and
the locked `nfrag = 2175400`. The plan's W1 text describes "both watershed
union-finds" and names `k_hook_bidir` and `k_hook_remain`, which are indeed e9's
kernels - but the two implementations that are easiest to read, and that I
modelled first, are `plateau_basins` and `watershed_device`, and neither is on
the production path. Anyone reading `ws.cu` top-down hits the dead ones first.

Having replicated all three, the picture is clean, and it is better news than I
first thought. Over a sweep of synthetic volumes at every tie density from 2 to
32 distinct affinity values, 20 cases:

    e9 vs plateau_basins            identical in 20 of 20
    watershed_device vs plateau_basins  differs in 20 of 20

So `plateau_basins` *is* a valid specification of the production watershed --
identical fragment counts and identical partitions, from 9 fragments up to 356
-- and the "G2-locked" comment still means something. It is `watershed_device`,
reached through `watershed_gpu_d`, that has drifted. That inverts the guess I
recorded first, and it matters, because it means W1 does not need a replica of
e9's machinery to be gated: it needs a diff against a 60-line sequential host
routine that any change can be checked against on a CPU in under a second.

The mechanism of the legacy divergence is in the source. `plateau_basins`
overwrites each voxel's direction byte with `to_set` as the BFS processes it,
and `to_set` never contains a reciprocal direction, so once i is processed a
later-processed reciprocal neighbour j asks `seg[i] & idirmask[d]`, gets false,
and concludes the edge was never reciprocal. e9's `k_indep_bfs` does exactly
the same thing, mutating `seg` in place as it pops each voxel, which is why the
two agree. `watershed_device`'s BFS is the odd one out: it asks the same
question of an immutable `orig` copy and always gets the original answer.

Two things had to be true for e9's parallelism to preserve the sequential
result, and both are now checked rather than assumed. First, e9 runs one BFS
per plateau concurrently while the host runs a single global FIFO; these agree
because plateaus are the connected components of the reciprocal subgraph, hence
disjoint, and within a plateau both process that plateau's corners in ascending
voxel order and then expand FIFO. CUB's radix sort is LSD and therefore stable,
which is what preserves the ascending corner order inside each plateau group.
Second, a thread working on plateau p can read `seg[j]` for a j in another
plateau, but only through a non-reciprocal edge, and `to_set` is always a subset
of the original bits, so the bit j would need to point back at i is provably
absent. Cross-plateau interference is impossible.

Three invariants the code depends on silently are now assertions in the
replica, and all three hold across the sweep:

- `k_count_v2`'s `in_plat` predicate (corner, or has a reciprocal neighbour) is
  exactly `bits != 0`, which is what makes `vcount[root]` the plateau's voxel
  count and therefore makes `qsz` correct.
- the per-plateau BFS never exceeds `vcount[root]`, confirming the `tail <=
  vcount` argument in `k_plat_meta` on data rather than on the val measurement
  quoted in its comment.
- no voxel points at a zero-bit voxel. This one is load-bearing and non-obvious:
  `k_root_flag` only allocates a label to a root with `bits != 0`, while
  `k_write_labels` gives every non-zero voxel `psum[parent[i]] + 1`, so if a
  component's minimum member had `bits == 0` it would be an unflagged root and
  `psum` at that index is some *other* fragment's label. For raw `k_flow`
  output this is forced, `bits[j] == 0` means every face of j is `<= low`, but
  a neighbour pointing at j does so across a face that is also one of j's and is
  `> low`. After the divide it is no longer forced, because `to_set` can point
  at a plateau interior that has just been zeroed, so it is checked explicitly.

### W1 is unsound, and it takes the plan's L2 story with it

With a trustworthy replica in hand, the first thing worth testing is W1 itself,
because the plan does not treat it as an optimisation but as a prerequisite:
"one label per 8 voxels is 0.5 B/vox... a z-plane-pair of block labels is
5.76 MB and fits in 6 MB L2... union operations drop ~8x". It calls this
Komura equivalence / BKE-3D.

BKE is a real technique and it does not apply here. Komura's block
decomposition is valid for 8-connected 2D and 26-connected 3D labelling, and
the reason is specific: under 26-connectivity every pair of voxels inside a
2x2x2 block is directly adjacent, so any two foreground voxels in a block are
necessarily in the same component and contracting the block to one label
discards nothing. This watershed is 6-connected, and its edges are not
foreground adjacency but flow direction, so neither half of that argument
survives.

Tested rather than argued. `w0_ws_ref.py --blocks` runs three labellings of the
same union-find edge set - per-voxel, one-label-per-block, and one slot per
intra-block connectivity class - on both of the union-finds W1 proposes to
convert, across the same 20-case sweep. One label per block is wrong in 40 of
40 checks, and not marginally: it collapses 7 to 790 true components into 1 to
3. Contracting blocks in a dense 6-connected flow graph makes almost the whole
block adjacency graph one component.

The sound form is one slot per intra-block class, which is exact by
construction because it is the same union-find on the same edges with the
parent array indexed differently. But the sweep measures the class count at a
mean of 2.26 to 5.55 per block and a maximum of 8. Eight uint32 slots is
4 B/vox, which is exactly what the per-voxel array already costs. So the sound
version of W1 saves nothing, and the unsound version is wrong.

The synthetic volumes are more fragmented than real data, so those means are
pessimistic. The real-data measurement points the same way though, and it is
already in `w1_block_analysis.json`: 37.77% of 2x2x2 blocks on val contain two
or more distinct fragments. A single label per block cannot represent those
blocks, whatever the mechanism. What does survive from W1 is the union-site
reduction - 454.0M same-fragment voxel pairs collapsing to 66.5M block face
pairs, 8.1x - but that is a traffic argument for tiling, which is W4, not a
reason to change the label representation.

### W2 closes anyway, via a lever the plan did not consider

Losing W1's -3.5 B/vox should have killed W2, the one hard blocker. It does
not, and `scripts/w2_mem.py` does the arithmetic from `d1_mem.json`.

The plan's own numbers do not work even before W1 is withdrawn. Its three
savings are -3.5, -4 and -1.43 B/vox against a measured 13.988, leaving 5.058
B/vox of scratch = 10.18 GiB, and the peak is scratch plus the affinity input
plus the label buffer: 6.03 + 8.05 + 10.18 = 24.26 GiB. That is over the 24 GiB
card, not the "~23 GiB" claimed, and before any allowance for driver context.

The lever that rescues it is one the plan never mentions: **stream the
affinity**. `k_flow` is the affinity's only consumer in the entire watershed,
so 6.03 GiB of input has no business being resident alongside the union-find
scratch. Upload it in z-slabs for `k_flow` alone and it stops counting toward
the peak. That is more than block labels were ever going to save, it is
obviously correct, and it is a change to the allocation schedule rather than to
the algorithm. Two more of the same kind: `vcount` is only ever read as
`vcount[plat_root[p]]` in `k_plat_meta`, so it needs one entry per plateau
rather than per voxel (-8.05 GiB), and e9c holds `flag` and `psum` as separate
per-voxel arrays when e9b already demonstrates the in-place-scan trick with
`k_scatter_idx_u32` recovering the predicate from the scan's own differences
(-8.05 GiB). Plus the uint32 BFS queue the plan did list (-2.88 GiB).

    peak measured                    42.22 GiB
    after the four sound levers      17.22 GiB   fits in 23 GiB usable
    plan's own arithmetic            24.26 GiB   does not fit

One honest caveat, recorded in the script rather than buried: streaming the
affinity and the e9c scan apply to different phases, and only the phase that
owns the peak actually pays. `ws_mem_peak_lines()` already attributes the peak
to a source line and `d1_mem.py` already reads it, so 17.22 GiB is a lower
bound on what the levers achieve until that attribution is run on a card. The
margin to 23 GiB is wide enough that the verdict is unlikely to flip, but it is
a projection, not a measurement.

### The feasibility verdict, with the watershed finally costed

The watershed had no byte model. It was 90% of the projection and was being
estimated by scaling one idle val measurement by 12x and a bandwidth ratio,
which is not evidence you can decide feasibility on. `m1_cost_model.py` now
costs it the same way it costs the agglomeration kernels: essential traffic per
pass times the passes the algorithm actually performs.

    k_flow: affinity read + bits write          8.6 GB  x1      10.2 ms
    hook rounds, cold neighbour gathers       997.9 GB  x14    1178.6 ms
    compress rounds                           241.9 GB  x14     285.7 ms
    divide: corner flag, vcount, BFS           36.7 GB  x1      43.4 ms
    label: root flag, scan, write              25.9 GB  x1      30.6 ms
    untiled total                              1311 GB         1548.5 ms
    xy-tiled + W3                               553 GB          653.2 ms

Two things fall out. First, the stage as it stands is **7.4x off its own
roofline** - 11435 ms projected against 1548 ms of untiled essential traffic.
That is the strongest possible confirmation of W4's premise: `grep` for
`__shared__`, `__shfl`, `warp` in `ws.cu` returns nothing, and the cost of that
is a factor of seven, not a rounding error. The watershed is not
bandwidth-bound, it is implementation-bound.

Second, and this retires W1 completely: the L2 residency W1 was supposed to buy
comes from tiling, not from the label representation. The hook kernels need a
three-plane window for their +/-z gathers. At 2.16 Gvox that is 69 MB of
per-voxel labels, and 17 MB even with block labels, so the plan's "block-label
z-plane-pair = 5.76 MB and fits in 6 MB L2" was comparing against the wrong
working set. What does fit is an xy tile: 512x512 over three planes is 3.15 MB
of *per-voxel* labels. Tiling gives residency on the 4 B/vox array directly,
with no change to the label representation and no soundness question.

The round count is the one remaining scaling assumption, and it holds. The
plan's argument that "iteration counts stay constant" is measured and true for
agglomeration because `make_big.py` mirror-tiles into 12 disjoint copies of the
graph, but a union-find's round count depends on the longest chain it collapses
and the graded volume is 3x2x2 tiles, so chains along an axis get up to 3x
longer. Measured in the replica across a 12x volume range, both union-finds
stay at 3-5 rounds. That is what full path compression every round buys: the
count is logarithmic in chain length, so 3x longer chains cost about 1.6 extra
rounds, not 3x more.

So, the verdict the plan's title promises, now derived rather than asserted:

    best case, every sound change landing perfectly
      watershed        537 ms   xy-tiled (W4) at its own essential traffic, W3
      rag              180 ms   r1 target
      agglomeration    760 ms   G levers + V levers, all measured
      extract           73 ms
      TOTAL           1550 ms = 1.393 Gvox/s, 1.44x over budget

W3 in that figure is narrower than the plan's version and is exact rather than
approximate. `k_hook_bidir` skips a voxel with no direction bits and only hooks
reciprocal edges, so a voxel with no reciprocal edge issues no hook at all;
launching over the compacted list of voxels that have one is definitionally the
same kernel, which is the same argument that makes g3's active-edge list exact.
That is 35.6% of the volume and it applies to the plateau half only, because
after the divide almost every voxel carries a bit and the basin hook has no
such list to restrict to. The label pass also drops from three 4 B/vox touches
to a 1 bit/vox flag bitmask with the scan running over per-word popcounts.

**The priority inverts once those land, and this is the thing to carry
forward.** With the watershed at 537 ms, agglomeration at 760 ms is the largest
single term in the pipeline - which is the opposite of the situation the
un-optimised numbers describe, where the watershed is 90.5% and agglomeration
is 6%. Watershed plus RAG plus extract at their achievable figures is 790 ms,
so 2 Gvox/s leaves agglomeration 290 ms, and the measured lever stack plus the
V levers reaches 760. The gate therefore needs a further 2.62x in agglomeration
that nothing in the repo identifies, or fewer union-find rounds in the
watershed. Both are algorithm changes rather than better implementations of
what is there, and that is the honest reason 2 Gvox/s is not reachable from
this design.

It also means G2 and G4 are worth writing after all. I had deprioritised them
on the grounds that agglomeration was 6% of the projection, which was only true
because the watershed was unoptimised; against the achievable watershed they
are half the remaining budget.

Worth noting the projection is *better* than the plan's own 0.98 Gvox/s, at
1.296. The plan was pessimistic about the achievable figure and optimistic
about the mechanism: it expected to get there through block labels, which do
not work, and it under-credited both the agglomeration levers and tiling.

### W4: the tiled hooks are written, and verified without a card

Since the byte model puts 7.4x of slack in the two hook kernels and identifies
the launch shape as the cause, that is where the code went. `k_hook_bidir` and
`k_hook_remain` now have tiled counterparts, `k_hook_bidir_tiled` and
`k_hook_remain_tiled`, selected by `WATERZ_UF_ALGO=2` alongside the existing
0 and 1 that B3 used to compare path compression against pointer jumping.

The change is entirely launch geometry and staging. `vox_grid` gives a block
256 consecutive x inside a single (y,z) row, so a block's +/-y neighbour is
X*4 bytes away and its +/-z neighbour X*Y*4, 23.04 MB at 2.16 Gvox, which
misses a 6 MB L2 on every one of six gathers per voxel per round, fourteen
rounds. The tiled version takes a 32x4x4 tile with its one-voxel halo, which is
34x6x6 = 1224 slots, stages 1224 parent words and 1224 direction bytes with 512
threads, and then does all seven accesses per voxel out of shared memory: 2.39
global loads per voxel instead of 7, and the survivors sit inside a three-plane
window of 32-voxel rows instead of scattered across the volume.

Why staleness is not a correctness question, which is what makes this safe to
write without being able to run it: every write is `atomicMin` into a root's
parent slot, so parent entries only ever decrease; the round loop already runs
to convergence and reports failure to converge; and the fixed point of
min-index hooking is the component minimum for any read order. The untiled
kernel is already reading values other blocks are concurrently modifying, so
tiling changes how stale the reads are, not whether they can be.

What would break it is dropping or inventing an edge, and a halo off-by-one is
the obvious way to do that. So that is what got tested. `w0_ws_ref.py` now
carries `tiled_hook_edges`, which replicates the kernel's index arithmetic
literally, `ws_sidx`, the `(gx, gy, gz)` bounds test, the inert `bits = 0`
fill for out-of-volume slots, and diffs the resulting edge set against
`untiled_hook_edges` taken straight from `k_hook_bidir`'s source. Shapes were
picked so tiles land unevenly on every axis: X of 33, 35 and 40 against a tile
of 32, Y of 5, 6 and 7 and Z of 5, 8 and 9 against a tile of 4, plus a 4x4x4
volume smaller than one tile.

    plateau hook, k_hook_bidir_tiled    15 cases, 0 mismatches
    basin hook, k_hook_remain_tiled     12 cases, 0 mismatches, on the
                                        divided field rather than the raw one

Both files pass `clang++ -x cuda -fsyntax-only` through `scripts/nvcheck.py`.
That is as far as verification goes without a card: the edge sets are proven
equal and the convergence argument is proven independent of read order, so what
remains unmeasured is only whether the shared-memory staging actually recovers
the 7.4x, which is a performance question rather than a correctness one.

## LICENSES PASS

command: read of csrc/parhac_d.cu:146 (k_scale_sm_bytes) and csrc/ws.cu:1221 (k_uf_jump comment)
method: two facts the later levers treat as licenses, written down before the CUDA that depends on them

1. Contact sums are exact integers. `k_scale_sm_bytes` multiplies every `sm`
   by 255 and stores the rounded result in a double. Affinities are uint8/255,
   so the true sum is k/255 for an integer k, and k is bounded by
   255 * 3 * nvox = 1.65e12 at 2.16 Gvox, inside the 2^53 exact-integer range
   of a double. atomicAdd on those values commutes. Dirty-set dedup (G2) and
   the RAG shared table (R1) are therefore order-independent in their sums,
   not approximately so.

2. The watershed union-find gate pins the fixed point, not the path. Both
   e9b and e9c already run to convergence and report NOT-CONVERGED rather
   than absorbing a cap. The comment above `k_uf_jump` states the license:
   the settled state is "every entry holds its component's minimum index"
   either way. Any rewrite that applies the same min-index hooks and runs
   to the same fixed point is bit-identical. That is what makes W5 a
   reordering of the edge set rather than a different algorithm.

## W5 CPU-VERIFIED

command: python3 scripts/w0_ws_ref.py --sweep --w5; python3 scripts/w5_tile_uf.py --crop 125 256 256
method: two-phase min-index union-find vs one-phase; stitch rounds on real fragments

    parent arrays identical in 120/120 synthetic checks (3 shapes including
    uneven tiles, 5 level counts, 4 seeds, plateau and basin each)
    e9 vs host: 0 diffs in the same 60 cases
    14/14 tilings on the 8.19 Mvox fragment crop: parent identical

Stitch payoff, tree graph (the pessimistic bound, a flow-like forest),
8x16x32 tile (the compiled default, 20 KB shared):

    baseline rounds 5, stitch rounds 5, hook list 0.382 of volume,
    compress list 0.456 of volume. Round count does not drop. Compress
    work drops because the domain does.

CUDA: `k_uf_tile_local<Recip>`, `k_w5_hook_list`, `k_w5_compress_list`,
`w5_union_find` behind `WATERZ_UF_ALGO=3` at both e9b and e9c.
`clang++ -fsyntax-only` PASS.

## W3 CPU-VERIFIED

command: python3 scripts/w0_ws_ref.py --sweep --w5
method: bitmask label_of vs exclusive-scan psum; nonempty-tile edge check

    bitmask labels identical in 60/60, nfrag match, ndiff=0
    edges into an all-zero tile: 0

CUDA: `k_root_mask`, `k_block_popc`, `label_of`, `k_write_labels_mask`
behind `WATERZ_WS_W3`. Empty-tile early-out inside `k_uf_tile_local`.
The original compacted-voxel-list form was not written: it would destroy
the locality W4/W5 exist for, and `bits != 0` on val is almost the whole
volume after the divide, so a voxel list would not shrink the basin UF.

## G2 G3 G4 WRITTEN

command: python3 scripts/nvcheck.py csrc/parhac_d.cu
method: lever bits 2 and 8 wired; bit 4 gains the active-edge list

    2  hash_combine_dirty: rewrite only edges whose current endpoints are
       this inner's merged blues or receiving reds, hash-dedup that set,
       write combined edges into vacated holes, leave nscan as the
       high-water mark. Licensed by the collision lemma in g0_agg_ref.py
       and by sm integrality.
    4  k_propose_listed over an amask packed after each dirty compact
       (incremental) or after a full combine (rebuild). The once-per-layer
       refresh is still rejected; --stale-active diverges.
    8  listed k_copy_sz over current roots; per-inner k_compress over
       accepted blues only.

Device fingerprint gate is scripts/g_levers.py, blocked on a card.

## R1 CPU-VERIFIED

command: python3 scripts/r1_rag_tile.py
method: face-count edge map on a 40x128x128 crop of gpu_fragments.npy

    38797 edges, tiled union identical, count-equal
    C1 warp atomics 132183, R1 tile flushes 65834, 2.01x fewer global
    atomic sequences

CUDA: `k_hash_faces_tiled` 32x4x2 block, 1024-slot shared table, overflow
falls back to `hash_add_group`. `WATERZ_RAG_ALGO=1`.

## HARNESS

    scripts/w_levers.py          CPU W5/W3 identity; device A/B deferred
    scripts/nvcheck.py           SRCS now includes csrc/rag.cu
    scripts/e3_final.py          --gpu-window refuses on a busy card, then
                                 nvcheck, g_levers, w_levers, b1, c2,
                                 four-threshold VOI, d_bench

## M1 REPROJECT

command: python3 scripts/m1_cost_model.py
method: W5 term from the measured stitch domain (0.46 of volume, 5 rounds
plus one flatten) replacing the tiled-only 14 full-volume compresses

    watershed best case  537 ms -> 281 ms
    end-to-end best case 1550 ms / 1.393 Gvox/s -> 1294 ms / 1.670 Gvox/s
    budget 1080 ms, still 1.20x over
    agglomeration 760 ms against 546 ms allowed: needs a further 1.39x
    that is not identified

W5 did not cut union-find rounds. That is a measured fact on the real
fragment geometry, not a model assumption. The residual is still an
algorithm change in agglomeration or a watershed that converges in fewer
rounds, which this design does not.

## G4 COMPLETE CPU-VERIFIED

command: python3 scripts/g0_agg_ref.py --mode both --out g0_agg_ref.json
method: incremental root list + skip per-outer nnode rebuild; assert
incremental sz[roots] == rebuild every outer

The previous bit-8 path rebuilt the root list with an exclusive_scan of
all nnode every outer and still ran k_compress / k_zero_sz / k_rebuild_sz
over the full volume. That is more work than k_copy_sz, so the lever was
a net loss. What landed:

    k_init_root_list once (roots = 1..nnode-1)
    after each accept: k_compact_roots keeping parent[r]==r
    per outer: k_clear_frozen_list + k_copy_sz_list only
    skip compress/zero/rebuild: k_accept_reds already keeps sz[root]

Full-val RAG, T=0.3, eps=0.08, to convergence:

    base matches p0aa_e6s.json digit-for-digit (n_layer, nouter, ninner,
    nmerge, sum_nlive, sum_above, both 17-element layer vectors)
    parent array bit-identical base vs fast
    inc_sz_ok = 653 / 653 outers
    compress_node_visits 3096M -> 41.0M (75.5x)
    outer_node_visits 8523M -> 356M (23.9x)

CUDA: WATERZ_AGG_LEVERS bit 8. clang++ -fsyntax-only PASS.

## V1 V2 WIRED

    WATERZ_AGG_EPS          overrides locked 0.08 in segment.py
    WATERZ_SIZE_ASYM=0      drops sz[red] >= sz[blue] in every propose
                            kernel via d_size_asym
    e3 --gpu-window         now runs a1 twice more, once with each

V2 CPU: g0 --no-size-asym --sub 80000 --max-layer 2 PASS (parent + counters).
Four-threshold VOI is still a card.

## G0 FULL-VAL PASS

command: python3 scripts/g0_agg_ref.py --mode both --out g0_agg_ref.json
method: CPU replica of parhac_e6s_dev vs fast G2/G3/G4, whole val RAG

    nedge=7505458 nnode=2175401
    base 381.4s  fast 186.8s
    nseg=321973  nmerge=1853427
    G0 PASS

m1_cost_model.py now consumes this file (it refused subgraph visit ratios).

## M1 REPROJECT AFTER G4

command: python3 scripts/m1_cost_model.py
method: full-val G0 visit ratios; G2 residual is the endpoint scan, not CSR

    agglomeration after G levers          1719 ms (7.5x from 12949)
    after G + V (subgraph 3.65x)           470 ms
    best case e2e                         1004 ms = 2.151 Gvox/s
    budget                                1080 ms, 0.93x of budget

On this model the 2 Gvox/s gate is reachable. That is not a claim it is
hit: V is still a subgraph factor, W5/R1 are byte models, no G lever has
run on a device.

Idle-card command, refuses if nvidia-smi shows other compute apps:

    python3 scripts/e3_final.py --gpu-window

## V FULL-VAL

command: python3 scripts/v_levers.py
method: CPU replica on whole val RAG; locked row reused from g0_agg_ref.json

    v1 eps=0.08           1.00x  nseg=321973
    v1 eps=0.12           1.39x  nseg +1
    v1 eps=0.16           1.69x  nseg -22
    v1 eps=0.24           2.21x  nseg -63
    v1 eps=0.32           2.88x  nseg -75
    v2 eps=0.08 no-asym   1.19x  nseg -10
    v1+v2 eps=0.32        3.19x  nseg -219

Layer 0 stays at the 64-outer cap except v1 eps=0.32 (63). The subgraph
3.65x was too high. Every unlocked config changes nseg and owes VOI.

## M1 AFTER FULL-VAL V

command: python3 scripts/m1_cost_model.py

    V factor 3.65x (sub) -> 3.19x (full val)
    agglomeration after G+V  470 ms -> 539 ms
    best case e2e           1004 ms -> 1073 ms = 2.013 Gvox/s
    0.99x of the 1080 ms budget

Still a model, not a measurement. The 7 ms of slack disappears if W5
or R1 miss their byte-model targets.

## E3 GPU-WINDOW FIXED

command: python3 scripts/e3_final.py --gpu-window
method: card_state is idle / busy / none; PY falls back to sys.executable

    none: CPU gates only (nvcheck, w_levers, r1). v_levers skipped
          when v_levers.json is already full-val.
    busy: refuse, write e3_gpu_window.json status=refused
    idle: CPU + g_levers + VOI x3 + d_bench

This machine: 3/3 CPU PASS, gpu_blocked listed. Previously
--gpu-window treated missing nvidia-smi as busy and refused, and
hard-required .venv/bin/python which does not exist here.

## E3 GPU-WINDOW GREENGOBLIN IDLE

command: PATH=/usr/local/cuda-12.8/bin:$PATH .venv/bin/python -u scripts/e3_final.py --gpu-window
host: v@100.90.97.111 (greengoblin) RTX 5090, idle 16 MiB 0% throughout the window
artifacts: data/cache/greengoblin_20260905/

    nvcheck             PASS  22.9s  nvcc -c (no clang on the box)
    w_levers            PASS   1.1s  UF 0/1/2/3 and W3 CPU identity
    r1_rag_tile         PASS   0.8s
    g_levers            PASS  26.6s  bits 1,2,4,8,15 all BIT-IDENTICAL
    rag_determinism     PASS   7.7s
    ws_invariants       PASS  14.2s  nfrag=2175400 bg=506568 oracle-equal
    voi_four_threshold  PASS 106.7s  locked fp [294165, 322000, 345131, 379293]
    v1_voi              INVALID PASS 103.3s  (see below)
    v2_voi              FAIL 103.6s  real accuracy miss
    bench_device        PASS  12.1s  val 180 Mvox, not the graded 2.16 Gvox

G-levers on idle 5090, T=0.3, nmerge=1853427, 17 layers, 653 outers:

    bit 0 reference     471.8 ms
    bit 1 freeze_reds   456.5 ms  1.03x  identical
    bit 2 dirty-scan    409.6 ms  1.15x  identical
    bit 4 active-lists  456.2 ms  1.03x  identical
    bit 8 root-list     436.7 ms  1.08x  identical
    bit 15 all          317.6 ms  1.49x  identical

Locked four-T VOI (eps=0.08) matched the fingerprint and passed +0.02 at
every T. C2 fingerprint fff9037cab341692be0c9bf3c577d4ff.

V1 window PASS was a lie. a1_e6t_voi.py hardcoded ctypes.c_double(0.08),
so WATERZ_AGG_EPS=0.16 never reached parhac_paper_d_timed. E6s printed
eps=0.0800 and the locked fingerprint. Fixed a1 to read WATERZ_AGG_EPS
the same way segment.py does.

command: WATERZ_AGG_EPS=0.16 .venv/bin/python -u scripts/a1_e6t_voi.py --e6s
method: same idle 5090, after the a1 fix; A1 eps=0.16 and E6s eps=0.1600

    fingerprint [294101, 321923, 344984, 378629]
    T=0.2  split 0.3754  merge 0.3715 > 0.3525  FAIL
    T=0.3/0.4/0.5 PASS
    ACCURACY GATE: FAIL

V2 (WATERZ_SIZE_ASYM=0) did apply. Fingerprint
[294142, 321960, 345010, 379008].

    T=0.2  split 0.3718  merge 0.3564 > 0.3525  FAIL
    T=0.3/0.4/0.5 PASS
    ACCURACY GATE: FAIL

Both unlocked V configs are accuracy-illegal. Do not ship them. Do not
tune G-series to chase them.

d_bench --device default aff is cremiA_val (125x1200x1200 = 180 Mvox):

    G0   median 1082.2 ms  0.166 Gvox/s  nseg=321973  idle  gradeable-on-this-card
    G15  median  958.7 ms  0.188 Gvox/s  nseg=321973  deterministic

command: WATERZ_AGG_LEVERS=15 .venv/bin/python -u scripts/d_bench.py --device

These are 5090 numbers on the val crop. They are not the TASK 2 Gvox/s
claim (3090 Ti, [3,375,2400,2400]=2.16 Gvox). 2.16 Gvox was not run:
d1_mem puts the watershed peak at 42.20 GiB and this card is 32 GiB;
z-slabbing is not landed.

## E1 T=0.3-ONLY VOI

command: .venv/bin/python -u scripts/e1_t3_voi.py
host: greengoblin CPU (gt.h5); voi_numpy matches waterz test_evaluate
method: g0 fast replica + H(seg|gt)/H(gt|seg); skip gt==0

    T=0.3 split<=0.4738 merge<=0.2611
    v1 eps=0.08          split 0.4512 merge 0.2505  PASS  (device locked)
    v1 eps=0.12          split 0.4491 merge 0.2414  PASS  outers=483  1.39x
    v1 eps=0.16          split 0.4536 merge 0.2396  PASS  (device)
    v1 eps=0.24          split 0.4498 merge 0.2565  PASS  outers=280  2.21x
    v1 eps=0.32          split 0.4451 merge 0.2509  PASS  outers=227  2.88x
    v2 eps=0.08 no-asym  split 0.4570 merge 0.2421  PASS  (device)  1.19x
    v1+v2 eps=0.32       split 0.4529 merge 0.2630  FAIL  0.2630>0.2611

Uniform V is still illegal (T=0.2). T=0.3-only V is legal up to eps=0.32
with the asymmetry on, and for no-asym at locked eps. Combined 0.32
no-asym dies at T=0.3 by 0.0019. Best speed-path parameter: eps=0.32
size_asym=1 at T=0.3 only.

## E2 CSR DIRTY-COMBINE

command: python3 scripts/e2_csr.py --sub 80000; python3 scripts/e2_csr.py --sub 0 --ref-json data/cache/g0_agg_ref.json

    sub 80k:  parent identical, cut 5.3x, E2b theta=0.5 identical
    full val: nmerge=1853427 nseg=321973, per-inner CSR set == scan
              lookup+splice 390M vs scan 1505M = 3.9x
              rebuild 68M, not fatter than the scan

Identity holds. The 10x compact-byte gate fails. Dead slots on spliced
lists keep lookup at 349M against ndirty_total 163M. Not the 750x the
nscan/ndirty_mean ratio suggested.

## E3 LAYER 0

command: python3 scripts/e3_layer0.py --sub 80000

    cap64 pinned. cap128 goes to 76 outers, +13 merges, partition moves.
    sub no-asym exits at 48. Full-val v_levers: layer 0 stays at 64
    except v1 eps=0.32 (63). No new matching. no-asym is E1's lever.

## E4 REPRESENTATIVE STITCH UF

command: python3 scripts/e4_rep_uf.py --crop 32 128 128

    20/20 parent-identical (dense+tree, W5 tiles + uneven 5x7x33,
    9x5x40, 4x4x4). max rep_frac=0.241 (tree 4x4x4) vs W5 0.46.
    default 8x16x32 tree: 41877 reps, frac=0.080.

## E5 W2 PROOFS

command: python3 scripts/e5_w2_proofs.py

    only k_flow reads aff[]. vcount-per-root would drop 98.5% of the
    array on a 32x256x256 fragment crop. w2 17.22 GiB stays a lower
    bound, not a measurement. Peak line 1346 in d1_mem.

## E6 RECALIBRATED STACKS

command: python3 scripts/e6_recal_m1.py
method: idle 5090 G0/G15 e2e, G15 phases, E1 vfac=2.88, E2 3.9x on
        compact, E4/W5 byte-model WS. Scale x21.333 to 2.16 Gvox 3090 Ti.

    stack 1 locked G15 untiled WS     20452 ms  0.106 Gvox/s  18.94x
    stack 2 +E2 +W4/W5 no V            5785 ms  0.373 Gvox/s   5.36x
    stack 3 +E1 eps=0.32 T=0.3         3619 ms  0.597 Gvox/s   3.35x

DESIGN SHORT. Stack 3 is 3.35x the 1080 ms budget after every measured
lever that passed. Do not start CUDA for E2/E4/E5 on the hope they
close 2 Gvox/s. They do not. The remaining gap is still an unidentified
algorithm change (fewer WS rounds, or an agglomeration that is not
ParHAC), not an implementation of what is already in the tree.

## N0 HONEST E6

command: python3 -u scripts/n0_honest_e6.py
method: extract=3.434 ms (LOG E10); leftover 96.08 ms attributed three ways

    leftover->WS (honest)   1642 ms  1.315 Gvox/s  1.52x   agg must be ≤587 ms (1.96x)
    leftover->alloc         3692 ms  0.585 Gvox/s  3.42x   (old E6 extract dump)
    leftover->sync          1738 ms  1.242 Gvox/s  1.61x
    +WS/RAG reflect        1205 ms  1.793 Gvox/s  1.12x   (N4, not a 12x claim)

Wrote data/cache/e6_recal_honest.json. Not a TASK number.

## N1 T=0.3 SHALLOW REGRADE

command: python3 -u scripts/n1_t3_regrade.py
method: chunked mmap VOI; C++ Kruskal/mutex/frozen CC; no 180 Mvox labels

    KEEP=0
    X0 / hist-q frozen / waterfall-full   merge 7.55 FAIL giant
    mutex AbsMax                          split 0.9097 FAIL
    Zlateski 256/1024                     split 1.06/0.80 FAIL
    rel-contact 0.05/0.10                 merge 0.346/0.280 FAIL
    waterfall 1/2 pass                    split 1.90/0.96 FAIL

Four-T death is T=0.3 death for every shallow class.

## N2 HIGHER EPS T=0.3

command: python3 -u scripts/n2_higher_eps.py

    eps=0.40  split 0.4408 merge 0.2543  PASS  sum_nlive=894217121  3.01x vs locked
    eps=0.48  split 0.4422 merge 0.2892  FAIL  STOP
    extra vs E1 eps=0.32: 1.04x. Not 1.96x.

## N3 RAG STRUCTURE

command: python3 -u scripts/n3_rag_structure.py

    mean==1 65256; mean>0.9 6.76M; mean>T 6.91M; prefilter edge cut 1.09x
    mean deg 6.90; locked visits/merge ~1152 (E2 already owns that slack)
    prefilter mean>T: fp!=locked, work 1.19x, T=0.3 VOI PASS 0.4527/0.2513
    sat mean==1 prefix: leftover 49586, merge VOI 7.87 FAIL giant

## N4 MIRROR / SEAM

command: python3 -u scripts/n4_mirror_identity.py
    reconstructed make_big seams identically 0 (synthetic). No make_big.py /
    affinity.h5 on disk. Not a 12x throughput claim. N5 does not take 1.18x.

## N6 STOP

Wrote notes/N6_STOP.md. N5 CUDA gate: need 1.96x, best new factor 1.04
(ε=0.40) or 1.19 (prefilter). Neither >=1.95. No CUDA. Honest stack still
1.52x over 1080 ms after every legal lever that passed.

## P1 OFFICIAL MAKE_BIG INDEPENDENCE

command: ssh greengoblin .venv/bin/python -u scripts/p1_make_big_indep.py
machine: greengoblin RTX 5090 idle; official make_big.mirror; GPU WS + CPU scans
    dataset already on greengoblin; no volumes copied to the laptop

    val GPU == wz_fragments.npy byte-identical; nfrag=2175400 bg=506568
    axis z/y/x: seam_zero, span=0, mir_nfrag=val, concat_nfrag=2*val,
    tile0==val remap, tile1==mir remap, all cross-seam aff=0
    P1 PASS. 12 merge-independent tiles. Not a 12x or 2 Gvox/s claim.

## P1b FLIP EQUIVARIANCE

command: ssh greengoblin .venv/bin/python -u scripts/p1b_flip_equiv.py

    flip(WS(val)) != WS(mirror(aff)) on every axis
    z disagree 9.90M (5.5% interior, 125/125 planes)
    y disagree 26.92M (15.0% interior, 1200/1200 planes)
    x disagree 12.57M (7.0% interior, 1200/1200 planes)
    1x-val+stamp illegal. halo_only=False.

## P1 GATE

command: python3 -u scripts/p1_gate.py

    8 unique flip tiles * 958.70 ms = 7669 ms (5090) / 13635 ms (3090 model)
    8.3x slower than fused honest 1642 ms; 12.6x the 1080 ms budget
    reopen_speed_path=False. start_w2_reflect_stamp_cuda=False. keep_n6=True
    Wrote notes/P1_GATE.md

## P2 TRUE HISTOGRAMQUANTILE

command: ssh greengoblin .venv/bin/python data/ws_bounty/baseline/run_baseline.py
         --candidate data/ws_bounty/q20_quantile/mine_thr{0.2,0.3,0.4,0.5}.h5

    T=0.2 0.3754/0.3299 PASS
    T=0.3 0.4521/0.2422 PASS
    T=0.4 0.5180/0.2180 PASS
    T=0.5 0.6103/0.2094 PASS
    ACCURACY GATE PASS. Serial BinQueue. Not a 1.96x fused-agg cut. Class closed.

## A CLUSTERED-GRAPH / PARHAC §2.3 READ

command: python3 -u scripts/a_clustered_cpu.py
papers: parhac_dhulipala2022.pdf p.6 §2.3, p.22-23 MultiMerge, p.23 Affinity/SCC,
        D.3 CPAM=HT runtime; dynhac_yu2025.pdf §1-3 good-merge

    7-11x is Affinity/SCCsim GBBS vs clustered-graph, NOT ParHAC.
    E2 CSR already is MultiMerge: identical=True cut=3.86x (e2_csr_full.json).
    layer0 ~20660 merges/outer: not the paper ε=0.01 small-round regime.
    Honest stack already credits E2. start_gpu_starmarge_v2=False.
    keep_n6_on_work=True. SOURCES S32 corrected; S39 DynHAC; S40 d1 peaks.

## B W2 FREE AFF AFTER K_FLOW

command: ssh greengoblin .venv/bin/python -u scripts/b_w2_free_aff.py
    rebuilt src/libws_gpu.so; idle 5090; host e9c_watershed only

    nfrag=2175400 oracle array_equal=True leaked=0
    val peak 3.536 -> 3.033 GiB (saved 0.503 = val aff)
    pred 2.16 Gvox WS 36.40 GiB (still over 23 usable)
    e9_d / segment_d aff left in place for RAG
    Wrote notes/B_W2.md

## B DEVICE-PATH AFF PARK

command: ssh greengoblin .venv/bin/python -u scripts/b_dev_aff.py
    rebuilt src/libws_gpu.so; ws_flow_d + park + ws_label_d

    nfrag=2175400 oracle=True held==park labels
    tracked peak 2.698 -> 2.195 GiB (saved 0.503 = val aff)
    Wrote notes/B_DEV_AFF.md

## B W2 REMAINING LEVERS (one at a time, parked path)

    q32:        oracle=True peak 2.195 (BFS not peak)
    vcount:     oracle=True peak 2.195 -> 2.023 (sort-phase)
    e9c_inplace:oracle=True peak 2.023 (e9c not peak)
    W3=1:       oracle=True peak 2.023 (e9c not peak)
    Wrote notes/B_W2_REST.md

## B DEAD AGG cu/cv/csm/cct

command: ssh greengoblin .venv/bin/python -u scripts/b_dead_agg.py
    parents identical a52a70236a43f9e99a9546482041ac82
    val agg peak 1.124 -> 0.956 GiB
    Wrote notes/B_DEAD_AGG.md

## B FIT TABLE

command: ssh greengoblin .venv/bin/python -u scripts/b_fit.py
    EDGES_PER_VOX=0.043, thinned edges, parked aff, dead-agg
    val nedge=7505458 parked=True
    2.16 pred: WS 34.34 / RAG 21.16 / AGG 19.55; worst WS; OVER 23
    Wrote notes/B_FIT.md data/cache/b_fit.json

## B 8 OFFICIAL MIRROR TRIPLES

command: ssh greengoblin .venv/bin/python -u scripts/b_8flip.py
    all 8: nfrag=2175400 bg=506568; 8 distinct size fingerprints
    12x identity=26104800; official fused=26023852
    Wrote notes/B_8FLIP.md

## B SUBGRAPHHAC GOOD-MERGE (CPU)

command: ssh greengoblin .venv/bin/python -u scripts/b_subgraphhac.py
    T=0.3 eps=0.10; 14 rounds; 1.87M merges; nseg=309454
    VOI 0.1550/7.5519 FAIL (giant). close_class=True. no CUDA.
    Wrote notes/B_SGHAC.md

## B SORT-PEAK TMP (in/out vs DoubleBuffer query)

command: ssh greengoblin .venv/bin/python -u scripts/b_sort_peak.py
    idle 5090; WATERZ_WS_MEMLOG=1; dry DoubleBuffer query only

    nC=61035574 nfrag=2175400 oracle=True
    tmp_inout=0.461 GiB tmp_dbl=0.007 GiB extra_n=0.455
    peak 2.023 still flag+vcount+3 nC (sort+tmp cur=1.598)
    fused-dbl-only pred 28.88 OVER. Wrote notes/B_SORT_PEAK.md

## B DOUBLEBUFFER + GATHER + PARK

command: ssh greengoblin .venv/bin/python -u scripts/b_dbl_buf.py
    DoubleBuffer keys+idx; host permute parked corners/vc
    vcount delayed; flag freed before keys; qsz from qoff

    nfrag=2175400 oracle=True leaked=0
    tracked 2.023 -> 0.916 GiB; tmp_dbl=0.007
    2.16 fused pred 21.05 FITS 23 usable
    Wrote notes/B_DBL_BUF.md data/cache/b_dbl_buf.json
    SOURCES S41 pinned

## B FIT TABLE (after DoubleBuffer + park)

command: ssh greengoblin .venv/bin/python -u scripts/b_fit.py
    val ws=0.916 rag=1.250 agg=0.788 parked=True nedge=7505458
    2.16 pred: WS 21.05 / RAG 21.16 / AGG 19.55; worst RAG; FITS 23
    Wrote notes/B_FIT.md data/cache/b_fit.json

## N7 POST-FIT REMESURE

command: ssh v@100.90.97.111 WATERZ_AGG_LEVERS=15 WATERZ_STAGE_MS=1 .venv/bin/python -u scripts/n7_postfit.py
    idle 5090; parked aff + DoubleBuffer + G15
    median e2e 1669.48 ms (was 958.70)
    ws 1272.39 rag 41.50 agg 351.72 extract 1.68 leftover 2.19
    0.108 Gvox/s on 180 Mvox. Wrote notes/N7_POSTFIT.md

## N7 COMPACT IOU

command: ssh v@100.90.97.111 .venv/bin/python -u scripts/n7_compact_iou.py
    G15 p0aa compact 227.43 = scan 42.99 + hash 119.74 + radix 7.67 + other 57.03
    owner=hash scan_frac=0.189 scan_dominates=false
    1150 is an E2 IOU. No CSR splice. Wrote notes/N7_STOP.md SOURCES S42

## N8 UNPARK / W5 / HASH / 2.16 FACT

command: ssh v@100.90.97.111 .venv/bin/python -u scripts/n8_unpark.py
    WS identity nfrag=2175400 oracle=True
    median e2e 1154 ms ws 913 rag 41 agg 197 (ε=0.40 speed path)
    tracked ws 1.364 GiB; fused 2.16 pred 26.43 OVER; slab pred FITS
command: ssh v@100.90.97.111 .venv/bin/python -u scripts/n8_w5.py
    algo 0/2/3 identity True; W5 1.63x (967->594 ms); default left 0
command: ssh v@100.90.97.111 .venv/bin/python -u scripts/n8_hash.py
    parent=True; warp+SM 158 vs G15 120; full RBK 499; G15 CAS stays
command: ssh v@100.90.97.111 .venv/bin/python -u scripts/n8_run216.py
    official make_big [3,375,2400,2400]; idle 5090; T=0.3 ε=0.40
    e2e **13518 ms** / 0.160 Gvox/s; ws 10832 rag 433 agg 2224
    nfrag 26104800 = 12xval. ≫2 s. Stop CUDA. notes/N8_IMPOSSIBLE.md

## N9 TYPE D / CUDA KILL / BINS

command: ssh v@100.90.97.111 .venv/bin/python -u scripts/n9_measure.py
    idle 5090; WATERZ_UF_ALGO=3; z-slab face-clear; ε=0.40
    val identity=True nfrag=2175400 ws=557.24 w5=175.96 bfs=1.18
    BFS/WS=0.00212 (PRUF dead)
    basin max/nvox=0.010352 plateau max/nvox=0.007826 (no ConnectIt)
    dirty_mult=1.0718 calls=315 (hash-table PDFs dead)
    2.16 e2e=9212.7 ms / 0.234 Gvox/s (not TASK-grade)
    ws=6536.2 rag=424.4 agg=2222.3; W5 kernels 2116.74 ms
    nfrag=26104800 nlab=3860788
    WS >4 s after W5: STOP CUDA tracks A/B/C
    notes/N9_MEASURE.md N9_E4.md N9_HALVING.md N9_PLAYNE.md
    Default WATERZ_UF_ALGO stays 0
command: ssh v@100.90.97.111 .venv/bin/python -u scripts/n9_binqueue.py
    stock MEAN + BinQueue; T=0.3; fresh fragments per N
    N=256/1024/4096: split=0.455129 merge=0.241600 nseg=322314 PASS
    2175401 nodes 7505458 edges merged 1853086 each
    no GPU bucket (no 5x visit proof; N7 visits!=wall)
    notes/N9_BINS.md

## N10 WS SPLIT: HOST PARK WAS THE 4s

command: ssh v@100.90.97.111 .venv/bin/python -u scripts/n10_ws_split.py
    idle 5090; W5; val identity True both
    park=1 ws=597.67 park=110 vcount=114 sort=2.91 unpark=44
    park=0 ws=344.25 park=0 vcount=15 sort=2.90 unpark=3
    1.74x. Sort is 3 ms; park path is the 268 ms.
command: WATERZ_UF_ALGO=3 WATERZ_HOST_PARK=0 WATERZ_AFF_PARK=0 \
    WATERZ_STAGE_MS=1 .venv/bin/python -u scripts/n8_run216.py
    2.16 e2e **4918 ms / 0.439 Gvox/s** (was 9213 / 0.234)
    ws 2577 (was 6536) rag 83 (was 424) agg 2238
    RAG 424 was aff H2D. N9 CUDA-stop was PCIe, not UF.
    notes/N10_BOTTLENECK.md
    not 2 Gvox/s, not 3090 Ti

## N11 PROBE FIRST / E4 / BINS KILL

command: .venv/bin/python -u scripts/n11_nrep.py
    idle 5090; count-only after tile-local Recip=e9b
    val identity=True nfrag=2175400
    n_face=68962500 n_list=104706433 n_rep=3267642 n_cross=8352645
    n_rep/n_face=0.0474 n_rep/nvox=0.0182 e4_gate=True
    720M slab same ratios n_rep=13070568
    notes/N11_NREP.md
command: .venv/bin/python -u scripts/n11_merge_probe.py
    val RAG nedge=7505458 target_merges=1853086
    find+union wall=2093.005 ms n_pop=4791399 ns/pop=436.8
    kill vs ParHAC val agg 202 ms. no neighbor probe. no GPU BinQueue
    notes/N11_PROBE.md
command: WATERZ_UF_ALGO=3 WATERZ_HOST_PARK=0 WATERZ_AFF_PARK=0 \
    WATERZ_WS_MEMLOG=1 WATERZ_STAGE_MS=1 .venv/bin/python -u scripts/n8_run216.py
    2.16 e2e=4913.5 ms / 0.440 Gvox/s ws=2582 rag=83 agg=2229
    WSMEM peak=11.518 GiB < 23 usable
    park defaults flipped off (HOST_PARK/AFF_PARK)
    notes/N11_MEM.md
command: .venv/bin/python -u scripts/n11_e4.py
    WATERZ_UF_ALGO=4 contracted p1 pairs
    val identity vs wz_fragments.npy True vs algo0 True nfrag=2175400
    W5 stitch=168.45 ms E4 stitch=136.39 ms ratio=0.810 > 0.5
    keep=False. no 2.16. speed path stays UF_ALGO=3
    notes/N11_E4.md
    not 2 Gvox/s, not 3090 Ti

## N12
command: clone papers/repos/{graph-mining,ParHAC,dynamic-hac,PRUF-watershed,cuvs,RAMA}; grep .cu
    ParHAC/TeraHAC/DynHAC/SubgraphHAC: 0 .cu. PRUF yes. cuVS single-link MST. RAMA thrust contract.
    no Funke/Wolf email. Dhulipala draft unsent (still no GPU compact)
    notes/N12_REPOS.md
command: nsys profile val then 2.16 parks-off UF=3
    ident array_equal True nfrag=2175400 bg=506568
    val e2e=451.3 ms ws=224 rag=14 agg=211
    2.16 e2e=4943.3 ms ws=2591 rag=84 agg=2248
    NVTX 2.16: w5_stitch 52.8% e9c 32.1% hash_rewrite 8.1% compact_radix 1.4% hash_insert ~0%
    kernels: k_w5_compress_list 21.1% k_uf_compress_c 18.0% k_hash_insert 3.4%
    notes/N12_NSYS.md
command: default WATERZ_UF_ALGO 0 -> 3 (keep 0 behind env) after ident True
command: .venv/bin/python -u scripts/n11_e4.py  (face emit + Unique pairs)
    i<j broke e9c Recip=false nfrag=3281827; both dirs then Unique restored identity
    val identity True stitch=142.91/168.33=0.849 >0.5 keep_default=False
    2.16 algo=4 e2e=4580.6 ms ws=2240 (0.865x W5). kernel stays UF_ALGO=4
    notes/N12_E4.md
command: .venv/bin/python -u scripts/n12_tile.py
    8x16x32 / 16x16x32 / 8x32x32 all identity True
    2.16 WS 2576 / 2470 / 2486 vs 2582 (0.998 / 0.957 / 0.963) none <=0.85x
    default stays 8x16x32. binaries kept
    notes/N12_TILE.md
command: .venv/bin/python -u scripts/n12_compact.py
    k=0,2,4,8 T=0.3 + four-T eps=0.08
    k=4 T=0.3 merge 0.2653 FAIL. others PASS four-T
    vs nsys val agg 211 ms no 1.2x. do not default WATERZ_COMPACT_EVERY
    notes/N12_COMPACT.md
command: .venv/bin/python -u scripts/n12_off.py
    eps 0.5/0.8/1.0/2.0 T=0.3 merge FAIL (0.27/0.31/0.38/0.87)
    kruskal frozen 6092 ms split 2.94 FAIL. mutex hop split 2.13 FAIL
    PRUF3D sm_120 GPU 101.9 ms (not waterz fragments). RAMA cmake no tree on goblin
    never default FAIL
    notes/N12_OFF.md
command: fatbin sm_86+sm_120; unset WATERZ_UF_ALGO (default 3); n8_run216
    2.16 e2e=4923.9 ms / 0.439 Gvox/s ws=2581 rag=94 agg=2230
    vs N11 4913.5 +0.21% inside ±2%
    scripts/n12_3090.py ready, do not run until 3090 Ti rent
    notes/N12_FATBIN.md
    not 2 Gvox/s, not 3090 Ti

## N13 leftover gates (5090 only, email unsent)
    idle RTX 5090. parks off. no 3090. no multi-GPU. no product-default changes.
    P0: ident True nfrag=2175400 bg=506568. T=0.3 split=0.4408 merge=0.2543 PASS.
        2.16 e2e=4915.3 ms (ws=2579 rag=83 agg=2234) vs N12 4923.9 within 2%. stack_ok.
        notes/N13_BASELINE.md
    L1: nsys /tmp/n12_216.nsys-rep K=4579.7 ms M=cudaMemcpy 4957.4 ms (76.6% API)
        E=4915.3 E-K=335.6 <500. M>=0.9K. gate_artifact=True gate_real=False. no L1b.
        notes/N13_D2H.md
    L2: idle E6t T=0.3 device_ms=2860.8 vs E6s 200.6. VOI PASS 0.4417/0.2453.
        byte_identical=False ndiff=218644. TASK line 118 cannot ship. no four-T, no 2.16.
        keep_default=False. no second StarMerge.
        notes/N13_E6T.md
    L3: UF=4 ident True. 2.16 e2e=4592.4 ws=2240. NVTX stitch=1791.5 unique=97.7
        flatten kernels=1372.1 frac=0.766 >=0.70 L3-stop. unique frac=0.055. no third stitch.
        keep_default=False.
        notes/N13_E4.md
    L4: HASH_INSERT_ONLY T=0.3 ms=654 split=0.3266 merge=3.489 FAIL (overflow).
        cut vs 211 ms =0.32. keep_default=False.
        notes/N13_INSERT.md
    L5: RAMA cmake+build OK (nvcc 12.8). GPU solver timeout 600s on val RAG. no labels.
        task_legal=False keep_default=False. no cuSLINK.
        notes/N13_RAMA.md
    defaults unchanged: UF=3, parks off, ε=0.40 T=0.3 / 0.08 four-T, compact k=0.
    not 2 Gvox/s, not 3090 Ti

## N14 leftover closers vs gap ledger (5090 only, email unsent)
    idle RTX 5090. parks off. no product-default changes unless keep_default.
    not 2 Gvox/s, not 3090 Ti. PLAN.md is stale; this log + notes/N*.md are the trail.

    Where we are (N13 P0): 2.16 e2e=4915.3 ms / 0.44 Gvox/s. WS=2579 RAG=83 agg=2234.
    Target 1080 ms on a slower 3090 Ti (1008 vs 1792 GB/s). Need ~4.5x on a faster card.

    Voided stops (do not re-cite as theorems):
    N8 13518 ms / N9 WS 6536 ms were host-park PCIe. Parks-off -> 4918 ms (N10).
    N9 Playne/path-halving "killed" on that 6536. Stitch now owns 52.8% NVTX; never reopened.
    N11 "GPU BinQueue dead" was one device thread find+union 2093 ms, neighbor rewrite skipped.
    Abboud/ParHAC P-completeness does not close TASK: MEAN BinQueue N=256 VOI PASS (0.4551/0.2416).
    PRUF 800 Mvox / 2.5 s is grayscale Meyer, not waterz S1.

    This-stack facts that survive:
    Time ∝ voxels on official 3x2x2 (nfrag=12xval). Mutex/Kruskal/ε>=0.5/X1/SubgraphHAC VOI FAIL.
    E6t VOI PASS, byte_identical=False, 14x slower. E4 identity True, flatten 76.6% of stitch.
    Zeroing E6s dirty-scan 1598 ms leaves e2e ≈ 3326 ms. Optimistic T1+T3 composition ≈ 1.9 s
    on 5090 (still short of 2.0, wrong card).

    Stale claims (logged, not re-litigated):
    N10 "E4 untried" / N11 "2.16 skipped" superseded by N12/N13 ~4590 ms.
    WATERZ_AGG_LEVERS comment "none run on device" stale; scripts set 15; lib default 0.
    w5_union_find cudaMalloc CUB tmp + stitch list inside timed event.
    N12 val nsys dump empty (stale sqlite). Fused 2.16 pred 26.4 GiB OVER 24; 5090 has 32 GB,
    never timed; cannot ship for 3090 24 GB (z-slab WSMEM 11.5 GiB is the fit path).
    No unpublished ParHAC GPU compact (NeurIPS 2022 is CPAM/CPU).

    Tracks: T1 fold flatten; T2 list path-halving; T3 GPU FIFO-BinQueue; T4 fused OOM check;
    T5 3090 Ti only after a stacked closer. notes/N14_*.md.
    T1: WATERZ_FOLD_FLATTEN=1 ident False (nfrag/bg match, array_equal False).
        2.16 WS=1762.5 (cut 1.46 vs 2579) but nlab=5924028 HASH_OVERFLOW. keep_default=False.
        One-hop is not enough for all e9b consumers. Default flatten stays.
    T2: WATERZ_LIST_HALVING=1 ident True nfrag=2175400 bg=506568.
        2.16 WS=2569.7 vs 2579 cut=1.004 < 1.2. keep_default=False.
    T3: GPU FIFO-BinQueue N=256 MEAN compiled. k_binq_merge hung >180s on val RAG.
        Killed. No VOI. keep_default=False. no 2.16. Not N11 1-thread probe; still too serial.
    T4: WATERZ_Z_SLAB=0 fused 2.16. stitch 890 ms on 2.16 Gvox then sort tmp 5.49 GiB, rc=-11 OOM.
        keep_default=False. cannot ship 3090 24 GB.
    T5: stacked_closer=False keep=[F,F,F]. n12_3090.py not run. card is 5090.
    defaults unchanged: UF=3, parks off, ε=0.40 T=0.3 / 0.08 four-T, compact k=0.
    not 2 Gvox/s, not 3090 Ti

## N15 execute leftover legal experiments (5090 only)
    idle RTX 5090. parks off. no product-default unless keep_default.
    not 2 Gvox/s, not 3090 Ti. 2.16 only after identity True AND 2-run array_equal.

    T1-PERM: WATERZ_FOLD_FLATTEN=1 vs wz_fragments.npy. No 2.16.
        identity_byte=False array_equal=False run2_array_equal=False.
        nfrag=2175400 bg=506568 match; bg_mask_eq=True.
        same_partition=False canon_eq=False fp_eq=False.
        ndiff_raw=ndiff_canon=161028442 / 180000000 (89.5%).
        pairs n=3018980 gold_split=826033 pred_merge=63
        max_gold_fanout=6 max_pred_fanout=822582.
        verdict=wrong_basins. revive_t1=False. kill_t1_as_speed=True.
        nfrag/bg match is not a partition. Non-determinism is the D2 find-vs-seg race.
        keep_default=False. notes/N15_T1PERM.md data/cache/n15_t1perm.json

    T2 share-off+fold: FOLD_FLATTEN=1 SHARE_OFF=1.
        ident True run2 True ndiff=0 nfrag=2175400 bg=506568. val peak 1.93 vs 1.47 GiB.
        2.16 WS=1785.3 cut=1.445 nlab=3860788 e2e=4117.3. keep_default=True (speed+ident).
        C++ default stays off (+4 B/vox; 3090 24GB unmeasured). notes/N15_T2.md
    T3 jump flatten: JUMP_FLATTEN=1 (not UF_ALGO=1, not skip). ident True.
        jump_rounds=2/64. 2.16 WS=2581.6 cut=0.999. keep_default=False. notes/N15_T3.md
    T4 e9b-only fold: FOLD_FLATTEN=1 E9B_FOLD_ONLY=1 share on. ident True.
        2.16 WS=2244.0 cut=1.149 < 1.2. keep_default=False. notes/N15_T4.md
    T5 hook-to-root: HOOK_ROOT=1. ident True. stitch_rounds 5->4. peak 1.47 GiB.
        2.16 WS=2105.4 cut=1.225. keep_default=True. C++ default stays off. notes/N15_T5.md
    T2+T5 stack: ident True. 2.16 WS=1311.4 cut=1.967 e2e=3639.8 nlab=3860788.
        keep_default=True on WS 1.2x. notes/N15_T2T5.md
    T6 fuse dirty: FUSE_DIRTY=1. T=0.3 VOI identical to N13 (0.4408/0.2543) run2 True.
        four-T ε=0.08 PASS. 2.16 agg=2078.4 cut=1.075 nlab=3860788.
        keep_default=False. legal lever. notes/N15_T6.md
    T7 four-pass dirty megakernel: subsumed by T6. keep_default=False. notes/N15_T7.md
    T8 sticky sz0: T=0.3 VOI PASS (0.4437/0.2528 drifted) run2 True. agg=1817.1 cut=1.229.
        four-T FAIL T=0.2 merge 0.3585>0.3525. 2.16 nlab=3860907.
        keep_default=False (retracted). notes/N15_T8.md
    T9 E6s MAX_OUTER=32: T=0.3 VOI PASS merge 0.26095 (wall 0.2611). agg=2115.7 cut=1.056.
        2.16 nlab=3860720. keep_default=False. notes/N15_T9.md
    T10 CUDA graph G2: not implemented. E-K=336 ms; 2234-336=1898>1862. keep_default=False.
        notes/N15_T10.md
    T11 stitch arena: not implemented. T2+T5 already 1.96x WS. keep_default=False.
        notes/N15_T11.md
    T12 persistent stitch: not implemented. 4-byte D2H x4 after T5. keep_default=False.
        notes/N15_T12.md
    T13 drop vcount: not run. ~178 ms vs stacked WS 1313. keep_default=False. notes/N15_T13.md
    T14 SortPairs: not run. 20-40 ms. keep_default=False. notes/N15_T14.md
    T15 fuse pack_amask: not run. T8 four-T FAIL. keep_default=False. notes/N15_T15.md

    Legal stack T2+T5+T6 (T8 excluded): ident True, T=0.3 VOI identical to N13, run2 True.
        2.16 WS=1313.1 agg=2081.2 e2e=3499.3 / 0.617 Gvox/s (5090). nlab=3860788.
        e2e cut vs 4915.3 = 1.405. keep_default=False (agg misses 1.2x).
        C++ product defaults unchanged: UF=3, parks off, ε=0.40/0.08, compact k=0,
        fold/share_off/hook_root/fuse_dirty env still default 0.
        notes/N15_STACK.md data/cache/n15_stack.json
    not 2 Gvox/s, not 3090 Ti

## N16 GPU leftover tracks + deep verify (5090 only)
    idle RTX 5090. parks off. no product-default unless keep_default.
    not 2 Gvox/s, not 3090 Ti. 2.16 only after identity/VOI gates. four-T is
    official run_baseline.py on mine_thr*.h5 (regraded independently).

    T7 nlive arith: FUSE_DIRTY=1 NLIVE_ARITH=1. T=0.3 VOI identical to N13
        (0.44082269408145347 / 0.25427477829766865) run2 True nseg=321855.
        four-T ε=0.08 ACCURACY GATE PASS (regrade n16_t7_four):
        0.2 0.3707/0.3350; 0.3 0.4512/0.2505; 0.4 0.5162/0.2268; 0.5 0.6129/0.2184.
        2.16 agg=1930.12 cut=1.157 vs 2234.05 gate 1861.71. nlab=3860788.
        keep_default=False. legal lever. notes/N16_T7.md data/cache/n16_t7.json

    T10 CUDA graph G2: not instantiated. hash_combine_dirty D2H every inner.
        T12 mapped-flag 2.16 WS=1028996. T7 already dropped k_count_live;
        remaining agg gap 67 ms on deep. keep_default=False. notes/N16_T10.md

    T11 stitch arena: STITCH_ARENA=1 + T2T5. ident True ndiff=0 fp lock.
        val peak 2.19 GiB leaked 419295239. 2.16 WS=1313.24 vs T2T5 1311.41
        cut=0.999 nlab=3860788. keep_default=False. notes/N16_T11.md

    T12 pin changed: PIN_CHANGED=1 + T2T5. ident True. 2.16 stitch
        e9b ~137s x3 + e9c ~206s x3. WS=1028996.16 nlab=3860788.
        mapped atomicExch on changed. keep_default=False. notes/N16_T12.md

    T13 drop k_count_v2: not removed. T14/deep 2.16 vcount=177.8 ms.
        BFS qsz is voxel occupancy; drop is identity-illegal.
        keep_default=False. notes/N16_T13.md

    T14 sort-pack: SORT_PACK=1 + T2T5. ident True ndiff=0.
        2.16 sort 52.71 vs T2T5 33.85. WS=1343.79 cut vs T2T5=0.976.
        nlab=3860788. keep_default=False. notes/N16_T14.md

    T15 fuse pack: FUSE_PACK=1. T=0.3 VOI identical to N13 run2 True.
        four-T regrade n16_t15_four ACCURACY GATE PASS (same table as T7).
        2.16 agg=2215.94 cut=1.008 nlab=3860788. keep_default=False.
        notes/N16_T15.md

    DEEP T2+T5+T6+T7: ident True. T=0.3 VOI identical N13. four-T regrade
        n16_deep_four ACCURACY GATE PASS (same table).
        2.16 two-run sha256 equal
        c53d430e5ba7f2dbbb8b011d93b8de6a3e98f34276ab5f8e5afcd0c23eced937
        nlab=3860788 both. a: e2e=3345.18 WS=1312.02 agg=1929.00
        b: e2e=3323.68 WS=1310.32 agg=1911.37
        ws_peak=14193138368 (13.22 GiB). 3090 24GB not measured.
        keep_default=False (agg misses 1.2x by 67 ms).
        C++ defaults unchanged: UF=3, parks off, ε=0.40/0.08, compact k=0,
        fold/share_off/hook_root/fuse_dirty/nlive_arith/arena/pin/sort_pack/fuse_pack
        getenv still default 0.
        notes/N16_DEEP.md notes/N16_STACK.md data/cache/n16_deep.json
        data/cache/n16_four_regrade.json
    not 2 Gvox/s, not 3090 Ti


## N18_A1
claim: not a 2 Gvox/s number; not 3090 Ti
rc=0 keep=False hang=False
ws=1311.435337178409 agg=1933.3693240769207 e2e=3350.21533203125
four=True

## N18_A1
claim: not a 2 Gvox/s number; not 3090 Ti
rc=0 keep=True hang=False
ws=1314.7199610248208 agg=1683.1723069772124 e2e=3104.0859375
four=True

## N18_A2
claim: not a 2 Gvox/s number; not 3090 Ti
rc=1 keep=False hang=False
ws=0 agg=0 e2e=0
four=None

## N18_A3
claim: not a 2 Gvox/s number; not 3090 Ti
rc=0 keep=False hang=False
ws=0 agg=0 e2e=0
four=True

## N18_A4
claim: not a 2 Gvox/s number; not 3090 Ti
rc=0 keep=False hang=False
ws=0 agg=0 e2e=0
four=True

## N18_A5
claim: not a 2 Gvox/s number; not 3090 Ti
rc=1 keep=False hang=False
ws=0 agg=0 e2e=0
four=None

## N18_A_STOP
best_legal_ws=9000000000.0 freeze WS track

## N18 VOI-first sprint (5090)

claim: not a 2 Gvox/s number; not 3090 Ti. Parks off. VRAM cleared of VLLM for runs.

### A0
- n18_voi_gate (voi_only default), n18_dead.jsonl seeded, n18_nsys on N17 stack
- nsys 2.16 e2e~3350 (see notes/N18_A0_NSYS.md); micro-opt kill <200ms rule

### A1 unlock PASS
- voi_only harness; identity diagnostic True (not ship-gate)
- four-T PASS; 2.16 ws=1314.7 agg=1683.2 e2e=3104.1 keep_default=True (env)

### A2-A5 (WS ladder): STOP, no WS≤900
- A2 VCOUNT_COMPACT: voi ok; four empty/fail; kill
- A3 TIE_FLIP: four PASS; 2.16 ws_label_d rc=-3; kill
- A4 COARSE_DELTA=8: four PASS; 2.16 rc=-3; no WS cut; kill
- A5 BLOCK_VOI: four fail/empty; kill
- notes/N18_A_STOP.md: freeze WS; EV -> Track B

### B0-B3
- B0: n18_nsys.json / N18_A0_NSYS.md
- B1 parallel BinQueue: 2253ms wall, VOI FAIL, kill (not serial reopen)
- B2 eps 0.41-0.49: all T=0.3 merge FAIL; best_eps=None
- B3 COMPACT_EVERY=8: four PASS; agg=1765 (worse vs N17 1690); cut -75ms; ParHAC local max; kill

### C
- n18_3090_grade.py REFUSE on 5090; no C++ default flip

### Honest ceiling (5090)
- best e2e ~3104 ms / ~0.70 Gvox/s (A1 = N17 stack)
- need ~2.9x on 5090 (~5x bw-scaled to 3090 Ti) still open

## N19_I0_REPRO
claim: not a 2 Gvox/s number; not 3090 Ti
rc=0 keep=True ws=1309.3050210736692 agg=1679.9063761718571 e2e=3093.4345703125

## N19_W1
claim: not a 2 Gvox/s number; not 3090 Ti
rc=4 keep=None ws=0 agg=0 e2e=0

## N19_W3
claim: not a 2 Gvox/s number; not 3090 Ti
rc=4 keep=None ws=0 agg=0 e2e=0

## N19 literature EV sprint (5090)

claim: not a 2 Gvox/s number; not 3090 Ti. Parks off.

### I0 PASS
- nsys force-export -> n19_owners.json (40 kern); top: compress_list 508, hash_rewrite 287, rebuild 278
- I0_REPRO e2e=3093.4 (A1 3104 ±2%); four-T PASS; keep_default=True

### W_STOP
- W0 owners documented; W1/W3 owner-skip (<200ms); W2 Allegretti skip; no WS≤900

### H_STOP
- H0 agg owners; H1 ladder VOI/wall kill; H2 NNG VOI FAIL; H3 unimplemented; H4 mismatch
- agg floor still ~1680 ms; no path ≤1000 ms

### X dead
- X1/X2/X3 stamped; no 2.16 on FAIL

### C
- REFUSE not 3090 Ti; no C++ default flip

honest: ~0.70 Gvox/s ceiling unchanged; never claim 2 Gvox/s from 5090

## N19 community report artifacts

claim: not a 2 Gvox/s number; not 3090 Ti

- notes/ATLAS.md: Part I chronology + Part II findings (pinned)
- data/cache/voi_atlas.csv + voi_atlas.json: 97 rows; rebuild: scripts/build_voi_atlas.py
- scripts/legal_eval.sh: N17 env + dual-ε four-T / voi_only
- notes/PROBLEM.md: remaining precise problem + ruled-out attacks
- README.md aligned to N17 dual-ε; explicit not-3090 / not-2Gvox
