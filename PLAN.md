# PLAN: GPU waterz (2026-08-30, stale)

Living status: [notes/WHERE_WE_ARE.md](notes/WHERE_WE_ARE.md).
Library docs: [README.md](README.md). This file is the original design note.

Contract: `TASK.md`. Quotes: `papers/SOURCES.md`. If this file
softens a TASK number, delete the sentence.

Status 2026-08-30: spec locked to waterz source + listing. Dataset
tarball **not** on disk. No kernel written. No VOI or timing claimed.

---

## 0. What we ship

One function, one CLI, one eval+bench script.

```
segment(aff, thresholds, aff_low=1e-4, aff_high=0.9999) -> list of uint32 [Z,Y,X]
```

`aff`: uint8 or fp32 `[3,Z,Y,X]`, numpy or torch CUDA.
Scale uint8 by `/255` in fp32. Never `a*255+0.5` in bf16/fp16.

Output: final labels (not fragments), background 0, deterministic.
IDs need not match waterz. Partition is graded.

Gates (do not restate as softer):

| Gate | Pass |
|---|---|
| Accuracy | `voi_split <= base+0.02` **and** `voi_merge <= base+0.02` at aff 0.2, 0.3, 0.4, 0.5 on shipped CREMI-A val |
| Speed | median of 5 CUDA-event runs >= 2 Gvox/s on `[3,375,2400,2400]` @ aff 0.3, 3090 Ti, aff already in VRAM |
| Fit | 3090 Ti 24 GB |
| Determinism | `np.array_equal` two runs |
| API / CLI / script / README | as TASK.md |

Trading split vs merge fails. No post except min-size.

---

## 1. What the reference actually does

Verified against `funkey/waterz` `a0184d2` (S1-S6). Agrees with TASK.md
except the stale "maximum affinity" comment in `region_graph.hpp`.

```
aff [3,Z,Y,X]  ──►  flow dirs (6-bit)  ──►  plateau BFS  ──►  basin BFS
                         │
                         ▼
                   fragments I>=1, bg=0
                         │
                         ▼
              RAG: 3 negative dirs, mean aff, drop bg
                         │
                         ▼
         min-heap on score=1-mean; merge while score < T
         shared-neighbour: keep cheaper, area-weighted mean, stale
                         │
                         ▼
                   UF compress, uint32 labels
```

Two facts that decide the design:

1. **Mean affinity is not Kruskal.** Kruskal sorts frozen weights once.
   Waterz reweights every shared-neighbour edge by contact area when
   regions grow (`notifyEdgeMerge` returns true -> stale -> rescore).
   Frozen-weight sort is an approximation. TASK.md allows it *if* VOI
   holds at all four thresholds.

2. **The exact heap is serial.** 26 023 852 fragments / 90 323 139 edges
   on the graded volume. A single-thread GPU heap will miss 1.08 s.
   Watershed + RAG are bandwidth-parallel. Agglomeration is the only
   stage that can fail the speed gate by algorithm, not by coding.

---

## 2. Hardware (measured)

Dev machine **greengoblin** (S12): RTX 5090 32 GB, sm_120, driver
580.159.03, CUDA 13.0 / nvcc 12.8.61, 125 GB RAM, 1.4 T free, Python
3.12.3, `uv` present. Path: `/home/v/proj/petya/waterz`.

5090 is **not** the graded card. TASK.md: develop anywhere, report
3090 Ti. 5090 is faster and fatter (32 vs 24 GB). A kernel that
scrapes 2.0 Gvox/s here can fail on a 3090 Ti.

Roofline we design against (3090 Ti, not measured here):

| | 3090 Ti (TASK) | 5090 (dev, measured) |
|---|---|---|
| VRAM | 24 GB | 32607 MiB |
| Leftover after aff+labels | ~9 GB | ~17 GB |
| Speed gate | 2 Gvox/s | **not graded** |

If the pipeline is bandwidth-bound, 5090 numbers overstate 3090 Ti.
Target on 5090: **>= 4 Gvox/s** median on the 2.16 Gvox volume before
we claim the speed gate is likely. That is a local go/no-go, not a
TASK number.

GPU was occupied at inspect (lego `gate3_pass_at_k.py`, ~8.8 GB).
Do not kill it. Val (180 Mvox) fits in the free ~23 GB. Full bench
waits until the card is idle.

No PyTorch install is dedicated to this tree yet. System python has
no torch. Existing torch lives in other projects (`zebra`,
`toneking_`). We make our own `.venv` with `uv`. Do not reuse those
envs.

---

## 3. Design (shortest thing that can pass both gates)

Ponytail: one algorithm, no mutex, no multi-threshold hierarchy, no
new deps beyond PyTorch + CUDA.

### 3.1 Affinity

Keep uint8 in VRAM. Promote **one voxel-neighbour** to fp32 at the
point of use (`aff_u8 * (1/255)`). Never write a 6.5 GB fp32 copy
(that is +19.5 GB and blows 24 GB).

### 3.2 Watershed: GPU union-find on waterz flow, not intensity

Do **not** implement an image-gradient watershed (S8). Implement
TASK.md / S1 flow:

1. Kernel: 6 loads, `m`, background if `m <= low`, else set dir bits
   where `aff_d == m || aff_d >= high`.
2. Plateau: same corner rule as S1. Break ties with a **fixed**
   direction order `(-z,-y,-x,+z,+y,+x)` then lower linear index.
   Pointer-jump / union-find along the single remaining parent, not
   host BFS. Deterministic because the parent is a function of the
   bits + the order, not of enqueue time.
3. Connected components of the parent forest -> fragment IDs from 1.
   Write into the uint32 labels buffer. No second 8.6 GB volume.

This is S8's PRUF *pattern* (path reduction + UF) applied to S1's
affinity flow. Expected: watershed ≪ 1 s on 2.16 Gvox. Not the
hard part.

### 3.3 RAG: 3-dir voxel scan, 90 M-slot hash

One thread per voxel, three negative dirs, skip `id==0` and `id1==id2`.
Atomic add `(sum, count)` into a hash of `key = (min<<32)|max`.
90 323 139 edges x ~32 B ≈ 2.9 GB; hash load 0.5 -> ~6 GB. Fits the
9 GB leftover on a 3090 Ti if we do **not** also keep a float aff
copy.

After the scan: compact to a dense edge list
`(u, v, sum, count, mean=sum/count, score=1-mean)`.

### 3.4 Agglomeration: live-mean Borůvka, exact-heap tail only if VOI fails

**Primary (ship this first):**

Repeat until no edge has `mean > aff_thr`:

1. Each current root picks its cheapest legal neighbour
   (deterministic: min `(score, min(u,v), max(u,v))`).
2. If `u` picked `v` and `v` picked `u`, or we take a consistent
   orientation (lower root absorbs), merge that pair.
3. Independent pairs merge in parallel (Borůvka).
4. On merge: exclusive edges retarget; shared edges
   `mean = (s1+s2)/(n1+n2)` exactly as S3. Recompute score.

This updates means. It is **not** waterz's global min-heap order.
It is the cheapest parallel algorithm that still implements the
S3 statistic. TASK.md explicitly allows "bucketed or sorted-edge
agglomeration, approximate merge order".

**Fallback, only if E5 VOI fails at 0.2 or 0.5:**

After Borůvka has collapsed the graph (edge count should drop
well below 90 M), run waterz's stale-heap loop on the **residual**
graph. 1e6-edge heap is fine. Do not run the heap on 90 M edges.

**Rejected as first attempt:**

| Idea | Why not first |
|---|---|
| Frozen Kruskal (sort once, no mean update) | Wrong statistic; likely fail aff 0.2 or 0.5 (S3 vs S7) |
| Mutex watershed (S7) | Different objective; needs repulsive edges we would have to invent from 3 channels; VOI vs *mean-affinity* waterz is the gate |
| Exact GPU heap, one thread | Misses 1.08 s |
| Blockwise agglomerate + stitch | z is 4x4x40 nm; seams in z move merge-VOI |
| `discretize_queue>0` as the product | Allowed; keep as a *second* fallback, not the design |

### 3.5 Extract

Path-compress UF, one gather into the labels buffer. uint32.

### 3.6 Min-size

Only if E5 needs it and the grader allows it. Default: off.
Do not tune it to launder a bad agglomerator.

### 3.7 Memory on the 2.16 Gvox volume (3090 Ti 24 GB)

```
uint8 aff            6.5 GB   keep
uint32 labels        8.6 GB   also fragment IDs
hash + edge list     ≤ 6 GB
UF parent + size     0.2 GB
────────────────────────────
                     ≤ 21.3 / 24 GB
```

No fp32 aff. No second label volume. If hash peaks over 9 GB leftover,
chunk the RAG **scan** (not the volume) by z-slabs into the same hash.

Chunking the *volume* is allowed (TASK.md) but the clock covers the
whole volume. Do not introduce a stitch that changes the partition
until a profiler says the unchunked path OOMs.

### 3.8 Stack

Python 3.12 + PyTorch (CUDA 12.8 wheel that supports sm_120) +
CUDA C++ for the three kernels (flow, RAG, relabel). Triton only
if a kernel is a one-liner; the hash RAG is C++.

No new library. waterz is a **dev/eval** dependency, not a runtime
dependency of `segment()`.

---

## 4. Threshold convention (must verify, not assume)

TASK.md: waterz `thresholds` are **scores** `1 - mean`; the listing
quotes **affinity**. Our API takes affinity (TASK.md `segment(...,
thresholds)` with the 0.2/0.3/0.4/0.5 table).

E0 reads `baseline/run_baseline.py` and the tarball README and
writes the exact `waterz.agglomerate(..., thresholds=...)` list
they pass. Until that file is on disk, treat

```
waterz_score_thr = 1 - aff_thr
# 0.2->0.8, 0.3->0.7, 0.4->0.6, 0.5->0.5
```

as the **hypothesis**, not a fact. If the shipped script passes
`[0.2,0.3,0.4,0.5]` straight into waterz, the hypothesis is wrong
and we follow the script. TASK.md says the tarball README was
"verified line-by-line against the waterz source": that file
wins over this paragraph.

---

## 5. Experiments

No result is recorded until the command has been run. Each row is
a pass/fail. Do them in order. Do not skip E0.

### E0: lock the tarball

```
# on greengoblin, after GPU is not the issue (this is CPU/disk)
gdown 1zbGpyr9M5Pvhgfy96V9erQwAeRZo23hW -O data/waterz_bounty.tar
# or the exact fetch the tarball README / Drive page states
tar tf data/waterz_bounty.tar | head
```

Pass:

- listing matches TASK.md: `cremiA_val/{affinity,gt,raw}.h5`,
  `baseline/{labels_thr*.h5,voi.csv,run_baseline.py}`, `make_big.py`,
  `viz/`, `README.md`
- val shapes `[3,125,1200,1200]` uint8 and `[125,1200,1200]` uint32
- `baseline/voi.csv` equals the TASK.md table to printed precision
- tarball README vs S1-S5: list every discrepancy. If any, TASK.md
  *listing* still wins for gates; the discrepancy is a note

Also dump `run_baseline.py` threshold list -> resolve §4.

### E0b: reproduce stock VOI

```
uv pip install waterz h5py numpy
python baseline/run_baseline.py
```

Pass: printed VOI within ~1e-05 of `voi.csv` (TASK.md jitter).
Time the four stages. This is the CPU baseline, not a gate.

### E1: synthetic semantics (no GPU required)

Tiny volumes we write, run through **stock** `waterz.agglomerate`
and through our CPU reference of S1-S4 (same math, our tie-break).

Cases:

| ID | Volume | Checks |
|---|---|---|
| E1a | 1 voxel | label 0 if aff missing/low |
| E1b | 2 voxels, one edge | merge iff mean > aff_thr |
| E1c | uniform plateau | one fragment; deterministic ID |
| E1d | sealed background ring | bg stays 0, never in RAG |
| E1e | 3 fragments, 2 shared neighbours | after one merge, surviving edge mean equals S3 formula |
| E1f | anisotropic: weak z, strong xy | fragment count changes when z channel is zeroed |

Pass: CPU reference matches the formulas in S1-S4 bit-exactly on
these volumes. Stock waterz may differ on E1c (order). That is
allowed. We record both.

### E2: GPU watershed vs waterz fragments on val

Run stock waterz with agglomeration threshold high enough that
**no** merge happens (score threshold 1.0 / aff_thr 0.0: confirm
in E0 which call does "fragments only"). Compare:

- fragment count vs 2 175 400 (expect \|Δ\| in the tens, TASK.md
  ~20 in 824k)
- VOI of *our* fragments vs *their* fragments (not vs GT). This
  is a diagnostic, not a gate.

Pass (local): our fragment count within 1% of 2 175 400 and no
crash. Do not block on bit-identity.

### E3: GPU RAG vs waterz RAG on val

Edge count vs 7 505 458. Mean of means within 1e-4 if we use the
same fragments. If fragments differ, compare only on a run that
feeds waterz fragments into our RAG (TASK.md allows `fragments=`
in waterz; we do the inverse: their labels -> our mean).

Pass: edge count within 0.1% on **identical** fragments; means
match S3.

### E4: exact agglomeration correctness lock (CPU heap on val)

Implement S4 on CPU (or a slow GPU heap) on the val RAG. Four
thresholds. `waterz.evaluate`.

Pass: all four pairs within +0.005 of baseline (tighter than the
gate). This proves our statistic + stop rule, independent of
approximation. If E4 fails, the bug is in mean/threshold/bg, not
in Borůvka. **Do not start E5 until E4 passes.**

### E5: approximate agglomeration VOI (the real accuracy gate)

Borůvka live-mean (§3.4) on val, four thresholds, dump
`mine_thr{0.2,0.3,0.4,0.5}.h5`, run the shipped grader.

| Result | Action |
|---|---|
| PASS all 8 numbers | freeze agglomerator; go to E6 |
| fail only 0.2 or only 0.5 | add exact-heap **tail** on residual; re-grade |
| fail split on one thr and merge on another | do **not** retune a single threshold; that is the forbidden trade |
| still fail | `discretize_queue`-style buckets on the residual, then stop. Mutex is last and needs a written reason |

### E6: speed, 180 Mvox, 5090

CUDA events, warmup + 5 runs, aff already on device, one threshold
0.3. Per-stage: flow, plateau/UF, RAG, agglomerate, relabel.

Pass (local): median < 50 ms (3.6 Gvox/s). If a stage is > 50% of
the time, that stage is the only thing we rewrite.

### E7: speed, 2.16 Gvox, 5090

`python make_big.py` -> `[3,375,2400,2400]`. Same timing protocol
as TASK.md. Also `make_big.py --factors 2 2 2` (report only).

Pass (local): median >= 4 Gvox/s and peak alloc < 24 GB (the 3090
Ti cap), not merely < 32 GB.

If we need > 24 GB, chunk before we celebrate a 5090 number.

### E8: determinism + memory

Two runs, `np.array_equal`. `nvidia-smi` / `torch.cuda.max_memory_allocated`
with and without the labels buffer. Document.

Pass: equal labels; peak on the 2.16 Gvox path ≤ 24 GB.

### E9: 3090 Ti (graded)

Same binary, same script. Record exact card + driver. Median of 5
>= 2 Gvox/s. This is the only speed number that may appear in the
submission README.

There is no 3090 Ti on greengoblin. E9 is a machine we do not have
today. Do not invent a 3090 Ti number from a 5090 run.

---

## 6. Work order (files we will actually add)

Do not create these until the previous experiment's pass is in
the log.

| When | Add | Why |
|---|---|---|
| now | TASK.md PLAN.md papers/ SOURCES.md | this commit |
| E0 | `data/` (gitignored), copy of tarball README into `notes/TARBALL_README.md` | lock §4 |
| E1 | `src/ref_cpu.py`: S1-S4 on numpy, no GPU | correctness oracle |
| E4 pass | `src/segment.py` API stub calling GPU ops | TASK.md API |
| E2-E5 | `csrc/*.cu` flow, uf, rag, boruvka | the product |
| E6 | `scripts/bench.py` CUDA events | TASK.md timing |
| E5 pass | `scripts/eval.sh` one command -> VOI table + timing table | TASK.md |
| E8 pass | README: algorithm, divergence, memory, failure modes | TASK.md |

No `setup.py` until the API is real. No tests framework. E1 is
`assert` in `scripts/e1_synthetic.py`.

---

## 7. Kill shots (ranked)

1. **E5 VOI at 0.2 or 0.5** after live-mean Borůvka. Mean-drift vs
   heap order is largest at the ends of the threshold list.
2. **RAG hash OOM / 24 GB** on 90 M edges if we keep a fp32 aff.
3. **5090-only speed.** 4 Gvox/s here, 1.6 on a 3090 Ti -> fail E9.
4. **Threshold unit error.** Passing aff 0.3 into waterz as a score
   (or the reverse) silently shifts every baseline. E0/E4 catch this.
5. **Anisotropic z.** An isotropic 26-neighbour kernel, or treating
   z-aff like xy-aff in a mutex rewrite, moves merge-VOI.
6. **Busy GPU / stolen job.** Do not `kill` PID 1880797.

---

## 8. What we will not build

- Mutex as the first agglomerator (S7 != S3).
- Training, meshing, multi-GPU, CPU fallback, ONNX, TensorRT.
- Bit-identity with waterz.
- A second label volume, a fp32 aff copy, a new Python dependency.
- Fly-brain simulator work in this tree.
- Any number in the submission that was not produced by E0-E9.

---

## 9. Consistency with TASK.md

| TASK.md requirement | Where this plan meets it |
|---|---|
| final labels, not fragments | `segment()` return |
| uint8 / fp32, numpy / torch CUDA | §3.1, API |
| aff_low 1e-4, aff_high 0.9999 | defaults |
| VOI +0.02 both halves, four thrs | E5, shipped grader |
| min-size only extra post | §3.6 |
| `waterz.evaluate` uint64 | E4/E5 |
| 2 Gvox/s, 2.16 Gvox, 3090 Ti, CUDA events, median of 5 | E7 local, E9 graded |
| 1.44 Gvox report | E7 |
| 15.1 + 9 GB | §3.7, E8 |
| deterministic | §3.2 tie-break, E8 |
| one-command bench+eval | §6 `scripts/eval.sh` |
| README contents | §6 last row |
| no affinity-net training | §8 |
| no mutex **input** | input stays 3ch |
| Python 3.10+, PyTorch, Triton/CUDA | §3.8 |
| uint8 `/255`, no bf16 wrap | §3.1 |
| uint32, bg 0 | extract |
| 26 M frags / 90 M edges | §3.3-3.4 |
| 24 GB | E7/E8 cap is 24, not 32 |
| `make_big.py` that volume | E7 |
| `np.array_equal` | E8 |

Nothing in §3 changes a TASK.md number.
