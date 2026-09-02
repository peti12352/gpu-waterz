# gpu-waterz

GPU-assisted watershed + mean-affinity agglomeration matching stock `waterz` quality.

**Contract:** [TASK.md](TASK.md). **Log:** [notes/LOG.md](notes/LOG.md). **Sources:** [papers/SOURCES.md](papers/SOURCES.md).

## API

```
from segment import segment
labs = segment(aff, [0.2, 0.3, 0.4, 0.5])  # list of uint32 [Z,Y,X]
```

`aff` is uint8 or float32 `[3,Z,Y,X]`, numpy (torch CUDA tensors are copied to host). uint8 scale is `/255` in fp32 at the point of use. CLI:

```
python src/segment.py cremiA_val/affinity.h5 --out-dir . --thresholds 0.2 0.3 0.4 0.5
python baseline/run_baseline.py --candidate mine_thr0.2.h5 mine_thr0.3.h5 mine_thr0.4.h5 mine_thr0.5.h5
```

`scripts/eval.sh` runs segment, the shipped grader, determinism, and the val bench.

## Algorithm

Same statistic as waterz `OneMinus<MeanAffinity>` (`funkey/waterz` `a0184d2`):

1. **Flow (GPU).** 6-neighbour affinities, OOB=`low`, bits where `aff==m || aff>=high`. Background iff `m<=low`.
2. **Plateau + basins (host, S1).** Corner BFS rewrite, then basin BFS. Dir order `(-z,-y,-x,+z,+y,+x)`. Deterministic.
3. **RAG (GPU atomic hash).** Three negative dirs, drop bg. Device hash of `key=(min<<32)|max` with `atomicAdd` on `(sum,count)`. Mean is `sum/count`.
4. **Agglomeration (CPU ParHAC ε=0.01).** Matching of (1+ε)-heavy waterz-mean edges, S3-contract. Same ε at all T. RAC (ε→0) also PASSes VOI but is more serial. Exact S4 heap is 18s on val (the merge loop, not the wrapper). Rejected: frozen CC, union-all-in-band, mutual-in-bucket without fallback.

Thresholds in the API are **affinity**. waterz scores are `1-aff` (see `notes/THRESHOLD.md`).

## Where it diverges from waterz

- Flow bits are computed on GPU; plateau/basin match the vendored C++ (not waterz enqueue-order jitter). Two `segment()` calls are byte-identical (G5).
- Fragment IDs need not match waterz. The partition is graded.
- Agglomeration is ParHAC ε=0.01 (Y2 grader PASS). RAC also PASSes. Not the 18s S4 heap and not live-mean Borůvka.

## Accuracy (CREMI-A val, shipped grader)

`ACCURACY GATE: PASS` at aff 0.2/0.3/0.4/0.5 (G4). Both halves within +0.02 of `voi.csv`.

## Speed and memory

Val 180 Mvox @ aff 0.3 on greengoblin RTX 5090 (wall, aff already in RAM; heap is CPU):

- median **36.6 s** (~0.005 Gvox/s) on the pre-hash-RAG binary. Gate is 50 ms local / 2 Gvox/s on a **3090 Ti**.
- After the atomic-hash RAG: RAG val ~0.4 s (was 2.6 s). Watershed still ~5.5 s (host plateau). Heap still ~28–36 s.

The serial S4 heap on 7.5 M (val) / 90 M (bench) edges is the speed blocker. No parallel agglomerator has passed all four VOI thresholds. No 3090 Ti number is claimed. Peak design: keep uint8 aff + uint32 labels; no fp32 aff copy. 24 GB cap on the 2.16 Gvox volume is not yet measured on a passing timed path.

### G9 (3090 Ti) — last mile, not run here

On a 24 GB RTX 3090 Ti, after `make_big.py` has written `[3,375,2400,2400]`:

```
python src/segment.py big/affinity.h5 --out-dir /tmp/wz --thresholds 0.3
python scripts/bench.py   # must be CUDA-event, aff already in VRAM; median of 5
nvidia-smi --query-gpu=name,driver_version --format=csv
```

Report median Gvox/s only from that card. Do not substitute a 5090 number.

## Known failure modes

- `libheap_cpu.so` built from `heap_s4.cpp` (vendored headers, `-DNDEBUG`). The earlier `heap_cpu.cpp` port over-merged at aff 0.2 (+0.017).
- Plateau still on the host: G2-accurate, ~5.5 s on val, too slow for 2.16 Gvox at 2 Gvox/s. GPU wavefront attempts over-split (2.3–12 M fragments).
- PyTorch CUDA wheel install on this box has been flaky (nvidia pypi timeouts). `segment()` does not require torch.

Dev: greengoblin RTX 5090, driver 580, nvcc 12.8, `sm_120`. Graded card: RTX 3090 Ti (not here).
