# Porting and optimizing on another GPU

Every published millisecond in this repo is **idle RTX 5090, 32 GB,
sm_120, driver 580, nvcc 12.8**, CUDA events, affinity already in VRAM,
parks off. Pin: `data/cache/N19_I0_REPRO.json`. Volume
`[3,375,2400,2400]` = 2.16 Gvox at affinity 0.3.

Do not multiply those times by a bandwidth ratio. Time the card you have.
Quality (four-T VOI, fragment identity) is independent of the card; speed
is not.

---

## 1. Build for the SM you run on

`scripts/build_cuda.sh` fatbins **sm_86** (Ampere) and **sm_120**
(Blackwell). That is SASS, not a PTX forward-compat blob. Ada, Hopper,
or anything else needs another `-gencode arch=compute_XX,code=sm_XX`.

```bash
# after adding your SM to ARCH_FLAGS in scripts/build_cuda.sh
bash scripts/build_cuda.sh
uv run python -c "import gpu_waterz as w; print(w.cuda_libs_ready())"
```

Lab nvcc is 12.8 (`WATERZ_NVCC` if yours lives elsewhere).

---

## 2. Fit in VRAM (measure peak; do not use listing arithmetic)

On 2.16 Gvox:

| Fact | Number | Source |
|---|---|---|
| Naive fused WS working set | ~42 GiB | `d1_mem.py` / SOURCES S40 |
| Legal stack WS peak (z-slab) | **13.22 GiB** | N17_DEEP |
| `SHARE_OFF=1` extra | +4 B/vox | ATLAS |
| Affinity uint8 + labels uint32 | 6.5 + 8.6 GiB | input/output only |
| 8-tile serial fallback | 8.3x slower; dead | ATLAS |

If you OOM: the fit path is **z-slab N=3** (slab Z=125, aff=0 seams), not
more serial tiles. Do not flip C++ buffer-sharing defaults to save 4 B/vox
until you have a peak on **this** card (`FOLD` without `SHARE_OFF` broke
basins: 89% voxels differed).

---

## 3. Time the same way we did, or the number is a different quantity

| Rule | Why (measured) |
|---|---|
| Affinity in VRAM before the event window | `HOST_PARK`/`AFF_PARK` put pageable PCIe inside the timer. N8 e2e 13518 ms vs N10 parks-off 4918 ms; 6 GiB H2D was booked as RAG |
| Idle GPU | Co-tenant (VLLM) moved one workload 1634-5018 ms (~3.1x) |
| CUDA events, median of 5 after warmup | same method as the 5090 pin |
| Four-T VOI **before** a 2.16 speed claim | N18 A4: T=0.3 VOI looked OK after nfrag 2.175M -> 3.14M |
| Ignore val cuts < 100 ms | N18 B3: `COMPACT_EVERY=8` "won" on val, **lost** on 2.16 (1765 vs 1690 ms agg) |
| Dual-eps 0.08 / 0.40 | Partition law, not a hardware knob. 0.41-0.49 fail merge VOI at T=0.3 |

Env for the measured stack (all getenv; see README):

```
WATERZ_UF_ALGO=3 WATERZ_HOST_PARK=0 WATERZ_AFF_PARK=0 WATERZ_AGG_LEVERS=15
WATERZ_FOLD_FLATTEN=1 WATERZ_SHARE_OFF=1 WATERZ_HOOK_ROOT=1
WATERZ_FUSE_DIRTY=1 WATERZ_NLIVE_ARITH=1 WATERZ_EMIT_HOLES=1
```

Do not default `WATERZ_LISTED_INSERT`, `WATERZ_LISTED_REBUILD`,
`WATERZ_SLOT_EMIT`, `WATERZ_LIST_JUMP`. Identity-true on val; not taken to
2.16.

---

## 4. What owns the 5090 pin

e2e **3093 ms (~0.70 Gvox/s)**: WS 1309, RAG 85, agg 1680, extract 18.

Largest nsys owners on that run (`n19_owners.json`):

| kernel | ms | note |
|---|---|---|
| `k_w5_compress_list` | 508 | WS; 71-105 M list entries. Jump-4 was slower. Do not drop plateau faces |
| `k_rebuild_active` | 278 | ParHAC, 438 launches |
| `k_rewrite_dirty_fuse` | 273 | still O(nscan); listing dirty edges does not delete this |
| `k_hash_insert` | 174 | |
| `k_hash_emit_holes` | 161 | |

If your nsys ranking is the same, you have the same problem on a different
clock. If compress explodes on a smaller L2, that is expected: 5090 L2 is
large (~96 MB); several gathers are cache-sensitive. HBM GB/s is the wrong
scaler.

Leftover if the heavy agg kernels vanished: still ~690-1130 ms agg plus
hundreds of ms of WS. New merge classes (mutex, Kruskal, RNN, StarMerge)
change the partition or do not close 1680 ms. See
[WHERE_WE_ARE.md](../notes/WHERE_WE_ARE.md).

---

## 5. Checklist on a new card

1. Add SM, rebuild, `cuda_libs_ready()`.
2. CREMI-A val four-T (`bash scripts/legal_eval.sh`) and fragment identity
   vs `wz_fragments.npy` if you have it.
3. Record `nvidia-smi` name, driver, VRAM, `ws_peak`.
4. Idle, parks off, CUDA-event e2e on the volume you care about.
5. nsys owners vs the table above.
6. Publish **card + volume + eps + thresholds** next to the milliseconds.
   Do not paste the README 0.70 Gvox/s onto another GPU.
