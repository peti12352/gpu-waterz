# N11 A: parks-off 2.16 WSMEM

Not a 2 Gvox/s number. Not 3090 Ti. Idle 5090.

```
WATERZ_UF_ALGO=3 WATERZ_HOST_PARK=0 WATERZ_AFF_PARK=0 \
WATERZ_WS_MEMLOG=1 WATERZ_STAGE_MS=1 \
.venv/bin/python -u scripts/n8_run216.py
```

- e2e **4913.5 ms / 0.440 Gvox/s**
- stages: WS 2582, RAG 83, agg 2229
- **WSMEM peak=11.518 GiB** (sort-tmp, aff resident, z-slab)
- 23 usable: **fits**. Parks-off is not 5090-only on this tracker.
- Product default of `WATERZ_HOST_PARK` / `WATERZ_AFF_PARK` flipped to **off**.

WSMEM is the `ws.cu` tracker (WS + credited aff). RAG/AGG allocations are not in these lines.
