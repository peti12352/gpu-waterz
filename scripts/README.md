# scripts/

Product entry points and CPU replicas. Campaign probes live in local
`archive/scripts/` (gitignored).

| Script | What it does |
|---|---|
| `build_cuda.sh` | nvcc `ws.cu` / `rag.cu` / `parhac_d.cu` into `src/lib*.so` |
| `legal_eval.sh` | four-T VOI + T=0.3 identity on the public API (`check.py`) |
| `check.py` | the gate `legal_eval.sh` calls; `--216` adds 2.16 Gvox timing |
| `voi_numpy.py` | VOI without the waterz C++ eval |
| `e1_synthetic.py` | tiny S1-S4 asserts, no GPU |
| `w0_ws_ref.py` | CPU replica of affinity-flow watershed |
| `g0_agg_ref.py` | CPU replica of device ParHAC |
| `nvcheck.py` | clang `-fsyntax-only` on the CUDA sources, no GPU |

```
bash scripts/build_cuda.sh
uv sync --extra eval    # h5py, for legal_eval
bash scripts/legal_eval.sh
```

Leftover kernels and closed attacks: [notes/lab.md](../notes/lab.md).
VOI atlas: [data/cache/voi_atlas.csv](../data/cache/voi_atlas.csv).
