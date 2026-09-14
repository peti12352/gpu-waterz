# scripts/

Product entry points and CPU replicas.

| Script | What it does |
|---|---|
| `build_cuda.sh` | nvcc `ws.cu` / `rag.cu` / `parhac_d.cu` into `src/lib*.so` |
| `eval.sh` | four-T VOI + T=0.3 identity on the public API (`check.py`) |
| `check.py` | the gate `eval.sh` calls; `--216` adds 2.16 Gvox timing |
| `make_big.py` | edge-aware 3x2x2 tile of val -> `data/cremiA_216/affinity.h5` |
| `voi_numpy.py` | VOI without the waterz C++ eval (same `evaluate.hpp` formula) |
| `e1_synthetic.py` | tiny S1-S4 asserts, no GPU |
| `w0_ws_ref.py` | CPU replica of affinity-flow watershed |
| `g0_agg_ref.py` | CPU replica of device ParHAC |
| `nvcheck.py` | clang `-fsyntax-only` on the CUDA sources, no GPU |
| `plot_speed.py` | rebuild `docs/speed_216.png` from the 5090 pin JSON |

`legal_eval.sh` is a wrapper that execs `eval.sh`.

```
bash scripts/build_cuda.sh
uv sync --extra eval
bash scripts/eval.sh
```

Leftover kernels and closed attacks: [notes/lab.md](../notes/lab.md).
VOI atlas: [data/cache/voi_atlas.csv](../data/cache/voi_atlas.csv).
