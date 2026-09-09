# Building CUDA libraries

Requires a CUDA toolkit with ``nvcc`` (lab default: 12.8). Fatbin targets
``sm_86`` and ``sm_120`` (see ``scripts/b_dev_aff.nvcc_arch_flags``).

```bash
uv sync
bash scripts/build_cuda.sh
# or: WATERZ_NVCC=/path/to/nvcc bash scripts/build_cuda.sh
```

Outputs (gitignored ``*.so``):

- ``src/libws_gpu.so`` from ``csrc/ws.cu``
- ``src/librag_gpu.so`` from ``csrc/rag.cu``
- ``src/libparhac_d.so`` from ``csrc/parhac_d.cu``

Check: ``uv run python -c "import gpu_waterz as w; print(w.cuda_libs_ready())"``.

``uv sync`` installs the Python package only; build the ``.so`` files separately.
