# N12 fatbin sm_86+sm_120

Not a 2 Gvox/s claim. Not 3090 Ti.

`nvcc -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120` in `b_dev_aff.build` and `e6r_parhac.compile_d`.

5090 parks-off UF=3 default: e2e **4923.9 ms** vs N11 4913.5 (**+0.21%**, inside ±2%). WS=2581 RAG=94 agg=2230.

`scripts/n12_3090.py` is median-of-5, asserts nvidia-smi contains `3090 Ti`. **Do not run until rented.** Host ≥64 GB, GPU 24 GB; parks-off OOM would mean N11 WSMEM 11.5 GiB missed RAG/AGG.
