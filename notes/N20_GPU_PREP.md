# N20 GPU prep (not a run)

N20 GPU prep only; not a run; not a throughput claim. host=greengoblin.

- go_star=False (must stay false; D1 layer-0 is not small-merge)
- go_gpu_rnn=False
- nvcc=missing
- smi: ['GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-aaabe7a6-e328-9869-cca8-a8e56a3d4591)']
- skeleton: `n20_rnn_gpu_prep.cu` exists=True
- Do not set N20_GPU_RUN. Do not nsys. Do not 2.16.
- Product default stays E6s. Do not reopen AGG_E6t.
