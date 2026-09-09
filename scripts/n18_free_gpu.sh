#!/bin/bash
# Hold the 5090 free of vllm for N18 runs. Kill EngineCore + parent serve.
set -e
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -9 -f 'vllm serve' 2>/dev/null || true
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
# Wait until no compute apps (or only ours)
for i in $(seq 1 30); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | wc -l)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${apps:-0}" -eq 0 ] && [ "${mem:-99999}" -lt 500 ]; then
    echo "GPU_FREE mem=${mem}MiB"
    exit 0
  fi
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
  pkill -9 -f 'vllm serve' 2>/dev/null || true
  sleep 1
done
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
echo "GPU_NOT_FREE"
exit 1
