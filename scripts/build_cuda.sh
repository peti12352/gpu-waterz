#!/usr/bin/env bash
# Build the CUDA shared libraries required by gpu_waterz / src/segment.py.
# Wraps the same nvcc flags used in scripts/b_dev_aff.py and scripts/e6r_parhac.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NVCC="${WATERZ_NVCC:-/usr/local/cuda-12.8/bin/nvcc}"
if [[ ! -x "$NVCC" ]]; then
  NVCC="$(command -v nvcc || true)"
fi
if [[ -z "${NVCC}" || ! -x "$NVCC" ]]; then
  echo "nvcc not found; set WATERZ_NVCC to your CUDA toolkit nvcc" >&2
  exit 1
fi
ARCH_FLAGS=(
  -gencode arch=compute_86,code=sm_86
  -gencode arch=compute_89,code=sm_89
  -gencode arch=compute_86,code=compute_86
  -gencode arch=compute_120,code=sm_120
)
INC=(-I/usr/local/cuda-12.8/targets/x86_64-linux/include)
if [[ ! -d /usr/local/cuda-12.8/targets/x86_64-linux/include ]]; then
  INC=()
fi

build_one() {
  local src="$1" dest="$2"
  echo "nvcc -> $dest"
  "$NVCC" -O3 "${ARCH_FLAGS[@]}" --shared -Xcompiler -fPIC \
    "${INC[@]}" -o "$dest" "$src"
}

mkdir -p "$ROOT/src"
build_one "$ROOT/csrc/ws.cu"      "$ROOT/src/libws_gpu.so"
build_one "$ROOT/csrc/rag.cu"     "$ROOT/src/librag_gpu.so"
build_one "$ROOT/csrc/parhac_d.cu" "$ROOT/src/libparhac_d.so"
echo "done: libws_gpu.so librag_gpu.so libparhac_d.so"
