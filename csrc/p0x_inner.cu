// P0x: dummy device scan+atomic matching. Launch-per-inner vs fused persistent.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>

__global__ void dummy_match(
    const uint32_t* u, const uint32_t* v, const float* w,
    int n, uint32_t* best, float* bestw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t a = u[i], b = v[i];
    float ww = w[i];
    if (ww > bestw[a]) {
        atomicMax((int*)&best[a], (int)b);
        atomicMax((int*)&bestw[a], __float_as_int(ww));
    }
    if (ww > bestw[b]) {
        atomicMax((int*)&best[b], (int)a);
        atomicMax((int*)&bestw[b], __float_as_int(ww));
    }
}

__global__ void dummy_match_fused(
    const uint32_t* u, const uint32_t* v, const float* w,
    int n, int niter, uint32_t* best, float* bestw)
{
    for (int it = 0; it < niter; ++it) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            uint32_t a = u[i], b = v[i];
            float ww = w[i];
            if (ww > bestw[a]) {
                atomicMax((int*)&best[a], (int)b);
                atomicMax((int*)&bestw[a], __float_as_int(ww));
            }
            if (ww > bestw[b]) {
                atomicMax((int*)&best[b], (int)a);
                atomicMax((int*)&bestw[b], __float_as_int(ww));
            }
        }
        __syncthreads();
    }
}

static float bench_launched(const uint32_t* u, const uint32_t* v, const float* w,
    int n, uint32_t* best, float* bestw, int niter, int warmup)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    for (int i = 0; i < warmup; ++i)
        dummy_match<<<blocks, threads>>>(u, v, w, n, best, bestw);
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < niter; ++i)
        dummy_match<<<blocks, threads>>>(u, v, w, n, best, bestw);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms;
}

static float bench_fused(const uint32_t* u, const uint32_t* v, const float* w,
    int n, uint32_t* best, float* bestw, int niter, int warmup)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dummy_match_fused<<<blocks, threads>>>(u, v, w, n, warmup, best, bestw);
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    dummy_match_fused<<<blocks, threads>>>(u, v, w, n, niter, best, bestw);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms;
}

extern "C" int p0x_microbench(const int32_t* ns, int n_sizes, int niter, float* ms_launch, float* ms_fused)
{
    if (n_sizes <= 0) return 0;
    for (int s = 0; s < n_sizes; ++s) {
        int n = ns[s];
        if (n < 1) return 0;
        std::vector<uint32_t> hu((size_t)n), hv((size_t)n);
        std::vector<float> hw((size_t)n);
        for (int i = 0; i < n; ++i) {
            hu[(size_t)i] = (uint32_t)(i % (n / 2 + 1));
            hv[(size_t)i] = (uint32_t)((i * 7 + 1) % (n / 2 + 1));
            hw[(size_t)i] = (float)((i * 17) % 1000) / 1000.f;
        }
        uint32_t *du, *dv, *dbest;
        float *dw, *dbw;
        cudaMalloc(&du, (size_t)n * 4);
        cudaMalloc(&dv, (size_t)n * 4);
        cudaMalloc(&dw, (size_t)n * 4);
        cudaMalloc(&dbest, (size_t)n * 4);
        cudaMalloc(&dbw, (size_t)n * 4);
        cudaMemcpy(du, hu.data(), (size_t)n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, hv.data(), (size_t)n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, hw.data(), (size_t)n * 4, cudaMemcpyHostToDevice);
        cudaMemset(dbest, 0xff, (size_t)n * 4);
        cudaMemset(dbw, 0, (size_t)n * 4);
        int iters = niter > 0 ? niter : 200;
        ms_launch[s] = bench_launched(du, dv, dw, n, dbest, dbw, iters, 10) / (float)iters;
        ms_fused[s] = bench_fused(du, dv, dw, n, dbest, dbw, iters, 10) / (float)iters;
        std::fprintf(stderr,
            "P0x_BENCH n=%d launch_ms=%.6f fused_ms=%.6f niter=%d\n",
            n, ms_launch[s], ms_fused[s], iters);
        cudaFree(du);
        cudaFree(dv);
        cudaFree(dw);
        cudaFree(dbest);
        cudaFree(dbw);
    }
    return 1;
}
