// Device atomic-hash RAG: 3 negative dirs, drop bg, S3 mean = sum/count.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cub/cub.cuh>

struct Slot {
    uint64_t key;
    float sum;
    uint32_t n;
};

__device__ inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

__device__ void hash_add(Slot* tab, uint64_t cap, uint64_t key, float a) {
    uint64_t h = mix64(key);
    for (uint64_t t = 0; t < 128; ++t) {
        uint64_t s = (h + t) & (cap - 1);
        uint64_t old = atomicCAS((unsigned long long*)&tab[s].key, 0ull, (unsigned long long)key);
        if (old == 0ull || old == key) {
            atomicAdd(&tab[s].sum, a);
            atomicAdd(&tab[s].n, 1u);
            return;
        }
    }
}

__global__ void k_hash_faces(
    const uint32_t* seg, const uint8_t* aff,
    int64_t Z, int64_t Y, int64_t X,
    Slot* tab, uint64_t cap)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    uint32_t id1 = seg[i];
    auto emit = [&](uint32_t id2, float a) {
        if (id1 == 0 || id2 == 0 || id1 == id2) return;
        uint32_t lo = id1 < id2 ? id1 : id2;
        uint32_t hi = id1 < id2 ? id2 : id1;
        uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;
        hash_add(tab, cap, key, a);
    };
    if (z > 0)
        emit(seg[i - yx], aff[(0 * Z + z) * yx + y * X + x] * (1.0f / 255.0f));
    if (y > 0)
        emit(seg[i - X], aff[(1 * Z + z) * yx + y * X + x] * (1.0f / 255.0f));
    if (x > 0)
        emit(seg[i - 1], aff[(2 * Z + z) * yx + y * X + x] * (1.0f / 255.0f));
}

__global__ void k_count_occ(const Slot* tab, uint64_t cap, uint32_t* flags) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= cap) return;
    flags[i] = (tab[i].key != 0 && tab[i].n > 0) ? 1u : 0u;
}

__global__ void k_scatter_edges(
    const Slot* tab, uint64_t cap, const uint32_t* psum,
    uint32_t* u, uint32_t* v, double* sm, int64_t* ct)
{
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= cap) return;
    if (tab[i].key == 0 || tab[i].n == 0) return;
    uint32_t o = psum[i];
    u[o] = (uint32_t)(tab[i].key >> 32);
    v[o] = (uint32_t)tab[i].key;
    ct[o] = (int64_t)tab[i].n;
    sm[o] = (double)tab[i].sum;
}

static uint64_t next_pow2(uint64_t x) {
    uint64_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

static int64_t rag_device(
    const uint8_t* aff_d, const uint32_t* seg_d,
    int64_t Z, int64_t Y, int64_t X,
    uint32_t* u_d, uint32_t* v_d, double* sm_d, int64_t* ct_d,
    int64_t max_edges)
{
    int64_t size = Z * Y * X;
    uint64_t cap = next_pow2((uint64_t)max_edges * 2);
    if (cap < 1024) cap = 1024;
    Slot* tab = nullptr;
    cudaMalloc(&tab, cap * sizeof(Slot));
    cudaMemset(tab, 0, cap * sizeof(Slot));
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    k_hash_faces<<<blocks, threads>>>(seg_d, aff_d, Z, Y, X, tab, cap);
    uint32_t* flags = nullptr;
    cudaMalloc(&flags, cap * 4);
    int b2 = (int)((cap + threads - 1) / threads);
    k_count_occ<<<b2, threads>>>(tab, cap, flags);
    uint32_t n = 0;
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, tmp_bytes, flags, (uint32_t*)nullptr, (int)cap);
        cudaMalloc(&tmp, tmp_bytes);
        uint32_t* n_d = nullptr;
        cudaMalloc(&n_d, 4);
        cub::DeviceReduce::Sum(tmp, tmp_bytes, flags, n_d, (int)cap);
        cudaMemcpy(&n, n_d, 4, cudaMemcpyDeviceToHost);
        cudaFree(tmp);
        cudaFree(n_d);
    }
    if ((int64_t)n > max_edges) {
        cudaFree(tab);
        cudaFree(flags);
        return -1;
    }
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, flags, flags, (int)cap);
        cudaMalloc(&tmp, tmp_bytes);
        cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, flags, flags, (int)cap);
        cudaFree(tmp);
    }
    k_scatter_edges<<<b2, threads>>>(tab, cap, flags, u_d, v_d, sm_d, ct_d);
    cudaFree(tab);
    cudaFree(flags);
    return (int64_t)n;
}

extern "C" int64_t rag_gpu_d(
    const uint8_t* aff_d, const uint32_t* seg_d,
    int64_t Z, int64_t Y, int64_t X,
    uint32_t* u_d, uint32_t* v_d, double* sm_d, int64_t* ct_d,
    int64_t max_edges)
{
    return rag_device(aff_d, seg_d, Z, Y, X, u_d, v_d, sm_d, ct_d, max_edges);
}

extern "C" int64_t rag_gpu(
    const uint8_t* aff_h, const uint32_t* seg_h,
    int64_t Z, int64_t Y, int64_t X,
    uint32_t* u_out, uint32_t* v_out, double* sum_out, int64_t* count_out,
    int64_t max_edges)
{
    int64_t size = Z * Y * X;
    uint8_t* aff_d = nullptr;
    uint32_t* seg_d = nullptr;
    uint32_t *u_d = nullptr, *v_d = nullptr;
    double* sm_d = nullptr;
    int64_t* ct_d = nullptr;
    cudaMalloc(&aff_d, (size_t)3 * size);
    cudaMalloc(&seg_d, (size_t)size * 4);
    cudaMalloc(&u_d, (size_t)max_edges * 4);
    cudaMalloc(&v_d, (size_t)max_edges * 4);
    cudaMalloc(&sm_d, (size_t)max_edges * 8);
    cudaMalloc(&ct_d, (size_t)max_edges * 8);
    cudaMemcpy(aff_d, aff_h, (size_t)3 * size, cudaMemcpyHostToDevice);
    cudaMemcpy(seg_d, seg_h, (size_t)size * 4, cudaMemcpyHostToDevice);
    int64_t n = rag_device(aff_d, seg_d, Z, Y, X, u_d, v_d, sm_d, ct_d, max_edges);
    if (n > 0) {
        cudaMemcpy(u_out, u_d, (size_t)n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(v_out, v_d, (size_t)n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(sum_out, sm_d, (size_t)n * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(count_out, ct_d, (size_t)n * 8, cudaMemcpyDeviceToHost);
    }
    cudaFree(aff_d);
    cudaFree(seg_d);
    cudaFree(u_d);
    cudaFree(v_d);
    cudaFree(sm_d);
    cudaFree(ct_d);
    return n;
}
