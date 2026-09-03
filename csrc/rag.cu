// Device atomic-hash RAG: 3 negative dirs, drop bg, S3 mean = sum/count.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <map>
#include <cub/cub.cuh>

#include "vox.cuh"

// Device allocation accounting; see the matching note in ws.cu. The RAG's hash
// table is the single largest buffer in the pipeline at large volumes, so its
// exact size matters for the 24 GB question.
static size_t g_rag_cur = 0;
static size_t g_rag_peak = 0;
static std::map<void*, size_t>& rag_book() {
    static std::map<void*, size_t> m;
    return m;
}

static cudaError_t rag_tracked_malloc(void** p, size_t n) {
    cudaError_t e = cudaMalloc(p, n);
    if (e == cudaSuccess && *p) {
        rag_book()[*p] = n;
        g_rag_cur += n;
        if (g_rag_cur > g_rag_peak) g_rag_peak = g_rag_cur;
    }
    return e;
}

static cudaError_t rag_tracked_free(void* p) {
    auto it = rag_book().find(p);
    if (it != rag_book().end()) {
        g_rag_cur -= it->second;
        rag_book().erase(it);
    }
    return cudaFree(p);
}

extern "C" void rag_mem_reset(void) { g_rag_peak = g_rag_cur; }
extern "C" size_t rag_mem_peak(void) { return g_rag_peak; }
extern "C" size_t rag_mem_cur(void) { return g_rag_cur; }

#define cudaMalloc(p, n) rag_tracked_malloc((void**)(p), (n))
#define cudaFree(p) rag_tracked_free((void*)(p))

// isum accumulates the RAW uint8 affinity bytes, not byte/255 as a float.
//
// Float addition is not associative, so the previous `float sum` accumulated by
// atomicAdd gave a different result whenever two runs interleaved their atomics
// differently: measured 740853 of 7505458 val edges disagreeing between two
// runs on identical input, max drift 5.2e-3. That propagates into every merge
// decision and breaks TASK.md's "same input -> byte-identical labels".
//
// An integer sum is exact, so the result is independent of atomic order. The
// struct stays 16 bytes, so the table costs no extra memory.
//
// Range: isum <= 255 * n. uint32 overflows only past 16.8M faces on a single
// fragment pair; total faces across the whole 2.16 Gvox volume is 3 * nvox =
// 6.5e9 spread over ~90M edges (mean 72), so this cannot be approached. The
// invariant is asserted in k_scatter_edges rather than assumed.
struct Slot {
    uint64_t key;
    uint32_t isum;
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

// Insert a already-summed contribution: `sum` of affinity bytes over `cnt`
// faces sharing one key. Open addressing, linear probing, power-of-two table.
__device__ inline void hash_add_group(Slot* tab, uint64_t cap, uint64_t key,
                                      uint32_t sum, uint32_t cnt) {
    uint64_t h = mix64(key);
    for (uint64_t t = 0; t < 128; ++t) {
        uint64_t s = (h + t) & (cap - 1);
        uint64_t old = atomicCAS((unsigned long long*)&tab[s].key, 0ull, (unsigned long long)key);
        if (old == 0ull || old == key) {
            atomicAdd(&tab[s].isum, sum);
            atomicAdd(&tab[s].n, cnt);
            return;
        }
    }
}

__device__ void hash_add(Slot* tab, uint64_t cap, uint64_t key, uint32_t a) {
    hash_add_group(tab, cap, key, a, 1u);
}

// The same insertion, but summed across the warp first.
//
// A fragment averages ~83 voxels at val, so its surface is far larger than the
// number of distinct neighbours it has: ~84M faces are emitted for ~7.5M
// distinct edges, about 11 atomic sequences per edge. Worse, they arrive
// together. A warp spans 32 consecutive x, so when it crosses a y or z boundary
// every lane is looking at the same pair of sheets and emits the *same* key,
// and 32 atomicAdds to one slot serialise on that slot.
//
// Matching lanes on the key and letting the lowest one add the group's total
// collapses each such burst to a single atomic sequence. This is exact rather
// than approximate: both accumulators are integers, so pre-summing inside the
// warp gives the same total as summing at the slot, and the result stays
// independent of order.
//
// Every lane must reach __match_any_sync, including lanes with nothing to
// emit, which is why the caller passes `valid` instead of returning early and
// why the mask is the full warp rather than __activemask().
__device__ inline void hash_add_warp(Slot* tab, uint64_t cap, uint64_t key,
                                     uint32_t a, bool valid) {
    // A real key is (lo << 32) | hi with both ids nonzero, so 0 is free to mark
    // "nothing to emit" and those lanes group together harmlessly.
    const uint64_t k = valid ? key : 0ull;
    const unsigned peers = __match_any_sync(0xffffffffu, k);
    if (!valid) return;
    // Only lanes of this group reach these, and they all pass the same mask,
    // which is what __reduce_add_sync requires. sm_80 and up; the dev card is
    // sm_120 and the graded 3090 Ti is sm_86.
    const uint32_t gsum = __reduce_add_sync(peers, a);
    const uint32_t gcnt = __reduce_add_sync(peers, 1u);
    const unsigned lane = threadIdx.x & 31u;
    if (__popc(peers & ((1u << lane) - 1u)) != 0) return;  // not the lowest peer
    hash_add_group(tab, cap, key, gsum, gcnt);
}

__global__ void k_hash_faces(
    const uint32_t* seg, const uint8_t* aff,
    int64_t Z, int64_t Y, int64_t X,
    Slot* tab, uint64_t cap)
{
    const Vox v = vox_of(Y, X);
    const int64_t yx = Y * X;
    // Out-of-range lanes stay in the warp rather than returning: the
    // aggregation in hash_add_warp needs all 32 lanes to reach the match.
    const bool ok = v.ok;
    const int64_t i = ok ? v.i : 0;
    const int64_t z = v.z, y = v.y, x = v.x;
    const uint32_t id1 = ok ? seg[i] : 0u;
    auto emit = [&](bool face, uint32_t id2, uint32_t a) {
        const bool valid = face && id1 != 0 && id2 != 0 && id1 != id2;
        const uint32_t lo = id1 < id2 ? id1 : id2;
        const uint32_t hi = id1 < id2 ? id2 : id1;
        const uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;
        hash_add_warp(tab, cap, key, a, valid);
    };
    const bool fz = ok && z > 0;
    const bool fy = ok && y > 0;
    const bool fx = ok && x > 0;
    emit(fz, fz ? seg[i - yx] : 0u,
         fz ? aff[(0 * Z + z) * yx + y * X + x] : 0u);
    emit(fy, fy ? seg[i - X] : 0u,
         fy ? aff[(1 * Z + z) * yx + y * X + x] : 0u);
    emit(fx, fx ? seg[i - 1] : 0u,
         fx ? aff[(2 * Z + z) * yx + y * X + x] : 0u);
}

__global__ void k_count_occ(const Slot* tab, uint64_t cap, uint32_t* flags) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= cap) return;
    flags[i] = (tab[i].key != 0 && tab[i].n > 0) ? 1u : 0u;
}

__global__ void k_scatter_edges(
    const Slot* tab, uint64_t cap, const uint32_t* psum,
    uint32_t* u, uint32_t* v, double* sm, int64_t* ct, int* overflow)
{
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= cap) return;
    if (tab[i].key == 0 || tab[i].n == 0) return;
    uint32_t o = psum[i];
    u[o] = (uint32_t)(tab[i].key >> 32);
    v[o] = (uint32_t)tab[i].key;
    ct[o] = (int64_t)tab[i].n;
    // Silent uint32 wraparound would corrupt weights invisibly, so check the
    // isum <= 255*n invariant instead of trusting the range argument.
    if (tab[i].isum > 255u * tab[i].n) atomicExch(overflow, 1);
    sm[o] = (double)tab[i].isum / 255.0;
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
    uint64_t cap = next_pow2((uint64_t)max_edges * 2);
    if (cap < 1024) cap = 1024;
    Slot* tab = nullptr;
    cudaMalloc(&tab, cap * sizeof(Slot));
    cudaMemset(tab, 0, cap * sizeof(Slot));
    int threads = 256;
    k_hash_faces<<<vox_grid(Z, Y, X, threads), threads>>>(
        seg_d, aff_d, Z, Y, X, tab, cap);
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
    int* ovf = nullptr;
    cudaMalloc(&ovf, 4);
    cudaMemset(ovf, 0, 4);
    k_scatter_edges<<<b2, threads>>>(tab, cap, flags, u_d, v_d, sm_d, ct_d, ovf);
    int h_ovf = 0;
    cudaMemcpy(&h_ovf, ovf, 4, cudaMemcpyDeviceToHost);
    cudaFree(ovf);
    cudaFree(tab);
    cudaFree(flags);
    if (h_ovf) {
        std::fprintf(stderr, "RAG_ISUM_OVERFLOW: a contact sum exceeded 255*n\n");
        return -2;
    }
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

// Device time of the last rag_gpu call. Exposed as a separate reader rather
// than an out-parameter so the existing callers' signatures stay put.
static float g_rag_last_ms = 0.0f;

extern "C" float rag_last_ms() { return g_rag_last_ms; }

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
    // Device time around the kernels only. Wall clock here would be dominated
    // by the 540 MB of affinity this entry point copies in, which the graded
    // path does not do, so it would hide whatever the kernels did.
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    int64_t n = rag_device(aff_d, seg_d, Z, Y, X, u_d, v_d, sm_d, ct_d, max_edges);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&g_rag_last_ms, ev0, ev1);
    fprintf(stderr, "RAG device_ms=%.2f nedge=%lld\n", g_rag_last_ms,
            (long long)n);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
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
