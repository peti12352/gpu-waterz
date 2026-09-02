// GPU S1 watershed: flow + wavefront plateau rewrite + UF basins.
// uint8 aff [3,Z,Y,X] stays on device. bits stored as uint8.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <map>

// Device allocation accounting.
//
// TASK's graded volume is 2.16 Gvox on a 24 GB 3090 Ti, and the footprint
// cannot be measured there directly because it does not fit. Routing every
// allocation in this file through a counter gives an exact peak at a size that
// does fit, from which the per-voxel coefficients, and therefore the larger
// volumes, follow. Counting beats hand-tracing the allocation list: these
// functions call each other and free at different depths, so the peak is not
// obvious by inspection.
//
// The macros are defined after all includes, so header and template code is
// already parsed and unaffected. CUB is always called here with explicit temp
// storage, so it performs no hidden allocations of its own.
static size_t g_mem_cur = 0;
static size_t g_mem_peak = 0;
static std::map<void*, size_t>& mem_book() {
    static std::map<void*, size_t> m;
    return m;
}

static cudaError_t ws_tracked_malloc(void** p, size_t n) {
    cudaError_t e = cudaMalloc(p, n);
    if (e == cudaSuccess && *p) {
        mem_book()[*p] = n;
        g_mem_cur += n;
        if (g_mem_cur > g_mem_peak) g_mem_peak = g_mem_cur;
    }
    return e;
}

static cudaError_t ws_tracked_free(void* p) {
    auto it = mem_book().find(p);
    if (it != mem_book().end()) {
        g_mem_cur -= it->second;
        mem_book().erase(it);
    }
    return cudaFree(p);
}

extern "C" void ws_mem_reset(void) { g_mem_peak = g_mem_cur; }
extern "C" size_t ws_mem_peak(void) { return g_mem_peak; }
extern "C" size_t ws_mem_cur(void) { return g_mem_cur; }

// A total peak says how much is needed but not which point in the pipeline
// demands it, which is what decides where to shorten a buffer's lifetime.
// WATERZ_WS_MEMLOG=1 prints live and peak bytes at each labelled checkpoint.
static void ws_mem_mark(const char* label) {
    static int on = -1;
    if (on < 0) on = getenv("WATERZ_WS_MEMLOG") != nullptr;
    if (!on) return;
    fprintf(stderr, "WSMEM %-22s cur=%8.3f GiB peak=%8.3f GiB\n", label,
            g_mem_cur / 1073741824.0, g_mem_peak / 1073741824.0);
}

#define cudaMalloc(p, n) ws_tracked_malloc((void**)(p), (n))
#define cudaFree(p) ws_tracked_free((void*)(p))

#ifndef SENT
#define SENT 0xffffffffu
#endif

__device__ __constant__ uint32_t DBIT[6] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20};
__device__ __constant__ uint32_t RBIT[6] = {0x08, 0x10, 0x20, 0x01, 0x02, 0x04};

__device__ inline int64_t neigh_i(int64_t i, int d, int64_t Y, int64_t X) {
    int64_t yx = Y * X;
    if (d == 0) return i - yx;
    if (d == 1) return i - X;
    if (d == 2) return i - 1;
    if (d == 3) return i + yx;
    if (d == 4) return i + X;
    return i + 1;
}

__device__ inline bool oob_d(int d, int64_t z, int64_t y, int64_t x,
                             int64_t Z, int64_t Y, int64_t X) {
    if (d == 0) return z == 0;
    if (d == 1) return y == 0;
    if (d == 2) return x == 0;
    if (d == 3) return z == Z - 1;
    if (d == 4) return y == Y - 1;
    return x == X - 1;
}

__global__ void k_flow(
    const uint8_t* aff, int64_t Z, int64_t Y, int64_t X,
    float low, float high, uint8_t* bits)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    auto aat = [&](int c, int64_t zz, int64_t yy, int64_t xx) -> float {
        return aff[((c * Z + zz) * Y + yy) * X + xx] * (1.0f / 255.0f);
    };
    float nz = (z > 0) ? aat(0, z, y, x) : low;
    float ny = (y > 0) ? aat(1, z, y, x) : low;
    float nx = (x > 0) ? aat(2, z, y, x) : low;
    float pz = (z < Z - 1) ? aat(0, z + 1, y, x) : low;
    float py = (y < Y - 1) ? aat(1, z, y + 1, x) : low;
    float px = (x < X - 1) ? aat(2, z, y, x + 1) : low;
    float m = fmaxf(fmaxf(fmaxf(nx, ny), nz), fmaxf(fmaxf(px, py), pz));
    uint8_t id = 0;
    if (m > low) {
        if (nz == m || nz >= high) id |= 0x01;
        if (ny == m || ny >= high) id |= 0x02;
        if (nx == m || nx >= high) id |= 0x04;
        if (pz == m || pz >= high) id |= 0x08;
        if (py == m || py >= high) id |= 0x10;
        if (px == m || px >= high) id |= 0x20;
    }
    bits[i] = id;
}

__global__ void k_mark_corners(
    uint8_t* bits, int64_t Z, int64_t Y, int64_t X, int* changed)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    uint8_t b = bits[i];
    if (b == 0 || (b & 0x40)) return;
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        if (!(bits[j] & RBIT[d])) {
            bits[i] = (uint8_t)(b | 0x40);
            return;
        }
    }
}

__global__ void k_spread(
    uint8_t* bits, uint32_t* reach, int64_t Z, int64_t Y, int64_t X, int* changed)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    uint8_t b = bits[i];
    if (b == 0 || (b & 0x40)) return;
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    int64_t best = -1;
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        if ((bits[j] & RBIT[d]) && (bits[j] & 0x40)) {
            if (best < 0 || j < best) best = j;
        }
    }
    if (best >= 0) {
        bits[i] = (uint8_t)(b | 0x40);
        reach[i] = (uint32_t)best;
        *changed = 1;
    }
}

__global__ void k_rewrite(
    const uint8_t* bits_in, uint8_t* bits_out, int64_t Z, int64_t Y, int64_t X)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    uint8_t b = bits_in[i];
    if (b == 0) {
        bits_out[i] = 0;
        return;
    }
    if (!(b & 0x40)) {
        bits_out[i] = (uint8_t)(b & 0x3f);
        return;
    }
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    uint8_t exit_dir = 0;
    uint8_t toward = 0;
    int64_t best_j = (int64_t)1 << 62;
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        if (!(bits_in[j] & RBIT[d])) {
            exit_dir = (uint8_t)DBIT[d];
        } else if (toward == 0 || j < best_j) {
            toward = (uint8_t)DBIT[d];
            best_j = j;
        }
    }
    bits_out[i] = exit_dir ? exit_dir : toward;
}

__device__ uint32_t uf_find(uint32_t* p, uint32_t x) {
    if (x == SENT) return SENT;
    uint32_t r = x;
    for (int k = 0; k < 4096; ++k) {
        uint32_t n = p[r];
        if (n == r || n == SENT) {
            if (n == SENT) return SENT;
            break;
        }
        r = n;
    }
    uint32_t y = x;
    for (int k = 0; k < 4096 && y != r && y != SENT; ++k) {
        uint32_t n = p[y];
        p[y] = r;
        y = n;
    }
    return r;
}

__device__ void uf_unite(uint32_t* p, uint32_t a, uint32_t b) {
    for (int it = 0; it < 64; ++it) {
        a = uf_find(p, a);
        b = uf_find(p, b);
        if (a == b) return;
        if (b == SENT || (a != SENT && a > b)) {
            uint32_t t = a;
            a = b;
            b = t;
        }
        if (b == SENT) return;
        uint32_t old = atomicCAS(&p[b], b, a);
        if (old == b) return;
    }
}

__global__ void k_set_parent(
    uint32_t* parent, const uint8_t* bits, int64_t Z, int64_t Y, int64_t X)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    uint8_t b = bits[i];
    if (b == 0) {
        parent[i] = SENT;
        return;
    }
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    int64_t dest = i;
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        dest = neigh_i(i, d, Y, X);
    }
    if (dest == i || bits[dest] == 0) parent[i] = (uint32_t)i;
    else parent[i] = (uint32_t)dest;
}

__global__ void k_apply_reach(uint32_t* parent, const uint32_t* reach, const uint8_t* bits, int64_t size) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    if (bits[i] == 0) return;
    uint32_t r = reach[i];
    if (r != SENT) parent[i] = r;
}

__global__ void k_break_cycles(uint32_t* parent, int64_t size) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    uint32_t p = parent[i];
    if (p == SENT || p == (uint32_t)i) return;
    if (parent[p] == (uint32_t)i) {
        uint32_t r = p < (uint32_t)i ? p : (uint32_t)i;
        parent[i] = r;
        parent[p] = r;
    }
}

__global__ void k_jump_once(uint32_t* parent, int64_t size) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    uint32_t p = parent[i];
    if (p == SENT || p == (uint32_t)i) return;
    uint32_t pp = parent[p];
    if (pp != SENT) parent[i] = pp;
}

__global__ void k_mark_roots(const uint32_t* parent, uint32_t* flags, int64_t size) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    uint32_t p = parent[i];
    flags[i] = (p == (uint32_t)i) ? 1u : 0u;
}

__global__ void k_scatter_ids(
    const uint32_t* parent, const uint32_t* flags, const uint32_t* psum,
    uint32_t* seg, int64_t size)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    uint32_t p = parent[i];
    if (p == SENT) {
        seg[i] = 0;
        return;
    }
    seg[i] = psum[p] + 1u;
}

#include <cub/cub.cuh>

__global__ void k_corner_flag(const uint8_t* bits, uint32_t* flag,
    int64_t Z, int64_t Y, int64_t X);
__global__ void k_scatter_idx(const uint32_t* flag, const uint32_t* psum,
    int64_t* idx, int64_t size);
__global__ void k_mark_vis_frontier(const int64_t* front, int n, unsigned int* vis);
__global__ void k_expand_atomic(
    const int64_t* front, int n, const uint8_t* orig, unsigned int* vis,
    int64_t* next, int* nnext, int64_t Y, int64_t X);
__global__ void k_rewrite_visited(
    const uint8_t* orig, const unsigned int* vis, uint8_t* out,
    int64_t Z, int64_t Y, int64_t X);

__device__ inline uint32_t uf_find_ro(const uint32_t* p, uint32_t x) {
    while (p[x] != x) x = p[x];
    return x;
}

__device__ inline void uf_hook(uint32_t* p, uint32_t a, uint32_t b) {
    while (true) {
        a = uf_find_ro(p, a);
        b = uf_find_ro(p, b);
        if (a == b) return;
        uint32_t lo = a < b ? a : b;
        uint32_t hi = a < b ? b : a;
        uint32_t old = atomicCAS(&p[hi], hi, lo);
        if (old == hi) return;
    }
}

__global__ void k_uf_init(uint32_t* p, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = (uint32_t)i;
}

__global__ void k_uf_link(
    uint32_t* p, const uint8_t* bits, int64_t Z, int64_t Y, int64_t X)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    uint8_t b = bits[i];
    if (!b) return;
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        uf_hook(p, (uint32_t)i, (uint32_t)neigh_i(i, d, Y, X));
    }
}

__global__ void k_uf_link_zero_orig(
    uint32_t* p, const uint8_t* orig, const uint8_t* rewritten,
    int64_t Z, int64_t Y, int64_t X)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    if (rewritten[i] != 0 || orig[i] == 0) return;
    uint8_t b = orig[i];
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    int64_t dest_any = -1, dest_first = -1, dest_last = -1;
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        dest_any = j;
        if (rewritten[j] != 0 || orig[j] == 0) {
            if (dest_first < 0) dest_first = j;
            dest_last = j;
        }
    }
    if (dest_last >= 0)
        uf_hook(p, (uint32_t)i, (uint32_t)dest_last);
    else if (dest_any >= 0)
        uf_hook(p, (uint32_t)i, (uint32_t)dest_any);
}

__global__ void k_uf_compress(uint32_t* p, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = uf_find(p, (uint32_t)i);
}

__global__ void k_mark_main_root(
    const uint32_t* p, const uint8_t* rewritten, uint8_t* mainroot, int64_t n)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (rewritten[i] != 0) mainroot[p[i]] = 1;
}

__global__ void k_collect_extra_target(
    const uint32_t* p, const uint8_t* orig, const uint8_t* rewritten,
    const uint8_t* mainroot, uint32_t* target,
    int64_t Z, int64_t Y, int64_t X)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    if (orig[i] == 0) return;
    uint32_t r = p[i];
    if (mainroot[r]) return;
    int64_t yx = Y * X;
    int64_t z = i / yx, rr = i % yx, y = rr / X, x = rr % X;
    uint32_t best = 0xffffffffu;
    for (int d = 0; d < 6; ++d) {
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        if (mainroot[p[j]] && (uint32_t)j < best) best = (uint32_t)j;
        int64_t zj = j / yx, rj = j % yx, yj = rj / X, xj = rj % X;
        for (int d2 = 0; d2 < 6; ++d2) {
            if (oob_d(d2, zj, yj, xj, Z, Y, X)) continue;
            int64_t k = neigh_i(j, d2, Y, X);
            if (mainroot[p[k]] && (uint32_t)k < best) best = (uint32_t)k;
        }
    }
    if (best != 0xffffffffu) atomicMin(&target[r], best);
}

__global__ void k_hook_extra(
    uint32_t* p, const uint32_t* target, const uint8_t* mainroot, int64_t n)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (p[i] != (uint32_t)i || mainroot[i]) return;
    if (target[i] != 0xffffffffu) uf_hook(p, (uint32_t)i, target[i]);
}

__global__ void k_count_sz(const uint32_t* p, uint32_t* sz, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&sz[p[i]], 1u);
}

__global__ void k_merge_tiny(
    uint32_t* p, const uint32_t* sz, const uint8_t* orig,
    int64_t Z, int64_t Y, int64_t X)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    uint32_t r = p[i];
    if (r != (uint32_t)i || orig[i] == 0 || sz[r] > 2) return;
    int64_t yx = Y * X;
    int64_t z = i / yx, rr = i % yx, y = rr / X, x = rr % X;
    int64_t best = -1;
    for (int d = 0; d < 6; ++d) {
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        if (orig[j] == 0) continue;
        if (p[j] == r) continue;
        if (best < 0 || j < best) best = j;
    }
    if (best >= 0) uf_hook(p, r, p[best]);
}

__global__ void k_mark_bg_root(
    const uint32_t* p, const uint8_t* bits, uint8_t* bgroot, int64_t n)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (bits[i] == 0) bgroot[p[i]] = 1;
}

__global__ void k_root_flag(
    const uint32_t* p, const uint8_t* bgroot, uint32_t* flag, int64_t n)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    flag[i] = (p[i] == (uint32_t)i && !bgroot[i]) ? 1u : 0u;
}

__global__ void k_apply_labels(
    const uint32_t* p, const uint8_t* bgroot, const uint32_t* psum,
    uint32_t* seg, int64_t n)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t r = p[i];
    if (bgroot[r]) seg[i] = 0;
    else seg[i] = psum[r] + 1u;
}

static int plateau_bfs_parallel(
    const uint8_t* orig_d, uint8_t* out_d,
    int64_t Z, int64_t Y, int64_t X)
{
    int64_t size = Z * Y * X;
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    uint32_t* flag = nullptr;
    uint32_t* psum = nullptr;
    cudaMalloc(&flag, (size_t)size * 4);
    cudaMalloc(&psum, (size_t)size * 4);
    k_corner_flag<<<blocks, threads>>>(orig_d, flag, Z, Y, X);
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, flag, psum, (int)size);
        cudaMalloc(&tmp, tmp_bytes);
        cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, flag, psum, (int)size);
        cudaFree(tmp);
    }
    uint32_t last_f = 0, last_p = 0;
    cudaMemcpy(&last_f, flag + size - 1, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_p, psum + size - 1, 4, cudaMemcpyDeviceToHost);
    int nfront = (int)(last_p + last_f);
    const int cap = (int)std::min(size, (int64_t)1 << 26);
    int64_t* front = nullptr;
    int64_t* next = nullptr;
    cudaMalloc(&front, (size_t)cap * 8);
    cudaMalloc(&next, (size_t)cap * 8);
    if (nfront > cap) nfront = cap;
    if (nfront > 0)
        k_scatter_idx<<<blocks, threads>>>(flag, psum, front, size);
    unsigned int* vis = nullptr;
    int64_t nwords = (size + 31) / 32;
    cudaMalloc(&vis, (size_t)nwords * 4);
    cudaMemset(vis, 0, (size_t)nwords * 4);
    if (nfront > 0) {
        int fb = (nfront + 255) / 256;
        k_mark_vis_frontier<<<fb, 256>>>(front, nfront, vis);
    }
    int* nnext_d = nullptr;
    cudaMalloc(&nnext_d, 4);
    int nit = 0;
    while (nfront > 0 && nit < 4096) {
        int fb = (nfront + 255) / 256;
        cudaMemset(nnext_d, 0, 4);
        k_expand_atomic<<<fb, 256>>>(front, nfront, orig_d, vis, next, nnext_d, Y, X);
        int nnext = 0;
        cudaMemcpy(&nnext, nnext_d, 4, cudaMemcpyDeviceToHost);
        if (nnext > cap) nnext = cap;
        std::swap(front, next);
        nfront = nnext;
        ++nit;
    }
    k_rewrite_visited<<<blocks, threads>>>(orig_d, vis, out_d, Z, Y, X);
    cudaFree(flag);
    cudaFree(psum);
    cudaFree(front);
    cudaFree(next);
    cudaFree(vis);
    cudaFree(nnext_d);
    return nit;
}

static int watershed_device(
    const uint8_t* aff_d, int64_t Z, int64_t Y, int64_t X,
    float low, float high, uint32_t* seg_d)
{
    int64_t size = Z * Y * X;
    uint8_t* bits0 = nullptr;
    uint8_t* bits1 = nullptr;
    uint32_t* parent = nullptr;
    uint32_t* flags = nullptr;
    uint8_t* bgroot = nullptr;
    cudaMalloc(&bits0, (size_t)size);
    cudaMalloc(&bits1, (size_t)size);
    cudaMalloc(&parent, (size_t)size * 4);
    cudaMalloc(&flags, (size_t)size * 4);
    cudaMalloc(&bgroot, (size_t)size);
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    k_flow<<<blocks, threads>>>(aff_d, Z, Y, X, low, high, bits0);
    int nit = plateau_bfs_parallel(bits0, bits1, Z, Y, X);
    k_uf_init<<<blocks, threads>>>(parent, size);
    k_uf_link<<<blocks, threads>>>(parent, bits1, Z, Y, X);
    k_uf_link_zero_orig<<<blocks, threads>>>(parent, bits0, bits1, Z, Y, X);
    k_uf_compress<<<blocks, threads>>>(parent, size);
    k_uf_link<<<blocks, threads>>>(parent, bits1, Z, Y, X);
    k_uf_link_zero_orig<<<blocks, threads>>>(parent, bits0, bits1, Z, Y, X);
    k_uf_compress<<<blocks, threads>>>(parent, size);
    uint8_t* mainroot = nullptr;
    cudaMalloc(&mainroot, (size_t)size);
    cudaMemset(mainroot, 0, (size_t)size);
    k_mark_main_root<<<blocks, threads>>>(parent, bits1, mainroot, size);
    uint32_t* extra_tgt = nullptr;
    cudaMalloc(&extra_tgt, (size_t)size * 4);
    cudaMemset(extra_tgt, 0xff, (size_t)size * 4);
    k_collect_extra_target<<<blocks, threads>>>(parent, bits0, bits1, mainroot, extra_tgt, Z, Y, X);
    k_hook_extra<<<blocks, threads>>>(parent, extra_tgt, mainroot, size);
    k_uf_compress<<<blocks, threads>>>(parent, size);
    uint32_t* sz = nullptr;
    cudaMalloc(&sz, (size_t)size * 4);
    cudaMemset(sz, 0, (size_t)size * 4);
    k_count_sz<<<blocks, threads>>>(parent, sz, size);
    k_merge_tiny<<<blocks, threads>>>(parent, sz, bits0, Z, Y, X);
    k_uf_compress<<<blocks, threads>>>(parent, size);
    cudaFree(sz);
    cudaFree(extra_tgt);
    cudaFree(mainroot);
    cudaMemset(bgroot, 0, (size_t)size);
    k_mark_bg_root<<<blocks, threads>>>(parent, bits0, bgroot, size);
    k_root_flag<<<blocks, threads>>>(parent, bgroot, flags, size);
    uint32_t nfrag = 0;
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, tmp_bytes, flags, (uint32_t*)nullptr, (int)size);
        cudaMalloc(&tmp, tmp_bytes);
        uint32_t* n_d = nullptr;
        cudaMalloc(&n_d, 4);
        cub::DeviceReduce::Sum(tmp, tmp_bytes, flags, n_d, (int)size);
        cudaMemcpy(&nfrag, n_d, 4, cudaMemcpyDeviceToHost);
        cudaFree(tmp);
        cudaFree(n_d);
    }
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, flags, flags, (int)size);
        cudaMalloc(&tmp, tmp_bytes);
        cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, flags, flags, (int)size);
        cudaFree(tmp);
    }
    k_apply_labels<<<blocks, threads>>>(parent, bgroot, flags, seg_d, size);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    fprintf(stderr, "ws device_ms=%.1f nfrag=%u bfs_levels=%d\n", ms, nfrag, nit);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(bits0);
    cudaFree(bits1);
    cudaFree(parent);
    cudaFree(flags);
    cudaFree(bgroot);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "ws cuda: %s\n", cudaGetErrorString(e));
        return -1;
    }
    return (int)nfrag;
}

extern "C" int watershed_gpu_d(
    const uint8_t* aff_d, int64_t Z, int64_t Y, int64_t X,
    float low, float high, uint32_t* seg_d)
{
    return watershed_device(aff_d, Z, Y, X, low, high, seg_d);
}

// Exact S1 plateau+basin on host (G2-locked). GPU flow bits only.
static uint32_t plateau_basins(uint32_t* seg, int64_t Z, int64_t Y, int64_t X) {
    const int64_t size = Z * Y * X;
    const int64_t yx = Y * X;
    const int64_t dir[6] = {-yx, -X, -1, yx, X, 1};
    const uint32_t dirmask[6] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20};
    const uint32_t idirmask[6] = {0x08, 0x10, 0x20, 0x01, 0x02, 0x04};
    std::vector<int64_t> bfs;
    bfs.reserve(size / 8);
    for (int64_t i = 0; i < size; ++i) {
        for (int d = 0; d < 6; ++d) {
            if (seg[i] & dirmask[d]) {
                if (!(seg[i + dir[d]] & idirmask[d])) {
                    seg[i] |= 0x40;
                    bfs.push_back(i);
                    break;
                }
            }
        }
    }
    size_t bi = 0;
    while (bi < bfs.size()) {
        int64_t i = bfs[bi];
        uint32_t to_set = 0;
        for (int d = 0; d < 6; ++d) {
            if (seg[i] & dirmask[d]) {
                if (seg[i + dir[d]] & idirmask[d]) {
                    if (!(seg[i + dir[d]] & 0x40)) {
                        bfs.push_back(i + dir[d]);
                        seg[i + dir[d]] |= 0x40;
                    }
                } else {
                    to_set = dirmask[d];
                }
            }
        }
        seg[i] = to_set;
        ++bi;
    }
    bfs.clear();
    const uint32_t HIGH = 0x80000000u;
    uint32_t next_id = 1;
    for (int64_t i = 0; i < size; ++i) {
        if (seg[i] == 0) {
            seg[i] |= HIGH;
            continue;
        }
        if (!(seg[i] & HIGH) && seg[i]) {
            bfs.push_back(i);
            bi = 0;
            seg[i] |= 0x40;
            while (bi < bfs.size()) {
                int64_t me = bfs[bi];
                for (int d = 0; d < 6; ++d) {
                    if (seg[me] & dirmask[d]) {
                        int64_t him = me + dir[d];
                        if (seg[him] & HIGH) {
                            for (auto it : bfs) seg[it] = seg[him];
                            bfs.clear();
                            d = 6;
                        } else if (!(seg[him] & 0x40)) {
                            seg[him] |= 0x40;
                            bfs.push_back(him);
                        }
                    }
                }
                ++bi;
            }
            if (!bfs.empty()) {
                uint32_t lab = HIGH | next_id;
                for (auto it : bfs) seg[it] = lab;
                ++next_id;
                bfs.clear();
            }
        }
    }
    for (int64_t i = 0; i < size; ++i) seg[i] &= 0x7fffffffu;
    return next_id - 1;
}

extern "C" int watershed_gpu(
    const uint8_t* aff_h, int64_t Z, int64_t Y, int64_t X,
    float low, float high, uint32_t* seg_h)
{
    int64_t size = Z * Y * X;
    uint8_t* aff_d = nullptr;
    uint8_t* bits_d = nullptr;
    cudaMalloc(&aff_d, (size_t)3 * size);
    cudaMalloc(&bits_d, (size_t)size);
    cudaMemcpy(aff_d, aff_h, (size_t)3 * size, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    k_flow<<<blocks, threads>>>(aff_d, Z, Y, X, low, high, bits_d);
    std::vector<uint8_t> bits(size);
    cudaMemcpy(bits.data(), bits_d, (size_t)size, cudaMemcpyDeviceToHost);
    cudaFree(aff_d);
    cudaFree(bits_d);
    for (int64_t i = 0; i < size; ++i) seg_h[i] = bits[i];
    return (int)plateau_basins(seg_h, Z, Y, X);
}

// ---- W1: bit-exact first BFS (host) + compact-frontier BFS (device) ----

static void plateau_bfs_only(uint32_t* seg, int64_t Z, int64_t Y, int64_t X) {
    const int64_t size = Z * Y * X;
    const int64_t yx = Y * X;
    const int64_t dir[6] = {-yx, -X, -1, yx, X, 1};
    const uint32_t dirmask[6] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20};
    const uint32_t idirmask[6] = {0x08, 0x10, 0x20, 0x01, 0x02, 0x04};
    std::vector<int64_t> bfs;
    bfs.reserve(size / 8);
    for (int64_t i = 0; i < size; ++i) {
        for (int d = 0; d < 6; ++d) {
            if (seg[i] & dirmask[d]) {
                if (!(seg[i + dir[d]] & idirmask[d])) {
                    seg[i] |= 0x40;
                    bfs.push_back(i);
                    break;
                }
            }
        }
    }
    size_t bi = 0;
    while (bi < bfs.size()) {
        int64_t i = bfs[bi];
        uint32_t to_set = 0;
        for (int d = 0; d < 6; ++d) {
            if (seg[i] & dirmask[d]) {
                if (seg[i + dir[d]] & idirmask[d]) {
                    if (!(seg[i + dir[d]] & 0x40)) {
                        bfs.push_back(i + dir[d]);
                        seg[i + dir[d]] |= 0x40;
                    }
                } else {
                    to_set = dirmask[d];
                }
            }
        }
        seg[i] = to_set;
        ++bi;
    }
}

__device__ inline bool vis_get(const unsigned int* vis, int64_t i) {
    return (vis[i >> 5] >> (i & 31)) & 1u;
}

__device__ inline bool vis_set(unsigned int* vis, int64_t i) {
    unsigned int bit = 1u << (unsigned)(i & 31);
    unsigned int old = atomicOr(&vis[i >> 5], bit);
    return (old & bit) == 0;
}

__global__ void k_corner_flag(
    const uint8_t* bits, uint32_t* flag,
    int64_t Z, int64_t Y, int64_t X)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    uint8_t b = bits[i];
    if (!b) {
        flag[i] = 0;
        return;
    }
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    uint32_t f = 0;
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        if (!(bits[j] & (uint8_t)RBIT[d])) {
            f = 1;
            break;
        }
    }
    flag[i] = f;
}

__global__ void k_scatter_idx(
    const uint32_t* flag, const uint32_t* psum, int64_t* idx, int64_t size)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    if (flag[i]) idx[psum[i]] = i;
}

__global__ void k_mark_vis_frontier(
    const int64_t* front, int n, unsigned int* vis)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    vis_set(vis, front[t]);
}

__global__ void k_expand_atomic(
    const int64_t* front, int n, const uint8_t* orig, unsigned int* vis,
    int64_t* next, int* nnext, int64_t Y, int64_t X)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int64_t i = front[t];
    uint8_t b = orig[i];
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        int64_t j = neigh_i(i, d, Y, X);
        if (orig[j] & (uint8_t)RBIT[d]) {
            if (vis_set(vis, j)) {
                int pos = atomicAdd(nnext, 1);
                next[pos] = j;
            }
        }
    }
}

__global__ void k_rewrite_visited(
    const uint8_t* orig, const unsigned int* vis, uint8_t* out,
    int64_t Z, int64_t Y, int64_t X)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    if (!vis_get(vis, i)) {
        out[i] = orig[i];
        return;
    }
    uint8_t b = orig[i];
    uint8_t to_set = 0;
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) {
            to_set = (uint8_t)DBIT[d];
            continue;
        }
        int64_t j = neigh_i(i, d, Y, X);
        if (orig[j] & (uint8_t)RBIT[d]) {
            // reciprocal
        } else {
            to_set = (uint8_t)DBIT[d];
        }
    }
    out[i] = to_set;
}

__global__ void k_or40(uint8_t* bits, const int64_t* idx, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    bits[idx[t]] |= 0x40;
}

__global__ void k_bfs_hostlike(
    uint8_t* seg, int64_t* q, int nseed, int64_t Y, int64_t X)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int tail = nseed;
    int bi = 0;
    while (bi < tail) {
        int64_t i = q[bi];
        uint8_t b = seg[i];
        uint8_t to_set = 0;
        for (int d = 0; d < 6; ++d) {
            if (!(b & DBIT[d])) continue;
            int64_t j = neigh_i(i, d, Y, X);
            if (seg[j] & (uint8_t)RBIT[d]) {
                if (!(seg[j] & 0x40)) {
                    q[tail++] = j;
                    seg[j] |= 0x40;
                }
            } else {
                to_set = (uint8_t)DBIT[d];
            }
        }
        seg[i] = to_set;
        ++bi;
    }
}

static int plateau_bfs_device(
    const uint8_t* orig_d, uint8_t* out_d,
    int64_t Z, int64_t Y, int64_t X, float* ms_out)
{
    int64_t size = Z * Y * X;
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    uint32_t* flag = nullptr;
    uint32_t* psum = nullptr;
    cudaMalloc(&flag, (size_t)size * 4);
    cudaMalloc(&psum, (size_t)size * 4);
    k_corner_flag<<<blocks, threads>>>(orig_d, flag, Z, Y, X);
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, flag, psum, (int)size);
        cudaMalloc(&tmp, tmp_bytes);
        cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, flag, psum, (int)size);
        cudaFree(tmp);
    }
    uint32_t last_f = 0, last_p = 0;
    cudaMemcpy(&last_f, flag + size - 1, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_p, psum + size - 1, 4, cudaMemcpyDeviceToHost);
    int nfront = (int)(last_p + last_f);
    int64_t* q = nullptr;
    cudaMalloc(&q, (size_t)size * 8);
    if (nfront > 0)
        k_scatter_idx<<<blocks, threads>>>(flag, psum, q, size);
    cudaMemcpy(out_d, orig_d, (size_t)size, cudaMemcpyDeviceToDevice);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    if (nfront > 0) {
        int fb = (nfront + 255) / 256;
        k_or40<<<fb, 256>>>(out_d, q, nfront);
        k_bfs_hostlike<<<1, 1>>>(out_d, q, nfront, Y, X);
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (ms_out) *ms_out = ms;
    fprintf(stderr, "W1 hostlike_bfs nseed=%d device_ms=%.2f\n", nfront, ms);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(flag);
    cudaFree(psum);
    cudaFree(q);
    return 0;
}

extern "C" int w1_plateau_bfs(
    const uint8_t* aff_h, int64_t Z, int64_t Y, int64_t X,
    float low, float high,
    uint8_t* host_out, uint8_t* gpu_out)
{
    int64_t size = Z * Y * X;
    uint8_t* aff_d = nullptr;
    uint8_t* bits_d = nullptr;
    uint8_t* gpu_d = nullptr;
    cudaMalloc(&aff_d, (size_t)3 * size);
    cudaMalloc(&bits_d, (size_t)size);
    cudaMalloc(&gpu_d, (size_t)size);
    cudaMemcpy(aff_d, aff_h, (size_t)3 * size, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    k_flow<<<blocks, threads>>>(aff_d, Z, Y, X, low, high, bits_d);
    std::vector<uint8_t> bits(size);
    cudaMemcpy(bits.data(), bits_d, (size_t)size, cudaMemcpyDeviceToHost);
    std::vector<uint32_t> host(size);
    for (int64_t i = 0; i < size; ++i) host[i] = bits[i];
    plateau_bfs_only(host.data(), Z, Y, X);
    for (int64_t i = 0; i < size; ++i) host_out[i] = (uint8_t)(host[i] & 0xff);
    float ms = 0;
    plateau_bfs_device(bits_d, gpu_d, Z, Y, X, &ms);
    cudaMemcpy(gpu_out, gpu_d, (size_t)size, cudaMemcpyDeviceToHost);
    cudaFree(aff_d);
    cudaFree(bits_d);
    cudaFree(gpu_d);
    int64_t mism = 0;
    for (int64_t i = 0; i < size; ++i)
        if (host_out[i] != gpu_out[i]) ++mism;
    fprintf(stderr, "W1 mismatch=%lld / %lld\n", (long long)mism, (long long)size);
    return mism == 0 ? 1 : 0;
}

// ---- E9b/E9c: independent-plateau divide + findbasins after divide ----

__global__ void k_parent_init(uint32_t* p, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = (uint32_t)i;
}

// changed lets the caller stop as soon as the plateau union-find has
// converged instead of running a fixed round count. One atomicExch on a
// single word per changed round is immaterial next to the sweep itself.
__global__ void k_hook_bidir(const uint8_t* bits, uint32_t* parent, int* changed,
                             int64_t Z, int64_t Y, int64_t X) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    uint8_t b = bits[i];
    if (!b) return;
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    uint32_t pi = parent[i];
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        if (!(bits[j] & (uint8_t)RBIT[d])) continue;
        uint32_t pj = parent[j];
        if (pi == pj) continue;
        uint32_t old = (pi < pj) ? atomicMin(&parent[pj], pi)
                                 : atomicMin(&parent[pi], pj);
        if (old > (pi < pj ? pi : pj)) atomicExch(changed, 1);
    }
}

__global__ void k_uf_compress_c(uint32_t* p, int* changed, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t was = p[i];
    uint32_t now = uf_find(p, (uint32_t)i);
    p[i] = now;
    if (now != was) atomicExch(changed, 1);
}

__global__ void k_count_v2(
    const uint8_t* bits, const uint32_t* flag, const uint32_t* parent,
    uint32_t* vcount, int64_t Z, int64_t Y, int64_t X)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    uint8_t b = bits[i];
    if (!b) return;
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    int in_plat = flag[i] ? 1 : 0;
    if (!in_plat) {
        for (int d = 0; d < 6; ++d) {
            if (!(b & DBIT[d])) continue;
            if (oob_d(d, z, y, x, Z, Y, X)) continue;
            int64_t j = neigh_i(i, d, Y, X);
            if (bits[j] & (uint8_t)RBIT[d]) {
                in_plat = 1;
                break;
            }
        }
    }
    if (in_plat) atomicAdd(&vcount[parent[i]], 1u);
}

__global__ void k_keys_from_parent(
    const int64_t* idx, const uint32_t* parent, uint32_t* keys, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    keys[t] = parent[idx[t]];
}

// The divide stage stores corner and queue entries as uint32 voxel indices
// rather than int64, which halves its two largest per-corner arrays and the
// radix sort's payload traffic. e9b_divide_d rejects volumes that would not
// fit that range.
__global__ void k_keys_from_parent_u32(
    const uint32_t* idx, const uint32_t* parent, uint32_t* keys, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    keys[t] = parent[idx[t]];
}

__global__ void k_or40_u32(uint8_t* bits, const uint32_t* idx, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    bits[idx[t]] |= 0x40;
}

// An in-place exclusive scan overwrites the corner flags, but they stay
// recoverable: voxel i was a corner iff ps[i+1] > ps[i], with the final voxel
// covered by last_f read before the scan. So one uint32-per-voxel buffer does
// the work of the separate flag and psum arrays.
__global__ void k_scatter_idx_u32(
    const uint32_t* ps, uint32_t last_f, uint32_t* idx, int64_t size)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    uint32_t here = ps[i];
    bool corner = (i + 1 < size) ? (ps[i + 1] > here) : (last_f != 0);
    if (corner) idx[here] = (uint32_t)i;
}

__global__ void k_run_start(const uint32_t* keys, uint32_t* start, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    start[t] = (t == 0 || keys[t] != keys[t - 1]) ? 1u : 0u;
}

__global__ void k_scatter_plat(
    const uint32_t* start, const uint32_t* psum, const uint32_t* keys,
    int* plat_begin, uint32_t* plat_root, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    if (!start[t]) return;
    int p = (int)psum[t];
    plat_begin[p] = t;
    plat_root[p] = keys[t];
}

__global__ void k_plat_meta(
    const int* plat_begin, const uint32_t* plat_root, const uint32_t* vcount,
    int* plat_nseed, uint32_t* qsz, int P, int nC)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int b = plat_begin[p];
    int e = (p + 1 < P) ? plat_begin[p + 1] : nC;
    plat_nseed[p] = e - b;
    // vcount[root] is exactly the number of voxels in the plateau, and the BFS
    // pushes each of them at most once: the seeds are pre-marked 0x40 by k_or40
    // so they cannot be re-pushed, and any j it does push is reciprocally
    // linked to a popped voxel, hence in the same union-find component and
    // itself counted in vcount. So tail <= vcount[root] and no slack is needed.
    // Measured over all 55,032,772 val plateaus: max(tail - vcount) == 0, and
    // the previous vcount + nseed + 8 sizing was 8.84x oversized.
    // vcount[root] >= nseed >= 1 because every seed is flagged, hence counted,
    // so the total is bounded by the voxel count and both qsz and its scan fit
    // in uint32 under the volume guard in e9b_divide_d.
    qsz[p] = vcount[plat_root[p]];
}

// The queue is sized per plateau from vcount, so an undersized qsz would run
// one plateau's BFS into the next one's slot and corrupt the segmentation
// silently. cap/overflow turn that into a loud failure, and qused reports the
// true high-water mark so the sizing can be measured rather than guessed.
__global__ void k_indep_bfs(
    uint8_t* seg, const uint32_t* corners, const int* plat_begin,
    const int* plat_nseed, int64_t* q, const uint32_t* qoff,
    const uint32_t* qsz, int* overflow, int64_t* qused,
    int P, int64_t Y, int64_t X)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int nseed = plat_nseed[p];
    if (nseed <= 0) return;
    int c0 = plat_begin[p];
    int64_t* qp = q + qoff[p];
    int64_t cap = qsz[p];
    int tail = 0;
    for (int s = 0; s < nseed; ++s) {
        if ((int64_t)tail >= cap) {
            atomicExch(overflow, 1);
            if (qused) qused[p] = tail;
            return;
        }
        qp[tail++] = corners[c0 + s];
    }
    int bi = 0;
    while (bi < tail) {
        int64_t i = qp[bi];
        uint8_t b = seg[i];
        uint8_t to_set = 0;
        for (int d = 0; d < 6; ++d) {
            if (!(b & DBIT[d])) continue;
            int64_t j = neigh_i(i, d, Y, X);
            if (seg[j] & (uint8_t)RBIT[d]) {
                if (!(seg[j] & 0x40)) {
                    if ((int64_t)tail >= cap) {
                        atomicExch(overflow, 1);
                        if (qused) qused[p] = tail;
                        return;
                    }
                    qp[tail++] = j;
                    seg[j] |= 0x40;
                }
            } else {
                to_set = (uint8_t)DBIT[d];
            }
        }
        seg[i] = to_set;
        ++bi;
    }
    if (qused) qused[p] = tail;
}

// Diagnostic for the queue-sizing measurement: how much of each plateau's
// allocated slot the BFS actually used. qsz is vcount, so a positive value
// anywhere would mean the tail <= vcount argument is wrong.
__global__ void k_qdiag(
    const uint32_t* qsz, const int64_t* qused, const int* plat_nseed,
    int64_t* over_vc, int P)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    over_vc[p] = qused[p] - (int64_t)qsz[p];
}

static int e9b_divide_d(uint8_t* bits_d, int64_t Z, int64_t Y, int64_t X, float* ms_out) {
    int64_t size = Z * Y * X;
    // Corner and queue entries are uint32 voxel indices. 2.16 Gvox is well
    // inside that range; anything larger must fail loudly, not wrap silently.
    if (size > 4294967295LL) {
        fprintf(stderr, "E9b FATAL volume %lld voxels exceeds uint32 indexing\n",
                (long long)size);
        return -4;
    }
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    uint32_t* parent = nullptr;
    uint32_t* flag = nullptr;
    uint32_t* vcount = nullptr;
    ws_mem_mark("divide/enter");
    cudaMalloc(&parent, (size_t)size * 4);
    cudaMalloc(&flag, (size_t)size * 4);
    cudaMalloc(&vcount, (size_t)size * 4);
    cudaMemset(vcount, 0, (size_t)size * 4);
    ws_mem_mark("divide/uf");
    k_parent_init<<<blocks, threads>>>(parent, size);
    // This loop ran a fixed 40 rounds. Each round sweeps every voxel and its
    // six neighbours twice over, so a round costs tens of GB of traffic and
    // spare rounds are the most expensive kind of idle work in the stage.
    // Plateaus are shallow, so convergence comes far sooner than 40; run until
    // a round changes nothing instead. The bound stays as a safety net, and
    // not converging within it is reported rather than silently accepted,
    // because everything downstream needs parent flattened to true roots:
    // vcount is indexed by parent[i] and an unflattened parent would split a
    // plateau's count across nodes and undersize its BFS queue.
    int* uf_changed = nullptr;
    cudaMalloc(&uf_changed, 4);
    const int uf_cap = 64;
    int uf_rounds = 0;
    cudaEvent_t uev0, uev1;
    cudaEventCreate(&uev0);
    cudaEventCreate(&uev1);
    cudaEventRecord(uev0);
    for (int r = 0; r < uf_cap; ++r) {
        cudaMemset(uf_changed, 0, 4);
        k_hook_bidir<<<blocks, threads>>>(bits_d, parent, uf_changed, Z, Y, X);
        k_uf_compress_c<<<blocks, threads>>>(parent, uf_changed, size);
        int h = 0;
        cudaMemcpy(&h, uf_changed, 4, cudaMemcpyDeviceToHost);
        ++uf_rounds;
        if (!h) break;
    }
    cudaEventRecord(uev1);
    cudaEventSynchronize(uev1);
    float uf_ms = 0;
    cudaEventElapsedTime(&uf_ms, uev0, uev1);
    cudaEventDestroy(uev0);
    cudaEventDestroy(uev1);
    cudaFree(uf_changed);
    fprintf(stderr, "E9b uf_rounds=%d/%d uf_ms=%.2f%s\n", uf_rounds, uf_cap,
            uf_ms, uf_rounds >= uf_cap ? " NOT-CONVERGED" : "");
    k_corner_flag<<<blocks, threads>>>(bits_d, flag, Z, Y, X);
    k_count_v2<<<blocks, threads>>>(bits_d, flag, parent, vcount, Z, Y, X);
    // Capture the last flag before the scan overwrites it, then scan flag into
    // itself: k_scatter_idx_u32 recovers the corner predicate from the scan's
    // own differences, so no second per-voxel array is needed.
    uint32_t last_f = 0, last_p = 0;
    cudaMemcpy(&last_f, flag + size - 1, 4, cudaMemcpyDeviceToHost);
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, flag, flag, (int)size);
        cudaMalloc(&tmp, tmp_bytes);
        cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, flag, flag, (int)size);
        cudaFree(tmp);
    }
    uint32_t* psum = flag;
    cudaMemcpy(&last_p, psum + size - 1, 4, cudaMemcpyDeviceToHost);
    int nC = (int)(last_p + last_f);
    if (nC <= 0) {
        cudaFree(parent);
        cudaFree(flag);
        cudaFree(vcount);
        if (ms_out) *ms_out = 0;
        return 0;
    }
    uint32_t* corners_in = nullptr;
    uint32_t* corners_out = nullptr;
    uint32_t* keys_in = nullptr;
    uint32_t* keys_out = nullptr;
    // Only the two sort *inputs* are needed while parent and flag are still
    // live; the outputs are not touched until the sort itself. Allocating them
    // here would overlap 2 * nC uint32s, 0.49 GiB at val, with the 1.34 GiB of
    // parent and flag for no reason, and that overlap is the pipeline's peak.
    cudaMalloc(&corners_in, (size_t)nC * 4);
    cudaMalloc(&keys_in, (size_t)nC * 4);
    ws_mem_mark("divide/corners");
    k_scatter_idx_u32<<<blocks, threads>>>(psum, last_f, corners_in, size);
    int cb = (nC + 255) / 256;
    k_keys_from_parent_u32<<<cb, 256>>>(corners_in, parent, keys_in, nC);
    // parent and the flag/psum buffer are both dead here and are one uint32
    // per voxel each. Holding them across the 61M-pair radix sort below cost
    // 1.3 GiB of peak for nothing. cudaFree synchronizes, so the two launches
    // above have completed before the storage is released.
    cudaFree(parent);
    parent = nullptr;
    cudaFree(flag);
    flag = nullptr;
    psum = nullptr;
    cudaMalloc(&corners_out, (size_t)nC * 4);
    cudaMalloc(&keys_out, (size_t)nC * 4);
    ws_mem_mark("divide/pre-sort");
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, tmp_bytes, keys_in, keys_out, corners_in, corners_out, nC);
        cudaMalloc(&tmp, tmp_bytes);
        ws_mem_mark("divide/sort-tmp");
        cub::DeviceRadixSort::SortPairs(
            tmp, tmp_bytes, keys_in, keys_out, corners_in, corners_out, nC);
        cudaFree(tmp);
    }
    cudaFree(corners_in);
    corners_in = nullptr;
    cudaFree(keys_in);
    keys_in = nullptr;
    ws_mem_mark("divide/post-sort");
    uint32_t* start = nullptr;
    uint32_t* start_ps = nullptr;
    cudaMalloc(&start, (size_t)nC * 4);
    cudaMalloc(&start_ps, (size_t)nC * 4);
    k_run_start<<<cb, 256>>>(keys_out, start, nC);
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, start, start_ps, nC);
        cudaMalloc(&tmp, tmp_bytes);
        cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, start, start_ps, nC);
        cudaFree(tmp);
    }
    uint32_t sl = 0, sp = 0;
    cudaMemcpy(&sl, start + nC - 1, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&sp, start_ps + nC - 1, 4, cudaMemcpyDeviceToHost);
    int P = (int)(sp + sl);
    int* plat_begin = nullptr;
    int* plat_nseed = nullptr;
    uint32_t* plat_root = nullptr;
    uint32_t* qsz = nullptr;
    uint32_t* qoff = nullptr;
    cudaMalloc(&plat_begin, (size_t)P * 4);
    cudaMalloc(&plat_nseed, (size_t)P * 4);
    cudaMalloc(&plat_root, (size_t)P * 4);
    k_scatter_plat<<<cb, 256>>>(start, start_ps, keys_out, plat_begin, plat_root, nC);
    // keys_out and the run-start arrays are per-corner and dead once the
    // plateau table exists, so they are released before the per-plateau queue
    // arrays are allocated rather than overlapping with them.
    cudaFree(keys_out);
    keys_out = nullptr;
    cudaFree(start);
    start = nullptr;
    cudaFree(start_ps);
    start_ps = nullptr;
    cudaMalloc(&qsz, (size_t)P * 4);
    cudaMalloc(&qoff, (size_t)P * 4);
    int pb = (P + 255) / 256;
    k_plat_meta<<<pb, 256>>>(plat_begin, plat_root, vcount, plat_nseed, qsz, P, nC);
    cudaFree(vcount);
    vcount = nullptr;
    ws_mem_mark("divide/pre-queue");
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, qsz, qoff, P);
        cudaMalloc(&tmp, tmp_bytes);
        cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, qsz, qoff, P);
        cudaFree(tmp);
    }
    uint32_t last_qs = 0, last_qo = 0;
    cudaMemcpy(&last_qs, qsz + P - 1, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_qo, qoff + P - 1, 4, cudaMemcpyDeviceToHost);
    int64_t qtot = (int64_t)last_qo + (int64_t)last_qs;
    if (qtot < 1) qtot = 1;
    int64_t* q = nullptr;
    cudaMalloc(&q, (size_t)qtot * 8);
    int* overflow = nullptr;
    cudaMalloc(&overflow, 4);
    cudaMemset(overflow, 0, 4);
    const bool qdiag = getenv("WATERZ_WS_QDIAG") != nullptr;
    int64_t* qused = nullptr;
    if (qdiag) {
        cudaMalloc(&qused, (size_t)P * 8);
        cudaMemset(qused, 0, (size_t)P * 8);
    }
    ws_mem_mark("divide/bfs");
    k_or40_u32<<<cb, 256>>>(bits_d, corners_out, nC);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    k_indep_bfs<<<pb, 256>>>(bits_d, corners_out, plat_begin, plat_nseed, q,
                             qoff, qsz, overflow, qused, P, Y, X);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (ms_out) *ms_out = ms;
    int ovf = 0;
    cudaMemcpy(&ovf, overflow, 4, cudaMemcpyDeviceToHost);
    fprintf(stderr, "E9b ncorner=%d nplat=%d qtot=%lld bfs_ms=%.2f\n",
            nC, P, (long long)qtot, ms);
    if (qdiag) {
        int64_t* over_vc = nullptr;
        cudaMalloc(&over_vc, (size_t)P * 8);
        k_qdiag<<<pb, 256>>>(qsz, qused, plat_nseed, over_vc, P);
        int64_t* red = nullptr;
        cudaMalloc(&red, 8);
        void* tmp = nullptr;
        size_t tb = 0;
        int64_t h_sum = 0, h_max_over = 0, h_max_used = 0;
        cub::DeviceReduce::Sum(nullptr, tb, qused, red, P);
        cudaMalloc(&tmp, tb);
        cub::DeviceReduce::Sum(tmp, tb, qused, red, P);
        cudaMemcpy(&h_sum, red, 8, cudaMemcpyDeviceToHost);
        cub::DeviceReduce::Max(tmp, tb, over_vc, red, P);
        cudaMemcpy(&h_max_over, red, 8, cudaMemcpyDeviceToHost);
        cub::DeviceReduce::Max(tmp, tb, qused, red, P);
        cudaMemcpy(&h_max_used, red, 8, cudaMemcpyDeviceToHost);
        fprintf(stderr,
                "E9bQ qtot=%lld qused_sum=%lld qused_max=%lld "
                "max(qused-vcount)=%lld waste=%.3f\n",
                (long long)qtot, (long long)h_sum, (long long)h_max_used,
                (long long)h_max_over, (double)qtot / (double)(h_sum ? h_sum : 1));
        cudaFree(tmp);
        cudaFree(red);
        cudaFree(over_vc);
        cudaFree(qused);
    }
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(overflow);
    // Each pointer below is either still live or was nulled when released
    // early, and freeing null is a no-op, so one cleanup serves both exits.
    cudaFree(parent);
    cudaFree(flag);
    cudaFree(vcount);
    cudaFree(corners_in);
    cudaFree(corners_out);
    cudaFree(keys_in);
    cudaFree(keys_out);
    cudaFree(start);
    cudaFree(start_ps);
    cudaFree(plat_begin);
    cudaFree(plat_nseed);
    cudaFree(plat_root);
    cudaFree(qsz);
    cudaFree(qoff);
    cudaFree(q);
    if (ovf) {
        fprintf(stderr, "E9b FATAL plateau BFS queue overflow\n");
        return -3;
    }
    return 0;
}

__global__ void k_hook_remain(const uint8_t* bits, uint32_t* parent, int64_t Z, int64_t Y, int64_t X) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    uint8_t b = bits[i];
    if (!b) return;
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    uint32_t pi = parent[i];
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        uint32_t pj = parent[j];
        if (pi == pj) continue;
        if (pi < pj) atomicMin(&parent[pj], pi);
        else atomicMin(&parent[pi], pj);
    }
}

__global__ void k_root_flag(const uint8_t* bits, const uint32_t* parent, uint32_t* flag, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    flag[i] = (bits[i] && parent[i] == (uint32_t)i) ? 1u : 0u;
}

__global__ void k_write_labels(
    const uint8_t* bits, const uint32_t* parent, const uint32_t* psum,
    uint32_t* seg, int64_t n)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!bits[i]) {
        seg[i] = 0;
        return;
    }
    uint32_t r = parent[i];
    seg[i] = psum[r] + 1u;
}

static int g_sv_rounds = 40;

static int e9c_basins_d(const uint8_t* bits_d, uint32_t* seg_d, int64_t Z, int64_t Y, int64_t X, uint32_t* nfrag) {
    int64_t size = Z * Y * X;
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    uint32_t* parent = nullptr;
    uint32_t* flag = nullptr;
    uint32_t* psum = nullptr;
    cudaMalloc(&parent, (size_t)size * 4);
    cudaMalloc(&flag, (size_t)size * 4);
    cudaMalloc(&psum, (size_t)size * 4);
    k_parent_init<<<blocks, threads>>>(parent, size);
    int nsv = g_sv_rounds;
    if (nsv < 1) nsv = 1;
    if (nsv > 40) nsv = 40;
    for (int r = 0; r < nsv; ++r) {
        k_hook_remain<<<blocks, threads>>>(bits_d, parent, Z, Y, X);
        k_uf_compress<<<blocks, threads>>>(parent, size);
    }
    k_root_flag<<<blocks, threads>>>(bits_d, parent, flag, size);
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, flag, psum, (int)size);
        cudaMalloc(&tmp, tmp_bytes);
        cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, flag, psum, (int)size);
        cudaFree(tmp);
    }
    uint32_t last_f = 0, last_p = 0;
    cudaMemcpy(&last_f, flag + size - 1, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_p, psum + size - 1, 4, cudaMemcpyDeviceToHost);
    uint32_t nf = last_p + last_f;
    k_write_labels<<<blocks, threads>>>(bits_d, parent, psum, seg_d, size);
    if (nfrag) *nfrag = nf;
    cudaFree(parent);
    cudaFree(flag);
    cudaFree(psum);
    return 0;
}

extern "C" int e9b_divide(
    const uint8_t* aff_h, int64_t Z, int64_t Y, int64_t X,
    float low, float high, uint8_t* host_out, uint8_t* gpu_out)
{
    int64_t size = Z * Y * X;
    uint8_t* aff_d = nullptr;
    uint8_t* bits_d = nullptr;
    cudaMalloc(&aff_d, (size_t)3 * size);
    cudaMalloc(&bits_d, (size_t)size);
    cudaMemcpy(aff_d, aff_h, (size_t)3 * size, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    k_flow<<<blocks, threads>>>(aff_d, Z, Y, X, low, high, bits_d);
    std::vector<uint8_t> bits(size);
    cudaMemcpy(bits.data(), bits_d, (size_t)size, cudaMemcpyDeviceToHost);
    std::vector<uint32_t> host(size);
    for (int64_t i = 0; i < size; ++i) host[i] = bits[i];
    plateau_bfs_only(host.data(), Z, Y, X);
    for (int64_t i = 0; i < size; ++i) host_out[i] = (uint8_t)(host[i] & 0xff);
    float ms = 0;
    e9b_divide_d(bits_d, Z, Y, X, &ms);
    cudaMemcpy(gpu_out, bits_d, (size_t)size, cudaMemcpyDeviceToHost);
    cudaFree(aff_d);
    cudaFree(bits_d);
    int64_t mism = 0;
    for (int64_t i = 0; i < size; ++i)
        if (host_out[i] != gpu_out[i]) ++mism;
    fprintf(stderr, "E9b mismatch=%lld / %lld bfs_ms=%.2f\n",
            (long long)mism, (long long)size, ms);
    return mism == 0 ? 1 : 0;
}

extern "C" int e9c_watershed(
    const uint8_t* aff_h, int64_t Z, int64_t Y, int64_t X,
    float low, float high, uint32_t* seg_h)
{
    int64_t size = Z * Y * X;
    uint8_t* aff_d = nullptr;
    uint8_t* bits_d = nullptr;
    uint32_t* seg_d = nullptr;
    cudaMalloc(&aff_d, (size_t)3 * size);
    cudaMalloc(&bits_d, (size_t)size);
    cudaMalloc(&seg_d, (size_t)size * 4);
    cudaMemcpy(aff_d, aff_h, (size_t)3 * size, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    k_flow<<<blocks, threads>>>(aff_d, Z, Y, X, low, high, bits_d);
    float ms = 0;
    e9b_divide_d(bits_d, Z, Y, X, &ms);
    uint32_t nfrag = 0;
    e9c_basins_d(bits_d, seg_d, Z, Y, X, &nfrag);
    cudaMemcpy(seg_h, seg_d, (size_t)size * 4, cudaMemcpyDeviceToHost);
    int64_t bg = 0;
    for (int64_t i = 0; i < size; ++i)
        if (seg_h[i] == 0) ++bg;
    fprintf(stderr, "E9c nfrag=%u bg=%lld divide_ms=%.2f\n",
            nfrag, (long long)bg, ms);
    cudaFree(aff_d);
    cudaFree(bits_d);
    cudaFree(seg_d);
    return (int)nfrag;
}

extern "C" int watershed_gpu_e9(
    const uint8_t* aff_h, int64_t Z, int64_t Y, int64_t X,
    float low, float high, uint32_t* seg_h)
{
    return e9c_watershed(aff_h, Z, Y, X, low, high, seg_h);
}

__global__ void k_extract(
    const uint32_t* seg, const uint32_t* parent, uint32_t* out, int64_t n)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = parent[seg[i]];
}

extern "C" int extract_gpu(
    const uint32_t* seg_h, const uint32_t* parent_h, int64_t n, uint32_t max_id,
    uint32_t* out_h, float* ms_out)
{
    uint32_t* seg_d = nullptr;
    uint32_t* par_d = nullptr;
    uint32_t* out_d = nullptr;
    cudaMalloc(&seg_d, (size_t)n * 4);
    cudaMalloc(&par_d, (size_t)(max_id + 1) * 4);
    cudaMalloc(&out_d, (size_t)n * 4);
    cudaMemcpy(seg_d, seg_h, (size_t)n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(par_d, parent_h, (size_t)(max_id + 1) * 4, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    k_extract<<<blocks, threads>>>(seg_d, par_d, out_d, n);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (ms_out) *ms_out = ms;
    cudaMemcpy(out_h, out_d, (size_t)n * 4, cudaMemcpyDeviceToHost);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(seg_d);
    cudaFree(par_d);
    cudaFree(out_d);
    return 1;
}

extern "C" int extract_gpu_d(
    const uint32_t* seg_d, const uint32_t* parent_d, int64_t n, uint32_t* out_d)
{
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    k_extract<<<blocks, threads>>>(seg_d, parent_d, out_d, n);
    return 1;
}

__global__ void k_count_diff(
    const uint32_t* a, const uint32_t* b, int64_t n, unsigned long long* out)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (a[i] != b[i]) atomicAdd(out, 1ull);
}

__global__ void k_pop_hist(
    const uint8_t* bits, int64_t n,
    unsigned long long* n0, unsigned long long* n1, unsigned long long* n2)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t b = bits[i];
    if (!b) {
        atomicAdd(n0, 1ull);
        return;
    }
    int pc = __popc((unsigned int)b);
    if (pc == 1) atomicAdd(n1, 1ull);
    else atomicAdd(n2, 1ull);
}

extern "C" int p0_ws_diag(
    const uint8_t* aff_h, int64_t Z, int64_t Y, int64_t X,
    float low, float high, int64_t* out12)
{
    int64_t size = Z * Y * X;
    uint8_t* aff_d = nullptr;
    uint8_t* bits_d = nullptr;
    uint32_t* parent = nullptr;
    uint32_t* prev = nullptr;
    unsigned long long* dchg = nullptr;
    cudaMalloc(&aff_d, (size_t)3 * size);
    cudaMalloc(&bits_d, (size_t)size);
    cudaMalloc(&parent, (size_t)size * 4);
    cudaMalloc(&prev, (size_t)size * 4);
    cudaMalloc(&dchg, 8);
    cudaMemcpy(aff_d, aff_h, (size_t)3 * size, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    k_flow<<<blocks, threads>>>(aff_d, Z, Y, X, low, high, bits_d);
    unsigned long long h0 = 0, h1 = 0, h2 = 0;
    unsigned long long *dn0 = nullptr, *dn1 = nullptr, *dn2 = nullptr;
    cudaMalloc(&dn0, 8);
    cudaMalloc(&dn1, 8);
    cudaMalloc(&dn2, 8);
    cudaMemset(dn0, 0, 8);
    cudaMemset(dn1, 0, 8);
    cudaMemset(dn2, 0, 8);
    float dms = 0;
    e9b_divide_d(bits_d, Z, Y, X, &dms);
    k_pop_hist<<<blocks, threads>>>(bits_d, size, dn0, dn1, dn2);
    cudaMemcpy(&h0, dn0, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h1, dn1, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h2, dn2, 8, cudaMemcpyDeviceToHost);
    k_parent_init<<<blocks, threads>>>(parent, size);
    int first_zero = -1;
    for (int r = 0; r < 40; ++r) {
        cudaMemcpy(prev, parent, (size_t)size * 4, cudaMemcpyDeviceToDevice);
        k_hook_remain<<<blocks, threads>>>(bits_d, parent, Z, Y, X);
        k_uf_compress<<<blocks, threads>>>(parent, size);
        cudaMemset(dchg, 0, 8);
        k_count_diff<<<blocks, threads>>>(prev, parent, size, dchg);
        unsigned long long ch = 0;
        cudaMemcpy(&ch, dchg, 8, cudaMemcpyDeviceToHost);
        fprintf(stderr, "P0c basin_sv r=%d change=%llu\n", r, (unsigned long long)ch);
        if (ch == 0 && first_zero < 0) first_zero = r;
    }
    if (out12) {
        out12[0] = (int64_t)h0;
        out12[1] = (int64_t)h1;
        out12[2] = (int64_t)h2;
        out12[3] = (int64_t)first_zero;
        out12[4] = (int64_t)(dms * 1000.0f);
    }
    fprintf(stderr, "P0d bg=%llu unique=%llu multibit=%llu first_zero_sv=%d divide_ms=%.2f\n",
            (unsigned long long)h0, (unsigned long long)h1, (unsigned long long)h2,
            first_zero, dms);
    cudaFree(aff_d);
    cudaFree(bits_d);
    cudaFree(parent);
    cudaFree(prev);
    cudaFree(dchg);
    cudaFree(dn0);
    cudaFree(dn1);
    cudaFree(dn2);
    return 1;
}

extern "C" int watershed_gpu_e9_d(
    const uint8_t* aff_d, int64_t Z, int64_t Y, int64_t X,
    float low, float high, uint32_t* seg_d, uint32_t* nfrag, float* ms_out)
{
    int64_t size = Z * Y * X;
    uint8_t* bits_d = nullptr;
    cudaMalloc(&bits_d, (size_t)size);
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    k_flow<<<blocks, threads>>>(aff_d, Z, Y, X, low, high, bits_d);
    float dms = 0;
    e9b_divide_d(bits_d, Z, Y, X, &dms);
    uint32_t nf = 0;
    e9c_basins_d(bits_d, seg_d, Z, Y, X, &nf);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (ms_out) *ms_out = ms;
    if (nfrag) *nfrag = nf;
    cudaFree(bits_d);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return (int)nf;
}

extern "C" void ws_set_sv_rounds(int n) {
    g_sv_rounds = n;
}
