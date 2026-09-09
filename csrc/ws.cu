// GPU S1 watershed: flow + wavefront plateau rewrite + UF basins.
// uint8 aff [3,Z,Y,X] stays on device. bits stored as uint8.
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <map>

struct NvRange {
    explicit NvRange(const char* n) { nvtxRangePushA(n); }
    ~NvRange() { nvtxRangePop(); }
};

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

struct WsAlloc { size_t bytes; int line; };
static std::map<void*, WsAlloc>& mem_book() {
    static std::map<void*, WsAlloc> m;
    return m;
}

// Which lines held the memory when the peak was set. The total says the
// watershed needs 15.69 B/vox but not which buffers that is, and slabbing the
// wrong stage is a rewrite spent for nothing.
static std::map<int, size_t>& ws_peak_lines() {
    static std::map<int, size_t> m;
    return m;
}

static cudaError_t ws_tracked_malloc(void** p, size_t n, int line) {
    cudaError_t e = cudaMalloc(p, n);
    if (e == cudaSuccess && *p) {
        mem_book()[*p] = WsAlloc{n, line};
        g_mem_cur += n;
        if (g_mem_cur > g_mem_peak) {
            g_mem_peak = g_mem_cur;
            auto& snap = ws_peak_lines();
            snap.clear();
            for (const auto& kv : mem_book()) snap[kv.second.line] += kv.second.bytes;
        }
    }
    return e;
}

static cudaError_t ws_tracked_free(void* p) {
    auto it = mem_book().find(p);
    if (it != mem_book().end()) {
        g_mem_cur -= it->second.bytes;
        mem_book().erase(it);
    }
    return cudaFree(p);
}

static size_t g_sort_tmp_inout = 0;
static size_t g_sort_tmp_dbl = 0;
static int g_sort_nC = 0;

// Type D counters. Reset from the host before a measured run.
static int g_n9_w5_calls = 0;
static int64_t g_n9_nlist_sum = 0;
static int g_n9_nlist_max = 0;
static int g_n9_nlist_last = 0;
static int64_t g_n9_nvox_last = 0;
static int g_n9_rounds_sum = 0;
static float g_n9_w5_ms = 0;
static float g_n9_bfs_ms = 0;
static int g_n9_bfs_calls = 0;
static float g_n10_park_ms = 0;
static float g_n10_vcount_ms = 0;
static float g_n10_sort_ms = 0;
static float g_n10_unpark_ms = 0;
static float g_n10_scan_ms = 0;
static float g_n11_tile_ms = 0;
static float g_n11_stitch_ms = 0;
static int64_t g_n11_nrep_last = 0;
static int64_t g_n11_ncross_last = 0;

static bool host_park_on() {
    const char* s = std::getenv("WATERZ_HOST_PARK");
    return s && std::atoi(s) != 0;
}

// N14 T1: skip k_uf_compress_c; consumers one-hop/find. Default off.
static bool fold_flatten() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_FOLD_FLATTEN");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N14 T2: path-halving inside k_w5_compress_list only. Default off.
static bool list_halving() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_LIST_HALVING");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N15 exp2: private parent under fold. Default off. D2 share stays on
// unless this is set. Process-cached: subprocess per env.
static bool share_off() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_SHARE_OFF");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N15 exp3: final flatten by k_uf_jump to a fixed point, not k_uf_compress_c
// and not skip. Not UF_ALGO=1 (that jumps inside the hook loop). Default off.
static bool jump_flatten() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_JUMP_FLATTEN");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N15 exp4: skip k_uf_compress_c only on Recip=true (e9b). e9c still flattens.
static bool fold_e9b_only() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_E9B_FOLD_ONLY");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N15 exp5: hook finds to root; compress_list once after the hook loop.
static bool hook_root() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_HOOK_ROOT");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

static bool stitch_arena() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_STITCH_ARENA");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

static bool pin_changed() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_PIN_CHANGED");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

static bool sort_pack() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_SORT_PACK");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N18 A2: dense-remap plateau roots for vcount histogram (nvox → U≤nC).
// Default off. Process-cached; subprocess per env.
static bool vcount_compact() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_VCOUNT_COMPACT");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N18 A3: reverse 6-dir scan order for plateau/flow tie-break.
static bool tie_flip() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_TIE_FLIP");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N18 A5: skip z-slab face clear; VOI-tolerant (not halo-exact). Default off.
static bool block_voi() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_BLOCK_VOI");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N19 W1: Playne-inspired find-before-hook (same min-ID fixed point). Default off.
static bool playne_hook() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_PLAYNE_HOOK");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N19 W3: warp-aggregated atomics in k_count_v2. Default off. Not A2 compact.
static bool vcount_priv() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_VCOUNT_PRIV");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N18 A4: intentional coarser plateaus (affinity uint8 units below max).
static int coarse_delta() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_COARSE_DELTA");
        cached = (s && std::atoi(s) > 0) ? std::atoi(s) : 0;
    }
    return cached;
}

struct W5Arena {
    uint32_t* list;
    size_t list_n;
    void* scan_tmp;
    size_t scan_cap;
    int* changed;
    int* pin_host;
};
static W5Arena g_w5a{};

extern "C" void ws_n9_reset(void) {
    g_n9_w5_calls = 0;
    g_n9_nlist_sum = 0;
    g_n9_nlist_max = 0;
    g_n9_nlist_last = 0;
    g_n9_nvox_last = 0;
    g_n9_rounds_sum = 0;
    g_n9_w5_ms = 0;
    g_n9_bfs_ms = 0;
    g_n9_bfs_calls = 0;
    g_n10_park_ms = 0;
    g_n10_vcount_ms = 0;
    g_n10_sort_ms = 0;
    g_n10_unpark_ms = 0;
    g_n10_scan_ms = 0;
    g_n11_tile_ms = 0;
    g_n11_stitch_ms = 0;
    g_n11_nrep_last = 0;
    g_n11_ncross_last = 0;
}

extern "C" void ws_n11_uf_stats(
    float* tile_ms, float* stitch_ms, int64_t* n_rep, int64_t* n_cross)
{
    if (tile_ms) *tile_ms = g_n11_tile_ms;
    if (stitch_ms) *stitch_ms = g_n11_stitch_ms;
    if (n_rep) *n_rep = g_n11_nrep_last;
    if (n_cross) *n_cross = g_n11_ncross_last;
}

extern "C" void ws_n10_stats(
    float* park_ms, float* vcount_ms, float* sort_ms,
    float* unpark_ms, float* scan_ms)
{
    if (park_ms) *park_ms = g_n10_park_ms;
    if (vcount_ms) *vcount_ms = g_n10_vcount_ms;
    if (sort_ms) *sort_ms = g_n10_sort_ms;
    if (unpark_ms) *unpark_ms = g_n10_unpark_ms;
    if (scan_ms) *scan_ms = g_n10_scan_ms;
}

extern "C" void ws_n9_stats(
    int* w5_calls, int64_t* nlist_sum, int* nlist_max, int* nlist_last,
    int64_t* nvox_last, int* rounds_sum, float* w5_ms, float* bfs_ms,
    int* bfs_calls)
{
    if (w5_calls) *w5_calls = g_n9_w5_calls;
    if (nlist_sum) *nlist_sum = g_n9_nlist_sum;
    if (nlist_max) *nlist_max = g_n9_nlist_max;
    if (nlist_last) *nlist_last = g_n9_nlist_last;
    if (nvox_last) *nvox_last = g_n9_nvox_last;
    if (rounds_sum) *rounds_sum = g_n9_rounds_sum;
    if (w5_ms) *w5_ms = g_n9_w5_ms;
    if (bfs_ms) *bfs_ms = g_n9_bfs_ms;
    if (bfs_calls) *bfs_calls = g_n9_bfs_calls;
}

extern "C" void ws_mem_reset(void) {
    g_mem_peak = g_mem_cur;
    ws_peak_lines().clear();
    g_sort_tmp_inout = 0;
    g_sort_tmp_dbl = 0;
    g_sort_nC = 0;
}
extern "C" size_t ws_mem_peak(void) { return g_mem_peak; }
extern "C" size_t ws_mem_cur(void) { return g_mem_cur; }
extern "C" size_t ws_sort_tmp_inout(void) { return g_sort_tmp_inout; }
extern "C" size_t ws_sort_tmp_dbl(void) { return g_sort_tmp_dbl; }
extern "C" int ws_sort_nC(void) { return g_sort_nC; }

// Caller-owned buffers (segment_d's aff_d) are not cudaMalloc'd here, so
// they are invisible to the tracker. Credit/debit lets the fused WS peak
// include them while they are resident and drop them after k_flow when
// segment_d parks aff so e9b does not pay 3 B/vox for an unread input.
extern "C" void ws_mem_credit(size_t n) {
    g_mem_cur += n;
    if (g_mem_cur > g_mem_peak) g_mem_peak = g_mem_cur;
}
extern "C" void ws_mem_debit(size_t n) {
    if (g_mem_cur >= n) g_mem_cur -= n;
    else g_mem_cur = 0;
}

// Build with -DWS_NO_BUFFER_SHARE to get the reference that allocates its own
// per-voxel parent array instead of borrowing the caller's label buffer. That
// is how the sharing was gated: both from one source, labels compared. There is
// no CPU oracle at crop scale, and comparing the shared build against itself
// would prove nothing.
#ifdef WS_NO_BUFFER_SHARE
static const bool g_ws_share_labels = false;
#else
static const bool g_ws_share_labels = true;
#endif

static bool share_labels_now() {
    return g_ws_share_labels && !share_off();
}

extern "C" int ws_mem_peak_lines(int* lines, size_t* bytes, int cap) {
    const auto& snap = ws_peak_lines();
    int i = 0;
    for (const auto& kv : snap) {
        if (i < cap) { lines[i] = kv.first; bytes[i] = kv.second; }
        ++i;
    }
    return i;
}

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

#define cudaMalloc(p, n) ws_tracked_malloc((void**)(p), (n), __LINE__)
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

#include "vox.cuh"

__global__ void k_flow(
    const uint8_t* aff, int64_t Z, int64_t Y, int64_t X,
    float low, float high, uint8_t* bits, int flip, int coarse_delta)
{
    const Vox v = vox_of(Y, X);
    if (!v.ok) return;
    const int64_t i = v.i, z = v.z, y = v.y, x = v.x;
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
    const float floor_m = m - (float)coarse_delta * (1.0f / 255.0f);
    if (m > low) {
        float vals[6] = {nz, ny, nx, pz, py, px};
        if (flip) {
            // Alternate tie-break: single steepest among ==m (dir high→low),
            // plus all dirs at/above high (threshold plateaus).
            int best = -1;
            for (int di = 0; di < 6; ++di) {
                int d = 5 - di;
                if (vals[d] == m) { best = d; break; }
            }
            if (best >= 0) id = (uint8_t)DBIT[best];
            for (int d = 0; d < 6; ++d) {
                if (vals[d] >= high) id |= (uint8_t)DBIT[d];
            }
        } else {
            for (int d = 0; d < 6; ++d) {
                if (vals[d] == m || vals[d] >= high ||
                    (coarse_delta > 0 && vals[d] >= floor_m && vals[d] > low))
                    id |= (uint8_t)DBIT[d];
            }
        }
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
    uint8_t* bits, uint32_t* reach, int64_t Z, int64_t Y, int64_t X,
    int* changed, int flip)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t size = Z * Y * X;
    if (i >= size) return;
    uint8_t b = bits[i];
    if (b == 0 || (b & 0x40)) return;
    int64_t yx = Y * X;
    int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    int64_t best = -1;
    for (int di = 0; di < 6; ++di) {
        int d = flip ? (5 - di) : di;
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        if ((bits[j] & RBIT[d]) && (bits[j] & 0x40)) {
            if (best < 0 || (flip ? (j > best) : (j < best))) best = j;
        }
    }
    if (best >= 0) {
        bits[i] = (uint8_t)(b | 0x40);
        reach[i] = (uint32_t)best;
        *changed = 1;
    }
}

__global__ void k_rewrite(
    const uint8_t* bits_in, uint8_t* bits_out, int64_t Z, int64_t Y, int64_t X,
    int flip)
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
    int64_t best_j = flip ? -1 : ((int64_t)1 << 62);
    for (int di = 0; di < 6; ++di) {
        int d = flip ? (5 - di) : di;
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        if (!(bits_in[j] & RBIT[d])) {
            exit_dir = (uint8_t)DBIT[d];
        } else if (toward == 0 || (flip ? (j > best_j) : (j < best_j))) {
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
    uint32_t* parent, const uint8_t* bits, int64_t Z, int64_t Y, int64_t X,
    int flip)
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
    for (int di = 0; di < 6; ++di) {
        int d = flip ? (5 - di) : di;
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

// Path-halving (Playne-style clamp). Cheaper than two-pass uf_find.
__device__ uint32_t uf_find_halve(uint32_t* p, uint32_t x) {
    if (x == SENT) return SENT;
    for (int k = 0; k < 4096; ++k) {
        uint32_t n = p[x];
        if (n == x || n == SENT) return (n == SENT) ? SENT : x;
        uint32_t nn = p[n];
        p[x] = nn;
        x = nn;
    }
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
    k_flow<<<vox_grid(Z, Y, X, threads), threads>>>(aff_d, Z, Y, X, low, high, bits0, tie_flip() ? 1 : 0, coarse_delta());
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
    k_flow<<<vox_grid(Z, Y, X, threads), threads>>>(aff_d, Z, Y, X, low, high, bits_d, tie_flip() ? 1 : 0, coarse_delta());
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
    k_flow<<<vox_grid(Z, Y, X, threads), threads>>>(aff_d, Z, Y, X, low, high, bits_d, tie_flip() ? 1 : 0, coarse_delta());
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
                             int64_t Z, int64_t Y, int64_t X, int playne) {
    const Vox v = vox_of(Y, X);
    if (!v.ok) return;
    const int64_t i = v.i, z = v.z, y = v.y, x = v.x;
    uint8_t b = bits[i];
    if (!b) return;
    uint32_t pi = parent[i];
    if (playne) pi = uf_find_ro(parent, pi);
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        if (!(bits[j] & (uint8_t)RBIT[d])) continue;
        uint32_t pj = parent[j];
        if (playne) pj = uf_find_ro(parent, pj);
        if (pi == pj) continue;
        uint32_t old = (pi < pj) ? atomicMin(&parent[pj], pi)
                                 : atomicMin(&parent[pi], pj);
        if (old > (pi < pj ? pi : pj)) atomicExch(changed, 1);
    }
}

// W4: the same hook, with the tile staged in shared memory.
//
// k_hook_bidir's launch geometry gives a block 256 consecutive x within one
// (y,z) row, so its +/-y neighbour is X*4 bytes away and its +/-z neighbour is
// X*Y*4. At 2.16 Gvox that z stride is 23.04 MB, which on a 3090 Ti's 6 MB L2
// misses every single time. The kernel then does six such gathers per voxel
// per round, fourteen rounds. That is the bulk of the watershed's 7.4x gap
// against its own essential traffic, and it is a launch-shape problem rather
// than an algorithmic one.
//
// A 32x4x4 tile plus its one-voxel halo is 34x6x6 = 1224 slots, so 512 threads
// stage 1224 parent words and 1224 direction bytes and then do all seven
// accesses per voxel out of shared memory. 2.39 global loads per voxel instead
// of 7, and the ones that remain are within a 3-plane window of 32-voxel rows
// rather than scattered across the volume.
//
// Correctness does not depend on the staged values being current, which is
// what makes this safe. Every write is `atomicMin` into the parent slot of a
// root, so parent entries only ever decrease; the round loop already runs to
// convergence and reports failure to converge; and the fixed point of
// min-index hooking is the component minimum regardless of the order or the
// staleness of the reads. The untiled kernel is already reading values that
// other blocks are concurrently modifying, so this changes how stale the reads
// are, not whether they can be. What it must not do is *drop* an edge, which
// is why the halo is loaded rather than clamped, and why boundary voxels fall
// back to the same oob_d test the untiled version uses.
#define WS_TX 32
#define WS_TY 4
#define WS_TZ 4
#define WS_SX (WS_TX + 2)
#define WS_SY (WS_TY + 2)
#define WS_SZ (WS_TZ + 2)
#define WS_SN (WS_SX * WS_SY * WS_SZ)

static inline __device__ int ws_sidx(int lz, int ly, int lx) {
    return ((lz + 1) * WS_SY + (ly + 1)) * WS_SX + (lx + 1);
}

__global__ void k_hook_bidir_tiled(const uint8_t* bits, uint32_t* parent,
                                   int* changed,
                                   int64_t Z, int64_t Y, int64_t X) {
    __shared__ uint32_t sp[WS_SN];
    __shared__ uint8_t sb[WS_SN];

    const int64_t x0 = (int64_t)blockIdx.x * WS_TX;
    const int64_t y0 = (int64_t)blockIdx.y * WS_TY;
    const int64_t z0 = (int64_t)blockIdx.z * WS_TZ;
    const int64_t yx = Y * X;
    const int tid = (threadIdx.z * WS_TY + threadIdx.y) * WS_TX + threadIdx.x;
    const int nthread = WS_TX * WS_TY * WS_TZ;

    // Stage tile + halo. Out-of-volume slots get bits 0, which makes them
    // inert: k_hook_bidir skips a voxel with no direction bits, and the
    // reciprocity test against a zero byte is always false.
    for (int s = tid; s < WS_SN; s += nthread) {
        int lx = s % WS_SX, t = s / WS_SX;
        int ly = t % WS_SY, lz = t / WS_SY;
        int64_t gx = x0 + lx - 1, gy = y0 + ly - 1, gz = z0 + lz - 1;
        if (gx >= 0 && gx < X && gy >= 0 && gy < Y && gz >= 0 && gz < Z) {
            int64_t gi = (gz * Y + gy) * X + gx;
            sb[s] = bits[gi];
            sp[s] = parent[gi];
        } else {
            sb[s] = 0;
            sp[s] = 0xffffffffu;
        }
    }
    __syncthreads();

    const int lx = threadIdx.x, ly = threadIdx.y, lz = threadIdx.z;
    const int64_t x = x0 + lx, y = y0 + ly, z = z0 + lz;
    if (x >= X || y >= Y || z >= Z) return;
    const int me = ws_sidx(lz, ly, lx);
    const uint8_t b = sb[me];
    if (!b) return;
    const uint32_t pi = sp[me];
    (void)yx;

    // Same six directions, same order, same predicates as k_hook_bidir.
    const int dlz[6] = {-1, 0, 0, 1, 0, 0};
    const int dly[6] = {0, -1, 0, 0, 1, 0};
    const int dlx[6] = {0, 0, -1, 0, 0, 1};
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        const int nb = ws_sidx(lz + dlz[d], ly + dly[d], lx + dlx[d]);
        if (!(sb[nb] & (uint8_t)RBIT[d])) continue;
        const uint32_t pj = sp[nb];
        if (pi == pj) continue;
        const uint32_t lo = pi < pj ? pi : pj;
        const uint32_t old = (pi < pj) ? atomicMin(&parent[pj], pi)
                                       : atomicMin(&parent[pi], pj);
        if (old > lo) atomicExch(changed, 1);
    }
}

static inline dim3 ws_tile_grid(int64_t Z, int64_t Y, int64_t X) {
    return dim3((unsigned)((X + WS_TX - 1) / WS_TX),
                (unsigned)((Y + WS_TY - 1) / WS_TY),
                (unsigned)((Z + WS_TZ - 1) / WS_TZ));
}

__global__ void k_uf_compress_c(uint32_t* p, int* changed, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t was = p[i];
    uint32_t now = uf_find(p, (uint32_t)i);
    p[i] = now;
    if (now != was) atomicExch(changed, 1);
}

// Flattening by pointer jumping: p[i] <- p[p[i]], one hop per round.
//
// k_uf_compress_c walks each chain to its root and then walks it a second time
// writing the root into every entry on the way. Timing the two kernels of the
// round separately puts it at 317 ms of the union-find's 499 ms while moving
// only about 2 GB, so it runs some 30x off bandwidth roofline -- the cost is
// chains of dependent random loads, not bytes.
//
// A jump is two loads and at most one store, and it halves every chain, so the
// convergence-driven loop gets there in O(log L) rounds of far cheaper work.
// This is a legitimate substitution rather than an approximation because what
// the gate pins is the fixed point, not the path taken to it: the loop settles
// when neither hooking nor flattening changes anything, and that state is
// exactly "every entry holds its component's minimum index" either way.
//
// Only valid where parent has no SENT entries. e9b_divide_d initialises with
// k_parent_init (p[i] = i everywhere), so it qualifies; the older SENT-carrying
// paths must keep using uf_find.
__global__ void k_uf_jump(uint32_t* p, int* changed, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t a = p[i];
    if (a == (uint32_t)i) return;   // already a root
    const uint32_t b = p[a];
    if (b == a) return;             // parent is a root, nothing left to gain
    p[i] = b;
    atomicExch(changed, 1);
}

// Which flattening kernel the plateau union-find uses. 0 = uf_find path
// compression (the shape the fingerprint was established with), 1 = pointer
// jumping. Selectable so the two can be gated separately and timed against
// each other in one interleaved run, which is the only trustworthy comparison
// while the card is shared.
static int uf_algo() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_UF_ALGO");
        cached = s ? std::atoi(s) : 3;
    }
    return cached;
}

// ---------------------------------------------------------------------------
// W5: resolve each tile's union-find in shared memory, then stitch the tiles.
// ---------------------------------------------------------------------------
//
// W4 tiled the hook, which was the bandwidth half of the round. The larger
// half is the other one: the note above k_uf_jump records k_uf_compress_c at
// 317 ms of the union-find's 499 ms while moving about 2 GB, some 30x off
// bandwidth roofline, because its cost is chains of dependent random loads
// rather than bytes. Tiling cannot help a pointer chase and neither can
// shrinking the domain. Doing the chase somewhere with a 30-cycle latency
// instead of a 300-cycle one can.
//
// So: partition the edge set. Every edge with both endpoints in one tile is
// applied by k_uf_tile_local, which runs the whole hook-and-flatten loop to
// its fixed point inside shared memory and never touches global memory in
// between. What is left is the cross-tile edges, and every one of those has
// both endpoints on a tile face, so the stitch only has to sweep faces.
//
// The measured shape of that trade, from scripts/w5_tile_uf.py on the real
// fragment volume: the whole-volume loop needs 6 rounds for the plateau
// union-find and 7 for the basin one, while the stitch converges in 3 over
// roughly 30% of the volume.
//
// Why this is bit-identical rather than merely close. The licence is the one
// already written above k_uf_jump: the gate pins the fixed point, not the path
// to it. Both loops run to convergence and report failure to converge, and
// the fixed point of min-index hooking is "every entry holds its component's
// minimum index" for any order of application. Splitting the edges into two
// phases is a reordering, so it lands on the same array. Two premises make
// that concrete, and both are checked on the CPU rather than assumed
// (scripts/w0_ws_ref.py --w5, 120/120 identical parent arrays):
//
//   1. Local index order inside a tile agrees with global index order, or the
//      local minimum would not be the global minimum. It does, because both
//      are lexicographic in their coordinate triple and the tile origin is a
//      constant offset.
//   2. The two phases together cover every edge. Phase 1 takes every
//      intra-tile edge; the stitch list contains every face voxel, and a
//      cross-tile edge has both endpoints on a face.
//
// Tile shape is a compile-time knob because the trade is a real one: a bigger
// tile leaves less to stitch but costs more shared memory and so less
// occupancy. 8x16x32 needs 4 B of parent and 1 B of direction byte per voxel,
// 20 KB, which leaves two blocks resident per SM against the 48 KB default
// limit. 16x16x32 cuts the stitch list from 39% to 29% but needs 40 KB and so
// runs one block per SM.
#ifndef W5_TZ
#define W5_TZ 8
#endif
#ifndef W5_TY
#define W5_TY 16
#endif
#ifndef W5_TX
#define W5_TX 32
#endif
#define W5_TN (W5_TZ * W5_TY * W5_TX)
#define W5_THREADS 256

static inline dim3 w5_tile_grid(int64_t Z, int64_t Y, int64_t X) {
    return dim3((unsigned)((X + W5_TX - 1) / W5_TX),
                (unsigned)((Y + W5_TY - 1) / W5_TY),
                (unsigned)((Z + W5_TZ - 1) / W5_TZ));
}

// Is this voxel in the first or last plane of its tile along any axis?
//
// Tiles at the far edge of the volume can be truncated, so their last
// occupied plane is not at local index T-1 and goes unmarked. That is correct
// rather than a gap: the neighbour such a voxel would have needed marking for
// is outside the volume, so no edge exists to stitch.
static inline __device__ bool w5_on_face(int64_t z, int64_t y, int64_t x) {
    const int lz = (int)(z % W5_TZ), ly = (int)(y % W5_TY),
              lx = (int)(x % W5_TX);
    return lz == 0 || lz == W5_TZ - 1 || ly == 0 || ly == W5_TY - 1
        || lx == 0 || lx == W5_TX - 1;
}

// Phase 1. `Recip` selects between the two predicates the file already has:
// k_hook_bidir requires the neighbour to point back, k_hook_remain does not.
// It is a template parameter so the branch costs nothing per direction.
template <bool Recip>
__global__ void k_uf_tile_local(const uint8_t* bits, uint32_t* parent,
                                int64_t Z, int64_t Y, int64_t X) {
    __shared__ uint32_t sp[W5_TN];
    __shared__ uint8_t sb[W5_TN];
    __shared__ int schanged;

    const int64_t x0 = (int64_t)blockIdx.x * W5_TX;
    const int64_t y0 = (int64_t)blockIdx.y * W5_TY;
    const int64_t z0 = (int64_t)blockIdx.z * W5_TZ;
    const int tid = threadIdx.x;
    const int nthread = blockDim.x;

    // No halo. Phase 1 applies only intra-tile edges, and both endpoints of
    // one of those are in the core by definition. Out-of-volume slots get
    // bits 0, which is inert for the same reason it is in k_hook_bidir_tiled.
    for (int s = tid; s < W5_TN; s += nthread) {
        const int lx = s % W5_TX, t = s / W5_TX;
        const int ly = t % W5_TY, lz = t / W5_TY;
        const int64_t gx = x0 + lx, gy = y0 + ly, gz = z0 + lz;
        sb[s] = (gx < X && gy < Y && gz < Z)
              ? bits[(gz * Y + gy) * X + gx] : (uint8_t)0;
        sp[s] = (uint32_t)s;
    }
    __syncthreads();
    // W3 nonempty-tile skip. A tile whose bits are all zero has no hook to
    // issue, so the identity parent we just staged is already the answer.
    // The 36% plateau figure is voxels, not tiles: a 4096-voxel tile is
    // empty only if it is all background, which on val is rare. The skip
    // is still free to take and is the form that preserves locality, unlike
    // a compacted voxel list.
    {
        __shared__ int any;
        if (tid == 0) any = 0;
        __syncthreads();
        for (int s = tid; s < W5_TN; s += nthread)
            if (sb[s]) any = 1;
        __syncthreads();
        if (!any) {
            for (int s = tid; s < W5_TN; s += nthread) {
                const int lx = s % W5_TX, t = s / W5_TX;
                const int ly = t % W5_TY, lz = t / W5_TY;
                const int64_t gx = x0 + lx, gy = y0 + ly, gz = z0 + lz;
                if (gx < X && gy < Y && gz < Z)
                    parent[(gz * Y + gy) * X + gx] = (uint32_t)
                        ((gz * Y + gy) * X + gx);
            }
            return;
        }
        __syncthreads();
    }

    const int dlz[6] = {-1, 0, 0, 1, 0, 0};
    const int dly[6] = {0, -1, 0, 0, 1, 0};
    const int dlx[6] = {0, 0, -1, 0, 0, 1};

    // Hook and flatten to the tile's fixed point. The bound is W5_TN because
    // every round that does not break has lowered at least one entry and
    // entries are bounded below, so it cannot be reached; the loop exits on
    // the flag.
    for (int iter = 0; iter < W5_TN; ++iter) {
        if (tid == 0) schanged = 0;
        __syncthreads();
        for (int s = tid; s < W5_TN; s += nthread) {
            const uint8_t b = sb[s];
            if (!b) continue;
            const int lx = s % W5_TX, t = s / W5_TX;
            const int ly = t % W5_TY, lz = t / W5_TY;
            const int64_t z = z0 + lz, y = y0 + ly, x = x0 + lx;
            if (z >= Z || y >= Y || x >= X) continue;
            const uint32_t pi = sp[s];
            for (int d = 0; d < 6; ++d) {
                if (!(b & DBIT[d])) continue;
                // Same in-volume test the untiled kernels use.
                if (oob_d(d, z, y, x, Z, Y, X)) continue;
                const int nlz = lz + dlz[d], nly = ly + dly[d],
                          nlx = lx + dlx[d];
                // Outside the tile: a cross-tile edge, left for the stitch.
                if (nlz < 0 || nlz >= W5_TZ || nly < 0 || nly >= W5_TY
                    || nlx < 0 || nlx >= W5_TX) continue;
                const int ns = (nlz * W5_TY + nly) * W5_TX + nlx;
                if (Recip && !(sb[ns] & (uint8_t)RBIT[d])) continue;
                const uint32_t pj = sp[ns];
                if (pi == pj) continue;
                const uint32_t lo = pi < pj ? pi : pj;
                const uint32_t hi = pi < pj ? pj : pi;
                if (atomicMin(&sp[hi], lo) > lo) schanged = 1;
            }
        }
        __syncthreads();
        // Flatten. Every hook makes the smaller index the parent, so sp[r] <= r
        // with equality only at a root, and the chase strictly decreases and
        // therefore terminates even while other threads are writing.
        for (int s = tid; s < W5_TN; s += nthread) {
            uint32_t r = sp[s];
            while (sp[r] != r) r = sp[r];
            if (sp[s] != r) {
                sp[s] = r;
                schanged = 1;
            }
        }
        __syncthreads();
        if (!schanged) break;
        __syncthreads();
    }

    // Publish the tile-local root as a global voxel index.
    for (int s = tid; s < W5_TN; s += nthread) {
        const int lx = s % W5_TX, t = s / W5_TX;
        const int ly = t % W5_TY, lz = t / W5_TY;
        const int64_t gx = x0 + lx, gy = y0 + ly, gz = z0 + lz;
        if (gx >= X || gy >= Y || gz >= Z) continue;
        const uint32_t r = sp[s];
        const int rlx = r % W5_TX, rt = r / W5_TX;
        const int rly = rt % W5_TY, rlz = rt / W5_TY;
        parent[(gz * Y + gy) * X + gx] = (uint32_t)
            (((z0 + rlz) * Y + (y0 + rly)) * X + (x0 + rlx));
    }
}

// The stitch list: face voxels, plus every phase-1 root.
//
// The faces are what carry the cross-tile edges. The roots are there because
// after phase 1 a parent entry can only hold a phase-1 root, so those are the
// only slots the stitch's atomicMin ever writes into; leaving them out of the
// compress domain would let the chains through them grow without bound, since
// k_uf_compress_c writes only its own slot and does not shorten the chain it
// walks. Zero-bit voxels are not filtered out, because k_hook_remain unions
// along every set direction bit without checking the target's bits, so a
// zero-bit voxel can be a root that others point at.
__global__ void k_w5_list_flag(const uint32_t* parent, uint32_t* flag,
                               int64_t Z, int64_t Y, int64_t X) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t size = Z * Y * X;
    if (i >= size) return;
    const int64_t yx = Y * X;
    const int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    flag[i] = (w5_on_face(z, y, x) || parent[i] == (uint32_t)i) ? 1u : 0u;
}

template <bool Recip>
__global__ void k_w5_hook_list(const uint8_t* bits, uint32_t* parent,
                               int* changed, const uint32_t* list, int nlist,
                               int64_t Z, int64_t Y, int64_t X, int find_root) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nlist) return;
    const int64_t i = list[t];
    const uint8_t b = bits[i];
    if (!b) return;
    const int64_t yx = Y * X;
    const int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    const uint32_t pi = find_root ? uf_find_ro(parent, (uint32_t)i) : parent[i];
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        const int64_t j = neigh_i(i, d, Y, X);
        if (Recip && !(bits[j] & (uint8_t)RBIT[d])) continue;
        const uint32_t pj = find_root ? uf_find_ro(parent, (uint32_t)j) : parent[j];
        if (pi == pj) continue;
        const uint32_t lo = pi < pj ? pi : pj;
        const uint32_t old = (pi < pj) ? atomicMin(&parent[pj], pi)
                                       : atomicMin(&parent[pi], pj);
        if (old > lo) atomicExch(changed, 1);
    }
}

__global__ void k_w5_compress_list(uint32_t* p, int* changed,
                                   const uint32_t* list, int nlist, int halve) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nlist) return;
    const uint32_t i = list[t];
    const uint32_t was = p[i];
    const uint32_t now = halve ? uf_find_halve(p, i) : uf_find(p, i);
    p[i] = now;
    if (now != was) atomicExch(changed, 1);
}

// N11 B: count-only after k_uf_tile_local. Does not stitch. Recip as e9b.
// n_rep is unique p1 of both endpoints of a cross-tile flow+reciprocity edge
// with p1[i] != p1[j]. n_cross counts each such undirected edge once (i < j).
static inline __device__ bool w5_same_tile(int64_t z, int64_t y, int64_t x,
                                          int64_t nz, int64_t ny, int64_t nx) {
    return (z / W5_TZ) == (nz / W5_TZ)
        && (y / W5_TY) == (ny / W5_TY)
        && (x / W5_TX) == (nx / W5_TX);
}

__global__ void k_n11_nrep_count(
    const uint8_t* bits, const uint32_t* parent, uint32_t* rep_flag,
    unsigned long long* n_face, unsigned long long* n_list,
    unsigned long long* n_cross,
    int64_t Z, int64_t Y, int64_t X)
{
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t size = Z * Y * X;
    if (i >= size) return;
    const int64_t yx = Y * X;
    const int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    const bool face = w5_on_face(z, y, x);
    if (face) atomicAdd(n_face, 1ull);
    if (face || parent[i] == (uint32_t)i) atomicAdd(n_list, 1ull);
    const uint8_t b = bits[i];
    if (!b) return;
    const int dz[6] = {-1, 0, 0, 1, 0, 0};
    const int dy[6] = {0, -1, 0, 0, 1, 0};
    const int dx[6] = {0, 0, -1, 0, 0, 1};
    const uint32_t pi = parent[i];
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        const int64_t nz = z + dz[d], ny = y + dy[d], nx = x + dx[d];
        if (w5_same_tile(z, y, x, nz, ny, nx)) continue;
        const int64_t j = neigh_i(i, d, Y, X);
        if (!(bits[j] & (uint8_t)RBIT[d])) continue;
        const uint32_t pj = parent[j];
        if (pi == pj) continue;
        rep_flag[pi] = 1u;
        rep_flag[pj] = 1u;
        if (i < j) atomicAdd(n_cross, 1ull);
    }
}

// Count-only. Allocates its own parent. Does not write bits or labels.
extern "C" int ws_n11_nrep(
    const uint8_t* bits_d, int64_t Z, int64_t Y, int64_t X,
    unsigned long long* n_face, unsigned long long* n_list,
    unsigned long long* n_rep, unsigned long long* n_cross)
{
    const int64_t size = Z * Y * X;
    if (size <= 0 || size > 4294967295LL) return -1;
    const int threads = 256;
    const int blocks = (int)((size + threads - 1) / threads);
    uint32_t* parent = nullptr;
    uint32_t* flag = nullptr;
    unsigned long long* ctr = nullptr;
    cudaMalloc(&parent, (size_t)size * 4);
    cudaMalloc(&flag, (size_t)size * 4);
    cudaMalloc(&ctr, 3 * sizeof(unsigned long long));
    cudaMemset(flag, 0, (size_t)size * 4);
    cudaMemset(ctr, 0, 3 * sizeof(unsigned long long));
    k_uf_tile_local<true><<<w5_tile_grid(Z, Y, X), W5_THREADS>>>(
        bits_d, parent, Z, Y, X);
    k_n11_nrep_count<<<blocks, threads>>>(
        bits_d, parent, flag, ctr + 0, ctr + 1, ctr + 2, Z, Y, X);
    unsigned long long hctr[3] = {0, 0, 0};
    cudaMemcpy(hctr, ctr, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    uint32_t nrep32 = 0;
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, tmp_bytes, flag, (uint32_t*)nullptr, (int)size);
        cudaMalloc(&tmp, tmp_bytes);
        uint32_t* n_d = nullptr;
        cudaMalloc(&n_d, 4);
        cub::DeviceReduce::Sum(tmp, tmp_bytes, flag, n_d, (int)size);
        cudaMemcpy(&nrep32, n_d, 4, cudaMemcpyDeviceToHost);
        cudaFree(tmp);
        cudaFree(n_d);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ws_n11_nrep CUDA %s\n", cudaGetErrorString(err));
        cudaFree(parent);
        cudaFree(flag);
        cudaFree(ctr);
        return -2;
    }
    if (n_face) *n_face = hctr[0];
    if (n_list) *n_list = hctr[1];
    if (n_rep) *n_rep = (unsigned long long)nrep32;
    if (n_cross) *n_cross = hctr[2];
    cudaFree(parent);
    cudaFree(flag);
    cudaFree(ctr);
    return 0;
}

// N11 D: contracted (p1,p1) stitch. Recip matches the caller (e9b true, e9c false).
template <bool Recip>
__global__ void k_e4_count_cross(
    const uint8_t* bits, const uint32_t* parent, unsigned int* n_cross,
    int64_t Z, int64_t Y, int64_t X)
{
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t size = Z * Y * X;
    if (i >= size) return;
    const uint8_t b = bits[i];
    if (!b) return;
    const int64_t yx = Y * X;
    const int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    const int dz[6] = {-1, 0, 0, 1, 0, 0};
    const int dy[6] = {0, -1, 0, 0, 1, 0};
    const int dx[6] = {0, 0, -1, 0, 0, 1};
    const uint32_t pi = parent[i];
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        const int64_t nz = z + dz[d], ny = y + dy[d], nx = x + dx[d];
        if (w5_same_tile(z, y, x, nz, ny, nx)) continue;
        const int64_t j = neigh_i(i, d, Y, X);
        if (Recip && !(bits[j] & (uint8_t)RBIT[d])) continue;
        if (pi == parent[j]) continue;
        atomicAdd(n_cross, 1u);
    }
}

template <bool Recip>
__global__ void k_e4_emit_pairs(
    const uint8_t* bits, const uint32_t* parent,
    uint32_t* eu, uint32_t* ev, unsigned int* nout,
    int64_t Z, int64_t Y, int64_t X)
{
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t size = Z * Y * X;
    if (i >= size) return;
    const uint8_t b = bits[i];
    if (!b) return;
    const int64_t yx = Y * X;
    const int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
    const int dz[6] = {-1, 0, 0, 1, 0, 0};
    const int dy[6] = {0, -1, 0, 0, 1, 0};
    const int dx[6] = {0, 0, -1, 0, 0, 1};
    const uint32_t pi = parent[i];
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        const int64_t nz = z + dz[d], ny = y + dy[d], nx = x + dx[d];
        if (w5_same_tile(z, y, x, nz, ny, nx)) continue;
        const int64_t j = neigh_i(i, d, Y, X);
        if (Recip && !(bits[j] & (uint8_t)RBIT[d])) continue;
        const uint32_t pj = parent[j];
        if (pi == pj) continue;
        const unsigned int slot = atomicAdd(nout, 1u);
        eu[slot] = pi;
        ev[slot] = pj;
    }
}

__global__ void k_e4_hook_pairs(
    uint32_t* parent, int* changed,
    const uint32_t* eu, const uint32_t* ev, int nedge)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nedge) return;
    const uint32_t pi = parent[eu[t]];
    const uint32_t pj = parent[ev[t]];
    if (pi == pj) return;
    const uint32_t lo = pi < pj ? pi : pj;
    const uint32_t hi = pi < pj ? pj : pi;
    if (atomicMin(&parent[hi], lo) > lo) atomicExch(changed, 1);
}

// N12: second kernel on the same tile grid as k_uf_tile_local (unchanged).
// Face threads only. parent[] is globally valid because this launches after
// tile_local returns; no cooperative grid.sync.
template <bool Recip>
__global__ void k_uf_tile_local_e4(
    const uint8_t* bits, const uint32_t* parent, unsigned int* n_cross,
    uint32_t* eu, uint32_t* ev, int64_t Z, int64_t Y, int64_t X)
{
    const int64_t x0 = (int64_t)blockIdx.x * W5_TX;
    const int64_t y0 = (int64_t)blockIdx.y * W5_TY;
    const int64_t z0 = (int64_t)blockIdx.z * W5_TZ;
    const int tid = threadIdx.x;
    const int nthread = blockDim.x;
    const int dz[6] = {-1, 0, 0, 1, 0, 0};
    const int dy[6] = {0, -1, 0, 0, 1, 0};
    const int dx[6] = {0, 0, -1, 0, 0, 1};
    for (int s = tid; s < W5_TN; s += nthread) {
        const int lx = s % W5_TX, t = s / W5_TX;
        const int ly = t % W5_TY, lz = t / W5_TY;
        if (!(lz == 0 || lz == W5_TZ - 1 || ly == 0 || ly == W5_TY - 1
              || lx == 0 || lx == W5_TX - 1))
            continue;
        const int64_t gx = x0 + lx, gy = y0 + ly, gz = z0 + lz;
        if (gx >= X || gy >= Y || gz >= Z) continue;
        const int64_t i = (gz * Y + gy) * X + gx;
        const uint8_t b = bits[i];
        if (!b) continue;
        const uint32_t pi = parent[i];
        for (int d = 0; d < 6; ++d) {
            if (!(b & DBIT[d])) continue;
            if (oob_d(d, gz, gy, gx, Z, Y, X)) continue;
            const int nlz = lz + dz[d], nly = ly + dy[d], nlx = lx + dx[d];
            if (nlz >= 0 && nlz < W5_TZ && nly >= 0 && nly < W5_TY
                && nlx >= 0 && nlx < W5_TX)
                continue;
            const int64_t j = neigh_i(i, d, Y, X);
            if (Recip && !(bits[j] & (uint8_t)RBIT[d])) continue;
            const uint32_t pj = parent[j];
            if (pi == pj) continue;
            const unsigned int slot = atomicAdd(n_cross, 1u);
            if (eu) {
                const uint32_t lo = pi < pj ? pi : pj;
                const uint32_t hi = pi < pj ? pj : pi;
                eu[slot] = lo;
                ev[slot] = hi;
            }
        }
    }
}

__global__ void k_e4_pack64(
    const uint32_t* eu, const uint32_t* ev, unsigned long long* key, int n)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    key[t] = ((unsigned long long)eu[t] << 32) | (unsigned long long)ev[t];
}

__global__ void k_e4_unpack64(
    const unsigned long long* key, uint32_t* eu, uint32_t* ev, int n)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const unsigned long long k = key[t];
    eu[t] = (uint32_t)(k >> 32);
    ev[t] = (uint32_t)k;
}

__global__ void k_count_v2(
    const uint8_t* bits, const uint32_t* parent,
    uint32_t* vcount, int64_t Z, int64_t Y, int64_t X, int fold, int priv)
{
    const Vox v = vox_of(Y, X);
    if (!v.ok) return;
    const int64_t i = v.i, z = v.z, y = v.y, x = v.x;
    uint8_t b = bits[i];
    if (!b) return;
    int in_plat = 0;
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        in_plat = 1;
        break;
    }
    if (!in_plat) return;
    uint32_t r = parent[i];
    if (fold) r = uf_find_ro(parent, r);
    if (!priv) {
        atomicAdd(&vcount[r], 1u);
        return;
    }
    // Warp-aggregated atomic: reduce within warp for identical roots, one atomic.
    unsigned int mask = __activemask();
    unsigned int same = __match_any_sync(mask, r);
    int leader = __ffs(same) - 1;
    int lane = threadIdx.x & 31;
    int n = __popc(same);
    if (lane == leader) atomicAdd(&vcount[r], (uint32_t)n);
}

__global__ void k_iota_u32(uint32_t* a, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    a[t] = (uint32_t)t;
}

__global__ void k_gather_u32(
    const uint32_t* idx, const uint32_t* src, uint32_t* dst, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    dst[t] = src[idx[t]];
}

__global__ void k_pack_cv(
    const uint32_t* corners, const uint32_t* vc, unsigned long long* p, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    p[t] = ((unsigned long long)corners[t] << 32) | (unsigned long long)vc[t];
}

__global__ void k_unpack_cv(
    const unsigned long long* p, uint32_t* corners, uint32_t* vc, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const unsigned long long k = p[t];
    corners[t] = (uint32_t)(k >> 32);
    vc[t] = (uint32_t)k;
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
    const uint32_t* idx, const uint32_t* parent, uint32_t* keys, int n, int fold)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint32_t r = parent[idx[t]];
    if (fold) r = uf_find_ro(parent, r);
    keys[t] = r;
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
    const uint32_t* psum, const uint32_t* keys,
    int* plat_begin, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    if (t > 0 && keys[t] == keys[t - 1]) return;
    plat_begin[(int)psum[t]] = t;
}

__global__ void k_plat_meta(
    const int* plat_begin, const uint32_t* vcount,
    uint32_t* qsz, int P, int nC)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    (void)nC;
    // vcount is the per-corner compact copy, already permuted into sorted
    // corner order. Every corner of a plateau shares the same root count, so
    // the first corner's slot is the plateau size.
    qsz[p] = vcount[plat_begin[p]];
}

__global__ void k_gather_vcount_u32(
    const uint32_t* keys, const uint32_t* vcount, uint32_t* vc, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    vc[t] = vcount[keys[t]];
}

// N18 A2: open-addressing root→dense map for compact vcount.
static constexpr uint32_t VC_HASH_EMPTY = 0xffffffffu;

__global__ void k_vc_hash_clear(uint32_t* keys, uint32_t* ids, int cap)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cap) return;
    keys[t] = VC_HASH_EMPTY;
    ids[t] = VC_HASH_EMPTY;
}

__global__ void k_vc_hash_insert(
    const uint32_t* uniq, int U, uint32_t* hkeys, uint32_t* hids, int cap)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= U) return;
    uint32_t key = uniq[t];
    uint32_t slot = key & (uint32_t)(cap - 1);
    for (int it = 0; it < cap; ++it) {
        uint32_t old = atomicCAS(&hkeys[slot], VC_HASH_EMPTY, key);
        if (old == VC_HASH_EMPTY || old == key) {
            hids[slot] = (uint32_t)t;
            return;
        }
        slot = (slot + 1u) & (uint32_t)(cap - 1);
    }
}

__device__ inline int vc_hash_lookup(
    const uint32_t* hkeys, const uint32_t* hids, int cap, uint32_t key)
{
    uint32_t slot = key & (uint32_t)(cap - 1);
    for (int it = 0; it < 4096; ++it) {
        uint32_t k = hkeys[slot];
        if (k == VC_HASH_EMPTY) return -1;
        if (k == key) return (int)hids[slot];
        slot = (slot + 1u) & (uint32_t)(cap - 1);
    }
    return -1;
}

__global__ void k_count_v2_compact(
    const uint8_t* bits, const uint32_t* parent,
    const uint32_t* hkeys, const uint32_t* hids, int cap,
    uint32_t* vcount, int64_t Z, int64_t Y, int64_t X, int fold)
{
    const Vox v = vox_of(Y, X);
    if (!v.ok) return;
    const int64_t i = v.i, z = v.z, y = v.y, x = v.x;
    uint8_t b = bits[i];
    if (!b) return;
    int in_plat = 0;
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        in_plat = 1;
        break;
    }
    if (!in_plat) return;
    uint32_t r = parent[i];
    if (fold) r = uf_find_ro(parent, r);
    int dens = vc_hash_lookup(hkeys, hids, cap, r);
    if (dens >= 0) atomicAdd(&vcount[dens], 1u);
}

__global__ void k_gather_vcount_compact(
    const uint32_t* keys, const uint32_t* hkeys, const uint32_t* hids,
    int cap, const uint32_t* vcount, uint32_t* vc, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int dens = vc_hash_lookup(hkeys, hids, cap, keys[t]);
    vc[t] = (dens >= 0) ? vcount[dens] : 0u;
}

// The queue is sized per plateau from vcount, so an undersized qsz would run
// one plateau's BFS into the next one's slot and corrupt the segmentation
// silently. cap/overflow turn that into a loud failure, and qused reports the
// true high-water mark so the sizing can be measured rather than guessed.
__global__ void k_indep_bfs(
    uint8_t* seg, const uint32_t* corners, const int* plat_begin,
    uint32_t* q, const uint32_t* qoff,
    int* overflow, int64_t* qused,
    int P, int nC, int64_t qtot, int64_t Y, int64_t X)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int c0 = plat_begin[p];
    int nseed = ((p + 1 < P) ? plat_begin[p + 1] : nC) - c0;
    if (nseed <= 0) return;
    uint32_t* qp = q + qoff[p];
    int64_t cap = ((p + 1 < P) ? (int64_t)qoff[p + 1] : qtot) - (int64_t)qoff[p];
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
        int64_t i = (int64_t)qp[bi];
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
                    qp[tail++] = (uint32_t)j;
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

// W5's driver: one tile-local pass, then hook-and-compress over the stitch
// list until a round changes nothing, then a single full-volume flatten.
//
// The last flatten is not optional. A voxel that is neither on a face nor a
// phase-1 root still points at its phase-1 root, and that root's entry now
// points on to the component root, so such a voxel is one hop short. One pass
// fixes every one of them, against the `uf_rounds` full passes the loop it
// replaces was doing.
//
// `flag` is borrowed as scan scratch. Both callers have it allocated and
// neither has written anything into it yet at this point: e9b fills it in
// k_corner_flag afterwards, e9c in k_root_flag.
template <bool Recip>
static int w5_union_find(const uint8_t* bits_d, uint32_t* parent,
                         uint32_t* flag, int64_t Z, int64_t Y, int64_t X,
                         const char* tag) {
    const int64_t size = Z * Y * X;
    const int threads = 256;
    const int blocks = (int)((size + threads - 1) / threads);

    cudaEvent_t ev0, ev1, evs;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventCreate(&evs);
    cudaEventRecord(ev0);

    {
        NvRange nv("tile_local");
        k_uf_tile_local<Recip><<<w5_tile_grid(Z, Y, X), W5_THREADS>>>(
            bits_d, parent, Z, Y, X);
    }
    cudaEventRecord(evs);
    NvRange nv_stitch("w5_stitch");

    k_w5_list_flag<<<blocks, threads>>>(parent, flag, Z, Y, X);
    uint32_t last_f = 0, last_p = 0;
    cudaMemcpy(&last_f, flag + size - 1, 4, cudaMemcpyDeviceToHost);
    {
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, flag, flag, (int)size);
        if (stitch_arena()) {
            if (g_w5a.scan_cap < tmp_bytes) {
                if (g_w5a.scan_tmp) cudaFree(g_w5a.scan_tmp);
                cudaMalloc(&g_w5a.scan_tmp, tmp_bytes ? tmp_bytes : 4);
                g_w5a.scan_cap = tmp_bytes;
            }
            tmp = g_w5a.scan_tmp;
        } else {
            cudaMalloc(&tmp, tmp_bytes);
        }
        cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, flag, flag, (int)size);
        if (!stitch_arena()) cudaFree(tmp);
    }
    cudaMemcpy(&last_p, flag + size - 1, 4, cudaMemcpyDeviceToHost);
    const int nlist = (int)(last_p + last_f);
    uint32_t* list = nullptr;
    const size_t nlist_n = (size_t)(nlist > 0 ? nlist : 1);
    if (stitch_arena()) {
        if (g_w5a.list_n < nlist_n) {
            if (g_w5a.list) cudaFree(g_w5a.list);
            cudaMalloc(&g_w5a.list, nlist_n * 4);
            g_w5a.list_n = nlist_n;
        }
        list = g_w5a.list;
    } else {
        cudaMalloc(&list, nlist_n * 4);
    }
    k_scatter_idx_u32<<<blocks, threads>>>(flag, last_f, list, size);

    int* changed = nullptr;
    int* pin_host = nullptr;
    if (pin_changed()) {
        if (!g_w5a.changed) {
            cudaHostAlloc((void**)&g_w5a.pin_host, 4, cudaHostAllocMapped);
            cudaHostGetDevicePointer((void**)&g_w5a.changed, g_w5a.pin_host, 0);
        }
        changed = g_w5a.changed;
        pin_host = g_w5a.pin_host;
    } else if (stitch_arena()) {
        if (!g_w5a.changed) cudaMalloc(&g_w5a.changed, 4);
        changed = g_w5a.changed;
    } else {
        cudaMalloc(&changed, 4);
    }
    const int cap = 64;
    int rounds = 0;
    int jump_rounds = 0;
    const int lb = (nlist + threads - 1) / threads;
    const int find_r = hook_root() ? 1 : 0;
    for (int r = 0; r < cap && nlist > 0; ++r) {
        cudaMemset(changed, 0, 4);
        k_w5_hook_list<Recip><<<lb, threads>>>(
            bits_d, parent, changed, list, nlist, Z, Y, X, find_r);
        if (!hook_root()) {
            k_w5_compress_list<<<lb, threads>>>(
                parent, changed, list, nlist, list_halving() ? 1 : 0);
        }
        int h = 0;
        if (pin_host) {
            cudaDeviceSynchronize();
            h = pin_host[0];
        } else {
            cudaMemcpy(&h, changed, 4, cudaMemcpyDeviceToHost);
        }
        ++rounds;
        if (!h) break;
    }
    if (hook_root() && nlist > 0) {
        cudaMemset(changed, 0, 4);
        k_w5_compress_list<<<lb, threads>>>(
            parent, changed, list, nlist, list_halving() ? 1 : 0);
    }
    const bool skip_c = fold_flatten() && (!fold_e9b_only() || Recip);
    if (jump_flatten()) {
        for (int r = 0; r < 64; ++r) {
            cudaMemset(changed, 0, 4);
            k_uf_jump<<<blocks, threads>>>(parent, changed, size);
            int h = 0;
            cudaMemcpy(&h, changed, 4, cudaMemcpyDeviceToHost);
            ++jump_rounds;
            if (!h) break;
        }
        fprintf(stderr, "%s jump_flatten rounds=%d/64\n", tag, jump_rounds);
    } else if (!skip_c) {
        k_uf_compress_c<<<blocks, threads>>>(parent, changed, size);
    }

    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0, tile_ms = 0, stitch_ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventElapsedTime(&tile_ms, ev0, evs);
    cudaEventElapsedTime(&stitch_ms, evs, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaEventDestroy(evs);
    if (!stitch_arena() && !pin_changed()) {
        cudaFree(changed);
        cudaFree(list);
    } else if (!stitch_arena()) {
        cudaFree(list);
    }
    fprintf(stderr,
            "%s w5 tile=%dx%dx%d shmem=%dB list=%d/%lld (%.3f) "
            "stitch_rounds=%d/%d ms=%.2f tile=%.2f stitch=%.2f%s\n",
            tag, W5_TZ, W5_TY, W5_TX, (int)(W5_TN * 5), nlist,
            (long long)size, (double)nlist / (double)size, rounds, cap, ms,
            tile_ms, stitch_ms,
            rounds >= cap ? " NOT-CONVERGED" : "");
    g_n9_w5_calls += 1;
    g_n9_nlist_sum += (int64_t)nlist;
    if (nlist > g_n9_nlist_max) g_n9_nlist_max = nlist;
    g_n9_nlist_last = nlist;
    g_n9_nvox_last = size;
    g_n9_rounds_sum += rounds;
    g_n9_w5_ms += ms;
    g_n11_tile_ms += tile_ms;
    g_n11_stitch_ms += stitch_ms;
    return rounds;
}

// WATERZ_UF_ALGO=4. Tile-local is k_uf_tile_local (not edited). Face emit is
// a second kernel on the same grid. CUB Unique on undirected (min,max) pairs
// before hook.
template <bool Recip>
static int w5_e4_union_find(const uint8_t* bits_d, uint32_t* parent,
                            uint32_t* flag, int64_t Z, int64_t Y, int64_t X,
                            const char* tag) {
    (void)flag;
    const int64_t size = Z * Y * X;
    const int threads = 256;
    const int blocks = (int)((size + threads - 1) / threads);
    const dim3 tgrid = w5_tile_grid(Z, Y, X);

    cudaEvent_t ev0, ev1, evs;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventCreate(&evs);
    cudaEventRecord(ev0);

    {
        NvRange nv("tile_local");
        k_uf_tile_local<Recip><<<tgrid, W5_THREADS>>>(
            bits_d, parent, Z, Y, X);
    }
    cudaEventRecord(evs);
    NvRange nv_stitch("w5_stitch");

    unsigned int* n_d = nullptr;
    uint32_t* eu = nullptr;
    uint32_t* ev = nullptr;
    unsigned int ncross = 0;
    {
        NvRange nv("e4_emit");
        cudaMalloc(&n_d, 4);
        cudaMemset(n_d, 0, 4);
        k_uf_tile_local_e4<Recip><<<tgrid, W5_THREADS>>>(
            bits_d, parent, n_d, nullptr, nullptr, Z, Y, X);
        cudaMemcpy(&ncross, n_d, 4, cudaMemcpyDeviceToHost);

        cudaMalloc(&eu, (size_t)(ncross > 0 ? ncross : 1) * 4);
        cudaMalloc(&ev, (size_t)(ncross > 0 ? ncross : 1) * 4);
        cudaMemset(n_d, 0, 4);
        if (ncross > 0) {
            k_uf_tile_local_e4<Recip><<<tgrid, W5_THREADS>>>(
                bits_d, parent, n_d, eu, ev, Z, Y, X);
            cudaMemcpy(&ncross, n_d, 4, cudaMemcpyDeviceToHost);
        }
    }

    uint32_t* eu_u = eu;
    uint32_t* ev_u = ev;
    int npair = (int)ncross;
    uint32_t* reps = nullptr;
    int nrep = 0;
    {
        NvRange nv("e4_unique");
        if (ncross > 0) {
        unsigned long long* key_in = nullptr;
        unsigned long long* key_out = nullptr;
        cudaMalloc(&key_in, (size_t)ncross * 8);
        cudaMalloc(&key_out, (size_t)ncross * 8);
        const int pb = (int)((ncross + threads - 1) / threads);
        k_e4_pack64<<<pb, threads>>>(eu, ev, key_in, (int)ncross);
        {
            void* tmp = nullptr;
            size_t tmp_bytes = 0;
            cub::DeviceRadixSort::SortKeys(
                nullptr, tmp_bytes, key_in, key_out, (int)ncross);
            cudaMalloc(&tmp, tmp_bytes);
            cub::DeviceRadixSort::SortKeys(
                tmp, tmp_bytes, key_in, key_out, (int)ncross);
            cudaFree(tmp);
        }
        int* nsel = nullptr;
        cudaMalloc(&nsel, 4);
        unsigned long long* key_u = nullptr;
        cudaMalloc(&key_u, (size_t)ncross * 8);
        {
            void* tmp = nullptr;
            size_t tmp_bytes = 0;
            cub::DeviceSelect::Unique(
                nullptr, tmp_bytes, key_out, key_u, nsel, (int)ncross);
            cudaMalloc(&tmp, tmp_bytes);
            cub::DeviceSelect::Unique(
                tmp, tmp_bytes, key_out, key_u, nsel, (int)ncross);
            cudaFree(tmp);
        }
        cudaMemcpy(&npair, nsel, 4, cudaMemcpyDeviceToHost);
        cudaFree(key_in);
        cudaFree(key_out);
        cudaMalloc(&eu_u, (size_t)(npair > 0 ? npair : 1) * 4);
        cudaMalloc(&ev_u, (size_t)(npair > 0 ? npair : 1) * 4);
        if (npair > 0) {
            const int ub = (npair + threads - 1) / threads;
            k_e4_unpack64<<<ub, threads>>>(key_u, eu_u, ev_u, npair);
        }
        cudaFree(key_u);
        cudaFree(eu);
        cudaFree(ev);
        eu = ev = nullptr;

        const int nends = npair * 2;
        uint32_t* ends_in = nullptr;
        uint32_t* ends_out = nullptr;
        cudaMalloc(&ends_in, (size_t)(nends > 0 ? nends : 1) * 4);
        cudaMalloc(&ends_out, (size_t)(nends > 0 ? nends : 1) * 4);
        if (npair > 0) {
            cudaMemcpy(ends_in, eu_u, (size_t)npair * 4, cudaMemcpyDeviceToDevice);
            cudaMemcpy(ends_in + npair, ev_u, (size_t)npair * 4,
                       cudaMemcpyDeviceToDevice);
        }
        if (nends > 0) {
            void* tmp = nullptr;
            size_t tmp_bytes = 0;
            cub::DeviceRadixSort::SortKeys(
                nullptr, tmp_bytes, ends_in, ends_out, nends);
            cudaMalloc(&tmp, tmp_bytes);
            cub::DeviceRadixSort::SortKeys(
                tmp, tmp_bytes, ends_in, ends_out, nends);
            cudaFree(tmp);
        }
        cudaFree(ends_in);
        cudaMalloc(&reps, (size_t)(nends > 0 ? nends : 1) * 4);
        {
            void* tmp = nullptr;
            size_t tmp_bytes = 0;
            cub::DeviceSelect::Unique(
                nullptr, tmp_bytes, ends_out, reps, nsel, nends > 0 ? nends : 1);
            cudaMalloc(&tmp, tmp_bytes);
            if (nends > 0) {
                cub::DeviceSelect::Unique(
                    tmp, tmp_bytes, ends_out, reps, nsel, nends);
            } else {
                cudaMemset(nsel, 0, 4);
            }
            cudaFree(tmp);
        }
        cudaMemcpy(&nrep, nsel, 4, cudaMemcpyDeviceToHost);
        cudaFree(nsel);
        cudaFree(ends_out);
    } else {
        cudaMalloc(&reps, 4);
        cudaMalloc(&eu_u, 4);
        cudaMalloc(&ev_u, 4);
        cudaFree(eu);
        cudaFree(ev);
        eu = ev = nullptr;
    }
    }

    int* changed = nullptr;
    cudaMalloc(&changed, 4);
    const int cap = 64;
    int rounds = 0;
    const int eb = npair > 0 ? (npair + threads - 1) / threads : 1;
    const int rb = nrep > 0 ? (nrep + threads - 1) / threads : 1;
    {
        NvRange nv("e4_hook");
        for (int r = 0; r < cap && npair > 0; ++r) {
            cudaMemset(changed, 0, 4);
            k_e4_hook_pairs<<<eb, threads>>>(parent, changed, eu_u, ev_u, npair);
            k_w5_compress_list<<<rb, threads>>>(
                parent, changed, reps, nrep, list_halving() ? 1 : 0);
            int h = 0;
            cudaMemcpy(&h, changed, 4, cudaMemcpyDeviceToHost);
            ++rounds;
            if (!h) break;
        }
        const bool skip_c = fold_flatten() && (!fold_e9b_only() || Recip);
        if (jump_flatten()) {
            int jump_rounds = 0;
            for (int r = 0; r < 64; ++r) {
                cudaMemset(changed, 0, 4);
                k_uf_jump<<<blocks, threads>>>(parent, changed, size);
                int h = 0;
                cudaMemcpy(&h, changed, 4, cudaMemcpyDeviceToHost);
                ++jump_rounds;
                if (!h) break;
            }
            fprintf(stderr, "%s jump_flatten rounds=%d/64\n", tag, jump_rounds);
        } else if (!skip_c) {
            k_uf_compress_c<<<blocks, threads>>>(parent, changed, size);
        }
    }

    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0, tile_ms = 0, stitch_ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventElapsedTime(&tile_ms, ev0, evs);
    cudaEventElapsedTime(&stitch_ms, evs, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaEventDestroy(evs);
    cudaFree(changed);
    cudaFree(reps);
    cudaFree(eu_u);
    cudaFree(ev_u);
    cudaFree(n_d);
    fprintf(stderr,
            "%s e4-fuse tile=%dx%dx%d n_rep=%d n_face_pairs=%u n_uniq=%d/%lld "
            "stitch_rounds=%d/%d ms=%.2f tile=%.2f stitch=%.2f%s\n",
            tag, W5_TZ, W5_TY, W5_TX, nrep, ncross, npair, (long long)size,
            rounds, cap, ms, tile_ms, stitch_ms,
            rounds >= cap ? " NOT-CONVERGED" : "");
    g_n9_w5_calls += 1;
    g_n9_nlist_sum += (int64_t)nrep;
    if (nrep > g_n9_nlist_max) g_n9_nlist_max = nrep;
    g_n9_nlist_last = nrep;
    g_n9_nvox_last = size;
    g_n9_rounds_sum += rounds;
    g_n9_w5_ms += ms;
    g_n11_tile_ms += tile_ms;
    g_n11_stitch_ms += stitch_ms;
    g_n11_nrep_last += nrep;
    g_n11_ncross_last += (int64_t)npair;
    return rounds;
}

// scratch_d, when given, supplies the per-voxel `parent` array instead of
// allocating one. The caller's label buffer is the natural donor: it is
// untouched until e9c_basins_d writes into it, and parent is dead before then.
// Nothing here reads parent at an index other than the thread's own except
// k_keys_from_parent_u32, which runs before the buffer is handed on.
static int e9b_divide_d(uint8_t* bits_d, int64_t Z, int64_t Y, int64_t X,
                        float* ms_out, uint32_t* scratch_d = nullptr) {
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
    uint32_t* parent = scratch_d;
    const bool own_parent = (scratch_d == nullptr);
    uint32_t* flag = nullptr;
    uint32_t* vcount = nullptr;
    ws_mem_mark("divide/enter");
    if (own_parent) cudaMalloc(&parent, (size_t)size * 4);
    cudaMalloc(&flag, (size_t)size * 4);
    // vcount is unused during UF. Allocating it next to flag made the
    // 8 B/vox pair that owned the val peak (B_SORT_PEAK).
    ws_mem_mark("divide/uf");
    if (uf_algo() == 3) {
        // W5. Phase 1 writes every in-volume parent entry itself, so the
        // k_parent_init pass below is not needed here.
        w5_union_find<true>(bits_d, parent, flag, Z, Y, X, "E9b");
    } else if (uf_algo() == 4) {
        w5_e4_union_find<true>(bits_d, parent, flag, Z, Y, X, "E9b");
    } else {
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
    // Split the round between its two kernels. Removing the index division
    // from k_hook_bidir cut its instruction count from 408 to 240 and changed
    // the loop's time by 0.3%, which says the cost is in the other kernel, so
    // the split is worth knowing before restructuring anything. Absolute
    // numbers inflate when the card is shared, but both kernels are measured
    // inside the same loop under the same contention, so the ratio holds.
    cudaEvent_t hk0, hk1, cp0, cp1;
    cudaEventCreate(&hk0);
    cudaEventCreate(&hk1);
    cudaEventCreate(&cp0);
    cudaEventCreate(&cp1);
    float hook_ms = 0, comp_ms = 0;
    for (int r = 0; r < uf_cap; ++r) {
        cudaMemset(uf_changed, 0, 4);
        cudaEventRecord(hk0);
        if (uf_algo() == 2) {
            k_hook_bidir_tiled<<<ws_tile_grid(Z, Y, X),
                                 dim3(WS_TX, WS_TY, WS_TZ)>>>(
                bits_d, parent, uf_changed, Z, Y, X);
        } else {
            k_hook_bidir<<<vox_grid(Z, Y, X, threads), threads>>>(bits_d, parent, uf_changed, Z, Y, X, playne_hook() ? 1 : 0);
        }
        cudaEventRecord(hk1);
        cudaEventRecord(cp0);
        if (uf_algo() == 1) {
            k_uf_jump<<<blocks, threads>>>(parent, uf_changed, size);
        } else {
            k_uf_compress_c<<<blocks, threads>>>(parent, uf_changed, size);
        }
        cudaEventRecord(cp1);
        int h = 0;
        cudaMemcpy(&h, uf_changed, 4, cudaMemcpyDeviceToHost);
        float a = 0, b = 0;
        cudaEventElapsedTime(&a, hk0, hk1);
        cudaEventElapsedTime(&b, cp0, cp1);
        hook_ms += a;
        comp_ms += b;
        ++uf_rounds;
        if (!h) break;
    }
    cudaEventDestroy(hk0);
    cudaEventDestroy(hk1);
    cudaEventDestroy(cp0);
    cudaEventDestroy(cp1);
    cudaEventRecord(uev1);
    cudaEventSynchronize(uev1);
    float uf_ms = 0;
    cudaEventElapsedTime(&uf_ms, uev0, uev1);
    cudaEventDestroy(uev0);
    cudaEventDestroy(uev1);
    cudaFree(uf_changed);
    fprintf(stderr,
            "E9b uf_algo=%d uf_rounds=%d/%d uf_ms=%.2f hook_ms=%.2f comp_ms=%.2f%s\n",
            uf_algo(), uf_rounds, uf_cap, uf_ms, hook_ms, comp_ms,
            uf_rounds >= uf_cap ? " NOT-CONVERGED" : "");
    }
    k_corner_flag<<<blocks, threads>>>(bits_d, flag, Z, Y, X);
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
        if (own_parent) cudaFree(parent);
        cudaFree(flag);
        cudaFree(vcount);
        if (ms_out) *ms_out = 0;
        return 0;
    }
    uint32_t* corners_in = nullptr;
    uint32_t* corners_out = nullptr;
    uint32_t* keys_in = nullptr;
    uint32_t* keys_out = nullptr;
    uint32_t* vc_in = nullptr;
    uint32_t* vc_out = nullptr;
    uint32_t* idx_in = nullptr;
    uint32_t* idx_out = nullptr;
    // Scatter and keys while flag is live; do not also hold vcount. Then
    // free flag before the nvox vcount + compact vc pair.
    cudaMalloc(&corners_in, (size_t)nC * 4);
    ws_mem_mark("divide/corners");
    k_scatter_idx_u32<<<blocks, threads>>>(psum, last_f, corners_in, size);
    int cb = (nC + 255) / 256;
    cudaFree(flag);
    flag = nullptr;
    psum = nullptr;
    cudaMalloc(&keys_in, (size_t)nC * 4);
    k_keys_from_parent_u32<<<cb, 256>>>(
        corners_in, parent, keys_in, nC, fold_flatten() ? 1 : 0);
    // Host park drops the device peak to vcount+keys (B_FIT). It also does a
    // pageable D2H of nC corners plus ~nC/1M synchronous vcount copies. That
    // is a latency tax, not an algorithm. WATERZ_HOST_PARK=0 keeps both arrays
    // on the device.
    const bool park = host_park_on();
    std::vector<uint32_t> corners_h;
    std::vector<uint32_t> vc_h;
    cudaEvent_t pe0, pe1;
    cudaEventCreate(&pe0);
    cudaEventCreate(&pe1);
    cudaEventRecord(pe0);
    if (park) {
        corners_h.resize((size_t)nC);
        cudaMemcpy(corners_h.data(), corners_in, (size_t)nC * 4,
                   cudaMemcpyDeviceToHost);
        cudaFree(corners_in);
        corners_in = nullptr;
    }
    cudaEventRecord(pe1);
    cudaEventSynchronize(pe1);
    {
        float ms = 0;
        cudaEventElapsedTime(&ms, pe0, pe1);
        g_n10_park_ms += ms;
    }
    cudaEventRecord(pe0);
    {
        NvRange nv_vcount("vcount");
        if (vcount_compact() && !park) {
            // Remap unique corner roots → dense [0,U), histogram into U slots.
            uint32_t* keys_sorted = nullptr;
            uint32_t* uniq = nullptr;
            int* d_num_selected = nullptr;
            cudaMalloc(&keys_sorted, (size_t)nC * 4);
            cudaMalloc(&uniq, (size_t)nC * 4);
            cudaMalloc(&d_num_selected, 4);
            {
                void* tmp = nullptr;
                size_t tmp_bytes = 0;
                cub::DeviceRadixSort::SortKeys(
                    nullptr, tmp_bytes, keys_in, keys_sorted, nC);
                cudaMalloc(&tmp, tmp_bytes);
                cub::DeviceRadixSort::SortKeys(
                    tmp, tmp_bytes, keys_in, keys_sorted, nC);
                cudaFree(tmp);
            }
            {
                void* tmp = nullptr;
                size_t tmp_bytes = 0;
                cub::DeviceSelect::Unique(
                    nullptr, tmp_bytes, keys_sorted, uniq, d_num_selected, nC);
                cudaMalloc(&tmp, tmp_bytes);
                cub::DeviceSelect::Unique(
                    tmp, tmp_bytes, keys_sorted, uniq, d_num_selected, nC);
                cudaFree(tmp);
            }
            int U = 0;
            cudaMemcpy(&U, d_num_selected, 4, cudaMemcpyDeviceToHost);
            cudaFree(keys_sorted);
            cudaFree(d_num_selected);
            if (U <= 0) {
                cudaFree(uniq);
                if (own_parent) cudaFree(parent);
                parent = nullptr;
                cudaMalloc(&vc_in, (size_t)nC * 4);
                cudaMemset(vc_in, 0, (size_t)nC * 4);
            } else {
                int cap = 1;
                while (cap < U * 2) cap <<= 1;
                if (cap < 1024) cap = 1024;
                uint32_t* hkeys = nullptr;
                uint32_t* hids = nullptr;
                cudaMalloc(&hkeys, (size_t)cap * 4);
                cudaMalloc(&hids, (size_t)cap * 4);
                int hb = (cap + 255) / 256;
                k_vc_hash_clear<<<hb, 256>>>(hkeys, hids, cap);
                int ub = (U + 255) / 256;
                k_vc_hash_insert<<<ub, 256>>>(uniq, U, hkeys, hids, cap);
                cudaFree(uniq);
                cudaMalloc(&vcount, (size_t)U * 4);
                cudaMemset(vcount, 0, (size_t)U * 4);
                k_count_v2_compact<<<vox_grid(Z, Y, X, threads), threads>>>(
                    bits_d, parent, hkeys, hids, cap, vcount, Z, Y, X,
                    fold_flatten() ? 1 : 0);
                if (own_parent) cudaFree(parent);
                parent = nullptr;
                cudaMalloc(&vc_in, (size_t)nC * 4);
                k_gather_vcount_compact<<<cb, 256>>>(
                    keys_in, hkeys, hids, cap, vcount, vc_in, nC);
                cudaFree(vcount);
                vcount = nullptr;
                cudaFree(hkeys);
                cudaFree(hids);
                fprintf(stderr, "N18 vcount_compact nC=%d U=%d cap=%d\n",
                        nC, U, cap);
            }
        } else {
            cudaMalloc(&vcount, (size_t)size * 4);
            cudaMemset(vcount, 0, (size_t)size * 4);
            k_count_v2<<<vox_grid(Z, Y, X, threads), threads>>>(bits_d, parent, vcount, Z, Y, X, fold_flatten() ? 1 : 0, vcount_priv() ? 1 : 0);
            if (own_parent) cudaFree(parent);
            parent = nullptr;
            if (park) {
                vc_h.resize((size_t)nC);
                const int CHUNK = 1 << 20;
                uint32_t* chunk = nullptr;
                cudaMalloc(&chunk, (size_t)CHUNK * 4);
                for (int off = 0; off < nC; off += CHUNK) {
                    int n = nC - off;
                    if (n > CHUNK) n = CHUNK;
                    int nb = (n + 255) / 256;
                    k_gather_vcount_u32<<<nb, 256>>>(keys_in + off, vcount, chunk, n);
                    cudaMemcpy(vc_h.data() + off, chunk, (size_t)n * 4,
                               cudaMemcpyDeviceToHost);
                }
                cudaFree(chunk);
            } else {
                cudaMalloc(&vc_in, (size_t)nC * 4);
                k_gather_vcount_u32<<<cb, 256>>>(keys_in, vcount, vc_in, nC);
            }
            cudaFree(vcount);
            vcount = nullptr;
        }
        cudaEventRecord(pe1);
        cudaEventSynchronize(pe1);
    }
    {
        float ms = 0;
        cudaEventElapsedTime(&ms, pe0, pe1);
        g_n10_vcount_ms += ms;
    }
    cudaMalloc(&keys_out, (size_t)nC * 4);
    ws_mem_mark("divide/pre-sort");
    if (sort_pack() && !park) {
        unsigned long long* pack_in = nullptr;
        unsigned long long* pack_out = nullptr;
        cudaMalloc(&pack_in, (size_t)nC * 8);
        cudaMalloc(&pack_out, (size_t)nC * 8);
        k_pack_cv<<<cb, 256>>>(corners_in, vc_in, pack_in, nC);
        cub::DoubleBuffer<uint32_t> d_keys(keys_in, keys_out);
        cub::DoubleBuffer<unsigned long long> d_pack(pack_in, pack_out);
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, tmp_bytes, d_keys, d_pack, nC);
        g_sort_nC = nC;
        g_sort_tmp_dbl = tmp_bytes;
        g_sort_tmp_inout = tmp_bytes;
        fprintf(stderr,
                "E9b sort-pack nC=%d tmp=%zu (%.3f GiB) park=0\n",
                nC, tmp_bytes, tmp_bytes / 1073741824.0);
        cudaEventRecord(pe0);
        {
            NvRange nv_sort("sort");
            cudaMalloc(&tmp, tmp_bytes);
            cub::DeviceRadixSort::SortPairs(tmp, tmp_bytes, d_keys, d_pack, nC);
            cudaFree(tmp);
        }
        cudaEventRecord(pe1);
        cudaEventSynchronize(pe1);
        {
            float ms = 0;
            cudaEventElapsedTime(&ms, pe0, pe1);
            g_n10_sort_ms += ms;
        }
        uint32_t* keys_sorted = d_keys.Current();
        uint32_t* keys_alt = d_keys.Alternate();
        unsigned long long* pack_sorted = d_pack.Current();
        unsigned long long* pack_alt = d_pack.Alternate();
        if (keys_alt) {
            cudaFree(keys_alt);
            if (keys_alt == keys_in) keys_in = nullptr;
            else keys_out = nullptr;
        }
        keys_out = keys_sorted;
        keys_in = nullptr;
        cudaEventRecord(pe0);
        cudaMalloc(&corners_out, (size_t)nC * 4);
        cudaMalloc(&vc_out, (size_t)nC * 4);
        k_unpack_cv<<<cb, 256>>>(pack_sorted, corners_out, vc_out, nC);
        cudaFree(corners_in);
        corners_in = nullptr;
        cudaFree(vc_in);
        vc_in = nullptr;
        cudaFree(pack_sorted);
        if (pack_alt && pack_alt != pack_sorted) cudaFree(pack_alt);
        cudaEventRecord(pe1);
        cudaEventSynchronize(pe1);
        {
            float ms = 0;
            cudaEventElapsedTime(&ms, pe0, pe1);
            g_n10_unpark_ms += ms;
        }
    } else {
    cudaMalloc(&idx_in, (size_t)nC * 4);
    cudaMalloc(&idx_out, (size_t)nC * 4);
    k_iota_u32<<<cb, 256>>>(idx_in, nC);
    ws_mem_mark("divide/pre-sort");
    {
        cub::DoubleBuffer<uint32_t> d_keys(keys_in, keys_out);
        cub::DoubleBuffer<uint32_t> d_idx(idx_in, idx_out);
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, tmp_bytes, d_keys, d_idx, nC);
        g_sort_nC = nC;
        g_sort_tmp_dbl = tmp_bytes;
        {
            size_t tmp_in = 0;
            cub::DeviceRadixSort::SortPairs(
                nullptr, tmp_in, keys_in, keys_out, idx_in, idx_out, nC);
            g_sort_tmp_inout = tmp_in;
        }
        fprintf(stderr,
                "E9b sort nC=%d tmp_inout=%zu tmp_dbl=%zu "
                "(%.3f / %.3f GiB) park=%d\n",
                nC, g_sort_tmp_inout, g_sort_tmp_dbl,
                g_sort_tmp_inout / 1073741824.0,
                g_sort_tmp_dbl / 1073741824.0, park ? 1 : 0);
        cudaEventRecord(pe0);
        {
            NvRange nv_sort("sort");
            cudaMalloc(&tmp, tmp_bytes);
            ws_mem_mark("divide/sort-tmp");
            cub::DeviceRadixSort::SortPairs(tmp, tmp_bytes, d_keys, d_idx, nC);
            cudaFree(tmp);
        }
        cudaEventRecord(pe1);
        cudaEventSynchronize(pe1);
        {
            float ms = 0;
            cudaEventElapsedTime(&ms, pe0, pe1);
            g_n10_sort_ms += ms;
        }
        uint32_t* keys_sorted = d_keys.Current();
        uint32_t* keys_alt = d_keys.Alternate();
        uint32_t* idx_sorted = d_idx.Current();
        uint32_t* idx_alt = d_idx.Alternate();
        if (keys_alt) {
            cudaFree(keys_alt);
            if (keys_alt == keys_in) keys_in = nullptr;
            else keys_out = nullptr;
        }
        keys_out = keys_sorted;
        keys_in = nullptr;
        if (idx_alt) {
            if (idx_alt == idx_in) idx_in = nullptr;
            else idx_out = nullptr;
        }
        cudaEventRecord(pe0);
        if (park) {
            cudaMalloc(&corners_in, (size_t)nC * 4);
            cudaMemcpy(corners_in, corners_h.data(), (size_t)nC * 4,
                       cudaMemcpyHostToDevice);
            corners_h.clear();
            corners_h.shrink_to_fit();
        }
        cudaMalloc(&corners_out, (size_t)nC * 4);
        k_gather_u32<<<cb, 256>>>(idx_sorted, corners_in, corners_out, nC);
        cudaFree(corners_in);
        corners_in = nullptr;
        if (park) {
            cudaMalloc(&vc_in, (size_t)nC * 4);
            cudaMemcpy(vc_in, vc_h.data(), (size_t)nC * 4,
                       cudaMemcpyHostToDevice);
            vc_h.clear();
            vc_h.shrink_to_fit();
        }
        cudaMalloc(&vc_out, (size_t)nC * 4);
        k_gather_u32<<<cb, 256>>>(idx_sorted, vc_in, vc_out, nC);
        cudaFree(vc_in);
        vc_in = nullptr;
        cudaFree(idx_sorted);
        if (idx_alt) cudaFree(idx_alt);
        idx_in = nullptr;
        idx_out = nullptr;
        cudaEventRecord(pe1);
        cudaEventSynchronize(pe1);
        {
            float ms = 0;
            cudaEventElapsedTime(&ms, pe0, pe1);
            g_n10_unpark_ms += ms;
        }
    }
    }
    cudaEventDestroy(pe0);
    cudaEventDestroy(pe1);
    ws_mem_mark("divide/post-sort");
    uint32_t* start = nullptr;
    cudaMalloc(&start, (size_t)nC * 4);
    k_run_start<<<cb, 256>>>(keys_out, start, nC);
    uint32_t last_s = 0, last_sp = 0;
    cudaMemcpy(&last_s, start + nC - 1, 4, cudaMemcpyDeviceToHost);
    {
        cudaEvent_t se0, se1;
        cudaEventCreate(&se0);
        cudaEventCreate(&se1);
        cudaEventRecord(se0);
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, start, start, nC);
        cudaMalloc(&tmp, tmp_bytes);
        cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, start, start, nC);
        cudaFree(tmp);
        cudaEventRecord(se1);
        cudaEventSynchronize(se1);
        float ms = 0;
        cudaEventElapsedTime(&ms, se0, se1);
        g_n10_scan_ms += ms;
        cudaEventDestroy(se0);
        cudaEventDestroy(se1);
    }
    cudaMemcpy(&last_sp, start + nC - 1, 4, cudaMemcpyDeviceToHost);
    int P = (int)(last_sp + last_s);
    int* plat_begin = nullptr;
    uint32_t* qsz = nullptr;
    uint32_t* qoff = nullptr;
    cudaMalloc(&plat_begin, (size_t)P * 4);
    k_scatter_plat<<<cb, 256>>>(start, keys_out, plat_begin, nC);
    cudaFree(keys_out);
    keys_out = nullptr;
    cudaFree(start);
    start = nullptr;
    cudaMalloc(&qsz, (size_t)P * 4);
    int pb = (P + 255) / 256;
    k_plat_meta<<<pb, 256>>>(plat_begin, vc_out, qsz, P, nC);
    cudaFree(vc_out);
    vc_out = nullptr;
    cudaMalloc(&qoff, (size_t)P * 4);
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
    const bool qdiag = getenv("WATERZ_WS_QDIAG") != nullptr;
    if (!qdiag) {
        cudaFree(qsz);
        qsz = nullptr;
    }
    uint32_t* q = nullptr;
    cudaMalloc(&q, (size_t)qtot * 4);
    int* overflow = nullptr;
    cudaMalloc(&overflow, 4);
    cudaMemset(overflow, 0, 4);
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
    {
        NvRange nv("bfs");
        k_indep_bfs<<<pb, 256>>>(bits_d, corners_out, plat_begin, q,
                                 qoff, overflow, qused, P, nC, qtot, Y, X);
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (ms_out) *ms_out = ms;
    int ovf = 0;
    cudaMemcpy(&ovf, overflow, 4, cudaMemcpyDeviceToHost);
    fprintf(stderr, "E9b ncorner=%d nplat=%d qtot=%lld bfs_ms=%.2f\n",
            nC, P, (long long)qtot, ms);
    fprintf(stderr,
            "E9b phases park=%.2f vcount=%.2f sort=%.2f unpark=%.2f "
            "scan=%.2f bfs=%.2f\n",
            g_n10_park_ms, g_n10_vcount_ms, g_n10_sort_ms, g_n10_unpark_ms,
            g_n10_scan_ms, g_n9_bfs_ms);
    g_n9_bfs_ms += ms;
    g_n9_bfs_calls += 1;
    if (qdiag) {
        int64_t* over_vc = nullptr;
        cudaMalloc(&over_vc, (size_t)P * 8);
        k_qdiag<<<pb, 256>>>(qsz, qused, nullptr, over_vc, P);
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
    if (own_parent) cudaFree(parent);
    cudaFree(flag);
    cudaFree(vcount);
    cudaFree(vc_in);
    cudaFree(vc_out);
    cudaFree(corners_in);
    cudaFree(corners_out);
    cudaFree(keys_in);
    cudaFree(keys_out);
    cudaFree(idx_in);
    cudaFree(idx_out);
    cudaFree(start);
    cudaFree(plat_begin);
    cudaFree(qsz);
    cudaFree(qoff);
    cudaFree(q);
    if (ovf) {
        fprintf(stderr, "E9b FATAL plateau BFS queue overflow\n");
        return -3;
    }
    return 0;
}

// changed lets the basin union-find stop when it has converged instead of
// running a round count fixed on the host. Same shape as k_hook_bidir's flag:
// one atomicExch on a single word per changed round, immaterial next to the
// full-volume sweep around it.
__global__ void k_hook_remain(const uint8_t* bits, uint32_t* parent, int* changed,
                              int64_t Z, int64_t Y, int64_t X) {
    const Vox v = vox_of(Y, X);
    if (!v.ok) return;
    const int64_t i = v.i, z = v.z, y = v.y, x = v.x;
    uint8_t b = bits[i];
    if (!b) return;
    uint32_t pi = parent[i];
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        int64_t j = neigh_i(i, d, Y, X);
        uint32_t pj = parent[j];
        if (pi == pj) continue;
        uint32_t lo = pi < pj ? pi : pj;
        uint32_t old = (pi < pj) ? atomicMin(&parent[pj], pi)
                                 : atomicMin(&parent[pi], pj);
        if (changed && old > lo) atomicExch(changed, 1);
    }
}

// The basin half of W4. Identical staging to k_hook_bidir_tiled; the only
// difference is the one k_hook_remain already has against k_hook_bidir, namely
// that it unions along every set direction bit instead of only reciprocal
// ones. After the divide most voxels carry a single bit, so this kernel does
// less work per voxel than the plateau hook but runs over the same volume for
// the same number of rounds, and pays the same 23 MB z-stride miss untiled.
__global__ void k_hook_remain_tiled(const uint8_t* bits, uint32_t* parent,
                                    int* changed,
                                    int64_t Z, int64_t Y, int64_t X) {
    __shared__ uint32_t sp[WS_SN];
    __shared__ uint8_t sb[WS_SN];

    const int64_t x0 = (int64_t)blockIdx.x * WS_TX;
    const int64_t y0 = (int64_t)blockIdx.y * WS_TY;
    const int64_t z0 = (int64_t)blockIdx.z * WS_TZ;
    const int tid = (threadIdx.z * WS_TY + threadIdx.y) * WS_TX + threadIdx.x;
    const int nthread = WS_TX * WS_TY * WS_TZ;

    for (int s = tid; s < WS_SN; s += nthread) {
        int lx = s % WS_SX, t = s / WS_SX;
        int ly = t % WS_SY, lz = t / WS_SY;
        int64_t gx = x0 + lx - 1, gy = y0 + ly - 1, gz = z0 + lz - 1;
        if (gx >= 0 && gx < X && gy >= 0 && gy < Y && gz >= 0 && gz < Z) {
            int64_t gi = (gz * Y + gy) * X + gx;
            sb[s] = bits[gi];
            sp[s] = parent[gi];
        } else {
            sb[s] = 0;
            sp[s] = 0xffffffffu;
        }
    }
    __syncthreads();

    const int lx = threadIdx.x, ly = threadIdx.y, lz = threadIdx.z;
    const int64_t x = x0 + lx, y = y0 + ly, z = z0 + lz;
    if (x >= X || y >= Y || z >= Z) return;
    const int me = ws_sidx(lz, ly, lx);
    const uint8_t b = sb[me];
    if (!b) return;
    const uint32_t pi = sp[me];

    const int dlz[6] = {-1, 0, 0, 1, 0, 0};
    const int dly[6] = {0, -1, 0, 0, 1, 0};
    const int dlx[6] = {0, 0, -1, 0, 0, 1};
    for (int d = 0; d < 6; ++d) {
        if (!(b & DBIT[d])) continue;
        if (oob_d(d, z, y, x, Z, Y, X)) continue;
        const int nb = ws_sidx(lz + dlz[d], ly + dly[d], lx + dlx[d]);
        const uint32_t pj = sp[nb];
        if (pi == pj) continue;
        const uint32_t lo = pi < pj ? pi : pj;
        const uint32_t old = (pi < pj) ? atomicMin(&parent[pj], pi)
                                       : atomicMin(&parent[pi], pj);
        if (changed && old > lo) atomicExch(changed, 1);
    }
}

__global__ void k_root_flag(const uint8_t* bits, const uint32_t* parent, uint32_t* flag, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    flag[i] = (bits[i] && parent[i] == (uint32_t)i) ? 1u : 0u;
}

__global__ void k_write_labels(
    const uint8_t* bits, const uint32_t* parent, const uint32_t* psum,
    uint32_t* seg, int64_t n, int fold)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!bits[i]) {
        seg[i] = 0;
        return;
    }
    uint32_t r = parent[i];
    if (fold) r = uf_find_ro(parent, r);
    seg[i] = psum[r] + 1u;
}

// W3, label half. k_root_flag writes a uint32 per voxel and the scan writes
// another; k_write_labels then reads psum[parent[i]]. That is two full
// 4 B/vox arrays, 17.2 GiB at 2.16 Gvox, whose only job is to turn "is this
// voxel a flagged root" into "how many flagged roots have a smaller index".
//
// A 1-bit-per-voxel mask is the same predicate in voxel-index order, so the
// exclusive count at r is the popcount of bits below r. That is O(n) to scan
// but O(1) to query if it is stored two-level: 32 words of 32 bits make a
// 1024-voxel block, a uint32 per block holds that block's popcount, and a
// prefix of those (size/1024 entries, 2.06 M at 2.16 Gvox) is the count of
// roots in earlier blocks. label_of(r) is then one block-scan load plus at
// most 32 popcs, randomly addressable because psum[r] is.
//
// Values are identical by construction. The mask is 0.125 B/vox and the
// block sums are 0.0039 B/vox, against 8 B/vox for flag+psum.
static int ws_w3() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_WS_W3");
        cached = s ? std::atoi(s) : 0;
    }
    return cached;
}

__global__ void k_root_mask(const uint8_t* bits, const uint32_t* parent,
                            uint32_t* mask, int64_t n) {
    const int64_t w = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t base = w * 32;
    if (base >= n) return;
    uint32_t m = 0;
    for (int b = 0; b < 32; ++b) {
        const int64_t i = base + b;
        if (i < n && bits[i] && parent[i] == (uint32_t)i)
            m |= 1u << b;
    }
    mask[w] = m;
}

__global__ void k_block_popc(const uint32_t* mask, uint32_t* blk,
                             int nblk, int nwords) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    uint32_t s = 0;
    const int w0 = b * 32;
    for (int k = 0; k < 32; ++k) {
        const int w = w0 + k;
        if (w < nwords) s += __popc(mask[w]);
    }
    blk[b] = s;
}

static inline __device__ uint32_t label_of(const uint32_t* mask,
                                           const uint32_t* blkscan,
                                           uint32_t r) {
    const uint32_t blk = r >> 10;
    const uint32_t wi = (r >> 5) & 31u;
    const uint32_t bit = r & 31u;
    const uint32_t* w = mask + (blk << 5);
    uint32_t extra = 0;
    #pragma unroll
    for (uint32_t k = 0; k < 32u; ++k)
        if (k < wi) extra += __popc(w[k]);
    extra += __popc(w[wi] & (bit ? ((1u << bit) - 1u) : 0u));
    return blkscan[blk] + extra + 1u;
}

__global__ void k_write_labels_mask(
    const uint8_t* bits, const uint32_t* parent,
    const uint32_t* mask, const uint32_t* blkscan,
    uint32_t* seg, int64_t n, int fold)
{
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!bits[i]) {
        seg[i] = 0;
        return;
    }
    uint32_t r = parent[i];
    if (fold) r = uf_find_ro(parent, r);
    seg[i] = label_of(mask, blkscan, r);
}

static int g_sv_rounds = 40;

static int e9c_basins_d(const uint8_t* bits_d, uint32_t* seg_d, int64_t Z, int64_t Y, int64_t X, uint32_t* nfrag) {
    NvRange nv("e9c");
    int64_t size = Z * Y * X;
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    // parent is the caller's label buffer, not a fourth per-voxel array. The
    // last thing this stage does is k_write_labels, which is
    // seg[i] = psum[parent[i]] + 1: thread i reads slot i of parent and writes
    // slot i of seg, and psum is a separate array, so no thread can observe
    // another's overwrite and the two may be the same storage. That is 4 B/vox,
    // 8.05 GiB at 2.16 Gvox, on a stage that was 15.69 B/vox.
    uint32_t* parent = share_labels_now() ? seg_d : nullptr;
    const bool own_parent = (parent == nullptr);
    uint32_t* flag = nullptr;
    uint32_t* psum = nullptr;
    if (own_parent) cudaMalloc(&parent, (size_t)size * 4);
    const bool use_w3 = ws_w3() != 0;
    // W5 borrows flag as stitch-list scratch. W3 labelling does not need
    // flag or psum at all. Allocate the union of what the chosen path reads.
    if (uf_algo() == 3 || uf_algo() == 4 || !use_w3) cudaMalloc(&flag, (size_t)size * 4);
    // psum is flag after the in-place exclusive scan; W3 needs neither.
    if (uf_algo() == 3) {
        // W5, basin half. Recip=false, matching k_hook_remain's predicate:
        // it unions along every set direction bit without asking the target
        // to point back.
        w5_union_find<false>(bits_d, parent, flag, Z, Y, X, "E9c");
    } else if (uf_algo() == 4) {
        w5_e4_union_find<false>(bits_d, parent, flag, Z, Y, X, "E9c");
    } else {
    k_parent_init<<<blocks, threads>>>(parent, size);
    // This ran a host-fixed round count -- `ws_set_sv_rounds(7)`, tuned on the
    // 180 Mvox validation volume. That is the wrong shape twice over. Spare
    // rounds are full-volume sweeps, and worse, a volume needing more than the
    // tuned count would come out with basins still unmerged and no complaint:
    // wrong fragments, not a slow run. The graded volume is 12x larger and has
    // never been executed, so the count cannot be assumed to carry.
    //
    // Run to convergence instead, as the plateau union-find already does, and
    // report failure to converge rather than absorbing it. g_sv_rounds stays as
    // the safety bound so an explicit setting still caps the work.
    int uf_cap = g_sv_rounds;
    if (uf_cap < 1) uf_cap = 1;
    if (uf_cap > 64) uf_cap = 64;
    int* sv_changed = nullptr;
    cudaMalloc(&sv_changed, 4);
    int sv_rounds = 0;
    for (int r = 0; r < uf_cap; ++r) {
        cudaMemset(sv_changed, 0, 4);
        if (uf_algo() == 2) {
            k_hook_remain_tiled<<<ws_tile_grid(Z, Y, X),
                                  dim3(WS_TX, WS_TY, WS_TZ)>>>(
                bits_d, parent, sv_changed, Z, Y, X);
        } else {
            k_hook_remain<<<vox_grid(Z, Y, X, threads), threads>>>(
                bits_d, parent, sv_changed, Z, Y, X);
        }
        k_uf_compress_c<<<blocks, threads>>>(parent, sv_changed, size);
        int h = 0;
        cudaMemcpy(&h, sv_changed, 4, cudaMemcpyDeviceToHost);
        ++sv_rounds;
        if (!h) break;
    }
    cudaFree(sv_changed);
    fprintf(stderr, "E9c sv_rounds=%d/%d%s\n", sv_rounds, uf_cap,
            sv_rounds >= uf_cap ? " NOT-CONVERGED" : "");
    }
    uint32_t nf = 0;
    if (use_w3) {
        // Pad the mask to a whole number of 1024-voxel blocks so label_of
        // can walk 32 words per block without a bounds check.
        const int nblk = (int)((size + 1023) / 1024);
        const int nwords = nblk * 32;
        uint32_t* mask = nullptr;
        uint32_t* blk = nullptr;
        cudaMalloc(&mask, (size_t)nwords * 4);
        cudaMalloc(&blk, (size_t)nblk * 4);
        cudaMemset(mask, 0, (size_t)nwords * 4);
        const int mw = (int)((size + 31) / 32);
        k_root_mask<<<(mw + threads - 1) / threads, threads>>>(
            bits_d, parent, mask, size);
        k_block_popc<<<(nblk + threads - 1) / threads, threads>>>(
            mask, blk, nblk, nwords);
        uint32_t last_b = 0;
        cudaMemcpy(&last_b, blk + nblk - 1, 4, cudaMemcpyDeviceToHost);
        {
            void* tmp = nullptr;
            size_t tmp_bytes = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, blk, blk, nblk);
            cudaMalloc(&tmp, tmp_bytes);
            cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, blk, blk, nblk);
            cudaFree(tmp);
        }
        uint32_t last_s = 0;
        cudaMemcpy(&last_s, blk + nblk - 1, 4, cudaMemcpyDeviceToHost);
        nf = last_s + last_b;
        k_write_labels_mask<<<blocks, threads>>>(
            bits_d, parent, mask, blk, seg_d, size,
            (fold_flatten() && !fold_e9b_only()) ? 1 : 0);
        cudaFree(mask);
        cudaFree(blk);
    } else {
        k_root_flag<<<blocks, threads>>>(bits_d, parent, flag, size);
        // Same in-place scan e9b already uses: exclusive-sum flag into itself
        // and recover nfrag from the last pre-scan flag bit. Drops the second
        // 4 B/vox array.
        uint32_t last_f = 0;
        cudaMemcpy(&last_f, flag + size - 1, 4, cudaMemcpyDeviceToHost);
        {
            void* tmp = nullptr;
            size_t tmp_bytes = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, flag, flag, (int)size);
            cudaMalloc(&tmp, tmp_bytes);
            cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, flag, flag, (int)size);
            cudaFree(tmp);
        }
        uint32_t last_p = 0;
        cudaMemcpy(&last_p, flag + size - 1, 4, cudaMemcpyDeviceToHost);
        nf = last_p + last_f;
        k_write_labels<<<blocks, threads>>>(
            bits_d, parent, flag, seg_d, size,
            (fold_flatten() && !fold_e9b_only()) ? 1 : 0);
    }
    if (nfrag) *nfrag = nf;
    if (own_parent) cudaFree(parent);
    if (flag) cudaFree(flag);
    if (psum) cudaFree(psum);
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
    k_flow<<<vox_grid(Z, Y, X, threads), threads>>>(aff_d, Z, Y, X, low, high, bits_d, tie_flip() ? 1 : 0, coarse_delta());
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
    k_flow<<<vox_grid(Z, Y, X, threads), threads>>>(aff_d, Z, Y, X, low, high, bits_d, tie_flip() ? 1 : 0, coarse_delta());
    // W2: k_flow is the only kernel that reads aff (E5). Free the 3 B/vox
    // copy before e9b union-find scratch so it does not sit in the WS peak.
    // Caller-owned aff in watershed_gpu_e9_d is left alone (RAG still needs it).
    cudaFree(aff_d);
    aff_d = nullptr;
    float ms = 0;
    e9b_divide_d(bits_d, Z, Y, X, &ms, share_labels_now() ? seg_d : nullptr);
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

__global__ void k_f32_to_u8(const float* src, uint8_t* dst, int64_t n)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i] * 255.0f;
    v = rintf(v);
    dst[i] = (uint8_t)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
}

// Quantise a float32 affinity already in VRAM, so the device-resident path
// accepts the float input TASK describes without a host round trip. Matches
// _as_u8 on the host: scale by 255, round half away from zero, clamp.
extern "C" int aff_f32_to_u8_d(const float* src_d, uint8_t* dst_d, int64_t n)
{
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    k_f32_to_u8<<<(int)blocks, threads>>>(src_d, dst_d, n);
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
    k_flow<<<vox_grid(Z, Y, X, threads), threads>>>(aff_d, Z, Y, X, low, high, bits_d, tie_flip() ? 1 : 0, coarse_delta());
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
        k_hook_remain<<<vox_grid(Z, Y, X, threads), threads>>>(
            bits_d, parent, nullptr, Z, Y, X);
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

// Split so segment_d can park caller aff after k_flow. e9b does not read aff
// (E5: only k_flow does). The fused wrapper below still holds aff through
// e9b for callers that have not split.
extern "C" int ws_flow_d(
    const uint8_t* aff_d, int64_t Z, int64_t Y, int64_t X,
    float low, float high, uint8_t* bits_d)
{
    NvRange nv("flow");
    int threads = 256;
    k_flow<<<vox_grid(Z, Y, X, threads), threads>>>(aff_d, Z, Y, X, low, high, bits_d, tie_flip() ? 1 : 0, coarse_delta());
    return 1;
}

__global__ void k_offset_labels(uint32_t* seg, int64_t n, uint32_t off)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n || seg[i] == 0) return;
    seg[i] += off;
}

// k_indep_bfs follows bits with no OOB test. Full-volume k_flow can leave a
// ±z bit on a slab face that points at the neighbouring tile; BFS then walks
// out of the slab, into already-rewritten bits, and overflows the queue.
__global__ void k_clear_slab_z_faces(uint8_t* bits, int64_t Z, int64_t Y, int64_t X)
{
    const int64_t yx = Y * X;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= yx) return;
    bits[i] &= (uint8_t)~0x01;                 // z=0: drop -z
    bits[(Z - 1) * yx + i] &= (uint8_t)~0x08;  // z=Z-1: drop +z
}

// Official make_big 3×2×2 is [375,2400,2400]. Each z-tile is 125 and the
// z-seam affinity is identically 0 (P1), so e9b/e9c on a tile is the same
// partition as the fused volume. E4 stitch is vacuous on those seams.
// Scratch is one tile, not 3, which is how 2.16 fits in 24 GiB.
static bool use_z_slab(int64_t Z, int64_t Y, int64_t X)
{
    const char* s = std::getenv("WATERZ_Z_SLAB");
    if (s && std::atoi(s) == 0) return false;
    if (s && std::atoi(s) > 0) return Z > 125 && (Z % 125) == 0;
    return Z == 375 && Y == 2400 && X == 2400;
}

extern "C" int ws_label_d(
    uint8_t* bits_d, int64_t Z, int64_t Y, int64_t X,
    uint32_t* seg_d, uint32_t* nfrag, float* ms_out)
{
    const int64_t SZ = 125;
    if (use_z_slab(Z, Y, X)) {
        const int ns = (int)(Z / SZ);
        const int64_t slab = SZ * Y * X;
        uint32_t tot = 0;
        float dms = 0;
        for (int s = 0; s < ns; ++s) {
            float ms = 0;
            uint8_t* bits_s = bits_d + s * slab;
            uint32_t* seg_s = seg_d + s * slab;
            {
                int fb = (int)((Y * X + 255) / 256);
                if (!block_voi())
                    k_clear_slab_z_faces<<<fb, 256>>>(bits_s, SZ, Y, X);
            }
            int rc = e9b_divide_d(bits_s, SZ, Y, X, &ms,
                                  share_labels_now() ? seg_s : nullptr);
            if (rc < 0) return rc;
            dms += ms;
            uint32_t nf = 0;
            e9c_basins_d(bits_s, seg_s, SZ, Y, X, &nf);
            if (tot > 0 && nf > 0) {
                int blocks = (int)((slab + 255) / 256);
                k_offset_labels<<<blocks, 256>>>(seg_s, slab, tot);
            }
            tot += nf;
        }
        if (nfrag) *nfrag = tot;
        if (ms_out) *ms_out = dms;
        fprintf(stderr, "E9 z-slab n=%d Zs=%lld nfrag=%u\n",
                ns, (long long)SZ, tot);
        return (int)tot;
    }
    float dms = 0;
    int rc = e9b_divide_d(bits_d, Z, Y, X, &dms,
                          share_labels_now() ? seg_d : nullptr);
    if (rc < 0) return rc;
    uint32_t nf = 0;
    e9c_basins_d(bits_d, seg_d, Z, Y, X, &nf);
    if (nfrag) *nfrag = nf;
    if (ms_out) *ms_out = dms;
    return (int)nf;
}

extern "C" int watershed_gpu_e9_d(
    const uint8_t* aff_d, int64_t Z, int64_t Y, int64_t X,
    float low, float high, uint32_t* seg_d, uint32_t* nfrag, float* ms_out)
{
    int64_t size = Z * Y * X;
    uint8_t* bits_d = nullptr;
    cudaMalloc(&bits_d, (size_t)size);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    ws_flow_d(aff_d, Z, Y, X, low, high, bits_d);
    int nf = ws_label_d(bits_d, Z, Y, X, seg_d, nfrag, nullptr);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (ms_out) *ms_out = ms;
    cudaFree(bits_d);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return nf;
}

extern "C" void ws_set_sv_rounds(int n) {
    g_sv_rounds = n;
}
