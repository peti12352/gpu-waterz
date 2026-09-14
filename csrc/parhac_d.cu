// Device paper-ParHAC ContractLayer, waterz contact-mean, eps=0.08.
// Live-subgraph compact + combine. Same control flow as parhac_paper_cpu.
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>

struct NvRange {
    explicit NvRange(const char* n) { nvtxRangePushA(n); }
    ~NvRange() { nvtxRangePop(); }
};

// Device allocation accounting; see the matching note in ws.cu.
//
// Only this file's explicit cudaMalloc calls are counted. thrust allocates its
// own scratch for sort_by_key and reduce_by_key through a separate path, so the
// reported figure is a lower bound on the agglomeration's true peak and is
// labelled as such by the reporting script.
static size_t g_agg_cur = 0;
static size_t g_agg_peak = 0;

struct AggAlloc { size_t bytes; int line; };
static std::map<void*, AggAlloc>& agg_book() {
    static std::map<void*, AggAlloc> m;
    return m;
}

// Which source lines held the memory at the moment of the peak. A single total
// says the agglomeration wants 17.6 GiB at 2.16 Gvox but not which of the ~30
// live buffers that is, and narrowing the wrong one saves nothing.
static std::map<int, size_t>& agg_peak_lines() {
    static std::map<int, size_t> m;
    return m;
}

static cudaError_t agg_tracked_malloc(void** p, size_t n, int line) {
    cudaError_t e = cudaMalloc(p, n);
    if (e == cudaSuccess && *p) {
        agg_book()[*p] = AggAlloc{n, line};
        g_agg_cur += n;
        if (g_agg_cur > g_agg_peak) {
            g_agg_peak = g_agg_cur;
            auto& snap = agg_peak_lines();
            snap.clear();
            for (const auto& kv : agg_book()) snap[kv.second.line] += kv.second.bytes;
        }
    }
    return e;
}

static cudaError_t agg_tracked_free(void* p) {
    auto it = agg_book().find(p);
    if (it != agg_book().end()) {
        g_agg_cur -= it->second.bytes;
        agg_book().erase(it);
    }
    return cudaFree(p);
}

static unsigned long long g_n9_dirty_e = 0;
static unsigned long long g_n9_dirty_u = 0;
static int g_n9_dirty_n = 0;
static int g_n9_dirty_e_max = 0;
static int g_n9_dirty_u_max = 0;

static void n9_dirty_hit(int ndirty, int nunique)
{
    if (ndirty < 0) ndirty = 0;
    if (nunique < 0) nunique = 0;
    g_n9_dirty_e += (unsigned long long)ndirty;
    g_n9_dirty_u += (unsigned long long)nunique;
    g_n9_dirty_n += 1;
    if (ndirty > g_n9_dirty_e_max) g_n9_dirty_e_max = ndirty;
    if (nunique > g_n9_dirty_u_max) g_n9_dirty_u_max = nunique;
}

extern "C" void agg_n9_reset(void) {
    g_n9_dirty_e = 0;
    g_n9_dirty_u = 0;
    g_n9_dirty_n = 0;
    g_n9_dirty_e_max = 0;
    g_n9_dirty_u_max = 0;
}

extern "C" void agg_n9_dirty(
    unsigned long long* dirty_e, unsigned long long* dirty_u, int* n,
    int* dirty_e_max, int* dirty_u_max)
{
    if (dirty_e) *dirty_e = g_n9_dirty_e;
    if (dirty_u) *dirty_u = g_n9_dirty_u;
    if (n) *n = g_n9_dirty_n;
    if (dirty_e_max) *dirty_e_max = g_n9_dirty_e_max;
    if (dirty_u_max) *dirty_u_max = g_n9_dirty_u_max;
}

enum { N21_INNER_CAP = 8192 };
static int g_n21_n = 0;
static int g_n21_nscan[N21_INNER_CAP];
static int g_n21_nlive[N21_INNER_CAP];
static int g_n21_ndirty[N21_INNER_CAP];
static int g_n21_nuniq[N21_INNER_CAP];
static int g_n21_nact[N21_INNER_CAP];
static int g_n21_ntab[N21_INNER_CAP];
static int g_n21_last_ndirty = 0;
static int g_n21_last_nuniq = 0;
static int g_n21_last_ntab = 0;

static void n21_note_combine(int ndirty, int nuniq, int ntab)
{
    g_n21_last_ndirty = ndirty;
    g_n21_last_nuniq = nuniq;
    g_n21_last_ntab = ntab;
}

static void n21_inner_push(int nscan, int nlive, int nact)
{
    if (g_n21_n >= N21_INNER_CAP) return;
    const int i = g_n21_n++;
    g_n21_nscan[i] = nscan;
    g_n21_nlive[i] = nlive;
    g_n21_ndirty[i] = g_n21_last_ndirty;
    g_n21_nuniq[i] = g_n21_last_nuniq;
    g_n21_nact[i] = nact;
    g_n21_ntab[i] = g_n21_last_ntab;
}

extern "C" void agg_n21_reset(void)
{
    g_n21_n = 0;
    g_n21_last_ndirty = 0;
    g_n21_last_nuniq = 0;
    g_n21_last_ntab = 0;
}

extern "C" int agg_n21_count(void) { return g_n21_n; }

extern "C" void agg_n21_get(
    int i, int* nscan, int* nlive, int* ndirty, int* nuniq, int* nact, int* ntab)
{
    if (i < 0 || i >= g_n21_n) {
        if (nscan) *nscan = 0;
        if (nlive) *nlive = 0;
        if (ndirty) *ndirty = 0;
        if (nuniq) *nuniq = 0;
        if (nact) *nact = 0;
        if (ntab) *ntab = 0;
        return;
    }
    if (nscan) *nscan = g_n21_nscan[i];
    if (nlive) *nlive = g_n21_nlive[i];
    if (ndirty) *ndirty = g_n21_ndirty[i];
    if (nuniq) *nuniq = g_n21_nuniq[i];
    if (nact) *nact = g_n21_nact[i];
    if (ntab) *ntab = g_n21_ntab[i];
}

extern "C" void agg_mem_reset(void) {
    g_agg_peak = g_agg_cur;
    agg_peak_lines().clear();
}
extern "C" size_t agg_mem_peak(void) { return g_agg_peak; }
extern "C" size_t agg_mem_cur(void) { return g_agg_cur; }

// Copy out the peak breakdown: line numbers and bytes, largest first is left to
// the caller. Returns how many entries exist, so a short buffer is detectable.
extern "C" int agg_mem_peak_lines(int* lines, size_t* bytes, int cap) {
    const auto& snap = agg_peak_lines();
    int i = 0;
    for (const auto& kv : snap) {
        if (i < cap) { lines[i] = kv.first; bytes[i] = kv.second; }
        ++i;
    }
    return i;
}

#define cudaMalloc(p, n) agg_tracked_malloc((void**)(p), (n), __LINE__)
#define cudaFree(p) agg_tracked_free((void*)(p))

#include <cub/cub.cuh>

struct KeepOn {
    __host__ __device__ bool operator()(uint8_t k) const { return k != 0; }
};

static inline __device__ uint32_t dfind_nocomp(const uint32_t* p, uint32_t x)
{
    while (p[x] != x) x = p[x];
    return x;
}

static inline __device__ uint32_t dfind(uint32_t* p, uint32_t x)
{
    uint32_t r = dfind_nocomp(p, x);
    while (p[x] != r) {
        uint32_t n = p[x];
        p[x] = r;
        x = n;
    }
    return r;
}

__global__ void k_compress(uint32_t* parent, int nnode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    parent[i] = dfind(parent, (uint32_t)i);
}

// V2: drop the sz[red] >= sz[blue] propose predicate. Default on, matching
// the locked fingerprint. WATERZ_SIZE_ASYM=0 is the accuracy-affecting
// lever; it is off the G bitmask on purpose.
static __device__ int d_size_asym = 1;

static void set_size_asym_from_env()
{
    int v = 1;
    if (const char* s = std::getenv("WATERZ_SIZE_ASYM"))
        v = std::atoi(s) != 0;
    cudaMemcpyToSymbol(d_size_asym, &v, sizeof(int));
}

__global__ void k_init_parent(uint32_t* parent, uint32_t* sz, int nnode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    parent[i] = (uint32_t)i;
    sz[i] = (i == 0) ? 0 : 1;
}

// Rescale contact sums into integral units of "affinity bytes" so that every
// subsequent accumulation is exact and therefore order-independent.
//
// StarMerge folds a merged edge's weight in with atomicAdd on a double. Double
// addition is not associative, so racing atomics give bit-different sums,
// which flips `mean < TL` for edges sitting near the layer threshold and makes
// the whole run nondeterministic. Affinities are uint8/255, so the true contact
// sum is k/255 for an integer k; storing k instead makes each partial sum an
// exactly representable integer, and atomicAdd on exact integers held in a
// double is order-independent.
//
// Headroom: k is bounded by 255 * 3 * nvox = 1.65e12 for the 2.16 Gvox volume,
// well inside the 2^53 = 9.0e15 exact-integer range of a double.
//
// Every threshold comparison in this file is relative (`sm/ct` against a TL
// derived from a wmax that is itself computed from `sm/ct`), so the only value
// needing a matching 255x is the externally supplied affinity threshold.
__global__ void k_scale_sm_bytes(double* sm, int64_t n)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    sm[i] = (double)llround(sm[i] * 255.0);
}

static const double SM_BYTE_SCALE = 255.0;

__global__ void k_wmax_live(
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    uint32_t* parent, int64_t n, double* blk)
{
    __shared__ double sh[256];
    double m = 0;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        if (ct[i] < 1) continue;
        uint32_t a = dfind_nocomp(parent, u[i]), b = dfind_nocomp(parent, v[i]);
        if (a == b || a == 0 || b == 0) continue;
        double mean = sm[i] / (double)ct[i];
        if (mean > m) m = mean;
    }
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s && sh[threadIdx.x + s] > sh[threadIdx.x])
            sh[threadIdx.x] = sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) blk[blockIdx.x] = sh[0];
}

__global__ void k_rewrite(
    uint32_t* u, uint32_t* v, const double* sm, const int64_t* ct,
    uint32_t* parent, int64_t n, uint8_t* keep, double TL)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ct[i] < 1) {
        keep[i] = 0;
        return;
    }
    uint32_t a = dfind_nocomp(parent, u[i]), b = dfind_nocomp(parent, v[i]);
    u[i] = a;
    v[i] = b;
    if (a == b || a == 0 || b == 0) {
        keep[i] = 0;
        return;
    }
    if (a > b) {
        u[i] = b;
        v[i] = a;
    }
    keep[i] = (sm[i] / (double)ct[i] >= TL) ? 1 : 0;
}

struct EdgeLess {
    __host__ __device__ bool operator()(
        const thrust::tuple<uint32_t, uint32_t, double, int64_t>& a,
        const thrust::tuple<uint32_t, uint32_t, double, int64_t>& b) const
    {
        uint32_t au = thrust::get<0>(a), av = thrust::get<1>(a);
        uint32_t bu = thrust::get<0>(b), bv = thrust::get<1>(b);
        if (au != bu) return au < bu;
        return av < bv;
    }
};

struct EdgeKeyEq {
    __host__ __device__ bool operator()(
        const thrust::tuple<uint32_t, uint32_t>& a,
        const thrust::tuple<uint32_t, uint32_t>& b) const
    {
        return thrust::get<0>(a) == thrust::get<0>(b)
            && thrust::get<1>(a) == thrust::get<1>(b);
    }
};

struct SumPair {
    __host__ __device__ thrust::tuple<double, int64_t> operator()(
        const thrust::tuple<double, int64_t>& a,
        const thrust::tuple<double, int64_t>& b) const
    {
        return thrust::make_tuple(
            thrust::get<0>(a) + thrust::get<0>(b),
            thrust::get<1>(a) + thrust::get<1>(b));
    }
};

__global__ void k_zero_sz(uint32_t* sz, int nnode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnode) sz[i] = 0;
}

__global__ void k_rebuild_sz(const uint32_t* parent, uint32_t* sz, int nnode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= 0 || i >= nnode) return;
    uint32_t r = dfind_nocomp(parent, (uint32_t)i);
    atomicAdd(&sz[r], 1u);
}

static inline __device__ uint8_t color_of(uint32_t i, uint64_t seed)
{
    // Must mix seed through multiply: XOR with id*odd is just (i^seed) parity,
    // so even-even leftover edges stay same-color forever.
    uint64_t x = (uint64_t)i * 0x9E3779B97F4A7C15ull;
    x ^= seed * 0xBF58476D1CE4E5B9ull;
    x ^= x >> 30;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 27;
    return (x & 1ull) ? 1 : 2;
}

// Proposal priority, seeded from edge CONTENT rather than array position.
//
// The original form hashed the edge's index `i`. Under E6s that is harmless,
// because compact_radix re-sorts the edge array by (u,v) every inner, so `i`
// is a deterministic function of the graph. Under E6t it is fatal: StarMerge
// picks the surviving slot for a merged edge by atomicCAS race, so `i` is
// race-assigned and every downstream merge decision inherits that randomness.
// Measured cost: two E6t runs on a byte-identical cached RAG disagreed on
// 531431 of 2175401 parents, violating TASK's byte-identical requirement.
//
// Hashing the canonical root pair instead makes the priority position-free.
// The returned value is used only as raw bits: it is packed into the high half
// of `prop` and compared by unsigned atomicMax, carried through `pris` via
// __uint_as_float/__float_as_uint, and used as radix sort key bits. It is
// never an operand of float arithmetic, so a full 31-bit hash is usable and
// gives ~2^31 tie space instead of the previous 24 bits.
static inline __device__ unsigned prop_pri_bits(
    uint64_t seed, uint32_t a, uint32_t b, uint32_t r)
{
    uint32_t lo = a < b ? a : b;
    uint32_t hi = a < b ? b : a;
    uint64_t h = ((uint64_t)lo << 32) | (uint64_t)hi;
    h ^= seed;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 29;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= (uint64_t)r * 0xD1B54A32D192ED03ull;
    h ^= h >> 32;
    return (unsigned)(h >> 32) & 0x7fffffffu;
}

__global__ void k_color(uint8_t* color, const uint32_t* parent, int nnode, uint64_t seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    if (i == 0 || parent[i] != (uint32_t)i) {
        color[i] = 0;
        return;
    }
    color[i] = color_of((uint32_t)i, seed);
}

__global__ void k_copy_sz(const uint32_t* sz, uint32_t* sz0, int nnode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnode) sz0[i] = sz[i];
}

// nabove counts live edges that are above TL and join two distinct roots. That
// condition is independent of the colouring, so nabove == 0 means no colouring
// whatsoever can produce a proposal and every remaining outer round of the
// layer is a no-op. The caller uses it to leave the layer early; see the outer
// loop in parhac_e6s_dev.
//
// It has to be reduced across the whole block before any thread returns, which
// is why the eligibility test is now computed into a flag instead of a chain of
// early returns. The proposal logic below is unchanged.
__global__ void k_propose(
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    uint32_t* parent, const uint32_t* sz, const uint8_t* color, const uint8_t* frozen,
    int64_t n, double TL, unsigned long long* prop, uint64_t seed, uint64_t cseed,
    int* dbg, int* nabove)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int above = 0;
    uint32_t a = 0, b = 0;
    if (i < n && ct[i] >= 1) {
        if (dbg) atomicAdd(dbg + 0, 1);
        double mean = sm[i] / (double)ct[i];
        if (mean >= TL) {
            if (dbg) atomicAdd(dbg + 1, 1);
            a = dfind_nocomp(parent, u[i]);
            b = dfind_nocomp(parent, v[i]);
            if (a != b && a != 0 && b != 0) {
                if (dbg) atomicAdd(dbg + 2, 1);
                above = 1;
            }
        }
    }
    if (nabove) {
        // One shared counter per block, one global atomic per block. nabove is
        // a kernel argument so the branch is block-uniform and the barriers
        // below are reached by every thread.
        __shared__ int sh_above;
        if (threadIdx.x == 0) sh_above = 0;
        __syncthreads();
        if (above) atomicAdd(&sh_above, 1);
        __syncthreads();
        if (threadIdx.x == 0 && sh_above) atomicAdd(nabove, sh_above);
    }
    if (!above) return;
    uint8_t ca = color_of(a, cseed);
    uint8_t cb = color_of(b, cseed);
    (void)color;
    (void)frozen;
    if (ca == cb) return;
    if (dbg) atomicAdd(dbg + 3, 1);
    uint32_t r = (ca == 1) ? a : b;
    uint32_t bl = (ca == 1) ? b : a;
    (void)color;
    if (frozen[r]) return;
    if (d_size_asym && sz[r] < sz[bl]) return;
    if (dbg) atomicAdd(dbg + 4, 1);
    unsigned long long pack =
        ((unsigned long long)prop_pri_bits(seed, a, b, r) << 32)
        | (unsigned long long)r;
    atomicMax(&prop[bl], pack);
}

// E6w: same propose, first writer of a blue records it for listed pack/memset.
__global__ void k_propose_list(
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    uint32_t* parent, const uint32_t* sz, const uint8_t* frozen,
    int64_t n, double TL, unsigned long long* prop, uint32_t* blist, int* nlist,
    uint64_t seed, uint64_t cseed)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n || ct[i] < 1) return;
    double mean = sm[i] / (double)ct[i];
    if (mean < TL) return;
    uint32_t a = dfind_nocomp(parent, u[i]), b = dfind_nocomp(parent, v[i]);
    if (a == b || a == 0 || b == 0) return;
    uint8_t ca = color_of(a, cseed);
    uint8_t cb = color_of(b, cseed);
    if (ca == cb) return;
    uint32_t r = (ca == 1) ? a : b;
    uint32_t bl = (ca == 1) ? b : a;
    if (frozen[r]) return;
    if (d_size_asym && sz[r] < sz[bl]) return;
    unsigned long long pack =
        ((unsigned long long)prop_pri_bits(seed, a, b, r) << 32)
        | (unsigned long long)r;
    unsigned long long old = atomicMax(&prop[bl], pack);
    if (old == 0 && pack != 0) {
        int slot = atomicAdd(nlist, 1);
        blist[slot] = bl;
    }
}

__global__ void k_pack_prop(
    const unsigned long long* prop, const uint32_t* sz, int nnode,
    uint32_t* reds, uint32_t* blues, uint32_t* addsz, float* pris, int* nprop)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= 0 || i >= nnode) return;
    unsigned long long p = prop[i];
    if (p == 0) return;
    uint32_t r = (uint32_t)(p & 0xffffffffull);
    if (r == 0) return;
    int slot = atomicAdd(nprop, 1);
    reds[slot] = r;
    blues[slot] = (uint32_t)i;
    addsz[slot] = sz[i];
    pris[slot] = __uint_as_float((unsigned)(p >> 32));
}

// Packing for a single fused radix sort, replacing two comparison sorts over
// zip iterators. k_accept_reds needs only proposals grouped by red with each
// group in descending priority, and it never reads the priority itself, so the
// sort key and payload can be built directly:
//
//     key = red : (0x7fffffff - priority)      payload = blue : size
//
// One ascending 64-bit sort then produces exactly that order. The priority is
// a 31-bit hash from prop_pri_bits, so the subtraction is injective and simply
// reverses rank within a group.
//
// This also retires a latent hazard. The old path stored the priority as
// __uint_as_float of that hash and compared the results as floats, but bit
// patterns in 0x7f800000-0x7fffffff are inf or NaN, and NaN comparisons are
// false, leaving their relative order undefined. Ordering the hash as the
// integer it is has a defined total order. It is a different tie order, so
// this changes which merges happen and has to be re-graded on VOI rather than
// checked for bit-equality.
// Collects the winning proposals, and clears the proposal array behind it.
//
// The loop used to cudaMemset prop over all nnode before every propose: 8 bytes
// per node, 17.4 MB per inner iteration at val, ~24 GB across a threshold. This
// pass already reads every entry, so resetting the non-empty ones costs a store
// on those alone, the same trade k_hash_emit makes with the hash table.
// Entries are only ever written by an atomicMax from zero, so an entry that is
// zero here was never touched and needs no reset.
__global__ void k_pack_prop_fused(
    unsigned long long* prop, const uint32_t* sz, int nnode,
    unsigned long long* key, unsigned long long* pay, int* nprop)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= 0 || i >= nnode) return;
    unsigned long long p = prop[i];
    if (p == 0) return;
    prop[i] = 0;
    uint32_t r = (uint32_t)(p & 0xffffffffull);
    if (r == 0) return;
    unsigned pri = (unsigned)(p >> 32) & 0x7fffffffu;
    int slot = atomicAdd(nprop, 1);
    key[slot] = ((unsigned long long)r << 32)
              | (unsigned long long)(0x7fffffffu - pri);
    pay[slot] = ((unsigned long long)i << 32) | (unsigned long long)sz[i];
}

// G3, blue-list half. k_propose plus a record of which blues were proposed to,
// and the fused pack restricted to that record.
//
// k_pack_prop_fused scans all nnode dprop slots to collect a mean of 2584
// winners, which is 12 B x 26.1 M per inner iteration at the graded volume to
// find 0.01% of it. The set it is looking for is exactly the blues that
// k_propose wrote, and k_propose can say so for free: an entry is only ever
// raised from zero by atomicMax, so the thread that observes old == 0 is the
// unique first writer of that blue and can append it to a list.
//
// Bit-identical, not approximately so. The list is the exact support of the
// non-zero dprop entries, so the packed set is the same set. Its ORDER differs,
// but so does the order the existing kernel produces: both assign output slots
// by atomicAdd, and the radix sort that follows is keyed on
// (red, 0x7fffffff - priority) where the priority is a 31-bit hash of the
// canonical root pair. Order therefore only matters for exact key ties, which
// need a hash collision within one red's group, and the existing path is
// equally exposed to those. a2_determinism.json records byte-identical output
// across runs, so they do not occur here.
//
// Deliberately separate kernels rather than a nullptr branch inside k_propose:
// k_propose is the one kernel already measured to be at roofline (77 ms of
// 5809), and it is not worth spending a register on it to save thirty lines.
__global__ void k_propose_bluelist(
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    uint32_t* parent, const uint32_t* sz, const uint8_t* frozen,
    int64_t n, double TL, unsigned long long* prop,
    uint32_t* blist, int* nlist,
    uint64_t seed, uint64_t cseed, int* nabove)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int above = 0;
    uint32_t a = 0, b = 0;
    if (i < n && ct[i] >= 1) {
        double mean = sm[i] / (double)ct[i];
        if (mean >= TL) {
            a = dfind_nocomp(parent, u[i]);
            b = dfind_nocomp(parent, v[i]);
            if (a != b && a != 0 && b != 0) above = 1;
        }
    }
    // Same block-aggregated count as k_propose, and for the same reason: the
    // layer-exit test needs it and a per-thread atomic would dominate.
    if (nabove) {
        __shared__ int sh_above;
        if (threadIdx.x == 0) sh_above = 0;
        __syncthreads();
        if (above) atomicAdd(&sh_above, 1);
        __syncthreads();
        if (threadIdx.x == 0 && sh_above) atomicAdd(nabove, sh_above);
    }
    if (!above) return;
    uint8_t ca = color_of(a, cseed);
    uint8_t cb = color_of(b, cseed);
    if (ca == cb) return;
    uint32_t r = (ca == 1) ? a : b;
    uint32_t bl = (ca == 1) ? b : a;
    if (frozen[r]) return;
    if (d_size_asym && sz[r] < sz[bl]) return;
    unsigned long long pack =
        ((unsigned long long)prop_pri_bits(seed, a, b, r) << 32)
        | (unsigned long long)r;
    unsigned long long old = atomicMax(&prop[bl], pack);
    if (old == 0) {
        blist[atomicAdd(nlist, 1)] = bl;
    }
}

// The k_pack_prop_fused body over the recorded blues instead of over nnode.
// Clearing prop[b] here preserves the invariant the loop relies on, that dprop
// arrives empty at the next propose without ever being memset.
// Grid-stride over the list, so the launch can be a small fixed grid. Sizing it
// from the count would need a device-to-host copy between propose and pack, and
// the whole point is to stop touching nnode-sized things per inner iteration.
__global__ void k_pack_listed_fused(
    const uint32_t* blist, const int* nlist_d, unsigned long long* prop,
    const uint32_t* sz, unsigned long long* key, unsigned long long* pay,
    int* nprop)
{
    const int nlist = nlist_d ? *nlist_d : 0;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nlist;
         i += stride) {
        uint32_t b = blist[i];
        if (b == 0) continue;
        unsigned long long p = prop[b];
        if (p == 0) continue;
        prop[b] = 0;
        uint32_t r = (uint32_t)(p & 0xffffffffull);
        if (r == 0) continue;
        unsigned pri = (unsigned)(p >> 32) & 0x7fffffffu;
        int slot = atomicAdd(nprop, 1);
        key[slot] = ((unsigned long long)r << 32)
                  | (unsigned long long)(0x7fffffffu - pri);
        pay[slot] = ((unsigned long long)b << 32) | (unsigned long long)sz[b];
    }
}

// One-launch proposal sort for the common small case. A 64-bit CUB device
// radix sort is eight passes of a couple of kernels each, and the proposal
// count has a median of 0 and a mean of about 2700, so nearly all of that
// call's cost is launches rather than sorting. A single block sorts up to
// BLOCK_T * ITEMS keys in shared memory with one launch.
//
// Padding with all-ones sorts the empty slots to the end, leaving the first
// nprop entries in the same order the device sort produces: both are stable
// radix sorts, so equal keys keep their input order in either.
template <int BLOCK_T, int ITEMS>
__global__ void k_sort_prop_block(
    const unsigned long long* key_in, const unsigned long long* pay_in,
    unsigned long long* key_out, unsigned long long* pay_out, int nprop)
{
    using BlockSort = cub::BlockRadixSort<
        unsigned long long, BLOCK_T, ITEMS, unsigned long long>;
    __shared__ typename BlockSort::TempStorage tmp;
    unsigned long long k[ITEMS], p[ITEMS];
    const int base = (int)threadIdx.x * ITEMS;
    for (int j = 0; j < ITEMS; ++j) {
        int i = base + j;
        k[j] = i < nprop ? key_in[i] : ~0ull;
        p[j] = i < nprop ? pay_in[i] : 0ull;
    }
    BlockSort(tmp).Sort(k, p);
    for (int j = 0; j < ITEMS; ++j) {
        int i = base + j;
        if (i < nprop) {
            key_out[i] = k[j];
            pay_out[i] = p[j];
        }
    }
}

static const int PSORT_BLOCK_T = 256;
static const int PSORT_ITEMS = 8;
static const int PSORT_BLOCK_CAP = PSORT_BLOCK_T * PSORT_ITEMS;

__global__ void k_unpack_prop(
    const unsigned long long* key, const unsigned long long* pay,
    uint32_t* reds, uint32_t* blues, uint32_t* addsz, int nprop)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nprop) return;
    reds[i] = (uint32_t)(key[i] >> 32);
    blues[i] = (uint32_t)(pay[i] >> 32);
    addsz[i] = (uint32_t)(pay[i] & 0xffffffffull);
}

__global__ void k_accept_serial(
    const uint32_t* reds, const uint32_t* blues, const uint32_t* addsz,
    int nprop, uint32_t* parent, uint32_t* sz, const uint8_t* frozen,
    double eps, int* nmerge)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int i = 0;
    while (i < nprop) {
        uint32_t r0 = reds[i];
        int j = i + 1;
        while (j < nprop && reds[j] == r0) ++j;
        uint32_t r = dfind_nocomp(parent, r0);
        if (!frozen[r0] && r != 0) {
            uint64_t cap = (uint64_t)llround((double)sz[r] * eps);
            if (eps > 0 && cap == 0) cap = 1;
            uint64_t acc = 0;
            for (int k = i; k < j; ++k) {
                uint32_t b = dfind_nocomp(parent, blues[k]);
                if (b == r || b == 0) continue;
                uint32_t add = addsz[k];
                acc += add;
                parent[b] = r;
                sz[r] += add;
                ++(*nmerge);
                if (acc > cap) break;
            }
        }
        i = j;
    }
}

// S33: one thread per red group. Same take=k+1 as k_accept_serial / CPU.
__global__ void k_accept_reds(
    const uint32_t* reds, const uint32_t* blues, const uint32_t* addsz,
    int nprop, uint32_t* parent, uint32_t* sz, const uint8_t* frozen,
    double eps, int* nmerge)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nprop) return;
    if (i > 0 && reds[i] == reds[i - 1]) return;
    uint32_t r0 = reds[i];
    int j = i + 1;
    while (j < nprop && reds[j] == r0) ++j;
    uint32_t r = dfind_nocomp(parent, r0);
    if (frozen[r0] || r == 0) return;
    uint64_t cap = (uint64_t)llround((double)sz[r] * eps);
    if (eps > 0 && cap == 0) cap = 1;
    uint64_t acc = 0;
    int local = 0;
    for (int k = i; k < j; ++k) {
        uint32_t b = dfind_nocomp(parent, blues[k]);
        if (b == r || b == 0) continue;
        uint32_t add = addsz[k];
        acc += add;
        parent[b] = r;
        sz[r] += add;
        ++local;
        if (acc > cap) break;
    }
    if (local) atomicAdd(nmerge, local);
}

__global__ void k_mark_dirty(
    const uint32_t* blues, const uint32_t* reds, int nprop, const uint32_t* parent,
    uint8_t* dirty)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nprop) return;
    uint32_t b = blues[i];
    uint32_t r = reds[i];
    if (parent[b] != b) dirty[b] = 1;
    dirty[r] = 1;
}

__global__ void k_count_roots(const uint32_t* parent, int nnode, int* nact)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= 0 || i >= nnode) return;
    if (parent[i] == (uint32_t)i) atomicAdd(nact, 1);
}

// P0z: blues that actually merged this accept (parent[b] != b).
__global__ void k_mark_acc_blue(
    const uint32_t* blues, const uint32_t* reds, int nprop, const uint32_t* parent,
    uint8_t* dirty_blue, uint8_t* dirty_star)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nprop) return;
    uint32_t b = blues[i];
    uint32_t r = reds[i];
    if (b == 0 || parent[b] == b) return;
    dirty_blue[b] = 1;
    dirty_star[b] = 1;
    dirty_star[r] = 1;
}

__global__ void k_unmark_acc_blue(
    const uint32_t* blues, const uint32_t* reds, int nprop, const uint32_t* parent,
    uint8_t* dirty_blue, uint8_t* dirty_star)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nprop) return;
    uint32_t b = blues[i];
    uint32_t r = reds[i];
    if (b == 0 || parent[b] == b) return;
    dirty_blue[b] = 0;
    dirty_star[b] = 0;
    dirty_star[r] = 0;
}

__global__ void k_count_dirty_edges(
    const uint32_t* u, const uint32_t* v, const int64_t* ct, int64_t n,
    const uint8_t* dirty_blue, const uint8_t* dirty_star, int* n_blue, int* n_star)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n || ct[i] < 1) return;
    uint32_t a = u[i], b = v[i];
    if (dirty_blue[a] || dirty_blue[b]) atomicAdd(n_blue, 1);
    if (dirty_star[a] || dirty_star[b]) atomicAdd(n_star, 1);
}

// ---------------------------------------------------------------------------
// G2: rewrite and combine only the dirty edges, leave the rest in place.
// ---------------------------------------------------------------------------
//
// hash_combine_live inserts every live edge into the table every inner
// iteration. The collision lemma in scripts/g0_agg_ref.py is that a
// parallel-edge collision can only occur between two dirty edges, a
// non-dirty edge has both endpoints unchanged, so its key cannot meet a
// rewritten one unless that rewritten one is also incident to a dirty
// endpoint, in which case it is dirty too. Combined with k_scale_sm_bytes
// (sums are exact integers, so atomicAdd commutes), restricting the table
// to the dirty set is therefore bit-identical, not approximate.
//
// The CPU replica finds that set by a masked scan of the alive edges, not
// by walking a CSR: after the first rewrite, u,v hold current roots, and
// the dirty flags are on this inner's merged blues and receiving reds,
// which are also roots. A CSR built at layer entry would be indexed by
// then-current endpoints and would miss an edge that was rewritten onto a
// dirty root in an earlier inner, unless it was rebuilt or spliced. The
// scan of current endpoints is the same selection the replica uses and
// does not have that bug. k_deg / k_scatter_adj stay available for a
// later splice; they are not what makes this pass correct.
//
// Dead slots are marked ct = 0. Physical compaction stays at layer entry
// (compact_radix already skips ct < 1). nscan is the high-water mark the
// subsequent propose / wmax / compact_radix have to cover; nlive is the
// count of ct >= 1, which is what the nlive trace records.

__global__ void k_rewrite_dirty(
    uint32_t* u, uint32_t* v, int64_t* ct, const uint32_t* parent,
    const uint8_t* dirty, int64_t n, uint8_t* keep)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    keep[i] = 0;
    if (ct[i] < 1) return;
    uint32_t a = u[i], b = v[i];
    if (!dirty[a] && !dirty[b]) return;
    a = dfind_nocomp(parent, a);
    b = dfind_nocomp(parent, b);
    if (a == b || a == 0 || b == 0) {
        ct[i] = 0;
        return;
    }
    if (a > b) {
        uint32_t t = a;
        a = b;
        b = t;
    }
    u[i] = a;
    v[i] = b;
    keep[i] = 1;
}

// N15 exp6: record vacated (keep) slots as holes during rewrite so
// k_count_u8 and k_list_holes are not separate O(nscan) launches.
__global__ void k_rewrite_dirty_fuse(
    uint32_t* u, uint32_t* v, int64_t* ct, const uint32_t* parent,
    const uint8_t* dirty, int64_t n, uint8_t* keep,
    uint32_t* holes, int* nhole, int* nself, uint8_t* amask)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    keep[i] = 0;
    if (ct[i] < 1) return;
    uint32_t a = u[i], b = v[i];
    if (!dirty[a] && !dirty[b]) return;
    a = dfind_nocomp(parent, a);
    b = dfind_nocomp(parent, b);
    if (a == b || a == 0 || b == 0) {
        ct[i] = 0;
        if (amask) amask[i] = 0;
        if (nself) atomicAdd(nself, 1);
        return;
    }
    if (a > b) {
        uint32_t t = a;
        a = b;
        b = t;
    }
    u[i] = a;
    v[i] = b;
    keep[i] = 1;
    int s = atomicAdd(nhole, 1);
    holes[s] = (uint32_t)i;
}

// N23 A1: same fuse body over an index list (CSR dirty set). Not e6t_rebuild.
__global__ void k_rewrite_dirty_fuse_listed(
    const uint32_t* elist, int nlist,
    uint32_t* u, uint32_t* v, int64_t* ct, const uint32_t* parent,
    uint8_t* keep, uint32_t* holes, int* nhole, int* nself, uint8_t* amask)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nlist) return;
    uint32_t i = elist[t];
    keep[i] = 0;
    if (ct[i] < 1) return;
    uint32_t a = u[i], b = v[i];
    a = dfind_nocomp(parent, a);
    b = dfind_nocomp(parent, b);
    if (a == b || a == 0 || b == 0) {
        ct[i] = 0;
        if (amask) amask[i] = 0;
        if (nself) atomicAdd(nself, 1);
        return;
    }
    if (a > b) {
        uint32_t tswap = a;
        a = b;
        b = tswap;
    }
    u[i] = a;
    v[i] = b;
    keep[i] = 1;
    int s = atomicAdd(nhole, 1);
    holes[s] = (uint32_t)i;
}

// N23 A1: prepend-build incidence lists at layer entry. nxt[2*i+0] is u's
// next, nxt[2*i+1] is v's next. Matches scripts/g0_agg_ref.py _csr_build.
__global__ void k_csr_reset_head(int* head, int nnode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnode) head[i] = -1;
}

__global__ void k_csr_scatter(
    int nscan, const uint32_t* u, const uint32_t* v, const int64_t* ct,
    int* head, int* nxt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nscan || ct[i] < 1) return;
    int a = (int)u[i], b = (int)v[i];
    nxt[2 * i + 0] = atomicExch(head + a, 2 * i + 0);
    nxt[2 * i + 1] = atomicExch(head + b, 2 * i + 1);
}

// Unique edge ids incident on dirty roots. gen/g avoids memset of n_edges.
__global__ void k_csr_gather(
    const int* nodes, int n_nodes, const int* head, const int* nxt,
    const uint32_t* u, const uint32_t* v, const int64_t* ct,
    uint32_t* gen, uint32_t g, uint32_t* elist, int* nlist)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_nodes) return;
    int x = nodes[t];
    for (int p = head[x]; p >= 0; p = nxt[p]) {
        int eid = p >> 1;
        if (ct[eid] < 1) continue;
        if ((int)u[eid] != x && (int)v[eid] != x) continue;
        uint32_t old = gen[eid];
        if (old == g) continue;
        if (atomicCAS(gen + eid, old, g) == old) {
            int slot = atomicAdd(nlist, 1);
            elist[slot] = (uint32_t)eid;
        }
    }
}

// Vertex-disjoint matching except star: one red may take several blues.
// atomicExch concatenates chains; walkers run in a later kernel.
__global__ void k_csr_splice(
    const uint32_t* blues, const uint32_t* reds, int nprop, const uint32_t* parent,
    int* head, int* nxt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nprop) return;
    uint32_t b = blues[i];
    uint32_t r = reds[i];
    if (b == 0 || parent[b] == b || b == r) return;
    int hb = head[(int)b];
    if (hb < 0) return;
    int p = hb;
    while (nxt[p] >= 0) p = nxt[p];
    int old = atomicExch(head + (int)r, hb);
    nxt[p] = old;
    head[(int)b] = -1;
}

__global__ void k_vacate_kept(int64_t* ct, const uint8_t* keep, int64_t n)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n && keep[i]) ct[i] = 0;
}

__global__ void k_list_holes(const int64_t* ct, int64_t n,
                             uint32_t* holes, int* nhole)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n || ct[i] >= 1) return;
    int s = atomicAdd(nhole, 1);
    holes[s] = (uint32_t)i;
}

__global__ void k_fill_holes(
    const uint32_t* su, const uint32_t* sv, const double* ssm, const int64_t* sct,
    const uint32_t* holes, int nfill,
    uint32_t* u, uint32_t* v, double* sm, int64_t* ct)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nfill) return;
    uint32_t d = holes[j];
    u[d] = su[j];
    v[d] = sv[j];
    sm[d] = ssm[j];
    ct[d] = sct[j];
}

__global__ void k_count_live(const int64_t* ct, int64_t n, int* nout)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n && ct[i] >= 1) atomicAdd(nout, 1);
}

// G3: propose over an explicit edge list (the active set), not [0, nscan).
__global__ void k_propose_listed(
    const uint32_t* elist, int nlist,
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    uint32_t* parent, const uint32_t* sz, const uint8_t* frozen,
    double TL, unsigned long long* prop, uint32_t* blist, int* nblist,
    uint64_t seed, uint64_t cseed, int* nabove)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int above = 0;
    uint32_t a = 0, b = 0;
    if (t < nlist) {
        uint32_t i = elist[t];
        if (ct[i] >= 1) {
            double mean = sm[i] / (double)ct[i];
            if (mean >= TL) {
                a = dfind_nocomp(parent, u[i]);
                b = dfind_nocomp(parent, v[i]);
                if (a != b && a != 0 && b != 0) above = 1;
            }
        }
    }
    if (nabove) {
        __shared__ int sh;
        if (threadIdx.x == 0) sh = 0;
        __syncthreads();
        if (above) atomicAdd(&sh, 1);
        __syncthreads();
        if (threadIdx.x == 0 && sh) atomicAdd(nabove, sh);
    }
    if (!above) return;
    uint8_t ca = color_of(a, cseed);
    uint8_t cb = color_of(b, cseed);
    if (ca == cb) return;
    uint32_t r = (ca == 1) ? a : b;
    uint32_t bl = (ca == 1) ? b : a;
    if (frozen[r]) return;
    if (d_size_asym && sz[r] < sz[bl]) return;
    unsigned long long pack =
        ((unsigned long long)prop_pri_bits(seed, a, b, r) << 32)
        | (unsigned long long)r;
    unsigned long long old = atomicMax(&prop[bl], pack);
    if (blist && old == 0 && pack != 0) {
        int slot = atomicAdd(nblist, 1);
        blist[slot] = bl;
    }
}

__global__ void k_rebuild_active(
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    const uint8_t* keep, const uint8_t* dirty, int64_t n, double TL,
    uint32_t* alist, int* nact, uint8_t* amask)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    // A non-dirty live edge cannot have crossed TL: only a dirty edge's
    // mean changes. So this kernel only touches dirty slots, and the
    // caller is expected to have dropped the vacated ones from amask
    // already. keep==1 is a dirty survivor whose new mean we re-test;
    // keep==0 and dirty endpoints with ct==0 is a dead slot we drop.
    uint32_t a = u[i], b = v[i];
    const bool was = dirty[a] || dirty[b] || keep[i];
    if (!was) return;
    if (ct[i] < 1) {
        amask[i] = 0;
        return;
    }
    const uint8_t on = (sm[i] / (double)ct[i] >= TL) ? 1 : 0;
    amask[i] = on;
}

__global__ void k_pack_amask(const uint8_t* amask, const int64_t* ct,
                             int64_t n, uint32_t* alist, int* nact)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n || !amask[i] || ct[i] < 1) return;
    int s = atomicAdd(nact, 1);
    alist[s] = (uint32_t)i;
}

// N21 A2: listed rebuild. keep is a packed-this-inner flag (0/1), not the
// rewrite keep bit: caller zeros it on old alist or emit holes first.
__global__ void k_keep_zero_listed(const uint32_t* idx, int n, uint8_t* keep)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    keep[idx[t]] = 0;
}

__global__ void k_retest_holes(
    const uint32_t* holes, int nfill,
    const double* sm, const int64_t* ct, double TL, uint8_t* amask)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nfill) return;
    const uint32_t i = holes[j];
    if (ct[i] < 1) {
        amask[i] = 0;
        return;
    }
    amask[i] = (sm[i] / (double)ct[i] >= TL) ? 1 : 0;
}

__global__ void k_pack_listed(
    const uint32_t* src, int nsrc,
    const uint8_t* amask, const int64_t* ct, uint8_t* keep,
    uint32_t* dst, int* nact)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nsrc) return;
    const uint32_t i = src[t];
    if (!amask[i] || ct[i] < 1) return;
    int s = atomicAdd(nact, 1);
    dst[s] = i;
    keep[i] = 1;
}

__global__ void k_pack_listed_new(
    const uint32_t* holes, int nfill,
    const uint8_t* amask, const int64_t* ct, uint8_t* keep,
    uint32_t* dst, int* nact)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nfill) return;
    const uint32_t i = holes[j];
    if (!amask[i] || ct[i] < 1 || keep[i]) return;
    int s = atomicAdd(nact, 1);
    dst[s] = i;
    keep[i] = 1;
}

__global__ void k_rebuild_pack(
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    const uint8_t* keep, const uint8_t* dirty, int64_t n, double TL,
    uint32_t* alist, int* nact, uint8_t* amask)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t a = u[i], b = v[i];
    const bool was = dirty[a] || dirty[b] || keep[i];
    if (!was) {
        if (amask[i] && ct[i] >= 1) {
            int s = atomicAdd(nact, 1);
            alist[s] = (uint32_t)i;
        }
        return;
    }
    if (ct[i] < 1) {
        amask[i] = 0;
        return;
    }
    const uint8_t on = (sm[i] / (double)ct[i] >= TL) ? 1 : 0;
    amask[i] = on;
    if (on) {
        int s = atomicAdd(nact, 1);
        alist[s] = (uint32_t)i;
    }
}

__global__ void k_init_amask(const double* sm, const int64_t* ct,
                             int64_t n, double TL, uint8_t* amask)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    amask[i] = (ct[i] >= 1 && sm[i] / (double)ct[i] >= TL) ? 1 : 0;
}

// G4: listed variants of the per-outer node passes.
__global__ void k_flag_roots(const uint32_t* parent, int nnode, uint32_t* flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    flag[i] = (i != 0 && parent[i] == (uint32_t)i) ? 1u : 0u;
}

__global__ void k_scatter_roots(const uint32_t* ps, uint32_t last_f,
                                uint32_t* roots, int nnode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    uint32_t here = ps[i];
    bool is = (i + 1 < nnode) ? (ps[i + 1] > here) : (last_f != 0);
    if (is) roots[here] = (uint32_t)i;
}

__global__ void k_zero_sz_list(uint32_t* sz, const uint32_t* roots, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) sz[roots[t]] = 0;
}

__global__ void k_copy_sz_list(const uint32_t* sz, uint32_t* sz0,
                               const uint32_t* roots, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) sz0[roots[t]] = sz[roots[t]];
}

__global__ void k_clear_frozen_list(uint8_t* frozen, const uint32_t* roots, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) frozen[roots[t]] = 0;
}

__global__ void k_compress_list(uint32_t* parent, const uint32_t* list, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint32_t i = list[t];
    parent[i] = dfind(parent, i);
}

__global__ void k_init_root_list(uint32_t* roots, int nnode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + 1 >= nnode) return;
    roots[i] = (uint32_t)(i + 1);
}

__global__ void k_compact_roots(
    const uint32_t* in, uint32_t* out, const uint32_t* parent,
    int n, int* nout)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint32_t r = in[t];
    if (r != 0 && parent[r] == r) {
        int s = atomicAdd(nout, 1);
        out[s] = r;
    }
}

__global__ void k_count_u8(const uint8_t* a, int64_t n, int* nout)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n && a[i]) atomicAdd(nout, 1);
}

// --- E6t: official StarMerge (walk blues only, InsertOrUpdate) ---
struct EHash {
    unsigned long long key;
    int idx;
};

static inline __device__ unsigned long long ekey(uint32_t a, uint32_t b)
{
    if (a > b) {
        uint32_t t = a;
        a = b;
        b = t;
    }
    return ((unsigned long long)a << 32) | (unsigned long long)b;
}

static inline __device__ int ehash_slot(unsigned long long key, int mask)
{
    unsigned long long h = key ^ (key >> 33);
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return (int)(h & (unsigned long long)mask);
}

__global__ void k_ehash_clear(EHash* tab, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        tab[i].key = 0;
        tab[i].idx = -1;
    }
}

__global__ void k_ehash_build(
    const uint32_t* u, const uint32_t* v, const int64_t* ct, int64_t n,
    EHash* tab, int ntab, int* fail)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n || ct[i] < 1) return;
    unsigned long long key = ekey(u[i], v[i]);
    if (key <= 1ull) return;
    int mask = ntab - 1;
    int slot = ehash_slot(key, mask);
    for (int s = 0; s < 1024; ++s) {
        int idx = (slot + s) & mask;
        unsigned long long old = atomicCAS(&tab[idx].key, 0ull, key);
        if (old == 0 || old == key) {
            tab[idx].idx = (int)i;
            return;
        }
    }
    atomicExch(fail, 1);
}

__device__ int ehash_find(const EHash* tab, int ntab, unsigned long long key)
{
    int mask = ntab - 1;
    unsigned long long h = key ^ (key >> 33);
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    int slot = (int)(h & (unsigned long long)mask);
    for (int s = 0; s < 1024; ++s) {
        int idx = (slot + s) & mask;
        unsigned long long k = tab[idx].key;
        if (k == 0) return -1;
        if (k == key) return tab[idx].idx;
    }
    return -1;
}

__device__ void ehash_tombstone(EHash* tab, int ntab, unsigned long long key, int eid)
{
    int mask = ntab - 1;
    unsigned long long h = key ^ (key >> 33);
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    int slot = (int)(h & (unsigned long long)mask);
    for (int s = 0; s < 1024; ++s) {
        int idx = (slot + s) & mask;
        unsigned long long k = tab[idx].key;
        if (k == 0) return;
        if (k == key && tab[idx].idx == eid) {
            tab[idx].key = 1ull;
            tab[idx].idx = -1;
            return;
        }
    }
}

__global__ void k_deg(
    const uint32_t* u, const uint32_t* v, const int64_t* ct, int64_t n, int* deg)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n || ct[i] < 1) return;
    atomicAdd(&deg[u[i]], 1);
    atomicAdd(&deg[v[i]], 1);
}

__global__ void k_scatter_adj(
    const uint32_t* u, const uint32_t* v, const int64_t* ct, int64_t n,
    const int* off, int* cur, int* adj)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n || ct[i] < 1) return;
    int pu = atomicAdd(&cur[u[i]], 1);
    adj[off[u[i]] + pu] = (int)i;
    int pv = atomicAdd(&cur[v[i]], 1);
    adj[off[v[i]] + pv] = (int)i;
}

__global__ void k_gc_mark(
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    uint32_t* parent, int64_t n, double TL, uint8_t* keep)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    keep[i] = 0;
    if (ct[i] < 1) return;
    if (sm[i] / (double)ct[i] < TL) return;
    uint32_t a = dfind_nocomp(parent, u[i]), b = dfind_nocomp(parent, v[i]);
    if (a == b || a == 0 || b == 0) return;
    keep[i] = 1;
}

__global__ void k_iota_if(const uint8_t* keep, int64_t n, int* out, int* nout)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n || !keep[i]) return;
    int slot = atomicAdd(nout, 1);
    out[slot] = (int)i;
}

__global__ void k_propose_eid(
    const int* eids, int ne,
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    uint32_t* parent, const uint32_t* sz, const uint8_t* frozen,
    double TL, unsigned long long* prop, uint64_t seed, uint64_t cseed)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ne) return;
    int i = eids[j];
    if (i < 0 || ct[i] < 1) return;
    double mean = sm[i] / (double)ct[i];
    if (mean < TL) return;
    uint32_t a = dfind_nocomp(parent, u[i]), b = dfind_nocomp(parent, v[i]);
    if (a == b || a == 0 || b == 0) return;
    uint8_t ca = color_of(a, cseed);
    uint8_t cb = color_of(b, cseed);
    if (ca == cb) return;
    uint32_t r = (ca == 1) ? a : b;
    uint32_t bl = (ca == 1) ? b : a;
    if (frozen[r]) return;
    if (d_size_asym && sz[r] < sz[bl]) return;
    unsigned long long pack =
        ((unsigned long long)prop_pri_bits(seed, a, b, r) << 32)
        | (unsigned long long)r;
    atomicMax(&prop[bl], pack);
}

// E6w: first writer of a blue records it so pack/memset skip nnode.
__global__ void k_propose_eid_list(
    const int* eids, const int* dne,
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    uint32_t* parent, const uint32_t* sz, const uint8_t* frozen,
    double TL, unsigned long long* prop, uint32_t* blist, int* nlist,
    const uint64_t* dseed, const uint64_t* dcseed, const int* dinner)
{
    int ne = dne ? *dne : 0;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ne) return;
    int i = eids[j];
    uint64_t seed = (dseed ? *dseed : 0) + (dinner ? (uint64_t)(*dinner) * 17ull : 0);
    uint64_t cseed = dcseed ? *dcseed : 0;
    if (i < 0 || ct[i] < 1) return;
    double mean = sm[i] / (double)ct[i];
    if (mean < TL) return;
    uint32_t a = dfind_nocomp(parent, u[i]), b = dfind_nocomp(parent, v[i]);
    if (a == b || a == 0 || b == 0) return;
    uint8_t ca = color_of(a, cseed);
    uint8_t cb = color_of(b, cseed);
    if (ca == cb) return;
    uint32_t r = (ca == 1) ? a : b;
    uint32_t bl = (ca == 1) ? b : a;
    if (frozen[r]) return;
    if (d_size_asym && sz[r] < sz[bl]) return;
    unsigned long long pack =
        ((unsigned long long)prop_pri_bits(seed, a, b, r) << 32)
        | (unsigned long long)r;
    unsigned long long old = atomicMax(&prop[bl], pack);
    if (old == 0 && pack != 0) {
        int slot = atomicAdd(nlist, 1);
        blist[slot] = bl;
    }
}

__global__ void k_pack_listed(
    const uint32_t* blist, const int* nlist_d, const unsigned long long* prop,
    const uint32_t* sz, uint32_t* reds, uint32_t* blues, uint32_t* addsz,
    float* pris, int* nprop)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nlist = nlist_d ? *nlist_d : 0;
    if (i >= nlist) return;
    uint32_t b = blist[i];
    if (b == 0) return;
    unsigned long long p = prop[b];
    if (p == 0) return;
    uint32_t r = (uint32_t)(p & 0xffffffffull);
    if (r == 0) return;
    int slot = atomicAdd(nprop, 1);
    reds[slot] = r;
    blues[slot] = b;
    addsz[slot] = sz[b];
    pris[slot] = __uint_as_float((unsigned)(p >> 32));
}

__global__ void k_zero_listed_prop(const uint32_t* blist, int n, unsigned long long* prop)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && blist[i] != 0) prop[blist[i]] = 0;
}

__global__ void k_zero_listed_prop_d(const uint32_t* blist, const int* nptr, unsigned long long* prop)
{
    int n = nptr ? *nptr : 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && blist[i] != 0) prop[blist[i]] = 0;
}

__global__ void k_accept_reds_d(
    const uint32_t* reds, const uint32_t* blues, const uint32_t* addsz,
    const int* nprop_d, uint32_t* parent, uint32_t* sz, const uint8_t* frozen,
    double eps, int* nmerge)
{
    int nprop = nprop_d ? *nprop_d : 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nprop) return;
    if (i > 0 && reds[i] == reds[i - 1]) return;
    uint32_t r0 = reds[i];
    int j = i + 1;
    while (j < nprop && reds[j] == r0) ++j;
    uint32_t r = dfind_nocomp(parent, r0);
    if (frozen[r0] || r == 0) return;
    uint64_t cap = (uint64_t)llround((double)sz[r] * eps);
    if (eps > 0 && cap == 0) cap = 1;
    uint64_t acc = 0;
    int local = 0;
    for (int k = i; k < j; ++k) {
        uint32_t b = dfind_nocomp(parent, blues[k]);
        if (b == r || b == 0) continue;
        uint32_t add = addsz[k];
        acc += add;
        parent[b] = r;
        sz[r] += add;
        ++local;
        if (acc > cap) break;
    }
    if (local) atomicAdd(nmerge, local);
}

__global__ void k_freeze_reds(
    const uint32_t* reds, const int* nprop_d,
    uint32_t* sz, const uint32_t* sz0, uint8_t* frozen, double eps)
{
    int nprop = nprop_d ? *nprop_d : 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nprop) return;
    if (i > 0 && reds[i] == reds[i - 1]) return;
    uint32_t r = reds[i];
    if (r == 0) return;
    if ((double)sz[r] > (1.0 + eps) * (double)sz0[r]) frozen[r] = 1;
}

__global__ void k_fill_sort_keys(
    const uint32_t* reds, const float* pris, const int* nprop,
    unsigned long long* keys, int* idx)
{
    int n = nprop ? *nprop : 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned pri = __float_as_uint(pris[i]);
    keys[i] = ((unsigned long long)reds[i] << 32) | (unsigned long long)(~pri);
    idx[i] = i;
}

__global__ void k_radix_count(
    const unsigned long long* keys, const int* nptr, int shift,
    int* hist, int nblk)
{
    int n = nptr ? *nptr : 0;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ int sh[256];
    if (tid < 256) sh[tid] = 0;
    __syncthreads();
    for (int i = bid * (int)blockDim.x + tid; i < n; i += nblk * (int)blockDim.x) {
        int d = (int)((keys[i] >> shift) & 255ull);
        atomicAdd(&sh[d], 1);
    }
    __syncthreads();
    if (tid < 256) hist[bid * 256 + tid] = sh[tid];
}

__global__ void k_radix_scan_blocks(int* hist, int nblk)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int excl[256];
    int run = 0;
    for (int d = 0; d < 256; ++d) {
        int tot = 0;
        for (int b = 0; b < nblk; ++b) tot += hist[b * 256 + d];
        excl[d] = run;
        run += tot;
    }
    for (int d = 0; d < 256; ++d) {
        int prefix = 0;
        for (int b = 0; b < nblk; ++b) {
            int c = hist[b * 256 + d];
            hist[b * 256 + d] = excl[d] + prefix;
            prefix += c;
        }
    }
}

__global__ void k_radix_scatter(
    const unsigned long long* kin, const int* iin,
    unsigned long long* kout, int* iout,
    const int* nptr, int shift, const int* offsets, int nblk)
{
    int n = nptr ? *nptr : 0;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ int local[256];
    if (tid < 256) local[tid] = 0;
    __syncthreads();
    for (int i = bid * (int)blockDim.x + tid; i < n; i += nblk * (int)blockDim.x) {
        int d = (int)((kin[i] >> shift) & 255ull);
        int lr = atomicAdd(&local[d], 1);
        int dest = offsets[bid * 256 + d] + lr;
        kout[dest] = kin[i];
        iout[dest] = iin[i];
    }
}

__global__ void k_apply_perm(
    const int* idx, const int* nptr,
    const uint32_t* r0, const uint32_t* b0, const uint32_t* a0, const float* p0,
    uint32_t* r1, uint32_t* b1, uint32_t* a1, float* p1)
{
    int n = nptr ? *nptr : 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = idx[i];
    r1[i] = r0[s];
    b1[i] = b0[s];
    a1[i] = a0[s];
    p1[i] = p0[s];
}

static void sort_proposals_dev(
    uint32_t* reds, uint32_t* blues, uint32_t* addsz, float* pris, int* dnprop,
    unsigned long long* k0, unsigned long long* k1, int* i0, int* i1, int* hist,
    uint32_t* r1, uint32_t* b1, uint32_t* a1, float* p1,
    int nnode, int threads, cudaStream_t st)
{
    const int nblk = 256;
    int bn = (nnode + threads - 1) / threads;
    if (bn < 1) bn = 1;
    k_fill_sort_keys<<<bn, threads, 0, st>>>(reds, pris, dnprop, k0, i0);
    unsigned long long* kin = k0;
    unsigned long long* kout = k1;
    int* iin = i0;
    int* iout = i1;
    for (int shift = 0; shift < 64; shift += 8) {
        k_radix_count<<<nblk, threads, 0, st>>>(kin, dnprop, shift, hist, nblk);
        k_radix_scan_blocks<<<1, 1, 0, st>>>(hist, nblk);
        k_radix_scatter<<<nblk, threads, 0, st>>>(kin, iin, kout, iout, dnprop, shift, hist, nblk);
        unsigned long long* kt = kin;
        kin = kout;
        kout = kt;
        int* it = iin;
        iin = iout;
        iout = it;
    }
    k_apply_perm<<<bn, threads, 0, st>>>(iin, dnprop, reds, blues, addsz, pris, r1, b1, a1, p1);
}

__global__ void k_copy_int_n(const int* src, int* dst, const int* nptr)
{
    int n = nptr ? *nptr : 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void k_copy_int1(const int* src, int* dst)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *dst = *src;
}

__global__ void k_e6u_tick(
    cudaGraphConditionalHandle handle, const int* nprop, const int* nmerge,
    int* inner, int max_inner, int* ninner_tot, int* nmerge_tot, const int* fail)
{
    int hm = nmerge ? *nmerge : 0;
    int np = nprop ? *nprop : 0;
    int hf = fail ? *fail : 0;
    if (ninner_tot) atomicAdd(ninner_tot, 1);
    if (nmerge_tot && hm) atomicAdd(nmerge_tot, hm);
    int i = *inner + 1;
    *inner = i;
    unsigned go = (np > 0 && hm > 0 && hf == 0 && i < max_inner) ? 1u : 0u;
    cudaGraphSetConditional(handle, go);
}

__global__ void k_ovf_reset(int* ovf_head, int nnode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnode) ovf_head[i] = -1;
}

__global__ void k_starmarge_blue(
    const uint32_t* blues, const int* nprop_d, uint32_t* parent,
    uint32_t* u, uint32_t* v, double* sm, int64_t* ct,
    const int* adj_off, const int* adj_len, const int* adj,
    int* ovf_head, int* ovf_eid, int* ovf_nxt, int* ovf_used, int ovf_cap,
    EHash* tab, int ntab,
    int* touched, int* ntouch, int* fail, int* nkill)
{
    int nprop = nprop_d ? *nprop_d : 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nprop) return;
    uint32_t b = blues[i];
    if (b == 0 || parent[b] == b) return;
    auto walk = [&](int eid) {
        if (eid < 0) return;
        unsigned long long oldc = atomicExch((unsigned long long*)(ct + eid), 0ull);
        if ((int64_t)oldc < 1) return;
        double olds = sm[eid];
        uint32_t ou0 = u[eid], ov0 = v[eid];
        ehash_tombstone(tab, ntab, ekey(ou0, ov0), eid);
        uint32_t ou = dfind_nocomp(parent, ou0);
        uint32_t ov = dfind_nocomp(parent, ov0);
        if (ou == ov || ou == 0 || ov == 0) {
            atomicAdd(nkill, 1);
            return;
        }
        if (ou > ov) {
            uint32_t t = ou;
            ou = ov;
            ov = t;
        }
        unsigned long long key = ekey(ou, ov);
        int mask = ntab - 1;
        unsigned long long hh = key ^ (key >> 33);
        hh *= 0xff51afd7ed558ccdULL;
        hh ^= hh >> 33;
        int slot0 = (int)(hh & (unsigned long long)mask);
        int dest = -1;
        int is_new = 0;
        for (int s = 0; s < 1024; ++s) {
            int idx = (slot0 + s) & mask;
            unsigned long long k = tab[idx].key;
            if (k == key) {
                dest = tab[idx].idx;
                break;
            }
            if (k == 0 || k == 1ull) {
                unsigned long long old = atomicCAS(&tab[idx].key, k, key);
                if (old == k) {
                    tab[idx].idx = eid;
                    dest = eid;
                    is_new = 1;
                    break;
                }
                if (tab[idx].key == key) {
                    dest = tab[idx].idx;
                    break;
                }
            }
        }
        if (dest < 0) {
            atomicExch(fail, 1);
            sm[eid] = olds;
            ct[eid] = (int64_t)oldc;
            u[eid] = ou;
            v[eid] = ov;
            return;
        }
        if (dest != eid) {
            atomicAdd(&sm[dest], olds);
            atomicAdd((unsigned long long*)&ct[dest], oldc);
            atomicAdd(nkill, 1);
            int tslot = atomicAdd(ntouch, 1);
            touched[tslot] = dest;
            return;
        }
        u[eid] = ou;
        v[eid] = ov;
        sm[eid] = olds;
        ct[eid] = (int64_t)oldc;
        if (is_new) {
            if (ou != ou0 && ou != ov0) {
                int s = atomicAdd(ovf_used, 1);
                if (s >= ovf_cap) {
                    atomicExch(fail, 1);
                } else {
                    ovf_eid[s] = eid;
                    int prev = atomicExch(&ovf_head[ou], s);
                    ovf_nxt[s] = prev;
                }
            }
            if (ov != ou0 && ov != ov0) {
                int s = atomicAdd(ovf_used, 1);
                if (s >= ovf_cap) {
                    atomicExch(fail, 1);
                } else {
                    ovf_eid[s] = eid;
                    int prev = atomicExch(&ovf_head[ov], s);
                    ovf_nxt[s] = prev;
                }
            }
        }
        int tslot = atomicAdd(ntouch, 1);
        touched[tslot] = eid;
    };

    int off = adj_off[b];
    int n = adj_len[b];
    for (int k = 0; k < n; ++k) walk(adj[off + k]);
    for (int t = ovf_head[b]; t >= 0; t = ovf_nxt[t]) walk(ovf_eid[t]);
}

__global__ void k_filter_gc(
    const int* gin, const int* n_d, const uint32_t* u, const uint32_t* v,
    const double* sm, const int64_t* ct, uint32_t* parent, double TL,
    int* gout, int* nout)
{
    int n = n_d ? *n_d : 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int eid = gin[i];
    if (eid < 0 || ct[eid] < 1) return;
    if (sm[eid] / (double)ct[eid] < TL) return;
    uint32_t a = dfind_nocomp(parent, u[eid]), b = dfind_nocomp(parent, v[eid]);
    if (a == b || a == 0 || b == 0) return;
    int slot = atomicAdd(nout, 1);
    gout[slot] = eid;
}

__global__ void k_add_touched_gc(
    const int* touched, const int* n_d, const uint32_t* u, const uint32_t* v,
    const double* sm, const int64_t* ct, uint32_t* parent, double TL,
    int* gout, int* nout)
{
    int n = n_d ? *n_d : 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int eid = touched[i];
    if (eid < 0 || ct[eid] < 1) return;
    if (sm[eid] / (double)ct[eid] < TL) return;
    uint32_t a = dfind_nocomp(parent, u[eid]), b = dfind_nocomp(parent, v[eid]);
    if (a == b || a == 0 || b == 0) return;
    int slot = atomicAdd(nout, 1);
    gout[slot] = eid;
}

// Hash-combine live edges (S32 MultiMerge substitute: no full sort).
// 16 bytes, not 24. This table is the single largest allocation the
// agglomeration makes - next_pow2(2*n_edges) slots, 83 B/edge measured, 40% of
// the tracked peak - so a third off it is 2.4 GiB at 2.16 Gvox.
//
// Narrowing is exact, not an approximation. sm arrives already rescaled to a
// whole number of affinity bytes by the llround at the top of the layer, and ct
// is a face count, so both accumulators are integers and both atomicAdds are
// exact integer sums in either width. The only thing uint32 gives up is range:
// sm <= 255*ct overflows past 16.8 M faces on one contact. That is not provable
// at 2.16 Gvox, so k_hash_insert checks each add against the width instead of
// assuming, and the caller reports it - the same discipline rag.cu already uses
// for its own isum.
// Build with -DHSLOT_WIDE for the original 24-byte slot. That is what the
// overflow message tells you to do, and it is how the narrowing was gated:
// both widths built from this one source and their labels compared.
#ifdef HSLOT_WIDE
using hsm_t = double;
using hct_t = unsigned long long;
#else
using hsm_t = uint32_t;
using hct_t = uint32_t;
#endif

struct HSlot {
    unsigned long long key;
    hsm_t sm;
    hct_t ct;
};

// Bits in the hash overflow flag, so a probe-limit failure and an arithmetic
// one stay distinguishable instead of one overwriting the other.
enum { HOVF_PROBE = 1, HOVF_WIDTH = 2 };

// Did this contribution exceed the slot's width? Checks the incoming value as
// well as the sum, since a value already past uint32 would be lost on the cast
// rather than on the add. Always false for the wide slot, which cannot wrap at
// any volume that fits in memory.
static inline __device__ bool hslot_would_wrap(
    hsm_t olds, hsm_t adds, hct_t oldc, hct_t addc, double smd, int64_t cti)
{
#ifdef HSLOT_WIDE
    (void)olds; (void)adds; (void)oldc; (void)addc; (void)smd; (void)cti;
    return false;
#else
    return smd < 0.0 || smd > 4294967295.0 || cti > 4294967295ll
        || olds > 0xffffffffu - adds || oldc > 0xffffffffu - addc;
#endif
}

static inline __device__ unsigned long long edge_key(uint32_t u, uint32_t v)
{
    return ((unsigned long long)u << 32) | (unsigned long long)v;
}

static inline __device__ unsigned long long hmix64(unsigned long long h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

__global__ void k_hash_clear(HSlot* tab, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        tab[i].key = 0;
        tab[i].sm = 0;
        tab[i].ct = 0;
    }
}

__global__ void k_hash_insert(
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    const uint8_t* keep, int64_t n, HSlot* tab, int ntab, int* ovf,
    uint32_t* slots, int* nslots)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n || !keep[i] || ct[i] < 1) return;
    uint32_t a = u[i], b = v[i];
    if (a == 0 || b == 0 || a == b) return;
    if (a > b) {
        uint32_t t = a;
        a = b;
        b = t;
    }
    unsigned long long key = edge_key(a, b);
    unsigned long long h = hmix64(key);
    int mask = ntab - 1;
    int slot = (int)(h & (unsigned long long)mask);
    const double smd = sm[i];
    const int64_t cti = ct[i];
    const hsm_t adds = (hsm_t)smd;
    const hct_t addc = (hct_t)cti;
    for (int s = 0; s < 128; ++s) {
        int idx = (slot + s) & mask;
        unsigned long long old = atomicCAS(&tab[idx].key, 0ull, key);
        if (old == 0 || old == key) {
            if (old == 0 && slots && nslots) {
                int si = atomicAdd(nslots, 1);
                if (si >= 0 && si < ntab) slots[si] = (uint32_t)idx;
            }
            const hsm_t olds = atomicAdd(&tab[idx].sm, adds);
            const hct_t oldc = atomicAdd(&tab[idx].ct, addc);
            if (ovf && hslot_would_wrap(olds, adds, oldc, addc, smd, cti))
                atomicOr(ovf, HOVF_WIDTH);
            return;
        }
    }
    if (ovf) atomicOr(ovf, HOVF_PROBE);
}

static inline __device__ void hslot_insert(
    HSlot* tab, int ntab, unsigned long long key,
    hsm_t adds, hct_t addc, double smd, int64_t cti, int* ovf,
    uint32_t* slots = nullptr, int* nslots = nullptr,
    uint32_t* home = nullptr, uint32_t eid = 0xffffffffu)
{
    unsigned long long h = hmix64(key);
    int mask = ntab - 1;
    int slot = (int)(h & (unsigned long long)mask);
    for (int s = 0; s < 128; ++s) {
        int idx = (slot + s) & mask;
        unsigned long long old = atomicCAS(&tab[idx].key, 0ull, key);
        if (old == 0 || old == key) {
            if (old == 0 && slots && nslots) {
                int si = atomicAdd(nslots, 1);
                if (si >= 0 && si < ntab) slots[si] = (uint32_t)idx;
            }
            if (old == 0 && home) home[idx] = eid;
            const hsm_t olds = atomicAdd(&tab[idx].sm, adds);
            const hct_t oldc = atomicAdd(&tab[idx].ct, addc);
            if (ovf && hslot_would_wrap(olds, adds, oldc, addc, smd, cti))
                atomicOr(ovf, HOVF_WIDTH);
            return;
        }
    }
    if (ovf) atomicOr(ovf, HOVF_PROBE);
}

// N21 A1: same CAS insert as k_hash_insert, grid over holes[0:ndirty]
// (keep-survivors from k_rewrite_dirty_fuse). Not HASH_INSERT_ONLY.
// Not k_hash_insert_dirty (n8: 1.31x slower).
__global__ void k_hash_insert_listed(
    const uint32_t* holes, int ndirty,
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    HSlot* tab, int ntab, int* ovf, uint32_t* slots, int* nslots,
    uint32_t* home)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ndirty) return;
    const int64_t i = (int64_t)holes[j];
    if (ct[i] < 1) return;
    uint32_t a = u[i], b = v[i];
    if (a == 0 || b == 0 || a == b) return;
    if (a > b) {
        uint32_t t = a;
        a = b;
        b = t;
    }
    const double smd = sm[i];
    const int64_t cti = ct[i];
    hslot_insert(tab, ntab, edge_key(a, b), (hsm_t)smd, (hct_t)cti, smd, cti,
                 ovf, slots, nslots, home, (uint32_t)i);
}

// N23 A1: write combined sums back onto the CAS-win edge index so CSR
// incidence lists stay attached to the same slots. Prefix-emit into
// holes[0:m] permutes (u,v) onto foreign slots and breaks splice.
__global__ void k_hash_commit_home(
    const uint32_t* holes, int ndirty,
    uint32_t* u, uint32_t* v, double* sm, int64_t* ct,
    HSlot* tab, int ntab, const uint32_t* home, int* nout, uint8_t* amask)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ndirty) return;
    const uint32_t eid = holes[j];
    if (ct[eid] < 1) return;
    uint32_t a = u[eid], b = v[eid];
    if (a > b) {
        uint32_t t = a;
        a = b;
        b = t;
    }
    unsigned long long key = edge_key(a, b);
    unsigned long long h = hmix64(key);
    int mask = ntab - 1;
    int slot = (int)(h & (unsigned long long)mask);
    for (int s = 0; s < 128; ++s) {
        int idx = (slot + s) & mask;
        if (tab[idx].key != key) continue;
        if (home[idx] == eid) {
            sm[eid] = (double)tab[idx].sm;
            ct[eid] = (int64_t)tab[idx].ct;
            tab[idx].key = 0;
            tab[idx].sm = 0;
            tab[idx].ct = 0;
            atomicAdd(nout, 1);
        } else {
            ct[eid] = 0;
            if (amask) amask[eid] = 0;
        }
        return;
    }
    ct[eid] = 0;
    if (amask) amask[eid] = 0;
}

// Dirty-path insert: warp-aggregate equal keys (C1), then a 512-slot
// shared table, then one global CAS per surviving group. Integer sums,
// so the combined (sm, ct) equals a per-thread CAS of the same edges.
#ifndef HASH_SH
#define HASH_SH 512
#endif

__global__ void k_hash_insert_dirty(
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    const uint8_t* keep, int64_t n, HSlot* tab, int ntab, int* ovf)
{
    __shared__ unsigned long long sh_key[HASH_SH];
    __shared__ hsm_t sh_sm[HASH_SH];
    __shared__ hct_t sh_ct[HASH_SH];
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    for (int s = tid; s < HASH_SH; s += nthr) {
        sh_key[s] = 0;
        sh_sm[s] = 0;
        sh_ct[s] = 0;
    }
    __syncthreads();

    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const bool in = (i < n);
    uint32_t a = 0, b = 0;
    bool valid = in && keep[i] && ct[i] >= 1;
    if (valid) {
        a = u[i];
        b = v[i];
        if (a > b) {
            uint32_t t = a;
            a = b;
            b = t;
        }
        valid = a != 0 && b != 0 && a != b;
    }
    const unsigned long long key = valid ? edge_key(a, b) : 0ull;
    const unsigned peers = __match_any_sync(0xffffffffu, key);
    hsm_t adds = 0;
    hct_t addc = 0;
    double smd = 0.0;
    int64_t cti = 0;
    bool lead = false;
    if (valid) {
        smd = sm[i];
        cti = ct[i];
        adds = (hsm_t)smd;
        addc = (hct_t)cti;
        adds = (hsm_t)__reduce_add_sync(peers, (unsigned)adds);
        addc = (hct_t)__reduce_add_sync(peers, (unsigned)addc);
        const unsigned lane = threadIdx.x & 31u;
        lead = __popc(peers & ((1u << lane) - 1u)) == 0;
        smd = (double)adds;
        cti = (int64_t)addc;
    }
    bool in_sh = false;
    if (lead) {
        unsigned long long h = hmix64(key);
        int mask = HASH_SH - 1;
        int slot = (int)(h & (unsigned long long)mask);
        for (int s = 0; s < 16; ++s) {
            int idx = (slot + s) & mask;
            unsigned long long old = atomicCAS(&sh_key[idx], 0ull, key);
            if (old == 0 || old == key) {
                atomicAdd(&sh_sm[idx], adds);
                atomicAdd(&sh_ct[idx], addc);
                in_sh = true;
                break;
            }
        }
    }
    __syncthreads();
    for (int s = tid; s < HASH_SH; s += nthr) {
        if (sh_key[s] != 0 && sh_ct[s] > 0)
            hslot_insert(tab, ntab, sh_key[s], sh_sm[s], sh_ct[s],
                         (double)sh_sm[s], (int64_t)sh_ct[s], ovf);
    }
    if (lead && !in_sh)
        hslot_insert(tab, ntab, key, adds, addc, smd, cti, ovf);
}

__global__ void k_hash_count(const HSlot* tab, int ntab, int* nout)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntab) return;
    if (tab[i].key != 0 && tab[i].ct > 0) atomicAdd(nout, 1);
}

// Emits the combined edges and leaves the table empty behind it.
//
// The caller used to run k_hash_clear over the whole table before every insert
// pass: a 24-byte store on roughly 2.7 slots per live edge, every inner
// iteration, which is about 65 B/edge of write traffic on a phase that is
// bandwidth-bound. This pass already has to read every slot, so emptying the
// occupied ones here replaces that with a store on the occupied slots alone.
//
// N22 records those occupied indices at insert CAS-win (WATERZ_SLOT_EMIT)
// so emit need not read empty slots. The invariant is the same: key == 0
// implies sm and ct are untouched zeros. Zeroing every occupied slot leaves
// the whole table clean, including the tail above a later, smaller ntab_use.
//
// The invariant it maintains: k_hash_insert only writes a slot after a CAS that
// set its key, so key == 0 implies sm and ct are untouched zeros. Zeroing every
// slot whose key is set therefore leaves the whole table clean, including the
// tail above a later, smaller ntab_use.
__global__ void k_hash_emit(
    HSlot* tab, int ntab, uint32_t* u, uint32_t* v, double* sm, int64_t* ct,
    int* nout)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntab) return;
    const unsigned long long key = tab[i].key;
    if (key == 0) return;
    const hsm_t s = tab[i].sm;
    const hct_t c = tab[i].ct;
    tab[i].key = 0;
    tab[i].sm = 0;
    tab[i].ct = 0;
    if (c == 0) return;
    int slot = atomicAdd(nout, 1);
    u[slot] = (uint32_t)(key >> 32);
    v[slot] = (uint32_t)(key & 0xffffffffull);
    sm[slot] = (double)s;
    ct[slot] = (int64_t)c;
}

// N17: emit combined dirty edges into the keep-holes from rewrite.
// Tail holes (m..nhole) stay vacated. Skips k_vacate_kept O(nscan) + k_fill_holes.
__global__ void k_hash_emit_holes(
    HSlot* tab, int ntab, const uint32_t* holes, int nhole,
    uint32_t* u, uint32_t* v, double* sm, int64_t* ct,
    int* nout, int* ovf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntab) return;
    const unsigned long long key = tab[i].key;
    if (key == 0) return;
    const hsm_t s = tab[i].sm;
    const hct_t c = tab[i].ct;
    tab[i].key = 0;
    tab[i].sm = 0;
    tab[i].ct = 0;
    if (c == 0) return;
    int slot = atomicAdd(nout, 1);
    if (slot >= nhole) {
        if (ovf) atomicExch(ovf, 1);
        return;
    }
    const uint32_t d = holes[slot];
    u[d] = (uint32_t)(key >> 32);
    v[d] = (uint32_t)(key & 0xffffffffull);
    sm[d] = (double)s;
    ct[d] = (int64_t)c;
}

// N22 A3: same emit as k_hash_emit_holes, grid over occupied table indices
// recorded at CAS-win in insert. Empty slots stay 0 from alloc / previous
// emit. Do not resurrect k_hash_clear. Default off (WATERZ_SLOT_EMIT).
__global__ void k_hash_emit_slots(
    HSlot* tab, int ntab, const uint32_t* slots, int nslot,
    const uint32_t* holes, int nhole,
    uint32_t* u, uint32_t* v, double* sm, int64_t* ct,
    int* nout, int* ovf)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nslot) return;
    const uint32_t i = slots[j];
    if ((int)i >= ntab) return;
    const unsigned long long key = tab[i].key;
    if (key == 0) return;
    const hsm_t s = tab[i].sm;
    const hct_t c = tab[i].ct;
    tab[i].key = 0;
    tab[i].sm = 0;
    tab[i].ct = 0;
    if (c == 0) return;
    int slot = atomicAdd(nout, 1);
    if (slot >= nhole) {
        if (ovf) atomicExch(ovf, 1);
        return;
    }
    const uint32_t d = holes[slot];
    u[d] = (uint32_t)(key >> 32);
    v[d] = (uint32_t)(key & 0xffffffffull);
    sm[d] = (double)s;
    ct[d] = (int64_t)c;
}

__global__ void k_zero_hole_tail(
    const uint32_t* holes, int m, int nhole, int64_t* ct, uint8_t* amask)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int src = m + j;
    if (src >= nhole) return;
    const uint32_t i = holes[src];
    ct[i] = 0;
    if (amask) amask[i] = 0;
}

__global__ void k_freeze(uint32_t* parent, uint32_t* sz, const uint32_t* sz0,
    const uint8_t* color, uint8_t* frozen, int nnode, double eps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= 0 || i >= nnode) return;
    if (color[i] != 1) return;
    uint32_t r = dfind_nocomp(parent, (uint32_t)i);
    double s0 = sz0[i] ? (double)sz0[i] : 1.0;
    if ((double)sz[r] > (1.0 + eps) * s0) frozen[i] = 1;
}

static int compact_and_combine(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct, uint8_t* dkeep,
    uint32_t* dparent, int nnode, int64_t n, int64_t* n_out, double TL,
    int be, int threads,
    uint32_t* tu, uint32_t* tv, double* tsm, int64_t* tct)
{
    int bn = (nnode + threads - 1) / threads;
    k_compress<<<bn, threads>>>(dparent, nnode);
    k_rewrite<<<be, threads>>>(du, dv, dsm, dct, dparent, n, dkeep, TL);
    thrust::device_ptr<uint32_t> pu(du), pv(dv);
    thrust::device_ptr<double> psm(dsm);
    thrust::device_ptr<int64_t> pct(dct);
    thrust::device_ptr<uint8_t> pk(dkeep);
    auto in = thrust::make_zip_iterator(thrust::make_tuple(pu, pv, psm, pct));
    auto out = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<uint32_t>(tu),
        thrust::device_ptr<uint32_t>(tv),
        thrust::device_ptr<double>(tsm),
        thrust::device_ptr<int64_t>(tct)));
    auto end = thrust::copy_if(in, in + n, pk, out, KeepOn());
    int64_t m = end - out;
    if (m <= 0) {
        *n_out = 0;
        return 1;
    }
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<uint32_t>(tu),
        thrust::device_ptr<uint32_t>(tv),
        thrust::device_ptr<double>(tsm),
        thrust::device_ptr<int64_t>(tct)));
    thrust::sort(zip, zip + m, EdgeLess());
    auto keys_in = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<uint32_t>(tu), thrust::device_ptr<uint32_t>(tv)));
    auto vals_in = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<double>(tsm), thrust::device_ptr<int64_t>(tct)));
    auto keys_out = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<uint32_t>(du), thrust::device_ptr<uint32_t>(dv)));
    auto vals_out = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<double>(dsm), thrust::device_ptr<int64_t>(dct)));
    auto red = thrust::reduce_by_key(
        keys_in, keys_in + m, vals_in, keys_out, vals_out, EdgeKeyEq(), SumPair());
    *n_out = red.first - keys_out;
    return 1;
}

static int next_pow2(int x);

static int hash_dedup_on()
{
    const char* s = std::getenv("WATERZ_HASH_DEDUP");
    return s ? std::atoi(s) : 1;
}

// CUB SortPairs + ReduceByKey on a compacted dirty set. Defined after
// k_key_from_sel / k_gather_smct / k_unpack_uvkey.
static int dirty_rbk(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct, uint8_t* dkeep,
    int64_t n, int ndirty_e, int* n_live, int be, int threads,
    uint32_t* tu, uint32_t* tv, double* tsm, int64_t* tct,
    uint32_t* dholes, int* dnhole, int* dnout);

// Deduplicate the live edge list with a hash table instead of a sort.
//
// compact_radix does an 8-pass 64-bit radix sort over the whole live list on
// every inner iteration for the sole purpose of putting duplicate keys next to
// each other so ReduceByKey can merge parallel edges. That is O(m log m)
// traffic and ~28 kernel launches to do an O(m) job. Inserting into a hash
// table keyed on the canonical (u,v) does the same merge in one pass.
//
// k_rewrite has already dropped self-loops, background-incident edges and
// empty counts via `keep`, and canonicalised u < v, so the guards inside
// k_hash_insert are redundant and the two paths combine the same edge
// multiset. The sums are order-independent: contact sums are integral
// affinity-byte counts held exactly in a double after k_scale_sm_bytes, and
// counts are integers, so the atomicAdds commute.
//
// What differs is the *order* of the emitted edge list, because k_hash_emit
// claims output slots with an atomic. That is only safe because propose picks
// per node by atomicMax on a content-derived priority, so it does not depend
// on edge position. That argument is tested by A2, not trusted.
// The edge pointers are taken by reference so the combined result can be
// adopted by swapping buffers rather than copied back over the input.
static int hash_combine_live(
    uint32_t*& du, uint32_t*& dv, double*& dsm, int64_t*& dct, uint8_t* dkeep,
    uint32_t* dparent, int nnode, int64_t n, int64_t* n_out,
    int be, int threads,
    uint32_t*& tu, uint32_t*& tv, double*& tsm, int64_t*& tct,
    HSlot* dtab, int ntab, int* dnout, int* dovf, bool recompress = true)
{
    int bn = (nnode + threads - 1) / threads;
    if (recompress) k_compress<<<bn, threads>>>(dparent, nnode);
    k_rewrite<<<be, threads>>>(du, dv, dsm, dct, dparent, n, dkeep, 0.0);
    // Size the table from the *current* live count, not from the original edge
    // count. Clearing is proportional to the table, and nlive falls by an
    // order of magnitude across the layers, so a table fixed at the initial
    // size would clear ~400 MB every iteration to hold a fraction of that.
    // Load factor stays at or below 0.5, which is what makes the 128-probe
    // bound in k_hash_insert safe.
    int ntab_use = next_pow2((int)(n * 2 + 1024));
    if (ntab_use > ntab) ntab_use = ntab;
    int tb = (ntab_use + threads - 1) / threads;
    // No clear pass: k_hash_emit leaves the table empty, and the caller clears
    // it once after allocation.
    k_hash_insert<<<be, threads>>>(
        du, dv, dsm, dct, dkeep, n, dtab, ntab_use, dovf, nullptr, nullptr);
    cudaMemset(dnout, 0, 4);
    k_hash_emit<<<tb, threads>>>(dtab, ntab_use, tu, tv, tsm, tct, dnout);
    int m = 0;
    cudaMemcpy(&m, dnout, 4, cudaMemcpyDeviceToHost);
    if (m <= 0) {
        *n_out = 0;
        return 1;
    }
    // The emitted arrays are already the complete compacted edge list, so
    // copying them back over the input was 48 B/edge of pure movement on every
    // inner iteration. Swapping which buffer is "live" is the same result and
    // costs nothing. Only the first m entries are ever read, so what the two
    // buffers hold past m does not matter.
    std::swap(du, tu);
    std::swap(dv, tv);
    std::swap(dsm, tsm);
    std::swap(dct, tct);
    *n_out = m;
    return 1;
}

// G2 inner compaction. n is the scan length (holes included). Combined
// dirty edges are written back into vacated slots. n_live is the count of
// ct >= 1 afterwards; the scan length does not shrink.
static bool fuse_dirty() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_FUSE_DIRTY");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N15 exp8: keep sz0 from the first outer of the layer. eps stays 0.40.
static bool sticky_sz0() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_STICKY_SZ0");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N16 T7: nlive = nlive_in - nself - ndirty + m. Requires fuse. Default off.
static bool nlive_arith() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_NLIVE_ARITH");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N16 T15: one kernel writes amask and packs alist. Default off.
static bool fuse_pack() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_FUSE_PACK");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N17: emit into rewrite holes; skip vacate+fill. Requires fuse. Default off.
static bool emit_holes() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_EMIT_HOLES");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

static bool csr_rewrite() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_CSR_REWRITE");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0 && fuse_dirty();
}

struct CsrInc {
    int* head;
    int* nxt;
    uint32_t* gen;
    uint32_t* elist;
    int* nodes;
    int* nnodes;
    int* nlist;
    int nnode;
    int64_t ncap;
    uint32_t g;
    bool on;
};

static void csr_rebuild(
    CsrInc& inc, const uint32_t* u, const uint32_t* v, const int64_t* ct,
    int nscan, int threads)
{
    NvRange nv("csr_rebuild");
    int bn = (inc.nnode + threads - 1) / threads;
    int be = (nscan + threads - 1) / threads;
    if (be < 1) be = 1;
    k_csr_reset_head<<<bn, threads>>>(inc.head, inc.nnode);
    cudaMemset(inc.gen, 0, (size_t)inc.ncap * 4);
    inc.g = 0;
    k_csr_scatter<<<be, threads>>>(nscan, u, v, ct, inc.head, inc.nxt);
}

// N22 A3: emit occupied hash slots from an insert-time index list instead of
// scanning ntab. Requires EMIT_HOLES. Default off.
static bool slot_emit() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_SLOT_EMIT");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0 && emit_holes();
}

// N21 A1: insert over holes[0:ndirty] instead of dense nscan. Default off.
static bool listed_insert() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_LISTED_INSERT");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

// N21 A2: rebuild/pack from old alist or emit holes. Default off.
// Requires fuse + EMIT_HOLES so holes are keep-survivors and vacated
// self-loops can drop amask in k_rewrite_dirty_fuse. Not FUSE_PACK.
static bool listed_rebuild() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_LISTED_REBUILD");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0 && fuse_dirty() && emit_holes() && !fuse_pack();
}

// N17: unmark last inner's blues/reds instead of memset nnode. Default off.
static bool dirty_unmark() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("WATERZ_DIRTY_UNMARK");
        cached = (s && std::atoi(s) != 0) ? 1 : 0;
    }
    return cached != 0;
}

static int hash_combine_dirty(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct, uint8_t* dkeep,
    uint32_t* dparent, const uint8_t* dirty, int64_t n, int nlive_in, int* n_live,
    int be, int threads,
    uint32_t* tu, uint32_t* tv, double* tsm, int64_t* tct,
    uint32_t* dholes, int* dnhole,
    HSlot* dtab, int ntab, int* dnout, int* dovf,
    uint32_t* dslots = nullptr, int* dnslot = nullptr,
    uint8_t* damask = nullptr, int* out_m = nullptr,
    double* scan_acc = nullptr, double* hash_acc = nullptr,
    CsrInc* cinc = nullptr)
{
    if (out_m) *out_m = 0;
    cudaEvent_t ea, eb;
    if (scan_acc || hash_acc) {
        cudaEventCreate(&ea);
        cudaEventCreate(&eb);
    }
    if (scan_acc) cudaEventRecord(ea);
    int ndirty_e = 0;
    int nself = 0;
    const bool fuse = fuse_dirty();
    const bool arith = fuse && nlive_arith() && nlive_in >= 0;
    const bool use_csr = fuse && cinc && cinc->on;
    {
        NvRange nv("hash_rewrite");
        if (fuse) {
            cudaMemset(dnhole, 0, 4);
            cudaMemset(dnout, 0, 4);
            if (use_csr) {
                cudaMemset(cinc->nnodes, 0, 4);
                cudaMemset(cinc->nlist, 0, 4);
                int bn = (cinc->nnode + threads - 1) / threads;
                k_iota_if<<<bn, threads>>>(
                    dirty, (int64_t)cinc->nnode, cinc->nodes, cinc->nnodes);
                int n_dirty_v = 0;
                cudaMemcpy(&n_dirty_v, cinc->nnodes, 4, cudaMemcpyDeviceToHost);
                cinc->g += 1;
                if (cinc->g == 0) {
                    cudaMemset(cinc->gen, 0, (size_t)cinc->ncap * 4);
                    cinc->g = 1;
                }
                if (n_dirty_v > 0) {
                    int bd = (n_dirty_v + threads - 1) / threads;
                    if (bd < 1) bd = 1;
                    k_csr_gather<<<bd, threads>>>(
                        cinc->nodes, n_dirty_v, cinc->head, cinc->nxt, du, dv,
                        dct, cinc->gen, cinc->g, cinc->elist, cinc->nlist);
                    int nlist = 0;
                    cudaMemcpy(&nlist, cinc->nlist, 4, cudaMemcpyDeviceToHost);
                    if (nlist > 0) {
                        int bl = (nlist + threads - 1) / threads;
                        if (bl < 1) bl = 1;
                        k_rewrite_dirty_fuse_listed<<<bl, threads>>>(
                            cinc->elist, nlist, du, dv, dct, dparent, dkeep,
                            dholes, dnhole, dnout,
                            listed_rebuild() ? damask : nullptr);
                    }
                }
            } else {
                k_rewrite_dirty_fuse<<<be, threads>>>(
                    du, dv, dct, dparent, dirty, n, dkeep, dholes, dnhole, dnout,
                    listed_rebuild() ? damask : nullptr);
            }
            cudaMemcpy(&ndirty_e, dnhole, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&nself, dnout, 4, cudaMemcpyDeviceToHost);
        } else {
            k_rewrite_dirty<<<be, threads>>>(du, dv, dct, dparent, dirty, n, dkeep);
            cudaMemset(dnout, 0, 4);
            k_count_u8<<<be, threads>>>(dkeep, n, dnout);
            cudaMemcpy(&ndirty_e, dnout, 4, cudaMemcpyDeviceToHost);
        }
    }
    if (scan_acc) {
        cudaEventRecord(eb);
        cudaEventSynchronize(eb);
        float ms = 0;
        cudaEventElapsedTime(&ms, ea, eb);
        *scan_acc += (double)ms;
    }
    if (ndirty_e <= 0) {
        n9_dirty_hit(0, 0);
        n21_note_combine(0, 0, 0);
        if (arith) {
            *n_live = nlive_in - nself;
            if (*n_live < 0) *n_live = 0;
        } else {
            cudaMemset(dnout, 0, 4);
            k_count_live<<<be, threads>>>(dct, n, dnout);
            cudaMemcpy(n_live, dnout, 4, cudaMemcpyDeviceToHost);
        }
        if (scan_acc || hash_acc) {
            cudaEventDestroy(ea);
            cudaEventDestroy(eb);
        }
        return 1;
    }
    // Size the table from the dirty set, not from nscan. Emit scans the
    // table, so a table sized for the live array would reintroduce the
    // traffic this lever exists to drop.
    if (hash_acc) cudaEventRecord(ea);
    const bool dedup = hash_dedup_on();
    if (dedup && ndirty_e < 32768 && !emit_holes()) {
        dirty_rbk(du, dv, dsm, dct, dkeep, n, ndirty_e, n_live,
                  be, threads, tu, tv, tsm, tct, dholes, dnhole, dnout);
        // n9_dirty_hit is inside dirty_rbk (hm vs unique m).
        n21_note_combine(ndirty_e, 0, 0);
    } else {
    int ntab_use = next_pow2(ndirty_e * 2 + 1024);
    if (ntab_use > ntab) ntab_use = ntab;
    int tb = (ntab_use + threads - 1) / threads;
    const bool se = slot_emit() && dslots && dnslot && !use_csr;
    // Warp/SM insert was parent-identical and 1.31x slower (n8_hash).
    // Full-dirty RBK was 4.2x slower. Large dirty sets stay on G15 CAS.
    {
        NvRange nv("hash_insert");
        if (se) cudaMemset(dnslot, 0, 4);
        if (listed_insert() || use_csr) {
            int bi = (ndirty_e + threads - 1) / threads;
            if (bi < 1) bi = 1;
            k_hash_insert_listed<<<bi, threads>>>(
                dholes, ndirty_e, du, dv, dsm, dct, dtab, ntab_use, dovf,
                se ? dslots : nullptr, se ? dnslot : nullptr,
                use_csr ? dslots : nullptr);
        } else {
            k_hash_insert<<<be, threads>>>(
                du, dv, dsm, dct, dkeep, n, dtab, ntab_use, dovf,
                se ? dslots : nullptr, se ? dnslot : nullptr);
        }
    }
    static int insert_only = -1;
    if (insert_only < 0) {
        const char* s = std::getenv("WATERZ_HASH_INSERT_ONLY");
        insert_only = (s && std::atoi(s)) ? 1 : 0;
    }
    if (!insert_only) {
    int m = 0;
    const bool eh = emit_holes() && fuse && ndirty_e > 0;
    if (eh) {
        cudaMemset(dnout, 0, 4);
        cudaMemset(dovf, 0, 4);
        if (use_csr && dslots) {
            int bi = (ndirty_e + threads - 1) / threads;
            if (bi < 1) bi = 1;
            k_hash_commit_home<<<bi, threads>>>(
                dholes, ndirty_e, du, dv, dsm, dct, dtab, ntab_use, dslots,
                dnout, listed_rebuild() ? damask : nullptr);
        } else if (se) {
            int nslot = 0;
            cudaMemcpy(&nslot, dnslot, 4, cudaMemcpyDeviceToHost);
            if (nslot > ntab) nslot = ntab;
            if (nslot > 0) {
                int bs = (nslot + threads - 1) / threads;
                if (bs < 1) bs = 1;
                k_hash_emit_slots<<<bs, threads>>>(
                    dtab, ntab_use, dslots, nslot, dholes, ndirty_e,
                    du, dv, dsm, dct, dnout, dovf);
            }
        } else {
            k_hash_emit_holes<<<tb, threads>>>(
                dtab, ntab_use, dholes, ndirty_e, du, dv, dsm, dct, dnout, dovf);
        }
        cudaMemcpy(&m, dnout, 4, cudaMemcpyDeviceToHost);
        int ov = 0;
        cudaMemcpy(&ov, dovf, 4, cudaMemcpyDeviceToHost);
        if (ov) {
            fprintf(stderr, "EMIT_HOLES overflow m=%d nhole=%d\n", m, ndirty_e);
            if (scan_acc || hash_acc) {
                cudaEventDestroy(ea);
                cudaEventDestroy(eb);
            }
            return 0;
        }
        if (m < ndirty_e && !use_csr) {
            int bt = (ndirty_e - m + threads - 1) / threads;
            if (bt < 1) bt = 1;
            k_zero_hole_tail<<<bt, threads>>>(
                dholes, m, ndirty_e, dct, listed_rebuild() ? damask : nullptr);
        }
        n9_dirty_hit(ndirty_e, m);
        n21_note_combine(ndirty_e, m, ntab_use);
    } else {
        k_vacate_kept<<<be, threads>>>(dct, dkeep, n);
        cudaMemset(dnout, 0, 4);
        k_hash_emit<<<tb, threads>>>(dtab, ntab_use, tu, tv, tsm, tct, dnout);
        cudaMemcpy(&m, dnout, 4, cudaMemcpyDeviceToHost);
        n9_dirty_hit(ndirty_e, m);
        n21_note_combine(ndirty_e, m, ntab_use);
        if (m > 0) {
            if (!fuse) {
                cudaMemset(dnhole, 0, 4);
                k_list_holes<<<be, threads>>>(dct, n, dholes, dnhole);
                int nh = 0;
                cudaMemcpy(&nh, dnhole, 4, cudaMemcpyDeviceToHost);
                (void)nh;
            }
            int bm = (m + threads - 1) / threads;
            if (bm < 1) bm = 1;
            k_fill_holes<<<bm, threads>>>(tu, tv, tsm, tct, dholes, m,
                                          du, dv, dsm, dct);
        }
    }
    if (out_m) *out_m = m;
    if (arith) {
        *n_live = nlive_in - nself - ndirty_e + m;
        if (*n_live < 0) *n_live = 0;
    } else {
        cudaMemset(dnout, 0, 4);
        k_count_live<<<be, threads>>>(dct, n, dnout);
        cudaMemcpy(n_live, dnout, 4, cudaMemcpyDeviceToHost);
    }
    } else {
        n9_dirty_hit(ndirty_e, 0);
        n21_note_combine(ndirty_e, 0, ntab_use);
        if (arith) {
            *n_live = nlive_in - nself - ndirty_e;
            if (*n_live < 0) *n_live = 0;
        } else {
            cudaMemset(dnout, 0, 4);
            k_count_live<<<be, threads>>>(dct, n, dnout);
            cudaMemcpy(n_live, dnout, 4, cudaMemcpyDeviceToHost);
        }
    }
    }
    if (hash_acc) {
        cudaEventRecord(eb);
        cudaEventSynchronize(eb);
        float ms = 0;
        cudaEventElapsedTime(&ms, ea, eb);
        *hash_acc += (double)ms;
    }
    if (scan_acc || hash_acc) {
        cudaEventDestroy(ea);
        cudaEventDestroy(eb);
    }
    return 1;
}

static int next_pow2(int x)
{
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

__global__ void k_pack_uvkey(
    const uint32_t* u, const uint32_t* v, unsigned long long* key, int64_t n)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t a = u[i], b = v[i];
    if (a > b) {
        uint32_t t = a;
        a = b;
        b = t;
    }
    key[i] = ((unsigned long long)a << 32) | (unsigned long long)b;
}

__global__ void k_unpack_uvkey(
    const unsigned long long* key, uint32_t* u, uint32_t* v, int64_t n)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    u[i] = (uint32_t)(key[i] >> 32);
    v[i] = (uint32_t)(key[i] & 0xffffffffull);
}

// Build the (u,v) sort key for a selected edge, reading through an index list
// instead of a compacted copy of the edge records.
__global__ void k_key_from_sel(
    const uint32_t* u, const uint32_t* v, const uint32_t* sel,
    unsigned long long* key, int64_t m)
{
    int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (j >= m) return;
    uint32_t e = sel[j];
    uint32_t a = u[e], b = v[e];
    if (a > b) {
        uint32_t t = a;
        a = b;
        b = t;
    }
    key[j] = ((unsigned long long)a << 32) | (unsigned long long)b;
}

// Gather the 16 B payload once, after the key sort has settled the order.
__global__ void k_gather_smct(
    const double* sm, const int64_t* ct, const uint32_t* sel,
    double* smo, int64_t* cto, int64_t m)
{
    int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (j >= m) return;
    uint32_t e = sel[j];
    smo[j] = sm[e];
    cto[j] = ct[e];
}

static int dirty_rbk(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct, uint8_t* dkeep,
    int64_t n, int ndirty_e, int* n_live, int be, int threads,
    uint32_t* tu, uint32_t* tv, double* tsm, int64_t* tct,
    uint32_t* dholes, int* dnhole, int* dnout)
{
    (void)ndirty_e;
    uint32_t* sel = tu;
    uint32_t* sel2 = tv;
    void* tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceSelect::Flagged(
        nullptr, tmp_bytes, thrust::make_counting_iterator<uint32_t>(0),
        dkeep, sel, dnout, (int)n);
    cudaMalloc(&tmp, tmp_bytes ? tmp_bytes : 4);
    cub::DeviceSelect::Flagged(
        tmp, tmp_bytes, thrust::make_counting_iterator<uint32_t>(0),
        dkeep, sel, dnout, (int)n);
    int hm = 0;
    cudaMemcpy(&hm, dnout, 4, cudaMemcpyDeviceToHost);
    if (hm <= 0) {
        cudaFree(tmp);
        cudaMemset(dnout, 0, 4);
        k_count_live<<<be, threads>>>(dct, n, dnout);
        cudaMemcpy(n_live, dnout, 4, cudaMemcpyDeviceToHost);
        return 1;
    }
    unsigned long long* dkey = nullptr;
    unsigned long long* dkeyo = nullptr;
    double* smo = nullptr;
    int64_t* cto = nullptr;
    cudaMalloc(&dkey, (size_t)hm * 8);
    cudaMalloc(&dkeyo, (size_t)hm * 8);
    cudaMalloc(&smo, (size_t)hm * 8);
    cudaMalloc(&cto, (size_t)hm * 8);
    int bm = (hm + threads - 1) / threads;
    if (bm < 1) bm = 1;
    k_key_from_sel<<<bm, threads>>>(du, dv, sel, dkey, hm);
    size_t sb = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sb, dkey, dkeyo, sel, sel2, hm);
    if (sb > tmp_bytes) {
        cudaFree(tmp);
        cudaMalloc(&tmp, sb);
        tmp_bytes = sb;
    }
    cub::DeviceRadixSort::SortPairs(tmp, tmp_bytes, dkey, dkeyo, sel, sel2, hm);
    k_gather_smct<<<bm, threads>>>(dsm, dct, sel2, tsm, tct, hm);
    cub::DeviceReduce::ReduceByKey(
        tmp, tmp_bytes, dkeyo, dkey, tsm, smo, dnout, cub::Sum(), hm);
    int m = 0;
    cudaMemcpy(&m, dnout, 4, cudaMemcpyDeviceToHost);
    cub::DeviceReduce::ReduceByKey(
        tmp, tmp_bytes, dkeyo, dkey, tct, cto, dnout, cub::Sum(), hm);
    n9_dirty_hit(hm, m);
    if (m > 0) {
        int bu = (m + threads - 1) / threads;
        if (bu < 1) bu = 1;
        k_unpack_uvkey<<<bu, threads>>>(dkey, tu, tv, m);
        cudaMemcpy(tsm, smo, (size_t)m * 8, cudaMemcpyDeviceToDevice);
        cudaMemcpy(tct, cto, (size_t)m * 8, cudaMemcpyDeviceToDevice);
    }
    cudaFree(dkey);
    cudaFree(dkeyo);
    cudaFree(smo);
    cudaFree(cto);
    cudaFree(tmp);
    k_vacate_kept<<<be, threads>>>(dct, dkeep, n);
    if (m > 0) {
        cudaMemset(dnhole, 0, 4);
        k_list_holes<<<be, threads>>>(dct, n, dholes, dnhole);
        int nh = 0;
        cudaMemcpy(&nh, dnhole, 4, cudaMemcpyDeviceToHost);
        int bm2 = (m + threads - 1) / threads;
        if (bm2 < 1) bm2 = 1;
        k_fill_holes<<<bm2, threads>>>(tu, tv, tsm, tct, dholes, m,
                                       du, dv, dsm, dct);
        (void)nh;
    }
    cudaMemset(dnout, 0, 4);
    k_count_live<<<be, threads>>>(dct, n, dnout);
    cudaMemcpy(n_live, dnout, 4, cudaMemcpyDeviceToHost);
    return 1;
}

// Measurement hook, not part of the algorithm. The inner loop makes four
// device-to-host copies per iteration: the proposal count, the merge count,
// and the two counts inside compaction. WATERZ_SYNC_PROBE=k adds k more per
// iteration so the marginal cost of one can be read off a slope instead of
// guessed at. See scripts/a4_sync_cost.py.
//
// Each probe launches a kernel first. Without it the probe copies land on a
// stream the preceding real copy already drained, so they return immediately
// and measure nothing; the cost of a round-trip is the pipeline drain, which
// only exists when there is queued work to drain. The atomicAdd of zero
// leaves the counter's value alone.
__global__ void k_probe_touch(int* p)
{
    if (threadIdx.x == 0) atomicAdd(p, 0);
}

static int sync_probe()
{
    const char* s = std::getenv("WATERZ_SYNC_PROBE");
    return s ? std::atoi(s) : 0;
}

// CUB scratch for compact_radix, owned by the caller. compact_radix runs once
// per inner iteration, so allocating inside it would reintroduce exactly the
// per-call cost this is meant to remove.
struct CompactScratch {
    void* tmp;
    size_t bytes;
    int* cnt;  // two slots: selected count, then run count
};

static void compact_scratch_init(CompactScratch& csr, int64_t n)
{
    csr.tmp = nullptr;
    csr.bytes = 0;
    csr.cnt = nullptr;
    if (n < 1) n = 1;
    unsigned long long* k = nullptr;
    uint32_t* s = nullptr;
    double* d = nullptr;
    int* c = nullptr;
    size_t a = 0, b = 0, e = 0;
    cub::DeviceSelect::Flagged(
        nullptr, a, thrust::make_counting_iterator<uint32_t>(0),
        (uint8_t*)nullptr, s, c, (int)n);
    cub::DeviceRadixSort::SortPairs(nullptr, b, k, k, s, s, (int)n);
    cub::DeviceReduce::ReduceByKey(
        nullptr, e, k, k, d, d, c, cub::Sum(), (int)n);
    csr.bytes = a > b ? a : b;
    if (e > csr.bytes) csr.bytes = e;
    cudaMalloc(&csr.tmp, csr.bytes);
    cudaMalloc(&csr.cnt, 8);
}

static void compact_scratch_free(CompactScratch& csr)
{
    cudaFree(csr.tmp);
    cudaFree(csr.cnt);
    csr.tmp = nullptr;
    csr.cnt = nullptr;
    csr.bytes = 0;
}

static int compact_radix(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct, uint8_t* dkeep,
    uint32_t* dparent, int nnode, int64_t n, int64_t* n_out,
    int be, int threads,
    uint32_t* tu, uint32_t* tv, double* tsm, int64_t* tct,
    unsigned long long* dkey, unsigned long long* dkeyo, CompactScratch& csr,
    double TL = 0.0, bool recompress = true)
{
    NvRange nv("compact_radix");
    int bn = (nnode + threads - 1) / threads;
    // k_compress walks each node to its root, so after one pass every entry
    // already points at a root and a second pass cannot change anything. The
    // inner loop compresses immediately before calling here and only k_freeze
    // runs in between, which touches sz and frozen but never parent, so that
    // caller passes recompress=false. It is a full pointer-chasing sweep of
    // all nnode entries and it ran twice per iteration.
    if (recompress) k_compress<<<bn, threads>>>(dparent, nnode);
    k_rewrite<<<be, threads>>>(du, dv, dsm, dct, dparent, n, dkeep, TL);

    // Select surviving edge INDICES rather than edge records.
    //
    // This previously ran copy_if over a zip_iterator of (u,v,sum,count) and
    // then sort_by_key with a zip_iterator of (sum,count) as the value. Both
    // drag a 24 B / 16 B tuple payload through every pass as strided tuple
    // loads and stores, which is the slow path in thrust. Sorting a 64-bit key
    // against a 32-bit index, then gathering the payload once at the end,
    // moves far less traffic and lets thrust dispatch a plain radix sort.
    //
    // `tu` is unused from here on and is exactly n uint32s, so it carries the
    // index list with no extra allocation, and `tv` takes the sorted copy.
    //
    // These four steps used to be thrust calls. Three of them return an
    // iterator or a count, so thrust had to synchronize the device to hand it
    // back, and each also allocated its own temporaries; at ~1840 compactions
    // that was 8766 ms, the largest phase left after the proposal sort was
    // fused. The CUB equivalents write their counts to device memory and take
    // caller-owned scratch, which leaves just two D2H copies per compaction,
    // for the two counts that genuinely decide later launch geometry.
    uint32_t* sel = tu;
    uint32_t* sel2 = tv;
    cub::DeviceSelect::Flagged(
        csr.tmp, csr.bytes, thrust::make_counting_iterator<uint32_t>(0),
        dkeep, sel, csr.cnt, (int)n);
    int hm = 0;
    cudaMemcpy(&hm, csr.cnt, 4, cudaMemcpyDeviceToHost);
    int64_t m = hm;
    if (m <= 0) {
        *n_out = 0;
        return 1;
    }
    int bm = (int)((m + threads - 1) / threads);
    if (bm < 1) bm = 1;
    k_key_from_sel<<<bm, threads>>>(du, dv, sel, dkey, m);
    cub::DeviceRadixSort::SortPairs(
        csr.tmp, csr.bytes, dkey, dkeyo, sel, sel2, (int)m);
    k_gather_smct<<<bm, threads>>>(dsm, dct, sel2, tsm, tct, m);

    // Two plain ReduceByKey passes instead of one zip pass: the segment
    // boundaries are recomputed, but each pass is a contiguous scan over a
    // single array. Sums are integral affinity-byte counts held exactly in a
    // double (see k_scale_sm_bytes), so the reduction is exact either way.
    // dkey is free once the sort has consumed it, so it takes the unique keys.
    cub::DeviceReduce::ReduceByKey(
        csr.tmp, csr.bytes, dkeyo, dkey, tsm, dsm, csr.cnt + 1, cub::Sum(),
        (int)m);
    int hm2 = 0;
    cudaMemcpy(&hm2, csr.cnt + 1, 4, cudaMemcpyDeviceToHost);
    int64_t m2 = hm2;
    cub::DeviceReduce::ReduceByKey(
        csr.tmp, csr.bytes, dkeyo, dkey, tct, dct, csr.cnt + 1, cub::Sum(),
        (int)m);
    k_unpack_uvkey<<<(int)((m2 + threads - 1) / threads), threads>>>(dkey, du, dv, m2);
    *n_out = m2;
    return 1;
}

struct P0yProf {
    double compact_ms;
    double propose_ms;
    double accept_ms;
    double d2h_ms;
    double memset_ms;
    double debug_ms;
    int64_t* hist_nlive;
    int64_t* hist_nprop;
    int64_t* hist_nmerge;
    // Optional. The per-node sweeps cost nnode regardless of how many roots
    // are still live, so the live-root count is what says whether working off
    // a list would help. Counting it costs an extra nnode sweep per inner
    // iteration, so it is only done when this is supplied. Results are
    // staged on the device and copied back once, to avoid adding 1800
    // synchronizations to the loop being measured.
    int64_t* hist_nact;
    int hist_cap;
    int hist_n;
    int skip_debug;
};

struct EvAccum {
    cudaEvent_t a, b;
    double* dest;
    explicit EvAccum(double* d) : dest(d)
    {
        cudaEventCreate(&a);
        cudaEventCreate(&b);
    }
    ~EvAccum()
    {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    void start() { if (dest) cudaEventRecord(a); }
    void stop()
    {
        if (!dest) return;
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        *dest += (double)ms;
    }
};

static int parhac_dev(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct, int64_t n_edges,
    const double* aff_thr, int n_thr, double eps,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out,
    P0yProf* prof = nullptr)
{
    const int nnode = (int)max_id + 1;
    uint32_t *dparent, *dsz, *dsz0, *dreds, *dblues, *dadd;
    unsigned long long *dprop;
    uint32_t *tu, *tv;
    double *dblk, *tsm;
    int64_t *tct;
    uint8_t *dkeep, *dcolor, *dfrozen;
    float *dpris;
    int *dnmerge, *dnprop, *ddbg;
    cudaMalloc(&dparent, (size_t)nnode * 4);
    cudaMalloc(&dsz, (size_t)nnode * 4);
    cudaMalloc(&dsz0, (size_t)nnode * 4);
    cudaMalloc(&dcolor, (size_t)nnode);
    cudaMalloc(&dfrozen, (size_t)nnode);
    cudaMalloc(&dprop, (size_t)nnode * 8);
    cudaMalloc(&dreds, (size_t)nnode * 4);
    cudaMalloc(&dblues, (size_t)nnode * 4);
    cudaMalloc(&dadd, (size_t)nnode * 4);
    cudaMalloc(&dpris, (size_t)nnode * 4);
    cudaMalloc(&dnmerge, 4);
    cudaMalloc(&dnprop, 4);
    cudaMalloc(&ddbg, 20);
    cudaMalloc(&dkeep, (size_t)n_edges);
    cudaMalloc(&tu, (size_t)n_edges * 4);
    cudaMalloc(&tv, (size_t)n_edges * 4);
    cudaMalloc(&tsm, (size_t)n_edges * 8);
    cudaMalloc(&tct, (size_t)n_edges * 8);
    int nblk = 256;
    cudaMalloc(&dblk, (size_t)nblk * 8);

    int threads = 256;
    int bn = (nnode + threads - 1) / threads;
    k_init_parent<<<bn, threads>>>(dparent, dsz, nnode);
    cudaDeviceSynchronize();

    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });
    std::vector<double> hblk((size_t)nblk);
    std::vector<uint32_t> hparent((size_t)nnode);
    int64_t nlive = n_edges;
    EvAccum ev_compact(prof ? &prof->compact_ms : nullptr);
    EvAccum ev_propose(prof ? &prof->propose_ms : nullptr);
    EvAccum ev_accept(prof ? &prof->accept_ms : nullptr);
    EvAccum ev_d2h(prof ? &prof->d2h_ms : nullptr);
    EvAccum ev_memset(prof ? &prof->memset_ms : nullptr);
    EvAccum ev_debug(prof ? &prof->debug_ms : nullptr);

    int* dnacts = nullptr;
    if (prof && prof->hist_nact && prof->hist_cap > 0) {
        cudaMalloc(&dnacts, (size_t)prof->hist_cap * 4);
        cudaMemset(dnacts, 0, (size_t)prof->hist_cap * 4);
    }

    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti];
        int64_t nmerge = 0, ninner = 0, nouter = 0;
        for (int layer = 0; layer < 10000; ++layer) {
            int be = (int)((nlive + threads - 1) / threads);
            if (be < 1) be = 1;
            k_compress<<<bn, threads>>>(dparent, nnode);
            k_wmax_live<<<nblk, 256>>>(du, dv, dsm, dct, dparent, nlive, dblk);
            ev_d2h.start();
            cudaMemcpy(hblk.data(), dblk, (size_t)nblk * 8, cudaMemcpyDeviceToHost);
            ev_d2h.stop();
            double wmax = 0;
            for (double x : hblk) if (x > wmax) wmax = x;
            if ((layer & 7) == 0 && !(prof && prof->skip_debug))
                std::fprintf(stderr, "E6r_WMAX T=%.2f layer=%d wmax=%.6f nlive=%lld\n",
                    T, layer, wmax, (long long)nlive);
            if (wmax <= T) break;
            double TL = wmax / (1.0 + eps);
            if (TL < T) TL = T;
            // Full live graph only; k_propose applies the TL cutoff (CPU layer is a view).
            ev_compact.start();
            compact_and_combine(du, dv, dsm, dct, dkeep, dparent, nnode, nlive, &nlive, 0.0,
                be, threads, tu, tv, tsm, tct);
            ev_compact.stop();
            if (nlive <= 0) break;
            be = (int)((nlive + threads - 1) / threads);
            if (be < 1) be = 1;
            int layer_merges = 0;
            for (int outer = 0; outer < 64; ++outer) {
                k_compress<<<bn, threads>>>(dparent, nnode);
                k_zero_sz<<<bn, threads>>>(dsz, nnode);
                k_rebuild_sz<<<bn, threads>>>(dparent, dsz, nnode);
                ev_memset.start();
                cudaMemset(dcolor, 0, (size_t)nnode);
                cudaMemset(dfrozen, 0, (size_t)nnode);
                ev_memset.stop();
                k_color<<<bn, threads>>>(dcolor, dparent, nnode,
                    0xC0FFEEULL + (uint64_t)layer * 10007 + (uint64_t)outer);
                k_copy_sz<<<bn, threads>>>(dsz, dsz0, nnode);
                for (int inner = 0; inner < 64; ++inner) {
                    ev_memset.start();
                    cudaMemset(dprop, 0, (size_t)nnode * 8);
                    cudaMemset(dnmerge, 0, 4);
                    cudaMemset(dnprop, 0, 4);
                    cudaMemset(ddbg, 0, 20);
                    ev_memset.stop();
                    ev_propose.start();
                    k_propose<<<be, threads>>>(
                        du, dv, dsm, dct, dparent, dsz, dcolor, dfrozen,
                        nlive, TL, dprop,
                        0xA5A5ULL + (uint64_t)inner * 17 + (uint64_t)outer,
                        0xC0FFEEULL + (uint64_t)layer * 10007 + (uint64_t)outer,
                        ddbg, nullptr);
                    k_pack_prop<<<bn, threads>>>(
                        dprop, dsz, nnode, dreds, dblues, dadd, dpris, dnprop);
                    ev_propose.stop();
                    int nprop = 0;
                    ev_d2h.start();
                    cudaMemcpy(&nprop, dnprop, 4, cudaMemcpyDeviceToHost);
                    ev_d2h.stop();
                    if (nprop == 0 && inner == 0 && (outer == 0 || layer_merges == 0)
                        && !(prof && prof->skip_debug)) {
                        int hdbg[5] = {0};
                        cudaMemcpy(hdbg, ddbg, 20, cudaMemcpyDeviceToHost);
                        std::fprintf(stderr,
                            "E6r_DBG T=%.2f L=%d o=%d live=%lld TL=%.4f "
                            "alive=%d geTL=%d inter=%d rb=%d szok=%d\n",
                            T, layer, outer, (long long)nlive, TL,
                            hdbg[0], hdbg[1], hdbg[2], hdbg[3], hdbg[4]);
                    }
                    if (nprop > 0) {
                        ev_accept.start();
                        thrust::sort_by_key(
                            thrust::device_ptr<float>(dpris),
                            thrust::device_ptr<float>(dpris) + nprop,
                            thrust::make_zip_iterator(thrust::make_tuple(
                                thrust::device_ptr<uint32_t>(dreds),
                                thrust::device_ptr<uint32_t>(dblues),
                                thrust::device_ptr<uint32_t>(dadd))),
                            thrust::greater<float>());
                        thrust::stable_sort_by_key(
                            thrust::device_ptr<uint32_t>(dreds),
                            thrust::device_ptr<uint32_t>(dreds) + nprop,
                            thrust::make_zip_iterator(thrust::make_tuple(
                                thrust::device_ptr<uint32_t>(dblues),
                                thrust::device_ptr<uint32_t>(dadd),
                                thrust::device_ptr<float>(dpris))));
                        k_accept_serial<<<1, 1>>>(
                            dreds, dblues, dadd, nprop, dparent, dsz, dfrozen, eps, dnmerge);
                        ev_accept.stop();
                    }
                    int hm = 0;
                    ev_d2h.start();
                    cudaMemcpy(&hm, dnmerge, 4, cudaMemcpyDeviceToHost);
                    ev_d2h.stop();
                    k_compress<<<bn, threads>>>(dparent, nnode);
                    k_freeze<<<bn, threads>>>(dparent, dsz, dsz0, dcolor, dfrozen, nnode, eps);
                    if (prof && prof->hist_n < prof->hist_cap) {
                        int hi = prof->hist_n++;
                        if (prof->hist_nlive) prof->hist_nlive[hi] = nlive;
                        if (prof->hist_nprop) prof->hist_nprop[hi] = nprop;
                        if (prof->hist_nmerge) prof->hist_nmerge[hi] = hm;
                        if (dnacts) k_count_roots<<<bn, threads>>>(
                            dparent, nnode, dnacts + hi);
                    }
                    ++ninner;
                    nmerge += hm;
                    layer_merges += hm;
                    if (hm == 0) break;
                    ev_compact.start();
                    compact_and_combine(du, dv, dsm, dct, dkeep, dparent, nnode, nlive, &nlive, 0.0,
                        be, threads, tu, tv, tsm, tct);
                    ev_compact.stop();
                    be = (int)((nlive + threads - 1) / threads);
                    if (be < 1) be = 1;
                    if (nlive <= 0) break;
                }
                ++nouter;
                if (nlive <= 0) break;
            }
            if (layer_merges == 0) {
                std::fprintf(stderr, "E6r_STUCK T=%.2f layer=%d TL=%.4f nlive=%lld\n",
                    T, layer, TL, (long long)nlive);
                break;
            }
        }
        cudaMemcpy(hparent.data(), dparent, (size_t)nnode * 4, cudaMemcpyDeviceToHost);
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (int i = 0; i < nnode; ++i) {
            uint32_t x = (uint32_t)i;
            int guard = 0;
            while (hparent[x] != x && guard++ < 64) x = hparent[x];
            dst[i] = x;
        }
        if (stats_out) {
            stats_out[ti * 3 + 0] = nouter;
            stats_out[ti * 3 + 1] = nmerge;
            stats_out[ti * 3 + 2] = ninner;
        }
        if (!(prof && prof->skip_debug)) {
            ev_debug.start();
            k_compress<<<bn, threads>>>(dparent, nnode);
            std::vector<uint32_t> hsz((size_t)nnode), hp((size_t)nnode);
            std::vector<uint32_t> hu((size_t)nlive), hv((size_t)nlive);
            std::vector<double> hsm((size_t)nlive);
            std::vector<int64_t> hct((size_t)nlive);
            cudaMemcpy(hsz.data(), dsz, (size_t)nnode * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hp.data(), dparent, (size_t)nnode * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hu.data(), du, (size_t)nlive * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hv.data(), dv, (size_t)nlive * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hsm.data(), dsm, (size_t)nlive * 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(hct.data(), dct, (size_t)nlive * 8, cudaMemcpyDeviceToHost);
            int64_t n_hi = 0, n_fit = 0, n_small = 0;
            double mx = 0;
            for (int64_t i = 0; i < nlive; ++i) {
                if (hct[(size_t)i] < 1) continue;
                uint32_t a = hu[(size_t)i], b = hv[(size_t)i];
                while (hp[a] != a) a = hp[a];
                while (hp[b] != b) b = hp[b];
                if (a == b || a == 0 || b == 0) continue;
                double mean = hsm[(size_t)i] / (double)hct[(size_t)i];
                if (mean > mx) mx = mean;
                if (mean < 0.9) continue;
                ++n_hi;
                uint32_t sa = hsz[a], sb = hsz[b];
                uint32_t lo = sa < sb ? sa : sb, hi = sa < sb ? sb : sa;
                if (hi > 0 && (uint64_t)lo * 100 <= (uint64_t)hi * 8) ++n_fit;
                if (lo <= 4) ++n_small;
            }
            std::fprintf(stderr,
                "E6r_CEN T=%.2f wmax=%.6f n_mean09=%lld n_fit_cap=%lld n_small=%lld\n",
                T, mx, (long long)n_hi, (long long)n_fit, (long long)n_small);
            ev_debug.stop();
        }
        std::fprintf(stderr,
            "E6r T=%.2f eps=%.4f outer=%lld inner=%lld merges=%lld nlive=%lld\n",
            T, eps, (long long)nouter, (long long)ninner, (long long)nmerge,
            (long long)nlive);
    }
    if (dnacts) {
        std::vector<int> hn((size_t)prof->hist_cap);
        cudaMemcpy(hn.data(), dnacts, (size_t)prof->hist_cap * 4,
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < prof->hist_n; ++i)
            prof->hist_nact[i] = (int64_t)hn[(size_t)i];
        cudaFree(dnacts);
    }
    cudaFree(dparent); cudaFree(dsz); cudaFree(dsz0); cudaFree(dcolor);
    cudaFree(dfrozen); cudaFree(dprop); cudaFree(dpris);
    cudaFree(dreds); cudaFree(dblues); cudaFree(dadd);
    cudaFree(dnmerge); cudaFree(dnprop); cudaFree(ddbg); cudaFree(dkeep);
    cudaFree(tu); cudaFree(tv); cudaFree(tsm); cudaFree(tct); cudaFree(dblk);
    return 1;
}

struct P0zProf {
    int64_t* hist_nlive;
    int64_t* hist_nprop;
    int64_t* hist_nmerge;
    int64_t* hist_ngc;
    int64_t* hist_nact;
    int64_t* hist_ndirty;
    int64_t* hist_nstar;
    int hist_cap;
    int hist_n;
    int n_layer;
};

struct P0aaProf {
    double compact_ms;
    double propose_ms;
    double pack_ms;
    double sort_ms;
    double accept_ms;
    double compress_ms;
    double freeze_ms;
    double color_ms;
    double memset_ms;
    double d2h_ms;
    double host_ms;
    int n_layer;
    double rewrite_scan_ms;
    double hash_ms;
    double compact_radix_ms;
    int layer_outers[64];
    int layer_merges[64];
    // A2 work accounting. These are counts, not times, so they are unaffected
    // by anything else running on the card.
    //   layer_first_zero  outer index at which the layer ran out of above-TL
    //                     edges, i.e. where the outer loop could have stopped;
    //                     -1 if it never did within the cap
    //   sum_nlive         live edges visited summed over inner iterations, the
    //                     quantity a dirty-set compaction has to reduce
    //   sum_above         of those, how many were actually eligible to merge
    int layer_first_zero[64];
    int64_t sum_nlive;
    int64_t sum_above;
};

static double g_iou_scan = 0;
static double g_iou_hash = 0;
static double g_iou_radix = 0;

extern "C" void parhac_compact_iou(double* scan, double* hash, double* radix)
{
    if (scan) *scan = g_iou_scan;
    if (hash) *hash = g_iou_hash;
    if (radix) *radix = g_iou_radix;
}

static int parhac_e6s_dev(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct, int64_t n_edges,
    const double* aff_thr, int n_thr, double eps,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out,
    P0zProf* zprof = nullptr, P0aaProf* aa = nullptr, int max_outer = 64)
{
    set_size_asym_from_env();
    const int nnode = (int)max_id + 1;
    uint32_t *dparent, *dsz, *dsz0, *dreds, *dblues, *dadd;
    unsigned long long *dprop;
    uint32_t *tu, *tv;
    double *dblk, *tsm;
    int64_t *tct;
    uint8_t *dkeep, *dcolor, *dfrozen;
    int *dnmerge, *dnprop, *ddbg, *dnact, *dndirty, *dnstar;
    uint8_t *ddirty_blue, *ddirty_star;
    int ntab = next_pow2((int)(n_edges * 2 + 1024));
    if (ntab < 1024) ntab = 1024;
    HSlot* dtab = nullptr;
    int* dnout = nullptr;
    uint32_t* dslots = nullptr;
    int* dnslot = nullptr;
    cudaMalloc(&dparent, (size_t)nnode * 4);
    cudaMalloc(&dsz, (size_t)nnode * 4);
    cudaMalloc(&dsz0, (size_t)nnode * 4);
    cudaMalloc(&dcolor, (size_t)nnode);
    cudaMalloc(&dfrozen, (size_t)nnode);
    cudaMalloc(&dprop, (size_t)nnode * 8);
    cudaMalloc(&dreds, (size_t)nnode * 4);
    cudaMalloc(&dblues, (size_t)nnode * 4);
    cudaMalloc(&dadd, (size_t)nnode * 4);
    cudaMalloc(&dnmerge, 4);
    // Two adjacent words: the proposal count and the above-TL edge count. They
    // are produced in the same inner iteration and consumed together, so
    // keeping them adjacent lets the layer-exit test ride along on the
    // device-to-host copy the loop already makes for nprop instead of adding a
    // second round-trip per iteration.
    cudaMalloc(&dnprop, 8);
    int* dnabove = dnprop + 1;
    // Proposal sort buffers and CUB scratch, sized once for the largest
    // possible proposal count. Sizing per call was the whole problem: thrust
    // allocated its own temporaries and synchronized on every one of the ~3700
    // sort calls, and each of those syncs waited for the other processes on the
    // card to drain too. CUB's byte requirement is monotonic in item count, so
    // the query at nnode bounds every smaller call and the buffer is reused.
    unsigned long long *pkey_in = nullptr, *pkey_out = nullptr;
    unsigned long long *ppay_in = nullptr, *ppay_out = nullptr;
    void* psort_tmp = nullptr;
    size_t psort_bytes = 0;
    cudaMalloc(&pkey_in, (size_t)nnode * 8);
    cudaMalloc(&pkey_out, (size_t)nnode * 8);
    cudaMalloc(&ppay_in, (size_t)nnode * 8);
    cudaMalloc(&ppay_out, (size_t)nnode * 8);
    cub::DeviceRadixSort::SortPairs(nullptr, psort_bytes, pkey_in, pkey_out,
                                    ppay_in, ppay_out, nnode);
    cudaMalloc(&psort_tmp, psort_bytes);
    CompactScratch csr;
    compact_scratch_init(csr, n_edges);
    ddbg = dnact = dndirty = dnstar = nullptr;
    ddirty_blue = ddirty_star = nullptr;
    if (zprof) {
        cudaMalloc(&ddbg, 20);
        cudaMalloc(&dnact, 4);
        cudaMalloc(&dndirty, 4);
        cudaMalloc(&dnstar, 4);
        cudaMalloc(&ddirty_blue, (size_t)nnode);
        cudaMalloc(&ddirty_star, (size_t)nnode);
    }
    cudaMalloc(&dkeep, (size_t)n_edges);
    cudaMalloc(&tu, (size_t)n_edges * 4);
    cudaMalloc(&tv, (size_t)n_edges * 4);
    cudaMalloc(&tsm, (size_t)n_edges * 8);
    cudaMalloc(&tct, (size_t)n_edges * 8);
    cudaMalloc(&dtab, (size_t)ntab * sizeof(HSlot));
    cudaMalloc(&dnout, 4);
    if (slot_emit() || csr_rewrite()) {
        cudaMalloc(&dslots, (size_t)ntab * 4);
        if (slot_emit())
            cudaMalloc(&dnslot, 4);
    }
    // Both of these are cleared once here rather than once per inner
    // iteration: k_hash_emit and k_pack_prop_fused each reset the entries they
    // consume, so both arrive empty.
    {
        int tb0 = (ntab + 255) / 256;
        k_hash_clear<<<tb0, 256>>>(dtab, ntab);
    }
    cudaMemset(dprop, 0, (size_t)nnode * 8);
    // hash_combine_live adopts its output by swapping du/dv/dsm/dct with
    // tu/tv/tsm/tct, so after an odd number of compactions the t* names hold
    // the caller's arrays and the d* names hold these allocations. Remember
    // what was allocated here and free that, or the swap turns into a free of
    // memory this function does not own.
    uint32_t* alloc_tu = tu;
    uint32_t* alloc_tv = tv;
    double* alloc_tsm = tsm;
    int64_t* alloc_tct = tct;
    unsigned long long *dkey, *dkeyo;
    cudaMalloc(&dkey, (size_t)n_edges * 8);
    cudaMalloc(&dkeyo, (size_t)n_edges * 8);
    int nblk = 256;
    cudaMalloc(&dblk, (size_t)nblk * 8);

    int threads = 256;
    int bn = (nnode + threads - 1) / threads;
    k_init_parent<<<bn, threads>>>(dparent, dsz, nnode);
    cudaDeviceSynchronize();

    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });
    std::vector<double> hblk((size_t)nblk);
    std::vector<uint32_t> hparent((size_t)nnode);
    int64_t nlive = n_edges;
    const int nprobe = sync_probe();
    // Hash combine is the default for the inner-loop compaction; the radix one
    // is kept reachable with WATERZ_E6S_HASH=0 so the two can still be A/B'd,
    // since the hash path changes the emitted edge order and the radix path is
    // the reference that established the fingerprint.
    const char* hs = std::getenv("WATERZ_E6S_HASH");
    const bool use_hash = !(hs && std::atoi(hs) == 0);
    const char* ce = std::getenv("WATERZ_COMPACT_EVERY");
    const int compact_k = ce ? std::atoi(ce) : 0;
    // G-series work-efficiency levers, as a bitmask so each can be A/B'd on
    // its own against this build. Every one of them is bit-identical by
    // construction rather than by tuning, see the equivalence argument next
    // to each, and scripts/g0_agg_ref.py checks that claim on the real val
    // RAG with a CPU replica of this loop. None has been run on a device yet,
    // so they default off: the locked fingerprint in a1_e6s_voi.json is what
    // this file is for, and a lever that cannot be measured must not be able
    // to change it. scripts/g_levers.py turns them on one at a time.
    //
    //   1  k_freeze_reds for k_freeze, which also retires dcolor and k_color
    //   2  dirty-set hash combine (G2); physical compact stays at layer entry
    //   4  candidate-blue list AND the active-edge list (G3). The edge list
    //      is rebuilt from the dirty set after each G2 compact; without G2
    //      it is rebuilt from scratch after the full combine, which is
    //      still cheaper than proposing over every live edge.
    //   8  incremental root list (G4): skip the per-outer full-nnode
    //      compress/zero/rebuild, copy sz and clear frozen over live
    //      roots only, compact the list after each accept, compress
    //      accepted blues only. sz is the incremental value k_accept_reds
    //      already maintains.
    const char* lvs = std::getenv("WATERZ_AGG_LEVERS");
    const int levers = lvs ? std::atoi(lvs) : 0;
    const bool lv_freeze_reds = (levers & 1) != 0;
    const bool lv_dirty = (levers & 2) != 0;
    const bool lv_bluelist = (levers & 4) != 0;
    const bool lv_rootlist = (levers & 8) != 0;
    uint32_t* dblist = nullptr;
    int* dnlist = nullptr;
    if (lv_bluelist) {
        // At most one entry per blue, and a blue is a distinct root, so nnode
        // bounds the list for any inner iteration.
        cudaMalloc(&dblist, (size_t)nnode * 4);
        cudaMalloc(&dnlist, 4);
    }
    uint8_t* ddirty = nullptr;
    uint32_t* dholes = nullptr;
    int* dnhole = nullptr;
    uint8_t* damask = nullptr;
    uint32_t* dalist = nullptr;
    uint32_t* dalist2 = nullptr;
    int* dnact_l = nullptr;
    uint32_t* droots = nullptr;
    uint32_t* drootflag = nullptr;
    int* dnroot = nullptr;
    int nroot = nnode > 0 ? nnode - 1 : 0;
    int nact_l = 0;
    int64_t nscan = nlive;
    if (lv_dirty) {
        cudaMalloc(&ddirty, (size_t)nnode);
        cudaMalloc(&dholes, (size_t)n_edges * 4);
        cudaMalloc(&dnhole, 4);
    }
    CsrInc inc{};
    inc.nnode = nnode;
    inc.ncap = n_edges;
    inc.on = csr_rewrite() && lv_dirty;
    if (inc.on) {
        cudaMalloc(&inc.head, (size_t)nnode * 4);
        cudaMalloc(&inc.nxt, (size_t)n_edges * 2 * 4);
        cudaMalloc(&inc.gen, (size_t)n_edges * 4);
        cudaMalloc(&inc.elist, (size_t)n_edges * 4);
        cudaMalloc(&inc.nodes, (size_t)nnode * 4);
        cudaMalloc(&inc.nnodes, 4);
        cudaMalloc(&inc.nlist, 4);
        if (!inc.head || !inc.nxt || !inc.gen || !inc.elist || !inc.nodes
            || !inc.nnodes || !inc.nlist) {
            std::fprintf(stderr, "WATERZ_CSR_REWRITE OOM n_edges=%lld nnode=%d\n",
                         (long long)n_edges, nnode);
            cudaFree(inc.head); cudaFree(inc.nxt); cudaFree(inc.gen);
            cudaFree(inc.elist); cudaFree(inc.nodes);
            cudaFree(inc.nnodes); cudaFree(inc.nlist);
            return 0;
        }
    }
    if (lv_bluelist) {
        cudaMalloc(&damask, (size_t)n_edges);
        cudaMalloc(&dalist, (size_t)n_edges * 4);
        cudaMalloc(&dnact_l, 4);
        if (listed_rebuild())
            cudaMalloc(&dalist2, (size_t)n_edges * 4);
    }
    if (lv_rootlist) {
        cudaMalloc(&droots, (size_t)nnode * 4);
        cudaMalloc(&drootflag, (size_t)nnode * 4);
        cudaMalloc(&dnroot, 4);
        k_init_root_list<<<bn, threads>>>(droots, nnode);
    }
    int* dovf = nullptr;
    cudaMalloc(&dovf, 4);
    cudaMemset(dovf, 0, 4);
    if (max_outer < 1) max_outer = 64;
    {
        const char* s = std::getenv("WATERZ_MAX_OUTER");
        if (s && *s) {
            int v = std::atoi(s);
            if (v > 0) max_outer = v;
        }
    }
    {
        int be0 = (int)((n_edges + threads - 1) / threads);
        if (be0 < 1) be0 = 1;
        k_scale_sm_bytes<<<be0, threads>>>(dsm, n_edges);
    }
    EvAccum ev_compact(aa ? &aa->compact_ms : nullptr);
    EvAccum ev_radix(aa ? &aa->compact_radix_ms : nullptr);
    EvAccum ev_propose(aa ? &aa->propose_ms : nullptr);
    EvAccum ev_pack(aa ? &aa->pack_ms : nullptr);
    EvAccum ev_sort(aa ? &aa->sort_ms : nullptr);
    EvAccum ev_accept(aa ? &aa->accept_ms : nullptr);
    EvAccum ev_compress(aa ? &aa->compress_ms : nullptr);
    EvAccum ev_freeze(aa ? &aa->freeze_ms : nullptr);
    EvAccum ev_color(aa ? &aa->color_ms : nullptr);
    EvAccum ev_memset(aa ? &aa->memset_ms : nullptr);
    EvAccum ev_d2h(aa ? &aa->d2h_ms : nullptr);
    EvAccum ev_host(aa ? &aa->host_ms : nullptr);

    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti] * SM_BYTE_SCALE;
        int64_t nmerge = 0, ninner = 0, nouter = 0;
        for (int layer = 0; layer < 10000; ++layer) {
            int be = (int)((nscan + threads - 1) / threads);
            if (be < 1) be = 1;
            k_compress<<<bn, threads>>>(dparent, nnode);
            k_wmax_live<<<nblk, 256>>>(du, dv, dsm, dct, dparent, nscan, dblk);
            ev_d2h.start();
            cudaMemcpy(hblk.data(), dblk, (size_t)nblk * 8, cudaMemcpyDeviceToHost);
            ev_d2h.stop();
            double wmax = 0;
            for (double x : hblk) if (x > wmax) wmax = x;
            if (wmax <= T) break;
            if (zprof) zprof->n_layer += 1;
            if (aa) aa->n_layer += 1;
            double TL = wmax / (1.0 + eps);
            if (TL < T) TL = T;
            ev_compact.start();
            ev_radix.start();
            compact_radix(du, dv, dsm, dct, dkeep, dparent, nnode, nscan, &nlive,
                be, threads, tu, tv, tsm, tct, dkey, dkeyo, csr);
            ev_radix.stop();
            ev_compact.stop();
            if (nlive <= 0) break;
            nscan = nlive;
            be = (int)((nscan + threads - 1) / threads);
            if (be < 1) be = 1;
            if (inc.on)
                csr_rebuild(inc, du, dv, dct, (int)nscan, threads);
            if (lv_bluelist) {
                k_init_amask<<<be, threads>>>(dsm, dct, nscan, TL, damask);
                cudaMemset(dnact_l, 0, 4);
                k_pack_amask<<<be, threads>>>(damask, dct, nscan, dalist, dnact_l);
                cudaMemcpy(&nact_l, dnact_l, 4, cudaMemcpyDeviceToHost);
            }
            int layer_merges = 0;
            int layer_outers = 0;
            int layer_first_zero = -1;
            bool layer_done = false;
            for (int outer = 0; outer < max_outer; ++outer) {
                ev_compress.start();
                // G4: k_accept_reds already keeps sz[root] equal to the
                // member count. Rebuilding it from all nnode is the same
                // numbers written back onto the same roots. Frozen and
                // sz0 are only read at current roots. parent[i] for a
                // never-merged fragment is still i, and after a merge the
                // only stale parent slots are on absorbed blues, which
                // propose finds through. Skipping the three full-nnode
                // passes is therefore a no-op for every decision.
                if (!lv_rootlist) {
                    k_compress<<<bn, threads>>>(dparent, nnode);
                    k_zero_sz<<<bn, threads>>>(dsz, nnode);
                    k_rebuild_sz<<<bn, threads>>>(dparent, dsz, nnode);
                }
                ev_compress.stop();
                ev_memset.start();
                // dcolor exists only for k_freeze. k_propose derives a node's
                // colour inline from color_of() and marks the array (void), so
                // once k_freeze_reds is in play nothing reads dcolor at all and
                // both the clear and k_color are dead. nullptr is passed to
                // k_propose in that case so a future reader faults loudly
                // instead of reading a stale buffer.
                if (!lv_freeze_reds) cudaMemset(dcolor, 0, (size_t)nnode);
                if (lv_rootlist) {
                    int br = (nroot + threads - 1) / threads;
                    if (br < 1) br = 1;
                    k_clear_frozen_list<<<br, threads>>>(
                        dfrozen, droots, nroot);
                } else {
                    cudaMemset(dfrozen, 0, (size_t)nnode);
                }
                ev_memset.stop();
                ev_color.start();
                if (!lv_freeze_reds)
                    k_color<<<bn, threads>>>(dcolor, dparent, nnode,
                        0xC0FFEEULL + (uint64_t)layer * 10007 + (uint64_t)outer);
                if (lv_rootlist) {
                    int br = (nroot + threads - 1) / threads;
                    if (br < 1) br = 1;
                    if (outer == 0 || !sticky_sz0())
                        k_copy_sz_list<<<br, threads>>>(dsz, dsz0, droots, nroot);
                } else {
                    if (outer == 0 || !sticky_sz0())
                        k_copy_sz<<<bn, threads>>>(dsz, dsz0, nnode);
                }
                ev_color.stop();
                for (int inner = 0; inner < 64; ++inner) {
                    ev_memset.start();
                    // dprop is not cleared here: k_pack_prop_fused resets the
                    // entries it consumes, so it arrives empty.
                    cudaMemset(dnmerge, 0, 4);
                    cudaMemset(dnprop, 0, 8);
                    if (zprof) cudaMemset(ddbg, 0, 20);
                    ev_memset.stop();
                    ev_propose.start();
                    {
                    NvRange nv("propose");
                    if (lv_bluelist && nact_l > 0) {
                        cudaMemset(dnlist, 0, 4);
                        int ba = (nact_l + threads - 1) / threads;
                        if (ba < 1) ba = 1;
                        k_propose_listed<<<ba, threads>>>(
                            dalist, nact_l, du, dv, dsm, dct, dparent, dsz,
                            dfrozen, TL, dprop, dblist, dnlist,
                            0xA5A5ULL + (uint64_t)inner * 17 + (uint64_t)outer,
                            0xC0FFEEULL + (uint64_t)layer * 10007 + (uint64_t)outer,
                            dnabove);
                    } else if (lv_bluelist) {
                        cudaMemset(dnlist, 0, 4);
                        k_propose_bluelist<<<be, threads>>>(
                            du, dv, dsm, dct, dparent, dsz, dfrozen,
                            nscan, TL, dprop, dblist, dnlist,
                            0xA5A5ULL + (uint64_t)inner * 17 + (uint64_t)outer,
                            0xC0FFEEULL + (uint64_t)layer * 10007 + (uint64_t)outer,
                            dnabove);
                    } else {
                        k_propose<<<be, threads>>>(
                            du, dv, dsm, dct, dparent, dsz,
                            lv_freeze_reds ? nullptr : dcolor, dfrozen,
                            nscan, TL, dprop,
                            0xA5A5ULL + (uint64_t)inner * 17 + (uint64_t)outer,
                            0xC0FFEEULL + (uint64_t)layer * 10007 + (uint64_t)outer,
                            zprof ? ddbg : nullptr, dnabove);
                    }
                    }
                    ev_propose.stop();
                    ev_pack.start();
                    {
                    NvRange nv("pack");
                    if (lv_bluelist) {
                        k_pack_listed_fused<<<bn < 256 ? bn : 256, threads>>>(
                            dblist, dnlist, dprop, dsz, pkey_in, ppay_in,
                            dnprop);
                    } else {
                        k_pack_prop_fused<<<bn, threads>>>(
                            dprop, dsz, nnode, pkey_in, ppay_in, dnprop);
                    }
                    }
                    ev_pack.stop();
                    int hnp[2] = {0, 0};
                    ev_d2h.start();
                    cudaMemcpy(hnp, dnprop, 8, cudaMemcpyDeviceToHost);
                    ev_d2h.stop();
                    int nprop = hnp[0];
                    const int nabove = hnp[1];
                    if (aa) {
                        aa->sum_nlive += nlive;
                        aa->sum_above += nabove;
                        if (nabove == 0 && layer_first_zero < 0)
                            layer_first_zero = layer_outers;
                    }
                    // No edge is above TL between two distinct roots, so no
                    // colouring can propose one and the layer is finished. The
                    // outer loop had no such test and ran its full cap on every
                    // layer, which measured 1088 rounds where 653 do the work.
                    //
                    // Leaving here is a no-op for the result, not an
                    // approximation: the rounds it skips would re-run
                    // compress/rebuild_sz (idempotent functions of parent),
                    // re-colour, and propose nothing, so parent, sz and nlive
                    // are already at the values the next layer would have read.
                    if (nabove == 0) layer_done = true;
                    ++ninner;
                    int ngc = 0, nact = 0, ndirty = 0, nstar = 0;
                    if (zprof) {
                        int hdbg[5] = {0};
                        cudaMemcpy(hdbg, ddbg, 20, cudaMemcpyDeviceToHost);
                        ngc = hdbg[2];
                        cudaMemset(dnact, 0, 4);
                        k_count_roots<<<bn, threads>>>(dparent, nnode, dnact);
                        cudaMemcpy(&nact, dnact, 4, cudaMemcpyDeviceToHost);
                    }
                    if (nprop <= 0) {
                        if (zprof && zprof->hist_n < zprof->hist_cap) {
                            int hi = zprof->hist_n++;
                            if (zprof->hist_nlive) zprof->hist_nlive[hi] = nlive;
                            if (zprof->hist_nprop) zprof->hist_nprop[hi] = 0;
                            if (zprof->hist_nmerge) zprof->hist_nmerge[hi] = 0;
                            if (zprof->hist_ngc) zprof->hist_ngc[hi] = ngc;
                            if (zprof->hist_nact) zprof->hist_nact[hi] = nact;
                            if (zprof->hist_ndirty) zprof->hist_ndirty[hi] = 0;
                            if (zprof->hist_nstar) zprof->hist_nstar[hi] = 0;
                        }
                        break;
                    }
                    int bp = (nprop + threads - 1) / threads;
                    if (bp < 1) bp = 1;
                    ev_sort.start();
                    {
                    NvRange nv("sort_prop");
                    if (nprop <= PSORT_BLOCK_CAP) {
                        k_sort_prop_block<PSORT_BLOCK_T, PSORT_ITEMS>
                            <<<1, PSORT_BLOCK_T>>>(
                                pkey_in, ppay_in, pkey_out, ppay_out, nprop);
                    } else {
                        cub::DeviceRadixSort::SortPairs(
                            psort_tmp, psort_bytes, pkey_in, pkey_out,
                            ppay_in, ppay_out, nprop);
                    }
                    k_unpack_prop<<<bp, threads>>>(
                        pkey_out, ppay_out, dreds, dblues, dadd, nprop);
                    }
                    ev_sort.stop();
                    ev_accept.start();
                    {
                    NvRange nv("accept");
                    k_accept_reds<<<bp, threads>>>(
                        dreds, dblues, dadd, nprop, dparent, dsz, dfrozen, eps, dnmerge);
                    }
                    ev_accept.stop();
                    ev_compress.start();
                    if (lv_rootlist)
                        k_compress_list<<<bp, threads>>>(dparent, dblues, nprop);
                    else
                        k_compress<<<bn, threads>>>(dparent, nnode);
                    ev_compress.stop();
                    ev_freeze.start();
                    if (lv_freeze_reds) {
                        // G1. k_freeze sweeps all nnode nodes to find the reds;
                        // the reds are already listed, sorted, in dreds.
                        //
                        // Equivalent, not approximately so:
                        // - k_freeze acts only on color[i]==1, and k_color
                        //    colours only roots, so its domain is the red roots.
                        // - reds and blues are disjoint by colour and only
                        //    blues are reparented, so a red stays a root for the
                        //    whole outer and find(i)==i. That makes k_freeze's
                        //    dfind_nocomp and its sz[r]/frozen[i] indexing agree
                        //    with k_freeze_reds' direct use of reds[i].
                        // - a red absent from dreds received no proposal, so
                        //    sz[r] still equals sz0[r] and the strict > test
                        //    cannot fire. Skipping it changes nothing.
                        // - k_freeze's `sz0[i] ? sz0[i] : 1.0` guard, which
                        //    k_freeze_reds lacks, is unreachable here: every
                        //    root is its own member so k_rebuild_sz leaves
                        //    sz0[root] >= 1.
                        k_freeze_reds<<<bp, threads>>>(
                            dreds, dnprop, dsz, dsz0, dfrozen, eps);
                    } else {
                        k_freeze<<<bn, threads>>>(dparent, dsz, dsz0, dcolor, dfrozen, nnode, eps);
                    }
                    ev_freeze.stop();
                    int hm = 0;
                    ev_d2h.start();
                    cudaMemcpy(&hm, dnmerge, 4, cudaMemcpyDeviceToHost);
                    ev_d2h.stop();
                    for (int q = 0; q < nprobe; ++q) {
                        int probe = 0;
                        k_probe_touch<<<1, 32>>>(dnmerge);
                        cudaMemcpy(&probe, dnmerge, 4, cudaMemcpyDeviceToHost);
                    }
                    if (zprof && hm > 0) {
                        cudaMemset(ddirty_blue, 0, (size_t)nnode);
                        cudaMemset(ddirty_star, 0, (size_t)nnode);
                        cudaMemset(dndirty, 0, 4);
                        cudaMemset(dnstar, 0, 4);
                        k_mark_acc_blue<<<bp, threads>>>(
                            dblues, dreds, nprop, dparent, ddirty_blue, ddirty_star);
                        k_count_dirty_edges<<<be, threads>>>(
                            du, dv, dct, nlive, ddirty_blue, ddirty_star, dndirty, dnstar);
                        cudaMemcpy(&ndirty, dndirty, 4, cudaMemcpyDeviceToHost);
                        cudaMemcpy(&nstar, dnstar, 4, cudaMemcpyDeviceToHost);
                    }
                    if (zprof && zprof->hist_n < zprof->hist_cap) {
                        int hi = zprof->hist_n++;
                        if (zprof->hist_nlive) zprof->hist_nlive[hi] = nlive;
                        if (zprof->hist_nprop) zprof->hist_nprop[hi] = nprop;
                        if (zprof->hist_nmerge) zprof->hist_nmerge[hi] = hm;
                        if (zprof->hist_ngc) zprof->hist_ngc[hi] = ngc;
                        if (zprof->hist_nact) zprof->hist_nact[hi] = nact;
                        if (zprof->hist_ndirty) zprof->hist_ndirty[hi] = ndirty;
                        if (zprof->hist_nstar) zprof->hist_nstar[hi] = nstar;
                    }
                    nmerge += hm;
                    layer_merges += hm;
                    if (lv_rootlist && hm > 0) {
                        cudaMemset(dnroot, 0, 4);
                        int brc = (nroot + threads - 1) / threads;
                        if (brc < 1) brc = 1;
                        k_compact_roots<<<brc, threads>>>(
                            droots, drootflag, dparent, nroot, dnroot);
                        std::swap(droots, drootflag);
                        cudaMemcpy(&nroot, dnroot, 4,
                                   cudaMemcpyDeviceToHost);
                    }
                    if (hm == 0) break;
                    ev_compact.start();
                    if (lv_dirty) {
                        if (!dirty_unmark() || inner == 0)
                            cudaMemset(ddirty, 0, (size_t)nnode);
                        k_mark_acc_blue<<<bp, threads>>>(
                            dblues, dreds, nprop, dparent, ddirty, ddirty);
                        int n_kept = 0;
                        int emit_m = 0;
                        const int nact_old = nact_l;
                        hash_combine_dirty(
                            du, dv, dsm, dct, dkeep, dparent, ddirty, nscan,
                            nlive, &n_kept, be, threads, tu, tv, tsm, tct,
                            dholes, dnhole, dtab, ntab, dnout, dovf,
                            dslots, dnslot, damask, &emit_m,
                            aa ? &aa->rewrite_scan_ms : nullptr,
                            aa ? &aa->hash_ms : nullptr,
                            inc.on ? &inc : nullptr);
                        nlive = n_kept;
                        if (compact_k > 0 && ((inner + 1) % compact_k) == 0) {
                            compact_radix(
                                du, dv, dsm, dct, dkeep, dparent, nnode, nscan,
                                &nlive, be, threads, tu, tv, tsm, tct,
                                dkey, dkeyo, csr, 0.0, false);
                            nscan = nlive;
                            if (inc.on)
                                csr_rebuild(inc, du, dv, dct, (int)nscan, threads);
                        } else if (inc.on && hm > 0) {
                            k_csr_splice<<<bp, threads>>>(
                                dblues, dreds, nprop, dparent, inc.head, inc.nxt);
                        }
                        if (lv_bluelist) {
                            cudaMemset(dnact_l, 0, 4);
                            if (listed_rebuild() && dalist2) {
                                if (nact_old > 0) {
                                    cudaMemcpy(
                                        dalist2, dalist,
                                        (size_t)nact_old * 4,
                                        cudaMemcpyDeviceToDevice);
                                    int ba = (nact_old + threads - 1) / threads;
                                    k_keep_zero_listed<<<ba, threads>>>(
                                        dalist2, nact_old, dkeep);
                                }
                                if (emit_m > 0) {
                                    int bh = (emit_m + threads - 1) / threads;
                                    k_keep_zero_listed<<<bh, threads>>>(
                                        dholes, emit_m, dkeep);
                                    k_retest_holes<<<bh, threads>>>(
                                        dholes, emit_m, dsm, dct, TL, damask);
                                }
                                if (nact_old > 0) {
                                    int ba = (nact_old + threads - 1) / threads;
                                    k_pack_listed<<<ba, threads>>>(
                                        dalist2, nact_old, damask, dct, dkeep,
                                        dalist, dnact_l);
                                }
                                if (emit_m > 0) {
                                    int bh = (emit_m + threads - 1) / threads;
                                    k_pack_listed_new<<<bh, threads>>>(
                                        dholes, emit_m, damask, dct, dkeep,
                                        dalist, dnact_l);
                                }
                            } else if (fuse_pack()) {
                                k_rebuild_pack<<<be, threads>>>(
                                    du, dv, dsm, dct, dkeep, ddirty, nscan, TL,
                                    dalist, dnact_l, damask);
                            } else {
                                k_rebuild_active<<<be, threads>>>(
                                    du, dv, dsm, dct, dkeep, ddirty, nscan, TL,
                                    dalist, dnact_l, damask);
                                k_pack_amask<<<be, threads>>>(
                                    damask, dct, nscan, dalist, dnact_l);
                            }
                            cudaMemcpy(&nact_l, dnact_l, 4,
                                       cudaMemcpyDeviceToHost);
                        }
                        if (dirty_unmark()) {
                            k_unmark_acc_blue<<<bp, threads>>>(
                                dblues, dreds, nprop, dparent, ddirty, ddirty);
                        }
                        n21_inner_push((int)nscan, nlive, nact_l);
                    } else if (use_hash) {
                        hash_combine_live(du, dv, dsm, dct, dkeep, dparent, nnode,
                            nlive, &nlive, be, threads, tu, tv, tsm, tct,
                            dtab, ntab, dnout, dovf, false);
                        nscan = nlive;
                        if (lv_bluelist) {
                            be = (int)((nscan + threads - 1) / threads);
                            if (be < 1) be = 1;
                            k_init_amask<<<be, threads>>>(
                                dsm, dct, nscan, TL, damask);
                            cudaMemset(dnact_l, 0, 4);
                            k_pack_amask<<<be, threads>>>(
                                damask, dct, nscan, dalist, dnact_l);
                            cudaMemcpy(&nact_l, dnact_l, 4,
                                       cudaMemcpyDeviceToHost);
                        }
                    } else {
                        compact_radix(du, dv, dsm, dct, dkeep, dparent, nnode, nlive, &nlive,
                            be, threads, tu, tv, tsm, tct, dkey, dkeyo, csr, 0.0, false);
                        nscan = nlive;
                    }
                    ev_compact.stop();
                    be = (int)((nscan + threads - 1) / threads);
                    if (be < 1) be = 1;
                    if (nlive <= 0) break;
                }
                ++nouter;
                ++layer_outers;
                if (nlive <= 0) break;
                if (layer_done) break;
            }
            if (aa && aa->n_layer > 0 && aa->n_layer <= 64) {
                int li = aa->n_layer - 1;
                aa->layer_outers[li] = layer_outers;
                aa->layer_merges[li] = layer_merges;
                aa->layer_first_zero[li] = layer_first_zero;
            }
            if (layer_merges == 0) break;
        }
        cudaMemcpy(hparent.data(), dparent, (size_t)nnode * 4, cudaMemcpyDeviceToHost);
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (int i = 0; i < nnode; ++i) {
            uint32_t x = (uint32_t)i;
            int guard = 0;
            while (hparent[x] != x && guard++ < 64) x = hparent[x];
            dst[i] = x;
        }
        if (stats_out) {
            stats_out[ti * 3 + 0] = nouter;
            stats_out[ti * 3 + 1] = nmerge;
            stats_out[ti * 3 + 2] = ninner;
        }
        std::fprintf(stderr,
            "E6s T=%.2f eps=%.4f outer=%lld inner=%lld merges=%lld nlive=%lld\n",
            T, eps, (long long)nouter, (long long)ninner, (long long)nmerge,
            (long long)nlive);
    }
    cudaFree(dparent); cudaFree(dsz); cudaFree(dsz0); cudaFree(dcolor);
    {
        int hovf = 0;
        cudaMemcpy(&hovf, dovf, 4, cudaMemcpyDeviceToHost);
        if (hovf & HOVF_PROBE)
            std::fprintf(stderr, "E6s_HASH_OVERFLOW probe limit hit\n");
        if (hovf & HOVF_WIDTH)
            std::fprintf(stderr, "E6s_HASH_OVERFLOW uint32 sm/ct exceeded; "
                                 "weights are wrong, widen HSlot\n");
    }
    cudaFree(dovf);
    cudaFree(dblist); cudaFree(dnlist);
    cudaFree(ddirty); cudaFree(dholes); cudaFree(dnhole);
    cudaFree(inc.head); cudaFree(inc.nxt); cudaFree(inc.gen);
    cudaFree(inc.elist); cudaFree(inc.nodes);
    cudaFree(inc.nnodes); cudaFree(inc.nlist);
    cudaFree(damask); cudaFree(dalist); cudaFree(dalist2); cudaFree(dnact_l);
    cudaFree(droots); cudaFree(drootflag); cudaFree(dnroot);
    cudaFree(dfrozen); cudaFree(dprop);
    cudaFree(pkey_in); cudaFree(pkey_out);
    cudaFree(ppay_in); cudaFree(ppay_out); cudaFree(psort_tmp);
    compact_scratch_free(csr);
    cudaFree(dreds); cudaFree(dblues); cudaFree(dadd);
    cudaFree(dnmerge); cudaFree(dnprop); cudaFree(dkeep);
    // The allocations, not the current values of the names; see alloc_t* above.
    cudaFree(alloc_tu); cudaFree(alloc_tv);
    cudaFree(alloc_tsm); cudaFree(alloc_tct); cudaFree(dblk);
    cudaFree(dtab); cudaFree(dnout); cudaFree(dslots); cudaFree(dnslot);
    cudaFree(dkey); cudaFree(dkeyo);
    if (zprof) {
        cudaFree(ddbg); cudaFree(dnact); cudaFree(dndirty); cudaFree(dnstar);
        cudaFree(ddirty_blue); cudaFree(ddirty_star);
    }
    return 1;
}

static void e6t_rebuild(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct, int64_t nlive,
    uint32_t* dparent, int nnode, int threads,
    int* ddeg, int* dcur, int* doff, int* dlen, int* dadj,
    int* ovf_head, int* dovf_used,
    EHash* dtab, int ntab, int* dfail,
    uint8_t* dkeep, int* dgc, int* dngc, double TL)
{
    int bn = (nnode + threads - 1) / threads;
    int be = (int)((nlive + threads - 1) / threads);
    if (be < 1) be = 1;
    int tb = (ntab + threads - 1) / threads;
    cudaMemset(ddeg, 0, (size_t)nnode * 4);
    cudaMemset(dcur, 0, (size_t)nnode * 4);
    cudaMemset(dfail, 0, 4);
    k_ehash_clear<<<tb, threads>>>(dtab, ntab);
    k_ehash_build<<<be, threads>>>(du, dv, dct, nlive, dtab, ntab, dfail);
    k_deg<<<be, threads>>>(du, dv, dct, nlive, ddeg);
    cudaMemcpy(dlen, ddeg, (size_t)nnode * 4, cudaMemcpyDeviceToDevice);
    thrust::exclusive_scan(
        thrust::device_ptr<int>(ddeg),
        thrust::device_ptr<int>(ddeg) + nnode,
        thrust::device_ptr<int>(doff));
    k_scatter_adj<<<be, threads>>>(du, dv, dct, nlive, doff, dcur, dadj);
    k_ovf_reset<<<bn, threads>>>(ovf_head, nnode);
    cudaMemset(dovf_used, 0, 4);
    k_gc_mark<<<be, threads>>>(du, dv, dsm, dct, dparent, nlive, TL, dkeep);
    cudaMemset(dngc, 0, 4);
    k_iota_if<<<be, threads>>>(dkeep, nlive, dgc, dngc);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "E6t_REBUILD cuda=%s nlive=%lld ntab=%d\n",
            cudaGetErrorString(err), (long long)nlive, ntab);
    }
}

static int env_max_outer(int defv)
{
    const char* s = std::getenv("WATERZ_MAX_OUTER");
    if (!s || !*s) return defv;
    int v = std::atoi(s);
    return v > 0 ? v : defv;
}

// E6u: capture once per TL/layer; launch per outer.
static cudaGraph_t e6u_graph = nullptr;
static cudaGraphExec_t e6u_exec = nullptr;
static double e6u_tl = -1.0;

static void e6u_reset()
{
    if (e6u_exec) cudaGraphExecDestroy(e6u_exec);
    if (e6u_graph) cudaGraphDestroy(e6u_graph);
    e6u_exec = nullptr;
    e6u_graph = nullptr;
    e6u_tl = -1.0;
}

static int e6u_run_outer(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct,
    uint32_t* dparent, uint32_t* dsz, uint32_t* dsz0, uint8_t* dfrozen,
    unsigned long long* dprop, uint32_t* dlist, int* dnlist,
    uint32_t* dreds, uint32_t* dblues, uint32_t* dadd, float* dpris,
    int* dnprop, int* dnmerge, uint64_t* dseed, uint64_t* dcseed, int* dinner,
    int* dgc, int* dngc, int* dgc2, int* dngc2,
    int* dtouched, int* dntouch, int* dfail, int* dnkill,
    int* doff, int* dlen, int* dadj, int* ovf_head, int* ovf_eid, int* ovf_nxt,
    int* dovf_used, int ovf_cap, EHash* dtab, int ntab,
    unsigned long long* dk0, unsigned long long* dk1, int* di0, int* di1, int* dhist,
    uint32_t* dr1, uint32_t* db1, uint32_t* da1, float* dp1,
    int* dninner_tot, int* dnmerge_tot,
    int nnode, int n_edges, int threads, double TL, double eps, int max_inner,
    int* ninner_add, int* nmerge_add, int* hfail)
{
    int bn = (nnode + threads - 1) / threads;
    int be_max = (int)((n_edges + threads - 1) / threads);
    if (be_max < 1) be_max = 1;
    cudaGraph_t graph = e6u_graph;
    cudaGraphExec_t exec = e6u_exec;
    cudaError_t err = cudaSuccess;
    if (!(exec && graph && e6u_tl == TL)) {
    e6u_reset();
    graph = nullptr;
    exec = nullptr;
    cudaGraphNode_t condNode;
    err = cudaGraphCreate(&graph, 0);
    if (err != cudaSuccess) return 0;
    cudaGraphConditionalHandle handle;
    err = cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        return 0;
    }
    cudaGraphNodeParams cParams{};
    cParams.type = cudaGraphNodeTypeConditional;
    cParams.conditional.handle = handle;
    cParams.conditional.type = cudaGraphCondTypeWhile;
    cParams.conditional.size = 1;
    err = cudaGraphAddNode(&condNode, graph, NULL, 0, &cParams);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        return 0;
    }
    cudaGraph_t body = cParams.conditional.phGraph_out[0];
    cudaStream_t cs;
    cudaStreamCreate(&cs);
    err = cudaStreamBeginCaptureToGraph(cs, body, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        cudaStreamDestroy(cs);
        cudaGraphDestroy(graph);
        return 0;
    }
    k_zero_listed_prop_d<<<bn, threads, 0, cs>>>(dlist, dnlist, dprop);
    cudaMemsetAsync(dnmerge, 0, 4, cs);
    cudaMemsetAsync(dnprop, 0, 4, cs);
    cudaMemsetAsync(dnlist, 0, 4, cs);
    cudaMemsetAsync(dntouch, 0, 4, cs);
    cudaMemsetAsync(dnkill, 0, 4, cs);
    cudaMemsetAsync(dfail, 0, 4, cs);
    k_propose_eid_list<<<be_max, threads, 0, cs>>>(
        dgc, dngc, du, dv, dsm, dct, dparent, dsz, dfrozen,
        TL, dprop, dlist, dnlist, dseed, dcseed, dinner);
    k_pack_listed<<<bn, threads, 0, cs>>>(
        dlist, dnlist, dprop, dsz, dreds, dblues, dadd, dpris, dnprop);
    sort_proposals_dev(
        dreds, dblues, dadd, dpris, dnprop,
        dk0, dk1, di0, di1, dhist, dr1, db1, da1, dp1,
        nnode, threads, cs);
    k_accept_reds_d<<<bn, threads, 0, cs>>>(
        dr1, db1, da1, dnprop, dparent, dsz, dfrozen, eps, dnmerge);
    k_compress<<<bn, threads, 0, cs>>>(dparent, nnode);
    k_freeze_reds<<<bn, threads, 0, cs>>>(dr1, dnprop, dsz, dsz0, dfrozen, eps);
    k_starmarge_blue<<<bn, threads, 0, cs>>>(
        db1, dnprop, dparent, du, dv, dsm, dct,
        doff, dlen, dadj, ovf_head, ovf_eid, ovf_nxt, dovf_used, ovf_cap,
        dtab, ntab, dtouched, dntouch, dfail, dnkill);
    cudaMemsetAsync(dngc2, 0, 4, cs);
    k_filter_gc<<<be_max, threads, 0, cs>>>(
        dgc, dngc, du, dv, dsm, dct, dparent, TL, dgc2, dngc2);
    k_add_touched_gc<<<be_max, threads, 0, cs>>>(
        dtouched, dntouch, du, dv, dsm, dct, dparent, TL, dgc2, dngc2);
    k_copy_int_n<<<be_max, threads, 0, cs>>>(dgc2, dgc, dngc2);
    k_copy_int1<<<1, 1, 0, cs>>>(dngc2, dngc);
    k_e6u_tick<<<1, 1, 0, cs>>>(
        handle, dnprop, dnmerge, dinner, max_inner, dninner_tot, dnmerge_tot, dfail);
    cudaGraph_t unused = nullptr;
    err = cudaStreamEndCapture(cs, &unused);
    cudaStreamDestroy(cs);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        return 0;
    }
    err = cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        return 0;
    }
    e6u_graph = graph;
    e6u_exec = exec;
    e6u_tl = TL;
    } // reuse cached graph when TL matches
    cudaMemset(dinner, 0, 4);
    cudaMemset(dninner_tot, 0, 4);
    cudaMemset(dnmerge_tot, 0, 4);
    cudaMemset(dfail, 0, 4);
    err = cudaGraphLaunch(e6u_exec, 0);
    cudaDeviceSynchronize();
    int ni = 0, nm = 0, hf = 0;
    cudaMemcpy(&ni, dninner_tot, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&nm, dnmerge_tot, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&hf, dfail, 4, cudaMemcpyDeviceToHost);
    if (ninner_add) *ninner_add = ni;
    if (nmerge_add) *nmerge_add = nm;
    if (hfail) *hfail = hf;
    return err == cudaSuccess ? 1 : 0;
}

static int parhac_e6t_dev(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct, int64_t n_edges,
    const double* aff_thr, int n_thr, double eps,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out)
{
    set_size_asym_from_env();
    const int nnode = (int)max_id + 1;
    uint32_t *dparent, *dsz, *dsz0, *dreds, *dblues, *dadd;
    unsigned long long *dprop;
    uint32_t *tu, *tv;
    double *dblk, *tsm;
    int64_t *tct;
    uint8_t *dkeep, *dcolor, *dfrozen;
    float *dpris;
    int *dnmerge, *dnprop;
    unsigned long long *dkey, *dkeyo;
    cudaMalloc(&dparent, (size_t)nnode * 4);
    cudaMalloc(&dsz, (size_t)nnode * 4);
    cudaMalloc(&dsz0, (size_t)nnode * 4);
    cudaMalloc(&dcolor, (size_t)nnode);
    cudaMalloc(&dfrozen, (size_t)nnode);
    cudaMalloc(&dprop, (size_t)nnode * 8);
    cudaMalloc(&dreds, (size_t)nnode * 4);
    cudaMalloc(&dblues, (size_t)nnode * 4);
    cudaMalloc(&dadd, (size_t)nnode * 4);
    cudaMalloc(&dpris, (size_t)nnode * 4);
    cudaMalloc(&dnmerge, 4);
    cudaMalloc(&dnprop, 4);
    cudaMalloc(&dkeep, (size_t)n_edges);
    cudaMalloc(&tu, (size_t)n_edges * 4);
    cudaMalloc(&tv, (size_t)n_edges * 4);
    cudaMalloc(&tsm, (size_t)n_edges * 8);
    cudaMalloc(&tct, (size_t)n_edges * 8);
    cudaMalloc(&dkey, (size_t)n_edges * 8);
    cudaMalloc(&dkeyo, (size_t)n_edges * 8);
    int nblk = 256;
    cudaMalloc(&dblk, (size_t)nblk * 8);
    CompactScratch csr;
    compact_scratch_init(csr, n_edges);

    int ntab = next_pow2((int)(n_edges * 4 + 1024));
    if (ntab < 2048) ntab = 2048;
    EHash* dtab = nullptr;
    int *ddeg, *dcur, *doff, *dlen, *dadj, *ovf_head, *ovf_eid, *ovf_nxt;
    int *dovf_used, *dfail, *dgc, *dgc2, *dngc, *dntouch, *dtouched, *dnkill;
    int adj_cap = (int)(n_edges * 2 + 16);
    int ovf_cap = (int)(n_edges * 4 + 16);
    if (cudaMalloc(&dtab, (size_t)ntab * sizeof(EHash)) != cudaSuccess) {
        std::fprintf(stderr, "E6t_OOM hash ntab=%d\n", ntab);
        compact_scratch_free(csr);
        return 0;
    }
    cudaMalloc(&ddeg, (size_t)nnode * 4);
    cudaMalloc(&dcur, (size_t)nnode * 4);
    cudaMalloc(&doff, (size_t)nnode * 4);
    cudaMalloc(&dlen, (size_t)nnode * 4);
    cudaMalloc(&dadj, (size_t)adj_cap * 4);
    cudaMalloc(&ovf_head, (size_t)nnode * 4);
    cudaMalloc(&ovf_eid, (size_t)ovf_cap * 4);
    cudaMalloc(&ovf_nxt, (size_t)ovf_cap * 4);
    cudaMalloc(&dovf_used, 4);
    cudaMalloc(&dfail, 4);
    cudaMalloc(&dgc, (size_t)n_edges * 4);
    cudaMalloc(&dgc2, (size_t)n_edges * 4);
    cudaMalloc(&dngc, 4);
    cudaMalloc(&dntouch, 4);
    cudaMalloc(&dtouched, (size_t)n_edges * 4);
    cudaMalloc(&dnkill, 4);
    uint32_t* dlist = nullptr;
    int* dnlist = nullptr;
    cudaMalloc(&dlist, (size_t)nnode * 4);
    cudaMalloc(&dnlist, 4);
    uint32_t *dr1 = nullptr, *db1 = nullptr, *da1 = nullptr;
    float* dp1 = nullptr;
    unsigned long long *dk0 = nullptr, *dk1 = nullptr;
    int *di0 = nullptr, *di1 = nullptr, *dhist = nullptr;
    uint64_t *dseed = nullptr, *dcseed = nullptr;
    int *dinner = nullptr, *dninner_tot = nullptr, *dnmerge_tot = nullptr;
    int* dngc2 = nullptr;
    cudaMalloc(&dr1, (size_t)nnode * 4);
    cudaMalloc(&db1, (size_t)nnode * 4);
    cudaMalloc(&da1, (size_t)nnode * 4);
    cudaMalloc(&dp1, (size_t)nnode * 4);
    cudaMalloc(&dk0, (size_t)nnode * 8);
    cudaMalloc(&dk1, (size_t)nnode * 8);
    cudaMalloc(&di0, (size_t)nnode * 4);
    cudaMalloc(&di1, (size_t)nnode * 4);
    cudaMalloc(&dhist, (size_t)256 * 256 * 4);
    cudaMalloc(&dseed, 8);
    cudaMalloc(&dcseed, 8);
    cudaMalloc(&dinner, 4);
    cudaMalloc(&dninner_tot, 4);
    cudaMalloc(&dnmerge_tot, 4);
    cudaMalloc(&dngc2, 4);
    int max_outer = env_max_outer(64);
    int use_graph = !std::getenv("WATERZ_NO_GRAPH");

    int threads = 256;
    int bn = (nnode + threads - 1) / threads;
    k_init_parent<<<bn, threads>>>(dparent, dsz, nnode);
    cudaDeviceSynchronize();

    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });
    std::vector<double> hblk((size_t)nblk);
    std::vector<uint32_t> hparent((size_t)nnode);
    int64_t nlive = n_edges;
    {
        int be0 = (int)((n_edges + threads - 1) / threads);
        if (be0 < 1) be0 = 1;
        k_scale_sm_bytes<<<be0, threads>>>(dsm, n_edges);
    }

    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti] * SM_BYTE_SCALE;
        int64_t nmerge = 0, ninner = 0, nouter = 0;
        for (int layer = 0; layer < 10000; ++layer) {
            int be = (int)((nlive + threads - 1) / threads);
            if (be < 1) be = 1;
            k_compress<<<bn, threads>>>(dparent, nnode);
            k_wmax_live<<<nblk, 256>>>(du, dv, dsm, dct, dparent, nlive, dblk);
            cudaMemcpy(hblk.data(), dblk, (size_t)nblk * 8, cudaMemcpyDeviceToHost);
            double wmax = 0;
            for (double x : hblk) if (x > wmax) wmax = x;
            if (wmax <= T) break;
            double TL = wmax / (1.0 + eps);
            if (TL < T) TL = T;
            compact_radix(du, dv, dsm, dct, dkeep, dparent, nnode, nlive, &nlive,
                be, threads, tu, tv, tsm, tct, dkey, dkeyo, csr);
            if (nlive <= 0) break;
            be = (int)((nlive + threads - 1) / threads);
            if (be < 1) be = 1;
            e6t_rebuild(du, dv, dsm, dct, nlive, dparent, nnode, threads,
                ddeg, dcur, doff, dlen, dadj, ovf_head, dovf_used,
                dtab, ntab, dfail, dkeep, dgc, dngc, TL);
            cudaMemset(dprop, 0, (size_t)nnode * 8);
            cudaMemset(dnlist, 0, 4);
            int hfail = 0;
            cudaMemcpy(&hfail, dfail, 4, cudaMemcpyDeviceToHost);
            if (hfail) {
                std::fprintf(stderr, "E6t_HASHFAIL layer rebuild T=%.2f nlive=%lld ntab=%d\n",
                    T, (long long)nlive, ntab);
                break;
            }
            int layer_merges = 0;
            int nkill_acc = 0;
            int be_max = (int)((n_edges + threads - 1) / threads);
            if (be_max < 1) be_max = 1;
            for (int outer = 0; outer < max_outer; ++outer) {
                k_compress<<<bn, threads>>>(dparent, nnode);
                k_zero_sz<<<bn, threads>>>(dsz, nnode);
                k_rebuild_sz<<<bn, threads>>>(dparent, dsz, nnode);
                cudaMemset(dfrozen, 0, (size_t)nnode);
                k_copy_sz<<<bn, threads>>>(dsz, dsz0, nnode);
                uint64_t hseed = 0xA5A5ULL + (uint64_t)outer;
                uint64_t hcseed = 0xC0FFEEULL + (uint64_t)layer * 10007ull + (uint64_t)outer;
                cudaMemcpy(dseed, &hseed, 8, cudaMemcpyHostToDevice);
                cudaMemcpy(dcseed, &hcseed, 8, cudaMemcpyHostToDevice);
                cudaMemset(dinner, 0, 4);
                if (use_graph) {
                    int ni = 0, nm = 0, hf = 0;
                    int gok = e6u_run_outer(
                        du, dv, dsm, dct, dparent, dsz, dsz0, dfrozen,
                        dprop, dlist, dnlist, dreds, dblues, dadd, dpris,
                        dnprop, dnmerge, dseed, dcseed, dinner,
                        dgc, dngc, dgc2, dngc2, dtouched, dntouch, dfail, dnkill,
                        doff, dlen, dadj, ovf_head, ovf_eid, ovf_nxt,
                        dovf_used, ovf_cap, dtab, ntab,
                        dk0, dk1, di0, di1, dhist, dr1, db1, da1, dp1,
                        dninner_tot, dnmerge_tot,
                        nnode, (int)n_edges, threads, TL, eps, 64,
                        &ni, &nm, &hf);
                    if (gok) {
                        ninner += ni;
                        nmerge += nm;
                        layer_merges += nm;
                        nkill_acc += 0;
                        if (hf || (nkill_acc * 2 > (int)nlive)) {
                            compact_radix(du, dv, dsm, dct, dkeep, dparent, nnode, nlive, &nlive,
                                be, threads, tu, tv, tsm, tct, dkey, dkeyo, csr, 0.0);
                            be = (int)((nlive + threads - 1) / threads);
                            if (be < 1) be = 1;
                            e6t_rebuild(du, dv, dsm, dct, nlive, dparent, nnode, threads,
                                ddeg, dcur, doff, dlen, dadj, ovf_head, dovf_used,
                                dtab, ntab, dfail, dkeep, dgc, dngc, TL);
                            nkill_acc = 0;
                        }
                        ++nouter;
                        if (nlive <= 0) break;
                        continue;
                    }
                    use_graph = 0;
                }
                for (int inner = 0; inner < 64; ++inner) {
                    k_zero_listed_prop_d<<<bn, threads>>>(dlist, dnlist, dprop);
                    cudaMemset(dnmerge, 0, 4);
                    cudaMemset(dnprop, 0, 4);
                    cudaMemset(dnlist, 0, 4);
                    k_propose_eid_list<<<be_max, threads>>>(
                        dgc, dngc, du, dv, dsm, dct, dparent, dsz, dfrozen,
                        TL, dprop, dlist, dnlist, dseed, dcseed, dinner);
                    k_pack_listed<<<bn, threads>>>(
                        dlist, dnlist, dprop, dsz, dreds, dblues, dadd, dpris, dnprop);
                    sort_proposals_dev(
                        dreds, dblues, dadd, dpris, dnprop,
                        dk0, dk1, di0, di1, dhist, dr1, db1, da1, dp1,
                        nnode, threads, 0);
                    k_accept_reds_d<<<bn, threads>>>(
                        dr1, db1, da1, dnprop, dparent, dsz, dfrozen, eps, dnmerge);
                    k_compress<<<bn, threads>>>(dparent, nnode);
                    k_freeze_reds<<<bn, threads>>>(dr1, dnprop, dsz, dsz0, dfrozen, eps);
                    int nprop = 0, hm = 0;
                    cudaMemcpy(&nprop, dnprop, 4, cudaMemcpyDeviceToHost);
                    ++ninner;
                    if (nprop <= 0) break;
                    cudaMemcpy(&hm, dnmerge, 4, cudaMemcpyDeviceToHost);
                    nmerge += hm;
                    layer_merges += hm;
                    if (hm == 0) break;
                    cudaMemset(dntouch, 0, 4);
                    cudaMemset(dnkill, 0, 4);
                    cudaMemset(dfail, 0, 4);
                    k_starmarge_blue<<<bn, threads>>>(
                        db1, dnprop, dparent, du, dv, dsm, dct,
                        doff, dlen, dadj, ovf_head, ovf_eid, ovf_nxt, dovf_used, ovf_cap,
                        dtab, ntab, dtouched, dntouch, dfail, dnkill);
                    int nkill = 0;
                    cudaMemcpy(&hfail, dfail, 4, cudaMemcpyDeviceToHost);
                    cudaMemcpy(&nkill, dnkill, 4, cudaMemcpyDeviceToHost);
                    nkill_acc += nkill;
                    int rebuild = hfail || (nkill_acc * 2 > (int)nlive);
                    if (rebuild) {
                        compact_radix(du, dv, dsm, dct, dkeep, dparent, nnode, nlive, &nlive,
                            be, threads, tu, tv, tsm, tct, dkey, dkeyo, csr, 0.0);
                        be = (int)((nlive + threads - 1) / threads);
                        if (be < 1) be = 1;
                        e6t_rebuild(du, dv, dsm, dct, nlive, dparent, nnode, threads,
                            ddeg, dcur, doff, dlen, dadj, ovf_head, dovf_used,
                            dtab, ntab, dfail, dkeep, dgc, dngc, TL);
                        nkill_acc = 0;
                    } else {
                        cudaMemset(dngc2, 0, 4);
                        k_filter_gc<<<be_max, threads>>>(
                            dgc, dngc, du, dv, dsm, dct, dparent, TL, dgc2, dngc2);
                        k_add_touched_gc<<<be_max, threads>>>(
                            dtouched, dntouch, du, dv, dsm, dct, dparent, TL, dgc2, dngc2);
                        k_copy_int_n<<<be_max, threads>>>(dgc2, dgc, dngc2);
                        k_copy_int1<<<1, 1>>>(dngc2, dngc);
                    }
                    int inext = inner + 1;
                    cudaMemcpy(dinner, &inext, 4, cudaMemcpyHostToDevice);
                    if (nlive <= 0) break;
                }
                ++nouter;
                if (nlive <= 0) break;
            }
                        if (layer_merges == 0) break;
        }
        cudaMemcpy(hparent.data(), dparent, (size_t)nnode * 4, cudaMemcpyDeviceToHost);
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (int i = 0; i < nnode; ++i) {
            uint32_t x = (uint32_t)i;
            int guard = 0;
            while (hparent[x] != x && guard++ < 64) x = hparent[x];
            dst[i] = x;
        }
        if (stats_out) {
            stats_out[ti * 3 + 0] = nouter;
            stats_out[ti * 3 + 1] = nmerge;
            stats_out[ti * 3 + 2] = ninner;
        }
        std::fprintf(stderr,
            "E6t T=%.2f eps=%.4f outer=%lld inner=%lld merges=%lld nlive=%lld\n",
            T, eps, (long long)nouter, (long long)ninner, (long long)nmerge,
            (long long)nlive);
    }
    cudaFree(dparent); cudaFree(dsz); cudaFree(dsz0); cudaFree(dcolor);
    cudaFree(dfrozen); cudaFree(dprop); cudaFree(dpris);
    cudaFree(dreds); cudaFree(dblues); cudaFree(dadd);
    cudaFree(dnmerge); cudaFree(dnprop); cudaFree(dkeep);
    cudaFree(tu); cudaFree(tv); cudaFree(tsm); cudaFree(tct); cudaFree(dblk);
    cudaFree(dkey); cudaFree(dkeyo);
    compact_scratch_free(csr);
    cudaFree(dtab); cudaFree(ddeg); cudaFree(dcur); cudaFree(doff); cudaFree(dlen);
    cudaFree(dadj); cudaFree(ovf_head); cudaFree(ovf_eid); cudaFree(ovf_nxt);
    cudaFree(dovf_used); cudaFree(dfail); cudaFree(dgc); cudaFree(dgc2);
    cudaFree(dngc); cudaFree(dntouch); cudaFree(dtouched); cudaFree(dnkill);
    cudaFree(dlist); cudaFree(dnlist);
    cudaFree(dr1); cudaFree(db1); cudaFree(da1); cudaFree(dp1);
    cudaFree(dk0); cudaFree(dk1); cudaFree(di0); cudaFree(di1); cudaFree(dhist);
    cudaFree(dseed); cudaFree(dcseed); cudaFree(dinner);
    cudaFree(dninner_tot); cudaFree(dnmerge_tot); cudaFree(dngc2);
    e6u_reset();
    return 1;
}

extern "C" int parhac_paper_d_timed(
    const uint32_t* u_h, const uint32_t* v_h,
    const double* sm_h, const int64_t* ct_h,
    int64_t n_edges, const double* aff_thr, int n_thr, double eps,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out, double* device_ms)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    uint32_t *du, *dv;
    double *dsm;
    int64_t *dct;
    cudaMalloc(&du, (size_t)n_edges * 4);
    cudaMalloc(&dv, (size_t)n_edges * 4);
    cudaMalloc(&dsm, (size_t)n_edges * 8);
    cudaMalloc(&dct, (size_t)n_edges * 8);
    cudaMemcpy(du, u_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsm, sm_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dct, ct_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    int rc = std::getenv("WATERZ_PAPER_E6T")
        ? parhac_e6t_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
            parent_out, max_id, stats_out)
        : parhac_e6s_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
            parent_out, max_id, stats_out);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (device_ms) *device_ms = (double)ms;
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(du); cudaFree(dv); cudaFree(dsm); cudaFree(dct);
    return rc;
}

extern "C" int parhac_e6s(
    const uint32_t* u_h, const uint32_t* v_h,
    const double* sm_h, const int64_t* ct_h,
    int64_t n_edges, const double* aff_thr, int n_thr, double eps,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    uint32_t *du, *dv;
    double *dsm;
    int64_t *dct;
    cudaMalloc(&du, (size_t)n_edges * 4);
    cudaMalloc(&dv, (size_t)n_edges * 4);
    cudaMalloc(&dsm, (size_t)n_edges * 8);
    cudaMalloc(&dct, (size_t)n_edges * 8);
    cudaMemcpy(du, u_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsm, sm_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dct, ct_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    int rc = parhac_e6s_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
        parent_out, max_id, stats_out);
    cudaFree(du); cudaFree(dv); cudaFree(dsm); cudaFree(dct);
    return rc;
}

extern "C" int parhac_paper_d(
    const uint32_t* u_h, const uint32_t* v_h,
    const double* sm_h, const int64_t* ct_h,
    int64_t n_edges, const double* aff_thr, int n_thr, double eps,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    uint32_t *du, *dv;
    double *dsm;
    int64_t *dct;
    cudaMalloc(&du, (size_t)n_edges * 4);
    cudaMalloc(&dv, (size_t)n_edges * 4);
    cudaMalloc(&dsm, (size_t)n_edges * 8);
    cudaMalloc(&dct, (size_t)n_edges * 8);
    cudaMemcpy(du, u_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsm, sm_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dct, ct_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    int rc = std::getenv("WATERZ_PAPER_E6T")
        ? parhac_e6t_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
            parent_out, max_id, stats_out)
        : parhac_e6s_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
            parent_out, max_id, stats_out);
    cudaFree(du); cudaFree(dv); cudaFree(dsm); cudaFree(dct);
    return rc;
}

// Consumes its input: the edge arrays are the working arrays, rewritten in
// place, and hold garbage on return.
//
// The previous version allocated a second full set and copied device-to-device
// into it, so two complete edge sets were live across the whole run. At
// 2.16 Gvox that is 2.06 GiB of duplicate held for the entire agglomeration,
// which is more than narrowing every payload to uint32 would save, plus 2 GiB
// of pointless copy. The only caller is segment_d, which frees these arrays as
// soon as this returns and never reads them again.
extern "C" int parhac_paper_d_dev(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct,
    int64_t n_edges, const double* aff_thr, int n_thr, double eps,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    return std::getenv("WATERZ_PAPER_E6T")
        ? parhac_e6t_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
            parent_out, max_id, stats_out)
        : parhac_e6s_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
            parent_out, max_id, stats_out);
}

extern "C" int parhac_e6s_profile(
    const uint32_t* u_h, const uint32_t* v_h,
    const double* sm_h, const int64_t* ct_h,
    int64_t n_edges, const double* aff_thr, int n_thr, double eps,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out,
    int64_t* hist_nlive, int64_t* hist_nprop, int64_t* hist_nmerge,
    int64_t* hist_ngc, int64_t* hist_nact, int64_t* hist_ndirty,
    int64_t* hist_nstar, int hist_cap, int* hist_n, int* n_layer)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    uint32_t *du, *dv;
    double *dsm;
    int64_t *dct;
    cudaMalloc(&du, (size_t)n_edges * 4);
    cudaMalloc(&dv, (size_t)n_edges * 4);
    cudaMalloc(&dsm, (size_t)n_edges * 8);
    cudaMalloc(&dct, (size_t)n_edges * 8);
    cudaMemcpy(du, u_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsm, sm_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dct, ct_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    P0zProf z{};
    z.hist_nlive = hist_nlive;
    z.hist_nprop = hist_nprop;
    z.hist_nmerge = hist_nmerge;
    z.hist_ngc = hist_ngc;
    z.hist_nact = hist_nact;
    z.hist_ndirty = hist_ndirty;
    z.hist_nstar = hist_nstar;
    z.hist_cap = hist_cap;
    z.hist_n = 0;
    z.n_layer = 0;
    int rc = parhac_e6s_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
        parent_out, max_id, stats_out, &z);
    if (hist_n) *hist_n = z.hist_n;
    if (n_layer) *n_layer = z.n_layer;
    cudaFree(du); cudaFree(dv); cudaFree(dsm); cudaFree(dct);
    return rc;
}

extern "C" int parhac_e6s_p0aa(
    const uint32_t* u_h, const uint32_t* v_h,
    const double* sm_h, const int64_t* ct_h,
    int64_t n_edges, const double* aff_thr, int n_thr, double eps,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out,
    double* phase_ms, int* n_layer, int* layer_outers, int* layer_merges,
    int* layer_first_zero, int64_t* work_counts)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    uint32_t *du, *dv;
    double *dsm;
    int64_t *dct;
    cudaMalloc(&du, (size_t)n_edges * 4);
    cudaMalloc(&dv, (size_t)n_edges * 4);
    cudaMalloc(&dsm, (size_t)n_edges * 8);
    cudaMalloc(&dct, (size_t)n_edges * 8);
    cudaMemcpy(du, u_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsm, sm_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dct, ct_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    P0aaProf aa{};
    int rc = parhac_e6s_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
        parent_out, max_id, stats_out, nullptr, &aa, 64);
    if (phase_ms) {
        phase_ms[0] = aa.compact_ms;
        phase_ms[1] = aa.propose_ms;
        phase_ms[2] = aa.pack_ms;
        phase_ms[3] = aa.sort_ms;
        phase_ms[4] = aa.accept_ms;
        phase_ms[5] = aa.compress_ms;
        phase_ms[6] = aa.freeze_ms;
        phase_ms[7] = aa.color_ms;
        phase_ms[8] = aa.memset_ms;
        phase_ms[9] = aa.d2h_ms;
        phase_ms[10] = aa.host_ms;
    }
    if (n_layer) *n_layer = aa.n_layer;
    if (layer_outers) {
        for (int i = 0; i < aa.n_layer && i < 64; ++i) layer_outers[i] = aa.layer_outers[i];
    }
    if (layer_merges) {
        for (int i = 0; i < aa.n_layer && i < 64; ++i) layer_merges[i] = aa.layer_merges[i];
    }
    if (layer_first_zero) {
        for (int i = 0; i < aa.n_layer && i < 64; ++i)
            layer_first_zero[i] = aa.layer_first_zero[i];
    }
    if (work_counts) {
        work_counts[0] = aa.sum_nlive;
        work_counts[1] = aa.sum_above;
    }
    g_iou_scan = aa.rewrite_scan_ms;
    g_iou_hash = aa.hash_ms;
    g_iou_radix = aa.compact_radix_ms;
    cudaFree(du); cudaFree(dv); cudaFree(dsm); cudaFree(dct);
    return rc;
}

extern "C" int parhac_e6s_cap(
    const uint32_t* u_h, const uint32_t* v_h,
    const double* sm_h, const int64_t* ct_h,
    int64_t n_edges, const double* aff_thr, int n_thr, double eps,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out, int max_outer)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    uint32_t *du, *dv;
    double *dsm;
    int64_t *dct;
    cudaMalloc(&du, (size_t)n_edges * 4);
    cudaMalloc(&dv, (size_t)n_edges * 4);
    cudaMalloc(&dsm, (size_t)n_edges * 8);
    cudaMalloc(&dct, (size_t)n_edges * 8);
    cudaMemcpy(du, u_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsm, sm_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dct, ct_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    int rc = parhac_e6s_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
        parent_out, max_id, stats_out, nullptr, nullptr, max_outer);
    cudaFree(du); cudaFree(dv); cudaFree(dsm); cudaFree(dct);
    return rc;
}

extern "C" int parhac_paper_d_profile(
    const uint32_t* u_h, const uint32_t* v_h,
    const double* sm_h, const int64_t* ct_h,
    int64_t n_edges, const double* aff_thr, int n_thr, double eps,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out,
    double* phase_ms, int64_t* hist_nlive, int64_t* hist_nprop,
    int64_t* hist_nmerge, int hist_cap, int* hist_n, int skip_debug,
    int64_t* hist_nact)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    uint32_t *du, *dv;
    double *dsm;
    int64_t *dct;
    cudaMalloc(&du, (size_t)n_edges * 4);
    cudaMalloc(&dv, (size_t)n_edges * 4);
    cudaMalloc(&dsm, (size_t)n_edges * 8);
    cudaMalloc(&dct, (size_t)n_edges * 8);
    cudaMemcpy(du, u_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsm, sm_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dct, ct_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    P0yProf prof{};
    prof.hist_nlive = hist_nlive;
    prof.hist_nprop = hist_nprop;
    prof.hist_nmerge = hist_nmerge;
    prof.hist_nact = hist_nact;
    prof.hist_cap = hist_cap;
    prof.hist_n = 0;
    prof.skip_debug = skip_debug;
    int rc = parhac_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
        parent_out, max_id, stats_out, &prof);
    if (phase_ms) {
        phase_ms[0] = prof.compact_ms;
        phase_ms[1] = prof.propose_ms;
        phase_ms[2] = prof.accept_ms;
        phase_ms[3] = prof.d2h_ms;
        phase_ms[4] = prof.memset_ms;
        phase_ms[5] = prof.debug_ms;
    }
    if (hist_n) *hist_n = prof.hist_n;
    cudaFree(du); cudaFree(dv); cudaFree(dsm); cudaFree(dct);
    return rc;
}
