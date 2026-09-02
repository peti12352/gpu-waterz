// Device paper-ParHAC ContractLayer, waterz contact-mean, ε=0.08.
// Live-subgraph compact + combine. Same control flow as parhac_paper_cpu.
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
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
    // Must mix seed through multiply — XOR with id*odd is just (i^seed) parity,
    // so even–even leftover edges stay same-color forever.
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

__global__ void k_propose(
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    uint32_t* parent, const uint32_t* sz, const uint8_t* color, const uint8_t* frozen,
    int64_t n, double TL, unsigned long long* prop, uint64_t seed, uint64_t cseed, int* dbg)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n || ct[i] < 1) return;
    if (dbg) atomicAdd(dbg + 0, 1);
    double mean = sm[i] / (double)ct[i];
    if (mean < TL) return;
    if (dbg) atomicAdd(dbg + 1, 1);
    uint32_t a = dfind_nocomp(parent, u[i]), b = dfind_nocomp(parent, v[i]);
    if (a == b || a == 0 || b == 0) return;
    if (dbg) atomicAdd(dbg + 2, 1);
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
    if (sz[r] < sz[bl]) return;
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
    if (sz[r] < sz[bl]) return;
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
    if (sz[r] < sz[bl]) return;
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
    if (sz[r] < sz[bl]) return;
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
struct HSlot {
    unsigned long long key;
    double sm;
    unsigned long long ct;
};

static inline __device__ unsigned long long edge_key(uint32_t u, uint32_t v)
{
    return ((unsigned long long)u << 32) | (unsigned long long)v;
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
    const uint8_t* keep, int64_t n, HSlot* tab, int ntab)
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
    unsigned long long h = key * 0x9E3779B97F4A7C15ull;
    int mask = ntab - 1;
    int slot = (int)(h & (unsigned long long)mask);
    for (int s = 0; s < 128; ++s) {
        int idx = (slot + s) & mask;
        unsigned long long old = atomicCAS(&tab[idx].key, 0ull, key);
        if (old == 0 || old == key) {
            atomicAdd(&tab[idx].sm, sm[i]);
            atomicAdd(&tab[idx].ct, (unsigned long long)ct[i]);
            return;
        }
    }
}

__global__ void k_hash_count(const HSlot* tab, int ntab, int* nout)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntab) return;
    if (tab[i].key != 0 && tab[i].ct > 0) atomicAdd(nout, 1);
}

__global__ void k_hash_emit(
    const HSlot* tab, int ntab, uint32_t* u, uint32_t* v, double* sm, int64_t* ct,
    int* nout)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntab) return;
    if (tab[i].key == 0 || tab[i].ct == 0) return;
    int slot = atomicAdd(nout, 1);
    u[slot] = (uint32_t)(tab[i].key >> 32);
    v[slot] = (uint32_t)(tab[i].key & 0xffffffffull);
    sm[slot] = tab[i].sm;
    ct[slot] = (int64_t)tab[i].ct;
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

static int hash_combine_live(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct, uint8_t* dkeep,
    uint32_t* dparent, int nnode, int64_t n, int64_t* n_out,
    int be, int threads,
    uint32_t* tu, uint32_t* tv, double* tsm, int64_t* tct,
    HSlot* dtab, int ntab, int* dnout)
{
    int bn = (nnode + threads - 1) / threads;
    k_compress<<<bn, threads>>>(dparent, nnode);
    k_rewrite<<<be, threads>>>(du, dv, dsm, dct, dparent, n, dkeep, 0.0);
    int tb = (ntab + threads - 1) / threads;
    k_hash_clear<<<tb, threads>>>(dtab, ntab);
    k_hash_insert<<<be, threads>>>(du, dv, dsm, dct, dkeep, n, dtab, ntab);
    cudaMemset(dnout, 0, 4);
    k_hash_emit<<<tb, threads>>>(dtab, ntab, tu, tv, tsm, tct, dnout);
    int m = 0;
    cudaMemcpy(&m, dnout, 4, cudaMemcpyDeviceToHost);
    if (m <= 0) {
        *n_out = 0;
        return 1;
    }
    cudaMemcpy(du, tu, (size_t)m * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dv, tv, (size_t)m * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dsm, tsm, (size_t)m * 8, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dct, tct, (size_t)m * 8, cudaMemcpyDeviceToDevice);
    *n_out = m;
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

static int compact_radix(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct, uint8_t* dkeep,
    uint32_t* dparent, int nnode, int64_t n, int64_t* n_out,
    int be, int threads,
    uint32_t* tu, uint32_t* tv, double* tsm, int64_t* tct,
    unsigned long long* dkey, unsigned long long* dkeyo, double TL = 0.0)
{
    int bn = (nnode + threads - 1) / threads;
    k_compress<<<bn, threads>>>(dparent, nnode);
    k_rewrite<<<be, threads>>>(du, dv, dsm, dct, dparent, n, dkeep, TL);
    thrust::device_ptr<uint32_t> pu(du), pv(dv);
    auto in = thrust::make_zip_iterator(thrust::make_tuple(
        pu, pv, thrust::device_ptr<double>(dsm), thrust::device_ptr<int64_t>(dct)));
    auto out = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<uint32_t>(tu), thrust::device_ptr<uint32_t>(tv),
        thrust::device_ptr<double>(tsm), thrust::device_ptr<int64_t>(tct)));
    auto end = thrust::copy_if(in, in + n, thrust::device_ptr<uint8_t>(dkeep), out, KeepOn());
    int64_t m = end - out;
    if (m <= 0) {
        *n_out = 0;
        return 1;
    }
    int bm = (int)((m + threads - 1) / threads);
    if (bm < 1) bm = 1;
    k_pack_uvkey<<<bm, threads>>>(tu, tv, dkey, m);
    auto vals = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<double>(tsm), thrust::device_ptr<int64_t>(tct)));
    thrust::sort_by_key(
        thrust::device_ptr<unsigned long long>(dkey),
        thrust::device_ptr<unsigned long long>(dkey) + m, vals);
    auto red = thrust::reduce_by_key(
        thrust::device_ptr<unsigned long long>(dkey),
        thrust::device_ptr<unsigned long long>(dkey) + m,
        vals,
        thrust::device_ptr<unsigned long long>(dkeyo),
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_ptr<double>(dsm), thrust::device_ptr<int64_t>(dct))),
        thrust::equal_to<unsigned long long>(), SumPair());
    int64_t m2 = red.first - thrust::device_ptr<unsigned long long>(dkeyo);
    k_unpack_uvkey<<<(int)((m2 + threads - 1) / threads), threads>>>(dkeyo, du, dv, m2);
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
                        0xC0FFEEULL + (uint64_t)layer * 10007 + (uint64_t)outer, ddbg);
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
    int layer_outers[64];
    int layer_merges[64];
};

static int parhac_e6s_dev(
    uint32_t* du, uint32_t* dv, double* dsm, int64_t* dct, int64_t n_edges,
    const double* aff_thr, int n_thr, double eps,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out,
    P0zProf* zprof = nullptr, P0aaProf* aa = nullptr, int max_outer = 64)
{
    const int nnode = (int)max_id + 1;
    uint32_t *dparent, *dsz, *dsz0, *dreds, *dblues, *dadd;
    unsigned long long *dprop;
    uint32_t *tu, *tv, *cu, *cv;
    double *dblk, *tsm, *csm;
    int64_t *tct, *cct;
    uint8_t *dkeep, *dcolor, *dfrozen;
    float *dpris;
    int *dnmerge, *dnprop, *ddbg, *dnact, *dndirty, *dnstar;
    uint8_t *ddirty_blue, *ddirty_star;
    int ntab = next_pow2((int)(n_edges * 2 + 1024));
    if (ntab < 1024) ntab = 1024;
    HSlot* dtab = nullptr;
    int* dnout = nullptr;
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
    cudaMalloc(&cu, (size_t)n_edges * 4);
    cudaMalloc(&cv, (size_t)n_edges * 4);
    cudaMalloc(&csm, (size_t)n_edges * 8);
    cudaMalloc(&cct, (size_t)n_edges * 8);
    cudaMalloc(&dtab, (size_t)ntab * sizeof(HSlot));
    cudaMalloc(&dnout, 4);
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
    if (max_outer < 1) max_outer = 64;
    {
        int be0 = (int)((n_edges + threads - 1) / threads);
        if (be0 < 1) be0 = 1;
        k_scale_sm_bytes<<<be0, threads>>>(dsm, n_edges);
    }
    EvAccum ev_compact(aa ? &aa->compact_ms : nullptr);
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
            int be = (int)((nlive + threads - 1) / threads);
            if (be < 1) be = 1;
            k_compress<<<bn, threads>>>(dparent, nnode);
            k_wmax_live<<<nblk, 256>>>(du, dv, dsm, dct, dparent, nlive, dblk);
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
            compact_radix(du, dv, dsm, dct, dkeep, dparent, nnode, nlive, &nlive,
                be, threads, tu, tv, tsm, tct, dkey, dkeyo);
            ev_compact.stop();
            if (nlive <= 0) break;
            be = (int)((nlive + threads - 1) / threads);
            if (be < 1) be = 1;
            int layer_merges = 0;
            int layer_outers = 0;
            for (int outer = 0; outer < max_outer; ++outer) {
                ev_compress.start();
                k_compress<<<bn, threads>>>(dparent, nnode);
                k_zero_sz<<<bn, threads>>>(dsz, nnode);
                k_rebuild_sz<<<bn, threads>>>(dparent, dsz, nnode);
                ev_compress.stop();
                ev_memset.start();
                cudaMemset(dcolor, 0, (size_t)nnode);
                cudaMemset(dfrozen, 0, (size_t)nnode);
                ev_memset.stop();
                ev_color.start();
                k_color<<<bn, threads>>>(dcolor, dparent, nnode,
                    0xC0FFEEULL + (uint64_t)layer * 10007 + (uint64_t)outer);
                k_copy_sz<<<bn, threads>>>(dsz, dsz0, nnode);
                ev_color.stop();
                for (int inner = 0; inner < 64; ++inner) {
                    ev_memset.start();
                    cudaMemset(dprop, 0, (size_t)nnode * 8);
                    cudaMemset(dnmerge, 0, 4);
                    cudaMemset(dnprop, 0, 4);
                    if (zprof) cudaMemset(ddbg, 0, 20);
                    ev_memset.stop();
                    ev_propose.start();
                    k_propose<<<be, threads>>>(
                        du, dv, dsm, dct, dparent, dsz, dcolor, dfrozen,
                        nlive, TL, dprop,
                        0xA5A5ULL + (uint64_t)inner * 17 + (uint64_t)outer,
                        0xC0FFEEULL + (uint64_t)layer * 10007 + (uint64_t)outer,
                        zprof ? ddbg : nullptr);
                    ev_propose.stop();
                    ev_pack.start();
                    k_pack_prop<<<bn, threads>>>(
                        dprop, dsz, nnode, dreds, dblues, dadd, dpris, dnprop);
                    ev_pack.stop();
                    int nprop = 0;
                    ev_d2h.start();
                    cudaMemcpy(&nprop, dnprop, 4, cudaMemcpyDeviceToHost);
                    ev_d2h.stop();
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
                    ev_sort.start();
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
                    ev_sort.stop();
                    int bp = (nprop + threads - 1) / threads;
                    if (bp < 1) bp = 1;
                    ev_accept.start();
                    k_accept_reds<<<bp, threads>>>(
                        dreds, dblues, dadd, nprop, dparent, dsz, dfrozen, eps, dnmerge);
                    ev_accept.stop();
                    ev_compress.start();
                    k_compress<<<bn, threads>>>(dparent, nnode);
                    ev_compress.stop();
                    ev_freeze.start();
                    k_freeze<<<bn, threads>>>(dparent, dsz, dsz0, dcolor, dfrozen, nnode, eps);
                    ev_freeze.stop();
                    int hm = 0;
                    ev_d2h.start();
                    cudaMemcpy(&hm, dnmerge, 4, cudaMemcpyDeviceToHost);
                    ev_d2h.stop();
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
                    if (hm == 0) break;
                    ev_compact.start();
                    compact_radix(du, dv, dsm, dct, dkeep, dparent, nnode, nlive, &nlive,
                        be, threads, tu, tv, tsm, tct, dkey, dkeyo, 0.0);
                    ev_compact.stop();
                    be = (int)((nlive + threads - 1) / threads);
                    if (be < 1) be = 1;
                    if (nlive <= 0) break;
                }
                ++nouter;
                ++layer_outers;
                if (nlive <= 0) break;
            }
            if (aa && aa->n_layer > 0 && aa->n_layer <= 64) {
                int li = aa->n_layer - 1;
                aa->layer_outers[li] = layer_outers;
                aa->layer_merges[li] = layer_merges;
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
    cudaFree(dfrozen); cudaFree(dprop); cudaFree(dpris);
    cudaFree(dreds); cudaFree(dblues); cudaFree(dadd);
    cudaFree(dnmerge); cudaFree(dnprop); cudaFree(dkeep);
    cudaFree(tu); cudaFree(tv); cudaFree(tsm); cudaFree(tct); cudaFree(dblk);
    cudaFree(dtab); cudaFree(dnout); cudaFree(dkey); cudaFree(dkeyo);
    cudaFree(cu); cudaFree(cv); cudaFree(csm); cudaFree(cct);
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

    int ntab = next_pow2((int)(n_edges * 4 + 1024));
    if (ntab < 2048) ntab = 2048;
    EHash* dtab = nullptr;
    int *ddeg, *dcur, *doff, *dlen, *dadj, *ovf_head, *ovf_eid, *ovf_nxt;
    int *dovf_used, *dfail, *dgc, *dgc2, *dngc, *dntouch, *dtouched, *dnkill;
    int adj_cap = (int)(n_edges * 2 + 16);
    int ovf_cap = (int)(n_edges * 4 + 16);
    if (cudaMalloc(&dtab, (size_t)ntab * sizeof(EHash)) != cudaSuccess) {
        std::fprintf(stderr, "E6t_OOM hash ntab=%d\n", ntab);
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
                be, threads, tu, tv, tsm, tct, dkey, dkeyo);
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
                                be, threads, tu, tv, tsm, tct, dkey, dkeyo, 0.0);
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
                            be, threads, tu, tv, tsm, tct, dkey, dkeyo, 0.0);
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

extern "C" int parhac_paper_d_dev(
    const uint32_t* du_in, const uint32_t* dv_in,
    const double* dsm_in, const int64_t* dct_in,
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
    cudaMemcpy(du, du_in, (size_t)n_edges * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dv, dv_in, (size_t)n_edges * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dsm, dsm_in, (size_t)n_edges * 8, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dct, dct_in, (size_t)n_edges * 8, cudaMemcpyDeviceToDevice);
    int rc = std::getenv("WATERZ_PAPER_E6T")
        ? parhac_e6t_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
            parent_out, max_id, stats_out)
        : parhac_e6s_dev(du, dv, dsm, dct, n_edges, aff_thr, n_thr, eps,
            parent_out, max_id, stats_out);
    cudaFree(du); cudaFree(dv); cudaFree(dsm); cudaFree(dct);
    return rc;
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
    double* phase_ms, int* n_layer, int* layer_outers, int* layer_merges)
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
    int64_t* hist_nmerge, int hist_cap, int* hist_n, int skip_debug)
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
