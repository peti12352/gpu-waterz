// N19 H1: affinity bin-ladder MEAN. Work-efficient: find min non-empty bin in
// parallel, drain that bin with block merge, abort_ms between rounds.
// NOT N18 multi-bin unordered pop. Not a 2 Gvox/s claim. Not 3090 Ti.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <chrono>

static constexpr int NBIN = 256;
static constexpr int BLK = 256;

static inline __device__ __host__ int bin_of(double score) {
    int i = (int)(score * (double)NBIN);
    if (i < 0) i = 0;
    if (i >= NBIN) i = NBIN - 1;
    return i;
}

static inline __device__ uint32_t dfind(uint32_t* p, uint32_t x) {
    uint32_t r = x;
    while (p[r] != r) r = p[r];
    while (p[x] != r) {
        uint32_t n = p[x];
        p[x] = r;
        x = n;
    }
    return r;
}

__global__ void k_init_nodes(uint32_t* parent, uint32_t n) {
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) parent[i] = i;
}

__global__ void k_clear_bins(int* bin_head, int* bin_count) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < NBIN) {
        bin_head[i] = -1;
        bin_count[i] = 0;
    }
}

__global__ void k_enqueue_ladder(
    const double* sm, const int64_t* ct, int64_t nedge,
    int* bin_head, int* bin_next, int* bin_count, uint8_t* deleted)
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nedge) return;
    deleted[e] = 0;
    double mean = (ct[e] > 0) ? (sm[e] / (double)ct[e]) : 0.0;
    double score = 1.0 - mean;
    int b = bin_of(score);
    bin_next[e] = atomicExch(&bin_head[b], (int)e);
    atomicAdd(&bin_count[b], 1);
}

__global__ void k_min_nonempty(const int* bin_count, int* out_min) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int m = NBIN;
        for (int b = 0; b < NBIN; ++b) {
            if (bin_count[b] > 0) { m = b; break; }
        }
        *out_min = m;
    }
}

// Drain one bin: thread0 pops FIFO-style via head, merges when valid.
__global__ void k_drain_bin(
    uint32_t* parent,
    const uint32_t* u, const uint32_t* v,
    const double* sm, const int64_t* ct,
    int* bin_head, int* bin_next, int* bin_count,
    uint8_t* deleted, double score_thr, int bin,
    int* n_merge, int max_m)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int h = bin_head[bin];
    int local = 0;
    while (h >= 0 && local < max_m) {
        int e = h;
        h = bin_next[e];
        atomicAdd(&bin_count[bin], -1);
        if (deleted[e]) continue;
        uint32_t ra = dfind(parent, u[e]);
        uint32_t rb = dfind(parent, v[e]);
        if (ra == rb) {
            deleted[e] = 1;
            continue;
        }
        double mean = (ct[e] > 0) ? (sm[e] / (double)ct[e]) : 0.0;
        double score = 1.0 - mean;
        if (score >= score_thr) {
            // put back and stop draining this bin for this thr
            bin_next[e] = h;
            bin_head[bin] = e;
            atomicAdd(&bin_count[bin], 1);
            break;
        }
        if (ra > rb) {
            uint32_t t = ra;
            ra = rb;
            rb = t;
        }
        parent[rb] = ra;
        deleted[e] = 1;
        int slot = atomicAdd(n_merge, 1);
        (void)slot;
        local++;
    }
    bin_head[bin] = h;
}

__global__ void k_compress_all(uint32_t* parent, uint32_t n) {
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) parent[i] = dfind(parent, i);
}

extern "C" int binladder_mean_d(
    const uint32_t* u_h, const uint32_t* v_h,
    const double* sm_h, const int64_t* ct_h,
    int64_t n_edges, const double* aff_thr, int n_thr,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out,
    int abort_ms)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    uint32_t nnode = max_id + 1;
    uint32_t *du = nullptr, *dv = nullptr, *dparent = nullptr;
    double *dsm = nullptr;
    int64_t *dct = nullptr;
    int *bh = nullptr, *bn = nullptr, *bc = nullptr, *dmin = nullptr, *nm = nullptr;
    uint8_t *del = nullptr;

    cudaMalloc(&du, (size_t)n_edges * 4);
    cudaMalloc(&dv, (size_t)n_edges * 4);
    cudaMalloc(&dsm, (size_t)n_edges * 8);
    cudaMalloc(&dct, (size_t)n_edges * 8);
    cudaMalloc(&dparent, (size_t)nnode * 4);
    cudaMalloc(&bh, NBIN * 4);
    cudaMalloc(&bn, (size_t)n_edges * 4);
    cudaMalloc(&bc, NBIN * 4);
    cudaMalloc(&del, (size_t)n_edges);
    cudaMalloc(&dmin, 4);
    cudaMalloc(&nm, 4);

    cudaMemcpy(du, u_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsm, sm_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dct, ct_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);

    const int threads = BLK;
    const int nb = (int)((nnode + threads - 1) / threads);
    const int eb = (int)((n_edges + threads - 1) / threads);
    auto t0 = std::chrono::steady_clock::now();
    int rc = 1;

    for (int t = 0; t < n_thr; ++t) {
        k_init_nodes<<<nb, threads>>>(dparent, nnode);
        k_clear_bins<<<1, NBIN>>>(bh, bc);
        cudaMemset(del, 0, (size_t)n_edges);
        k_enqueue_ladder<<<eb, threads>>>(dsm, dct, n_edges, bh, bn, bc, del);

        double T = aff_thr[t];
        double score_thr = 1.0 - T;
        int64_t total_m = 0;
        int rounds = 0;
        while (rounds < 500000) {
            auto now = std::chrono::steady_clock::now();
            int elapsed = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                now - t0).count();
            if (abort_ms > 0 && elapsed > abort_ms) {
                fprintf(stderr, "N19 H1 abort %d ms rounds=%d\n", elapsed, rounds);
                rc = 0;
                break;
            }
            k_min_nonempty<<<1, 1>>>(bc, dmin);
            int mb = NBIN;
            cudaMemcpy(&mb, dmin, 4, cudaMemcpyDeviceToHost);
            if (mb >= NBIN) break;
            cudaMemset(nm, 0, 4);
            k_drain_bin<<<1, BLK>>>(
                dparent, du, dv, dsm, dct, bh, bn, bc, del, score_thr, mb, nm, 64);
            int nml = 0;
            cudaMemcpy(&nml, nm, 4, cudaMemcpyDeviceToHost);
            if (nml <= 0) {
                // force count refresh: if bin empty continue
                int cnt = 0;
                cudaMemcpy(&cnt, bc + mb, 4, cudaMemcpyDeviceToHost);
                if (cnt <= 0) {
                    // mark empty
                    int zero = 0;
                    cudaMemcpy(bc + mb, &zero, 4, cudaMemcpyHostToDevice);
                }
                // if no merges and bin still has edges above thr, advance
                if (nml <= 0 && cnt <= 0) {
                    /* next bin via loop */
                } else if (nml <= 0) {
                    break; // remaining edges above threshold
                }
            }
            total_m += nml;
            ++rounds;
            if (nml <= 0) break;
        }
        k_compress_all<<<nb, threads>>>(dparent, nnode);
        cudaMemcpy(parent_out + (size_t)t * nnode, dparent,
                   (size_t)nnode * 4, cudaMemcpyDeviceToHost);
        if (stats_out) {
            stats_out[t * 3 + 0] = rounds;
            stats_out[t * 3 + 1] = total_m;
            stats_out[t * 3 + 2] = rounds;
        }
        if (!rc) break;
    }

    cudaFree(du); cudaFree(dv); cudaFree(dsm); cudaFree(dct);
    cudaFree(dparent); cudaFree(bh); cudaFree(bn); cudaFree(bc);
    cudaFree(del); cudaFree(dmin); cudaFree(nm);
    return rc;
}
