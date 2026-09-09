// N18 B1: parallel multi-bin MEAN BinQueue. Host abort via abort_ms between iters.
// Not single-thread drain (N14 T3 dead). Not a 2 Gvox/s claim. Not 3090 Ti.
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

__global__ void k_init_heads(int* head, uint32_t n) {
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) head[i] = -1;
}

__global__ void k_build_inc(
    const uint32_t* u, const uint32_t* v, int64_t nedge,
    int* head, int* next)
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nedge) return;
    uint32_t a = u[e], b = v[e];
    next[2 * e] = atomicExch(&head[a], (int)(2 * e));
    next[2 * e + 1] = atomicExch(&head[b], (int)(2 * e + 1));
}

// Parallel enqueue: each edge inserts itself into its bin via atomic head push
// (LIFO within bin — still multi-bin parallel, not serial FIFO drain).
__global__ void k_enqueue_par(
    const double* sm, const int64_t* ct, int64_t nedge,
    int* bin_head, int* bin_next, uint8_t* deleted)
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nedge) return;
    deleted[e] = 0;
    double mean = (ct[e] > 0) ? (sm[e] / (double)ct[e]) : 0.0;
    double score = 1.0 - mean;
    int b = bin_of(score);
    bin_next[e] = atomicExch(&bin_head[b], (int)e);
}

__global__ void k_clear_bins(int* bin_head) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < NBIN) bin_head[i] = -1;
}

// One block per bin: pop up to MAX_POP candidates that are live and below thr.
__global__ void k_bin_pop_batch(
    uint32_t* parent,
    const uint32_t* u, const uint32_t* v,
    const double* sm, const int64_t* ct,
    int* bin_head, const int* bin_next,
    uint8_t* deleted, double score_thr,
    int* merge_u, int* merge_v, int* n_merge, int max_merge)
{
    int b = (int)blockIdx.x;
    if (b >= NBIN) return;
    __shared__ int s_n;
    if (threadIdx.x == 0) s_n = 0;
    __syncthreads();
    // Thread 0 walks this bin's list and records eligible merges.
    if (threadIdx.x == 0) {
        int h = bin_head[b];
        while (h >= 0 && s_n < 8) {
            int e = h;
            h = bin_next[e];
            if (deleted[e]) continue;
            uint32_t ra = dfind(parent, u[e]);
            uint32_t rb = dfind(parent, v[e]);
            if (ra == rb) {
                deleted[e] = 1;
                continue;
            }
            double mean = (ct[e] > 0) ? (sm[e] / (double)ct[e]) : 0.0;
            double score = 1.0 - mean;
            if (score >= score_thr) break;
            int slot = atomicAdd(n_merge, 1);
            if (slot < max_merge) {
                if (ra > rb) {
                    uint32_t t = ra;
                    ra = rb;
                    rb = t;
                }
                merge_u[slot] = (int)ra;
                merge_v[slot] = (int)rb;
                deleted[e] = 1;
                s_n++;
            } else {
                atomicAdd(n_merge, -1);
                break;
            }
        }
        bin_head[b] = h;
    }
}

__global__ void k_apply_merges(
    uint32_t* parent, const int* merge_u, const int* merge_v, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    uint32_t a = dfind(parent, (uint32_t)merge_u[i]);
    uint32_t b = dfind(parent, (uint32_t)merge_v[i]);
    if (a == b) return;
    if (a > b) {
        uint32_t t = a;
        a = b;
        b = t;
    }
    atomicMin(&parent[b], a);
}

__global__ void k_compress_all(uint32_t* parent, uint32_t n) {
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) parent[i] = dfind(parent, i);
}

extern "C" int binqueue_par_mean_d(
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
    int *head = nullptr, *next = nullptr, *bh = nullptr, *bn = nullptr;
    uint8_t *del = nullptr;
    int *merge_u = nullptr, *merge_v = nullptr, *n_merge = nullptr;

    cudaMalloc(&du, (size_t)n_edges * 4);
    cudaMalloc(&dv, (size_t)n_edges * 4);
    cudaMalloc(&dsm, (size_t)n_edges * 8);
    cudaMalloc(&dct, (size_t)n_edges * 8);
    cudaMalloc(&dparent, (size_t)nnode * 4);
    cudaMalloc(&head, (size_t)nnode * 4);
    cudaMalloc(&next, (size_t)n_edges * 2 * 4);
    cudaMalloc(&bh, NBIN * 4);
    cudaMalloc(&bn, (size_t)n_edges * 4);
    cudaMalloc(&del, (size_t)n_edges);
    const int MAXM = 4096;
    cudaMalloc(&merge_u, MAXM * 4);
    cudaMalloc(&merge_v, MAXM * 4);
    cudaMalloc(&n_merge, 4);

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
        k_init_heads<<<nb, threads>>>(head, nnode);
        k_clear_bins<<<1, NBIN>>>(bh);
        cudaMemset(del, 0, (size_t)n_edges);
        k_build_inc<<<eb, threads>>>(du, dv, n_edges, head, next);
        k_enqueue_par<<<eb, threads>>>(dsm, dct, n_edges, bh, bn, del);

        double T = aff_thr[t];
        double score_thr = 1.0 - T;
        int64_t total_merges = 0;
        int rounds = 0;
        const int ROUND_CAP = 200000;
        while (rounds < ROUND_CAP) {
            auto now = std::chrono::steady_clock::now();
            int elapsed = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                now - t0).count();
            if (abort_ms > 0 && elapsed > abort_ms) {
                fprintf(stderr, "N18 B1 abort after %d ms rounds=%d\n",
                        elapsed, rounds);
                rc = 0;
                break;
            }
            cudaMemset(n_merge, 0, 4);
            k_bin_pop_batch<<<NBIN, BLK>>>(
                dparent, du, dv, dsm, dct, bh, bn, del, score_thr,
                merge_u, merge_v, n_merge, MAXM);
            int nm = 0;
            cudaMemcpy(&nm, n_merge, 4, cudaMemcpyDeviceToHost);
            if (nm <= 0) break;
            if (nm > MAXM) nm = MAXM;
            int mb = (nm + 255) / 256;
            k_apply_merges<<<mb, 256>>>(dparent, merge_u, merge_v, nm);
            total_merges += nm;
            ++rounds;
        }
        k_compress_all<<<nb, threads>>>(dparent, nnode);
        cudaMemcpy(parent_out + (size_t)t * nnode, dparent,
                   (size_t)nnode * 4, cudaMemcpyDeviceToHost);
        if (stats_out) {
            stats_out[t * 3 + 0] = rounds;
            stats_out[t * 3 + 1] = total_merges;
            stats_out[t * 3 + 2] = rounds;
        }
        if (!rc) break;
    }

    cudaFree(du); cudaFree(dv); cudaFree(dsm); cudaFree(dct);
    cudaFree(dparent); cudaFree(head); cudaFree(next);
    cudaFree(bh); cudaFree(bn); cudaFree(del);
    cudaFree(merge_u); cudaFree(merge_v); cudaFree(n_merge);
    return rc;
}
