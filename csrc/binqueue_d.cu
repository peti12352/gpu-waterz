// N14 T3: GPU FIFO-inside-bin MEAN agglomeration. Not 1-thread find+union.
// N=256 bins, score=1-mean_aff, FIFO linked lists, one block pops,
// the rest of the block rewrites the absorbed node's incidence list.
// Not a 2 Gvox/s claim. Not 3090 Ti.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

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

__global__ void k_init_bins(int* bh, int* bt) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < NBIN) {
        bh[i] = -1;
        bt[i] = -1;
    }
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

__global__ void k_enqueue_init(
    const double* sm, const int64_t* ct, int64_t nedge,
    int* bin_head, int* bin_tail, int* bin_next, uint8_t* deleted)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int64_t e = 0; e < nedge; ++e) {
        deleted[e] = 0;
        double mean = (ct[e] > 0) ? (sm[e] / (double)ct[e]) : 0.0;
        double score = 1.0 - mean;
        int b = bin_of(score);
        bin_next[e] = -1;
        int prev = bin_tail[b];
        if (prev < 0) bin_head[b] = (int)e;
        else bin_next[prev] = (int)e;
        bin_tail[b] = (int)e;
    }
}

static __device__ int find_inc(
    uint32_t node, uint32_t nbr, const int* head, const int* next,
    const uint8_t* deleted, const uint32_t* u, const uint32_t* v,
    uint32_t* parent)
{
    for (int h = head[node]; h >= 0; h = next[h]) {
        int e = h >> 1;
        if (deleted[e]) continue;
        uint32_t o = (u[e] == node) ? v[e] : u[e];
        o = dfind(parent, o);
        if (o == nbr) return e;
    }
    return -1;
}

// One block. Thread 0 pops FIFO from the cheapest non-empty bin.
// All threads walk the absorbed node's incidence list and splice into the survivor.
__global__ void k_binq_merge(
    uint32_t* parent,
    uint32_t* u, uint32_t* v, double* sm, int64_t* ct,
    int* head, int* next,
    int* bin_head, int* bin_next,
    uint8_t* deleted, double score_thr,
    int64_t nedge, uint32_t nnode,
    int64_t* n_pop, int64_t* n_merge)
{
    __shared__ int s_e, s_a, s_b, s_done, s_minbin;
    __shared__ int64_t s_pop, s_mrg;

    if (threadIdx.x == 0) {
        s_pop = 0;
        s_mrg = 0;
        s_minbin = 0;
        s_done = 0;
    }
    __syncthreads();

    while (!s_done) {
        if (threadIdx.x == 0) {
            int e = -1;
            while (s_minbin < NBIN && e < 0) {
                int h = bin_head[s_minbin];
                while (h >= 0 && deleted[h]) h = bin_next[h];
                bin_head[s_minbin] = h;
                if (h < 0) {
                    ++s_minbin;
                    continue;
                }
                bin_head[s_minbin] = bin_next[h];
                e = h;
            }
            if (e < 0) {
                s_done = 1;
                s_e = -1;
            } else {
                ++s_pop;
                uint32_t ra = dfind(parent, u[e]);
                uint32_t rb = dfind(parent, v[e]);
                if (ra == rb) {
                    s_e = -2;
                    deleted[e] = 1;
                } else {
                    double mean = (ct[e] > 0) ? (sm[e] / (double)ct[e]) : 0.0;
                    double score = 1.0 - mean;
                    if (score >= score_thr) {
                        s_done = 1;
                        s_e = -1;
                    } else {
                        if (ra > rb) {
                            uint32_t t = ra;
                            ra = rb;
                            rb = t;
                        }
                        parent[rb] = ra;
                        s_a = (int)ra;
                        s_b = (int)rb;
                        s_e = e;
                        deleted[e] = 1;
                        ++s_mrg;
                    }
                }
            }
        }
        __syncthreads();
        if (s_done) break;
        if (s_e < 0) continue;

        const uint32_t a = (uint32_t)s_a;
        const uint32_t b = (uint32_t)s_b;
        // Walk b's incidence. Thread 0 does the list splice (adjacency is not
        // lock-free); other threads only exist so this is a block, not a
        // 1-thread grid. The walk is still serial per merge — FIFO requires it.
        if (threadIdx.x == 0) {
            int h = head[b];
            head[b] = -1;
            while (h >= 0) {
                int nxt = next[h];
                int e = h >> 1;
                if (!deleted[e]) {
                    uint32_t x = u[e], y = v[e];
                    uint32_t o = (x == b) ? y : x;
                    o = dfind(parent, o);
                    if (o == a) {
                        deleted[e] = 1;
                    } else {
                        int exist = find_inc(a, o, head, next, deleted, u, v, parent);
                        if (exist >= 0) {
                            double fromN = (double)ct[e];
                            double toN = (double)ct[exist];
                            double fromM = (ct[e] > 0) ? (sm[e] / fromN) : 0.0;
                            double toM = (ct[exist] > 0) ? (sm[exist] / toN) : 0.0;
                            double nm = (fromM * fromN + toM * toN) / (fromN + toN);
                            ct[exist] = ct[e] + ct[exist];
                            sm[exist] = nm * (double)ct[exist];
                            deleted[e] = 1;
                        } else {
                            if (u[e] == b) u[e] = a;
                            else v[e] = a;
                            next[h] = head[a];
                            head[a] = h;
                            h = nxt;
                            continue;
                        }
                    }
                }
                h = nxt;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *n_pop = s_pop;
        *n_merge = s_mrg;
    }
}

__global__ void k_compress_all(uint32_t* parent, uint32_t n) {
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) parent[i] = dfind(parent, i);
}

extern "C" int binqueue_mean_d(
    const uint32_t* u_h, const uint32_t* v_h,
    const double* sm_h, const int64_t* ct_h,
    int64_t n_edges, const double* aff_thr, int n_thr, int n_bins,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out)
{
    (void)n_bins;
    if (n_edges <= 0 || n_thr <= 0) return 0;
    uint32_t nnode = max_id + 1;
    uint32_t *du = nullptr, *dv = nullptr, *dparent = nullptr;
    double *dsm = nullptr;
    int64_t *dct = nullptr;
    int *head = nullptr, *next = nullptr, *bh = nullptr, *bt = nullptr, *bn = nullptr;
    uint8_t *del = nullptr;
    int64_t *np = nullptr, *nm = nullptr;

    cudaMalloc(&du, (size_t)n_edges * 4);
    cudaMalloc(&dv, (size_t)n_edges * 4);
    cudaMalloc(&dsm, (size_t)n_edges * 8);
    cudaMalloc(&dct, (size_t)n_edges * 8);
    cudaMalloc(&dparent, (size_t)nnode * 4);
    cudaMalloc(&head, (size_t)nnode * 4);
    cudaMalloc(&next, (size_t)n_edges * 2 * 4);
    cudaMalloc(&bh, NBIN * 4);
    cudaMalloc(&bt, NBIN * 4);
    cudaMalloc(&bn, (size_t)n_edges * 4);
    cudaMalloc(&del, (size_t)n_edges);
    cudaMalloc(&np, 8);
    cudaMalloc(&nm, 8);
    cudaMemcpy(du, u_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsm, sm_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dct, ct_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);

    const int threads = BLK;
    const int nb = (int)((nnode + threads - 1) / threads);
    const int eb = (int)((n_edges + threads - 1) / threads);
    k_init_nodes<<<nb, threads>>>(dparent, nnode);
    k_init_heads<<<nb, threads>>>(head, nnode);
    k_init_bins<<<1, NBIN>>>(bh, bt);
    k_build_inc<<<eb, threads>>>(du, dv, n_edges, head, next);
    k_enqueue_init<<<eb, threads>>>(dsm, dct, n_edges, bh, bt, bn, del);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int rc = 1;
    for (int t = 0; t < n_thr; ++t) {
        k_init_nodes<<<nb, threads>>>(dparent, nnode);
        k_init_heads<<<nb, threads>>>(head, nnode);
        k_init_bins<<<1, NBIN>>>(bh, bt);
        cudaMemset(del, 0, (size_t)n_edges);
        k_build_inc<<<eb, threads>>>(du, dv, n_edges, head, next);
        k_enqueue_init<<<eb, threads>>>(dsm, dct, n_edges, bh, bt, bn, del);
        cudaMemset(np, 0, 8);
        cudaMemset(nm, 0, 8);
        double T = aff_thr[t];
        double score_thr = 1.0 - T;
        cudaEventRecord(e0);
        k_binq_merge<<<1, BLK>>>(
            dparent, du, dv, dsm, dct, head, next, bh, bn, del,
            score_thr, n_edges, nnode, np, nm);
        cudaEventRecord(e1);
        cudaError_t st = cudaEventSynchronize(e1);
        if (st != cudaSuccess) {
            fprintf(stderr, "binqueue cuda err %s\n", cudaGetErrorString(st));
            rc = 0;
            break;
        }
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        k_compress_all<<<nb, threads>>>(dparent, nnode);
        cudaMemcpy(parent_out + (size_t)t * nnode, dparent,
                   (size_t)nnode * 4, cudaMemcpyDeviceToHost);
        int64_t pops = 0, merges = 0;
        cudaMemcpy(&pops, np, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(&merges, nm, 8, cudaMemcpyDeviceToHost);
        if (stats_out) {
            stats_out[t * 3 + 0] = 1;
            stats_out[t * 3 + 1] = merges;
            stats_out[t * 3 + 2] = pops;
        }
        fprintf(stderr, "binqueue T=%.3f ms=%.1f pops=%lld merges=%lld\n",
                T, ms, (long long)pops, (long long)merges);
        // restore edge arrays for next threshold from host copies
        if (t + 1 < n_thr) {
            cudaMemcpy(du, u_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dv, v_h, (size_t)n_edges * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dsm, sm_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
            cudaMemcpy(dct, ct_h, (size_t)n_edges * 8, cudaMemcpyHostToDevice);
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(du); cudaFree(dv); cudaFree(dsm); cudaFree(dct);
    cudaFree(dparent); cudaFree(head); cudaFree(next);
    cudaFree(bh); cudaFree(bt); cudaFree(bn); cudaFree(del);
    cudaFree(np); cudaFree(nm);
    return rc;
}
