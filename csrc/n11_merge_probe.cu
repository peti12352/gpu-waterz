// N11 C: serial merge-cost probe. One thread. Not an agglomerator.
// Probe 0: find two roots + min-hook + size add.
// Probe 1: that plus one CSR neighbor walk of the absorbed endpoint.
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <cstdio>

static inline __device__ uint32_t dfind(uint32_t* p, uint32_t x)
{
    uint32_t r = x;
    while (p[r] != r) r = p[r];
    while (p[x] != r) {
        uint32_t n = p[x];
        p[x] = r;
        x = n;
    }
    return r;
}

__global__ void k_iota(uint32_t* p, uint32_t* sz, uint32_t n)
{
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    p[i] = i;
    sz[i] = 1u;
}

__global__ void k_deg(const uint32_t* u, const uint32_t* v, int64_t n, int* deg)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&deg[u[i]], 1);
    atomicAdd(&deg[v[i]], 1);
}

__global__ void k_scatter_adj(
    const uint32_t* u, const uint32_t* v, int64_t n,
    const int* off, int* cur, int* adj)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int pu = atomicAdd(&cur[u[i]], 1);
    adj[off[u[i]] + pu] = (int)i;
    int pv = atomicAdd(&cur[v[i]], 1);
    adj[off[v[i]] + pv] = (int)i;
}

// One thread. `walk`: 0 = find+union only; 1 = + CSR walk of absorbed node.
__global__ void k_serial_merge(
    uint32_t* parent, uint32_t* sz,
    const uint32_t* u, const uint32_t* v, int64_t nedge,
    int target_merges, int walk,
    const int* off, const int* adj,
    int64_t* n_pop, int64_t* n_merge, unsigned long long* sink)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int64_t pops = 0, merges = 0;
    unsigned long long acc = 0;
    for (int64_t i = 0; i < nedge && merges < (int64_t)target_merges; ++i) {
        ++pops;
        uint32_t a = dfind(parent, u[i]);
        uint32_t b = dfind(parent, v[i]);
        if (a == b) continue;
        if (a > b) {
            uint32_t t = a;
            a = b;
            b = t;
        }
        parent[b] = a;
        sz[a] += sz[b];
        ++merges;
        if (walk && off && adj) {
            int o = off[b];
            int n = off[b + 1] - o;
            for (int k = 0; k < n; ++k) acc += (unsigned long long)(uint32_t)adj[o + k];
        }
    }
    *n_pop = pops;
    *n_merge = merges;
    if (sink) *sink = acc;
}

extern "C" int n11_merge_probe(
    const uint32_t* u_h, const uint32_t* v_h, int64_t nedge,
    uint32_t nnode, int target_merges, int walk,
    double* wall_ms, int64_t* n_pop, int64_t* n_merge)
{
    if (nedge <= 0 || nnode == 0 || target_merges <= 0) return -1;
    uint32_t *u = nullptr, *v = nullptr, *parent = nullptr, *sz = nullptr;
    int *deg = nullptr, *off = nullptr, *cur = nullptr, *adj = nullptr;
    int64_t *np = nullptr, *nm = nullptr;
    unsigned long long *sink = nullptr;
    cudaMalloc(&u, (size_t)nedge * 4);
    cudaMalloc(&v, (size_t)nedge * 4);
    cudaMalloc(&parent, (size_t)nnode * 4);
    cudaMalloc(&sz, (size_t)nnode * 4);
    cudaMalloc(&np, 8);
    cudaMalloc(&nm, 8);
    cudaMalloc(&sink, 8);
    cudaMemcpy(u, u_h, (size_t)nedge * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(v, v_h, (size_t)nedge * 4, cudaMemcpyHostToDevice);
    const int threads = 256;
    const int nb = (int)((nnode + threads - 1) / threads);
    const int eb = (int)((nedge + threads - 1) / threads);
    k_iota<<<nb, threads>>>(parent, sz, nnode);

    if (walk) {
        cudaMalloc(&deg, (size_t)nnode * 4);
        cudaMalloc(&off, (size_t)(nnode + 1) * 4);
        cudaMalloc(&cur, (size_t)nnode * 4);
        cudaMemset(deg, 0, (size_t)nnode * 4);
        k_deg<<<eb, threads>>>(u, v, nedge, deg);
        void* tmp = nullptr;
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, deg, off, (int)nnode);
        cudaMalloc(&tmp, tmp_bytes);
        cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, deg, off, (int)nnode);
        cudaFree(tmp);
        int last_deg = 0, last_off = 0;
        cudaMemcpy(&last_deg, deg + (nnode - 1), 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_off, off + (nnode - 1), 4, cudaMemcpyDeviceToHost);
        const int nadj = last_off + last_deg;
        cudaMemcpy(off + nnode, &nadj, 4, cudaMemcpyHostToDevice);
        cudaMalloc(&adj, (size_t)(nadj > 0 ? nadj : 1) * 4);
        cudaMemset(cur, 0, (size_t)nnode * 4);
        k_scatter_adj<<<eb, threads>>>(u, v, nedge, off, cur, adj);
        cudaFree(deg);
        cudaFree(cur);
    }

    cudaDeviceSynchronize();
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    k_serial_merge<<<1, 1>>>(
        parent, sz, u, v, nedge, target_merges, walk,
        off, adj, np, nm, sink);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    int64_t hp = 0, hm = 0;
    cudaMemcpy(&hp, np, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&hm, nm, 8, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (wall_ms) *wall_ms = (double)ms;
    if (n_pop) *n_pop = hp;
    if (n_merge) *n_merge = hm;

    cudaFree(u);
    cudaFree(v);
    cudaFree(parent);
    cudaFree(sz);
    cudaFree(np);
    cudaFree(nm);
    cudaFree(sink);
    if (off) cudaFree(off);
    if (adj) cudaFree(adj);
    if (err != cudaSuccess) {
        fprintf(stderr, "n11_merge_probe CUDA %s\n", cudaGetErrorString(err));
        return -2;
    }
    return 0;
}
