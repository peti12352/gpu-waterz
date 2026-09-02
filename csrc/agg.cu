// Locked AGG=parhac-ε 0.01. Host matching of (1+ε)-heavy waterz-mean edges
// plus S3-contract lives in src/rac_agg.cpp (parhac_agg_cpu). This TU is the
// device-side hook: edges already on GPU are copied out and agglomerated
// with the locked CPU kernel so segment() stays one algorithm.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" int parhac_agg_cpu(
    const uint32_t* u_in, const uint32_t* v_in,
    const double* sum_in, const int64_t* count_in,
    int64_t n_edges, const double* aff_thr, int n_thr,
    double eps, int additive,
    uint32_t* parent_out, uint32_t max_id, int64_t* stats_out);

extern "C" int agg_parhac_d(
    const uint32_t* u_d, const uint32_t* v_d,
    const double* sm_d, const int64_t* ct_d,
    int64_t n_edges, const double* aff_thr, int n_thr,
    double eps, uint32_t* parent_out, uint32_t max_id)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    std::vector<uint32_t> u((size_t)n_edges), v((size_t)n_edges);
    std::vector<double> sm((size_t)n_edges);
    std::vector<int64_t> ct((size_t)n_edges);
    cudaMemcpy(u.data(), u_d, (size_t)n_edges * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(), v_d, (size_t)n_edges * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(sm.data(), sm_d, (size_t)n_edges * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(ct.data(), ct_d, (size_t)n_edges * 8, cudaMemcpyDeviceToHost);
    std::vector<int64_t> stats((size_t)n_thr * 3, 0);
    return parhac_agg_cpu(
        u.data(), v.data(), sm.data(), ct.data(), n_edges,
        aff_thr, n_thr, eps, 0, parent_out, max_id, stats.data());
}
