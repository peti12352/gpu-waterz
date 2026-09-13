// N20 GPU RNN PREP. Do not compile into the product. Do not launch.
// Gate: scripts/n20_gpu_prep.py and N20_D3.go_gpu_rnn.
// CPU oracle: n20_rnn_s3_cpu in src/rac_agg.cpp (S3 contact-mean, Graph unite).
// Bruynooghe 1977 / RAC Garg 2021: RNN matching is exact for reducible average.
//
// This file is a launch recipe, not a benchmark. WATERZ_BATCH_RNN stays off.
// Do not reopen N19_H3. New ID would be N20_RNN_GPU after CPU four-T + D3 go.
//
// Kernel plan (mirrors CPU loop; one graph per T; no NNG filter):
//   1. Upload RAG (u,v,sum,count) once.
//   2. For each T independently (not nested):
//        compact live edges with score < 1-T
//        k_scan_best: per-node argmax mean (atomicMax on packed score|id)
//        k_rnn_match: keep pair if w(uv)=wmax(u)=wmax(v) and u<v
//        k_s3_unite: lock-free DSU + notifyEdgeMerge pooling on incident edges
//        repeat until no merges or rounds==cap
//   3. Download parents. Grade with t3_memsafe / four-T. Never 2.16 first.
//
// Why this is usually a no-go: P0w hit cap 200; Abboud: exact average-linkage
// is CC-hard (no poly-log exact PRAM). GPU only if D3 rounds are tiny.
//
// StarMerge (ParHAC 2.3 clustered-graph) is a different port, EV-dead at
// eps=0.08 (N20_D1 layer0 71% of merges). Do not implement k_starmarge here.

#include <cstdint>
#include <cstdio>

extern "C" int n20_rnn_gpu_prep_version(void) {
    std::fprintf(stderr, "n20_rnn_gpu_prep: compile-only stub; refuse launch\n");
    return 0;
}

extern "C" int n20_rnn_gpu_run(
    const uint32_t*, const uint32_t*, const double*, const int64_t*,
    int64_t, const double*, int, int64_t, uint32_t*, uint32_t)
{
    std::fprintf(stderr, "N20_RNN_GPU refuse launch (CPU-only campaign)\n");
    return 0;
}
