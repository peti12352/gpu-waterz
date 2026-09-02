// Faithful S4: vendored IterativeRegionMerging + MeanAffinity + OneMinus.
// Wrapper-only: add edges before stats maps, mute iostream, time stages.
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>

#include "backend/RegionGraph.hpp"
#include "backend/IterativeRegionMerging.hpp"
#include "backend/MergeFunctions.hpp"
#include "backend/Operators.hpp"
#include "backend/PriorityQueue.hpp"

typedef uint32_t NodeId;
typedef float ScoreValue;
typedef RegionGraph<NodeId> RegionGraphType;
typedef MeanAffinityProvider<RegionGraphType, ScoreValue> StatsType;
typedef OneMinus<MeanAffinity<RegionGraphType, ScoreValue>> ScoringType;
typedef IterativeRegionMerging<NodeId, ScoreValue, PriorityQueue> MergingType;

struct SilentVisitor {
    void onPop(RegionGraphType::EdgeIdType, ScoreValue) {}
    void onDeletedEdgeFound(RegionGraphType::EdgeIdType) {}
    void onStaleEdgeFound(RegionGraphType::EdgeIdType, ScoreValue, ScoreValue) {}
    void onMerge(NodeId, NodeId, NodeId, ScoreValue) {}
};

struct IdVol {
    std::vector<NodeId> v;
    std::size_t num_elements() const { return v.size(); }
    NodeId* data() { return v.data(); }
};

static double secs(std::chrono::steady_clock::time_point a,
                   std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

extern "C" int heap_agglomerate_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    uint32_t* parent_out,
    uint32_t max_id)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    const NodeId nnode = max_id + 1;
    auto t0 = std::chrono::steady_clock::now();

    RegionGraphType rg(nnode);
    rg.reserveEdges((std::size_t)n_edges);
    std::vector<RegionGraphType::EdgeIdType> eids;
    eids.reserve((std::size_t)n_edges);
    for (int64_t i = 0; i < n_edges; ++i) {
        NodeId u = u_in[i], v = v_in[i];
        if (u > v) std::swap(u, v);
        if (u == 0 || u == v) continue;
        eids.push_back(rg.addEdge(u, v));
    }
    auto t1 = std::chrono::steady_clock::now();

    StatsType stats(rg);
    std::size_t k = 0;
    for (int64_t i = 0; i < n_edges; ++i) {
        NodeId u = u_in[i], v = v_in[i];
        if (u > v) std::swap(u, v);
        if (u == 0 || u == v) continue;
        float mean = 0.f;
        int64_t n = count_in[i];
        if (n > 0) mean = (float)(sum_in[i] / (double)n);
        if (n < 1) n = 1;
        stats.setEdge(eids[k], mean, (size_t)n);
        ++k;
    }
    auto t2 = std::chrono::steady_clock::now();

    ScoringType scoring(rg, stats);
    MergingType merging(rg);
    SilentVisitor vis;
    auto* oldbuf = std::cout.rdbuf(nullptr);

    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });

    double merge_s = 0, extract_s = 0;
    IdVol vol;
    vol.v.resize(nnode);
    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        ScoreValue tscore = (ScoreValue)(1.0 - aff_thr[ti]);
        auto tm0 = std::chrono::steady_clock::now();
        merging.mergeUntil(scoring, stats, tscore, vis);
        auto tm1 = std::chrono::steady_clock::now();
        for (NodeId i = 0; i < nnode; ++i) vol.v[i] = i;
        merging.extractSegmentation(vol);
        uint32_t* dest = parent_out + (size_t)ti * (size_t)nnode;
        for (NodeId i = 0; i < nnode; ++i) dest[i] = vol.v[i];
        auto tm2 = std::chrono::steady_clock::now();
        merge_s += secs(tm0, tm1);
        extract_s += secs(tm1, tm2);
    }
    std::cout.rdbuf(oldbuf);
    auto t3 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "H0 heap build=%.3f setEdge=%.3f merge=%.3f extract=%.3f total=%.3f n_edges=%lld nnode=%u\n",
        secs(t0, t1), secs(t1, t2), merge_s, extract_s, secs(t0, t3),
        (long long)n_edges, nnode);
    return 1;
}
