// Bucketed live-mean: process score buckets in order, S3-combine after each.
// B=1 is frozen CC. Large B approaches S4.
#include <cstdint>
#include <vector>
#include <algorithm>
#include <unordered_map>

struct Edge {
    uint32_t u, v;
    double sum;
    int64_t n;
    double score;  // 1 - mean
};

extern "C" int bucket_agg_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    int n_buckets,
    uint32_t* parent_out,  // [n_thr, max_id+1]
    uint32_t max_id)
{
    if (n_edges <= 0 || n_thr <= 0 || n_buckets < 1) return 0;
    const uint32_t nnode = max_id + 1;

    std::vector<Edge> base;
    base.reserve((size_t)n_edges);
    for (int64_t i = 0; i < n_edges; ++i) {
        uint32_t u = u_in[i], v = v_in[i];
        if (u > v) std::swap(u, v);
        if (u == 0 || u == v || count_in[i] < 1) continue;
        Edge e;
        e.u = u;
        e.v = v;
        e.sum = sum_in[i];
        e.n = count_in[i];
        double mean = e.sum / (double)e.n;
        e.score = 1.0 - mean;
        base.push_back(e);
    }

    for (int ti = 0; ti < n_thr; ++ti) {
        const double T = aff_thr[ti];
        const double tscore = 1.0 - T;
        std::vector<uint32_t> parent(nnode);
        for (uint32_t i = 0; i < nnode; ++i) parent[i] = i;
        auto find = [&](uint32_t x) {
            uint32_t r = x;
            while (parent[r] != r) r = parent[r];
            while (parent[x] != r) {
                uint32_t nxt = parent[x];
                parent[x] = r;
                x = nxt;
            }
            return r;
        };
        auto unite = [&](uint32_t a, uint32_t b) {
            a = find(a);
            b = find(b);
            if (a == b) return;
            if (a < b) parent[b] = a;
            else parent[a] = b;
        };

        std::vector<Edge> cur = base;
        const double bw = 1.0 / (double)n_buckets;
        for (int b = 0; b < n_buckets; ++b) {
            const double lo = (double)b * bw;
            const double hi = (b + 1 == n_buckets) ? 1.0000001 : (double)(b + 1) * bw;
            if (lo >= tscore) break;
            for (const Edge& e : cur) {
                if (e.score < lo || e.score >= hi) continue;
                if (e.score >= tscore) continue;
                uint32_t fu = find(e.u), fv = find(e.v);
                if (fu != fv) unite(fu, fv);
            }
            // contract: S3-combine remaining inter-component edges
            std::unordered_map<uint64_t, Edge> acc;
            acc.reserve(cur.size());
            for (const Edge& e : cur) {
                uint32_t fu = find(e.u), fv = find(e.v);
                if (fu == fv) continue;
                if (fu > fv) std::swap(fu, fv);
                uint64_t key = ((uint64_t)fu << 32) | (uint64_t)fv;
                auto it = acc.find(key);
                if (it == acc.end()) {
                    Edge ne = e;
                    ne.u = fu;
                    ne.v = fv;
                    ne.score = 1.0 - (ne.sum / (double)ne.n);
                    acc.emplace(key, ne);
                } else {
                    it->second.sum += e.sum;
                    it->second.n += e.n;
                    it->second.score = 1.0 - (it->second.sum / (double)it->second.n);
                }
            }
            cur.clear();
            cur.reserve(acc.size());
            for (auto& kv : acc) cur.push_back(kv.second);
            if (cur.empty()) break;
        }
        for (uint32_t i = 0; i < nnode; ++i) parent[i] = find(i);
        uint32_t* dst = parent_out + (size_t)ti * nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = parent[i];
    }
    return 0;
}
