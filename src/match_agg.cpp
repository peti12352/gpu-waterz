// Matching-in-bucket live-mean: within each score band, merge a
// vertex-disjoint cheapest-outgoing matching, then S3-contract.
// Safer than union-all-in-bucket (X1), still parallel-round.
#include <cstdint>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <limits>

struct Edge {
    uint32_t u, v;
    double sum;
    int64_t n;
    double score;
};

extern "C" int match_agg_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    int n_buckets,
    uint32_t* parent_out,
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
        e.score = 1.0 - e.sum / (double)e.n;
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
            if (a == b) return false;
            if (a < b) parent[b] = a;
            else parent[a] = b;
            return true;
        };

        std::vector<Edge> cur = base;
        const double bw = 1.0 / (double)n_buckets;
        for (int b = 0; b < n_buckets; ++b) {
            const double lo = (double)b * bw;
            const double hi = (b + 1 == n_buckets) ? 1.0000001 : (double)(b + 1) * bw;
            if (lo >= tscore) break;
            for (int round = 0; round < 64; ++round) {
                struct Cand {
                    double sc;
                    uint32_t a, b;
                    bool ok;
                };
                std::vector<Cand> best(nnode);
                for (uint32_t i = 0; i < nnode; ++i) best[i].ok = false;
                for (const Edge& e : cur) {
                    if (e.score < lo || e.score >= hi || e.score >= tscore) continue;
                    uint32_t fu = find(e.u), fv = find(e.v);
                    if (fu == fv) continue;
                    if (fu > fv) std::swap(fu, fv);
                    auto consider = [&](uint32_t x, uint32_t y) {
                        if (!best[x].ok || e.score < best[x].sc ||
                            (e.score == best[x].sc && y < best[x].b)) {
                            best[x].ok = true;
                            best[x].sc = e.score;
                            best[x].a = x;
                            best[x].b = y;
                        }
                    };
                    consider(fu, fv);
                    consider(fv, fu);
                }
                int merges = 0;
                for (uint32_t i = 0; i < nnode; ++i) {
                    if (!best[i].ok) continue;
                    uint32_t j = best[i].b;
                    // mutual only: both endpoints agree
                    if (!best[j].ok || best[j].b != i) continue;
                    if (i < j && unite(i, j)) ++merges;
                }
                // contract
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
                        ne.score = 1.0 - ne.sum / (double)ne.n;
                        acc.emplace(key, ne);
                    } else {
                        it->second.sum += e.sum;
                        it->second.n += e.n;
                        it->second.score = 1.0 - it->second.sum / (double)it->second.n;
                    }
                }
                cur.clear();
                cur.reserve(acc.size());
                for (auto& kv : acc) cur.push_back(kv.second);
                if (merges == 0) break;
            }
        }
        for (uint32_t i = 0; i < nnode; ++i) parent[i] = find(i);
        uint32_t* dst = parent_out + (size_t)ti * nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = parent[i];
    }
    return 0;
}
