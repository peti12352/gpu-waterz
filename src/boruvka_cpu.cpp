// CPU live-mean Borůvka (PLAN §3.4). No GPU.
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <limits>

extern "C" int boruvka_cpu(
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
    const uint32_t nnode = max_id + 1;

    struct Edge {
        uint32_t u, v;
        double sum;
        int64_t n;
        double mean;
    };

    for (int ti = 0; ti < n_thr; ++ti) {
        const double thr = aff_thr[ti];
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

        std::vector<Edge> cur;
        cur.reserve((size_t)n_edges);
        for (int64_t i = 0; i < n_edges; ++i) {
            uint32_t u = u_in[i], v = v_in[i];
            if (u > v) std::swap(u, v);
            if (u == 0 || u == v) continue;
            Edge e;
            e.u = u;
            e.v = v;
            e.sum = sum_in[i];
            e.n = count_in[i];
            e.mean = e.n ? e.sum / (double)e.n : 0;
            cur.push_back(e);
        }

        struct Cand {
            double sc;
            uint32_t a, b;
            bool ok;
        };
        std::vector<Cand> best(nnode);

        while (true) {
            for (uint32_t i = 0; i < nnode; ++i) {
                best[i].ok = false;
                best[i].sc = 0;
            }
            for (const Edge& e : cur) {
                uint32_t u = find(e.u), v = find(e.v);
                if (u == v || e.mean <= thr) continue;
                if (u > v) std::swap(u, v);
                double sc = 1.0 - e.mean;
                Cand c{sc, u, v, true};
                for (uint32_t node : {u, v}) {
                    if (!best[node].ok ||
                        c.sc < best[node].sc - 1e-18 ||
                        (c.sc <= best[node].sc + 1e-18 &&
                         (c.a < best[node].a ||
                          (c.a == best[node].a && c.b < best[node].b)))) {
                        best[node] = c;
                    }
                }
            }
            std::vector<std::pair<uint32_t, uint32_t>> pairs;
            std::unordered_map<uint64_t, char> seen;
            auto pk = [](uint32_t a, uint32_t b) -> uint64_t {
                return (uint64_t(a) << 32) | uint64_t(b);
            };
            for (uint32_t node = 1; node < nnode; ++node) {
                if (find(node) != node || !best[node].ok) continue;
                uint32_t a = find(best[node].a), b = find(best[node].b);
                if (a == b) continue;
                if (a > b) std::swap(a, b);
                uint64_t key = pk(a, b);
                if (seen.count(key)) continue;
                const Cand& ba = best[a];
                const Cand& bb = best[b];
                if (!ba.ok || !bb.ok) continue;
                uint32_t ba_a = find(ba.a), ba_b = find(ba.b);
                uint32_t bb_a = find(bb.a), bb_b = find(bb.b);
                if (ba_a > ba_b) std::swap(ba_a, ba_b);
                if (bb_a > bb_b) std::swap(bb_a, bb_b);
                bool a_picks = ba.ok && ba_a == a && ba_b == b;
                bool b_picks = bb.ok && bb_a == a && bb_b == b;
                if (!a_picks || !b_picks) continue;
                pairs.emplace_back(a, b);
                seen[key] = 1;
            }
            if (pairs.empty()) break;
            for (auto [lo, hi] : pairs) unite(lo, hi);

            std::unordered_map<uint64_t, Edge> nxt;
            nxt.reserve(cur.size());
            for (const Edge& e : cur) {
                uint32_t u = find(e.u), v = find(e.v);
                if (u == v) continue;
                if (u > v) std::swap(u, v);
                uint64_t key = (uint64_t(u) << 32) | uint64_t(v);
                auto it = nxt.find(key);
                if (it == nxt.end()) {
                    Edge ne{u, v, e.sum, e.n, 0};
                    ne.mean = ne.n ? ne.sum / (double)ne.n : 0;
                    nxt.emplace(key, ne);
                } else {
                    it->second.sum += e.sum;
                    it->second.n += e.n;
                    it->second.mean = it->second.n ? it->second.sum / (double)it->second.n : 0;
                }
            }
            cur.clear();
            cur.reserve(nxt.size());
            for (auto& kv : nxt) cur.push_back(kv.second);
        }

        uint32_t* dest = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dest[i] = find(i);
    }
    return 1;
}
