// CPU exact heap: port of waterz IterativeRegionMerging + MeanAffinity + OneMinus.
// ScoreValue is float. Lower ID absorbs higher. No GPU.
#include <cstdint>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <limits>

static uint64_t pk(uint32_t a, uint32_t b) {
    if (a > b) std::swap(a, b);
    return (uint64_t(a) << 32) | uint64_t(b);
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
    const uint32_t nnode = max_id + 1;

    struct E {
        uint32_t u, v;
        float mean;
        uint32_t n;
        float score;
        bool deleted;
        bool stale;
    };
    std::vector<E> edges;
    edges.reserve((size_t)n_edges);
    std::vector<std::vector<int>> inc(nnode);
    std::unordered_map<uint64_t, int> lookup;
    lookup.reserve((size_t)n_edges * 2);

    for (int64_t i = 0; i < n_edges; ++i) {
        uint32_t u = u_in[i], v = v_in[i];
        if (u > v) std::swap(u, v);
        if (u == 0 || u == v) continue;
        E e;
        e.u = u;
        e.v = v;
        e.n = (uint32_t)count_in[i];
        e.mean = e.n ? (float)(sum_in[i] / (double)e.n) : 0.f;
        e.score = 1.f - e.mean;
        e.deleted = false;
        e.stale = false;
        int id = (int)edges.size();
        edges.push_back(e);
        inc[u].push_back(id);
        inc[v].push_back(id);
        lookup[pk(u, v)] = id;
    }

    std::vector<uint32_t> parent(nnode);
    for (uint32_t i = 0; i < nnode; ++i) parent[i] = i;

    auto find_e = [&](uint32_t a, uint32_t b) -> int {
        auto it = lookup.find(pk(a, b));
        return it == lookup.end() ? -1 : it->second;
    };

    auto remove_inc = [&](uint32_t n, int e) {
        auto& v = inc[n];
        auto it = std::find(v.begin(), v.end(), e);
        if (it != v.end()) v.erase(it);
    };

    using Item = std::pair<float, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    for (int i = 0; i < (int)edges.size(); ++i)
        heap.push({edges[i].score, i});

    auto score_edge = [&](int e) {
        edges[e].score = 1.f - edges[e].mean;
        heap.push({edges[e].score, e});
        return edges[e].score;
    };

    auto merge_regions = [&](int e) {
        uint32_t a = edges[e].u;
        uint32_t b = edges[e].v;
        // waterz: u <= v already; absorb b (higher) into a (lower)
        if (a > b) std::swap(a, b);
        parent[b] = a;
        lookup.erase(pk(a, b));
        edges[e].deleted = true;
        remove_inc(a, e);
        remove_inc(b, e);

        std::vector<int> neigh = inc[b];
        for (int ne : neigh) {
            if (ne == e || edges[ne].deleted) continue;
            uint32_t nb = (edges[ne].u == b) ? edges[ne].v : edges[ne].u;
            if (nb == a) {
                edges[ne].deleted = true;
                remove_inc(edges[ne].u, ne);
                remove_inc(edges[ne].v, ne);
                lookup.erase(pk(edges[ne].u, edges[ne].v));
                continue;
            }
            int ae = find_e(a, nb);
            if (ae < 0) {
                remove_inc(b, ne);
                lookup.erase(pk(edges[ne].u, edges[ne].v));
                edges[ne].u = std::min(a, nb);
                edges[ne].v = std::max(a, nb);
                inc[a].push_back(ne);
                lookup[pk(a, nb)] = ne;
            } else {
                // keep cheaper (lower score), merge stats into it
                int keep, drop;
                if (edges[ne].score > edges[ae].score) {
                    keep = ae;
                    drop = ne;
                } else {
                    keep = ne;
                    drop = ae;
                }
                float fn = (float)edges[drop].n;
                float tn = (float)edges[keep].n;
                edges[keep].mean = (edges[drop].mean * fn + edges[keep].mean * tn) / (fn + tn);
                edges[keep].n += edges[drop].n;
                edges[keep].stale = true;
                edges[drop].deleted = true;
                remove_inc(edges[drop].u, drop);
                remove_inc(edges[drop].v, drop);
                lookup.erase(pk(edges[drop].u, edges[drop].v));
                if (keep == ne) {
                    remove_inc(edges[ae].u, ae);
                    remove_inc(edges[ae].v, ae);
                    lookup.erase(pk(edges[ae].u, edges[ae].v));
                    remove_inc(b, ne);
                    edges[ne].u = std::min(a, nb);
                    edges[ne].v = std::max(a, nb);
                    inc[a].push_back(ne);
                    lookup[pk(a, nb)] = ne;
                } else {
                    remove_inc(b, ne);
                    lookup.erase(pk(a, nb));
                    lookup[pk(a, nb)] = ae;
                }
            }
        }
        inc[b].clear();
    };

    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });

    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        float thr = 1.f - (float)aff_thr[ti];
        while (!heap.empty()) {
            int next = heap.top().second;
            float score = edges[next].score;
            if (score >= thr) break;
            heap.pop();
            if (edges[next].deleted) continue;
            if (edges[next].stale) {
                edges[next].stale = false;
                score_edge(next);
                continue;
            }
            merge_regions(next);
        }
        uint32_t* dest = parent_out + (size_t)ti * (size_t)nnode;
        // compress
        for (uint32_t i = 0; i < nnode; ++i) {
            uint32_t r = i;
            while (parent[r] != r) r = parent[r];
            uint32_t x = i;
            while (parent[x] != r) {
                uint32_t n = parent[x];
                parent[x] = r;
                x = n;
            }
            dest[i] = r;
        }
    }
    return 1;
}
