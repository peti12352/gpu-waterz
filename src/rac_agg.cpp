// Exact RAC (reciprocal nearest-neighbour matching) + global-min fallback.
// Waterz means: score = 1 - sum/n. Incremental S3-contract (s1+s2)/(n1+n2).
#include <cstdint>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <cstdio>
#include <queue>
#include <random>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>

static int64_t g_red_viol = 0;
static int g_p0 = 0;

struct P0Inner {
    int32_t layer;
    int32_t ec;
    int32_t merges;
    int32_t us;
};
static std::vector<P0Inner> g_p0_inners;

struct Edge {
    uint32_t u, v;
    double sum;
    int64_t n;
    double score;
    bool alive;
};

static inline uint64_t pair_key(uint32_t a, uint32_t b) {
    return ((uint64_t)a << 32) | (uint64_t)b;
}

struct Best {
    double score;
    uint32_t nbr;
    uint32_t lo, hi;
    bool ok;
};

static inline bool better(double sc, uint32_t lo, uint32_t hi, const Best& b) {
    if (!b.ok) return true;
    if (sc < b.score) return true;
    if (sc > b.score) return false;
    if (lo < b.lo) return true;
    if (lo > b.lo) return false;
    return hi < b.hi;
}

struct Graph {
    uint32_t nnode;
    std::vector<Edge> edges;
    std::vector<std::vector<int>> adj;
    std::unordered_map<uint64_t, int> lookup;
    std::vector<uint32_t> parent;
    std::vector<int> live;

    explicit Graph(uint32_t n) : nnode(n), adj(n), parent(n) {
        for (uint32_t i = 0; i < n; ++i) parent[i] = i;
    }

    uint32_t find(uint32_t x) {
        uint32_t r = x;
        while (parent[r] != r) r = parent[r];
        while (parent[x] != r) {
            uint32_t nxt = parent[x];
            parent[x] = r;
            x = nxt;
        }
        return r;
    }

    void add_edge(uint32_t u, uint32_t v, double sum, int64_t n) {
        if (u > v) std::swap(u, v);
        if (u == 0 || u == v || n < 1) return;
        int id = (int)edges.size();
        Edge e;
        e.u = u;
        e.v = v;
        e.sum = sum;
        e.n = n;
        e.score = 1.0 - sum / (double)n;
        e.alive = true;
        edges.push_back(e);
        adj[u].push_back(id);
        adj[v].push_back(id);
        lookup[pair_key(u, v)] = id;
        live.push_back(id);
    }

    bool unite(uint32_t a, uint32_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        if (a > b) std::swap(a, b);
        // a absorbs b: move/combine b's edges onto a
        for (int eid : adj[b]) {
            Edge& e = edges[(size_t)eid];
            if (!e.alive) continue;
            uint32_t other = (e.u == b) ? e.v : e.u;
            other = find(other);
            lookup.erase(pair_key(e.u, e.v));
            if (other == a) {
                e.alive = false;
                continue;
            }
            uint32_t lo = a < other ? a : other;
            uint32_t hi = a < other ? other : a;
            auto it = lookup.find(pair_key(lo, hi));
            if (it == lookup.end()) {
                e.u = lo;
                e.v = hi;
                e.score = 1.0 - e.sum / (double)e.n;
                adj[a].push_back(eid);
                lookup[pair_key(lo, hi)] = eid;
            } else {
                Edge& keep = edges[(size_t)it->second];
                keep.sum += e.sum;
                keep.n += e.n;
                keep.score = 1.0 - keep.sum / (double)keep.n;
                e.alive = false;
            }
        }
        adj[b].clear();
        parent[b] = a;
        return true;
    }

    bool unite_keep(uint32_t keep, uint32_t drop) {
        keep = find(keep);
        drop = find(drop);
        if (keep == drop) return false;
        for (int eid : adj[drop]) {
            Edge& e = edges[(size_t)eid];
            if (!e.alive) continue;
            uint32_t other = (e.u == drop) ? e.v : e.u;
            other = find(other);
            lookup.erase(pair_key(e.u, e.v));
            if (other == keep) {
                e.alive = false;
                continue;
            }
            uint32_t lo = keep < other ? keep : other;
            uint32_t hi = keep < other ? other : keep;
            auto it = lookup.find(pair_key(lo, hi));
            if (it == lookup.end()) {
                e.u = lo;
                e.v = hi;
                e.score = 1.0 - e.sum / (double)e.n;
                adj[keep].push_back(eid);
                lookup[pair_key(lo, hi)] = eid;
            } else {
                Edge& keep_e = edges[(size_t)it->second];
                double m1 = keep_e.n > 0 ? keep_e.sum / (double)keep_e.n : 0.0;
                double m2 = e.n > 0 ? e.sum / (double)e.n : 0.0;
                keep_e.sum += e.sum;
                keep_e.n += e.n;
                keep_e.score = 1.0 - keep_e.sum / (double)keep_e.n;
                double mn = keep_e.n > 0 ? keep_e.sum / (double)keep_e.n : 0.0;
                if (mn > std::max(m1, m2) + 1e-15) ++g_red_viol;
                e.alive = false;
            }
        }
        adj[drop].clear();
        parent[drop] = keep;
        return true;
    }

    void compact_live() {
        int w = 0;
        for (int eid : live) {
            if (edges[(size_t)eid].alive) live[w++] = eid;
        }
        live.resize((size_t)w);
    }

    void nbr_roots(uint32_t x, std::vector<uint32_t>& out) {
        x = find(x);
        for (int eid : adj[x]) {
            Edge& e = edges[(size_t)eid];
            if (!e.alive) continue;
            uint32_t fu = find(e.u), fv = find(e.v);
            if (fu == fv) {
                e.alive = false;
                continue;
            }
            uint32_t o = (fu == x) ? fv : fu;
            if (o != x) out.push_back(o);
        }
    }
};

static void recompute_best_one(Graph& g, uint32_t x, double tscore, Best& b) {
    b.ok = false;
    x = g.find(x);
    for (int eid : g.adj[x]) {
        Edge& e = g.edges[(size_t)eid];
        if (!e.alive) continue;
        if (e.score >= tscore) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) {
            e.alive = false;
            continue;
        }
        uint32_t o = (fu == x) ? fv : (fv == x ? fu : 0xffffffffu);
        if (o == 0xffffffffu) continue;
        uint32_t lo = x < o ? x : o;
        uint32_t hi = x < o ? o : x;
        if (better(e.score, lo, hi, b)) {
            b.ok = true;
            b.score = e.score;
            b.nbr = o;
            b.lo = lo;
            b.hi = hi;
        }
    }
}

static void load_graph(
    Graph& g,
    const uint32_t* u_in, const uint32_t* v_in,
    const double* sum_in, const int64_t* count_in, int64_t n_edges)
{
    g.edges.reserve((size_t)n_edges);
    g.live.reserve((size_t)n_edges);
    g.lookup.reserve((size_t)n_edges);
    for (int64_t i = 0; i < n_edges; ++i)
        g.add_edge(u_in[i], v_in[i], sum_in[i], count_in[i]);
}

static void scan_best(
    Graph& g, double tscore, std::vector<Best>& best,
    int64_t* n_rnn, int64_t* n_onesided, double* smin)
{
    for (uint32_t i = 0; i < g.nnode; ++i) best[i].ok = false;
    double gmin = std::numeric_limits<double>::infinity();
    bool any = false;
    for (int eid : g.live) {
        Edge& e = g.edges[(size_t)eid];
        if (!e.alive) continue;
        if (e.score >= tscore) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) {
            e.alive = false;
            continue;
        }
        uint32_t lo = fu < fv ? fu : fv;
        uint32_t hi = fu < fv ? fv : fu;
        if (!any || e.score < gmin) gmin = e.score;
        any = true;
        if (better(e.score, lo, hi, best[fu])) {
            best[fu].ok = true;
            best[fu].score = e.score;
            best[fu].nbr = fv;
            best[fu].lo = lo;
            best[fu].hi = hi;
        }
        if (better(e.score, lo, hi, best[fv])) {
            best[fv].ok = true;
            best[fv].score = e.score;
            best[fv].nbr = fu;
            best[fv].lo = lo;
            best[fv].hi = hi;
        }
    }
    int64_t rnn = 0, one = 0;
    for (uint32_t i = 0; i < g.nnode; ++i) {
        if (!best[i].ok) continue;
        uint32_t j = best[i].nbr;
        if (best[j].ok && best[j].nbr == i) {
            if (i < j) ++rnn;
        } else {
            ++one;
        }
    }
    *n_rnn = rnn;
    *n_onesided = one;
    *smin = any ? gmin : -1.0;
}

extern "C" int rac_census_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    double T,
    int n_rounds,
    uint32_t max_id,
    int64_t* n_rnn_out,
    int64_t* n_onesided_out,
    double* smin_out,
    int64_t* n_edges_out)
{
    if (n_edges <= 0 || n_rounds < 0) return 0;
    Graph g(max_id + 1);
    load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
    const double tscore = 1.0 - T;
    std::vector<Best> best(g.nnode);
    for (int r = 0; r <= n_rounds; ++r) {
        g.compact_live();
        int64_t nr = 0, no = 0;
        double sm = -1.0;
        scan_best(g, tscore, best, &nr, &no, &sm);
        n_rnn_out[r] = nr;
        n_onesided_out[r] = no;
        smin_out[r] = sm;
        n_edges_out[r] = (int64_t)g.live.size();
        if (r == n_rounds) break;
        if (nr == 0) break;
        int merges = 0;
        for (uint32_t i = 0; i < g.nnode; ++i) {
            if (!best[i].ok) continue;
            uint32_t j = best[i].nbr;
            if (!best[j].ok || best[j].nbr != i) continue;
            if (i < j && g.unite(i, j)) ++merges;
        }
        if (merges == 0) break;
    }
    return 1;
}

extern "C" int rac_agg_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    const uint32_t nnode = max_id + 1;
    Graph g(nnode);
    load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });
    std::vector<Best> best(nnode);

    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti];
        const double tscore = 1.0 - T;
        int64_t rounds = 0, rnn_merges = 0, singleton_merges = 0;
        const int64_t max_rounds = 200000;
        bool need_scan = true;
        int64_t last_nr = 0;
        struct HItem {
            double score;
            uint32_t lo, hi;
            int eid;
            bool operator>(const HItem& o) const {
                if (score != o.score) return score > o.score;
                if (lo != o.lo) return lo > o.lo;
                return hi > o.hi;
            }
        };
        std::priority_queue<HItem, std::vector<HItem>, std::greater<HItem>> heap;
        bool heap_ok = false;
        auto rebuild_heap = [&]() {
            while (!heap.empty()) heap.pop();
            g.compact_live();
            for (int eid : g.live) {
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive || e.score >= tscore) continue;
                uint32_t fu = g.find(e.u), fv = g.find(e.v);
                if (fu == fv) {
                    e.alive = false;
                    continue;
                }
                if (fu > fv) std::swap(fu, fv);
                heap.push({e.score, fu, fv, eid});
            }
            heap_ok = true;
        };
        auto pop_min = [&](uint32_t* lo, uint32_t* hi) -> bool {
            if (!heap_ok) rebuild_heap();
            while (!heap.empty()) {
                HItem h = heap.top();
                heap.pop();
                Edge& e = g.edges[(size_t)h.eid];
                if (!e.alive || e.score >= tscore) continue;
                uint32_t fu = g.find(e.u), fv = g.find(e.v);
                if (fu == fv) {
                    e.alive = false;
                    continue;
                }
                if (fu > fv) std::swap(fu, fv);
                if (e.score != h.score || fu != h.lo || fv != h.hi) {
                    heap.push({e.score, fu, fv, h.eid});
                    continue;
                }
                *lo = fu;
                *hi = fv;
                return true;
            }
            return false;
        };
        while (rounds < max_rounds) {
            int64_t nr = 0, no = 0;
            double sm = -1.0;
            if (need_scan) {
                g.compact_live();
                scan_best(g, tscore, best, &nr, &no, &sm);
                need_scan = false;
            } else {
                for (uint32_t i = 0; i < nnode; ++i) {
                    if (g.parent[i] != i || !best[i].ok) continue;
                    uint32_t j = g.find(best[i].nbr);
                    if (best[j].ok && g.find(best[j].nbr) == i) {
                        if (i < j) ++nr;
                    } else {
                        ++no;
                    }
                }
                sm = 0.0;
            }
            last_nr = nr;
            int merges = 0;
            if (nr > 0) {
                for (uint32_t i = 0; i < nnode; ++i) {
                    if (g.parent[i] != i || !best[i].ok) continue;
                    uint32_t j = g.find(best[i].nbr);
                    if (!best[j].ok || g.find(best[j].nbr) != i) continue;
                    if (i < j && g.unite(i, j)) ++merges;
                }
                rnn_merges += merges;
                need_scan = true;
                heap_ok = false;
            } else {
                uint32_t bu = 0, bv = 0;
                if (!pop_min(&bu, &bv)) break;
                std::vector<uint32_t> aff;
                aff.reserve(64);
                g.nbr_roots(bu, aff);
                g.nbr_roots(bv, aff);
                aff.push_back(bu);
                aff.push_back(bv);
                if (g.unite(bu, bv)) {
                    merges = 1;
                    ++singleton_merges;
                }
                uint32_t keep = g.find(bu);
                aff.push_back(keep);
                std::sort(aff.begin(), aff.end());
                aff.erase(std::unique(aff.begin(), aff.end()), aff.end());
                for (uint32_t x : aff) {
                    x = g.find(x);
                    recompute_best_one(g, x, tscore, best[x]);
                }
                // partners of affected nodes
                std::vector<uint32_t> extra;
                for (uint32_t x : aff) {
                    x = g.find(x);
                    if (best[x].ok) extra.push_back(best[x].nbr);
                }
                for (uint32_t y : extra) {
                    y = g.find(y);
                    recompute_best_one(g, y, tscore, best[y]);
                }
                need_scan = false;
            }
            ++rounds;
            if ((rounds % 2000) == 0)
                std::fprintf(stderr, "Y1 T=%.2f progress rounds=%lld rnn=%lld sing=%lld nr=%lld\n",
                    T, (long long)rounds, (long long)rnn_merges,
                    (long long)singleton_merges, (long long)last_nr);
            if (merges == 0) break;
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = g.find(i);
        if (stats_out) {
            stats_out[ti * 3 + 0] = rounds;
            stats_out[ti * 3 + 1] = rnn_merges;
            stats_out[ti * 3 + 2] = singleton_merges;
        }
        std::fprintf(stderr,
            "Y1 T=%.2f rounds=%lld rnn_merges=%lld singleton=%lld live=%zu\n",
            T, (long long)rounds, (long long)rnn_merges,
            (long long)singleton_merges, g.live.size());
    }
    return 1;
}

extern "C" int s4_fast_cpu(
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
    Graph g(nnode);
    load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });
    struct HItem {
        double score;
        uint32_t lo, hi;
        int eid;
        bool operator>(const HItem& o) const {
            if (score != o.score) return score > o.score;
            if (lo != o.lo) return lo > o.lo;
            return hi > o.hi;
        }
    };
    std::priority_queue<HItem, std::vector<HItem>, std::greater<HItem>> heap;
    auto push_live = [&](double tscore) {
        while (!heap.empty()) heap.pop();
        g.compact_live();
        for (int eid : g.live) {
            Edge& e = g.edges[(size_t)eid];
            if (!e.alive) continue;
            uint32_t fu = g.find(e.u), fv = g.find(e.v);
            if (fu == fv) {
                e.alive = false;
                continue;
            }
            if (e.score >= tscore) continue;
            if (fu > fv) std::swap(fu, fv);
            heap.push({e.score, fu, fv, eid});
        }
    };
    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double tscore = 1.0 - aff_thr[ti];
        push_live(tscore);
        int64_t merges = 0;
        while (!heap.empty()) {
            HItem h = heap.top();
            heap.pop();
            Edge& e = g.edges[(size_t)h.eid];
            if (!e.alive || e.score >= tscore) continue;
            uint32_t fu = g.find(e.u), fv = g.find(e.v);
            if (fu == fv) {
                e.alive = false;
                continue;
            }
            if (fu > fv) std::swap(fu, fv);
            if (e.score != h.score || fu != h.lo || fv != h.hi) {
                heap.push({e.score, fu, fv, h.eid});
                continue;
            }
            if (!g.unite(fu, fv)) continue;
            ++merges;
            uint32_t keep = g.find(fu);
            for (int eid : g.adj[keep]) {
                Edge& ne = g.edges[(size_t)eid];
                if (!ne.alive || ne.score >= tscore) continue;
                uint32_t a = g.find(ne.u), b = g.find(ne.v);
                if (a == b) {
                    ne.alive = false;
                    continue;
                }
                if (a > b) std::swap(a, b);
                heap.push({ne.score, a, b, eid});
            }
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = g.find(i);
        std::fprintf(stderr, "H0 s4_fast T=%.2f merges=%lld\n", aff_thr[ti], (long long)merges);
    }
    return 1;
}

extern "C" int parhac_agg_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    double eps,
    int additive,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0 || eps <= 0.0) return 0;
    const uint32_t nnode = max_id + 1;
    Graph g(nnode);
    load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });

    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti];
        const double tscore = 1.0 - T;
        int64_t rounds = 0, nmerge = 0;
        const int64_t max_rounds = 200000;
        while (rounds < max_rounds) {
            g.compact_live();
            double wmax = -1.0;
            double smin = 1e300;
            std::vector<int> legal;
            legal.reserve(g.live.size());
            for (int eid : g.live) {
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive) continue;
                uint32_t fu = g.find(e.u), fv = g.find(e.v);
                if (fu == fv) {
                    e.alive = false;
                    continue;
                }
                if (e.score >= tscore) continue;
                legal.push_back(eid);
                double mean = 1.0 - e.score;
                if (mean > wmax) wmax = mean;
                if (e.score < smin) smin = e.score;
            }
            if (legal.empty()) break;
            std::vector<int> layer;
            layer.reserve(legal.size());
            if (additive) {
                for (int eid : legal)
                    if (g.edges[(size_t)eid].score <= smin + eps) layer.push_back(eid);
            } else {
                const double floor_mean = wmax / (1.0 + eps);
                for (int eid : legal)
                    if ((1.0 - g.edges[(size_t)eid].score) >= floor_mean)
                        layer.push_back(eid);
            }
            std::sort(layer.begin(), layer.end(), [&](int a, int b) {
                const Edge& ea = g.edges[(size_t)a];
                const Edge& eb = g.edges[(size_t)b];
                if (ea.score != eb.score) return ea.score < eb.score;
                uint32_t au = g.find(ea.u), av = g.find(ea.v);
                uint32_t bu = g.find(eb.u), bv = g.find(eb.v);
                if (au > av) std::swap(au, av);
                if (bu > bv) std::swap(bu, bv);
                if (au != bu) return au < bu;
                return av < bv;
            });
            std::vector<char> used(nnode, 0);
            int merges = 0;
            for (int eid : layer) {
                const Edge& e = g.edges[(size_t)eid];
                if (!e.alive) continue;
                uint32_t fu = g.find(e.u), fv = g.find(e.v);
                if (fu == fv || used[fu] || used[fv]) continue;
                used[fu] = used[fv] = 1;
                if (g.unite(fu, fv)) ++merges;
            }
            ++rounds;
            nmerge += merges;
            if (merges == 0) break;
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = g.find(i);
        if (stats_out) {
            stats_out[ti * 3 + 0] = rounds;
            stats_out[ti * 3 + 1] = nmerge;
            stats_out[ti * 3 + 2] = 0;
        }
        std::fprintf(stderr,
            "Y2 T=%.2f eps=%.4f additive=%d rounds=%lld merges=%lld live=%zu\n",
            T, eps, additive, (long long)rounds, (long long)nmerge, g.live.size());
    }
    return 1;
}

// P1 Alg. 1+2 (Dhulipala et al. 2022). Weight = waterz S3 mean, not UPGMA.
// Size = fragment cardinality. Stop: while Wmax > T; TL = max(T, Wmax/(1+eps)).
// RNG seed 0 (G5). Returns 1 on success.

static double graph_wmax(Graph& g) {
    double w = -1.0;
    for (int eid : g.live) {
        Edge& e = g.edges[(size_t)eid];
        if (!e.alive) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) {
            e.alive = false;
            continue;
        }
        double mean = 1.0 - e.score;
        if (mean > w) w = mean;
    }
    return w;
}

static bool paper_unite(Graph& g, std::vector<uint32_t>& sz, uint32_t keep, uint32_t drop) {
    keep = g.find(keep);
    drop = g.find(drop);
    if (keep == drop) return false;
    if (!g.unite_keep(keep, drop)) return false;
    sz[keep] += sz[drop];
    return true;
}

struct D0Count {
    int64_t empty_ec = 0;
    int64_t merge_outer = 0;
    int64_t freeze_outer = 0;
    int64_t still_ge_tl = 0;
    int64_t still_edges_sum = 0;
    int64_t freeze_reds = 0;
};

static int contract_layer(
    Graph& g, std::vector<uint32_t>& sz, double TL, double eps,
    std::mt19937_64& rng, int64_t* nmerge, int64_t* nouter, int64_t* ninner,
    D0Count* d0)
{
    const uint32_t nnode = g.nnode;
    const int64_t max_outer = 20000;
    const int64_t max_inner = 20000;
    int64_t outer = 0;
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    g.compact_live();
    std::vector<int> layer;
    layer.reserve(g.live.size());
    for (int eid : g.live) {
        Edge& e = g.edges[(size_t)eid];
        if (!e.alive) continue;
        if ((1.0 - e.score) < TL) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) {
            e.alive = false;
            continue;
        }
        layer.push_back(eid);
    }
    if (layer.empty()) return 1;
    std::vector<char> on(g.edges.size(), 0);
    for (int eid : layer) on[(size_t)eid] = 1;
    while (outer < max_outer && !layer.empty()) {
        std::vector<uint8_t> color(nnode, 0);
        std::vector<char> active(nnode, 0);
        std::vector<uint32_t> act;
        int lw = 0;
        for (int eid : layer) {
            Edge& e = g.edges[(size_t)eid];
            if (!e.alive || (1.0 - e.score) < TL) {
                on[(size_t)eid] = 0;
                continue;
            }
            uint32_t fu = g.find(e.u), fv = g.find(e.v);
            if (fu == fv) {
                e.alive = false;
                on[(size_t)eid] = 0;
                continue;
            }
            layer[lw++] = eid;
            if (!active[fu]) {
                active[fu] = 1;
                act.push_back(fu);
            }
            if (!active[fv]) {
                active[fv] = 1;
                act.push_back(fv);
            }
        }
        layer.resize((size_t)lw);
        if (layer.empty()) return 1;
        std::sort(act.begin(), act.end());
        std::vector<uint32_t> reds;
        for (uint32_t i : act) {
            if (g.find(i) != i) continue;
            color[i] = (rng() & 1ull) ? 1 : 2;
            if (color[i] == 1) reds.push_back(i);
        }
        std::vector<uint32_t> sz0(nnode, 0);
        for (uint32_t i : act) sz0[i] = sz[i];
        std::vector<char> frozen(nnode, 0);
        int64_t inner = 0;
        int64_t merges_before = nmerge ? *nmerge : 0;
        int64_t freeze_before = d0 ? d0->freeze_reds : 0;
        while (inner < max_inner) {
            auto t_inner0 = std::chrono::steady_clock::now();
            int layer_n = (int)layer.size();
            int64_t merges_inner0 = nmerge ? *nmerge : 0;
            struct Trip {
                uint32_t r;
                double pi;
                uint32_t b;
            };
            std::vector<std::pair<uint32_t, uint32_t>> br;
            std::vector<int> ec;
            std::vector<uint32_t> dirty;
            for (int eid : layer) {
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive) continue;
                if ((1.0 - e.score) < TL) continue;
                uint32_t fu = g.find(e.u), fv = g.find(e.v);
                if (fu == fv) {
                    e.alive = false;
                    continue;
                }
                uint8_t cu = color[fu], cv = color[fv];
                if (cu == 0 || cv == 0 || cu == cv) continue;
                uint32_t r = (cu == 1) ? fu : fv;
                uint32_t b = (cu == 1) ? fv : fu;
                if (color[r] != 1 || color[b] != 2) continue;
                if (frozen[r]) continue;
                if (sz[r] < sz[b]) continue;
                ec.push_back(eid);
                br.push_back({b, r});
            }
            if (ec.empty()) {
                if (inner == 0 && d0) ++d0->empty_ec;
                if (g_p0) {
                    auto t1 = std::chrono::steady_clock::now();
                    int us = (int)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t_inner0).count();
                    g_p0_inners.push_back({layer_n, 0, 0, us});
                }
                break;
            }
            std::sort(br.begin(), br.end());
            std::vector<Trip> T;
            T.reserve(br.size());
            size_t bi = 0;
            while (bi < br.size()) {
                uint32_t b = br[bi].first;
                size_t bj = bi;
                while (bj < br.size() && br[bj].first == b) ++bj;
                uint32_t r = br[bi + (size_t)(rng() % (bj - bi))].second;
                T.push_back({r, unif(rng), b});
                bi = bj;
            }
            std::sort(T.begin(), T.end(), [](const Trip& a, const Trip& b) {
                if (a.r != b.r) return a.r < b.r;
                if (a.pi != b.pi) return a.pi < b.pi;
                return a.b < b.b;
            });
            size_t i = 0;
            while (i < T.size()) {
                uint32_t r = T[i].r;
                size_t j = i;
                while (j < T.size() && T[j].r == r) ++j;
                uint64_t acc = 0;
                uint64_t cap = (uint64_t)std::llround((double)sz[r] * eps);
                if (eps > 0.0 && cap == 0) cap = 1;
                size_t take = j;
                for (size_t k = i; k < j; ++k) {
                    acc += sz[T[k].b];
                    if (acc > cap) {
                        take = k + 1;
                        break;
                    }
                }
                if (acc <= cap) take = j;
                uint32_t rr = g.find(r);
                for (size_t k = i; k < take; ++k) {
                    uint32_t bb = g.find(T[k].b);
                    if (paper_unite(g, sz, rr, bb)) {
                        ++(*nmerge);
                        rr = g.find(rr);
                        dirty.push_back(rr);
                    }
                }
                i = j;
            }
            for (uint32_t r : reds) {
                uint32_t fr = g.find(r);
                uint64_t s0 = sz0[r];
                if (s0 == 0) s0 = 1;
                if ((double)sz[fr] > (1.0 + eps) * (double)s0) {
                    if (!frozen[r] && d0) ++d0->freeze_reds;
                    frozen[r] = 1;
                }
            }
            int lw2 = 0;
            for (int eid : layer) {
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive || (1.0 - e.score) < TL) {
                    on[(size_t)eid] = 0;
                    continue;
                }
                uint32_t fu = g.find(e.u), fv = g.find(e.v);
                if (fu == fv) {
                    e.alive = false;
                    on[(size_t)eid] = 0;
                    continue;
                }
                layer[lw2++] = eid;
            }
            layer.resize((size_t)lw2);
            for (uint32_t d : dirty) {
                d = g.find(d);
                for (int eid : g.adj[d]) {
                    if ((size_t)eid >= on.size()) continue;
                    if (on[(size_t)eid]) continue;
                    Edge& e = g.edges[(size_t)eid];
                    if (!e.alive || (1.0 - e.score) < TL) continue;
                    uint32_t fu = g.find(e.u), fv = g.find(e.v);
                    if (fu == fv) {
                        e.alive = false;
                        continue;
                    }
                    on[(size_t)eid] = 1;
                    layer.push_back(eid);
                }
            }
            ++inner;
            if (ninner) ++(*ninner);
            if (g_p0) {
                auto t1 = std::chrono::steady_clock::now();
                int us = (int)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t_inner0).count();
                int mg = nmerge ? (int)(*nmerge - merges_inner0) : 0;
                g_p0_inners.push_back({(int)layer.size(), (int)ec.size(), mg, us});
            }
        }
        if (inner >= max_inner) {
            std::fprintf(stderr, "E3 inner cap TL=%.6f\n", TL);
            return 0;
        }
        if (d0) {
            if (nmerge && *nmerge > merges_before) ++d0->merge_outer;
            if (d0->freeze_reds > freeze_before) ++d0->freeze_outer;
            int64_t still = 0;
            for (int eid : layer) {
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive) continue;
                if ((1.0 - e.score) < TL) continue;
                uint32_t fu = g.find(e.u), fv = g.find(e.v);
                if (fu == fv) continue;
                ++still;
            }
            if (still > 0) ++d0->still_ge_tl;
            d0->still_edges_sum += still;
        }
        ++outer;
        if (nouter) ++(*nouter);
    }
    if (outer >= max_outer) {
        std::fprintf(stderr, "E3 outer cap TL=%.6f\n", TL);
        return 0;
    }
    return 1;
}

extern "C" int parhac_paper_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    double eps,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0 || eps < 0.0) return 0;
    g_p0 = std::getenv("WATERZ_P0") ? 1 : 0;
    g_red_viol = 0;
    g_p0_inners.clear();
    const uint32_t nnode = max_id + 1;
    Graph g(nnode);
    load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
    std::vector<uint32_t> sz(nnode, 1u);
    if (nnode > 0) sz[0] = 0;
    std::mt19937_64 rng(0);
    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });
    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti];
        int64_t layers = 0, nmerge = 0, nouter = 0, ninner = 0;
        D0Count d0;
        const int64_t max_layers = 10000;
        while (layers < max_layers) {
            g.compact_live();
            double wmax = graph_wmax(g);
            if (wmax <= T) break;
            double TL = wmax / (1.0 + eps);
            if (TL < T) TL = T;
            if (contract_layer(g, sz, TL, eps, rng, &nmerge, &nouter, &ninner, &d0) != 1)
                return 0;
            ++layers;
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = g.find(i);
        if (stats_out) {
            stats_out[ti * 3 + 0] = layers;
            stats_out[ti * 3 + 1] = nmerge;
            stats_out[ti * 3 + 2] = ninner;
        }
        std::fprintf(stderr,
            "E3 T=%.2f eps=%.4f layers=%lld outer=%lld inner=%lld merges=%lld live=%zu\n",
            T, eps, (long long)layers, (long long)nouter, (long long)ninner,
            (long long)nmerge, g.live.size());
        std::fprintf(stderr,
            "D0 T=%.2f empty_ec=%lld merge_outer=%lld freeze_outer=%lld "
            "still_ge_tl=%lld still_edges_sum=%lld freeze_reds=%lld "
            "outer=%lld inner=%lld\n",
            T, (long long)d0.empty_ec, (long long)d0.merge_outer,
            (long long)d0.freeze_outer, (long long)d0.still_ge_tl,
            (long long)d0.still_edges_sum, (long long)d0.freeze_reds,
            (long long)nouter, (long long)ninner);
    }
    std::fprintf(stderr, "P0b red_viol=%lld\n", (long long)g_red_viol);
    if (g_p0 && !g_p0_inners.empty()) {
        int64_t n = (int64_t)g_p0_inners.size();
        int64_t sl = 0, se = 0, sm = 0, su = 0;
        int ml = 0, me = 0, mm = 0, mu = 0;
        std::vector<int> us;
        us.reserve((size_t)n);
        for (const auto& r : g_p0_inners) {
            sl += r.layer; se += r.ec; sm += r.merges; su += r.us;
            if (r.layer > ml) ml = r.layer;
            if (r.ec > me) me = r.ec;
            if (r.merges > mm) mm = r.merges;
            if (r.us > mu) mu = r.us;
            us.push_back(r.us);
        }
        std::sort(us.begin(), us.end());
        int p50 = us[(size_t)n / 2];
        std::fprintf(stderr,
            "P0a ninner=%lld layer_mean=%.1f layer_max=%d ec_mean=%.1f ec_max=%d "
            "merge_mean=%.2f merge_max=%d us_mean=%.1f us_p50=%d us_max=%d\n",
            (long long)n, (double)sl / (double)n, ml, (double)se / (double)n, me,
            (double)sm / (double)n, mm, (double)su / (double)n, p50, mu);
        int b_lt1k = 0, b_lt10k = 0, b_lt100k = 0, b_ge = 0;
        for (const auto& r : g_p0_inners) {
            if (r.layer < 1000) ++b_lt1k;
            else if (r.layer < 10000) ++b_lt10k;
            else if (r.layer < 100000) ++b_lt100k;
            else ++b_ge;
        }
        std::fprintf(stderr, "P0a layer_hist lt1k=%d lt10k=%d lt100k=%d ge100k=%d\n",
            b_lt1k, b_lt10k, b_lt100k, b_ge);
        for (size_t i = 0; i < g_p0_inners.size(); ++i)
            std::fprintf(stderr, "P0x_INNER i=%zu layer=%d ec=%d merges=%d us=%d\n",
                i, g_p0_inners[i].layer, g_p0_inners[i].ec,
                g_p0_inners[i].merges, g_p0_inners[i].us);
    }
    return 1;
}

extern "C" int parhac_p0_count(void)
{
    return (int)g_p0_inners.size();
}

extern "C" int parhac_p0_copy_layers(int32_t* layer_out, int n)
{
    int m = (int)g_p0_inners.size();
    if (n < m) m = n;
    for (int i = 0; i < m; ++i) layer_out[i] = g_p0_inners[(size_t)i].layer;
    return m;
}

// Official ParHAC UPGMA (released ParHac.h), not paper Alg. 1 and not waterz mean.
// Weight = contact sum; score = sum / (|A||B|). Blue proposes, F&A cap (1+ε)|red|.
// Fused inner. Stop when UPGMA Wmax <= T. seed 0. Early exit if Wmax < 0.01.

static bool almost_eq(double x, double y, int ulp) {
    double ax = std::fabs(x), ay = std::fabs(y);
    return std::fabs(x - y) <= std::numeric_limits<double>::epsilon() * (ax + ay) * (double)ulp
        || std::fabs(x - y) < std::numeric_limits<double>::min();
}

static double upgma_w(const Edge& e, const std::vector<uint32_t>& sz, uint32_t fu, uint32_t fv) {
    double sa = (double)sz[fu], sb = (double)sz[fv];
    if (sa < 1.0) sa = 1.0;
    if (sb < 1.0) sb = 1.0;
    return e.sum / (sa * sb);
}

static double graph_wmax_upgma(Graph& g, const std::vector<uint32_t>& sz) {
    double w = 0.0;
    for (int eid : g.live) {
        Edge& e = g.edges[(size_t)eid];
        if (!e.alive) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) {
            e.alive = false;
            continue;
        }
        double ww = upgma_w(e, sz, fu, fv);
        if (ww > w) w = ww;
    }
    return w;
}

static int contract_upgma(
    Graph& g, std::vector<uint32_t>& sz, double lower, double max_w, double eps,
    std::mt19937_64& rng, int64_t* nmerge, int64_t* ninner)
{
    if (max_w < 0.01) return 1;
    const uint32_t nnode = g.nnode;
    const double one_plus = 1.0 + eps;
    const int64_t max_inner = 20000;
    g.compact_live();
    std::vector<char> active(nnode, 0);
    std::vector<uint32_t> alive;
    for (int eid : g.live) {
        Edge& e = g.edges[(size_t)eid];
        if (!e.alive) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) {
            e.alive = false;
            continue;
        }
        if (upgma_w(e, sz, fu, fv) >= lower) {
            if (!active[fu]) {
                active[fu] = 1;
                alive.push_back(fu);
            }
            if (!active[fv]) {
                active[fv] = 1;
                alive.push_back(fv);
            }
        }
    }
    std::sort(alive.begin(), alive.end());
    std::vector<uint64_t> cas(nnode, 0);
    int64_t inner = 0;
    while (!alive.empty() && inner < max_inner) {
        std::vector<uint8_t> color(nnode, 0);
        for (uint32_t u : alive) {
            if (g.find(u) != u) continue;
            color[u] = (rng() & 1ull) ? 2 : 1;  // 2=red, 1=blue (official kBlue=1 kRed=2)
        }
        for (uint32_t u : alive) cas[u] = sz[g.find(u)];
        std::vector<std::pair<uint32_t, uint32_t>> merges;
        for (uint32_t u : alive) {
            uint32_t fu = g.find(u);
            if (fu != u || color[fu] != 1) continue;
            double our_size = (double)sz[fu];
            std::vector<std::pair<double, uint32_t>> nbrs;
            for (int eid : g.adj[fu]) {
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive) continue;
                uint32_t fv = g.find(e.u == fu ? e.v : e.u);
                if (fv == fu) continue;
                nbrs.push_back({e.sum, fv});
            }
            std::sort(nbrs.begin(), nbrs.end());
            for (size_t ni = 0; ni < nbrs.size(); ++ni) {
                double wgh = nbrs[ni].first;
                uint32_t v = g.find(nbrs[ni].second);
                if (v == fu) continue;
                double real = wgh / (our_size * (double)std::max(sz[v], 1u));
                if (!(real >= lower || almost_eq(real, lower, 3))) continue;
                if (color[v] == 1) continue;
                uint64_t upper = (uint64_t)std::llround(one_plus * (double)sz[v]);
                uint64_t add = (uint64_t)sz[fu];
                uint64_t old = cas[v];
                if (old <= upper) {
                    cas[v] = old + add;
                    merges.push_back({v, fu});
                    break;
                }
            }
        }
        for (auto [r, b] : merges) {
            if (paper_unite(g, sz, r, b)) ++(*nmerge);
        }
        std::vector<uint32_t> next;
        std::vector<char> seen(nnode, 0);
        g.compact_live();
        for (int eid : g.live) {
            Edge& e = g.edges[(size_t)eid];
            if (!e.alive) continue;
            uint32_t fu = g.find(e.u), fv = g.find(e.v);
            if (fu == fv) {
                e.alive = false;
                continue;
            }
            if (upgma_w(e, sz, fu, fv) >= lower) {
                if (!seen[fu]) {
                    seen[fu] = 1;
                    next.push_back(fu);
                }
                if (!seen[fv]) {
                    seen[fv] = 1;
                    next.push_back(fv);
                }
            }
        }
        std::sort(next.begin(), next.end());
        alive.swap(next);
        ++inner;
        if (ninner) ++(*ninner);
    }
    if (inner >= max_inner) {
        std::fprintf(stderr, "E13 inner cap lower=%.6f\n", lower);
        return 0;
    }
    return 1;
}

extern "C" int parhac_upgma_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    double eps,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0 || eps < 0.0) return 0;
    const uint32_t nnode = max_id + 1;
    Graph g(nnode);
    load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
    std::vector<uint32_t> sz(nnode, 1u);
    if (nnode > 0) sz[0] = 0;
    std::mt19937_64 rng(0);
    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });
    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti];
        int64_t layers = 0, nmerge = 0, ninner = 0;
        const int64_t max_layers = 10000;
        while (layers < max_layers) {
            g.compact_live();
            double wmax = graph_wmax_upgma(g, sz);
            if (wmax <= T) break;
            if (wmax < 0.01) break;
            double lower = wmax / (1.0 + eps);
            if (lower < T) lower = T;
            if (contract_upgma(g, sz, lower, wmax, eps, rng, &nmerge, &ninner) != 1)
                return 0;
            ++layers;
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = g.find(i);
        if (stats_out) {
            stats_out[ti * 3 + 0] = layers;
            stats_out[ti * 3 + 1] = nmerge;
            stats_out[ti * 3 + 2] = ninner;
        }
        std::fprintf(stderr,
            "E13 T=%.2f eps=%.4f layers=%lld inner=%lld merges=%lld live=%zu\n",
            T, eps, (long long)layers, (long long)ninner, (long long)nmerge,
            g.live.size());
    }
    return 1;
}

// TeraHAC (Dhulipala et al. arXiv:2308.03578) control flow from
// google/graph-mining terahac.cc + terahac_internal.h, S3 contact-mean
// instead of AverageLinkageWeight. Stop: snapshot when graph Wmax <= T.
// Partition: SizeConstrainedAffinity. Merges: SubgraphHAC good-edges only.

static double tera_mean(const Edge& e) { return 1.0 - e.score; }

static void tera_wmax(Graph& g, std::vector<double>& wmax) {
    std::fill(wmax.begin(), wmax.end(), -1.0);
    for (int eid : g.live) {
        Edge& e = g.edges[(size_t)eid];
        if (!e.alive) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) {
            e.alive = false;
            continue;
        }
        double m = tera_mean(e);
        if (m > wmax[fu]) wmax[fu] = m;
        if (m > wmax[fv]) wmax[fv] = m;
    }
}

static bool tera_good(double w, double wu, double wv, double Mu, double Mv, double eps) {
    double Muv = w;
    if (Mu < Muv) Muv = Mu;
    if (Mv < Muv) Muv = Mv;
    if (!(Muv > 0.0)) return false;
    double mx = wu > wv ? wu : wv;
    return mx / Muv <= 1.0 + eps;
}

static uint32_t tera_best(
    Graph& g, uint32_t v, double* wout)
{
    v = g.find(v);
    uint32_t best = v;
    double bw = -1.0;
    uint32_t blo = 0xffffffffu, bhi = 0xffffffffu;
    for (int eid : g.adj[v]) {
        Edge& e = g.edges[(size_t)eid];
        if (!e.alive) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) {
            e.alive = false;
            continue;
        }
        uint32_t o = (fu == v) ? fv : fu;
        if (o == v) continue;
        double m = tera_mean(e);
        uint32_t lo = v < o ? v : o;
        uint32_t hi = v < o ? o : v;
        bool better = m > bw;
        if (!better && m == bw) {
            if (lo < blo || (lo == blo && hi < bhi)) better = true;
        }
        if (better) {
            bw = m;
            best = o;
            blo = lo;
            bhi = hi;
        }
    }
    if (wout) *wout = bw;
    return best;
}

static void tera_affinity(
    Graph& g, const std::vector<uint32_t>& active, size_t cap,
    std::vector<uint32_t>& cid)
{
    const uint32_t n = g.nnode;
    std::vector<uint32_t> uf(n), sz(n, 1u);
    for (uint32_t i = 0; i < n; ++i) uf[i] = i;
    auto find = [&](uint32_t x) {
        uint32_t r = x;
        while (uf[r] != r) r = uf[r];
        while (uf[x] != r) {
            uint32_t n2 = uf[x];
            uf[x] = r;
            x = n2;
        }
        return r;
    };
    auto unite = [&](uint32_t a, uint32_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (a > b) std::swap(a, b);
        uf[b] = a;
        sz[a] += sz[b];
    };
    std::vector<uint32_t> bn(active.size());
    std::vector<float> bw(active.size());
    for (size_t i = 0; i < active.size(); ++i) {
        double w = -1.0;
        uint32_t v = active[i];
        uint32_t n1 = tera_best(g, v, &w);
        bn[i] = n1;
        bw[i] = (float)w;
        if (n1 != v && w > 0.0) unite(v, n1);
    }
    std::vector<std::tuple<uint32_t, float, uint32_t>> rows;
    rows.reserve(active.size());
    for (size_t i = 0; i < active.size(); ++i) {
        uint32_t v = active[i];
        rows.push_back({find(v), -bw[i], v});
    }
    std::sort(rows.begin(), rows.end());
    std::vector<uint32_t> uf2(n), sz2(n, 1u);
    for (uint32_t i = 0; i < n; ++i) uf2[i] = i;
    auto find2 = [&](uint32_t x) {
        uint32_t r = x;
        while (uf2[r] != r) r = uf2[r];
        while (uf2[x] != r) {
            uint32_t n2 = uf2[x];
            uf2[x] = r;
            x = n2;
        }
        return r;
    };
    size_t i = 0;
    while (i < rows.size()) {
        uint32_t lab = std::get<0>(rows[i]);
        size_t j = i;
        while (j < rows.size() && std::get<0>(rows[j]) == lab) ++j;
        for (size_t k = i; k < j; ++k) {
            uint32_t v = std::get<2>(rows[k]);
            double w = -1.0;
            uint32_t nb = tera_best(g, v, &w);
            uint32_t cv = find2(v);
            uint32_t cw = find2(nb);
            size_t sv = sz2[cv], sw = sz2[cw];
            if (cv != cw && sv + sw <= cap) {
                if (cv > cw) std::swap(cv, cw);
                uf2[cw] = cv;
                sz2[cv] = sv + sw;
            }
        }
        i = j;
    }
    cid.assign(n, 0xffffffffu);
    for (uint32_t v : active) cid[v] = find2(v);
}

static int tera_subgraph(
    Graph& g, std::vector<uint32_t>& sz, std::vector<double>& Mv,
    const std::vector<uint32_t>& members, std::vector<char>& in_c,
    double T, double eps, int64_t* nmerge)
{
    if (members.size() < 2) return 0;
    std::vector<double> wmax(g.nnode, -1.0);
    auto fill_wmax = [&]() {
        for (uint32_t v0 : members) {
            uint32_t v = g.find(v0);
            if (!in_c[v]) continue;
            double m = -1.0;
            for (int eid : g.adj[v]) {
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive) continue;
                uint32_t fu = g.find(e.u), fv = g.find(e.v);
                if (fu == fv) {
                    e.alive = false;
                    continue;
                }
                double mm = tera_mean(e);
                if (mm > m) m = mm;
            }
            wmax[v] = m;
        }
    };
    int local = 0;
    const int max_match = 100000;
    for (int it = 0; it < max_match; ++it) {
        fill_wmax();
        struct Cand {
            double good;
            uint32_t lo, hi;
            double w;
        };
        std::vector<Cand> cands;
        std::vector<char> seen_e(g.edges.size(), 0);
        for (uint32_t v0 : members) {
            uint32_t v = g.find(v0);
            if (!in_c[v]) continue;
            for (int eid : g.adj[v]) {
                if ((size_t)eid >= seen_e.size() || seen_e[(size_t)eid]) continue;
                seen_e[(size_t)eid] = 1;
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive) continue;
                uint32_t fu = g.find(e.u), fv = g.find(e.v);
                if (fu == fv) {
                    e.alive = false;
                    continue;
                }
                if (!in_c[fu] || !in_c[fv]) continue;
                double w = tera_mean(e);
                if (w <= T) continue;
                if (!tera_good(w, wmax[fu], wmax[fv], Mv[fu], Mv[fv], eps)) continue;
                uint32_t lo = fu < fv ? fu : fv;
                uint32_t hi = fu < fv ? fv : fu;
                double Muv = std::min(std::min(Mv[fu], Mv[fv]), w);
                cands.push_back({std::max(wmax[fu], wmax[fv]) / Muv, lo, hi, w});
            }
        }
        if (cands.empty()) break;
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
            if (a.good != b.good) return a.good < b.good;
            if (a.lo != b.lo) return a.lo < b.lo;
            return a.hi < b.hi;
        });
        std::vector<char> taken(g.nnode, 0);
        int got = 0;
        for (const Cand& c : cands) {
            uint32_t a = g.find(c.lo), b = g.find(c.hi);
            if (a == b || taken[a] || taken[b]) continue;
            if (!in_c[a] || !in_c[b]) continue;
            uint32_t keep = a < b ? a : b;
            uint32_t drop = a < b ? b : a;
            if (!paper_unite(g, sz, keep, drop)) continue;
            keep = g.find(keep);
            Mv[keep] = std::min(std::min(Mv[a], Mv[b]), c.w);
            in_c[keep] = 1;
            taken[keep] = 1;
            ++(*nmerge);
            ++local;
            ++got;
        }
        if (got == 0) break;
    }
    return local;
}

extern "C" int terahac_mean_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    double eps,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out,
    int64_t size_constraint)
{
    if (n_edges <= 0 || n_thr <= 0 || eps <= 0.0) return 0;
    g_red_viol = 0;
    const uint32_t nnode = max_id + 1;
    Graph g(nnode);
    load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
    std::vector<uint32_t> sz(nnode, 1u);
    if (nnode > 0) sz[0] = 0;
    std::vector<double> Mv(nnode, std::numeric_limits<double>::infinity());
    int64_t cap = size_constraint;
    if (cap <= 0) cap = std::max((int64_t)nnode / 100, (int64_t)1000000);
    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });
    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti];
        int64_t nmerge = 0, nouter = 0;
        int stuck = 0;
        const int64_t max_outer = 10000;
        while (nouter < max_outer) {
            g.compact_live();
            double wmax = graph_wmax(g);
            if (wmax <= T) break;
            std::vector<uint32_t> active;
            std::vector<char> seen(nnode, 0);
            for (int eid : g.live) {
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive) continue;
                uint32_t fu = g.find(e.u), fv = g.find(e.v);
                if (fu == fv) {
                    e.alive = false;
                    continue;
                }
                if (tera_mean(e) <= T) continue;
                if (!seen[fu]) {
                    seen[fu] = 1;
                    active.push_back(fu);
                }
                if (!seen[fv]) {
                    seen[fv] = 1;
                    active.push_back(fv);
                }
            }
            std::sort(active.begin(), active.end());
            if (active.size() <= 1) break;
            std::vector<uint32_t> cid;
            tera_affinity(g, active, (size_t)cap, cid);
            std::vector<uint32_t> keys = cid;
            std::sort(keys.begin(), keys.end());
            keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
            int64_t before = nmerge;
            for (uint32_t c : keys) {
                if (c == 0xffffffffu) continue;
                std::vector<uint32_t> mem;
                std::vector<char> in_c(nnode, 0);
                for (uint32_t v : active) {
                    if (cid[v] == c) {
                        uint32_t r = g.find(v);
                        if (!in_c[r]) {
                            in_c[r] = 1;
                            mem.push_back(r);
                        }
                    }
                }
                tera_subgraph(g, sz, Mv, mem, in_c, T, eps, &nmerge);
            }
            ++nouter;
            if (nouter <= 5 || nouter % 10 == 0) {
                std::fprintf(stderr,
                    "T14 progress T=%.2f outer=%lld dmerge=%lld tot=%lld active=%zu\n",
                    T, (long long)nouter, (long long)(nmerge - before),
                    (long long)nmerge, active.size());
                std::fflush(stderr);
            }
            if (nmerge == before) {
                ++stuck;
                if (stuck >= 3) {
                    std::fprintf(stderr, "T14 stuck T=%.2f outer=%lld wmax=%.6f\n",
                        T, (long long)nouter, wmax);
                    break;
                }
            } else {
                stuck = 0;
            }
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = g.find(i);
        if (stats_out) {
            stats_out[ti * 3 + 0] = nouter;
            stats_out[ti * 3 + 1] = nmerge;
            stats_out[ti * 3 + 2] = g_red_viol;
        }
        std::fprintf(stderr,
            "T14 T=%.2f eps=%.4f cap=%lld outer=%lld merges=%lld red_viol=%lld live=%zu\n",
            T, eps, (long long)cap, (long long)nouter, (long long)nmerge,
            (long long)g_red_viol, g.live.size());
    }
    return 1;
}

// Wolf Alg. 2 / GASP AbsMax on signed RAG. Attractive = mean, repulsive = 1-mean.
// Cut: ignore attractive with mean <= T.

extern "C" int mutex_rag_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    const uint32_t nnode = max_id + 1;
    struct SE {
        uint32_t u, v;
        float aw;
        uint8_t attr;
        float mean;
    };
    std::vector<SE> es;
    es.reserve((size_t)n_edges * 2);
    for (int64_t i = 0; i < n_edges; ++i) {
        uint32_t u = u_in[i], v = v_in[i];
        if (u > v) std::swap(u, v);
        if (u == 0 || u == v) continue;
        int64_t n = count_in[i];
        if (n < 1) continue;
        double mean = sum_in[i] / (double)n;
        float am = (float)mean;
        float rm = (float)(1.0 - mean);
        es.push_back({u, v, am, 1, am});
        es.push_back({u, v, rm, 0, am});
    }
    std::sort(es.begin(), es.end(), [](const SE& a, const SE& b) {
        if (a.aw != b.aw) return a.aw > b.aw;
        if (a.attr != b.attr) return a.attr > b.attr;
        if (a.u != b.u) return a.u < b.u;
        return a.v < b.v;
    });
    auto run_one = [&](double T, uint32_t* dst) {
        std::vector<uint32_t> parent(nnode), rnk(nnode, 0);
        for (uint32_t i = 0; i < nnode; ++i) parent[i] = i;
        auto find = [&](uint32_t x) {
            uint32_t r = x;
            while (parent[r] != r) r = parent[r];
            while (parent[x] != r) {
                uint32_t n2 = parent[x];
                parent[x] = r;
                x = n2;
            }
            return r;
        };
        auto unite = [&](uint32_t a, uint32_t b) {
            a = find(a);
            b = find(b);
            if (a == b) return false;
            if (rnk[a] < rnk[b]) std::swap(a, b);
            parent[b] = a;
            if (rnk[a] == rnk[b]) ++rnk[a];
            return true;
        };
        std::vector<std::unordered_set<uint32_t>> dam(nnode);
        auto blocked = [&](uint32_t a, uint32_t b) {
            a = find(a);
            b = find(b);
            if (a == b) return false;
            return dam[a].count(b) || dam[b].count(a);
        };
        auto add_dam = [&](uint32_t a, uint32_t b) {
            a = find(a);
            b = find(b);
            if (a == b) return;
            dam[a].insert(b);
            dam[b].insert(a);
        };
        auto merge_dam = [&](uint32_t keep, uint32_t drop) {
            if (dam[drop].empty()) return;
            for (uint32_t x : dam[drop]) {
                dam[x].erase(drop);
                if (x != keep) {
                    dam[keep].insert(x);
                    dam[x].insert(keep);
                }
            }
            dam[drop].clear();
        };
        int64_t nmerge = 0, nmutex = 0;
        for (const SE& e : es) {
            uint32_t a = find(e.u), b = find(e.v);
            if (a == b) continue;
            if (e.attr) {
                if ((double)e.mean <= T) continue;
                if (blocked(a, b)) continue;
                if (unite(a, b)) {
                    uint32_t keep = find(a);
                    uint32_t drop = (keep == a) ? b : a;
                    merge_dam(keep, drop);
                    ++nmerge;
                }
            } else {
                add_dam(a, b);
                ++nmutex;
            }
        }
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = find(i);
        return std::pair<int64_t, int64_t>{nmerge, nmutex};
    };
    for (int t = 0; t < n_thr; ++t) {
        auto [nm, nx] = run_one(aff_thr[t], parent_out + (size_t)t * (size_t)nnode);
        if (stats_out) {
            stats_out[t * 3 + 0] = 1;
            stats_out[t * 3 + 1] = nm;
            stats_out[t * 3 + 2] = nx;
        }
        std::fprintf(stderr, "M16 T=%.2f merges=%lld mutex=%lld\n",
            aff_thr[t], (long long)nm, (long long)nx);
    }
    return 1;
}

extern "C" int mutex_rag_hop_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const uint8_t* aff,
    const uint32_t* seg,
    int64_t Z, int64_t Y, int64_t X,
    const int* hops, int n_hops,
    const double* aff_thr,
    int n_thr,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    const uint32_t nnode = max_id + 1;
    struct SE {
        uint32_t u, v;
        float aw;
        uint8_t attr;
        float mean;
    };
    std::vector<SE> es;
    es.reserve((size_t)n_edges * 2 + (size_t)Z * (size_t)Y * (size_t)X);
    for (int64_t i = 0; i < n_edges; ++i) {
        uint32_t u = u_in[i], v = v_in[i];
        if (u > v) std::swap(u, v);
        if (u == 0 || u == v) continue;
        int64_t n = count_in[i];
        if (n < 1) continue;
        double mean = sum_in[i] / (double)n;
        es.push_back({u, v, (float)mean, 1, (float)mean});
        es.push_back({u, v, (float)(1.0 - mean), 0, (float)mean});
    }
    const int64_t yx = Y * X;
    const int64_t vol = Z * Y * X;
    for (int hi = 0; hi < n_hops; ++hi) {
        int k = hops[hi];
        if (k < 2) continue;
        for (int64_t i = 0; i < vol; ++i) {
            uint32_t a = seg[i];
            if (a == 0) continue;
            int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
            // aff layout: c=0 -z, c=1 -y, c=2 -x at this voxel
            auto hop_axis = [&](int c, int64_t dz, int64_t dy, int64_t dx) {
                int64_t z2 = z + dz, y2 = y + dy, x2 = x + dx;
                if (z2 < 0 || z2 >= Z || y2 < 0 || y2 >= Y || x2 < 0 || x2 >= X)
                    return;
                int64_t j = z2 * yx + y2 * X + x2;
                uint32_t b = seg[j];
                if (b == 0 || b == a) return;
                float mn = 1.0f;
                for (int s = 1; s <= k; ++s) {
                    int64_t zs = z + (dz / k) * s, ys = y + (dy / k) * s, xs = x + (dx / k) * s;
                    int64_t q = zs * yx + ys * X + xs;
                    uint8_t av = aff[(int64_t)c * vol + q];
                    float af = (float)av / 255.0f;
                    if (af < mn) mn = af;
                }
                uint32_t lo = a < b ? a : b, hi2 = a < b ? b : a;
                es.push_back({lo, hi2, 1.0f - mn, 0, mn});
            };
            if (z >= k) hop_axis(0, -k, 0, 0);
            if (y >= k) hop_axis(1, 0, -k, 0);
            if (x >= k) hop_axis(2, 0, 0, -k);
        }
    }
    std::sort(es.begin(), es.end(), [](const SE& a, const SE& b) {
        if (a.aw != b.aw) return a.aw > b.aw;
        if (a.attr != b.attr) return a.attr > b.attr;
        if (a.u != b.u) return a.u < b.u;
        return a.v < b.v;
    });
    for (int t = 0; t < n_thr; ++t) {
        const double T = aff_thr[t];
        std::vector<uint32_t> parent(nnode), rnk(nnode, 0);
        for (uint32_t i = 0; i < nnode; ++i) parent[i] = i;
        auto find = [&](uint32_t x) {
            uint32_t r = x;
            while (parent[r] != r) r = parent[r];
            while (parent[x] != r) {
                uint32_t n2 = parent[x];
                parent[x] = r;
                x = n2;
            }
            return r;
        };
        auto unite = [&](uint32_t a, uint32_t b) {
            a = find(a);
            b = find(b);
            if (a == b) return false;
            if (rnk[a] < rnk[b]) std::swap(a, b);
            parent[b] = a;
            if (rnk[a] == rnk[b]) ++rnk[a];
            return true;
        };
        std::vector<std::unordered_set<uint32_t>> dam(nnode);
        auto blocked = [&](uint32_t a, uint32_t b) {
            a = find(a);
            b = find(b);
            if (a == b) return false;
            return dam[a].count(b) || dam[b].count(a);
        };
        auto add_dam = [&](uint32_t a, uint32_t b) {
            a = find(a);
            b = find(b);
            if (a == b) return;
            dam[a].insert(b);
            dam[b].insert(a);
        };
        auto merge_dam = [&](uint32_t keep, uint32_t drop) {
            if (dam[drop].empty()) return;
            for (uint32_t x : dam[drop]) {
                dam[x].erase(drop);
                if (x != keep) {
                    dam[keep].insert(x);
                    dam[x].insert(keep);
                }
            }
            dam[drop].clear();
        };
        int64_t nmerge = 0, nmutex = 0;
        for (const SE& e : es) {
            uint32_t a = find(e.u), b = find(e.v);
            if (a == b) continue;
            if (e.attr) {
                if ((double)e.mean <= T) continue;
                if (blocked(a, b)) continue;
                if (unite(a, b)) {
                    uint32_t keep = find(a);
                    uint32_t drop = (keep == a) ? b : a;
                    merge_dam(keep, drop);
                    ++nmerge;
                }
            } else {
                add_dam(a, b);
                ++nmutex;
            }
        }
        uint32_t* dst = parent_out + (size_t)t * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = find(i);
        if (stats_out) {
            stats_out[t * 3 + 0] = 1;
            stats_out[t * 3 + 1] = nmerge;
            stats_out[t * 3 + 2] = nmutex;
        }
        std::fprintf(stderr, "M16b T=%.2f merges=%lld mutex=%lld\n",
            T, (long long)nmerge, (long long)nmutex);
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Matching-until-empty in a live-mean band. Not union-all (X1), not one-shot (X1b).
// Each root picks highest-mean in-band neighbour; greedy matching; S3-contract.
// ---------------------------------------------------------------------------

static int64_t matching_round_band(Graph& g, double lo, double hi)
{
    const uint32_t n = g.nnode;
    std::vector<uint32_t> bn(n, 0xffffffffu);
    std::vector<double> bw(n, -1.0);
    g.compact_live();
    for (int eid : g.live) {
        Edge& e = g.edges[(size_t)eid];
        if (!e.alive) continue;
        double m = 1.0 - e.score;
        if (!(m > lo && m <= hi + 1e-15)) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) {
            e.alive = false;
            continue;
        }
        auto consider = [&](uint32_t a, uint32_t b) {
            bool bet = m > bw[a];
            if (!bet && m == bw[a] && bn[a] != 0xffffffffu) {
                uint32_t clo = a < bn[a] ? a : bn[a];
                uint32_t chi = a < bn[a] ? bn[a] : a;
                uint32_t nlo = a < b ? a : b;
                uint32_t nhi = a < b ? b : a;
                bet = nlo < clo || (nlo == clo && nhi < chi);
            }
            if (bet) {
                bw[a] = m;
                bn[a] = b;
            }
        };
        consider(fu, fv);
        consider(fv, fu);
    }
    std::vector<uint32_t> act;
    act.reserve(n / 8);
    for (uint32_t i = 1; i < n; ++i) {
        if (g.parent[i] == i && bn[i] != 0xffffffffu) act.push_back(i);
    }
    std::sort(act.begin(), act.end(), [&](uint32_t a, uint32_t b) {
        if (bw[a] != bw[b]) return bw[a] > bw[b];
        return a < b;
    });
    std::vector<char> used(n, 0);
    int64_t nm = 0;
    for (uint32_t a : act) {
        a = g.find(a);
        uint32_t b = bn[a];
        if (b == 0xffffffffu) continue;
        b = g.find(b);
        if (a == b || used[a] || used[b]) continue;
        used[a] = used[b] = 1;
        uint32_t keep = a < b ? a : b;
        uint32_t drop = a < b ? b : a;
        if (g.unite_keep(keep, drop)) ++nm;
    }
    return nm;
}

static int64_t match_until_empty(Graph& g, double lo, double hi, int max_rounds, int64_t* nmerge)
{
    int64_t rounds = 0, nm_tot = 0;
    while (rounds < max_rounds) {
        int64_t nm = matching_round_band(g, lo, hi);
        ++rounds;
        nm_tot += nm;
        if (nm == 0) break;
    }
    if (nmerge) *nmerge += nm_tot;
    return rounds;
}

extern "C" int p0_agg_shape_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    uint32_t max_id)
{
    if (n_edges <= 0) return 0;
    const uint32_t nnode = max_id + 1;

    // P0e / P0j on the raw arrays (no contract)
    int64_t mean_hist[100];
    std::memset(mean_hist, 0, sizeof(mean_hist));
    int64_t area_lt2 = 0, area_lt8 = 0, area_lt32 = 0, area_lt128 = 0, area_ge = 0;
    int64_t n_ge[8] = {0};
    const double cuts[8] = {0.99, 0.95, 0.90, 0.80, 0.70, 0.55, 0.50, 0.30};
    int64_t area_max = 0;
    for (int64_t i = 0; i < n_edges; ++i) {
        if (count_in[i] < 1) continue;
        double m = sum_in[i] / (double)count_in[i];
        int b = (int)(m * 100.0);
        if (b < 0) b = 0;
        if (b > 99) b = 99;
        mean_hist[b]++;
        for (int k = 0; k < 8; ++k) if (m > cuts[k]) n_ge[k]++;
        int64_t a = count_in[i];
        if (a > area_max) area_max = a;
        if (a < 2) ++area_lt2;
        else if (a < 8) ++area_lt8;
        else if (a < 32) ++area_lt32;
        else if (a < 128) ++area_lt128;
        else ++area_ge;
    }
    std::fprintf(stderr, "P0e n_edges=%lld\n", (long long)n_edges);
    for (int k = 0; k < 8; ++k)
        std::fprintf(stderr, "P0e n_mean_gt_%.2f=%lld\n", cuts[k], (long long)n_ge[k]);
    std::fprintf(stderr, "P0e hist01");
    for (int b = 0; b < 100; ++b) std::fprintf(stderr, " %lld", (long long)mean_hist[b]);
    std::fprintf(stderr, "\n");
    std::fprintf(stderr,
        "P0j area_lt2=%lld lt8=%lld lt32=%lld lt128=%lld ge128=%lld max=%lld\n",
        (long long)area_lt2, (long long)area_lt8, (long long)area_lt32,
        (long long)area_lt128, (long long)area_ge, (long long)area_max);

    // P0f live BinMatch Δ=0.10 then Δ=0.05 from 1.0 down to 0.3
    auto run_bands = [&](double delta, const char* tag) {
        Graph g(nnode);
        load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
        int64_t tot_r = 0, tot_m = 0;
        for (double hi = 1.0; hi > 0.3 + 1e-9; hi -= delta) {
            double lo = hi - delta;
            if (lo < 0.3) lo = 0.3;
            int64_t nm = 0;
            int64_t r = match_until_empty(g, lo, hi, 200, &nm);
            tot_r += r;
            tot_m += nm;
            std::fprintf(stderr,
                "P0%s band=(%.2f,%.2f] rounds=%lld merges=%lld live=%zu\n",
                tag, lo, hi, (long long)r, (long long)nm, g.live.size());
        }
        std::fprintf(stderr, "P0%s total_rounds=%lld total_merges=%lld\n",
            tag, (long long)tot_r, (long long)tot_m);
        return tot_r;
    };
    int64_t r_f10 = run_bands(0.10, "f10");
    int64_t r_f05 = -1, r_g = -1;
    if (r_f10 <= 30) {
        r_f05 = run_bands(0.05, "f05");
    } else {
        std::fprintf(stderr, "P0f05 SKIP f10=%lld > 30\n", (long long)r_f10);
    }

    // P0g: 256 live-mean bins only if coarse bands already look shallow
    if (r_f10 <= 30) {
        Graph g(nnode);
        load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
        const int N = 256;
        int64_t tot_r = 0, tot_m = 0;
        int hi_bin = N - 1;
        int lo_bin = (int)(0.3 * N);
        for (int b = hi_bin; b > lo_bin; --b) {
            double lo = (double)b / (double)N;
            double hi = (double)(b + 1) / (double)N;
            int64_t nm = 0;
            int64_t r = match_until_empty(g, lo, hi, 200, &nm);
            tot_r += r;
            tot_m += nm;
            if (nm > 0 || r > 1)
                std::fprintf(stderr,
                    "P0g bin=%d (%.4f,%.4f] rounds=%lld merges=%lld\n",
                    b, lo, hi, (long long)r, (long long)nm);
        }
        r_g = tot_r;
        std::fprintf(stderr, "P0g total_rounds=%lld total_merges=%lld\n",
            (long long)tot_r, (long long)tot_m);
    } else {
        std::fprintf(stderr, "P0g SKIP f10=%lld > 30\n", (long long)r_f10);
    }
    std::fprintf(stderr, "P0_BRANCH f10=%lld f05=%lld g256=%lld\n",
        (long long)r_f10, (long long)r_f05, (long long)r_g);

    // P0h: first 50k S4-style merges; log Wmax S3 / UPGMA / minface
    {
        Graph g(nnode);
        load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
        std::vector<uint32_t> sz(nnode, 1u);
        if (nnode) sz[0] = 0;
        std::vector<double> minf(g.edges.size());
        for (size_t i = 0; i < g.edges.size(); ++i)
            minf[i] = g.edges[i].n > 0 ? g.edges[i].sum / (double)g.edges[i].n : 0.0;
        using QE = std::pair<double, int>;
        std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
        for (int eid : g.live) pq.push({g.edges[(size_t)eid].score, eid});
        auto wmax_s3 = [&]() {
            double w = -1;
            for (int eid : g.live) {
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive) continue;
                if (g.find(e.u) == g.find(e.v)) continue;
                double m = 1.0 - e.score;
                if (m > w) w = m;
            }
            return w;
        };
        auto wmax_upgma = [&]() {
            double w = -1;
            for (int eid : g.live) {
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive) continue;
                uint32_t fu = g.find(e.u), fv = g.find(e.v);
                if (fu == fv) continue;
                double den = (double)sz[fu] * (double)sz[fv];
                if (den < 1) continue;
                double u = e.sum / den;
                if (u > w) w = u;
            }
            return w;
        };
        auto wmax_minf = [&]() {
            double w = -1;
            for (int eid : g.live) {
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive) continue;
                if (g.find(e.u) == g.find(e.v)) continue;
                if (minf[(size_t)eid] > w) w = minf[(size_t)eid];
            }
            return w;
        };
        std::fprintf(stderr, "P0h step=0 s3=%.6f upgma=%.6f minface=%.6f (median skipped: no per-face list)\n",
            wmax_s3(), wmax_upgma(), wmax_minf());
        int64_t nmerge = 0;
        const int64_t cap = 50000;
        while (nmerge < cap && !pq.empty()) {
            auto [sc, eid] = pq.top();
            pq.pop();
            Edge& e = g.edges[(size_t)eid];
            if (!e.alive) continue;
            uint32_t fu = g.find(e.u), fv = g.find(e.v);
            if (fu == fv) {
                e.alive = false;
                continue;
            }
            if (std::fabs(e.score - sc) > 1e-12) {
                pq.push({e.score, eid});
                continue;
            }
            uint32_t keep = fu < fv ? fu : fv;
            uint32_t drop = fu < fv ? fv : fu;
            double mf = minf[(size_t)eid];
            if (!g.unite_keep(keep, drop)) continue;
            sz[keep] += sz[drop];
            for (int neid : g.adj[keep]) {
                Edge& ne = g.edges[(size_t)neid];
                if (!ne.alive) continue;
                minf[(size_t)neid] = std::min(minf[(size_t)neid], mf);
                pq.push({ne.score, neid});
            }
            ++nmerge;
            if (nmerge % 10000 == 0) {
                g.compact_live();
                std::fprintf(stderr,
                    "P0h step=%lld s3=%.6f upgma=%.6f minface=%.6f live=%zu\n",
                    (long long)nmerge, wmax_s3(), wmax_upgma(), wmax_minf(), g.live.size());
            }
        }
        g.compact_live();
        double s3 = wmax_s3(), up = wmax_upgma(), mf = wmax_minf();
        std::fprintf(stderr,
            "P0h step=%lld s3=%.6f upgma=%.6f minface=%.6f live=%zu\n",
            (long long)nmerge, s3, up, mf, g.live.size());
        int drop_up = (s3 >= 0.5 && up > 0 && s3 / up >= 2.0) ? 1 : 0;
        int drop_mf = (s3 >= 0.5 && mf > 0 && s3 / mf >= 2.0) ? 1 : 0;
        std::fprintf(stderr, "P0h drop_upgma=%d drop_minface=%d (need s3>=0.5 and ratio>=2)\n",
            drop_up, drop_mf);
    }

    // P0i: 50k-edge BFS subgraph, S4 unique scores + max parent-chain
    {
        Graph g(nnode);
        load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
        std::vector<char> in_v(nnode, 0), in_e(g.edges.size(), 0);
        std::queue<uint32_t> q;
        uint32_t seed = 1;
        while (seed < nnode && g.adj[seed].empty()) ++seed;
        if (seed < nnode) {
            in_v[seed] = 1;
            q.push(seed);
        }
        int64_t ne = 0;
        const int64_t want = 50000;
        while (!q.empty() && ne < want) {
            uint32_t x = q.front();
            q.pop();
            for (int eid : g.adj[x]) {
                if (in_e[(size_t)eid]) continue;
                Edge& e = g.edges[(size_t)eid];
                if (!e.alive) continue;
                in_e[(size_t)eid] = 1;
                ++ne;
                uint32_t o = (e.u == x) ? e.v : e.u;
                if (!in_v[o]) {
                    in_v[o] = 1;
                    q.push(o);
                }
                if (ne >= want) break;
            }
        }
        Graph sg(nnode);
        for (size_t i = 0; i < g.edges.size(); ++i) {
            if (!in_e[i]) continue;
            Edge& e = g.edges[i];
            sg.add_edge(e.u, e.v, e.sum, e.n);
        }
        using QE = std::pair<double, int>;
        std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
        for (int eid : sg.live) pq.push({sg.edges[(size_t)eid].score, eid});
        std::unordered_set<uint64_t> uniq;
        int64_t nmerge = 0;
        while (!pq.empty()) {
            auto [sc, eid] = pq.top();
            pq.pop();
            Edge& e = sg.edges[(size_t)eid];
            if (!e.alive) continue;
            uint32_t fu = sg.find(e.u), fv = sg.find(e.v);
            if (fu == fv) {
                e.alive = false;
                continue;
            }
            if (std::fabs(e.score - sc) > 1e-12) {
                pq.push({e.score, eid});
                continue;
            }
            uniq.insert((uint64_t)(e.score * 1e9));
            uint32_t keep = fu < fv ? fu : fv, drop = fu < fv ? fv : fu;
            if (!sg.unite_keep(keep, drop)) continue;
            for (int neid : sg.adj[keep]) {
                if (sg.edges[(size_t)neid].alive)
                    pq.push({sg.edges[(size_t)neid].score, neid});
            }
            ++nmerge;
        }
        int64_t max_h = 0;
        for (uint32_t i = 1; i < nnode; ++i) {
            if (!in_v[i]) continue;
            int64_t h = 0;
            uint32_t x = i;
            while (sg.parent[x] != x && h < 1000000) {
                x = sg.parent[x];
                ++h;
            }
            if (h > max_h) max_h = h;
        }
        std::fprintf(stderr,
            "P0i sub_edges=%lld merges=%lld unique_scores=%zu max_parent_chain=%lld\n",
            (long long)ne, (long long)nmerge, uniq.size(), (long long)max_h);
    }
    return 1;
}

extern "C" int binmatch_mean_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    int n_bins,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0 || n_bins < 2) return 0;
    const uint32_t nnode = max_id + 1;
    Graph g(nnode);
    load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });
    int64_t tot_rounds = 0;
    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti];
        int64_t nmerge = 0, rounds = 0;
        int lo_bin = (int)(T * n_bins);
        for (int b = n_bins - 1; b > lo_bin; --b) {
            double lo = (double)b / (double)n_bins;
            double hi = (double)(b + 1) / (double)n_bins;
            if (lo < T) lo = T;
            int64_t nm = 0;
            rounds += match_until_empty(g, lo, hi, 200, &nm);
            nmerge += nm;
        }
        tot_rounds += rounds;
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = g.find(i);
        if (stats_out) {
            stats_out[ti * 3 + 0] = rounds;
            stats_out[ti * 3 + 1] = nmerge;
            stats_out[ti * 3 + 2] = tot_rounds;
        }
        std::fprintf(stderr,
            "B18 T=%.2f bins=%d rounds=%lld merges=%lld live=%zu\n",
            T, n_bins, (long long)rounds, (long long)nmerge, g.live.size());
    }
    return 1;
}

// A17 dual-weight ParHAC: layer key = UPGMA (mode=0) or minface (mode=1).
// Merge predicate: S3 > T. Contract: S3. Size: cardinality (mode_sz=0) or area (1).

static double graph_wmax_upgma_sz(Graph& g, const std::vector<uint32_t>& sz)
{
    double w = -1;
    for (int eid : g.live) {
        Edge& e = g.edges[(size_t)eid];
        if (!e.alive) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) continue;
        double den = (double)sz[fu] * (double)sz[fv];
        if (den < 1) continue;
        double u = e.sum / den;
        if (u > w) w = u;
    }
    return w;
}

static int64_t matching_round_pi(
    Graph& g, std::vector<uint32_t>& sz, const std::vector<double>& minf,
    double lo_pi, double hi_pi, double T, int pi_mode, int64_t* nrefused)
{
    const uint32_t n = g.nnode;
    std::vector<uint32_t> bn(n, 0xffffffffu);
    std::vector<double> bw(n, -1.0);
    g.compact_live();
    for (int eid : g.live) {
        Edge& e = g.edges[(size_t)eid];
        if (!e.alive) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) {
            e.alive = false;
            continue;
        }
        double s3 = 1.0 - e.score;
        if (!(s3 > T)) continue;
        double pi = s3;
        if (pi_mode == 0) {
            double den = (double)sz[fu] * (double)sz[fv];
            pi = den >= 1 ? e.sum / den : 0.0;
        } else if (pi_mode == 1) {
            pi = (size_t)eid < minf.size() ? minf[(size_t)eid] : s3;
        }
        if (!(pi > lo_pi && pi <= hi_pi + 1e-15)) continue;
        auto consider = [&](uint32_t a, uint32_t b) {
            if (pi > bw[a]) {
                bw[a] = pi;
                bn[a] = b;
            }
        };
        consider(fu, fv);
        consider(fv, fu);
    }
    std::vector<uint32_t> act;
    for (uint32_t i = 1; i < n; ++i)
        if (g.parent[i] == i && bn[i] != 0xffffffffu) act.push_back(i);
    std::sort(act.begin(), act.end(), [&](uint32_t a, uint32_t b) {
        if (bw[a] != bw[b]) return bw[a] > bw[b];
        return a < b;
    });
    std::vector<char> used(n, 0);
    int64_t nm = 0;
    for (uint32_t a : act) {
        a = g.find(a);
        uint32_t b = bn[a];
        if (b == 0xffffffffu) continue;
        b = g.find(b);
        if (a == b || used[a] || used[b]) continue;
        used[a] = used[b] = 1;
        uint32_t keep = a < b ? a : b, drop = a < b ? b : a;
        if (g.unite_keep(keep, drop)) {
            sz[keep] += sz[drop];
            ++nm;
        } else if (nrefused) {
            ++*nrefused;
        }
    }
    return nm;
}

extern "C" int dual_parhac_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    double eps,
    int pi_mode,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0 || eps <= 0) return 0;
    const uint32_t nnode = max_id + 1;
    Graph g(nnode);
    load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
    std::vector<uint32_t> sz(nnode, 1u);
    if (nnode) sz[0] = 0;
    std::vector<double> minf(g.edges.size());
    for (size_t i = 0; i < g.edges.size(); ++i)
        minf[i] = g.edges[i].n > 0 ? g.edges[i].sum / (double)g.edges[i].n : 0.0;
    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });
    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti];
        int64_t layers = 0, nmerge = 0, nref = 0;
        const int64_t max_layers = 10000;
        while (layers < max_layers) {
            g.compact_live();
            double w_s3 = graph_wmax(g);
            if (w_s3 <= T) break;
            double w_pi = (pi_mode == 0) ? graph_wmax_upgma_sz(g, sz) : w_s3;
            if (pi_mode == 1) {
                w_pi = -1;
                for (int eid : g.live) {
                    Edge& e = g.edges[(size_t)eid];
                    if (!e.alive) continue;
                    if (g.find(e.u) == g.find(e.v)) continue;
                    if (minf[(size_t)eid] > w_pi) w_pi = minf[(size_t)eid];
                }
            }
            if (w_pi <= 0) break;
            double TL = w_pi / (1.0 + eps);
            int64_t nm = 0;
            int inner = 0;
            while (inner < 200) {
                int64_t r = matching_round_pi(g, sz, minf, TL, w_pi + 1e-15, T, pi_mode, &nref);
                nm += r;
                ++inner;
                if (r == 0) break;
            }
            nmerge += nm;
            ++layers;
            if (nm == 0) {
                // no S3>T edge in this π-band: drop π clock by forcing a layer skip
                // if S3 still high, shrink eps-band by accepting slightly lower π
                if (TL <= T + 1e-15) break;
            }
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = g.find(i);
        if (stats_out) {
            stats_out[ti * 3 + 0] = layers;
            stats_out[ti * 3 + 1] = nmerge;
            stats_out[ti * 3 + 2] = nref;
        }
        std::fprintf(stderr,
            "A17 T=%.2f eps=%.4f pi=%d layers=%lld merges=%lld refused=%lld live=%zu\n",
            T, eps, pi_mode, (long long)layers, (long long)nmerge,
            (long long)nref, g.live.size());
    }
    return 1;
}

// S21: GASP mean + cannot-link on mean<0.5 (weight 0.5-mean). Not AbsMax.
extern "C" int gasp_mean_signed_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    const uint32_t nnode = max_id + 1;
    Graph g(nnode);
    load_graph(g, u_in, v_in, sum_in, count_in, n_edges);
    std::vector<std::unordered_set<uint32_t>> dam(nnode);
    for (int eid : g.live) {
        Edge& e = g.edges[(size_t)eid];
        double m = 1.0 - e.score;
        if (m >= 0.5) continue;
        dam[e.u].insert(e.v);
        dam[e.v].insert(e.u);
    }
    auto blocked = [&](uint32_t a, uint32_t b) {
        a = g.find(a);
        b = g.find(b);
        if (a == b) return false;
        return dam[a].count(b) || dam[b].count(a);
    };
    using QE = std::pair<double, int>;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
    for (int eid : g.live) {
        if (1.0 - g.edges[(size_t)eid].score >= 0.5)
            pq.push({g.edges[(size_t)eid].score, eid});
    }
    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });
    int64_t nmerge = 0;
    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti];
        while (!pq.empty()) {
            auto [sc, eid] = pq.top();
            Edge& e = g.edges[(size_t)eid];
            if (!e.alive) {
                pq.pop();
                continue;
            }
            uint32_t fu = g.find(e.u), fv = g.find(e.v);
            if (fu == fv) {
                e.alive = false;
                pq.pop();
                continue;
            }
            if (std::fabs(e.score - sc) > 1e-12) {
                pq.pop();
                pq.push({e.score, eid});
                continue;
            }
            double m = 1.0 - e.score;
            if (m <= T) break;
            pq.pop();
            if (blocked(fu, fv)) continue;
            uint32_t keep = fu < fv ? fu : fv, drop = fu < fv ? fv : fu;
            if (!g.unite_keep(keep, drop)) continue;
            if (!dam[drop].empty()) {
                for (uint32_t x : dam[drop]) {
                    dam[x].erase(drop);
                    if (x != keep) {
                        dam[keep].insert(x);
                        dam[x].insert(keep);
                    }
                }
                dam[drop].clear();
            }
            for (int neid : g.adj[keep]) {
                Edge& ne = g.edges[(size_t)neid];
                if (ne.alive) pq.push({ne.score, neid});
            }
            ++nmerge;
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = g.find(i);
        if (stats_out) {
            stats_out[ti * 3 + 0] = 1;
            stats_out[ti * 3 + 1] = nmerge;
            stats_out[ti * 3 + 2] = 0;
        }
        std::fprintf(stderr, "S21 T=%.2f merges=%lld live=%zu\n",
            T, (long long)nmerge, g.live.size());
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Size-aware Kruskal / constrained-CC (P0m/n/r + predicates A27..H30)
// pred: 0 area-CC  1 SDSL-step  2 SDSL-smooth  3 FH  4 SRM
//       5 X1+size  6 Soille  7 waterfall  8 HEM  9 relative-contact
// n_bins <= 1 => serial Kruskal; else batched frozen-size UF.
// ---------------------------------------------------------------------------

enum {
    PRED_AREA = 0, PRED_SDSL = 1, PRED_SDSL_SM = 2, PRED_FH = 3, PRED_SRM = 4,
    PRED_X1S = 5, PRED_SOILLE = 6, PRED_WF = 7, PRED_HEM = 8, PRED_REL = 9
};

struct DSU {
    std::vector<uint32_t> p;
    std::vector<int64_t> vox;
    std::vector<double> a, b; // Int / R / min_w, max_w
    explicit DSU(uint32_t n) : p(n), vox(n, 0), a(n, 0), b(n, 0) {
        for (uint32_t i = 0; i < n; ++i) p[i] = i;
    }
    uint32_t find(uint32_t x) {
        uint32_t r = x;
        while (p[r] != r) r = p[r];
        while (p[x] != r) {
            uint32_t n = p[x];
            p[x] = r;
            x = n;
        }
        return r;
    }
    bool unite(uint32_t u, uint32_t v) {
        u = find(u);
        v = find(v);
        if (u == v) return false;
        if (u > v) std::swap(u, v);
        p[v] = u;
        vox[u] += vox[v];
        return true;
    }
};

struct ERec {
    uint32_t u, v;
    double mean;
    int64_t area;
    double w;
};

static void fill_edges(
    const uint32_t* u, const uint32_t* v, const double* sm, const int64_t* ct,
    int64_t n, std::vector<ERec>& es)
{
    es.clear();
    es.reserve((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        if (ct[i] < 1 || u[i] == 0 || v[i] == 0 || u[i] == v[i]) continue;
        ERec e;
        e.u = u[i];
        e.v = v[i];
        e.area = ct[i];
        e.mean = sm[i] / (double)ct[i];
        e.w = 1.0 - e.mean;
        es.push_back(e);
    }
}

static int64_t n_components(DSU& d, uint32_t nnode)
{
    int64_t n = 0;
    for (uint32_t i = 1; i < nnode; ++i)
        if (d.vox[i] > 0 && d.find(i) == i) ++n;
    return n;
}

static int64_t max_cc_vox(DSU& d, uint32_t nnode)
{
    int64_t m = 0;
    for (uint32_t i = 1; i < nnode; ++i)
        if (d.find(i) == i && d.vox[i] > m) m = d.vox[i];
    return m;
}

extern "C" int p0_kruskal_shape_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    const int64_t* frag_sz,
    int64_t n_edges,
    uint32_t max_id,
    int64_t total_vox)
{
    if (n_edges <= 0) return 0;
    const uint32_t nnode = max_id + 1;
    std::vector<ERec> es;
    fill_edges(u_in, v_in, sum_in, count_in, n_edges, es);

    const double taus[] = {0.999, 0.99, 0.95, 0.80, 0.50, 0.20, 0.30, 0.40};
    const int ntau = 8;
    const int64_t areas[] = {1, 2, 4, 8, 16};
    const int na = 5;

    auto run_cc = [&](double tau, int64_t amin, DSU& d, int64_t* nkeep) {
        d = DSU(nnode);
        for (uint32_t i = 0; i < nnode; ++i) d.vox[i] = frag_sz ? frag_sz[i] : 1;
        if (nnode) d.vox[0] = 0;
        int64_t keep = 0;
        for (const ERec& e : es) {
            if (e.mean > tau && e.area >= amin) {
                ++keep;
                d.unite(e.u, e.v);
            }
        }
        if (nkeep) *nkeep = keep;
    };

    for (int ti = 0; ti < ntau; ++ti) {
        for (int ai = 0; ai < na; ++ai) {
            DSU d(nnode);
            int64_t keep = 0;
            run_cc(taus[ti], areas[ai], d, &keep);
            int64_t ncc = n_components(d, nnode);
            int64_t mx = max_cc_vox(d, nnode);
            double gfrac = total_vox > 0 ? (double)mx / (double)total_vox : 0;
            std::fprintf(stderr,
                "P0m tau=%.3f a=%lld keep=%lld ncc=%lld max_vox=%lld giant=%.6f\n",
                taus[ti], (long long)areas[ai], (long long)keep,
                (long long)ncc, (long long)mx, gfrac);
        }
    }

    // P0n degrees
    auto deg_of = [&](double tau, const char* tag) {
        std::vector<int64_t> deg(nnode, 0);
        DSU d(nnode);
        for (uint32_t i = 0; i < nnode; ++i) d.vox[i] = 1;
        if (nnode) d.vox[0] = 0;
        int64_t ne = 0;
        for (const ERec& e : es) {
            if (e.mean > tau) {
                ++deg[e.u];
                ++deg[e.v];
                ++ne;
                d.unite(e.u, e.v);
            }
        }
        std::vector<int64_t> ds;
        ds.reserve(nnode);
        int64_t nact = 0;
        for (uint32_t i = 1; i < nnode; ++i)
            if (deg[i] > 0) {
                ds.push_back(deg[i]);
                ++nact;
            }
        std::sort(ds.begin(), ds.end());
        int64_t mx = ds.empty() ? 0 : ds.back();
        int64_t p50 = ds.empty() ? 0 : ds[ds.size() / 2];
        int64_t p99 = ds.empty() ? 0 : ds[(size_t)(0.99 * (double)(ds.size() - 1))];
        int64_t giant_v = 0;
        std::vector<int64_t> csz(nnode, 0);
        for (uint32_t i = 1; i < nnode; ++i)
            if (deg[i] > 0) ++csz[d.find(i)];
        for (uint32_t i = 1; i < nnode; ++i)
            if (csz[i] > giant_v) giant_v = csz[i];
        double gvf = nact > 0 ? (double)giant_v / (double)nact : 0;
        std::fprintf(stderr,
            "P0n %s ne=%lld nact=%lld deg_max=%lld p50=%lld p99=%lld giant_vfrac=%.6f\n",
            tag, (long long)ne, (long long)nact, (long long)mx,
            (long long)p50, (long long)p99, gvf);
    };
    deg_of(0.99, "m099");
    deg_of(0.50, "m050");
    deg_of(0.20, "m020");

    // P0r residual after non-giant constrained CC
    for (int ti = 0; ti < ntau; ++ti) {
        for (int ai = 0; ai < na; ++ai) {
            DSU d(nnode);
            int64_t keep = 0;
            run_cc(taus[ti], areas[ai], d, &keep);
            int64_t mx = max_cc_vox(d, nnode);
            double gfrac = total_vox > 0 ? (double)mx / (double)total_vox : 0;
            if (gfrac > 0.02) continue;
            int64_t nsuper = n_components(d, nnode);
            std::unordered_map<uint64_t, std::pair<double, int64_t>> agg;
            agg.reserve((size_t)es.size() / 2);
            for (const ERec& e : es) {
                uint32_t fu = d.find(e.u), fv = d.find(e.v);
                if (fu == fv || fu == 0 || fv == 0) continue;
                if (fu > fv) std::swap(fu, fv);
                auto& pr = agg[pair_key(fu, fv)];
                pr.first += e.mean * (double)e.area;
                pr.second += e.area;
            }
            const double Ts[4] = {0.2, 0.3, 0.4, 0.5};
            for (int k = 0; k < 4; ++k) {
                int64_t nres = 0;
                for (auto& kv : agg) {
                    if (kv.second.second < 1) continue;
                    if (kv.second.first / (double)kv.second.second > Ts[k]) ++nres;
                }
                std::fprintf(stderr,
                    "P0r tau=%.3f a=%lld nsuper=%lld residual_T=%.1f nedge=%lld giant=%.6f\n",
                    taus[ti], (long long)areas[ai], (long long)nsuper,
                    Ts[k], (long long)nres, gfrac);
            }
        }
    }
    return 1;
}

static bool pred_ok(
    int pred, double T, double p0, double p1,
    double mean, int64_t area, double w,
    int64_t s1, int64_t s2,
    double a1, double a2, double b1, double b2)
{
    if (s1 < 1) s1 = 1;
    if (s2 < 1) s2 = 1;
    int64_t mn = s1 < s2 ? s1 : s2;
    if (pred == PRED_AREA)
        return mean > T && area >= (int64_t)(p0 + 0.5);
    if (pred == PRED_SDSL)
        return mean > T && mn < (int64_t)(p0 + 0.5);
    if (pred == PRED_REL) {
        double need = p0 * std::pow((double)mn, p1);
        return mean > T && (double)area + 1e-15 >= need;
    }
    if (pred == PRED_SDSL_SM) {
        double alpha = T + p1 * std::log(1.0 + (double)mn / (p0 > 0 ? p0 : 1.0));
        return mean >= alpha && mean > T;
    }
    if (pred == PRED_FH) {
        double t1 = a1 + p0 / (double)s1;
        double t2 = a2 + p0 / (double)s2;
        double mint = t1 < t2 ? t1 : t2;
        return w <= (1.0 - T) && w <= mint;
    }
    if (pred == PRED_SRM) {
        if (!(mean > T)) return false;
        double q = p0 > 0 ? p0 : 32.0;
        double diff = a1 - a2;
        double rhs = (1.0 / (2.0 * q)) * (1.0 / (double)s1 + 1.0 / (double)s2);
        return diff * diff < rhs;
    }
    if (pred == PRED_X1S)
        return mean > T && mn < (int64_t)(p0 + 0.5);
    if (pred == PRED_SOILLE) {
        if (w > (1.0 - T) + 1e-15) return false;
        double mx = a1;
        if (a2 > mx) mx = a2;
        if (w > mx) mx = w;
        double mn_w = b1;
        if (b2 < mn_w) mn_w = b2;
        if (w < mn_w) mn_w = w;
        return (mx - mn_w) <= p0 + 1e-15;
    }
    return mean > T;
}

static void apply_merge_meta(int pred, DSU& d, uint32_t keep, uint32_t drop, double w, double mean)
{
    if (pred == PRED_FH) {
        double nw = w;
        if (d.a[keep] > nw) nw = d.a[keep];
        if (d.a[drop] > nw) nw = d.a[drop];
        d.a[keep] = nw;
    } else if (pred == PRED_SRM) {
        double n1 = (double)(d.vox[keep] > d.vox[drop] ? d.vox[keep] - d.vox[drop] : 1);
        // vox already summed; recover old keep size
        int64_t oldk = d.vox[keep] - d.vox[drop];
        if (oldk < 1) oldk = 1;
        d.a[keep] = (d.a[keep] * (double)oldk + d.a[drop] * (double)d.vox[drop])
            / (double)d.vox[keep];
        (void)n1;
        (void)mean;
    } else if (pred == PRED_SOILLE) {
        double mx = d.a[keep];
        if (d.a[drop] > mx) mx = d.a[drop];
        if (w > mx) mx = w;
        double mn = d.b[keep];
        if (d.b[drop] < mn) mn = d.b[drop];
        if (w < mn) mn = w;
        d.a[keep] = mx;
        d.b[keep] = mn;
    }
}

static int64_t kruskal_one(
    std::vector<ERec>& es, DSU& d, uint32_t nnode,
    int pred, int n_bins, double T, double p0, double p1)
{
    int64_t nmerge = 0;
    if (pred == PRED_AREA || (pred == PRED_X1S && n_bins <= 1)) {
        for (const ERec& e : es) {
            uint32_t fu = d.find(e.u), fv = d.find(e.v);
            if (fu == fv) continue;
            if (pred_ok(pred, T, p0, p1, e.mean, e.area, e.w,
                        d.vox[fu], d.vox[fv], d.a[fu], d.a[fv], d.b[fu], d.b[fv])) {
                if (d.unite(e.u, e.v)) {
                    apply_merge_meta(pred, d, d.find(e.u), fv, e.w, e.mean);
                    ++nmerge;
                }
            }
        }
        return nmerge;
    }
    if (pred == PRED_X1S && n_bins > 1) {
        int B = n_bins;
        for (int b = B - 1; b >= 0; --b) {
            double lo = (double)b / (double)B;
            double hi = (double)(b + 1) / (double)B;
            if (hi <= T) continue;
            if (lo < T) lo = T;
            std::vector<int64_t> snap(nnode);
            for (uint32_t i = 0; i < nnode; ++i) {
                uint32_t r = d.find(i);
                snap[i] = d.vox[r];
            }
            for (const ERec& e : es) {
                if (!(e.mean > lo && e.mean <= hi + 1e-15)) continue;
                uint32_t fu = d.find(e.u), fv = d.find(e.v);
                if (fu == fv) continue;
                int64_t s1 = snap[fu], s2 = snap[fv];
                if (pred_ok(PRED_X1S, T, p0, p1, e.mean, e.area, e.w, s1, s2, 0, 0, 0, 0)) {
                    if (d.unite(e.u, e.v)) ++nmerge;
                }
            }
        }
        return nmerge;
    }

    // sort: FH/SRM/Soille by increasing w; SDSL by decreasing mean
    std::vector<int> ord((int)es.size());
    for (int i = 0; i < (int)es.size(); ++i) ord[i] = i;
    if (pred == PRED_FH || pred == PRED_SRM || pred == PRED_SOILLE) {
        std::sort(ord.begin(), ord.end(), [&](int i, int j) {
            if (es[i].w != es[j].w) return es[i].w < es[j].w;
            return i < j;
        });
    } else {
        std::sort(ord.begin(), ord.end(), [&](int i, int j) {
            if (es[i].mean != es[j].mean) return es[i].mean > es[j].mean;
            return i < j;
        });
    }

    if (n_bins <= 1) {
        for (int idx : ord) {
            const ERec& e = es[idx];
            uint32_t fu = d.find(e.u), fv = d.find(e.v);
            if (fu == fv) continue;
            if (pred_ok(pred, T, p0, p1, e.mean, e.area, e.w,
                        d.vox[fu], d.vox[fv], d.a[fu], d.a[fv], d.b[fu], d.b[fv])) {
                if (d.unite(e.u, e.v)) {
                    uint32_t k = d.find(e.u);
                    apply_merge_meta(pred, d, k, (k == fu ? fv : fu), e.w, e.mean);
                    ++nmerge;
                }
            }
        }
        return nmerge;
    }

    // batched: B mean-bands, freeze sizes/meta at band start, union-all
    int B = n_bins;
    for (int b = B - 1; b >= 0; --b) {
        double lo = (double)b / (double)B;
        double hi = (double)(b + 1) / (double)B;
        if (hi <= T) continue;
        if (lo < T) lo = T;
        std::vector<int64_t> snap_s(nnode);
        std::vector<double> snap_a(nnode), snap_b(nnode);
        for (uint32_t i = 0; i < nnode; ++i) {
            uint32_t r = d.find(i);
            snap_s[i] = d.vox[r];
            snap_a[i] = d.a[r];
            snap_b[i] = d.b[r];
        }
        for (int idx : ord) {
            const ERec& e = es[idx];
            if (!(e.mean > lo && e.mean <= hi + 1e-15)) continue;
            uint32_t fu = d.find(e.u), fv = d.find(e.v);
            if (fu == fv) continue;
            if (pred_ok(pred, T, p0, p1, e.mean, e.area, e.w,
                        snap_s[fu], snap_s[fv], snap_a[fu], snap_a[fv],
                        snap_b[fu], snap_b[fv])) {
                if (d.unite(e.u, e.v)) {
                    uint32_t k = d.find(e.u);
                    apply_merge_meta(pred, d, k, (k == fu ? fv : fu), e.w, e.mean);
                    ++nmerge;
                }
            }
        }
    }
    return nmerge;
}

static int64_t waterfall_one(std::vector<ERec>& es, DSU& d, uint32_t nnode, double T)
{
    int64_t tot = 0;
    for (int r = 0; r < 30; ++r) {
        std::vector<double> best(nnode, -1.0);
        for (const ERec& e : es) {
            if (!(e.mean > T)) continue;
            uint32_t fu = d.find(e.u), fv = d.find(e.v);
            if (fu == fv) continue;
            if (e.mean > best[fu]) best[fu] = e.mean;
            if (e.mean > best[fv]) best[fv] = e.mean;
        }
        int64_t nm = 0;
        for (const ERec& e : es) {
            if (!(e.mean > T)) continue;
            uint32_t fu = d.find(e.u), fv = d.find(e.v);
            if (fu == fv) continue;
            if (std::fabs(e.mean - best[fu]) < 1e-12 || std::fabs(e.mean - best[fv]) < 1e-12) {
                if (d.unite(e.u, e.v)) ++nm;
            }
        }
        tot += nm;
        if (nm == 0) break;
    }
    return tot;
}

static int64_t hem_one(std::vector<ERec>& es, DSU& d, uint32_t nnode, double T)
{
    int64_t tot = 0;
    for (int lev = 0; lev < 20; ++lev) {
        std::unordered_map<uint64_t, std::pair<double, int64_t>> agg;
        agg.reserve(es.size() / 2);
        for (const ERec& e : es) {
            uint32_t fu = d.find(e.u), fv = d.find(e.v);
            if (fu == fv || fu == 0 || fv == 0) continue;
            uint32_t lo = fu < fv ? fu : fv, hi = fu < fv ? fv : fu;
            auto& pr = agg[pair_key(lo, hi)];
            pr.first += e.mean * (double)e.area;
            pr.second += e.area;
        }
        std::vector<uint32_t> bn(nnode, 0xffffffffu);
        std::vector<double> bw(nnode, -1.0);
        for (auto& kv : agg) {
            if (kv.second.second < 1) continue;
            double m = kv.second.first / (double)kv.second.second;
            if (!(m > T)) continue;
            uint32_t a = (uint32_t)(kv.first >> 32);
            uint32_t b = (uint32_t)(kv.first & 0xffffffffu);
            a = d.find(a);
            b = d.find(b);
            if (a == b) continue;
            if (m > bw[a]) {
                bw[a] = m;
                bn[a] = b;
            }
            if (m > bw[b]) {
                bw[b] = m;
                bn[b] = a;
            }
        }
        std::vector<uint32_t> act;
        for (uint32_t i = 1; i < nnode; ++i)
            if (d.p[i] == i && bn[i] != 0xffffffffu) act.push_back(i);
        std::sort(act.begin(), act.end(), [&](uint32_t x, uint32_t y) {
            if (bw[x] != bw[y]) return bw[x] > bw[y];
            return x < y;
        });
        std::vector<char> used(nnode, 0);
        int64_t nm = 0;
        for (uint32_t a : act) {
            a = d.find(a);
            uint32_t b = bn[a];
            if (b == 0xffffffffu) continue;
            b = d.find(b);
            if (a == b || used[a] || used[b]) continue;
            // size-doubling: refuse if both already large vs this level
            int64_t cap = (int64_t)1 << (lev + 6);
            if (d.vox[a] > cap && d.vox[b] > cap) continue;
            used[a] = used[b] = 1;
            if (d.unite(a, b)) ++nm;
        }
        tot += nm;
        if (nm == 0) break;
    }
    return tot;
}

extern "C" int kruskal_pred_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    const int64_t* frag_sz,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    int pred,
    int n_bins,
    double p0,
    double p1,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    const uint32_t nnode = max_id + 1;
    std::vector<ERec> es;
    fill_edges(u_in, v_in, sum_in, count_in, n_edges, es);

    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });

    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti];
        DSU d(nnode);
        for (uint32_t i = 0; i < nnode; ++i) d.vox[i] = frag_sz ? frag_sz[i] : 1;
        if (nnode) d.vox[0] = 0;
        if (pred == PRED_SRM) {
            for (const ERec& e : es) {
                if (e.mean > d.a[e.u]) d.a[e.u] = e.mean;
                if (e.mean > d.a[e.v]) d.a[e.v] = e.mean;
            }
        }
        if (pred == PRED_SOILLE) {
            for (uint32_t i = 0; i < nnode; ++i) {
                d.a[i] = 0.0;
                d.b[i] = 1.0;
            }
        }
        int64_t nmerge = 0;
        int64_t rounds = (n_bins <= 1) ? 1 : n_bins;
        if (pred == PRED_WF) {
            nmerge = waterfall_one(es, d, nnode, T);
            rounds = 30;
        } else if (pred == PRED_HEM) {
            nmerge = hem_one(es, d, nnode, T);
            rounds = 20;
        } else {
            nmerge = kruskal_one(es, d, nnode, pred, n_bins, T, p0, p1);
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = d.find(i);
        int64_t ncc = n_components(d, nnode);
        int64_t mx = max_cc_vox(d, nnode);
        if (stats_out) {
            stats_out[ti * 3 + 0] = rounds;
            stats_out[ti * 3 + 1] = nmerge;
            stats_out[ti * 3 + 2] = ncc;
        }
        std::fprintf(stderr,
            "KR pred=%d bins=%d T=%.2f p0=%.4f p1=%.4f merges=%lld ncc=%lld max_vox=%lld rounds=%lld\n",
            pred, n_bins, T, p0, p1, (long long)nmerge, (long long)ncc,
            (long long)mx, (long long)rounds);
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Leftover / relative-contact / block-S3 (P0s/t/u, L33, B34, V35)
// ---------------------------------------------------------------------------

static void aggregate_residual(
    const std::vector<ERec>& es, DSU& d,
    std::unordered_map<uint64_t, std::pair<double, int64_t>>& agg)
{
    agg.clear();
    agg.reserve(es.size() / 4);
    for (const ERec& e : es) {
        uint32_t fu = d.find(e.u), fv = d.find(e.v);
        if (fu == fv || fu == 0 || fv == 0) continue;
        if (fu > fv) std::swap(fu, fv);
        auto& pr = agg[pair_key(fu, fv)];
        pr.first += e.mean * (double)e.area;
        pr.second += e.area;
    }
}

static int64_t s4_heap_on_residual(
    uint32_t nnode,
    const std::unordered_map<uint64_t, std::pair<double, int64_t>>& agg,
    DSU& d,
    double T)
{
    Graph g(nnode);
    for (uint32_t i = 0; i < nnode; ++i) g.parent[i] = d.find(i);
    int64_t nadd = 0;
    for (const auto& kv : agg) {
        if (kv.second.second < 1) continue;
        double mean = kv.second.first / (double)kv.second.second;
        if (!(mean > T)) continue;
        uint32_t a = (uint32_t)(kv.first >> 32);
        uint32_t b = (uint32_t)(kv.first & 0xffffffffu);
        a = d.find(a);
        b = d.find(b);
        if (a == b || a == 0 || b == 0) continue;
        g.add_edge(a, b, kv.second.first, kv.second.second);
        ++nadd;
    }
    if (nadd == 0) return 0;
    const double tscore = 1.0 - T;
    struct HItem {
        double score;
        uint32_t lo, hi;
        int eid;
        bool operator>(const HItem& o) const {
            if (score != o.score) return score > o.score;
            if (lo != o.lo) return lo > o.lo;
            return hi > o.hi;
        }
    };
    std::priority_queue<HItem, std::vector<HItem>, std::greater<HItem>> heap;
    g.compact_live();
    for (int eid : g.live) {
        Edge& e = g.edges[(size_t)eid];
        if (!e.alive) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) {
            e.alive = false;
            continue;
        }
        if (e.score >= tscore) continue;
        if (fu > fv) std::swap(fu, fv);
        heap.push({e.score, fu, fv, eid});
    }
    int64_t merges = 0;
    while (!heap.empty()) {
        HItem h = heap.top();
        heap.pop();
        Edge& e = g.edges[(size_t)h.eid];
        if (!e.alive || e.score >= tscore) continue;
        uint32_t fu = g.find(e.u), fv = g.find(e.v);
        if (fu == fv) {
            e.alive = false;
            continue;
        }
        if (fu > fv) std::swap(fu, fv);
        if (e.score != h.score || fu != h.lo || fv != h.hi) {
            heap.push({e.score, fu, fv, h.eid});
            continue;
        }
        if (!g.unite(fu, fv)) continue;
        ++merges;
        uint32_t keep = g.find(fu);
        for (int eid : g.adj[keep]) {
            Edge& ne = g.edges[(size_t)eid];
            if (!ne.alive || ne.score >= tscore) continue;
            uint32_t a = g.find(ne.u), b = g.find(ne.v);
            if (a == b) {
                ne.alive = false;
                continue;
            }
            if (a > b) std::swap(a, b);
            heap.push({ne.score, a, b, eid});
        }
    }
    for (uint32_t i = 0; i < nnode; ++i) d.p[i] = g.find(i);
    return merges;
}

extern "C" int p0_leftover_block_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    const int64_t* frag_sz,
    const int32_t* tile0,
    const int32_t* tile1,
    const int32_t* tile2,
    int64_t n_edges,
    uint32_t max_id,
    int64_t total_vox)
{
    if (n_edges <= 0) return 0;
    const uint32_t nnode = max_id + 1;
    std::vector<ERec> es;
    fill_edges(u_in, v_in, sum_in, count_in, n_edges, es);
    const double Ts[4] = {0.2, 0.3, 0.4, 0.5};
    const int64_t S0s[3] = {256, 1024, 4096};

    for (int si = 0; si < 3; ++si) {
        double s0 = (double)S0s[si];
        for (int ti = 0; ti < 4; ++ti) {
            const double T = Ts[ti];
            DSU d(nnode);
            for (uint32_t i = 0; i < nnode; ++i) d.vox[i] = frag_sz ? frag_sz[i] : 1;
            if (nnode) d.vox[0] = 0;
            kruskal_one(es, d, nnode, PRED_SDSL, 1, T, s0, 0.0);
            std::unordered_map<uint64_t, std::pair<double, int64_t>> agg;
            aggregate_residual(es, d, agg);
            int64_t n_residual = 0, n_ll = 0, n_small = 0;
            for (const auto& kv : agg) {
                if (kv.second.second < 1) continue;
                double mean = kv.second.first / (double)kv.second.second;
                if (!(mean > T)) continue;
                ++n_residual;
                uint32_t a = (uint32_t)(kv.first >> 32);
                uint32_t b = (uint32_t)(kv.first & 0xffffffffu);
                a = d.find(a);
                b = d.find(b);
                int64_t s1 = d.vox[a], s2 = d.vox[b];
                int64_t mn = s1 < s2 ? s1 : s2;
                if (s1 >= S0s[si] && s2 >= S0s[si]) ++n_ll;
                if (mn < S0s[si]) ++n_small;
            }
            int64_t n_small_touch = 0;
            for (const ERec& e : es) {
                if (!(e.mean > T)) continue;
                if (d.find(e.u) == d.find(e.v)) continue;
                int64_t s1 = frag_sz ? frag_sz[e.u] : 1;
                int64_t s2 = frag_sz ? frag_sz[e.v] : 1;
                int64_t mn = s1 < s2 ? s1 : s2;
                if (mn < S0s[si]) ++n_small_touch;
            }
            std::fprintf(stderr,
                "P0s S0=%lld T=%.2f n_residual=%lld n_large_large=%lld n_small_touch=%lld\n",
                (long long)S0s[si], T, (long long)n_residual,
                (long long)n_ll, (long long)n_small_touch);
            (void)n_small;
        }
    }

    const double gammas[4] = {0.01, 0.02, 0.05, 0.1};
    const double alphas[2] = {0.50, 0.67};
    for (int gi = 0; gi < 4; ++gi) {
        for (int ai = 0; ai < 2; ++ai) {
            for (int ti = 0; ti < 4; ++ti) {
                const double T = Ts[ti];
                DSU d(nnode);
                for (uint32_t i = 0; i < nnode; ++i) d.vox[i] = frag_sz ? frag_sz[i] : 1;
                if (nnode) d.vox[0] = 0;
                int64_t nm = kruskal_one(es, d, nnode, PRED_REL, 1, T, gammas[gi], alphas[ai]);
                int64_t ncc = n_components(d, nnode);
                int64_t mx = max_cc_vox(d, nnode);
                double gfrac = total_vox > 0 ? (double)mx / (double)total_vox : 0;
                std::fprintf(stderr,
                    "P0t gamma=%.3f alpha=%.2f T=%.2f giant=%.6f ncc=%lld nmerge=%lld max_vox=%lld\n",
                    gammas[gi], alphas[ai], T, gfrac,
                    (long long)ncc, (long long)nm, (long long)mx);

                DSU db(nnode);
                for (uint32_t i = 0; i < nnode; ++i) db.vox[i] = frag_sz ? frag_sz[i] : 1;
                if (nnode) db.vox[0] = 0;
                int64_t nmb = kruskal_one(es, db, nnode, PRED_REL, 16, T, gammas[gi], alphas[ai]);
                int64_t nccb = n_components(db, nnode);
                int64_t mxb = max_cc_vox(db, nnode);
                double gfb = total_vox > 0 ? (double)mxb / (double)total_vox : 0;
                std::fprintf(stderr,
                    "P0t_b16 gamma=%.3f alpha=%.2f T=%.2f giant=%.6f ncc=%lld nmerge=%lld rounds=16\n",
                    gammas[gi], alphas[ai], T, gfb,
                    (long long)nccb, (long long)nmb);
            }
        }
    }

    const int32_t* tiles[3] = {tile0, tile1, tile2};
    const char* tagn[3] = {"8x32x32", "16x64x64", "32x128x128"};
    std::vector<int> ord((int)es.size());
    for (int i = 0; i < (int)es.size(); ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int i, int j) {
        if (es[i].mean != es[j].mean) return es[i].mean > es[j].mean;
        return i < j;
    });
    for (int k = 0; k < 3; ++k) {
        const int32_t* tid = tiles[k];
        if (!tid) continue;
        for (int ti = 0; ti < 4; ++ti) {
            const double T = Ts[ti];
            int64_t n_intra = 0, n_inter = 0;
            for (const ERec& e : es) {
                if (!(e.mean > T)) continue;
                int32_t tu = tid[e.u], tv = tid[e.v];
                if (tu == tv) ++n_intra;
                else ++n_inter;
            }
            double frac = (n_intra + n_inter) > 0
                ? (double)n_intra / (double)(n_intra + n_inter) : 0;
            DSU d(nnode);
            for (uint32_t i = 0; i < nnode; ++i) d.vox[i] = frag_sz ? frag_sz[i] : 1;
            if (nnode) d.vox[0] = 0;
            for (int idx : ord) {
                const ERec& e = es[idx];
                if (!(e.mean > T)) continue;
                if (tid[e.u] != tid[e.v]) continue;
                d.unite(e.u, e.v);
            }
            std::unordered_map<uint64_t, std::pair<double, int64_t>> agg;
            aggregate_residual(es, d, agg);
            int64_t n_res_inter = 0;
            std::vector<char> froz(nnode, 0);
            for (const auto& kv : agg) {
                if (kv.second.second < 1) continue;
                double mean = kv.second.first / (double)kv.second.second;
                if (!(mean > T)) continue;
                uint32_t a = (uint32_t)(kv.first >> 32);
                uint32_t b = (uint32_t)(kv.first & 0xffffffffu);
                a = d.find(a);
                b = d.find(b);
                bool inter = false;
                // residual after interior-only: any live S3>T is inter-or-refused
                // count as inter if any original contributing pair is inter-tile
                (void)a;
                (void)b;
                ++n_res_inter;
                froz[a] = 1;
                froz[b] = 1;
            }
            // recompute inter-only residual + frozen from original inter edges
            n_res_inter = 0;
            std::fill(froz.begin(), froz.end(), 0);
            std::unordered_map<uint64_t, std::pair<double, int64_t>> inter_agg;
            inter_agg.reserve(es.size() / 4);
            for (const ERec& e : es) {
                if (tid[e.u] == tid[e.v]) continue;
                uint32_t fu = d.find(e.u), fv = d.find(e.v);
                if (fu == fv || fu == 0 || fv == 0) continue;
                if (fu > fv) std::swap(fu, fv);
                auto& pr = inter_agg[pair_key(fu, fv)];
                pr.first += e.mean * (double)e.area;
                pr.second += e.area;
            }
            for (const auto& kv : inter_agg) {
                if (kv.second.second < 1) continue;
                double mean = kv.second.first / (double)kv.second.second;
                if (!(mean > T)) continue;
                ++n_res_inter;
                uint32_t a = d.find((uint32_t)(kv.first >> 32));
                uint32_t b = d.find((uint32_t)(kv.first & 0xffffffffu));
                froz[a] = 1;
                froz[b] = 1;
            }
            int64_t frozen = 0;
            for (uint32_t i = 1; i < nnode; ++i)
                if (d.find(i) == i && d.vox[i] > 0 && froz[i]) ++frozen;
            std::fprintf(stderr,
                "P0u tile=%s T=%.2f n_intra=%lld n_inter=%lld intra_frac=%.6f frozen=%lld n_residual_inter=%lld\n",
                tagn[k], T, (long long)n_intra, (long long)n_inter, frac,
                (long long)frozen, (long long)n_res_inter);
        }
    }
    return 1;
}

extern "C" int leftover_s4_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    const int64_t* frag_sz,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    double s0,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* residual_out)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    const uint32_t nnode = max_id + 1;
    std::vector<ERec> es;
    fill_edges(u_in, v_in, sum_in, count_in, n_edges, es);
    std::vector<int> order(n_thr);
    for (int i = 0; i < n_thr; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return aff_thr[a] > aff_thr[b];
    });
    for (int oi = 0; oi < n_thr; ++oi) {
        int ti = order[oi];
        const double T = aff_thr[ti];
        DSU d(nnode);
        for (uint32_t i = 0; i < nnode; ++i) d.vox[i] = frag_sz ? frag_sz[i] : 1;
        if (nnode) d.vox[0] = 0;
        kruskal_one(es, d, nnode, PRED_SDSL, 1, T, s0, 0.0);
        std::unordered_map<uint64_t, std::pair<double, int64_t>> agg;
        aggregate_residual(es, d, agg);
        int64_t nres = 0;
        for (const auto& kv : agg) {
            if (kv.second.second < 1) continue;
            if (kv.second.first / (double)kv.second.second > T) ++nres;
        }
        if (residual_out) residual_out[ti] = nres;
        int64_t nm = 0;
        if (nres > 0 && nres < 50000)
            nm = s4_heap_on_residual(nnode, agg, d, T);
        else if (nres >= 50000) {
            // still Kruskal the leftover (mean order, live sizes unused) so VOI exists
            for (const auto& kv : agg) {
                if (kv.second.second < 1) continue;
                double mean = kv.second.first / (double)kv.second.second;
                if (!(mean > T)) continue;
                uint32_t a = (uint32_t)(kv.first >> 32);
                uint32_t b = (uint32_t)(kv.first & 0xffffffffu);
                d.unite(a, b);
            }
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = d.find(i);
        std::fprintf(stderr,
            "L33 S0=%.0f T=%.2f n_residual=%lld s4_merges=%lld\n",
            s0, T, (long long)nres, (long long)nm);
    }
    return 1;
}

extern "C" int block_s3_naive_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    const int64_t* frag_sz,
    const int32_t* tile_id,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0 || !tile_id) return 0;
    const uint32_t nnode = max_id + 1;
    std::vector<ERec> es;
    fill_edges(u_in, v_in, sum_in, count_in, n_edges, es);
    std::vector<int> ord((int)es.size());
    for (int i = 0; i < (int)es.size(); ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int i, int j) {
        if (es[i].mean != es[j].mean) return es[i].mean > es[j].mean;
        return i < j;
    });
    for (int ti = 0; ti < n_thr; ++ti) {
        const double T = aff_thr[ti];
        DSU d(nnode);
        for (uint32_t i = 0; i < nnode; ++i) d.vox[i] = frag_sz ? frag_sz[i] : 1;
        if (nnode) d.vox[0] = 0;
        int64_t n_intra = 0;
        for (int idx : ord) {
            const ERec& e = es[idx];
            if (!(e.mean > T)) continue;
            if (tile_id[e.u] != tile_id[e.v]) continue;
            if (d.unite(e.u, e.v)) ++n_intra;
        }
        std::unordered_map<uint64_t, std::pair<double, int64_t>> agg;
        aggregate_residual(es, d, agg);
        int64_t nres = 0;
        for (const auto& kv : agg) {
            if (kv.second.second < 1) continue;
            if (kv.second.first / (double)kv.second.second > T) ++nres;
        }
        int64_t nm = 0;
        if (nres > 0 && nres < 50000)
            nm = s4_heap_on_residual(nnode, agg, d, T);
        else {
            for (const auto& kv : agg) {
                if (kv.second.second < 1) continue;
                double mean = kv.second.first / (double)kv.second.second;
                if (!(mean > T)) continue;
                uint32_t a = (uint32_t)(kv.first >> 32);
                uint32_t b = (uint32_t)(kv.first & 0xffffffffu);
                if (d.unite(a, b)) ++nm;
            }
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = d.find(i);
        if (stats_out) {
            stats_out[ti * 3 + 0] = 2;
            stats_out[ti * 3 + 1] = nres;
            stats_out[ti * 3 + 2] = nm;
        }
        std::fprintf(stderr,
            "B34naive T=%.2f intra_merges=%lld n_residual=%lld tail_merges=%lld\n",
            T, (long long)n_intra, (long long)nres, (long long)nm);
    }
    return 1;
}

extern "C" int block_s3_lu_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    const int64_t* frag_sz,
    const float* cz,
    const float* cy,
    const float* cx,
    int64_t n_edges,
    double dz0,
    double dy0,
    double dx0,
    const double* aff_thr,
    int n_thr,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0 || !cz) return 0;
    const uint32_t nnode = max_id + 1;
    std::vector<ERec> es;
    fill_edges(u_in, v_in, sum_in, count_in, n_edges, es);
    std::vector<int> ord((int)es.size());
    for (int i = 0; i < (int)es.size(); ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int i, int j) {
        if (es[i].mean != es[j].mean) return es[i].mean > es[j].mean;
        return i < j;
    });
    auto tile_of = [&](uint32_t i, double dz, double dy, double dx) -> int64_t {
        int tz = (int)std::floor((double)cz[i] / dz);
        int ty = (int)std::floor((double)cy[i] / dy);
        int tx = (int)std::floor((double)cx[i] / dx);
        if (tz < 0) tz = 0;
        if (ty < 0) ty = 0;
        if (tx < 0) tx = 0;
        return ((int64_t)tz << 42) | ((int64_t)ty << 21) | (int64_t)tx;
    };
    for (int ti = 0; ti < n_thr; ++ti) {
        const double T = aff_thr[ti];
        DSU d(nnode);
        for (uint32_t i = 0; i < nnode; ++i) d.vox[i] = frag_sz ? frag_sz[i] : 1;
        if (nnode) d.vox[0] = 0;
        double dz = dz0, dy = dy0, dx = dx0;
        int64_t max_res = 0, last_res = 0, nlev = 0;
        for (int lev = 0; lev < 8; ++lev) {
            ++nlev;
            std::vector<int64_t> tid(nnode, 0);
            for (uint32_t i = 1; i < nnode; ++i)
                tid[i] = tile_of(i, dz, dy, dx);
            std::vector<char> frozen(nnode, 0);
            for (const ERec& e : es) {
                if (e.u == 0 || e.v == 0) continue;
                if (tid[e.u] != tid[e.v]) {
                    frozen[d.find(e.u)] = 1;
                    frozen[d.find(e.v)] = 1;
                }
            }
            int64_t nmerge = 0;
            for (int idx : ord) {
                const ERec& e = es[idx];
                if (!(e.mean > T)) continue;
                uint32_t fu = d.find(e.u), fv = d.find(e.v);
                if (fu == fv) continue;
                if (frozen[fu] || frozen[fv]) {
                    frozen[fu] = frozen[fv] = 1;
                    continue;
                }
                if (d.unite(e.u, e.v)) ++nmerge;
            }
            std::unordered_map<uint64_t, std::pair<double, int64_t>> agg;
            aggregate_residual(es, d, agg);
            int64_t nres = 0;
            int64_t nfrozen = 0;
            std::vector<char> froz2(nnode, 0);
            for (const auto& kv : agg) {
                if (kv.second.second < 1) continue;
                if (!(kv.second.first / (double)kv.second.second > T)) continue;
                ++nres;
                uint32_t a = d.find((uint32_t)(kv.first >> 32));
                uint32_t b = d.find((uint32_t)(kv.first & 0xffffffffu));
                froz2[a] = froz2[b] = 1;
            }
            for (uint32_t i = 1; i < nnode; ++i)
                if (d.find(i) == i && d.vox[i] > 0 && froz2[i]) ++nfrozen;
            last_res = nres;
            if (nres > max_res) max_res = nres;
            std::fprintf(stderr,
                "B34lu T=%.2f lev=%d tile=%.0fx%.0fx%.0f merges=%lld residual=%lld frozen=%lld\n",
                T, lev, dz, dy, dx, (long long)nmerge,
                (long long)nres, (long long)nfrozen);
            if (nres < 50000) {
                if (nres > 0) s4_heap_on_residual(nnode, agg, d, T);
                break;
            }
            dz *= 2.0;
            dy *= 2.0;
            dx *= 2.0;
        }
        if (last_res >= 50000) {
            std::unordered_map<uint64_t, std::pair<double, int64_t>> agg;
            aggregate_residual(es, d, agg);
            for (const auto& kv : agg) {
                if (kv.second.second < 1) continue;
                if (!(kv.second.first / (double)kv.second.second > T)) continue;
                uint32_t a = (uint32_t)(kv.first >> 32);
                uint32_t b = (uint32_t)(kv.first & 0xffffffffu);
                d.unite(a, b);
            }
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = d.find(i);
        if (stats_out) {
            stats_out[ti * 3 + 0] = nlev;
            stats_out[ti * 3 + 1] = last_res;
            stats_out[ti * 3 + 2] = max_res;
        }
    }
    return 1;
}

extern "C" int hysteresis_cc_cpu(
    const uint8_t* aff,
    int zdim,
    int ydim,
    int xdim,
    double high,
    double lowT,
    uint32_t* labels)
{
    if (!aff || zdim < 1 || ydim < 1 || xdim < 1 || !labels) return 0;
    const int64_t plane = (int64_t)ydim * (int64_t)xdim;
    const int64_t vol = (int64_t)zdim * plane;
    const int64_t stride_c = vol;
    auto a_at = [&](int c, int z, int y, int x) -> double {
        return (double)aff[(int64_t)c * stride_c + (int64_t)z * plane
                           + (int64_t)y * (int64_t)xdim + (int64_t)x] / 255.0;
    };
    auto vid = [&](int z, int y, int x) -> uint32_t {
        return (uint32_t)((int64_t)z * plane + (int64_t)y * (int64_t)xdim + (int64_t)x);
    };
    std::vector<uint32_t> parent((size_t)vol);
    for (int64_t i = 0; i < vol; ++i) parent[(size_t)i] = (uint32_t)i;
    auto find = [&](uint32_t x) {
        uint32_t r = x;
        while (parent[r] != r) r = parent[r];
        while (parent[x] != r) {
            uint32_t n = parent[x];
            parent[x] = r;
            x = n;
        }
        return r;
    };
    auto unite = [&](uint32_t a, uint32_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (a > b) std::swap(a, b);
        parent[b] = a;
    };
    std::vector<uint8_t> seed((size_t)vol, 0);
    std::vector<uint8_t> bg((size_t)vol, 1);
    for (int z = 0; z < zdim; ++z) {
        for (int y = 0; y < ydim; ++y) {
            for (int x = 0; x < xdim; ++x) {
                double m = 0;
                double e[6];
                e[0] = (z > 0) ? a_at(0, z, y, x) : 0;
                e[1] = (y > 0) ? a_at(1, z, y, x) : 0;
                e[2] = (x > 0) ? a_at(2, z, y, x) : 0;
                e[3] = (z + 1 < zdim) ? a_at(0, z + 1, y, x) : 0;
                e[4] = (y + 1 < ydim) ? a_at(1, z, y + 1, x) : 0;
                e[5] = (x + 1 < xdim) ? a_at(2, z, y, x + 1) : 0;
                for (int k = 0; k < 6; ++k) {
                    if (e[k] > m) m = e[k];
                    if (e[k] >= high) seed[vid(z, y, x)] = 1;
                }
                if (m > 1e-4) bg[vid(z, y, x)] = 0;
            }
        }
    }
    std::vector<uint32_t> q;
    q.reserve((size_t)(vol / 8));
    std::vector<uint8_t> vis((size_t)vol, 0);
    for (int64_t i = 0; i < vol; ++i) {
        if (seed[(size_t)i]) {
            vis[(size_t)i] = 1;
            q.push_back((uint32_t)i);
        }
    }
    size_t head = 0;
    auto try_nb = [&](uint32_t src, int z, int y, int x, double w) {
        if (w < lowT) return;
        uint32_t dst = vid(z, y, x);
        unite(src, dst);
        if (!vis[dst]) {
            vis[dst] = 1;
            q.push_back(dst);
        }
    };
    while (head < q.size()) {
        uint32_t id = q[head++];
        int z = (int)(id / (uint32_t)plane);
        int rem = (int)(id % (uint32_t)plane);
        int y = rem / xdim;
        int x = rem % xdim;
        if (z > 0) try_nb(id, z - 1, y, x, a_at(0, z, y, x));
        if (y > 0) try_nb(id, z, y - 1, x, a_at(1, z, y, x));
        if (x > 0) try_nb(id, z, y, x - 1, a_at(2, z, y, x));
        if (z + 1 < zdim) try_nb(id, z + 1, y, x, a_at(0, z + 1, y, x));
        if (y + 1 < ydim) try_nb(id, z, y + 1, x, a_at(1, z, y + 1, x));
        if (x + 1 < xdim) try_nb(id, z, y, x + 1, a_at(2, z, y, x + 1));
    }
    uint32_t next_id = 1;
    std::vector<uint32_t> remap((size_t)vol, 0);
    for (int64_t i = 0; i < vol; ++i) {
        if (bg[(size_t)i] && !vis[(size_t)i]) {
            labels[i] = 0;
            continue;
        }
        if (!vis[(size_t)i]) {
            labels[i] = next_id++;
            continue;
        }
        uint32_t r = find((uint32_t)i);
        if (remap[r] == 0) remap[r] = next_id++;
        labels[i] = remap[r];
    }
    std::fprintf(stderr,
        "V35 high=%.2f low=%.2f nseg=%u seeds=%zu grown=%zu\n",
        high, lowT, (unsigned)(next_id - 1), q.size(), (size_t)0);
    return 1;
}

extern "C" int p0_rel_leftover_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    const int64_t* frag_sz,
    int64_t n_edges,
    uint32_t max_id,
    int64_t total_vox)
{
    if (n_edges <= 0) return 0;
    const uint32_t nnode = max_id + 1;
    std::vector<ERec> es;
    fill_edges(u_in, v_in, sum_in, count_in, n_edges, es);
    const double Ts[4] = {0.2, 0.3, 0.4, 0.5};
    const double gammas[2] = {0.10, 0.05};
    const double alpha = 0.67;
    for (int gi = 0; gi < 2; ++gi) {
        for (int ti = 0; ti < 4; ++ti) {
            const double T = Ts[ti];
            DSU d(nnode);
            for (uint32_t i = 0; i < nnode; ++i) d.vox[i] = frag_sz ? frag_sz[i] : 1;
            if (nnode) d.vox[0] = 0;
            kruskal_one(es, d, nnode, PRED_REL, 1, T, gammas[gi], alpha);
            std::unordered_map<uint64_t, std::pair<double, int64_t>> agg;
            aggregate_residual(es, d, agg);
            int64_t nres = 0, nhigh = 0;
            DSU dg(nnode);
            for (uint32_t i = 0; i < nnode; ++i) dg.vox[i] = frag_sz ? frag_sz[i] : 1;
            if (nnode) dg.vox[0] = 0;
            for (const auto& kv : agg) {
                if (kv.second.second < 1) continue;
                double mean = kv.second.first / (double)kv.second.second;
                if (!(mean > T)) continue;
                ++nres;
                if (mean > 0.9) ++nhigh;
                uint32_t a = (uint32_t)(kv.first >> 32);
                uint32_t b = (uint32_t)(kv.first & 0xffffffffu);
                dg.unite(a, b);
            }
            int64_t mx = max_cc_vox(dg, nnode);
            double gfrac = total_vox > 0 ? (double)mx / (double)total_vox : 0;
            std::fprintf(stderr,
                "P0v gamma=%.3f alpha=%.2f T=%.2f n_residual=%lld n_high=%.0f giant_if_union=%.6f\n",
                gammas[gi], alpha, T, (long long)nres, (double)nhigh, gfrac);
        }
    }
    return 1;
}

extern "C" int leftover_rel_s4_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    const int64_t* frag_sz,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    double gamma,
    double alpha,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* residual_out)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    const uint32_t nnode = max_id + 1;
    std::vector<ERec> es;
    fill_edges(u_in, v_in, sum_in, count_in, n_edges, es);
    for (int ti = 0; ti < n_thr; ++ti) {
        const double T = aff_thr[ti];
        DSU d(nnode);
        for (uint32_t i = 0; i < nnode; ++i) d.vox[i] = frag_sz ? frag_sz[i] : 1;
        if (nnode) d.vox[0] = 0;
        kruskal_one(es, d, nnode, PRED_REL, 1, T, gamma, alpha);
        std::unordered_map<uint64_t, std::pair<double, int64_t>> agg;
        aggregate_residual(es, d, agg);
        int64_t nres = 0;
        for (const auto& kv : agg) {
            if (kv.second.second < 1) continue;
            if (kv.second.first / (double)kv.second.second > T) ++nres;
        }
        if (residual_out) residual_out[ti] = nres;
        int64_t nm = 0;
        if (nres > 0 && nres < 50000)
            nm = s4_heap_on_residual(nnode, agg, d, T);
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = d.find(i);
        std::fprintf(stderr,
            "L36 gamma=%.3f alpha=%.2f T=%.2f n_residual=%lld s4_merges=%lld\n",
            gamma, alpha, T, (long long)nres, (long long)nm);
    }
    return 1;
}

extern "C" int nn_chain_s3_cpu(
    const uint32_t* u_in,
    const uint32_t* v_in,
    const double* sum_in,
    const int64_t* count_in,
    int64_t n_edges,
    const double* aff_thr,
    int n_thr,
    uint32_t* parent_out,
    uint32_t max_id,
    int64_t* stats_out)
{
    if (n_edges <= 0 || n_thr <= 0) return 0;
    const uint32_t nnode = max_id + 1;
    std::vector<ERec> es;
    fill_edges(u_in, v_in, sum_in, count_in, n_edges, es);
    for (int ti = 0; ti < n_thr; ++ti) {
        const double T = aff_thr[ti];
        DSU d(nnode);
        for (uint32_t i = 0; i < nnode; ++i) d.vox[i] = 1;
        if (nnode) d.vox[0] = 0;
        int64_t rounds = 0, nmerge = 0, maxchain = 0;
        const int64_t maxr = 200;
        while (rounds < maxr) {
            std::vector<uint32_t> bn(nnode, 0xffffffffu);
            std::vector<double> bw(nnode, -1.0);
            for (const ERec& e : es) {
                if (!(e.mean > T)) continue;
                uint32_t fu = d.find(e.u), fv = d.find(e.v);
                if (fu == fv) continue;
                if (e.mean > bw[fu]) {
                    bw[fu] = e.mean;
                    bn[fu] = fv;
                }
                if (e.mean > bw[fv]) {
                    bw[fv] = e.mean;
                    bn[fv] = fu;
                }
            }
            int64_t nm = 0, chain = 0;
            std::vector<char> used(nnode, 0);
            for (uint32_t i = 1; i < nnode; ++i) {
                if (d.p[i] != i || bn[i] == 0xffffffffu) continue;
                uint32_t a = i, b = bn[i];
                int steps = 0;
                while (b != 0xffffffffu && bn[b] != a && !used[a] && steps < 64) {
                    a = b;
                    b = bn[a];
                    ++steps;
                }
                if (steps > chain) chain = steps;
                a = d.find(a);
                if (b == 0xffffffffu) continue;
                b = d.find(b);
                if (a == b || used[a] || used[b]) continue;
                if (bn[a] != b && bn[b] != a) continue;
                used[a] = used[b] = 1;
                if (d.unite(a, b)) ++nm;
            }
            ++rounds;
            nmerge += nm;
            if (chain > maxchain) maxchain = chain;
            if (nm == 0) break;
        }
        uint32_t* dst = parent_out + (size_t)ti * (size_t)nnode;
        for (uint32_t i = 0; i < nnode; ++i) dst[i] = d.find(i);
        if (stats_out) {
            stats_out[ti * 3 + 0] = rounds;
            stats_out[ti * 3 + 1] = nmerge;
            stats_out[ti * 3 + 2] = maxchain;
        }
        std::fprintf(stderr,
            "P0w T=%.2f rounds=%lld merges=%lld maxchain=%lld ncc=%lld\n",
            T, (long long)rounds, (long long)nmerge, (long long)maxchain,
            (long long)n_components(d, nnode));
    }
    return 1;
}
