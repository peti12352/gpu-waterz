// Frozen CC: union every edge with mean > thr. Deterministic: lower id absorbs.
#include <cstdint>
#include <vector>

extern "C" int frozen_cc_cpu(
    const uint32_t* u,
    const uint32_t* v,
    const double* mean,
    int64_t n_edges,
    double thr,
    uint32_t* parent,
    uint32_t max_id)
{
    const uint32_t n = max_id + 1;
    for (uint32_t i = 0; i < n; ++i) parent[i] = i;

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

    for (int64_t i = 0; i < n_edges; ++i) {
        if (mean[i] <= thr) continue;
        uint32_t a = u[i], b = v[i];
        if (a == 0 || b == 0 || a == b) continue;
        a = find(a);
        b = find(b);
        if (a == b) continue;
        if (a < b) parent[b] = a;
        else parent[a] = b;
    }
    for (uint32_t i = 0; i < n; ++i) parent[i] = find(i);
    return 0;
}
