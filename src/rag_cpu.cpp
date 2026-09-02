// CPU RAG: 3 negative dirs, drop bg, waterz float incremental mean.
#include <cstdint>
#include <vector>
#include <unordered_map>

struct Acc {
    float mean;
    uint32_t n;
};

extern "C" int64_t rag_cpu(
    const float* aff, // [3,Z,Y,X]
    const uint32_t* seg,
    int64_t Z, int64_t Y, int64_t X,
    uint32_t* u_out,
    uint32_t* v_out,
    double* sum_out,
    int64_t* count_out,
    int64_t max_edges)
{
    const int64_t yx = Y * X;
    auto sid = [&](int64_t z, int64_t y, int64_t x) {
        return seg[z * yx + y * X + x];
    };
    auto aval = [&](int c, int64_t z, int64_t y, int64_t x) {
        return aff[(c * Z + z) * yx + y * X + x];
    };
    auto pk = [](uint32_t a, uint32_t b) -> uint64_t {
        return (uint64_t(a) << 32) | uint64_t(b);
    };

    std::unordered_map<uint64_t, Acc> acc;
    acc.reserve(1 << 23);

    for (int64_t z = 0; z < Z; ++z)
        for (int64_t y = 0; y < Y; ++y)
            for (int64_t x = 0; x < X; ++x) {
                uint32_t id1 = sid(z, y, x);
                if (z > 0) {
                    uint32_t id2 = sid(z - 1, y, x);
                    if (id1 != id2) {
                        uint32_t lo = id1 < id2 ? id1 : id2;
                        uint32_t hi = id1 < id2 ? id2 : id1;
                        if (lo != 0) {
                            Acc& a = acc[pk(lo, hi)];
                            float affv = aval(0, z, y, x);
                            a.mean = (affv + a.mean * (float)a.n) / (float)(a.n + 1);
                            a.n += 1;
                        }
                    }
                }
                if (y > 0) {
                    uint32_t id2 = sid(z, y - 1, x);
                    if (id1 != id2) {
                        uint32_t lo = id1 < id2 ? id1 : id2;
                        uint32_t hi = id1 < id2 ? id2 : id1;
                        if (lo != 0) {
                            Acc& a = acc[pk(lo, hi)];
                            float affv = aval(1, z, y, x);
                            a.mean = (affv + a.mean * (float)a.n) / (float)(a.n + 1);
                            a.n += 1;
                        }
                    }
                }
                if (x > 0) {
                    uint32_t id2 = sid(z, y, x - 1);
                    if (id1 != id2) {
                        uint32_t lo = id1 < id2 ? id1 : id2;
                        uint32_t hi = id1 < id2 ? id2 : id1;
                        if (lo != 0) {
                            Acc& a = acc[pk(lo, hi)];
                            float affv = aval(2, z, y, x);
                            a.mean = (affv + a.mean * (float)a.n) / (float)(a.n + 1);
                            a.n += 1;
                        }
                    }
                }
            }

    if ((int64_t)acc.size() > max_edges) return -1;
    int64_t i = 0;
    for (const auto& kv : acc) {
        uint32_t u = (uint32_t)(kv.first >> 32);
        uint32_t v = (uint32_t)(kv.first & 0xffffffffu);
        u_out[i] = u;
        v_out[i] = v;
        count_out[i] = (int64_t)kv.second.n;
        sum_out[i] = (double)kv.second.mean * (double)kv.second.n;
        ++i;
    }
    return i;
}
