// CPU watershed, same semantics as waterz basic_watershed.hpp. No GPU.
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cstddef>

extern "C" uint32_t watershed_cpu(
    const float* aff, // [3,Z,Y,X] C-order
    int64_t Z, int64_t Y, int64_t X,
    float low, float high,
    uint32_t* seg)
{
    const int64_t size = Z * Y * X;
    const int64_t yx = Y * X;
    auto idx = [&](int64_t z, int64_t y, int64_t x) { return z * yx + y * X + x; };
    auto aat = [&](int c, int64_t z, int64_t y, int64_t x) {
        return aff[(c * Z + z) * yx + y * X + x];
    };

    for (int64_t z = 0; z < Z; ++z)
        for (int64_t y = 0; y < Y; ++y)
            for (int64_t x = 0; x < X; ++x) {
                uint32_t& id = seg[idx(z, y, x)] = 0;
                float negz = (z > 0) ? aat(0, z, y, x) : low;
                float negy = (y > 0) ? aat(1, z, y, x) : low;
                float negx = (x > 0) ? aat(2, z, y, x) : low;
                float posz = (z < Z - 1) ? aat(0, z + 1, y, x) : low;
                float posy = (y < Y - 1) ? aat(1, z, y + 1, x) : low;
                float posx = (x < X - 1) ? aat(2, z, y, x + 1) : low;
                float m = std::max({negx, negy, negz, posx, posy, posz});
                if (m > low) {
                    if (negz == m || negz >= high) id |= 0x01;
                    if (negy == m || negy >= high) id |= 0x02;
                    if (negx == m || negx >= high) id |= 0x04;
                    if (posz == m || posz >= high) id |= 0x08;
                    if (posy == m || posy >= high) id |= 0x10;
                    if (posx == m || posx >= high) id |= 0x20;
                }
            }

    const int64_t dir[6] = {-yx, -X, -1, yx, X, 1};
    const uint32_t dirmask[6] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20};
    const uint32_t idirmask[6] = {0x08, 0x10, 0x20, 0x01, 0x02, 0x04};

    std::vector<int64_t> bfs;
    bfs.reserve(size / 8);
    for (int64_t i = 0; i < size; ++i) {
        for (int d = 0; d < 6; ++d) {
            if (seg[i] & dirmask[d]) {
                if (!(seg[i + dir[d]] & idirmask[d])) {
                    seg[i] |= 0x40;
                    bfs.push_back(i);
                    break;
                }
            }
        }
    }
    size_t bi = 0;
    while (bi < bfs.size()) {
        int64_t i = bfs[bi];
        uint32_t to_set = 0;
        for (int d = 0; d < 6; ++d) {
            if (seg[i] & dirmask[d]) {
                if (seg[i + dir[d]] & idirmask[d]) {
                    if (!(seg[i + dir[d]] & 0x40)) {
                        bfs.push_back(i + dir[d]);
                        seg[i + dir[d]] |= 0x40;
                    }
                } else {
                    to_set = dirmask[d];
                }
            }
        }
        seg[i] = to_set;
        ++bi;
    }
    bfs.clear();

    const uint32_t HIGH = 0x80000000u;
    uint32_t next_id = 1;
    for (int64_t i = 0; i < size; ++i) {
        if (seg[i] == 0) {
            seg[i] |= HIGH;
            continue;
        }
        if (!(seg[i] & HIGH) && seg[i]) {
            bfs.push_back(i);
            bi = 0;
            seg[i] |= 0x40;
            while (bi < bfs.size()) {
                int64_t me = bfs[bi];
                for (int d = 0; d < 6; ++d) {
                    if (seg[me] & dirmask[d]) {
                        int64_t him = me + dir[d];
                        if (seg[him] & HIGH) {
                            for (auto it : bfs) seg[it] = seg[him];
                            bfs.clear();
                            d = 6;
                        } else if (!(seg[him] & 0x40)) {
                            seg[him] |= 0x40;
                            bfs.push_back(him);
                        }
                    }
                }
                ++bi;
            }
            if (!bfs.empty()) {
                uint32_t lab = HIGH | next_id;
                for (auto it : bfs) seg[it] = lab;
                ++next_id;
                bfs.clear();
            }
        }
    }
    for (int64_t i = 0; i < size; ++i) seg[i] &= 0x7fffffffu;
    return next_id - 1;
}
