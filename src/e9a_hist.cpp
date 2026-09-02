// E9a: plateau size histogram after host flow matching k_flow / S1 bits.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

static inline float aat(
    const uint8_t* aff, int64_t Z, int64_t Y, int64_t X,
    int c, int64_t z, int64_t y, int64_t x)
{
    return aff[((c * Z + z) * Y + y) * X + x] * (1.0f / 255.0f);
}

static void host_flow(
    const uint8_t* aff, int64_t Z, int64_t Y, int64_t X,
    float low, float high, uint8_t* bits)
{
    const int64_t size = Z * Y * X;
    for (int64_t i = 0; i < size; ++i) {
        int64_t yx = Y * X;
        int64_t z = i / yx, r = i % yx, y = r / X, x = r % X;
        float nz = (z > 0) ? aat(aff, Z, Y, X, 0, z, y, x) : low;
        float ny = (y > 0) ? aat(aff, Z, Y, X, 1, z, y, x) : low;
        float nx = (x > 0) ? aat(aff, Z, Y, X, 2, z, y, x) : low;
        float pz = (z < Z - 1) ? aat(aff, Z, Y, X, 0, z + 1, y, x) : low;
        float py = (y < Y - 1) ? aat(aff, Z, Y, X, 1, z, y + 1, x) : low;
        float px = (x < X - 1) ? aat(aff, Z, Y, X, 2, z, y, x + 1) : low;
        float m = fmaxf(fmaxf(fmaxf(nx, ny), nz), fmaxf(fmaxf(px, py), pz));
        uint8_t id = 0;
        if (m > low) {
            if (nz == m || nz >= high) id |= 0x01;
            if (ny == m || ny >= high) id |= 0x02;
            if (nx == m || nx >= high) id |= 0x04;
            if (pz == m || pz >= high) id |= 0x08;
            if (py == m || py >= high) id |= 0x10;
            if (px == m || px >= high) id |= 0x20;
        }
        bits[i] = id;
    }
}

static uint32_t findp(std::vector<uint32_t>& p, uint32_t x) {
    uint32_t r = x;
    while (p[r] != r) r = p[r];
    while (p[x] != r) {
        uint32_t n = p[x];
        p[x] = r;
        x = n;
    }
    return r;
}

static void unite(std::vector<uint32_t>& p, uint32_t a, uint32_t b) {
    a = findp(p, a);
    b = findp(p, b);
    if (a == b) return;
    if (a > b) std::swap(a, b);
    p[b] = a;
}

extern "C" int e9a_plateau_hist(
    const uint8_t* aff, int64_t Z, int64_t Y, int64_t X,
    float low, float high, int64_t* out)
{
    const int64_t size = Z * Y * X;
    const int64_t yx = Y * X;
    const int64_t dir[6] = {-yx, -X, -1, yx, X, 1};
    const uint32_t dirmask[6] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20};
    const uint32_t idirmask[6] = {0x08, 0x10, 0x20, 0x01, 0x02, 0x04};
    std::vector<uint8_t> bits((size_t)size);
    host_flow(aff, Z, Y, X, low, high, bits.data());
    std::vector<uint32_t> parent((size_t)size);
    for (int64_t i = 0; i < size; ++i) parent[(size_t)i] = (uint32_t)i;
    std::vector<char> in_plat((size_t)size, 0);
    int64_t n_fg = 0, n_corner = 0, n_bg = 0;
    for (int64_t i = 0; i < size; ++i) {
        uint8_t b = bits[(size_t)i];
        if (!b) {
            ++n_bg;
            continue;
        }
        ++n_fg;
        bool corner = false;
        for (int d = 0; d < 6; ++d) {
            if (!(b & dirmask[d])) continue;
            int64_t j = i + dir[d];
            if (j < 0 || j >= size) continue;
            if (bits[(size_t)j] & idirmask[d]) {
                in_plat[(size_t)i] = 1;
                in_plat[(size_t)j] = 1;
                unite(parent, (uint32_t)i, (uint32_t)j);
            } else {
                corner = true;
            }
        }
        if (corner) ++n_corner;
    }
    std::vector<int64_t> sz((size_t)size, 0);
    std::vector<char> has_corner((size_t)size, 0);
    for (int64_t i = 0; i < size; ++i) {
        if (!in_plat[(size_t)i]) continue;
        uint32_t r = findp(parent, (uint32_t)i);
        ++sz[r];
        uint8_t b = bits[(size_t)i];
        for (int d = 0; d < 6; ++d) {
            if (!(b & dirmask[d])) continue;
            int64_t j = i + dir[d];
            if (j < 0 || j >= size) continue;
            if (!(bits[(size_t)j] & idirmask[d])) {
                has_corner[r] = 1;
                break;
            }
        }
    }
    std::vector<int64_t> sizes;
    sizes.reserve(1 << 20);
    int64_t n_cc = 0, n_closed = 0, n_closed_vox = 0, max_sz = 0;
    for (int64_t i = 0; i < size; ++i) {
        if (sz[(size_t)i] == 0) continue;
        ++n_cc;
        int64_t s = sz[(size_t)i];
        sizes.push_back(s);
        if (s > max_sz) max_sz = s;
        if (!has_corner[(size_t)i]) {
            ++n_closed;
            n_closed_vox += s;
        }
    }
    std::sort(sizes.begin(), sizes.end());
    int64_t p50 = 0, p99 = 0;
    if (!sizes.empty()) {
        p50 = sizes[(sizes.size() * 50) / 100];
        p99 = sizes[(sizes.size() * 99) / 100];
        if (p99 >= (int64_t)sizes.size()) p99 = sizes.back();
        p50 = sizes[std::min(sizes.size() - 1, (sizes.size() * 50) / 100)];
        p99 = sizes[std::min(sizes.size() - 1, (sizes.size() * 99) / 100)];
    }
    // out: n_fg, n_bg, n_corner, n_cc, n_closed, n_closed_vox, max, p50, p99
    if (out) {
        out[0] = n_fg;
        out[1] = n_bg;
        out[2] = n_corner;
        out[3] = n_cc;
        out[4] = n_closed;
        out[5] = n_closed_vox;
        out[6] = max_sz;
        out[7] = p50;
        out[8] = p99;
    }
    std::fprintf(stderr,
        "E9a n_fg=%lld n_bg=%lld n_corner=%lld n_plat_cc=%lld "
        "n_closed_cc=%lld n_closed_vox=%lld max=%lld p50=%lld p99=%lld "
        "max/fg=%.6f giant=%d\n",
        (long long)n_fg, (long long)n_bg, (long long)n_corner, (long long)n_cc,
        (long long)n_closed, (long long)n_closed_vox, (long long)max_sz,
        (long long)p50, (long long)p99,
        n_fg ? (double)max_sz / (double)n_fg : 0.0,
        (n_fg && max_sz > (int64_t)(0.2 * (double)n_fg)) ? 1 : 0);
    int64_t buckets[8] = {1, 4, 16, 64, 256, 1024, 4096, 1LL << 60};
    int64_t hist[8] = {0};
    for (int64_t s : sizes) {
        for (int k = 0; k < 8; ++k) {
            if (s <= buckets[k]) {
                ++hist[k];
                break;
            }
        }
    }
    std::fprintf(stderr,
        "E9a hist <=1,%lld <=4,%lld <=16,%lld <=64,%lld <=256,%lld "
        "<=1024,%lld <=4096,%lld >4096,%lld\n",
        (long long)hist[0], (long long)hist[1], (long long)hist[2],
        (long long)hist[3], (long long)hist[4], (long long)hist[5],
        (long long)hist[6], (long long)hist[7]);
    return 1;
}
