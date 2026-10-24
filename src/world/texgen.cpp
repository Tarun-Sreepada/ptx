#include "world/texgen.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>

#include "core/jobs.h"

namespace texgen {
namespace {
constexpr uint32_t VERSION = 2;  // bump when the generator changes: invalidates the cache

struct C3 { float r, g, b; };
C3 operator+(C3 a, C3 b) { return {a.r + b.r, a.g + b.g, a.b + b.b}; }
C3 operator*(C3 a, float s) { return {a.r * s, a.g * s, a.b * s}; }
C3 operator*(C3 a, C3 b) { return {a.r * b.r, a.g * b.g, a.b * b.b}; }
C3 mix(C3 a, C3 b, float t) { return a * (1.f - t) + b * t; }
float clamp01(float x) { return std::min(1.f, std::max(0.f, x)); }
float smooth(float a, float b, float x) { float t = clamp01((x - a) / (b - a)); return t * t * (3.f - 2.f * t); }

uint32_t hash(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
float h01(uint32_t x) { return (hash(x) & 0xFFFFFF) / float(0xFFFFFF); }

// tileable value noise: lattice of `period` cells over [0,1) in u, v
float tnoise(float u, float v, uint32_t period, uint32_t seed) {
    float x = u * period, y = v * period;
    int ix = (int)floorf(x), iy = (int)floorf(y);
    float fx = x - ix, fy = y - iy;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    auto H = [&](int i, int j) {
        uint32_t a = (uint32_t)(((i % (int)period) + (int)period) % (int)period), b = (uint32_t)(((j % (int)period) + (int)period) % (int)period);
        return h01(a * 73856093u ^ hash(b * 19349663u + seed));
    };
    float a = H(ix, iy), b = H(ix + 1, iy), c = H(ix, iy + 1), d = H(ix + 1, iy + 1);
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}
float tfbm(float u, float v, uint32_t period, int oct, uint32_t seed) {
    float s = 0, a = 0.5f, n = 0;
    for (int i = 0; i < oct; i++) { s += a * tnoise(u, v, period << i, seed + i * 1013u); n += a; a *= 0.5f; }
    return s / n;
}
// non-tiling noise in texel units (used inside a stone, whose stamp is evaluated in unwrapped coordinates)
float vn(float x, float y, uint32_t seed) {
    int ix = (int)floorf(x), iy = (int)floorf(y);
    float fx = x - ix, fy = y - iy;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    auto H = [&](int i, int j) { return h01((uint32_t)i * 73856093u ^ hash((uint32_t)j * 19349663u + seed)); };
    float a = H(ix, iy), b = H(ix + 1, iy), c = H(ix, iy + 1), d = H(ix + 1, iy + 1);
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

enum Shape { PEBBLE, ANGULAR, CHIP, COBBLE, STRAW };
enum Rock { GRANITE, BASALT, SANDSTONE, QUARTZ, RED, LIMESTONE, ROCK_COUNT };
const C3 rockBase[ROCK_COUNT] = {{0.40f, 0.39f, 0.37f}, {0.13f, 0.13f, 0.14f}, {0.52f, 0.42f, 0.29f},
                                 {0.72f, 0.70f, 0.66f}, {0.46f, 0.28f, 0.20f}, {0.60f, 0.58f, 0.52f}};

struct Stone {
    float cx, cy, R, aspect, rot, H, bury, n;  // centre / radius in texels, height in metres
    Shape shape; Rock rock;
    float harm[5], phase[5];                   // boundary wobble (lumpy outlines)
    float polyN[8][2], polyR[8]; int poly;     // convex facets for angular rocks
    C3 tint; uint32_t seed;
};

struct Canvas {
    uint32_t N; float texel;  // metres per texel
    std::vector<float> h; std::vector<C3> c; std::vector<uint8_t> mask;
};

void stamp(Canvas& cv, const Stone& s) {
    float ext = s.R * std::max(1.f, s.aspect) * 1.35f;
    int x0 = (int)floorf(s.cx - ext), x1 = (int)ceilf(s.cx + ext), y0 = (int)floorf(s.cy - ext), y1 = (int)ceilf(s.cy + ext);
    float cr = cosf(s.rot), sr = sinf(s.rot);
    int N = (int)cv.N;
    for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) {
        float dx = x + 0.5f - s.cx, dy = y + 0.5f - s.cy;
        float lx = (dx * cr + dy * sr) / (s.R * s.aspect), ly = (-dx * sr + dy * cr) / s.R;
        float d;
        if (s.shape == ANGULAR) {
            d = 0;
            for (int i = 0; i < s.poly; i++) d = std::max(d, (lx * s.polyN[i][0] + ly * s.polyN[i][1]) / s.polyR[i]);
        } else {
            d = powf(powf(fabsf(lx), s.n) + powf(fabsf(ly), s.n), 1.f / s.n);
            float th = atan2f(ly, lx), b = 1.f;
            for (int k = 0; k < 5; k++) b += s.harm[k] * cosf((k + 2) * th + s.phase[k]);
            d /= b;
        }
        d += (vn(dx * 0.35f, dy * 0.35f, s.seed) - 0.5f) * 0.08f;  // fine edge roughness
        if (d >= 1.f) continue;
        float t = 1.f - d, z;
        switch (s.shape) {
        case PEBBLE: z = powf(std::max(0.f, 1.f - d * d), 0.6f); break;
        case COBBLE: z = powf(std::max(0.f, 1.f - powf(d, 2.6f)), 0.45f); break;
        case ANGULAR: z = std::min(1.f, t * 2.6f) * (0.8f + 0.2f * vn(dx * 0.08f, dy * 0.08f, s.seed + 7)); break;
        case CHIP: z = smooth(0.f, 0.25f, t) * (0.9f + 0.1f * vn(dx * 0.1f, dy * 0.1f, s.seed + 3)); break;
        default: z = sqrtf(std::max(0.f, 1.f - ly * ly)); break;  // straw: round cross-section
        }
        float grain = vn(dx * 0.9f, dy * 0.9f, s.seed + 11), pit = vn(dx * 0.25f, dy * 0.25f, s.seed + 13);
        float rough = s.rock == BASALT ? -0.06f * smooth(0.6f, 0.8f, pit) : (s.rock == SANDSTONE ? 0.03f * grain : 0.015f * grain);
        float hh = s.H * (z + rough) - s.H * s.bury;
        int wx = ((x % N) + N) % N, wy = ((y % N) + N) % N;
        size_t i = (size_t)wy * N + wx;
        if (hh <= cv.h[i]) continue;
        cv.h[i] = hh;
        // colour: rock family + per-stone tint + surface pattern, dusty where low / near the rim
        C3 col = rockBase[s.rock] * s.tint;
        switch (s.rock) {
        case GRANITE: col = col * (0.75f + 0.5f * grain); if (grain > 0.82f) col = C3{0.08f, 0.08f, 0.08f}; else if (grain < 0.12f) col = C3{0.8f, 0.78f, 0.74f}; break;
        case BASALT: col = col * (1.f - 0.4f * smooth(0.6f, 0.8f, pit)); break;
        case SANDSTONE: {
            float band = sinf((lx * 0.7f + ly) * 9.f + vn(dx * 0.05f, dy * 0.05f, s.seed) * 6.f);
            col = col * (0.85f + 0.15f * band); break;
        }
        case QUARTZ: col = col * (0.9f + 0.15f * vn(dx * 0.2f, dy * 0.2f, s.seed + 5)); break;
        case RED: col = col * (0.8f + 0.35f * grain); break;
        default: col = col * (0.85f + 0.25f * pit); break;
        }
        if (s.shape == STRAW) col = col * (0.85f + 0.3f * vn(dx * 0.03f, dy * 1.f, s.seed));
        float dust = s.shape == STRAW ? 0.f : 0.45f * powf(1.f - z, 3.f);
        cv.c[i] = mix(col, cv.c[i], dust);
        cv.mask[i] = 255;
    }
}

Stone makeStone(std::mt19937& rng, float N, float Rm, float texel, bool soil) {
    std::uniform_real_distribution<float> U(0.f, 1.f);
    Stone s{};
    s.cx = U(rng) * N; s.cy = U(rng) * N;
    s.R = Rm / texel;
    s.rot = U(rng) * 6.2831853f;
    s.seed = (uint32_t)rng();
    float u = U(rng);
    s.shape = u < 0.38f ? PEBBLE : u < 0.68f ? ANGULAR : u < 0.85f ? CHIP : COBBLE;
    s.aspect = s.shape == CHIP ? 1.4f + 2.2f * U(rng) : 1.f + 0.9f * U(rng) * U(rng);
    s.n = s.shape == COBBLE ? 2.5f + 1.5f * U(rng) : 1.7f + 0.8f * U(rng);
    float lumpy = s.shape == COBBLE ? 0.12f : s.shape == PEBBLE ? 0.05f : 0.08f;
    for (int k = 0; k < 5; k++) { s.harm[k] = (U(rng) - 0.5f) * 2.f * lumpy / (1 + k * 0.6f); s.phase[k] = U(rng) * 6.2831853f; }
    s.poly = 5 + (int)(U(rng) * 4);
    for (int i = 0; i < s.poly; i++) {
        float a = (i + 0.5f * U(rng)) / s.poly * 6.2831853f;
        s.polyN[i][0] = cosf(a); s.polyN[i][1] = sinf(a); s.polyR[i] = 0.75f + 0.3f * U(rng);
    }
    float hf = s.shape == CHIP ? 0.18f : s.shape == ANGULAR ? 0.55f : 0.45f;
    s.H = Rm * hf * (0.7f + 0.6f * U(rng));
    s.bury = 0.15f + 0.45f * U(rng) + (soil ? 0.2f : 0.f);
    float r = U(rng);
    s.rock = (Rock)(r < 0.3f ? GRANITE : r < 0.45f ? BASALT : r < 0.65f ? SANDSTONE : r < 0.72f ? QUARTZ : r < 0.82f ? RED : LIMESTONE);
    float j = 0.85f + 0.3f * U(rng);
    s.tint = {j * (0.95f + 0.1f * U(rng)), j, j * (0.95f + 0.1f * U(rng))};
    return s;
}

uint8_t toSrgb(float x) {
    x = clamp01(x);
    float s = x <= 0.0031308f ? 12.92f * x : 1.055f * powf(x, 1.f / 2.4f) - 0.055f;
    return (uint8_t)lrintf(s * 255.f);
}

// separable wrap-around box blur via running sums
void boxBlur(const std::vector<float>& src, std::vector<float>& dst, uint32_t N, int r) {
    std::vector<float> tmp(src.size());
    float inv = 1.f / (2 * r + 1);
    jobs::parallelFor(N, 16, [&](uint32_t b, uint32_t e) {
        for (uint32_t y = b; y < e; y++) {
            const float* row = &src[(size_t)y * N];
            float s = 0;
            for (int k = -r; k <= r; k++) s += row[(k + N) % N];
            for (uint32_t x = 0; x < N; x++) {
                tmp[(size_t)y * N + x] = s * inv;
                s += row[(x + r + 1) % N] - row[(x + N - r) % N];
            }
        }
    });
    dst.resize(src.size());
    jobs::parallelFor(N, 16, [&](uint32_t b, uint32_t e) {
        for (uint32_t x = b; x < e; x++) {
            float s = 0;
            for (int k = -r; k <= r; k++) s += tmp[(size_t)((k + N) % N) * N + x];
            for (uint32_t y = 0; y < N; y++) {
                dst[(size_t)y * N + x] = s * inv;
                s += tmp[(size_t)((y + r + 1) % N) * N + x] - tmp[(size_t)((y + N - r) % N) * N + x];
            }
        }
    });
}
}  // namespace

Set generate(Kind k, const Params& p) {
    const uint32_t N = p.size;
    const bool soil = k == GROUND_SOIL;
    Canvas cv{N, p.tile / N, std::vector<float>((size_t)N * N), std::vector<C3>((size_t)N * N), std::vector<uint8_t>((size_t)N * N, 0)};
    uint32_t seed = hash(p.seed * 7919u + k * 104729u);

    // base: soil height (a few mm of clumps and crumbs) and colour (damp patches, sandy grains, organic dark)
    jobs::parallelFor(N, 8, [&](uint32_t b, uint32_t e) {
        for (uint32_t y = b; y < e; y++) for (uint32_t x = 0; x < N; x++) {
            float u = (x + 0.5f) / N, v = (y + 0.5f) / N;
            float big = tfbm(u, v, 8, 4, seed), mid = tfbm(u, v, 64, 3, seed + 50), fine = tnoise(u, v, 1024, seed + 90);
            float crumbs = smooth(0.62f, 0.8f, tnoise(u, v, 512, seed + 70));
            size_t i = (size_t)y * N + x;
            cv.h[i] = 0.004f * big + 0.003f * mid + 0.0015f * fine + 0.002f * crumbs;
            float damp = smooth(0.4f, 0.7f, tfbm(u, v, 4, 3, seed + 30));
            C3 c = soil ? mix(C3{0.16f, 0.11f, 0.07f}, C3{0.09f, 0.065f, 0.045f}, damp) : mix(C3{0.27f, 0.21f, 0.15f}, C3{0.17f, 0.13f, 0.09f}, damp);
            c = c * (0.8f + 0.4f * mid) * (0.9f + 0.2f * fine);
            if (!soil) c = mix(c, C3{0.36f, 0.33f, 0.28f}, 0.4f * smooth(0.55f, 0.8f, tfbm(u, v, 16, 3, seed + 40)));  // gray gravelly patches
            if (fine > 0.93f) c = soil ? C3{0.25f, 0.2f, 0.12f} : C3{0.45f, 0.42f, 0.36f};                                // sand grains
            cv.c[i] = c;
        }
    });

    // scatter: size classes from cobbles down to grit; density varies across the tile so it clumps
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    float area = p.tile * p.tile, sm = p.stones * (soil ? 0.25f : 1.f);
    struct Cls { float rmin, rmax, perM2; };
    const Cls cls[] = {{0.06f, 0.14f, 6.f}, {0.02f, 0.055f, 70.f}, {0.006f, 0.018f, 600.f}, {0.0015f, 0.005f, 5000.f}};
    std::vector<Stone> stones;
    for (const Cls& c : cls) {
        int n = (int)(c.perM2 * area * sm);
        for (int i = 0; i < n; i++) {
            float r = c.rmin * powf(c.rmax / c.rmin, U(rng) * U(rng));  // more small than large
            Stone s = makeStone(rng, (float)N, r, cv.texel, soil);
            float dens = tfbm(s.cx / N, s.cy / N, 4, 3, seed + 77);   // clumping
            if (U(rng) > smooth(0.25f, 0.65f, dens) * 1.3f) continue;
            // height map: in some patches stones stand proud, in others they are pressed flush into the soil
            float pop = smooth(0.2f, 0.8f, tfbm(s.cx / N, s.cy / N, 6, 3, seed + 123));
            float popFine = tnoise(s.cx / N, s.cy / N, 48, seed + 131);
            s.H *= (0.3f + 1.6f * pop) * (0.75f + 0.5f * popFine);
            s.bury = std::min(0.85f, s.bury + 0.35f * (1.f - pop));
            stones.push_back(s);
        }
    }
    if (soil) {  // dead grass straws and twig bits
        int n = (int)(900 * area);
        for (int i = 0; i < n; i++) {
            Stone s = makeStone(rng, (float)N, 0.0008f + 0.0012f * U(rng), cv.texel, true);
            s.shape = STRAW; s.aspect = 10.f + 30.f * U(rng); s.n = 2.f; s.H = 0.0012f; s.bury = 0.f;
            for (float& h : s.harm) h = 0.f;
            s.rock = LIMESTONE;
            float t = U(rng);
            s.tint = t < 0.5f ? C3{0.75f, 0.62f, 0.38f} : C3{0.45f, 0.33f, 0.2f};
            stones.push_back(s);
        }
    }
    std::sort(stones.begin(), stones.end(), [](const Stone& a, const Stone& b) { return a.R > b.R; });
    for (auto& s : stones) stamp(cv, s);

    // height -> normal, AO (cavity from blurred height), normalised height for parallax
    float hmin = 1e9f, hmax = -1e9f;
    for (float h : cv.h) { hmin = std::min(hmin, h); hmax = std::max(hmax, h); }
    std::vector<float> b1, b2;
    boxBlur(cv.h, b1, N, 6);
    boxBlur(cv.h, b2, N, 40);
    Set out;
    out.size = N;
    out.maxHeight = hmax - hmin;
    out.albedo.resize((size_t)N * N * 4);
    out.detail.resize((size_t)N * N * 4);
    float inv2t = 1.f / (2.f * cv.texel);
    jobs::parallelFor(N, 16, [&](uint32_t b, uint32_t e) {
        for (uint32_t y = b; y < e; y++) for (uint32_t x = 0; x < N; x++) {
            size_t i = (size_t)y * N + x;
            float dx = (cv.h[(size_t)y * N + (x + 1) % N] - cv.h[(size_t)y * N + (x + N - 1) % N]) * inv2t;
            float dy = (cv.h[(size_t)((y + 1) % N) * N + x] - cv.h[(size_t)((y + N - 1) % N) * N + x]) * inv2t;
            float il = 1.f / sqrtf(dx * dx + dy * dy + 1.f);
            float nx = -dx * il, nz = -dy * il;
            float cav = std::max(0.f, b1[i] - cv.h[i]) * 1.f / 0.004f + std::max(0.f, b2[i] - cv.h[i]) / 0.02f;
            float ao = clamp01(1.f - 0.7f * clamp01(cav));
            C3 c = cv.c[i] * (0.55f + 0.45f * ao);
            out.albedo[4 * i] = toSrgb(c.r); out.albedo[4 * i + 1] = toSrgb(c.g); out.albedo[4 * i + 2] = toSrgb(c.b);
            out.albedo[4 * i + 3] = cv.mask[i];
            out.detail[4 * i] = (uint8_t)lrintf(clamp01(nx * 0.5f + 0.5f) * 255.f);
            out.detail[4 * i + 1] = (uint8_t)lrintf(clamp01(nz * 0.5f + 0.5f) * 255.f);
            out.detail[4 * i + 2] = (uint8_t)lrintf(clamp01((cv.h[i] - hmin) / std::max(out.maxHeight, 1e-6f)) * 255.f);
            out.detail[4 * i + 3] = (uint8_t)lrintf(ao * 255.f);
        }
    });
    return out;
}

bool load(Kind k, const Params& p, const std::string& dir, Set& out) {
    char name[160];
    snprintf(name, sizeof name, "%s/ground%d_%u_%.2f_%u_%.2f_v%u.bin", dir.c_str(), (int)k, p.size, p.tile, p.seed, p.stones, VERSION);
    if (FILE* f = fopen(name, "rb")) {
        uint32_t hdr[2]; float mh;
        bool ok = fread(hdr, 4, 2, f) == 2 && hdr[0] == 0x54475031u && hdr[1] == p.size && fread(&mh, 4, 1, f) == 1;
        if (ok) {
            size_t n = (size_t)p.size * p.size * 4;
            out.size = p.size; out.maxHeight = mh;
            out.albedo.resize(n); out.detail.resize(n);
            ok = fread(out.albedo.data(), 1, n, f) == n && fread(out.detail.data(), 1, n, f) == n;
        }
        fclose(f);
        if (ok) return true;
    }
    out = generate(k, p);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (FILE* f = fopen(name, "wb")) {
        uint32_t hdr[2] = {0x54475031u, p.size};
        fwrite(hdr, 4, 2, f);
        fwrite(&out.maxHeight, 4, 1, f);
        fwrite(out.albedo.data(), 1, out.albedo.size(), f);
        fwrite(out.detail.data(), 1, out.detail.size(), f);
        fclose(f);
    }
    return false;
}
}  // namespace texgen
