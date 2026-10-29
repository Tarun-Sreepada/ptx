#include "world/worldmap.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <unordered_map>

#include "world/terrain.h"

namespace worldmap {
namespace {
constexpr int WN = N / 2;            // water grid: 4 m cells (rivers are wider than that)
constexpr float WCELL = CELL * 2.f;
std::string dir;
std::vector<float> H(size_t(N) * N, 0.f);
std::vector<Paint> P(size_t(N) * N);
std::vector<Water> W(size_t(WN) * WN);
std::string palette[PALETTE];
Rect dirtyR{1, 1, 0, 0};
bool unsaved = false;
uint32_t ver = 1;

// undo: the first value of every cell a stroke touched
struct Undo { std::unordered_map<uint32_t, std::pair<float, Paint>> cells; std::unordered_map<uint32_t, Water> water; };
std::vector<Undo> undos;

void markDirty(int x0, int z0, int x1, int z1) {
    x0 = std::max(x0, 0); z0 = std::max(z0, 0); x1 = std::min(x1, N - 1); z1 = std::min(z1, N - 1);
    if (x0 > x1 || z0 > z1) return;
    if (dirtyR.x0 > dirtyR.x1) dirtyR = {x0, z0, x1, z1};
    else dirtyR = {std::min(dirtyR.x0, x0), std::min(dirtyR.z0, z0), std::max(dirtyR.x1, x1), std::max(dirtyR.z1, z1)};
    unsaved = true;
    ver++;
}
void touch(uint32_t i) {
    if (undos.empty()) return;
    undos.back().cells.try_emplace(i, H[i], P[i]);
}
void touchW(uint32_t i) {
    if (undos.empty()) return;
    undos.back().water.try_emplace(i, W[i]);
}
// bilinear sample of a cell-centred grid
template <class F> float sample(float x, float z, int n, float cell, float outside, F get) {
    float fx = (x - ORIGIN) / cell - 0.5f, fz = (z - ORIGIN) / cell - 0.5f;
    if (!(fx >= 0 && fz >= 0 && fx < n - 1 && fz < n - 1)) return outside;  // written so NaN / inf land outside too
    int ix = (int)fx, iz = (int)fz;
    float tx = fx - ix, tz = fz - iz;
    float a = get(ix, iz), b = get(ix + 1, iz), c = get(ix, iz + 1), d = get(ix + 1, iz + 1);
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * tz;
}
float cellX(int i) { return ORIGIN + (i + 0.5f) * CELL; }
}  // namespace

void init(const std::string& worldDir) {
    dir = worldDir + "terrain/";
    // the world comes from the world seed alone now: hand edits in map.bin are no longer loaded (the editor is off too)
    constexpr bool LOAD_HAND_EDITS = false;
    FILE* f = LOAD_HAND_EDITS ? fopen((dir + "map.bin").c_str(), "rb") : nullptr;
    if (!f) return;
    char magic[8] = {};
    int n = 0;
    bool ok = fread(magic, 1, 8, f) == 8 && !memcmp(magic, "PTXMAP1", 7) && fread(&n, 4, 1, f) == 1 && n == N;
    for (int i = 0; ok && i < PALETTE; i++) {
        char name[64] = {};
        ok = fread(name, 1, 64, f) == 64;
        palette[i] = name;
    }
    ok = ok && fread(H.data(), sizeof(float), H.size(), f) == H.size() && fread(P.data(), sizeof(Paint), P.size(), f) == P.size() &&
         fread(W.data(), sizeof(Water), W.size(), f) == W.size();
    fclose(f);
    if (!ok) {
        fprintf(stderr, "worldmap: %smap.bin unreadable, starting flat\n", dir.c_str());
        std::fill(H.begin(), H.end(), 0.f);
        std::fill(P.begin(), P.end(), Paint{});
        std::fill(W.begin(), W.end(), Water{});
    }
    markDirty(0, 0, N - 1, N - 1);
    unsaved = false;
}

bool save(std::string& err) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::string tmp = dir + "map.bin.tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) { err = "can't write " + tmp; return false; }
    int n = N;
    fwrite("PTXMAP1", 1, 8, f);
    fwrite(&n, 4, 1, f);
    for (int i = 0; i < PALETTE; i++) {
        char name[64] = {};
        snprintf(name, sizeof name, "%s", palette[i].c_str());
        fwrite(name, 1, 64, f);
    }
    fwrite(H.data(), sizeof(float), H.size(), f);
    fwrite(P.data(), sizeof(Paint), P.size(), f);
    fwrite(W.data(), sizeof(Water), W.size(), f);
    bool ok = !ferror(f);
    fclose(f);
    if (!ok) { err = "write failed"; return false; }
    std::filesystem::rename(tmp, dir + "map.bin", ec);
    if (ec) { err = ec.message(); return false; }
    unsaved = false;
    return true;
}
bool dirty() { return unsaved; }

float height(float x, float z) { return sample(x, z, N, CELL, 0.f, [](int i, int k) { return H[(size_t)k * N + i]; }); }
float grassMul(float x, float z) { return sample(x, z, N, CELL, 1.f, [](int i, int k) { return P[(size_t)k * N + i].grass / 128.f; }); }
float treeMul(float x, float z) { return sample(x, z, N, CELL, 1.f, [](int i, int k) { return P[(size_t)k * N + i].trees / 128.f; }); }
float waterLevel(float x, float z) {
    // nearest cell (levels don't blend with "none")
    float fi = floorf((x - ORIGIN) / WCELL), fk = floorf((z - ORIGIN) / WCELL);
    if (!(fi >= 0 && fk >= 0 && fi < WN && fk < WN)) return -1000.f;
    int i = (int)fi, k = (int)fk;
    return W[(size_t)k * WN + i].level;
}
const std::string& paletteName(int i) { static std::string none; return i > 0 && i < PALETTE ? palette[i] : none; }
void setPalette(int i, const std::string& name) {
    if (i <= 0 || i >= PALETTE) return;
    palette[i] = name;
    unsaved = true;
    ver++;
    markDirty(0, 0, 0, 0);  // the renderer re-reads the palette with any upload
}

void beginStroke() {
    undos.push_back({});
    if (undos.size() > 16) undos.erase(undos.begin());
}
bool undo() {
    while (!undos.empty() && undos.back().cells.empty() && undos.back().water.empty()) undos.pop_back();
    if (undos.empty()) return false;
    Undo u = std::move(undos.back());
    undos.pop_back();
    int x0 = N, z0 = N, x1 = -1, z1 = -1;
    for (auto& [i, v] : u.cells) {
        H[i] = v.first;
        P[i] = v.second;
        int x = i % N, z = i / N;
        x0 = std::min(x0, x); x1 = std::max(x1, x); z0 = std::min(z0, z); z1 = std::max(z1, z);
    }
    for (auto& [i, w] : u.water) {
        W[i] = w;
        int x = (i % WN) * 2, z = (i / WN) * 2;
        x0 = std::min(x0, x); x1 = std::max(x1, x + 1); z0 = std::min(z0, z); z1 = std::max(z1, z + 1);
    }
    markDirty(x0, z0, x1, z1);
    return true;
}

void brush(Tool t, float x, float z, float radius, float strength, float dt, float flattenTo, int mat) {
    if (!std::isfinite(x) || !std::isfinite(z) || !std::isfinite(radius) || radius <= 0.f) return;
    int cx0 = (int)floorf((x - radius - ORIGIN) / CELL), cx1 = (int)floorf((x + radius - ORIGIN) / CELL);
    int cz0 = (int)floorf((z - radius - ORIGIN) / CELL), cz1 = (int)floorf((z + radius - ORIGIN) / CELL);
    cx0 = std::max(cx0, 1); cz0 = std::max(cz0, 1); cx1 = std::min(cx1, N - 2); cz1 = std::min(cz1, N - 2);
    if (cx0 > cx1 || cz0 > cz1) return;
    std::vector<float> src;
    if (t == SMOOTH) src = H;  // read the pre-stroke-step heights (a copy of the whole grid is ~16 MB: only per step)
    for (int k = cz0; k <= cz1; k++)
        for (int i = cx0; i <= cx1; i++) {
            float wx = cellX(i), wz = cellX(k), d = hypotf(wx - x, wz - z) / radius;
            if (d >= 1.f) continue;
            float w = 1.f - d * d * (3.f - 2.f * d);  // smooth falloff
            uint32_t c = (uint32_t)k * N + i;
            touch(c);
            float a = std::min(strength * dt * w, 1.f);
            auto paint = [&](uint8_t& v, float target) { v = (uint8_t)std::clamp(v + (target - v) * a, 0.f, 255.f); };
            switch (t) {
            case RAISE: H[c] += strength * dt * w; break;
            case LOWER: H[c] -= strength * dt * w; break;
            case SMOOTH: {
                float s = 0;
                for (int dz = -1; dz <= 1; dz++)
                    for (int dx = -1; dx <= 1; dx++) s += src[(size_t)(k + dz) * N + i + dx];
                H[c] += (s / 9.f - H[c]) * a;
                break;
            }
            case FLATTEN: H[c] += (flattenTo - (naturalHeight(wx, wz) + H[c])) * a; break;
            case NOISE: H[c] += strength * dt * w * (vnoise(wx * 0.08f, wz * 0.08f) * 2.f - 1.f) * 2.f; break;
            case PAINT_GRASS: paint(P[c].grass, 255.f); break;
            case ERASE_GRASS: paint(P[c].grass, 0.f); break;
            case PAINT_TREES: paint(P[c].trees, 255.f); break;
            case ERASE_TREES: paint(P[c].trees, 0.f); break;
            case PAINT_TEXTURE:
                if (P[c].mat != mat) { if (P[c].matW < 8) P[c].mat = (uint8_t)mat; else { paint(P[c].matW, 0.f); break; } }
                paint(P[c].matW, 255.f);
                break;
            case ERASE_TEXTURE: paint(P[c].matW, 0.f); break;
            case ERASE_WATER: {
                uint32_t wi = (uint32_t)(k / 2) * WN + i / 2;
                touchW(wi);
                if (w > 0.3f) W[wi] = Water{};
                break;
            }
            default: break;
            }
        }
    markDirty(cx0 - 1, cz0 - 1, cx1 + 1, cz1 + 1);
}

void addRiver(const std::vector<RiverPt>& in) {
    if (in.size() < 2) return;
    // resample every metre; the level only ever falls along the drawn direction (water runs downhill)
    std::vector<RiverPt> pts;
    std::vector<float> level;
    for (size_t s = 0; s + 1 < in.size(); s++) {
        float L = hypotf(in[s + 1].x - in[s].x, in[s + 1].z - in[s].z);
        int n = std::max(1, (int)ceilf(L));
        for (int j = 0; j < n; j++) {
            float u = (float)j / n;
            pts.push_back({in[s].x + (in[s + 1].x - in[s].x) * u, in[s].z + (in[s + 1].z - in[s].z) * u,
                           in[s].width + (in[s + 1].width - in[s].width) * u, in[s].depth + (in[s + 1].depth - in[s].depth) * u});
        }
    }
    pts.push_back(in.back());
    for (size_t i = 0; i < pts.size(); i++) {
        float g = naturalHeight(pts[i].x, pts[i].z) + height(pts[i].x, pts[i].z);
        float lv = g - 0.25f * pts[i].depth;
        level.push_back(i ? std::min(level.back(), lv) : lv);
    }
    const float BANK = 6.f;
    float maxW = 0;
    float bx0 = 1e30f, bz0 = 1e30f, bx1 = -1e30f, bz1 = -1e30f;
    for (auto& p : pts) {
        maxW = std::max(maxW, p.width);
        bx0 = std::min(bx0, p.x); bx1 = std::max(bx1, p.x); bz0 = std::min(bz0, p.z); bz1 = std::max(bz1, p.z);
    }
    float m = maxW + BANK + CELL;
    int cx0 = std::max(1, (int)floorf((bx0 - m - ORIGIN) / CELL)), cx1 = std::min(N - 2, (int)floorf((bx1 + m - ORIGIN) / CELL));
    int cz0 = std::max(1, (int)floorf((bz0 - m - ORIGIN) / CELL)), cz1 = std::min(N - 2, (int)floorf((bz1 + m - ORIGIN) / CELL));
    auto nearest = [&](float x, float z, float& dist, float& u, size_t& seg) {
        dist = 1e30f;
        for (size_t s = 0; s + 1 < pts.size(); s++) {
            float ax = pts[s].x, az = pts[s].z, bx = pts[s + 1].x - ax, bz = pts[s + 1].z - az;
            float l2 = bx * bx + bz * bz, t = l2 > 0 ? std::clamp(((x - ax) * bx + (z - az) * bz) / l2, 0.f, 1.f) : 0.f;
            float d = hypotf(x - ax - bx * t, z - az - bz * t);
            if (d < dist) { dist = d; u = t; seg = s; }
        }
    };
    for (int k = cz0; k <= cz1; k++)
        for (int i = cx0; i <= cx1; i++) {
            float wx = cellX(i), wz = cellX(k), d, u;
            size_t s;
            nearest(wx, wz, d, u, s);
            float w = pts[s].width + (pts[s + 1].width - pts[s].width) * u, dp = pts[s].depth + (pts[s + 1].depth - pts[s].depth) * u;
            float lv = level[s] + (level[s + 1] - level[s]) * u;
            if (d > w + BANK) continue;
            uint32_t c = (uint32_t)k * N + i;
            float nat = naturalHeight(wx, wz), cur = nat + H[c];
            float c2 = d / std::max(w, 0.1f);
            float prof = d < w ? lv - dp * (1.f - c2 * c2) : lv + 0.35f * (d - w);
            float tb = std::clamp((d - w) / BANK, 0.f, 1.f);
            float target = prof + (cur - prof) * tb * tb * (3 - 2 * tb);
            if (d < w || target < cur) {  // carve (banks only cut, never fill)
                touch(c);
                H[c] = (d < w ? std::min(target, cur) : target) - nat;
            }
            if (d < w + 0.5f * WCELL) {  // the water surface and its flow (downstream along the stroke)
                uint32_t wi = (uint32_t)(k / 2) * WN + i / 2;
                touchW(wi);
                float tx = pts[s + 1].x - pts[s].x, tz = pts[s + 1].z - pts[s].z, tl = std::max(hypotf(tx, tz), 1e-4f);
                float slope = std::max(level[std::max<size_t>(s, 1) - 1] - level[std::min(s + 2, pts.size() - 1)], 0.f) / 3.f;
                // flow follows the pen (first point -> last); drawn uphill the bed is cut deeper instead and it still runs
                float speed = std::clamp(0.8f + slope * 60.f, 0.8f, 3.f) * (1.f - 0.7f * c2 * c2);
                W[wi] = Water{lv, tx / tl * speed, tz / tl * speed, w};
            }
        }
    markDirty(cx0 - 1, cz0 - 1, cx1 + 1, cz1 + 1);
}

Rect takeDirty() { Rect r = dirtyR; dirtyR = {1, 1, 0, 0}; return r; }
const float* heightData() { return H.data(); }
const Paint* paintData() { return P.data(); }
const Water* waterData() { return W.data(); }
uint32_t version() { return ver; }
}  // namespace worldmap

// terrain.h: the CPU height adds the edits
float editHeight(float x, float z) { return worldmap::height(x, z); }
