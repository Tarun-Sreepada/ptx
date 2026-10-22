#include "matlib.h"

#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>

namespace matlib {
namespace {

constexpr uint32_t N = 256;  // texels per metre

std::string texDir;
std::vector<Entry> entries;
std::map<std::string, TextureData> cache;

uint32_t ihash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
float h2(int x, int y, uint32_t s = 0) { return ihash((uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^ s * 83492791u) / 4294967296.f; }
// value noise, tiling with period p cells
float tnoise(float x, float y, int p, uint32_t s = 0) {
    int ix = (int)floorf(x), iy = (int)floorf(y);
    float fx = x - ix, fy = y - iy;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    auto g = [&](int a, int b) { return h2(((a % p) + p) % p, ((b % p) + p) % p, s); };
    float a = g(ix, iy), b = g(ix + 1, iy), c = g(ix, iy + 1), d = g(ix + 1, iy + 1);
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}
// fractal tiling noise over the unit tile (u, v in 0..1)
float fbm(float u, float v, int base, int oct, uint32_t s = 0) {
    float r = 0, amp = 0.5f, tot = 0;
    for (int o = 0; o < oct; o++, base *= 2, amp *= 0.5f) { r += amp * tnoise(u * base, v * base, base, s + o); tot += amp; }
    return r / tot;
}
uint8_t srgb(float lin) {
    lin = std::clamp(lin, 0.f, 1.f);
    float s = lin <= 0.0031308f ? lin * 12.92f : 1.055f * powf(lin, 1.f / 2.4f) - 0.055f;
    return (uint8_t)(s * 255.f + 0.5f);
}
struct C { float r, g, b; };
C mix(C a, C b, float t) { return {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t}; }
C mul(C a, float k) { return {a.r * k, a.g * k, a.b * k}; }

// running-bond blocks: rows x cols per tile, joint width j (fraction of a block). Returns block id and joint 0..1.
float blocks(float u, float v, int rows, int cols, float jx, float jy, int& bx, int& by) {
    float y = v * rows;
    by = (int)floorf(y);
    float x = u * cols + (by & 1) * 0.5f;
    bx = ((int)floorf(x)) % cols;
    float fx = x - floorf(x), fy = y - by;
    float dx = std::min(fx, 1 - fx), dy = std::min(fy, 1 - fy);
    return std::max(1.f - std::clamp(dx / jx, 0.f, 1.f), 1.f - std::clamp(dy / jy, 0.f, 1.f));
}

// Albedo of procedural material n at (u, v), plus its relief height h (0..1), roughness r and metalness m.
C texel(const std::string& n, float u, float v, float& h, float& r, float& m) {
    int bx, by;
    m = 0.f;
    if (n == "brick") {
        float j = blocks(u, v, 16, 4, 0.03f, 0.1f, bx, by);
        float k = h2(bx, by, 1);
        C brick = mix({0.32f, 0.09f, 0.05f}, {0.45f, 0.17f, 0.09f}, k);
        float g = fbm(u, v, 32, 3, 2);
        brick = mul(brick, 0.8f + 0.4f * g);
        if (h2(bx, by, 9) < 0.08f) brick = mul(brick, 0.6f);  // the odd burnt brick
        C mortar = mul({0.55f, 0.53f, 0.48f}, 0.85f + 0.3f * fbm(u, v, 64, 2, 3));
        h = (1.f - std::clamp((j - 0.3f) / 0.5f, 0.f, 1.f)) * (0.85f + 0.15f * g);  // bevelled bricks, recessed joints
        r = j > 0.5f ? 0.95f : 0.82f - 0.1f * g;
        return mix(brick, mortar, j > 0.5f ? 1.f : j * 2.f * 0.3f);
    }
    if (n == "concrete") {
        float k = fbm(u, v, 8, 5, 4);
        C c = mul({0.42f, 0.41f, 0.39f}, 0.75f + 0.5f * k);
        bool pore = fbm(u, v, 128, 1, 5) > 0.82f;
        if (pore) c = mul(c, 0.55f);
        h = pore ? 0.f : 0.4f + 0.6f * fbm(u, v, 32, 3, 19);
        r = 0.88f + 0.08f * k;
        return c;
    }
    if (n == "plaster") {
        float k = fbm(u, v, 16, 4, 6);
        h = k;
        r = 0.8f + 0.1f * k;
        return mul({0.72f, 0.69f, 0.62f}, 0.9f + 0.18f * k);
    }
    if (n == "planks") {
        int b = (int)floorf(u * 8);
        float fu = u * 8 - b;
        float off = h2(b, 0, 7);
        int seg = (int)floorf(v * 2 + off);  // boards 0.5 m long, staggered
        float fv = v * 2 + off - seg;
        float grain = 0.5f + 0.5f * sinf((fu * 3.f + 6.f * tnoise(u * 8, v * 4, 4, 8 + b)) * 6.2831f * 3.f);
        C w = mix({0.30f, 0.17f, 0.08f}, {0.42f, 0.26f, 0.12f}, h2(b, seg, 3));
        w = mul(w, 0.8f + 0.25f * grain);
        float edge = std::min({fu, 1 - fu, (1 - fv) * 4.f, fv * 4.f});
        float seam = std::min({fu, 1 - fu}) < 0.03f || std::min(fv, 1 - fv) < 0.012f ? 1.f : 0.f;
        h = seam > 0.f ? 0.f : std::min(edge / 0.06f, 1.f) * (0.9f + 0.1f * grain);
        r = 0.55f + 0.2f * grain;
        return mix(w, {0.06f, 0.04f, 0.02f}, seam);
    }
    if (n == "stone") {
        float j = blocks(u, v, 4, 3, 0.04f, 0.05f, bx, by);
        C s = mix({0.30f, 0.29f, 0.26f}, {0.48f, 0.45f, 0.40f}, h2(bx, by, 11));
        float g = fbm(u, v, 16, 4, 12);
        s = mul(s, 0.7f + 0.5f * g);
        h = (1.f - std::clamp(j * 1.4f - 0.2f, 0.f, 1.f)) * (0.6f + 0.4f * g);  // pillowed blocks
        r = 0.75f + 0.15f * g;
        return mix(s, {0.12f, 0.11f, 0.10f}, std::clamp(j * 1.5f - 0.3f, 0.f, 1.f));
    }
    if (n == "metal") {
        float streak = tnoise(u * 2, v * 128, 128, 13) * 0.6f + fbm(u, v, 8, 3, 14) * 0.4f;
        C c = mul({0.62f, 0.63f, 0.65f}, 0.85f + 0.2f * streak);
        float pu = fmodf(u * 2, 1.f), pv = fmodf(v * 1, 1.f);
        bool seam = std::min({pu, 1 - pu, pv, 1 - pv}) < 0.006f;
        if (seam) c = mul(c, 0.4f);  // panel seams
        h = seam ? 0.f : 1.f;
        r = 0.3f + 0.25f * streak;   // brushed: rougher along the streaks
        m = seam ? 0.f : 1.f;
        return c;
    }
    if (n == "tiles") {
        float fx = fmodf(u * 8, 1.f), fy = fmodf(v * 8, 1.f);
        float e = std::min({fx, 1 - fx, fy, 1 - fy});
        float g = e < 0.035f ? 1.f : 0.f;
        C t = mul({0.78f, 0.78f, 0.75f}, 0.93f + 0.1f * h2((int)floorf(u * 8), (int)floorf(v * 8), 15));
        h = std::clamp((e - 0.03f) / 0.03f, 0.f, 1.f);  // rounded tile edges into the grout
        r = g > 0.f ? 0.9f : 0.12f;                     // glazed
        return mix(t, {0.35f, 0.34f, 0.32f}, g);
    }
    if (n == "roof") {
        float y = v * 10;
        int row = (int)floorf(y);
        float x = u * 6 + (row & 1) * 0.5f;
        int col = (int)floorf(x);
        float fx = x - col, fy = y - row;
        C c = mix({0.25f, 0.07f, 0.04f}, {0.36f, 0.12f, 0.06f}, h2(col, row, 16));
        c = mul(c, 0.6f + 0.5f * fy);  // each shingle shaded down to the one overlapping it
        bool gap = std::min(fx, 1 - fx) < 0.025f;
        if (gap) c = mul(c, 0.4f);
        h = gap ? 0.f : 1.f - fy;  // shingles overlap: thick at the bottom edge, thin under the next row
        r = 0.7f;
        return c;
    }
    if (n == "asphalt") {
        float k = fbm(u, v, 32, 4, 17);
        C a = mul({0.07f, 0.07f, 0.07f}, 0.7f + 0.6f * k);
        bool chip = h2((int)(u * 256), (int)(v * 256), 18) > 0.97f;
        if (chip) a = {0.25f, 0.25f, 0.24f};
        h = chip ? 1.f : k * 0.7f;
        r = 0.85f;
        return a;
    }
    // unknown: checker
    h = 0.5f;
    r = 0.8f;
    return ((int)floorf(u * 8) + (int)floorf(v * 8)) & 1 ? C{0.5f, 0.5f, 0.5f} : C{0.3f, 0.3f, 0.3f};
}

struct Proc { const char* name; float strength, relief; };  // relief: height map depth (m)
const Proc PROCS[] = {{"brick", 1.f, 0.012f}, {"concrete", 2.f, 0.003f}, {"plaster", 0.6f, 0.002f}, {"planks", 0.4f, 0.004f},
                      {"stone", 2.5f, 0.03f}, {"metal", 5.f, 0.002f}, {"tiles", 0.8f, 0.004f}, {"roof", 0.5f, 0.02f},
                      {"asphalt", 1.5f, 0.004f}};
float reliefOf(const std::string& n) {
    for (auto& p : PROCS) if (n == p.name) return p.relief;
    return 0.f;
}
std::map<std::string, Maps> mapCache;

TextureData generate(const std::string& n) {
    TextureData t;
    t.w = t.h = N;
    t.rgba.resize(N * N * 4);
    for (uint32_t y = 0; y < N; y++)
        for (uint32_t x = 0; x < N; x++) {
            float h, r, m;
            C c = texel(n, (x + 0.5f) / N, (y + 0.5f) / N, h, r, m);
            uint8_t* p = &t.rgba[(y * N + x) * 4];
            p[0] = srgb(c.r); p[1] = srgb(c.g); p[2] = srgb(c.b); p[3] = 255;
        }
    return t;
}

uint8_t unorm(float v) { return (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f); }

// Normal (from the height field's slopes at `relief` metres per unit height, tiling) + height in alpha, and
// ORM (r cavity occlusion, g roughness, b metalness), all linear.
Maps buildMaps(uint32_t w, uint32_t hh, const std::vector<float>& H, const std::vector<float>& R, const std::vector<float>& M, float relief,
               float metresPerTile) {
    Maps o;
    o.normal.w = o.orm.w = w;
    o.normal.h = o.orm.h = hh;
    o.normal.linear = o.orm.linear = true;
    o.normal.rgba.resize((size_t)w * hh * 4);
    o.orm.rgba.resize((size_t)w * hh * 4);
    float texel = metresPerTile / w;
    auto at = [&](int x, int y) { return H[(size_t)((y + hh) % hh) * w + (x + w) % w]; };
    for (uint32_t y = 0; y < hh; y++)
        for (uint32_t x = 0; x < w; x++) {
            float dx = (at(x + 1, y) - at(x - 1, y)) * relief / (2 * texel), dy = (at(x, y + 1) - at(x, y - 1)) * relief / (2 * texel);
            // uv v runs down the image while the tangent frame's bitangent follows +v: flip the y slope
            float nx = -dx, ny = dy, nz = 1.f, l = sqrtf(nx * nx + ny * ny + nz * nz);
            uint8_t* p = &o.normal.rgba[((size_t)y * w + x) * 4];
            p[0] = unorm(nx / l * 0.5f + 0.5f); p[1] = unorm(ny / l * 0.5f + 0.5f); p[2] = unorm(nz / l * 0.5f + 0.5f);
            p[3] = unorm(H[(size_t)y * w + x]);
            float avg = (at(x - 2, y) + at(x + 2, y) + at(x, y - 2) + at(x, y + 2)) * 0.25f;
            float cav = 1.f - std::clamp((avg - H[(size_t)y * w + x]) * 2.f, 0.f, 0.6f);  // crevices see less sky
            uint8_t* q = &o.orm.rgba[((size_t)y * w + x) * 4];
            q[0] = unorm(cav); q[1] = unorm(R[(size_t)y * w + x]); q[2] = unorm(M[(size_t)y * w + x]); q[3] = 255;
        }
    return o;
}

bool loadFile(const std::string& file, TextureData& t) {
    int w, h, ch;
    stbi_uc* px = stbi_load((texDir + file).c_str(), &w, &h, &ch, 4);
    if (!px) return false;
    t.w = (uint32_t)w; t.h = (uint32_t)h;
    t.rgba.assign(px, px + (size_t)w * h * 4);
    stbi_image_free(px);
    while (t.w > 1024 || t.h > 1024) {  // box-halve big photos
        uint32_t nw = std::max(1u, t.w / 2), nh = std::max(1u, t.h / 2);
        std::vector<uint8_t> n((size_t)nw * nh * 4);
        for (uint32_t y = 0; y < nh; y++)
            for (uint32_t x = 0; x < nw; x++)
                for (int c = 0; c < 4; c++) {
                    uint32_t x1 = std::min(2 * x + 1, t.w - 1), y1 = std::min(2 * y + 1, t.h - 1);
                    n[((size_t)y * nw + x) * 4 + c] = (uint8_t)((t.rgba[((size_t)2 * y * t.w + 2 * x) * 4 + c] + t.rgba[((size_t)2 * y * t.w + x1) * 4 + c] +
                                                                  t.rgba[((size_t)y1 * t.w + 2 * x) * 4 + c] + t.rgba[((size_t)y1 * t.w + x1) * 4 + c] + 2) / 4);
                }
        t.rgba.swap(n);
        t.w = nw; t.h = nh;
    }
    return true;
}

void swatch(const TextureData& t, float out[3]) {
    double s[3] = {0, 0, 0};
    size_t n = t.rgba.size() / 4;
    for (size_t i = 0; i < n; i++) for (int k = 0; k < 3; k++) s[k] += t.rgba[i * 4 + k];
    for (int k = 0; k < 3; k++) out[k] = n ? (float)(s[k] / n / 255.0) : 0.5f;
}

}  // namespace

void init(const std::string& worldDir) {
    texDir = worldDir + "textures/";
    rescan();
}

void rescan() {
    entries.clear();
    for (auto& p : PROCS) {
        Entry e{p.name, p.strength, {}};
        swatch(texture(p.name), e.swatch);
        entries.push_back(e);
    }
    std::error_code ec;
    std::vector<std::string> files;
    for (auto& f : std::filesystem::directory_iterator(texDir, ec)) {
        std::string ext = f.path().extension().string();
        for (auto& c : ext) c = (char)tolower(c);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") files.push_back(f.path().filename().string());
    }
    // PBR sets: <textures>/<name>/albedo.png (+ normal.png, orm.png), e.g. extracted from a glTF texture pack
    for (auto& f : std::filesystem::directory_iterator(texDir, ec))
        if (f.is_directory(ec) && (std::filesystem::exists(f.path() / "albedo.png", ec) || std::filesystem::exists(f.path() / "albedo.jpg", ec)))
            files.push_back("set:" + f.path().filename().string());
    std::sort(files.begin(), files.end());
    for (auto& f : files) {
        if (f.rfind("set:", 0) == 0) {
            Entry e{f, 1.f, {}};
            swatch(texture(e.name), e.swatch);
            entries.push_back(e);
            continue;
        }
        Entry e{"file:" + f, 1.f, {}};
        swatch(texture(e.name), e.swatch);
        entries.push_back(e);
    }
}

const std::vector<Entry>& list() { return entries; }

const Entry* find(const std::string& name) {
    for (auto& e : entries) if (e.name == name) return &e;
    return nullptr;
}

// file of a PBR set ("set:<dir>"): <dir>/<map>.png or .jpg
static bool loadSetMap(const std::string& set, const char* map, TextureData& t) {
    return loadFile(set + "/" + map + ".png", t) || loadFile(set + "/" + map + ".jpg", t);
}

const Maps& maps(const std::string& name) {
    auto it = mapCache.find(name);
    if (it != mapCache.end()) return it->second;
    if (name.rfind("set:", 0) == 0) {  // authored maps where the set has them, the photo fallback otherwise
        Maps m = maps("file:" + name.substr(4) + "/albedo.png");
        if (m.normal.rgba.empty()) m = maps("file:" + name.substr(4) + "/albedo.jpg");
        TextureData t;
        if (loadSetMap(name.substr(4), "normal", t)) {
            for (size_t i = 3; i < t.rgba.size(); i += 4) t.rgba[i] = 255;  // no height: alpha unused (heightScale 0)
            t.linear = true;
            m.normal = std::move(t);
        }
        if (loadSetMap(name.substr(4), "orm", t)) { t.linear = true; m.orm = std::move(t); }
        m.heightScale = 0.f;
        return mapCache.emplace(name, std::move(m)).first->second;
    }
    std::vector<float> H, R, M;
    uint32_t w, h;
    float relief = 0.f;
    if (name.rfind("file:", 0) == 0) {  // photo: luminance as a stand-in height, gentle normals only, no parallax
        const TextureData& t = texture(name);
        w = t.w; h = t.h;
        H.resize((size_t)w * h); R.assign(H.size(), 0.75f); M.assign(H.size(), 0.f);
        for (size_t i = 0; i < H.size(); i++)
            H[i] = (0.2126f * t.rgba[i * 4] + 0.7152f * t.rgba[i * 4 + 1] + 0.0722f * t.rgba[i * 4 + 2]) / 255.f;
        relief = 0.002f;
    } else {
        w = h = N;
        H.resize(N * N); R.resize(N * N); M.resize(N * N);
        for (uint32_t y = 0; y < N; y++)
            for (uint32_t x = 0; x < N; x++) texel(name, (x + 0.5f) / N, (y + 0.5f) / N, H[y * N + x], R[y * N + x], M[y * N + x]);
        relief = reliefOf(name);
    }
    Maps m = buildMaps(w, h, H, R, M, std::max(relief, 1e-4f), 1.f);
    m.heightScale = name.rfind("file:", 0) == 0 ? 0.f : relief;
    return mapCache.emplace(name, std::move(m)).first->second;
}

const TextureData& texture(const std::string& name) {
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    TextureData t;
    if (name.rfind("set:", 0) == 0) {
        if (!loadSetMap(name.substr(4), "albedo", t)) t = generate("");
    } else if (name.rfind("file:", 0) != 0 || !loadFile(name.substr(5), t)) t = generate(name.rfind("file:", 0) == 0 ? "" : name);
    return cache.emplace(name, std::move(t)).first->second;
}

}  // namespace matlib
