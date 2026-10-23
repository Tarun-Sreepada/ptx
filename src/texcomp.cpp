#include "texcomp.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <thread>

#include <bc7enc.h>

#include "prof.h"

namespace texcomp {
static std::atomic<bool> on{true};
static constexpr uint32_t VERSION = 1;  // bump when the encoder settings or mip filter change
static constexpr uint32_t VK_BC7_UNORM = 145, VK_BC7_SRGB = 146;  // VkFormat values

void init() { bc7enc_compress_block_init(); }
void setEnabled(bool v) { on = v; }
bool enabled() { return on; }

static uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

static float toLin(uint8_t v) { float c = v / 255.f; return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f); }
static uint8_t toSrgb(float c) {
    c = c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1.f / 2.4f) - 0.055f;
    return (uint8_t)std::clamp(c * 255.f + 0.5f, 0.f, 255.f);
}

// 2x2 box filter (in linear light for sRGB colour; alpha and linear data as stored)
static std::vector<uint8_t> downsample(const std::vector<uint8_t>& src, uint32_t w, uint32_t h, bool linear) {
    uint32_t nw = std::max(1u, w / 2), nh = std::max(1u, h / 2);
    std::vector<uint8_t> dst((size_t)nw * nh * 4);
    static float lut[256];
    static bool lutInit = [] { for (int i = 0; i < 256; i++) lut[i] = toLin((uint8_t)i); return true; }();
    (void)lutInit;
    for (uint32_t y = 0; y < nh; y++)
        for (uint32_t x = 0; x < nw; x++) {
            uint32_t x0 = std::min(2 * x, w - 1), x1 = std::min(2 * x + 1, w - 1), y0 = std::min(2 * y, h - 1), y1 = std::min(2 * y + 1, h - 1);
            const uint8_t* p[4] = {&src[((size_t)y0 * w + x0) * 4], &src[((size_t)y0 * w + x1) * 4], &src[((size_t)y1 * w + x0) * 4], &src[((size_t)y1 * w + x1) * 4]};
            uint8_t* o = &dst[((size_t)y * nw + x) * 4];
            for (int c = 0; c < 4; c++) {
                if (linear || c == 3) o[c] = (uint8_t)((p[0][c] + p[1][c] + p[2][c] + p[3][c] + 2) / 4);
                else o[c] = toSrgb((lut[p[0][c]] + lut[p[1][c]] + lut[p[2][c]] + lut[p[3][c]]) * 0.25f);
            }
        }
    return dst;
}

// one mip level -> BC7 blocks (edge texels clamped into partial blocks), rows of blocks split over all cores
static void encodeLevel(const std::vector<uint8_t>& px, uint32_t w, uint32_t h, bool linear, uint8_t* out) {
    uint32_t bw = (w + 3) / 4, bh = (h + 3) / 4;
    bc7enc_compress_block_params prm;
    bc7enc_compress_block_params_init(&prm);
    if (linear) bc7enc_compress_block_params_init_linear_weights(&prm);
    prm.m_uber_level = 0;
    prm.m_max_partitions = 16;
    std::atomic<uint32_t> next{0};
    auto work = [&] {
        uint8_t blk[64];
        for (uint32_t by; (by = next.fetch_add(1)) < bh;)
            for (uint32_t bx = 0; bx < bw; bx++) {
                for (uint32_t j = 0; j < 4; j++)
                    for (uint32_t i = 0; i < 4; i++) {
                        uint32_t x = std::min(bx * 4 + i, w - 1), y = std::min(by * 4 + j, h - 1);
                        memcpy(blk + (j * 4 + i) * 4, &px[((size_t)y * w + x) * 4], 4);
                    }
                bc7enc_compress_block(out + ((size_t)by * bw + bx) * 16, blk, &prm);
            }
    };
    uint32_t nt = std::clamp(std::thread::hardware_concurrency(), 1u, 16u);
    if (bh < 8) nt = 1;
    std::vector<std::thread> ts;
    for (uint32_t i = 1; i < nt; i++) ts.emplace_back(work);
    work();
    for (auto& t : ts) t.join();
}

// ---- KTX2 (uncompressed container of BC7 levels; spec: registry.khronos.org/KTX/specs/2.0)
static const uint8_t KTX2_ID[12] = {0xAB, 'K', 'T', 'X', ' ', '2', '0', 0xBB, '\r', '\n', 0x1A, '\n'};

static bool writeKtx2(const std::string& path, const TextureData& t) {
    uint32_t L = t.bcMips;
    // basic data format descriptor for BC7: 4x4 blocks of 16 bytes, one sample
    uint32_t dfd[11] = {44, 0, 40u << 16 | 2u, 136u | 1u << 8 | (t.linear ? 1u : 2u) << 16, 3u | 3u << 8, 16, 0, 127u << 16, 0, 0, 0xFFFFFFFFu};
    uint64_t levelIdx = 80, dfdOff = levelIdx + 24ull * L, dataOff = (dfdOff + sizeof dfd + 15) & ~15ull;
    FILE* f = fopen((path + ".tmp").c_str(), "wb");
    if (!f) return false;
    uint32_t hdr[9] = {t.linear ? VK_BC7_UNORM : VK_BC7_SRGB, 1, t.w, t.h, 0, 0, 1, L, 0};
    uint32_t idx[4] = {(uint32_t)dfdOff, (uint32_t)sizeof dfd, 0, 0};
    uint64_t sgd[2] = {0, 0};
    fwrite(KTX2_ID, 1, 12, f);
    fwrite(hdr, 4, 9, f);
    fwrite(idx, 4, 4, f);
    fwrite(sgd, 8, 2, f);
    // levels are stored smallest first; the index lists level 0 first
    std::vector<uint64_t> off(L);
    uint64_t o = dataOff;
    for (int l = (int)L - 1; l >= 0; l--) {
        uint64_t sz = (l + 1 < (int)L ? t.bcLevels[l + 1] : t.bc7.size()) - t.bcLevels[l];
        off[l] = o;
        o += (sz + 15) & ~15ull;
    }
    for (uint32_t l = 0; l < L; l++) {
        uint64_t sz = (l + 1 < L ? t.bcLevels[l + 1] : t.bc7.size()) - t.bcLevels[l];
        uint64_t e[3] = {off[l], sz, sz};
        fwrite(e, 8, 3, f);
    }
    fwrite(dfd, 4, 11, f);
    static const uint8_t zero[16] = {};
    fwrite(zero, 1, dataOff - (dfdOff + sizeof dfd), f);
    for (int l = (int)L - 1; l >= 0; l--) {
        uint64_t sz = (l + 1 < (int)L ? t.bcLevels[l + 1] : t.bc7.size()) - t.bcLevels[l];
        fwrite(t.bc7.data() + t.bcLevels[l], 1, sz, f);
        fwrite(zero, 1, ((sz + 15) & ~15ull) - sz, f);
    }
    bool ok = fclose(f) == 0;
    std::error_code ec;
    if (ok) std::filesystem::rename(path + ".tmp", path, ec);
    return ok && !ec;
}

static bool readKtx2(const std::string& path, TextureData& t) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint8_t id[12];
    uint32_t hdr[9], idx[4];
    uint64_t sgd[2];
    bool ok = fread(id, 1, 12, f) == 12 && !memcmp(id, KTX2_ID, 12) && fread(hdr, 4, 9, f) == 9 && fread(idx, 4, 4, f) == 4 && fread(sgd, 8, 2, f) == 2;
    ok = ok && hdr[0] == (t.linear ? VK_BC7_UNORM : VK_BC7_SRGB) && hdr[2] == t.w && hdr[3] == t.h && hdr[7] >= 1 && hdr[7] <= 16 && hdr[8] == 0;
    std::vector<uint64_t> lv;
    if (ok) {
        lv.resize(hdr[7] * 3);
        ok = fread(lv.data(), 8, lv.size(), f) == lv.size();
    }
    if (ok) {
        uint32_t L = hdr[7];
        t.bcLevels.resize(L);
        size_t total = 0;
        for (uint32_t l = 0; l < L; l++) { t.bcLevels[l] = total; total += lv[l * 3 + 1]; }
        t.bc7.resize(total);
        for (uint32_t l = 0; l < L && ok; l++)
            ok = fseek(f, (long)lv[l * 3], SEEK_SET) == 0 && fread(t.bc7.data() + t.bcLevels[l], 1, lv[l * 3 + 1], f) == lv[l * 3 + 1];
        t.bcMips = L;
    }
    fclose(f);
    if (!ok) { t.bc7.clear(); t.bcLevels.clear(); t.bcMips = 0; }
    return ok;
}

void prepare(TextureData& t, const char* cacheDir) {
    if (!on || !t.bc7.empty() || t.rgba.empty() || t.w == 0 || t.h == 0) return;
    PROF_ZONE("texcomp prepare");
    uint64_t h = fnv(t.rgba.data(), t.rgba.size());
    uint32_t meta[4] = {t.w, t.h, t.linear ? 1u : 0u, VERSION};
    h = fnv(meta, sizeof meta, h);
    char name[64];
    snprintf(name, sizeof name, "/bc7_%016llx.ktx2", (unsigned long long)h);
    std::string path = std::string(cacheDir) + name;
    if (readKtx2(path, t)) return;

    PROF_ZONE("texcomp encode");
    uint32_t mips = 1 + (uint32_t)floorf(log2f((float)std::max(t.w, t.h)));
    std::vector<uint8_t> cur = t.rgba;
    uint32_t w = t.w, hh = t.h;
    t.bcLevels.clear();
    size_t total = 0;
    for (uint32_t l = 0; l < mips; l++) {
        t.bcLevels.push_back(total);
        total += (size_t)((std::max(1u, t.w >> l) + 3) / 4) * ((std::max(1u, t.h >> l) + 3) / 4) * 16;
    }
    t.bc7.resize(total);
    for (uint32_t l = 0; l < mips; l++) {
        encodeLevel(cur, w, hh, t.linear, t.bc7.data() + t.bcLevels[l]);
        if (l + 1 < mips) { cur = downsample(cur, w, hh, t.linear); w = std::max(1u, w / 2); hh = std::max(1u, hh / 2); }
    }
    t.bcMips = mips;
    std::error_code ec;
    std::filesystem::create_directories(cacheDir, ec);
    if (!writeKtx2(path, t)) fprintf(stderr, "texcomp: could not write %s\n", path.c_str());
}
}  // namespace texcomp
