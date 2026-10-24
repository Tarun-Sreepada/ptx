#include "render/bake.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <vulkan/vulkan.h>
#include "world/world.h"
#include "app/settings.h"
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/vk.h"
#include "render/lights.h"
#include "scene/camera.h"

std::vector<BakeVol> bakeVols;

Buffer volBuf, probeBuf, probeTmp;

bool bakedGI = true, bakeEnabled = true, startLightsOff = false;

uint64_t bakeEnv = 0;   // hash of everything the bakes depend on (sun, sky, light scale)

std::string bakeDir;

uint32_t bakeOffLights = 0;

// ---------------------------------------------------------------- baked interiors
static bool inBox(const ProbeVolume& v, float x, float y, float z) {
    return x >= v.mn[0] && y >= v.mn[1] && z >= v.mn[2] && x < v.mx[0] && y < v.mx[1] && z < v.mx[2];
}

int volumeAt(float x, float y, float z) {
    for (size_t i = 0; i < bakeVols.size(); i++) if (inBox(bakeVols[i].v, x, y, z)) return (int)i;
    return -1;
}

static uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
    return h;
}

static std::string bakePath(const BakeVol& b, int st) {
    uint64_t h = fnv(b.v.mn, 24, fnv(b.dims, 12, fnv(&b.spacing, 4, bakeEnv ^ 5)));  // bump the xor to invalidate old caches
    char name[64];
    snprintf(name, sizeof name, "_%s_%016llx.bin", st ? "on" : "off", (unsigned long long)h);
    return bakeDir + b.v.name + name;
}

uint64_t bakeEnvHash() {
    // With the day / night cycle running the sun moves every frame: the bakes follow it in steps (a few game hours
    // each), and the cached bake of each step is found again the next day, instead of starting over every frame.
    constexpr float SUN_STEP_ELEV = 8.f, SUN_STEP_AZIM = 25.f;  // degrees
    float el = S.sunElev, az = S.sunAzim;
    if (S.dayCycle) { el = roundf(el / SUN_STEP_ELEV) * SUN_STEP_ELEV; az = roundf(az / SUN_STEP_AZIM) * SUN_STEP_AZIM; }
    // version 2: emitters' direct light left to ReSTIR DI (lights.glsl emissionWeight) -> older bakes are redone
    float e[] = {el, az, S.sunIntensity, S.sunRadiusDeg, S.skyIntensity, lightScale, 2.f};
    return fnv(e, sizeof e);
}

// Probe data of one volume + state: GPU <-> file (only called right after a frame fence)
static void bakeTransfer(BakeVol& b, int st, bool save) {
    VkDeviceSize size = (VkDeviceSize)b.count * 64, off = ((VkDeviceSize)b.first * 2 + (VkDeviceSize)st * b.count) * 64;
    Buffer stg = mkBuf(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::string path = bakePath(b, st);
    if (save) {
        vkDeviceWaitIdle(dev);
        oneShot([&](VkCommandBuffer cb) { VkBufferCopy c{off, 0, size}; vkCmdCopyBuffer(cb, probeBuf.b, stg.b, 1, &c); });
        std::filesystem::create_directories(bakeDir);
        std::ofstream f(path, std::ios::binary);
        f.write((const char*)stg.map, (std::streamsize)size);
        b.saved[st] = true;
    } else {
        std::ifstream f(path, std::ios::binary);
        if (f && f.read((char*)stg.map, (std::streamsize)size)) {
            vkDeviceWaitIdle(dev);
            oneShot([&](VkCommandBuffer cb) { VkBufferCopy c{0, off, size}; vkCmdCopyBuffer(cb, stg.b, probeBuf.b, 1, &c); });
            b.ready[st] = b.saved[st] = true;
        }
    }
    freeBuf(stg);
}

// New set of volumes (stream change) or new lighting environment: lay out probe grids, load cached bakes.
void rebuildBakeVolumes(bool envChanged) {
    const auto& vs = stream::volumes();
    bool same = !envChanged && vs.size() == bakeVols.size();
    for (size_t i = 0; same && i < vs.size(); i++) same = vs[i].name == bakeVols[i].v.name && !memcmp(vs[i].mn, bakeVols[i].v.mn, 24);
    if (same) return;
    std::vector<BakeVol> old;
    old.swap(bakeVols);
    // 25 cm probes; coarser everywhere if they don't fit
    double vol = 0;
    for (auto& v : vs) vol += (double)(v.mx[0] - v.mn[0]) * (v.mx[1] - v.mn[1]) * (v.mx[2] - v.mn[2]);
    float sp = std::max(0.25f, (float)cbrt(vol / (MAX_PROBES * 0.9)));
    uint32_t first = 0;
    for (auto& v : vs) {
        if (bakeVols.size() >= MAX_VOLUMES) break;
        BakeVol b;
        b.v = v;
        b.spacing = sp;
        b.on = !startLightsOff;
        for (int k = 0; k < 3; k++) b.dims[k] = std::max(1u, (uint32_t)ceilf((v.mx[k] - v.mn[k]) / sp));
        b.count = b.dims[0] * b.dims[1] * b.dims[2];
        if (first + b.count > MAX_PROBES) break;
        b.first = first;
        first += b.count;
        for (auto& o : old) if (o.v.name == v.name) b.on = o.on;  // keep switch states
        bakeVols.push_back(b);
    }
    for (auto& b : bakeVols) for (int st = 0; st < 2; st++) bakeTransfer(b, st, false);
}

// Lights inside a switched-off building leave the active list (they go after it, for the lights-on bake).
bool lightOff(const float p[3]) {
    int v = volumeAt(p[0], p[1], p[2]);
    return v >= 0 && !bakeVols[v].on;
}

void writeBakeVolumes() {
    float* o = (float*)volBuf.map;
    for (size_t i = 0; i < bakeVols.size(); i++) {
        auto& b = bakeVols[i];
        uint32_t d[4] = {b.dims[0], b.dims[1], b.dims[2], b.first};
        float e[16] = {b.v.mn[0], b.v.mn[1], b.v.mn[2], b.spacing, b.v.mx[0], b.v.mx[1], b.v.mx[2], b.on ? 1.f : 0.f,
                       0, 0, 0, 0, b.ready[0] ? 1.f : 0.f, b.ready[1] ? 1.f : 0.f, 0, 0};
        memcpy(e + 8, d, 16);
        memcpy(o + i * 16, e, sizeof e);
    }
}

BakeJob nextBakeJob() {
    BakeJob j;
    if (!bakeEnabled || bakeVols.empty()) return j;
    int here = volumeAt(cam.pos.x, cam.pos.y, cam.pos.z);
    std::vector<std::pair<int, int>> order;
    if (here >= 0) { order.push_back({here, bakeVols[here].on}); order.push_back({here, !bakeVols[here].on}); }
    for (int i = 0; i < (int)bakeVols.size(); i++) for (int st : {(int)bakeVols[i].on, (int)!bakeVols[i].on}) order.push_back({i, st});
    for (auto [v, st] : order) {
        BakeVol& b = bakeVols[v];
        if (b.ready[st]) continue;
        if (b.pass[st] >= BAKE_PASSES) {  // denoise: one whole-volume filter iteration per frame
            j = {v, st, 0, b.count, b.pass[st], true, b.count, b.first * 2 + (uint32_t)st * b.count};
            if (++b.pass[st] == BAKE_PASSES + BAKE_FILTERS) b.ready[st] = true;  // the GI pass sees it from the next frame on
            return j;
        }
        j = {v, st, b.next[st], std::min(BAKE_CHUNK, b.count - b.next[st]), b.pass[st]};
        b.next[st] += j.n;
        if (b.next[st] >= b.count) { b.next[st] = 0; b.pass[st]++; }
        return j;
    }
    return j;
}

void saveFinishedBakes() {
    for (auto& b : bakeVols) for (int st = 0; st < 2; st++) if (b.ready[st] && !b.saved[st]) bakeTransfer(b, st, true);
}
