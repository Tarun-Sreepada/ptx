#include "world/ground_surface.h"
#include <string>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include "world/worldmap.h"
#include "world/terrain.h"
#include "world/world.h"
#include "app/settings.h"
#include "sim/world_sim.h"
#include "scene/ground_mask.h"

// Craters next to the river fill from it instead of appearing full: the water level rises as river water pours in
// (a share of the river's discharge, falling off with the remaining head), and while it does, that share is
// missing downstream, so the river there slows and drops a little (water.glsl). A fresh crater, or one that grows,
// starts (part) empty again.
// Ground under a terrain contact: friction / restitution by what's there. Bare soil and grass give little bounce
// (soft, absorbs), hard surfaces (stone, paving, painted rock / concrete textures) more; loose crater soil less;
// water films make it slicker, snow swallows the bounce and is slippery.
static void surfaceOfMaterial(const std::string& n, float& f, float& e) {
    auto has = [&](const char* k) { return n.find(k) != std::string::npos; };
    if (has("ice")) { f = 0.05f; e = 0.1f; }
    else if (has("metal") || has("steel")) { f = 0.45f; e = 0.4f; }
    else if (has("stone") || has("rock") || has("concrete") || has("asphalt") || has("paving") || has("brick") || has("tile") || has("cobble")) { f = 0.62f; e = 0.35f; }
    else if (has("wood") || has("plank")) { f = 0.55f; e = 0.25f; }
    else if (has("sand") || has("gravel")) { f = 0.9f; e = 0.03f; }
    else if (has("mud")) { f = 0.95f; e = 0.0f; }
    else { f = 0.75f; e = 0.12f; }
}

void groundSurfaceAt(const float p[3], float& f, float& e) {
    float x = p[0], z = p[2];
    f = S.groundFriction; e = S.groundBounce;  // natural soil / grass
    const auto& zs = stream::zones();
    float zc = zoneCover(x, z, zs.data(), (int)zs.size());
    if (zc > 0.5f) {  // flattened (city) ground: paved, or its own texture
        float zf = 0.62f, ze = 0.3f;
        for (auto& zn : zs) if (zoneWeight(zn, x, z) > 0.5f && !zn.mat.empty()) surfaceOfMaterial(zn.mat, zf, ze);
        f = zf; e = ze;
    }
    // painted ground texture (world map)
    if (!std::isfinite(x) || !std::isfinite(z)) return;
    int cx = (int)floorf((x - worldmap::ORIGIN) / worldmap::CELL), cz = (int)floorf((z - worldmap::ORIGIN) / worldmap::CELL);
    if (cx >= 0 && cz >= 0 && cx < worldmap::N && cz < worldmap::N) {
        const worldmap::Paint& pt = worldmap::paintData()[(size_t)cz * worldmap::N + cx];
        if (pt.mat && pt.matW > 64 && !worldmap::paletteName(pt.mat).empty()) {
            float pf, pe;
            surfaceOfMaterial(worldmap::paletteName(pt.mat), pf, pe);
            float w = pt.matW / 255.f;
            f += (pf - f) * w; e += (pe - e) * w;
        }
    }
    if (inCrater(x, z, 0.f)) { f = std::max(f, 0.85f); e = std::min(e, 0.04f); }  // loose, churned soil
    if (simSnapValid) {
        int i = cx - simSnapX, k = cz - simSnapZ;
        if (i >= 0 && k >= 0 && i < SIM_RB && k < SIM_RB) {
            float snow = simSnap[k * SIM_RB + i][0], wet = simSnap[k * SIM_RB + i][1];
            f *= 1.f - 0.35f * std::clamp(wet, 0.f, 1.f);
            e *= 1.f - 0.3f * std::clamp(wet, 0.f, 1.f);
            float sk = std::clamp((snow - 0.02f) / 0.2f, 0.f, 1.f);
            f += (0.28f - f) * sk;
            e *= 1.f - sk;
        }
    }
}

static int footOfMaterial(const std::string& n) {
    auto has = [&](const char* k) { return n.find(k) != std::string::npos; };
    if (has("wood") || has("plank")) return FOOT_WOOD;
    if (has("stone") || has("rock") || has("concrete") || has("asphalt") || has("paving") || has("brick") || has("tile") ||
        has("cobble") || has("metal") || has("ice")) return FOOT_CONCRETE;
    if (has("grass") || has("moss")) return FOOT_GRASS;
    return FOOT_DIRT;  // sand, gravel, mud, soil
}

int footSurfaceAt(const float p[3], float feetAbove, float wade) {
    float x = p[0], z = p[2];
    if (wade > 0.01f) return FOOT_WATER;
    if (feetAbove > 0.12f) return FOOT_CONCRETE;  // on something built (floors, prims, rocks)
    if (!std::isfinite(x) || !std::isfinite(z)) return FOOT_DIRT;
    int cx = (int)floorf((x - worldmap::ORIGIN) / worldmap::CELL), cz = (int)floorf((z - worldmap::ORIGIN) / worldmap::CELL);
    if (simSnapValid) {
        int i = cx - simSnapX, k = cz - simSnapZ;
        if (i >= 0 && k >= 0 && i < SIM_RB && k < SIM_RB && simSnap[k * SIM_RB + i][0] > 0.04f) return FOOT_SNOW;
    }
    const auto& zs = stream::zones();
    if (zoneCover(x, z, zs.data(), (int)zs.size()) > 0.5f) {  // city ground: paved unless it has its own texture
        for (auto& zn : zs) if (zoneWeight(zn, x, z) > 0.5f && !zn.mat.empty()) return footOfMaterial(zn.mat);
        return FOOT_CONCRETE;
    }
    float grass = grassMaskCPU(x, z);
    if (cx >= 0 && cz >= 0 && cx < worldmap::N && cz < worldmap::N) {
        const worldmap::Paint& pt = worldmap::paintData()[(size_t)cz * worldmap::N + cx];
        if (pt.mat && pt.matW > 128 && !worldmap::paletteName(pt.mat).empty()) return footOfMaterial(worldmap::paletteName(pt.mat));
        grass *= std::min(pt.grass / 128.f, 1.f);
    }
    if (inCrater(x, z, 0.f)) return FOOT_DIRT;
    return grass > 0.35f ? FOOT_GRASS : FOOT_DIRT;
}
