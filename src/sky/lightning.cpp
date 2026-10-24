#include "sky/lightning.h"
#include "scene/vegetation.h"
#include "sim/critters.h"
#include "sim/fire.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <cstdint>
#include <cstddef>
#include "audio/audio.h"
#include "app/frame_state.h"
#include "app/settings.h"
#include "core/math.h"
#include "render/frame_ubo.h"
#include "scene/camera.h"
#include "world/terrain_query.h"

Bolt bolt;

bool boltRequest = false;

namespace {
constexpr float TREE_STRIKE_REACH = 8.f;    // m: a bolt this close to a tree goes down the tree
constexpr float TREE_FIRE_CHANCE = 0.5f;    // that it sets it burning (in dry weather)
constexpr float GROUND_FIRE_CHANCE = 0.12f; // that it lights the grass where it strikes open ground
constexpr float TREE_FIRE_BURN = 120.f, GROUND_FIRE_BURN = 30.f;  // s

// The strike may start a fire: a tree it hit, or the grass. Rain makes both less likely.
void igniteAt(V3 g, std::mt19937& rng) {
    std::uniform_real_distribution<float> U(0.f, 1.f);
    float dry = std::clamp(1.f - 0.7f * S.rain, 0.f, 1.f);
    const TreeInst* best = nullptr;
    float bd = TREE_STRIKE_REACH * TREE_STRIKE_REACH;
    for (const TreeInst& t : treeList) {
        float dx = t.p.x - g.x, dz = t.p.z - g.z;
        if (dx * dx + dz * dz < bd) { bd = dx * dx + dz * dz; best = &t; }
    }
    if (best) { if (U(rng) < TREE_FIRE_CHANCE * dry) startFire(best->p, 1.2f * best->scale, TREE_FIRE_BURN); }
    else if (riverDry(g.x, g.z, 0.f) > 0.f && U(rng) < GROUND_FIRE_CHANCE * dry) startFire(g, 0.5f, GROUND_FIRE_BURN);
}
}  // namespace

static void newBolt() {
    std::mt19937 rng((uint32_t)(simTime * 1000.0) ^ 0x9e3779b9u);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    float d = 150.f + 5850.f * U(rng) * U(rng), a = U(rng) * 6.2832f;
    V3 g{cam.pos.x + cosf(a) * d, 0.f, cam.pos.z + sinf(a) * d};
    g.y = heightFn(g.x, g.z);
    V3 top{g.x + (U(rng) - 0.5f) * 400.f, std::max(S.cloudBase, g.y + 300.f), g.z + (U(rng) - 0.5f) * 400.f};
    bolt = Bolt{};
    bolt.ground = g; bolt.top = top; bolt.t0 = simTime; bolt.dist = d;
    auto channel = [&](V3 p0, V3 p1, int n, float w, float br) {
        V3 prev = p0;
        float len = sqrtf(dot(p1 - p0, p1 - p0)) / n;
        for (int i = 1; i <= n && bolt.seg.size() < 48; i++) {
            V3 q = p0 + (p1 - p0) * ((float)i / n);
            if (i < n) q = q + V3{U(rng) - 0.5f, (U(rng) - 0.5f) * 0.3f, U(rng) - 0.5f} * (len * 0.9f);
            bolt.seg.push_back({prev.x, prev.y, prev.z, w, q.x, q.y, q.z, br});
            prev = q;
        }
    };
    channel(top, g, 28, 0.6f, 1.f);
    for (int b = 0; b < 4 && bolt.seg.size() < 44; b++) {
        auto& sg = bolt.seg[(size_t)(U(rng) * 20.f)];
        V3 s0{sg[0], sg[1], sg[2]};
        V3 e = s0 + V3{(U(rng) - 0.5f) * 500.f, -(100.f + 300.f * U(rng)), (U(rng) - 0.5f) * 500.f};
        channel(s0, e, 5, 0.25f, 0.35f);
    }
    bolt.nStrokes = 2 + (int)(U(rng) * 3.f);
    for (int k = 0; k < bolt.nStrokes; k++) bolt.strokes[k] = k == 0 ? 0.f : bolt.strokes[k - 1] + 0.04f + 0.12f * U(rng);
    igniteAt(g, rng);
}

static float boltFlash() {
    float t = (float)(simTime - bolt.t0), f = 0.f;
    for (int k = 0; k < bolt.nStrokes; k++) if (t >= bolt.strokes[k]) f += expf(-(t - bolt.strokes[k]) / 0.035f);
    return t < 2.f ? std::min(f, 1.5f) : 0.f;
}

void updateLightning(float dt) {
    float storm = S.clouds ? S.cloudCoverage * S.cumulonimbus : 0.f;
    std::uniform_real_distribution<float> U(0.f, 1.f);
    static std::mt19937 rng(77);
    float rate = S.lightning ? S.strikesPerMin / 60.f * std::clamp((S.rain - 0.3f) / 0.5f, 0.f, 1.f) * std::min(storm * 3.f, 1.f) : 0.f;
    if (boltRequest || (rate > 0.f && U(rng) < rate * dt && simTime - bolt.t0 > 2.0)) { newBolt(); boltRequest = false; }
    if (!bolt.thunder && bolt.t0 > -1e8 && simTime - bolt.t0 > bolt.dist / 343.f) {
        bolt.thunder = true;
        V3 dir = normalize(bolt.ground - cam.pos);
        V3 at = cam.pos + dir * 40.f;  // our own distance law (the spatializer would drown km-far sources)
        float gain = std::clamp(600.f / bolt.dist, 0.08f, 2.f);
        audio::emitterPlay(bolt.dist < 900.f ? audio::SND_THUNDER_NEAR : audio::SND_THUNDER_FAR, &at.x, gain, false, 0.9f + 0.2f * U(rng));
        critterAlarm(cam.pos, gain);  // thunder scatters the birds
    }
}

void writeLightning(UBO& u) {
    float f = boltFlash();
    u.boltInfo[0] = f > 0.f ? (float)bolt.seg.size() : 0.f;
    u.boltInfo[1] = f;
    u.boltInfo[2] = 0.4f;
    for (size_t i = 0; i < bolt.seg.size() && f > 0.f; i++) { memcpy(u.bolt[2 * i], &bolt.seg[i][0], 16); memcpy(u.bolt[2 * i + 1], &bolt.seg[i][4], 16); }
    V3 c = bolt.ground * 0.5f + bolt.top * 0.5f;
    set4(u.boltPos, c, 0.f);
}
