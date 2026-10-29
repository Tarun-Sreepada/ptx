#include "scene/audio_sync.h"
#include "sim/water_splash.h"
#include "sim/water.h"
#include "scene/ambience.h"
#include "sim/trample.h"
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "core/prof.h"
#include "audio/audio.h"
#include "world/world.h"
#include "app/settings.h"
#include "core/limits.h"
#include "core/math.h"
#include "render/lights.h"
#include "scene/camera.h"
#include "scene/models.h"
#include "scene/scene_gpu.h"
#include "tools/bench.h"
#include "world/terrain_query.h"
#include "world/ground_surface.h"
#include <filesystem>
#include <string>

std::vector<uint32_t> audioMeshRetire;

// acoustic instances keyed by (mesh, placement); diffed on every stream change so only edits reach the audio scene
struct AudioKey {
    const void* mesh; float t[3], r0;
    bool operator==(const AudioKey& o) const { return !memcmp(this, &o, sizeof *this); }
};

struct AudioKeyHash {
    size_t operator()(const AudioKey& k) const {
        size_t h = std::hash<const void*>()(k.mesh);
        for (float f : {k.t[0], k.t[1], k.t[2], k.r0}) h = h * 1000003u ^ std::hash<float>()(f);
        return h;
    }
};

static std::unordered_map<AudioKey, uint32_t, AudioKeyHash> audioInsts;

void syncAudioInstances() {
    std::unordered_map<AudioKey, uint32_t, AudioKeyHash> next;
    stream::forEachPlaced([&](const Placed& p) {
        auto* g = (GpuModel*)p.model->gpu;
        if (!g || (size_t)p.mesh >= g->audioMeshes.size()) return;
        AudioKey k{&p.model->data.meshes[p.mesh], {p.xf.m[0][3], p.xf.m[1][3], p.xf.m[2][3]}, p.xf.m[0][0]};
        auto it = audioInsts.find(k);
        if (it != audioInsts.end()) { next.emplace(k, it->second); audioInsts.erase(it); }
        else next.emplace(k, audio::instanceAdd(g->audioMeshes[p.mesh], p.xf));
    });
    for (auto& [k, id] : audioInsts) audio::instanceRemove(id);
    audioInsts.swap(next);
}

// ---------------------------------------------------------------- audio: listener, ground patch, footsteps, orb hums
void updateAudio(float dt) {
    PROF_FUNC();
    V3 f = cam.fwd(), u = cam.up();
    audio::setListener(&cam.pos.x, &f.x, &u.x);

    // acoustic ground: 96 m patch at 3 m spacing, rebuilt when the listener moves 16 m
    static float gx = 1e9f, gz = 1e9f;
    if (fabsf(cam.pos.x - gx) > 16.f || fabsf(cam.pos.z - gz) > 16.f) {
        gx = floorf(cam.pos.x / 3.f) * 3.f;
        gz = floorf(cam.pos.z / 3.f) * 3.f;
        const int N = 33;
        std::vector<float> v;
        std::vector<uint32_t> idx;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) {
                float x = gx + (i - N / 2) * 3.f, z = gz + (j - N / 2) * 3.f;
                v.insert(v.end(), {x, heightFn(x, z), z});
            }
        for (int j = 0; j + 1 < N; j++)
            for (int i = 0; i + 1 < N; i++) {
                uint32_t a = j * N + i;
                idx.insert(idx.end(), {a, a + N, a + 1, a + 1, a + N, a + N + 1});
            }
        audio::setTerrain(v.data(), (uint32_t)v.size() / 3, idx.data(), (uint32_t)idx.size() / 3);
    }

    // footsteps every stride while walking on the ground, a heavier one on landing; recorded steps per surface
    // (world/assets/sounds/footsteps/<surface>/, CC0), the synthesized ones where a folder is empty
    static std::vector<uint32_t> footSnd[FOOT_COUNT];
    static bool footInit = false;
    if (!footInit) {
        footInit = true;
        for (int k = 0; k < FOOT_COUNT; k++) {
            std::error_code ec;
            std::vector<std::string> files;
            for (auto& e : std::filesystem::directory_iterator(std::string(WORLD_DIR) + "assets/sounds/footsteps/" + FOOT_NAMES[k], ec))
                files.push_back(e.path().string());
            std::sort(files.begin(), files.end());
            for (auto& f : files) if (uint32_t id = audio::soundLoad(f.c_str()); id != UINT32_MAX) footSnd[k].push_back(id);
        }
    }
    static std::mt19937 rng(3);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    static V3 lastFeet = playerFeet;
    static float stride = 0;
    static bool wasGrounded = true;
    if (walkMode && !bench.running) {
        V3 d = playerFeet - lastFeet;
        float moved = sqrtf(d.x * d.x + d.z * d.z);
        if (grounded && moved < 1.f) stride += moved;
        float len = moved / std::max(dt, 1e-4f) > 5.f ? 1.0f : 0.75f;
        float wl = waterLevelAt(playerFeet.x, playerFeet.z), wade = wl - playerFeet.y;  // m of water at the feet
        float speed = moved / std::max(dt, 1e-4f);
        float feetAbove = playerFeet.y - heightFn(playerFeet.x, playerFeet.z);
        int surf = footSurfaceAt(&playerFeet.x, feetAbove, wade);
        auto step = [&](float gain) {
            const std::vector<uint32_t>& v = footSnd[surf];
            float run = std::clamp((speed - 2.f) / 4.f, 0.f, 1.f);  // running: harder, a little higher
            gain *= 0.8f + 0.5f * run;
            if (v.empty()) audio::emitterPlay(audio::SND_FOOTSTEP0 + (uint32_t)(U(rng) * 3.99f), &playerFeet.x, gain, false, 0.9f + 0.2f * U(rng));
            else audio::emitterPlay(v[std::min((size_t)(U(rng) * v.size()), v.size() - 1)], &playerFeet.x, gain, false, 0.92f + 0.1f * run + 0.12f * U(rng));
        };
        if (grounded && stride > len) {
            stride = 0;
            if (wade > 0.01f) step(0.5f + std::min(wade * 2.f, 0.5f));  // the slosh of the foot in the water
            if (wade > 0.02f) {  // wading: a splash and ripples instead of a footstep, bigger the deeper
                float v[3] = {d.x / std::max(dt, 1e-4f), -1.5f, d.z / std::max(dt, 1e-4f)};
                waterImpact(V3{playerFeet.x, wl, playerFeet.z}, std::min(0.15f + wade, 0.8f), v, std::min(0.4f + wade, 1.f));
            } else {
                step(0.5f + 0.2f * U(rng));
                static bool leftFoot = false;
                leftFoot = !leftFoot;
                if (moved > 1e-3f) footstep(playerFeet, atan2f(d.x, d.z), leftFoot);  // a print, and the grass stays trodden
            }
        }
        if (grounded && !wasGrounded) {
            if (wade > 0.02f) { float v[3] = {0.f, -4.f, 0.f}; waterImpact(V3{playerFeet.x, wl, playerFeet.z}, 0.5f, v); }
            else {
                audio::emitterPlay(audio::SND_LAND, &playerFeet.x, 0.6f, false, 0.95f + 0.1f * U(rng));  // the thump
                step(0.9f);                                                                      // and the surface
            }
        }
    }
    lastFeet = playerFeet;
    wasGrounded = grounded;
    if (!bench.running) updateAmbience(dt);  // birds, crickets, the river, leaves

    // light orbs hum: the nearest few get an emitter (positions from this frame's light buffer)
    constexpr int HUMS = 8;
    static uint32_t humEmitter[MAX_ORBS];
    static bool humInit = false;
    if (!humInit) { std::fill_n(humEmitter, MAX_ORBS, UINT32_MAX); humInit = true; }
    const float* L = (const float*)lightBuf.map;
    int orbs = std::min(S.lightCount, MAX_ORBS);
    std::vector<std::pair<float, int>> near;
    for (int i = 0; i < orbs; i++) {
        V3 p{L[i * 8], L[i * 8 + 1], L[i * 8 + 2]}, d = p - cam.pos;
        near.push_back({dot(d, d), i});
    }
    int keep = std::min(HUMS, (int)near.size());
    std::partial_sort(near.begin(), near.begin() + keep, near.end());
    bool want[MAX_ORBS] = {};
    for (int k = 0; k < keep; k++) want[near[k].second] = S.lightIntensity > 0.f;
    for (int i = 0; i < MAX_ORBS; i++) {
        uint32_t& e = humEmitter[i];
        if (e != UINT32_MAX && !audio::emitterAlive(e)) e = UINT32_MAX;
        if (!want[i]) { if (e != UINT32_MAX) { audio::emitterStop(e); e = UINT32_MAX; } continue; }
        const float* p = L + i * 8;
        float gain = 0.25f * lightBase[i].inten * std::min(S.lightIntensity, 4.f) * orbFade[i];
        if (e == UINT32_MAX) e = audio::emitterPlay(audio::SND_HUM, p, gain, true, 0.8f + 0.45f * fmodf(lightBase[i].phase, 1.f));
        else { audio::emitterSetPos(e, p); audio::emitterSetGain(e, gain); }
    }
    audio::setAmbience(S.windStrength * 0.35f * (0.4f + S.windSpeed / 6.f));
}
