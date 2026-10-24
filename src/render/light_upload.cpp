#include "render/light_upload.h"

#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <vulkan/vulkan.h>
#include "core/prof.h"
#include "core/jobs.h"
#include "core/simd.h"
#include "vegetation/trees.h"
#include "world/worldmap.h"
#include "world/terrain.h"
#include "world/world.h"
#include "app/frame_state.h"
#include "app/settings.h"
#include "core/limits.h"
#include "core/math.h"
#include "destruction/rocks.h"
#include "destruction/thrown.h"
#include "destruction/tree_damage.h"
#include "gpu/accel.h"
#include "gpu/swapchain.h"
#include "render/bake.h"
#include "render/crater_bins.h"
#include "render/frame_ubo.h"
#include "render/ground_textures.h"
#include "render/ground_vt.h"
#include "render/light_build.h"
#include "render/lights.h"
#include "render/targets.h"
#include "scene/camera.h"
#include "scene/grass.h"
#include "scene/ground_mask.h"
#include "scene/model_instances.h"
#include "scene/models.h"
#include "scene/rocks.h"
#include "scene/scene_gpu.h"
#include "scene/stream_sync.h"
#include "scene/vegetation.h"
#include "sim/effects.h"
#include "sim/fire.h"
#include "sim/water.h"
#include "sim/water_splash.h"
#include "sim/world_sim.h"
#include "sky/celestial.h"
#include "sky/lightning.h"
#include "world/terrain_query.h"

void uploadLights() {
    float* L = (float*)lightBuf.map;
    float t = (float)simTime;
    int orbs = std::min(S.lightCount, MAX_ORBS);
    for (int i = 0; i < orbs; i++) {
        auto& b = lightBase[i];
        const float R = S.orbRange;
        auto wrap = [R](float v) { return v - R * floorf(v / R + 0.5f); };
        float x = cam.pos.x + wrap(b.p.x * R + sinf(t * 0.31f + b.phase) * 1.6f - cam.pos.x);
        float z = cam.pos.z + wrap(b.p.z * R + cosf(t * 0.27f + b.phase * 1.3f) * 1.6f - cam.pos.z);
        float y = heightFn(x, z) + b.p.y + sinf(t * 0.83f + b.phase * 1.7f) * 0.35f;
        // fade out (light and orb size) toward the tile edge, where an orb wraps to the far side: no pop
        float e = std::max(fabsf(x - cam.pos.x), fabsf(z - cam.pos.z)) / (0.5f * R);
        float fade = 1.f - std::clamp((e - 0.75f) / 0.25f, 0.f, 1.f);
        fade = fade * fade * (3.f - 2.f * fade);
        orbFade[i] = fade;
        V3 c = b.col * (b.inten * S.lightIntensity * fade);
        float v[8] = {x, y, z, 0.06f * fade, c.x, c.y, c.z, 0};
        memcpy(L + i * 8, v, sizeof v);
    }
    // fires: dynamic like the orbs (not in the probe bakes), hidden from the camera
    int dyn = orbs + writeFireLights(L + orbs * 8, MAX_LIGHTS - orbs);
    // streamed lights: keep the strongest at the camera (radius 0 = not drawn as an orb)
    size_t nw = wlx.size(), room = MAX_LIGHTS - dyn;
    std::vector<uint32_t> pick;
    if (nw) {
        lightScores(wlx.data(), wly.data(), wlz.data(), wlLum.data(), nw, cam.pos.x, cam.pos.y, cam.pos.z, wlScore.data());
        pick.resize(nw);
        for (uint32_t i = 0; i < nw; i++) pick[i] = i;
        if (nw > room) {
            std::nth_element(pick.begin(), pick.begin() + room, pick.end(), [](uint32_t a, uint32_t b) { return wlScore[a] > wlScore[b]; });
            pick.resize(room);
        }
    }
    auto& wl = stream::lights();
    // switched-off lights go after the active ones: only the probe bake (lights-on state) samples them
    std::stable_partition(pick.begin(), pick.end(), [&](uint32_t i) { return !lightOff(wl[i].pos); });
    uint32_t active = 0;
    for (size_t k = 0; k < pick.size(); k++) {
        const WorldLight& w = wl[pick[k]];
        active += !lightOff(w.pos);
        // radius 4 cm: drawn as a small glowing bulb (hidden inside lamp models that have their own glowing mesh)
        float v[8] = {w.pos[0], w.pos[1], w.pos[2], 0.04f, w.color[0] * lightScale, w.color[1] * lightScale, w.color[2] * lightScale, 0};
        memcpy(L + (dyn + k) * 8, v, sizeof v);
    }
    numLights = dyn + active;
    numWorldLights = active;
    bakeOffLights = (uint32_t)pick.size() - active;
    buildLightTree(L);
    UBO* ub = (UBO*)uboBuf.map;
    ub->res[3] = numLights;
    ub->bake[0] = (uint32_t)bakeVols.size();
    ub->bake[1] = (uint32_t)dyn;
    ub->bake[2] = bakedGI ? 1u : 0u;
    ub->bake[3] = numLights + bakeOffLights;
    ub->lightInfo[0] = numTriLights;
    ub->lightInfo[1] = numTreeNodes;
    ub->lightInfo[2] = allEmittersListed ? 1u : 0u;
    writeBakeVolumes();
}
