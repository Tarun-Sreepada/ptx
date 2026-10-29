#include "render/ubo_fill.h"
#include "render/frame_data.h"

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
#include "sim/water.h"
#include "sim/water_splash.h"
#include "sim/world_sim.h"
#include "sky/celestial.h"
#include "sky/lightning.h"
#include "world/terrain_query.h"

void fillCameraSky(UBO& u) {
    auto camVecs = [](const Camera& c, float* pos, float* r, float* up, float* f, float time) {
        float aspect = (float)W / H, th = tanf(c.fov() * 0.5f * 3.14159265f / 180.f);
        set4(pos, c.pos, time);
        set4(r, c.right(), aspect);
        set4(up, c.up(), th);
        set4(f, c.fwd(), 0);
    };
    camVecs(cam, u.camPos, u.camRight, u.camUp, u.camFwd, (float)simTime);
    camVecs(prevCam, u.prevCamPos, u.prevRight, u.prevUp, u.prevFwd, 0);

    // sun and sky share one physical model: sky LUT is per unit sun irradiance, sun colour = irradiance * transmittance
    // key light: the sun, or the moon once it lights the ground more (night); both light the sky LUT
    V3 sd = sunDirection(), md = moonDirection();
    bool moonKey = keyIsMoon();
    float moonI = S.sunIntensity * moonlightScale() * S.nightBoost;
    if (moonKey) {
        set4(u.sunDir, md, cosf(0.26f * 3.14159265f / 180.f));
        V3 mc = sunTransmittance(md) * moonI;
        set4(u.sunColor, V3{mc.x, mc.y * 0.96f, mc.z * 0.9f}, S.skyIntensity);  // lunar regolith reddens a little
    } else {
        set4(u.sunDir, sd, cosf(S.sunRadiusDeg * 3.14159265f / 180.f));
        set4(u.sunColor, sunTransmittance(sd) * S.sunIntensity, S.skyIntensity);
    }
    set4(u.sun2, sd, S.sunIntensity);
    set4(u.moon0, md, cosf(0.26f * 3.14159265f / 180.f));
    {
        float elong = S.moonPhase * 2.f * 3.14159265f;
        set4(u.moon1, V3{fabsf(S.moonPhase - 0.5f) * 2.f * 3.14159265f, 0.5f * (1.f - cosf(elong)), moonI}, S.nightBoost);
        set4(u.night0, V3{S.stars, S.milkyWay, S.airglow}, S.nightBoost / S.sceneLum);
        // the sky turns with the sun: its angle about the pole in the equatorial frame (south = 0, east +)
        V3 P = celestialPole(), e1 = normalize(V3{0.f, cosf(S.latitude * 3.14159265f / 180.f), sinf(S.latitude * 3.14159265f / 180.f)}), e2 = cross(P, e1);
        set4(u.night1, V3{atan2f(dot(sd, e2), dot(sd, e1)), S.latitude * 3.14159265f / 180.f, S.twinkle}, moonKey ? 1.f : 0.f);
        // Purkinje: scene units -> physical cd/m^2; at night the scene is lit S.nightBoost x brighter than physical
        float boost = (moonKey || sd.y < 0.f) ? S.nightBoost : 1.f;
        set4(u.purk, V3{S.purkinje ? 1.f : 0.f, S.sceneLum / boost, S.nightBoost}, 0.f);
    }
    float wd = S.windDirDeg * 3.14159265f / 180.f;
    u.wind[0] = cosf(wd); u.wind[1] = sinf(wd); u.wind[2] = S.windStrength; u.wind[3] = S.windSpeed;
    u.params[0] = S.exposure; u.params[1] = (float)S.historyMax; u.params[2] = S.fireflyClamp; u.params[3] = S.spatialRadius;
    u.res[0] = W; u.res[1] = H; u.res[2] = frameIndex; u.res[3] = 0;  // light count, filled below
    u.settings[0] = (uint32_t)S.raysFirst; u.settings[1] = (uint32_t)S.bounces; u.settings[2] = flags(); u.settings[3] = (uint32_t)S.viewMode | (uint32_t)std::clamp(S.giCheckerCell, 1, 32) << 16 | (S.giQuarter ? 1u : 0u) << 24;  // view mode | checker cell (px) | quarter rate
    u.params2[0] = S.giRadius; u.params2[1] = S.giMaxM; u.params2[2] = S.sunIntensity; u.params2[3] = S.fogScale;
    set4(u.clouds, V3{S.cloudCoverage * S.cumulus, S.cloudDensity, S.cloudSpeed}, S.clouds ? 1.f : 0.f);
    set4(u.clouds2, V3{S.cloudBase, S.cloudThickness, S.cloudShadow}, 0);
    set4(u.clouds3, V3{S.cloudCoverage * S.stratus, S.stratusLumpy, S.stratusBase}, S.stratusThick);
    set4(u.clouds4, V3{S.cloudCoverage * S.cumulonimbus, S.cbTop, S.cloudCoverage * S.cirrus}, S.cirrusAlt);
    set4(u.clouds5, V3{S.cirrusLength, S.cirrusFine, S.cirrusPuff}, 0);
}
