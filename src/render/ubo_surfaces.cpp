#include "render/ubo_fill.h"

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

void fillSurfaceParams(UBO& u) {
    set4(u.grassC, V3{S.grassClump, S.grassClumpSize, S.grassWidth}, S.grassCurve);
    set4(u.grassD, V3{S.grassTwist, S.grassTilt, S.grassHeightVar}, S.grassDead);
    set4(u.ground2, V3{S.groundWarp, S.groundMottle, S.groundMottleScale}, S.darkSpots);
    set4(u.ground3, V3{S.darkSpotSize, S.groundGrit, S.groundCavity}, S.tileBreakup);
    set4(u.vegCol, V3{S.grassColorVar, S.groundSat, 0}, 0);
    set4(u.sssP, V3{S.sss ? 1.f : 0.f, (float)S.sssSteps, S.sssScale}, S.snowSSS ? 1.f : 0.f);
    for (int i = 0; i < 4; i++) u.groundTex[i] = groundSlots[i];
    u.groundP[0] = 4.f; u.groundP[1] = (float)groundTexSize; u.groundP[2] = groundMaxH[0]; u.groundP[3] = groundMaxH[1];
    u.groundS[0] = S.groundNormal; u.groundS[1] = S.parallax ? 1.f : 0.f; u.groundS[2] = S.parallaxDist;
    u.groundS[3] = (groundReady || S.groundMode == 3) && S.groundTextures ? 1.f : 0.f;
    updateGroundVT(u.vtRect);
    u.vtP[0] = VT_TEXEL0; u.vtP[1] = (float)VT_N; u.vtP[2] = (float)VT_L; u.vtP[3] = S.groundMode == 3 ? 1.f : 0.f;
    {   // wind map window follows the camera in whole cells; cells that scroll in are re-initialised by wind.comp
        static int prevOrigin[2] = {INT32_MIN / 2, INT32_MIN / 2};
        static double lastT = simTime;
        int ox = (int)floorf(cam.pos.x) - 128, oz = (int)floorf(cam.pos.z) - 128;
        u.windGrid[0] = ox; u.windGrid[1] = oz; u.windGrid[2] = prevOrigin[0]; u.windGrid[3] = prevOrigin[1];
        prevOrigin[0] = ox; prevOrigin[1] = oz;
        float fdt = std::clamp((float)(simTime - lastT), 0.f, 1.f / 20.f);
        lastT = simTime;
        u.windSim[0] = fdt * 0.5f; u.windSim[1] = S.gustStrength; u.windSim[2] = S.gustSpreadDeg * 3.14159265f / 180.f; u.windSim[3] = S.windMap ? 1.f : 0.f;
        u.windSim2[0] = S.windWaveSpeed * S.windWaveSpeed * 4.f; u.windSim2[1] = S.windDamping; u.windSim2[2] = S.windRelax; u.windSim2[3] = (float)simTime;
    }
}
