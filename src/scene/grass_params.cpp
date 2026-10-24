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
#include "scene/grass.h"

// grass grid around the camera + view cone (fov + margin) used for culling; refit the BLAS while the camera rests
void fillGrassParams(UBO& u) {
    const float d2r = 3.14159265f / 180.f;
    float cs = 1.f / sqrtf(S.grassDensity), R = S.drawDist;
    grassGridN = (uint32_t)ceilf(2 * R / cs) + 1;
    float vfov = cam.fovDeg * d2r, hfov = 2 * atanf(tanf(vfov / 2) * W / H), m = S.fovMargin * d2r;
    u.grassA[0] = cs; u.grassA[1] = S.grassLength; u.grassA[2] = S.lodStart; u.grassA[3] = R;
    u.grassB[0] = tanf(std::min((hfov + m) / 2, 1.5f)); u.grassB[1] = tanf(std::min((vfov + m) / 2, 1.5f)); u.grassB[2] = S.keepRadius; u.grassB[3] = S.offscreenDensity;
    u.grassGrid[0] = (int)floorf((cam.pos.x - R) / cs); u.grassGrid[1] = (int)floorf((cam.pos.z - R) / cs);
    {
        V3 f = cam.fwd();
        float dx = cam.pos.x - grassBuildPos[0], dy = cam.pos.y - grassBuildPos[1], dz = cam.pos.z - grassBuildPos[2];
        bool moved = dx * dx + dy * dy + dz * dz > 1.f || f.x * grassBuildFwd[0] + f.y * grassBuildFwd[1] + f.z * grassBuildFwd[2] < 0.996f;  // 1 m, ~5 deg
        grassRefit = S.grassRefit && grassBuilt && !moved && !terrainDirty && ++grassFrames < GRASS_REBUILD;
        if (grassRefit) bladeCap = grassBuildCap;  // an update keeps the primitive count
        else {
            grassFrames = 0;
            grassBuildCap = bladeCap;
            grassBuildPos[0] = cam.pos.x; grassBuildPos[1] = cam.pos.y; grassBuildPos[2] = cam.pos.z;
            grassBuildFwd[0] = f.x; grassBuildFwd[1] = f.y; grassBuildFwd[2] = f.z;
        }
    }
    u.grassGrid[2] = (int)grassGridN; u.grassGrid[3] = (int)bladeCap;
}
