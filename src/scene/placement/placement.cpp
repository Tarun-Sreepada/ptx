#include "scene/placement/placement.h"

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

std::vector<std::vector<PlacedCell>> placeRows(float cell, float range, const CellFn& fn) {
    int z0 = (int)floorf((cam.pos.z - range) / cell), z1 = (int)floorf((cam.pos.z + range) / cell);
    int x0 = (int)floorf((cam.pos.x - range) / cell), x1 = (int)floorf((cam.pos.x + range) / cell);
    std::vector<std::vector<PlacedCell>> rows(z1 - z0 + 1);
    jobs::parallelFor((uint32_t)rows.size(), 4, [&](uint32_t b, uint32_t e) {
        for (uint32_t r = b; r < e; r++)
            for (int ix = x0; ix <= x1; ix++) {
                PlacedCell pl;
                if (fn(ix, z0 + (int)r, pl)) rows[r].push_back(pl);
            }
    });
    return rows;
}

void placeScene() {
    // rocks: hashed 7 m grid out to the terrain's edge, TLAS instances + raster transforms. No fixed radius: each
    // rock is kept while it covers >= ~1.5 output pixels (screen-size LOD), so big boulders stay visible across the
    // field and only rocks too small to resolve drop out, where the pop is invisible.
    const float RR = S.viewDist;
    const float pixAngle = 2.f * tanf(cam.fovDeg * 0.5f * 3.14159265f / 180.f) / (float)OH, keepPx = 1.5f;
    // Placement is ~70k grid cells of noise + height lookups (several ms): redo it only after moving 2 m or a
    // world / setting change. The TLAS instances and raster transforms stay valid in their buffers meanwhile.
    static V3 placedAt{1e9f, 0, 0};
    static uint32_t placedModels = UINT32_MAX;
    static float placedDensity = -1.f, placedPix = 0.f, placedView = 0.f;
    V3 mv = cam.pos - placedAt;
    float step = std::max(2.f, S.viewDist / 300.f);  // a far view costs more to place: redo it less often (it's far away)
    if (!placeDirty && dot(mv, mv) < step * step && placedModels == numModelInst && placedDensity == S.treeDensity && placedPix == pixAngle &&
        placedView == S.viewDist)
        return;
    placedView = S.viewDist;
    placeDirty = false;
    placedAt = cam.pos;
    placedModels = numModelInst;
    placedDensity = S.treeDensity;
    placedPix = pixAngle;
    PlacementCtx c{RR, pixAngle, keepPx, (VkAccelerationStructureInstanceKHR*)instBuf.map, (float*)rockXf.map, stream::zones()};
    placeRocks(c);
    placeTrees(c);
    placePlants(c);
}
