#include "render/culling.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include "core/prof.h"
#include "core/simd.h"
#include "app/settings.h"
#include "core/limits.h"
#include "core/math.h"
#include "render/occlusion.h"
#include "render/veg_gpu.h"
#include "render/targets.h"
#include "scene/camera.h"
#include "scene/model_instances.h"
#include "scene/models.h"

// Frustum-cull model instances (AVX2), write raster transforms, group consecutive same-mesh items into instanced draws.
void frustumPlanes(float planes[5][4]) {
    float th = tanf(cam.fov() * 0.5f * 3.14159265f / 180.f), aspect = (float)W / H;
    V3 f = cam.fwd(), r = cam.right(), u = cam.up();
    V3 edges[4][2] = {{f + r * (th * aspect), u}, {f - r * (th * aspect), u}, {f + u * th, r}, {f - u * th, r}};
    for (int i = 0; i < 4; i++) {
        V3 n = normalize(cross(edges[i][0], edges[i][1]));
        if (dot(n, f) < 0) n = n * -1.f;  // orient inward regardless of handedness
        set4(planes[i], n, -dot(n, cam.pos));
    }
    set4(planes[4], f, -dot(f, cam.pos) + 0.05f);
}

// Previous-frame transform of raster instance k (second half of modelXfBuf): the vertex shader projects both and the
// difference goes into the G-buffer's velocity target. prevSlot: the mesh whose vertices it had (skinned meshes:
// last frame's skinned copy). Instances not given one get their current transform (static) in finishPrevXf.
static std::vector<uint8_t> prevSet(MAX_MODEL_INST, 0);

void setPrevXf(uint32_t k, const float m[3][4], uint32_t prevSlot) {
    float* d = (float*)modelXfBuf.map + XF_SLOTS * 16 + (size_t)k * 16;
    memcpy(d, m, 48);
    memcpy(d + 12, &prevSlot, 4);
    d[13] = d[14] = d[15] = 0;
    prevSet[k] = 1;
}

void finishPrevXf() {
    const float* cur = (const float*)modelXfBuf.map;
    for (uint32_t k = 0; k < numModelVisible; k++) {
        if (!prevSet[k]) memcpy((float*)modelXfBuf.map + XF_SLOTS * 16 + (size_t)k * 16, cur + (size_t)k * 16, 64);
        prevSet[k] = 0;
    }
}

void cullModels() {
    PROF_FUNC();
    drawRuns.clear();
    numModelVisible = 0;
    beginOcclusion(cam.pos);
    if (!numModelInst || !S.rasterPrimary) return;
    float planes[5][4];
    frustumPlanes(planes);
    cullAabbs(aabb[0].data(), aabb[1].data(), aabb[2].data(), aabb[3].data(), aabb[4].data(), aabb[5].data(), numModelInst, planes, 5,
              visibleItems.data());
    float* xf = (float*)modelXfBuf.map;
    for (uint32_t i = 0; i < numModelInst; i++) {
        if (!visibleItems[i]) continue;
        V3 lo{aabb[0][i], aabb[1][i], aabb[2][i]}, hi{aabb[3][i], aabb[4][i], aabb[5][i]}, e = (hi - lo) * 0.5f;
        if (occluded(lo + e, sqrtf(dot(e, e)))) continue;
        const DrawItem& it = drawItems[i];
        uint32_t k = numModelVisible++;
        memcpy(xf + k * 16, it.xf.m, 48);
        uint32_t slot = it.m->meshes[it.mesh].slot;
        memcpy(xf + k * 16 + 12, &slot, 4);
        xf[k * 16 + 13] = xf[k * 16 + 14] = xf[k * 16 + 15] = 0;  // no impostor fade
        if (!drawRuns.empty() && drawRuns.back().m == it.m && drawRuns.back().mesh == it.mesh) drawRuns.back().count++;
        else drawRuns.push_back({it.m, it.mesh, k, 1});
    }
}
