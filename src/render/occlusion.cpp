#include "render/occlusion.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include "app/settings.h"
#include "gpu/memory.h"

Buffer occBuf;
uint32_t occCulled = 0;

namespace {
constexpr float DEPTH_SLACK = 0.03f, DEPTH_SLACK_M = 0.5f;  // nearest sphere point must be this far past the occluder
constexpr float MAX_CAM_MOVE = 4.f;                           // camera jumped further than this since the grid: no culling
std::vector<float> grid;
V3 gPos, gRight, gUp, gFwd, camNow;
float sx = 1, sy = 1;  // tan(fov/2) * aspect, tan(fov/2)
uint32_t gw = 0, gh = 0;
bool valid = false;
}  // namespace

void createOcclusionBuffer() {
    VkDeviceSize n = (OCC_HEADER + OCC_MAX_W * OCC_MAX_H) * 4;
    occBuf = mkBuf(n, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memset(occBuf.map, 0, n);  // gw = gh = 0: nothing to test against yet
}

void beginOcclusion(V3 camPos) {
    occCulled = 0;
    const float* h = (const float*)occBuf.map;
    camNow = camPos;
    gPos = {h[0], h[1], h[2]};
    gRight = {h[4], h[5], h[6]};
    gUp = {h[8], h[9], h[10]};
    gFwd = {h[12], h[13], h[14]};
    sx = h[7] * h[11];
    sy = h[11];
    memcpy(&gw, h + 3, 4);
    memcpy(&gh, h + 15, 4);
    V3 d = camPos - gPos;
    valid = S.occlusionCull && S.rasterPrimary && gw && gh && gw <= OCC_MAX_W && gh <= OCC_MAX_H && dot(d, d) < MAX_CAM_MOVE * MAX_CAM_MOVE;
    if (valid) grid.assign(h + OCC_HEADER, h + OCC_HEADER + gw * gh);  // copy: the GPU overwrites it this frame
}

bool occluded(V3 c, float r) {
    if (!valid) return false;
    // the camera moved since the grid was made: something hidden then may peek out now, so grow the sphere by that
    V3 mv = camNow - gPos;
    r += sqrtf(dot(mv, mv));
    V3 d = c - gPos;
    float x = dot(d, gRight), y = dot(d, gUp), z = dot(d, gFwd);
    float zn = z - r;
    if (zn < 0.5f) return false;  // reaches near or behind the old camera
    float zf = z + r;
    // x/z over the box around the sphere is extreme at its corners: a conservative screen rectangle
    float x0 = std::min((x - r) / zn, (x - r) / zf) / sx, x1 = std::max((x + r) / zn, (x + r) / zf) / sx;
    float y0 = std::min((y - r) / zn, (y - r) / zf) / sy, y1 = std::max((y + r) / zn, (y + r) / zf) / sy;
    if (x0 < -1.f || x1 > 1.f || y0 < -1.f || y1 > 1.f) return false;  // partly outside what the grid saw
    // ndc -> tiles (y down, matching cameraRay)
    int tx0 = (int)((x0 * 0.5f + 0.5f) * gw), tx1 = std::min((int)((x1 * 0.5f + 0.5f) * gw), (int)gw - 1);
    int ty0 = (int)((0.5f - y1 * 0.5f) * gh), ty1 = std::min((int)((0.5f - y0 * 0.5f) * gh), (int)gh - 1);
    float nearest = sqrtf(dot(d, d)) - r;  // ray distance to the closest point of the sphere
    float limit = nearest / (1.f + DEPTH_SLACK) - DEPTH_SLACK_M;
    for (int ty = ty0; ty <= ty1; ty++)
        for (int tx = tx0; tx <= tx1; tx++)
            if (grid[(size_t)ty * gw + tx] >= limit) return false;
    occCulled++;
    return true;
}

bool occlusionValid() { return valid; }
