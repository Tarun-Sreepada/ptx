// Instance culling on the GPU (vegcull.comp, rockcull.comp): view frustum, and last frame's occlusion grid with the
// same rules as render/occlusion.cpp occluded(). occOn: the grid is valid this frame.
#ifndef SCENE_CULL
#define SCENE_CULL
layout(binding = 106, std430) readonly buffer Occ { float occ[]; };
const uint OCC_HEADER = 16u;
const float DEPTH_SLACK = 0.03, DEPTH_SLACK_M = 0.5, MAX_CAM_MOVE = 4.0;

bool inFrustum(vec3 c, float r) {
    vec3 d = c - F.camPos.xyz;
    float z = dot(d, F.camFwd.xyz);
    if (z < -r) return false;
    float th = F.camUp.w, tw = th * F.camRight.w;
    // side planes through the eye: |x| <= z * tan + r * sqrt(1 + tan^2)
    return abs(dot(d, F.camRight.xyz)) <= z * tw + r * sqrt(1.0 + tw * tw) && abs(dot(d, F.camUp.xyz)) <= z * th + r * sqrt(1.0 + th * th);
}

bool occluded(vec3 c, float r, bool occOn) {
    if (!occOn) return false;
    vec3 gPos = vec3(occ[0], occ[1], occ[2]), gRight = vec3(occ[4], occ[5], occ[6]), gUp = vec3(occ[8], occ[9], occ[10]),
         gFwd = vec3(occ[12], occ[13], occ[14]);
    uint gw = floatBitsToUint(occ[3]), gh = floatBitsToUint(occ[15]);
    float sx = occ[7] * occ[11], sy = occ[11];
    vec3 mv = F.camPos.xyz - gPos;
    if (dot(mv, mv) > MAX_CAM_MOVE * MAX_CAM_MOVE) return false;
    r += length(mv);
    vec3 d = c - gPos;
    float x = dot(d, gRight), y = dot(d, gUp), z = dot(d, gFwd);
    float zn = z - r;
    if (zn < 0.5) return false;
    float zf = z + r;
    float x0 = min((x - r) / zn, (x - r) / zf) / sx, x1 = max((x + r) / zn, (x + r) / zf) / sx;
    float y0 = min((y - r) / zn, (y - r) / zf) / sy, y1 = max((y + r) / zn, (y + r) / zf) / sy;
    if (x0 < -1.0 || x1 > 1.0 || y0 < -1.0 || y1 > 1.0) return false;
    int tx0 = int((x0 * 0.5 + 0.5) * float(gw)), tx1 = min(int((x1 * 0.5 + 0.5) * float(gw)), int(gw) - 1);
    int ty0 = int((0.5 - y1 * 0.5) * float(gh)), ty1 = min(int((0.5 - y0 * 0.5) * float(gh)), int(gh) - 1);
    float limit = (length(d) - r) / (1.0 + DEPTH_SLACK) - DEPTH_SLACK_M;
    // a big sphere close by spans many tiles: past 64 of them, don't bother (it's probably visible anyway)
    if ((tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 64) return false;
    for (int ty = ty0; ty <= ty1; ty++)
        for (int tx = tx0; tx <= tx1; tx++)
            if (occ[OCC_HEADER + uint(ty) * gw + uint(tx)] >= limit) return false;
    return true;
}

#endif
