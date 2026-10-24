// Camera rays (this and last frame), world positions from depth, reprojection.
#ifndef CORE_CAMERA
#define CORE_CAMERA
#include "core/frame.glsl"
#include "core/bindings.glsl"
#include "materials/model.glsl"

// ---------------------------------------------------------------- camera / sky
vec3 cameraRay(ivec2 p) {
    vec2 uv = (vec2(p) + 0.5 + F.jitter.xy) / vec2(F.res.xy);
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return normalize(F.camFwd.xyz + ndc.x * F.camUp.w * F.camRight.w * F.camRight.xyz + ndc.y * F.camUp.w * F.camUp.xyz);
}

vec3 prevCameraRay(ivec2 p) {
    vec2 uv = (vec2(p) + 0.5 + F.jitter.zw) / vec2(F.res.xy);
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return normalize(F.prevFwd.xyz + ndc.x * F.prevUp.w * F.prevRight.w * F.prevRight.xyz + ndc.y * F.prevUp.w * F.prevUp.xyz);
}

// World position reconstructed from depth: xyz = position, w = distance (-1 = sky)
vec4 loadPos(ivec2 p) {
    float t = imageLoad(gDepth, p).r;
    return vec4(F.camPos.xyz + cameraRay(p) * max(t, 0.0), t);
}
// Where pixel p's surface is in the ray-traced scene: the G-buffer position less the wind sway the raster pass gave
// vegetation (gEmis.w, gVel.w: its xz offset; the BVH holds plants at rest). Lighting rays start there, so a
// swaying leaf isn't shadowed by its own resting copy; reprojection keeps the position as seen (loadPos).
vec3 restOffset(ivec2 p) { return vec3(imageLoad(gEmis, p).w, 0.0, imageLoad(gVel, p).w); }
vec4 loadRayPos(ivec2 p) {
    vec4 P = loadPos(p);
    P.xyz -= restOffset(p);
    return P;
}
vec4 loadPrevPos(ivec2 p) {
    float t = imageLoad(prevDepth, p).r;
    return vec4(F.prevCamPos.xyz + prevCameraRay(p) * max(t, 0.0), t);
}

// Continuous pixel position of world point P in the previous frame's camera.
bool prevPixel(vec3 P, out vec2 pp) {
    vec3 d = P - F.prevCamPos.xyz;
    float z = dot(d, F.prevFwd.xyz);
    if (z <= 1e-3) return false;
    float x = dot(d, F.prevRight.xyz) / (z * F.prevUp.w * F.prevRight.w);
    float y = dot(d, F.prevUp.xyz) / (z * F.prevUp.w);
    pp = vec2(x * 0.5 + 0.5, 0.5 - y * 0.5) * vec2(F.res.xy) - 0.5 - F.jitter.zw;
    return true;
}
bool inScreen(ivec2 q) { return q.x >= 0 && q.y >= 0 && q.x < int(F.res.x) && q.y < int(F.res.y); }

bool similarPrev(ivec2 q, vec3 P, vec3 n, uint mat) {
    if (!inScreen(q)) return false;
    vec4 qp = loadPrevPos(q);
    if (qp.w < 0.0) return false;
    vec4 qn = imageLoad(prevNorm, q);
    if (uint(qn.w + 0.5) != mat) return false;
    float dist = length(P - F.camPos.xyz);
    if (abs(dot(P - qp.xyz, n)) > 0.02 + 0.01 * dist) return false;
    if (dot(n, qn.xyz) < (mat == MAT_GRASS ? 0.5 : 0.9)) return false;
    return true;
}
#endif
