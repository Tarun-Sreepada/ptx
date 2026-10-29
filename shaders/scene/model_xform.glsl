// The raster transform of one model vertex, shared by raster_model.vert and raster_model.mesh: instance inst's
// 3x4 transform + mesh slot from modelXf (last frame's in its second half, for motion vectors), wing flaps for
// critters and wind sway for plants (sway.glsl).
#ifndef SCENE_MODEL_XFORM
#define SCENE_MODEL_XFORM
#include "core/frame.glsl"
#include "materials/model.glsl"
#include "scene/sway.glsl"

// Flapping wings (birds, butterflies; sim/critters.h): what lies beyond the body (|x| > CRITTER_BODY, model space)
// turns about the body's long axis, both wings together. w: phase (fraction) + amplitude x 10 (integer, radians).
const float CRITTER_BODY = 0.06;
vec3 flapWing(vec3 p, float hz, float w, float t, inout vec3 n) {
    float ax = abs(p.x) - CRITTER_BODY;
    if (ax <= 0.0) return p;
    float s = sign(p.x), ang = sin(6.2831853 * (hz * t + fract(w))) * floor(w) * 0.1 * s;
    float c = cos(ang), sn = sin(ang);
    n = vec3(n.x * c - n.y * sn, n.x * sn + n.y * c, n.z);
    return vec3(s * CRITTER_BODY + s * ax * c, p.y + ax * sn * s, p.z);
}

struct ModelOut { vec3 P, N; vec2 uv; uint slot; float fade; vec3 prev; vec2 sway; };

// uniformScale: rotation x uniform scale only (vegetation), so the normal needs no inverse matrix
ModelOut modelXform(uint inst, uint vi, bool uniformScale) {
    ModelOut o;
    uint b = inst * 4u;
    mat3x4 xf = mat3x4(modelXf[b], modelXf[b + 1u], modelXf[b + 2u]);  // rows
    o.slot = floatBitsToUint(modelXf[b + 3u].x);
    o.fade = modelXf[b + 3u].y;
    ModelVertex v = modelVertex(meshes[o.slot], vi);
    vec4 sw = modelXf[b + 3u];  // z: plant height (m, 0 = rigid) or minus the wing beats per second; w: phase etc.
    vec3 lp = v.p, ln = v.n, prevN = v.n;
    if (sw.z < 0.0) lp = flapWing(v.p, -sw.z, sw.w, F.camPos.w, ln);
    o.P = vec4(lp, 1.0) * xf;
    mat3 m = transpose(mat3(xf));  // linear part, column-major
    o.N = uniformScale ? m * ln : transpose(inverse(m)) * ln;  // (normalised in the fragment shader)
    o.uv = v.uv;
    uint pb = uint(modelXf.length()) / 2u + b;
    mat3x4 pxf = mat3x4(modelXf[pb], modelXf[pb + 1u], modelXf[pb + 2u]);
    uint ps = floatBitsToUint(modelXf[pb + 3u].x);
    vec3 pp = ps == o.slot ? v.p : modelVertex(meshes[ps], vi).p;
    if (sw.z < 0.0) pp = flapWing(pp, -sw.z, sw.w, F.camPos.w - F.post2.w, prevN);
    o.prev = vec4(pp, 1.0) * pxf;
    o.sway = vec2(0);
    if (sw.z > 0.0) {
        vec3 base = vec3(xf[0].w, xf[1].w, xf[2].w), pbase = vec3(pxf[0].w, pxf[1].w, pxf[2].w);
        vec3 d = plantSway(o.P - base, base, sw.z, sw.w, F.camPos.w);
        o.prev += plantSway(o.prev - pbase, pbase, sw.z, sw.w, F.camPos.w - F.post2.w);
        o.P += d;
        o.sway = d.xz;
    }
    return o;
}
#endif
