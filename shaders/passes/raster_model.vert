#version 460
#extension GL_GOOGLE_include_directive : require
#include "core/frame.glsl"
#include "materials/model.glsl"
#include "scene/sway.glsl"

// Streamed model instances. No vertex input: vertices are pulled from the mesh's buffer address,
// the per-instance 3x4 transform + mesh slot come from modelXf (filled after CPU frustum culling), and for
// plants their height and phase for swaying in the wind (sway.glsl).
layout(location = 0) out vec3 oP;
layout(location = 1) out vec3 oN;
layout(location = 2) out vec2 oUV;
layout(location = 3) flat out uint oSlot;
layout(location = 4) flat out float oFade;  // > 0: tree in the impostor crossfade band
layout(location = 5) flat out uint oInst;
layout(location = 6) out vec3 oPrev;  // last frame's position: previous transform + previous vertices (second half of modelXf)
layout(location = 7) out vec2 oSway;  // xz the wind moved this vertex away from where the BVH has it

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

void main() {
    uint b = uint(gl_InstanceIndex) * 4u;
    mat3x4 xf = mat3x4(modelXf[b], modelXf[b + 1u], modelXf[b + 2u]);  // rows
    oSlot = floatBitsToUint(modelXf[b + 3u].x);
    oFade = modelXf[b + 3u].y;
    oInst = uint(gl_InstanceIndex);
    ModelVertex v = modelVertex(meshes[oSlot], uint(gl_VertexIndex));
    vec4 sw = modelXf[b + 3u];  // z: plant height (m, 0 = rigid) or minus the wing beats per second; w: phase etc.
    vec3 lp = v.p, ln = v.n, prevN = v.n;
    if (sw.z < 0.0) lp = flapWing(v.p, -sw.z, sw.w, F.camPos.w, ln);
    oP = vec4(lp, 1.0) * xf;
    mat3 m = transpose(mat3(xf));  // linear part, column-major
    oN = transpose(inverse(m)) * ln;
    oUV = v.uv;
    uint pb = uint(modelXf.length()) / 2u + b;
    mat3x4 pxf = mat3x4(modelXf[pb], modelXf[pb + 1u], modelXf[pb + 2u]);
    uint ps = floatBitsToUint(modelXf[pb + 3u].x);
    vec3 pp = ps == oSlot ? v.p : modelVertex(meshes[ps], uint(gl_VertexIndex)).p;
    if (sw.z < 0.0) pp = flapWing(pp, -sw.z, sw.w, F.camPos.w - F.post2.w, prevN);
    oPrev = vec4(pp, 1.0) * pxf;
    oSway = vec2(0);
    if (sw.z > 0.0) {
        vec3 base = vec3(xf[0].w, xf[1].w, xf[2].w), pbase = vec3(pxf[0].w, pxf[1].w, pxf[2].w);
        vec3 d = plantSway(oP - base, base, sw.z, sw.w, F.camPos.w);
        oPrev += plantSway(oPrev - pbase, pbase, sw.z, sw.w, F.camPos.w - F.post2.w);
        oP += d;
        oSway = d.xz;
    }
    gl_Position = project(oP);
}
