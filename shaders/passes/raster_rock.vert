#version 460
#extension GL_GOOGLE_include_directive : require
#include "core/frame.glsl"

// Instanced rocks; per-instance 3x4 transforms (rotation * uniform scale + translation), same as the TLAS, and a
// 4th row whose x is the variant's base vertex: all variants share the index list, the vertex comes from rv.
layout(binding = 29, std430) readonly buffer RockV { float rv[]; };  // pos.xyz, normal.xyz
layout(binding = 31, std430) readonly buffer RockXf { vec4 rockXf[]; };
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNrm;
layout(location = 0) out vec3 oP;
layout(location = 1) out vec3 oN;
layout(location = 2) flat out uint oMat;
layout(location = 3) out float oT;
layout(location = 4) flat out uint oBlade;
layout(location = 5) out vec3 oPrev;  // last frame's position (second half of rockXf)

void main() {
    uint b = uint(gl_InstanceIndex) * 4u;
    vec4 r0 = rockXf[b], r1 = rockXf[b + 1u], r2 = rockXf[b + 2u];
    uint v = 6u * (uint(gl_VertexIndex) + floatBitsToUint(rockXf[b + 3u].x));
    vec4 p = vec4(rv[v], rv[v + 1u], rv[v + 2u], 1.0);
    vec3 inN = vec3(rv[v + 3u], rv[v + 4u], rv[v + 5u]);
    oP = vec3(dot(r0, p), dot(r1, p), dot(r2, p));
    uint pb = uint(rockXf.length()) / 2u + b;
    oPrev = vec3(dot(rockXf[pb], p), dot(rockXf[pb + 1u], p), dot(rockXf[pb + 2u], p));
    oN = vec3(dot(r0.xyz, inN), dot(r1.xyz, inN), dot(r2.xyz, inN));
    oMat = 2u;
    oT = 0.0;
    oBlade = 0u;
    gl_Position = project(oP);
}
