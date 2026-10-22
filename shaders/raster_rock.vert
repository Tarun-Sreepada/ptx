#version 460
#extension GL_GOOGLE_include_directive : require
#include "frame.glsl"

// Instanced rocks; per-instance 3x4 transforms (rotation * uniform scale + translation), same as the TLAS.
layout(binding = 31, std430) readonly buffer RockXf { vec4 rockXf[]; };
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNrm;
layout(location = 0) out vec3 oP;
layout(location = 1) out vec3 oN;
layout(location = 2) flat out uint oMat;
layout(location = 3) out float oT;
layout(location = 4) flat out uint oBlade;

void main() {
    uint b = uint(gl_InstanceIndex) * 3u;
    vec4 r0 = rockXf[b], r1 = rockXf[b + 1u], r2 = rockXf[b + 2u];
    vec4 p = vec4(inPos, 1.0);
    oP = vec3(dot(r0, p), dot(r1, p), dot(r2, p));
    oN = vec3(dot(r0.xyz, inNrm), dot(r1.xyz, inNrm), dot(r2.xyz, inNrm));
    oMat = 2u;
    oT = 0.0;
    oBlade = 0u;
    gl_Position = project(oP);
}
