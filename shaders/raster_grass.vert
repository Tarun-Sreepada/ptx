#version 460
#extension GL_GOOGLE_include_directive : require
#include "frame.glsl"

// Same animated vertex buffer the grass BLAS is built from, so raster and RT see identical blades.
layout(binding = 6, std430) readonly buffer BladeSeed { uint bladeSeed[]; };
layout(location = 0) in vec3 inPos;
layout(location = 0) out vec3 oP;
layout(location = 1) out vec3 oN;
layout(location = 2) flat out uint oMat;
layout(location = 3) out float oT;
layout(location = 4) flat out uint oBlade;

void main() {
    uint v = uint(gl_VertexIndex);
    oP = inPos;
    oN = vec3(0);
    oMat = 1u;
    oT = float((v % 7u) / 2u) / 3.0; // 7 verts per blade, 2 per level, tip = level 3
    oBlade = bladeSeed[v / 7u];
    gl_Position = project(inPos);
}
