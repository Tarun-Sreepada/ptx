#version 460
#extension GL_GOOGLE_include_directive : require
#include "core/frame.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNrm;
layout(location = 0) out vec3 oP;
layout(location = 1) out vec3 oN;
layout(location = 2) flat out uint oMat;
layout(location = 3) out float oT;
layout(location = 4) flat out uint oBlade;
layout(location = 5) out vec3 oPrev;  // last frame's position (motion vectors)

void main() {
    oP = inPos;
    oPrev = inPos;
    oN = inNrm;
    oMat = 0u;
    oT = 0.0;
    oBlade = 0u;
    gl_Position = project(inPos);
}
