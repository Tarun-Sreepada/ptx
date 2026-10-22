#version 460
#extension GL_GOOGLE_include_directive : require
#include "frame.glsl"
#include "model.glsl"

// Streamed model instances. No vertex input: vertices are pulled from the mesh's buffer address,
// the per-instance 3x4 transform + mesh slot come from modelXf (filled after CPU frustum culling).
layout(location = 0) out vec3 oP;
layout(location = 1) out vec3 oN;
layout(location = 2) out vec2 oUV;
layout(location = 3) flat out uint oSlot;
layout(location = 4) flat out float oFade;  // > 0: tree in the impostor crossfade band
layout(location = 5) flat out uint oInst;

void main() {
    uint b = uint(gl_InstanceIndex) * 4u;
    mat3x4 xf = mat3x4(modelXf[b], modelXf[b + 1u], modelXf[b + 2u]);  // rows
    oSlot = floatBitsToUint(modelXf[b + 3u].x);
    oFade = modelXf[b + 3u].y;
    oInst = uint(gl_InstanceIndex);
    ModelVertex v = modelVertex(meshes[oSlot], uint(gl_VertexIndex));
    oP = vec4(v.p, 1.0) * xf;
    mat3 m = transpose(mat3(xf));  // linear part, column-major
    oN = transpose(inverse(m)) * v.n;
    oUV = v.uv;
    gl_Position = project(oP);
}
