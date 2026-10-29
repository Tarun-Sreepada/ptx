#version 460
#extension GL_GOOGLE_include_directive : require
#include "scene/model_xform.glsl"

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

void main() {
    ModelOut o = modelXform(uint(gl_InstanceIndex), uint(gl_VertexIndex), false);
    oP = o.P; oN = o.N; oUV = o.uv; oSlot = o.slot; oFade = o.fade; oInst = uint(gl_InstanceIndex); oPrev = o.prev; oSway = o.sway;
    gl_Position = project(oP);
}
