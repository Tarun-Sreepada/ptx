#version 460
#extension GL_GOOGLE_include_directive : require
#include "frame.glsl"
#include "impostor.glsl"

// Far trees: one camera-facing quad over the variant's bounding sphere; the fragment shader ray-marches
// the octahedral frames and writes the real surface position / normal into the G-buffer.
layout(location = 0) out vec3 oP;
layout(location = 1) flat out uint oInst;

void main() {
    const vec2 corners[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
    uint b = uint(gl_InstanceIndex) * 3u;
    vec4 a = impInst[b], c = impInst[b + 1u], bnd = impInst[b + 2u];
    float cy = cos(c.x), sy = sin(c.x);
    vec3 Cw = a.xyz + rotY(bnd.xyz, cy, sy) * a.w;
    float R = bnd.w * a.w * 1.08;  // a little margin for perspective
    vec3 toCam = normalize(F.camPos.xyz - Cw);
    vec3 r = normalize(cross(abs(toCam.y) > 0.99 ? vec3(0, 0, 1) : vec3(0, 1, 0), toCam)), u = cross(toCam, r);
    vec2 k = corners[gl_VertexIndex];
    oP = Cw + (r * k.x + u * k.y) * R;
    oInst = uint(gl_InstanceIndex);
    gl_Position = project(oP);
}
