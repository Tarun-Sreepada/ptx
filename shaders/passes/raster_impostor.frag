#version 460
#extension GL_GOOGLE_include_directive : require
#include "core/frame.glsl"
#include "materials/model.glsl"
#include "scene/impostor.glsl"

// Octahedral impostor G-buffer: the view ray is intersected with the planes of the 4 nearest frames, each
// frame is sampled at its own reprojected uv and the results are blended bilinearly. The reconstructed
// surface point (not the quad) is written as depth / position, so lighting and reprojection see the tree.
layout(binding = 45) uniform sampler2DArray impAlbedo;
layout(binding = 46) uniform sampler2DArray impNormal;
layout(location = 0) in vec3 oP;
layout(location = 1) flat in uint oInst;

layout(location = 0) out float outDepth;
layout(location = 1) out vec4 outNorm;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outEmis;
layout(location = 4) out vec4 outMat;
layout(location = 5) out vec4 outMat2;
layout(location = 6) out vec4 outTan;
layout(location = 7) out vec4 outVel;

void main() {
    uint b = oInst * 3u;
    vec4 a = impInst[b], c = impInst[b + 1u], bnd = impInst[b + 2u];
    float cy = cos(c.x), sy = sin(c.x), layer = c.y, fade = c.z;
    if (fadeNoise(gl_FragCoord.xy, F.res.z) >= fade) discard;  // mesh draws these pixels (crossfade band)
    vec3 C = bnd.xyz;
    float R = bnd.w;
    // view ray in the variant's local space (unrotated, unscaled)
    vec3 lo = invRotY(F.camPos.xyz - a.xyz, cy, sy) / a.w;
    vec3 ld = normalize(invRotY(oP - F.camPos.xyz, cy, sy));
    vec3 vc = normalize(lo - C);
    vec2 gp = (hemiOctEncode(vc) * 0.5 + 0.5) * float(IMP_G - 1);
    ivec2 g0 = clamp(ivec2(floor(gp)), ivec2(0), ivec2(IMP_G - 2));
    vec2 fr = clamp(gp - vec2(g0), 0.0, 1.0);

    float wSum = 0.0;
    vec3 alb = vec3(0), nrm = vec3(0), pos = vec3(0);
    for (int k = 0; k < 4; k++) {
        ivec2 o = ivec2(k & 1, k >> 1);
        float w = (o.x == 1 ? fr.x : 1.0 - fr.x) * (o.y == 1 ? fr.y : 1.0 - fr.y);
        if (w <= 1e-4) continue;
        ivec2 f = g0 + o;
        vec3 d = impDir(f), r, u;
        impBasis(d, r, u);
        float den = dot(ld, d);
        if (abs(den) < 1e-4) continue;
        vec3 q = lo + ld * (dot(C - lo, d) / den);
        vec2 s = vec2(dot(q - C, r), dot(q - C, u)) / R;
        if (abs(s.x) > 1.0 || abs(s.y) > 1.0) continue;
        vec2 uv = (vec2(f) + vec2(s.x, -s.y) * 0.5 + 0.5) / float(IMP_G);
        vec4 A = textureLod(impAlbedo, vec3(uv, layer), 0.0);
        vec4 N = textureLod(impNormal, vec3(uv, layer), 0.0);
        float wa = w * A.a;
        wSum += wa;
        alb += A.rgb * A.rgb * wa;
        nrm += N.xyz * wa;
        pos += (q - d * (N.w * R)) * wa;
    }
    if (wSum < 0.5) discard;
    alb /= wSum;
    pos /= wSum;
    vec3 P = a.xyz + rotY(pos, cy, sy) * a.w;
    vec3 n = normalize(rotY(nrm, cy, sy));
    vec3 v = P - F.camPos.xyz;
    if (dot(n, v) > 0.0) n = -n;
    float z = dot(v, F.camFwd.xyz);
    gl_FragDepth = 0.05 / max(z, 0.05);  // same reverse-Z as project()
    outDepth = length(v);
    outNorm = vec4(n, float(MAT_MODEL));
    outAlbedo = vec4(alb, 1.0);
    outEmis = vec4(0.0);
    outMat = vec4(0.8, 0.0, 1.0, 0.0);  // tree materials: the default dielectric
    outMat2 = vec4(0.0);
    vec3 t = abs(n.y) < 0.99 ? cross(vec3(0, 1, 0), n) : cross(vec3(1, 0, 0), n);
    outTan = vec4(normalize(t), -0.45);  // tree canopies: translucent leaves
    outVel = vec4(0);  // impostors are of standing trees
}
