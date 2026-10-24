// Volumetric fog in front of a pixel (volfog.comp output, depth-aware upsampling).
#ifndef LIGHTING_FOG
#define LIGHTING_FOG
#include "core/frame.glsl"
#include "core/bindings.glsl"

// Volumetric fog in front of render pixel p (volfog.comp, 1/F.fogRes.x resolution, bilinear): rgb in-scatter, a transmittance.
vec4 fogAt(ivec2 p) {
    if (F.fog0.w < 0.5) return vec4(0, 0, 0, 1);
    ivec2 hs = F.fogRes.yz;
    int div = F.fogRes.x;
    vec2 c = (vec2(p) + 0.5) / float(div) - 0.5;
    ivec2 i = ivec2(floor(c));
    vec2 f = c - vec2(i);
    if (div == 1) return imageLoad(fogImg, p);
    // bilinear, but taps whose depth (at the pixel volfog.comp marched) differs from ours count less: no fog
    // bleeding across silhouettes, which shows as stair steps at low fog resolution
    float z = imageLoad(gDepth, p).r;
    z = z < 0.0 ? 1e4 : z;
    vec4 sum = vec4(0);
    float wSum = 0.0;
    for (int k = 0; k < 4; k++) {
        ivec2 o = ivec2(k & 1, k >> 1);
        ivec2 q = clamp(i + o, ivec2(0), hs - 1);
        float zq = imageLoad(gDepth, min(q * div + div / 2, ivec2(F.res.xy) - 1)).r;
        zq = zq < 0.0 ? 1e4 : zq;
        vec2 t = mix(1.0 - f, f, vec2(o));
        float w = t.x * t.y / (1e-3 + abs(zq - z) / max(z, 0.1));
        sum += imageLoad(fogImg, q) * w;
        wSum += w;
    }
    return sum / wSum;
}
#endif
