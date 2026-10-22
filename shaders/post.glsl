// Post-processing at output resolution: the HDR scene (composite.comp / DLSS-RR, linear, scene-referred) ->
// exposure, bloom, sharpening, tonemapping and the display encoding.
#ifndef POST_GLSL
#define POST_GLSL
layout(binding = 83, std430) buffer Exposure {
    uint hist[128];  // log2-luminance histogram of the frame (exposure.comp)
    vec4 expo;       // x exposure multiplier, y metered log2 luminance (EV), z target log2 exposure, w initialised
};
layout(binding = 84) uniform sampler2D hdrTex;     // the HDR scene, bilinear (X_HDR)
layout(binding = 85) uniform sampler2D bloomTex;   // bloom chain, all mips
layout(binding = 86, rgba16f) uniform image2D bloomMip[8];
layout(binding = 59, rgba16f) uniform image2D hdrOut;  // HDR display output (PQ or scRGB), output resolution (X_HDROUT)

const float EV_MIN = -16.0, EV_MAX = 12.0;  // histogram range (log2 luminance)
const int BLOOM_MIPS = 7;

float exposureNow() { return F.post0.y > 0.5 ? expo.x : F.params.x; }

// ---------------------------------------------------------------- tonemappers (in: exposed scene-linear Rec.709)
// AgX (Troy Sobotka), polynomial fit of the default contrast curve (Benjamin Wrensch's minimal version).
vec3 agxContrast(vec3 x) {
    vec3 x2 = x * x, x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
vec3 agx(vec3 c, bool punchy) {
    const mat3 inset = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051, 0.0784335999999992, 0.878468636469772,
                            0.0784336, 0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const mat3 outset = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438, -0.0980208811401368, 1.15190312990417,
                             -0.0980434501171241, -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    const float lo = -12.47393, hi = 4.026069;
    vec3 v = inset * max(c, vec3(1e-10));
    v = (clamp(log2(v), lo, hi) - lo) / (hi - lo);
    v = agxContrast(v);
    if (punchy) {  // the "punchy" look: more contrast and saturation
        v = pow(max(v, vec3(0)), vec3(1.35));
        float l = dot(v, vec3(0.2126, 0.7152, 0.0722));
        v = l + 1.4 * (v - l);
    }
    v = outset * v;
    return pow(clamp(v, 0.0, 1.0), vec3(2.2));  // display-linear Rec.709
}
vec3 acesFit(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
vec3 tonemapSDR(vec3 c) {
    uint t = uint(F.post2.z + 0.5);
    if (t == 0u) return agx(c, false);
    if (t == 1u) return agx(c, true);
    if (t == 2u) return acesFit(c);
    return clamp(c, 0.0, 1.0);
}
// HDR displays: hue-preserving shoulder toward the display peak (in units of paper white); mid-tones stay linear.
vec3 tonemapHDR(vec3 c, float peak) {
    float y = dot(c, vec3(0.2126, 0.7152, 0.0722));
    if (y <= 0.0) return vec3(0);
    float yo = y / (1.0 + y / peak);
    vec3 o = c * (yo / y);
    float m = max(o.r, max(o.g, o.b));
    if (m > peak) o = mix(vec3(yo), o, (peak - yo) / max(m - yo, 1e-4));  // desaturate what would clip
    return o;
}
float srgbOETF(float x) { return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0 / 2.4) - 0.055; }
vec3 srgbOETF(vec3 c) { return vec3(srgbOETF(c.r), srgbOETF(c.g), srgbOETF(c.b)); }
vec3 pqOETF(vec3 nits) {  // SMPTE ST 2084
    vec3 y = clamp(nits / 10000.0, 0.0, 1.0);
    const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    vec3 ym = pow(y, vec3(m1));
    return pow((c1 + c2 * ym) / (1.0 + c3 * ym), vec3(m2));
}
const mat3 REC709_TO_2020 = mat3(0.6274, 0.0691, 0.0164, 0.3293, 0.9195, 0.0880, 0.0433, 0.0114, 0.8956);
#endif
