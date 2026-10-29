// Direct sun / moon light: shadow rays, cloud shadows, underwater caustics, the BRDF at primary hits.
#ifndef LIGHTING_SUN
#define LIGHTING_SUN
#include "core/frame.glsl"
#include "lighting/trace.glsl"
#include "sky/clouds.glsl"

// Diffuse cosine term; grass blades also transmit some light from behind.
// mat: material id with packed translucency (brdf.glsl matT).
float lambert(vec3 n, vec3 l, uint mat, out float side) {
    float c = dot(n, l);
    side = c >= 0.0 ? 1.0 : -1.0;
    if (c >= 0.0) return c;
    return -c * translOf(mat);
}

const float EPS = 0.004;

// Sunlight reaching x under the river's surface: absorbed along the refracted path, focused by the waves into a
// moving caustic net (two flow-advected layers half a period apart, like the surface detail in water.glsl).
float causticNet(vec2 p) {
    float a = 1.0 - abs(vnoise(p) * 2.0 - 1.0), b = 1.0 - abs(vnoise(p * 1.9 + 4.0) * 2.0 - 1.0);
    return pow(a, 6.0) + pow(b, 8.0) * 0.6;
}
vec3 underwaterSun(vec3 x) {
    float depth = waterLevelAt(x.xz) - x.y;
    if (depth <= 0.0) return vec3(1);
    vec3 rd = refract(-F.sunDir.xyz, vec3(0, 1, 0), 1.0 / 1.33);
    float path = depth / max(-rd.y, 0.2);
    vec2 e = x.xz - rd.xz * path;  // where the sunlight entered the water
    float dz;
    vec4 ri = riverInfoS(e, dz);
    // the river's current carries the net downstream; a still pond's only shimmers in place (ripples from the wind)
    vec2 flow = abs(ri.x) > ri.y + 1.0 ? vec2(0.11, 0.07) * (0.3 + F.wind.z)
                                       : vec2(1.0, dz) * inversesqrt(1.0 + dz * dz) * F.water.x * 0.9;
    float T = F.camPos.w, ph0 = fract(T / 2.0), ph1 = fract(T / 2.0 + 0.5), blend = abs(1.0 - 2.0 * ph0);
    vec2 q0 = e - flow * ph0 * 2.0, q1 = e - flow * ph1 * 2.0 + vec2(0.43, 0.71);
    float c = mix(causticNet(q0 * 1.7), causticNet(q1 * 1.7), blend);
    // sharper in shallow water, smeared out with depth; mean stays ~1 (light is redistributed, not added)
    float k = F.weather.w * exp(-0.5 * depth);
    float focus = 1.0 + k * (c * 3.2 - 0.7);
    vec3 sigma = F.waterCol.w * vec3(1.0, 0.42, 0.32) / max(F.water.y, 0.05);
    return exp(-sigma * path) * max(focus, 0.0);
}

// Demodulated sun contribution (Lo / albedo) with a soft cone-sampled shadow ray; u: point on the sun disc.
vec3 sunDirect(vec3 x, vec3 n, uint mat, vec2 u) {
    if (F.sunColor.r + F.sunColor.g + F.sunColor.b <= 0.0) return vec3(0);
    vec3 l = sampleConeU(F.sunDir.xyz, F.sunDir.w, u);
    float side;
    float c = lambert(n, l, mat, side);
    if (c <= 0.0) return vec3(0);
    if (!visible(x + n * (EPS * side), l, 1e4)) return vec3(0);  // path vertices: approximate in craters
    return F.sunColor.rgb * cloudShadow(x) * underwaterSun(x) * c / PI * (side < 0.0 ? TRANSLUCENT_TINT : vec3(1));
}

vec3 sunDirect(vec3 x, vec3 n, uint mat, inout uint s) {
    float u0 = rnd(s);
    return sunDirect(x, n, mat, vec2(u0, rnd(s)));
}

// GGX specular lobe (the waxy sheen on grass, dull highlight on rock).
float ggxSpec(vec3 n, vec3 v, vec3 l, float rough) {
    float ndl = dot(n, l), ndv = max(dot(n, v), 1e-3);
    if (ndl <= 0.0) return 0.0;
    vec3 h = normalize(l + v);
    float a2 = rough * rough * rough * rough;
    float nh = max(dot(n, h), 0.0);
    float dd = nh * nh * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * dd * dd);
    float k = rough * rough * 0.5;
    float G = ndv / (ndv * (1.0 - k) + k) * ndl / (ndl * (1.0 - k) + k);
    float Fr = 0.04 + 0.96 * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    return D * G * Fr / (4.0 * ndv);   // already multiplied by n.l
}

// Sun at the primary hit through the full BRDF, demodulated (divided by the pixel's albedo A).
// u: 2D sample for the point on the sun disc (blue noise); kd: share of the diffuse lobe (0 where subsurface
// scattering replaces it); ao: material occlusion for micro-shadowing (1 = none).
vec3 sunDirectPrimary(vec3 x, Surf sf, vec3 v, vec3 A, vec2 u, float kd, float ao) {
    if (F.sunColor.r + F.sunColor.g + F.sunColor.b <= 0.0) return vec3(0);
    vec3 l = sampleConeU(F.sunDir.xyz, F.sunDir.w, u);
    vec3 fd;
    vec3 f = evalBRDFSplit(sf, v, l, fd);
    f = (f - fd + fd * kd) * microShadow(ao, dot(sf.n, l));
    if (f.r + f.g + f.b <= 0.0) return vec3(0);
    float side = dot(sf.n, l) >= 0.0 ? 1.0 : -1.0;
    if (!visibleSun(x + sf.n * (EPS * side), l)) return vec3(0);
    return F.sunColor.rgb * cloudShadow(x) * underwaterSun(x) * f / A;
}
#endif
