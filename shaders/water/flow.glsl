// River flow field and surface noise (flow-mapped normals, caustic pattern).
#ifndef WATER_FLOW
#define WATER_FLOW
#include "core/frame.glsl"
#include "water/level.glsl"

const float WATER_PERIOD = 2.0;  // flow-map reset period (s)

struct Flow { vec2 v; float speed, shear; vec4 r; };  // r = riverInfo
Flow riverFlow(vec2 p) {
    Flow f;
    float dz;
    f.r = riverInfoS(p, dz);  // dz: the direction of the channel this point is in
    float drop;
    float divK = F.dynInfo.z > 0u ? riverDiversion(p.x, drop) : 1.0;
    vec2 dir = vec2(1.0, dz) * inversesqrt(1.0 + dz * dz);
    float c = min(abs(f.r.x) / f.r.y, 1.0);
    // more water after rain runs faster (the swell over the channel depth, roughly)
    float U = F.water.x * 1.3 * (F.river.x * F.river.y) / (f.r.y * f.r.z) * divK * (1.0 + 1.6 * F.sim2.w / max(F.river.y, 0.3));
    float k = max(1.0 - c * c, 0.02);
    f.speed = U * pow(k, 0.35) * step(c, 0.999);
    f.v = dir * f.speed;
    f.shear = min(U * 0.7 * c * pow(k, -0.65) / f.r.y, 4.0);  // |du/dn|
    return f;
}

float whash(vec2 i) { return float(hashu(uint(int(i.x)) * 7919u + uint(int(i.y)) * 104729u)) / 4294967295.0; }
// value noise with analytic gradient: x = value, yz = d/dp
vec3 vnoiseD(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f), du = 6.0 * f * (1.0 - f);
    float a = whash(i), b = whash(i + vec2(1, 0)), c = whash(i + vec2(0, 1)), d = whash(i + vec2(1, 1));
    float k1 = b - a, k2 = c - a, k3 = a - b - c + d;
    return vec3(a + k1 * u.x + k2 * u.y + k3 * u.x * u.y, du * (vec2(k1, k2) + k3 * u.yx));
}
vec2 waveGrad(vec2 p) {
    return 0.5 * 1.1 * vnoiseD(p * 1.1).yz + 0.25 * 2.7 * vnoiseD(p * 2.7 + 5.3).yz + 0.12 * 6.3 * vnoiseD(p * 6.3 - 2.1).yz;
}
float caustics(vec2 p) {
    float a = 1.0 - abs(vnoise(p) * 2.0 - 1.0), b = 1.0 - abs(vnoise(p * 1.9 + 4.0) * 2.0 - 1.0);
    return pow(a, 6.0) + pow(b, 8.0) * 0.6;
}
float foamNoise(vec2 p) { return vnoise(p * 2.3) * 0.6 + vnoise(p * 7.1) * 0.4; }

// Foam trail length (water2.x): wakes fade out smoothly before it instead of being cut off
float wakeFade(float a) { return 1.0 - smoothstep(F.water2.x * 0.3, F.water2.x, a); }
// Wake strength along the trail: decays over a good part of the trail length (bubbles take a while to pop),
// not a fixed few metres, so the white water thins out gradually all the way down
float wakeTail(float a, float R) { return smoothstep(-0.3 * R, R, a) * exp(-max(a, 0.0) / (0.3 * F.water2.x + 2.0 * R + 1.0)) * wakeFade(a); }
#endif
