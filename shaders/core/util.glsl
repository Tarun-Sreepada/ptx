// Small helpers: luminance, PCG random numbers, orthonormal bases, hemisphere / cone sampling.
#ifndef CORE_UTIL
#define CORE_UTIL
#include "core/frame.glsl"

#define PI 3.14159265359
// ---------------------------------------------------------------- utils
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

uint pcg(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float rnd(inout uint s) { return float(pcg(s)) * (1.0 / 4294967296.0); }
uint seedFor(ivec2 p, uint salt) {
    uint s = uint(p.x) * 1973u + uint(p.y) * 9277u + F.res.z * 26699u + salt * 104729u;
    pcg(s);
    return s;
}


void onb(vec3 n, out vec3 t, out vec3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float bb = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * bb, -s * n.x);
    b = vec3(bb, s + n.y * n.y * a, -n.y);
}
vec3 cosineSampleU(vec3 n, vec2 u) {
    float u1 = u.x, u2 = u.y;
    float r = sqrt(u1), phi = 2.0 * PI * u2;
    vec3 t, b; onb(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}
vec3 cosineSample(vec3 n, inout uint s) {
    float u1 = rnd(s), u2 = rnd(s);
    float r = sqrt(u1), phi = 2.0 * PI * u2;
    vec3 t, b; onb(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}
vec3 sampleConeU(vec3 axis, float cosMax, vec2 u) {
    float ct = mix(1.0, cosMax, u.x);
    float st = sqrt(max(0.0, 1.0 - ct * ct));
    float phi = 2.0 * PI * u.y;
    vec3 t, b; onb(axis, t, b);
    return normalize(t * (cos(phi) * st) + b * (sin(phi) * st) + axis * ct);
}
vec3 sampleCone(vec3 axis, float cosMax, inout uint s) {
    float ct = mix(1.0, cosMax, rnd(s));
    float st = sqrt(max(0.0, 1.0 - ct * ct));
    float phi = 2.0 * PI * rnd(s);
    vec3 t, b; onb(axis, t, b);
    return normalize(t * (cos(phi) * st) + b * (sin(phi) * st) + axis * ct);
}
#endif
