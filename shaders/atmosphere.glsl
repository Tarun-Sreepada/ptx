// Single-scattering Earth atmosphere (Rayleigh + Mie + ozone), Hillaire-style constants, metres.
// Mirrored by sunTransmittance() in main.cpp for the sun colour at the ground.
#ifndef ATMOSPHERE_GLSL
#define ATMOSPHERE_GLSL
const int SKY_W = 256, SKY_H = 128;
const float RG = 6360e3, RT = 6460e3, VIEW_H = 300.0;
const vec3 BR = vec3(5.802, 13.558, 33.1) * 1e-6;
const float HR = 8000.0, HM = 1200.0, BMS = 3.996e-6, BME = 4.40e-6;
const vec3 BO = vec3(0.650, 1.881, 0.085) * 1e-6;

vec3 extinctionAt(float h) {
    return BR * exp(-h / HR) + vec3(BME * exp(-h / HM)) + BO * max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);
}
// distance to exit sphere of radius R from inside (o inside)
float exitSphere(vec3 o, vec3 d, float R) {
    float b = dot(o, d), c = dot(o, o) - R * R;
    return -b + sqrt(max(b * b - c, 0.0));
}
// distance to the ground, or -1
float hitGround(vec3 o, vec3 d) {
    float b = dot(o, d), c = dot(o, o) - RG * RG, disc = b * b - c;
    if (disc < 0.0) return -1.0;
    float t = -b - sqrt(disc);
    return t > 0.0 ? t : -1.0;
}
vec3 transmittanceToSun(vec3 p, vec3 s) {
    if (hitGround(p, s) > 0.0) return vec3(0);
    float L = exitSphere(p, s, RT), ds = L / 8.0;
    vec3 od = vec3(0);
    for (int i = 0; i < 8; i++) od += extinctionAt(length(p + s * ((float(i) + 0.5) * ds)) - RG);
    return exp(-od * ds);
}
// sky texel v <-> elevation, denser near the horizon
float skyV(float el) { float x = sign(el) * sqrt(abs(el) / (PI * 0.5)); return x * 0.5 + 0.5; }
float skyEl(float v) { float x = v * 2.0 - 1.0; return sign(x) * x * x * PI * 0.5; }

vec3 atmosphere(vec3 dir, vec3 sunDir) {
    vec3 o = vec3(0.0, RG + VIEW_H, 0.0);
    float tmax = exitSphere(o, dir, RT), tg = hitGround(o, dir);
    if (tg > 0.0) tmax = tg;
    const int N = 24;
    float ds = tmax / float(N), mu = dot(dir, sunDir);
    float phR = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
    const float g = 0.8;
    float phM = (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * mu, 1.5));
    vec3 T = vec3(1), L = vec3(0);
    for (int i = 0; i < N; i++) {
        vec3 p = o + dir * ((float(i) + 0.5) * ds);
        float h = length(p) - RG;
        float r = exp(-h / HR), m = exp(-h / HM);
        vec3 Ts = transmittanceToSun(p, sunDir);
        vec3 scat = BR * r * phR + BMS * m * phM;
        vec3 ms = (BR * r + BMS * m) * (0.15 / (4.0 * PI)) * max(sunDir.y + 0.1, 0.0); // crude multiple scattering
        L += T * (scat * Ts + ms) * ds;
        T *= exp(-extinctionAt(h) * ds);
    }
    return L;
}
#endif
