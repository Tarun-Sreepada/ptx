// Cirrus: thin ice cloud 7-12 km up. What makes it read as cirrus rather than smeared noise:
//  - fibrous streaks, long along the wind shear and fine across it (fibratus), bent into arcs and hooks by a slow
//    large-scale warp (uncinus), with fine striations running along each streak; optional tufts (floccus)
//  - smooth gradient noise (no lattice creases), and octaves that fade out as they shrink below a pixel, so streaks
//    stay crisp overhead and melt into a soft veil toward the horizon instead of shimmering
//  - a thin layer, not a plane: three slices a few hundred metres apart give it parallax and depth at low angles
//  - ice optics: a strong forward lobe, weak backscatter and a faint 22-degree halo; lit by the sun at its own
//    altitude, so it keeps glowing orange-pink for minutes after the ground has gone into shadow at sunset
// Settings: F.clouds4.z amount, F.clouds4.w altitude (m), F.clouds5 = (strand length, fineness, puffiness).
#ifndef SKY_CIRRUS
#define SKY_CIRRUS
#include "core/frame.glsl"
#include "core/util.glsl"
#include "terrain/noise.glsl"
#include "sky/atmosphere.glsl"

const float CIRRUS_THICKNESS = 500.0;   // m between the lowest and highest slice
const float CIRRUS_TAU = 0.55;          // vertical optical depth of the densest strands
const float CIRRUS_JET = 2.5;           // drift speed relative to the low clouds (jet stream)
const float CIRRUS_BAND_SCALE = 22000.0, CIRRUS_WARP_SCALE = 9000.0, CIRRUS_TUFT_SCALE = 900.0;  // m

vec2 cirrusGrad(ivec2 i) {
    uint h = hashu(uint(i.x) * 0x8da6b343u ^ hashu(uint(i.y) * 0xd8163841u));
    return vec2(float(h & 0xFFFFu), float(h >> 16)) / 32767.5 - 1.0;
}
// Perlin gradient noise, quintic fade: no lattice-aligned creases (value noise ridges showed them), ~[-1, 1]
float cirrusNoise(vec2 p) {
    ivec2 i = ivec2(floor(p));
    vec2 f = fract(p), u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float a = dot(cirrusGrad(i), f), b = dot(cirrusGrad(i + ivec2(1, 0)), f - vec2(1, 0));
    float c = dot(cirrusGrad(i + ivec2(0, 1)), f - vec2(0, 1)), d = dot(cirrusGrad(i + ivec2(1, 1)), f - vec2(1, 1));
    return 1.6 * mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
// fbm whose octaves fade out once their period drops below ~2 pixel footprints (fp, in p's units): an octave the
// pixel can't resolve contributes its mean (0) instead of aliasing
float cirrusFbm(vec2 p, int octaves, float fp, float gain) {
    float s = 0.0, a = 0.5, period = 1.0, norm = 0.0;
    const mat2 R = mat2(0.80, -0.60, 0.60, 0.80);
    for (int i = 0; i < octaves; i++) {
        s += a * smoothstep(1.5 * fp, 4.0 * fp, period) * cirrusNoise(p);
        norm += a;
        a *= gain;
        period *= 0.5;
        p = R * p * 2.0 + vec2(13.7, 7.1);
    }
    return s / norm;
}

vec2 cirrusWindDir() { return normalize(vec2(F.wind.x, F.wind.y) + vec2(1e-4, 0.0)); }

// density 0..~1.2 of the layer at world xz; fp = pixel footprint there (m)
float cirrusDensity(vec2 xz, float fp) {
    float cov = F.clouds4.z;
    if (cov <= 0.0) return 0.0;
    vec2 w = cirrusWindDir(), wn = vec2(-w.y, w.x);
    vec2 c = xz + vec2(F.wind.x, F.wind.y) * (F.camPos.w * 12.0 * F.clouds.z * CIRRUS_JET);
    // where: broad bands across the sky, stretched along the wind
    vec2 qm = vec2(dot(c, w) * 0.35, dot(c, wn)) / CIRRUS_BAND_SCALE;
    float mask = cirrusFbm(qm, 4, fp / CIRRUS_BAND_SCALE, 0.5) * 0.5 + 0.5;
    mask = smoothstep(1.0 - cov * 0.85 - 0.05, 1.0 - cov * 0.85 + 0.3, mask);
    if (mask <= 0.001) return 0.0;
    // the streak frame, bent by a slow warp so streaks arc and hook
    vec2 warp = vec2(cirrusNoise(c / CIRRUS_WARP_SCALE + 3.1), cirrusNoise(c / CIRRUS_WARP_SCALE + 7.7)) * 2500.0 +
                vec2(cirrusNoise(c / 2600.0 + 1.3), cirrusNoise(c / 2600.0 + 5.9)) * 500.0;
    vec2 cw = c + warp;
    float stretch = 7.0 * max(F.clouds5.x, 0.2);       // along / across ratio of a strand
    float across = mix(1400.0, 450.0, F.clouds5.y);     // strand width scale (m)
    vec2 sf = vec2(dot(cw, w) / stretch, dot(cw, wn)) / across;
    float st = cirrusFbm(sf, 5, fp / across, 0.55) * 0.5 + 0.5;
    float strands = smoothstep(0.45, 0.85, st);
    // fibres: fine striations running along each strand
    float fscale = across * 0.11;
    vec2 ff = vec2(dot(cw, w) / (stretch * 5.0), dot(cw, wn)) / fscale;
    float fib = cirrusFbm(ff, 3, fp / fscale, 0.6) * 0.5 + 0.5;
    float dens = strands * mix(0.45, 1.2, fib);
    // tufts (floccus): rounded heads where the strands start
    float puff = F.clouds5.z;
    if (puff > 0.0) {
        float cells = cirrusFbm(cw / CIRRUS_TUFT_SCALE, 3, fp / CIRRUS_TUFT_SCALE, 0.5) * 0.5 + 0.5;
        dens = max(dens, smoothstep(0.58, 0.82, cells) * smoothstep(0.35, 0.6, st) * puff * 1.1);
    }
    // frayed, soft edges
    dens *= smoothstep(0.0, 0.3, dens + 0.15 * (fib - 0.5));
    return dens * mask;
}

// vertical optical depth for the cloud shadow on the ground (no fine detail: the shadow is soft anyway)
float cirrusShadowTau(vec2 xz) { return cirrusDensity(xz, 300.0) * CIRRUS_TAU * (0.5 + 0.5 * F.clouds4.z); }

float hgPhase(float mu, float g) { return (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * mu, 1.5)); }

// In-scattered light and transmittance of the cirrus layer along a view ray (pixelAngle: radians per pixel).
vec3 cirrusMarch(vec3 ro, vec3 rd, float pixelAngle, vec3 amb, out float T) {
    T = 1.0;
    if (F.clouds4.z <= 0.0 || abs(rd.y) < 1e-3) return vec3(0);
    float alt = F.clouds4.w;
    // sun (or moon) light at the cirrus altitude: the sun is still up there a few degrees after it set below
    vec3 key = F.night1.w > 0.5 ? F.sunColor.rgb : transmittanceToSun(vec3(0.0, RG + alt, 0.0), F.sun2.xyz) * F.sun2.w;
    vec3 kd = F.night1.w > 0.5 ? F.sunDir.xyz : F.sun2.xyz;
    float mu = dot(rd, kd);
    float halo = exp(-sq((acos(clamp(mu, -1.0, 1.0)) - 0.384) / 0.014)) * 0.12;  // 22-degree halo of hexagonal plates
    float phase = mix(hgPhase(mu, -0.25), hgPhase(mu, 0.85), 0.8) + halo / (4.0 * PI);
    vec3 Ls = key * phase * 2.6 + amb / (4.0 * PI) * 3.2;  // thin ice: single scattering x a little multiple scattering
    vec3 L = vec3(0);
    for (int k = 0; k < 3; k++) {
        float h = alt + (float(k) - 1.0) * 0.5 * CIRRUS_THICKNESS;
        float t = (h - ro.y) / rd.y;
        if (t <= 0.0 || t > 1.5e5) continue;
        float fp = t * pixelAngle / max(abs(rd.y), 0.05);  // footprint, stretched along the view near the horizon
        float tau = cirrusDensity((ro + rd * t).xz, fp) * CIRRUS_TAU * (0.5 + 0.5 * F.clouds4.z) / 3.0 / max(abs(rd.y), 0.06);
        if (tau <= 0.0) continue;
        float a = exp(-tau);
        L += T * Ls * (1.0 - a);
        T *= a;
    }
    return L;
}
#endif
