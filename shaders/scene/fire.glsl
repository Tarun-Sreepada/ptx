// Fires (sim/fire.cpp): flames and smoke as volumes, marched along the view ray through each fire's bounding
// cylinder. Flames: turbulent tongues rising off the fuel bed, hottest (yellow-white) low in the core, cooling to
// orange and deep red at their ragged tips, flickering as the noise rises through them; soot makes them a little
// opaque. Smoke: a grey column above, widening and thinning as it climbs and leaning with the wind, lit by the sky
// and by the glow below. F.fires[2i] = base xyz, radius; [2i+1] = heat, seed, flame height, smoke height.
#ifndef SCENE_FIRE
#define SCENE_FIRE
#include "core/frame.glsl"

const int FIRE_STEPS = 28;
const float FLAME_RADIANCE = 6.0;  // hot core radiance (scene units)
const float FLAME_SIGMA = 2.5;     // 1/m: soot in the flames
const float SMOKE_SIGMA = 0.9;     // 1/m: densest smoke just above the flames
const float SMOKE_ALBEDO = 0.3;
const float SMOKE_SPREAD = 0.22;   // smoke radius growth per metre of rise
const float SMOKE_LEAN = 0.45;     // m of downwind lean per metre of rise, per unit of wind
const float FLAME_RISE = 2.2;      // m/s the flame turbulence climbs
const float SMOKE_RISE = 1.1;      // m/s

float fireHash(vec3 i) {
    uvec3 c = uvec3(ivec3(i));
    return float(hashu(hashu(hashu(c.x) ^ c.y * 0x9e3779b9u) ^ c.z * 0x85ebca6bu) & 0xFFFFFFu) / 16777216.0;
}
float fireNoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = mix(fireHash(i), fireHash(i + vec3(1, 0, 0)), f.x), b = mix(fireHash(i + vec3(0, 1, 0)), fireHash(i + vec3(1, 1, 0)), f.x);
    float c = mix(fireHash(i + vec3(0, 0, 1)), fireHash(i + vec3(1, 0, 1)), f.x), d = mix(fireHash(i + vec3(0, 1, 1)), fireHash(i + vec3(1, 1, 1)), f.x);
    return mix(mix(a, b, f.y), mix(c, d, f.y), f.z);
}
float fireFbm(vec3 p, int octaves) {
    float s = 0.0, a = 0.5, n = 0.0;
    for (int i = 0; i < octaves; i++) { s += a * fireNoise(p); n += a; a *= 0.5; p = p * 2.03 + 17.1; }
    return s / n;
}

// flame colour from 0 (cool tip) to 1 (hot core): deep red, orange, yellow, near white (blackbody-like)
vec3 flameColor(float t) {
    t = clamp(t, 0.0, 1.0);
    return t < 0.33 ? mix(vec3(0.35, 0.03, 0.0), vec3(1.0, 0.25, 0.02), t / 0.33)
         : t < 0.66 ? mix(vec3(1.0, 0.25, 0.02), vec3(1.0, 0.62, 0.12), (t - 0.33) / 0.33)
                    : mix(vec3(1.0, 0.62, 0.12), vec3(1.0, 0.9, 0.6), (t - 0.66) / 0.34);
}

// Light the fires add along the ray o + t d (t < tMax, the surface behind) and how much of what's behind gets
// through: rgb = in-scattered and emitted light, a = transmittance. jit: 0..1 per pixel (step offset).
vec4 fireVolumes(vec3 o, vec3 d, float tMax, float jit, vec3 ambient) {
    uint n = uint(F.fireInfo.x);
    vec3 L = vec3(0);
    float T = 1.0;
    vec2 wind = F.wind.xy * F.wind.z;
    float time = F.camPos.w;
    for (uint i = 0u; i < n; i++) {
        vec4 a = F.fires[2u * i], b = F.fires[2u * i + 1u];
        vec3 c = a.xyz;
        float R = a.w, heat = b.x, seed = b.y, Hf = b.z, top = b.z + b.w;
        if (heat <= 0.0) continue;
        vec2 lean = wind * SMOKE_LEAN * top;  // the column's top, downwind of the base
        // ray against the vertical cylinder around the leaning column, base to top
        float Rb = R + SMOKE_SPREAD * top + 0.5 * length(lean);
        vec2 oc = o.xz - (c.xz + 0.5 * lean);
        float A = dot(d.xz, d.xz), B = dot(oc, d.xz), C = dot(oc, oc) - Rb * Rb;
        float t0 = 0.0, t1 = tMax;
        if (A > 1e-8) {
            float disc = B * B - A * C;
            if (disc <= 0.0) continue;
            float s = sqrt(disc);
            t0 = (-B - s) / A; t1 = (-B + s) / A;
        } else if (C > 0.0) continue;
        float y0 = c.y - 0.3, y1 = c.y + top;
        if (abs(d.y) > 1e-6) {
            float ta = (y0 - o.y) / d.y, tb = (y1 - o.y) / d.y;
            t0 = max(t0, min(ta, tb)); t1 = min(t1, max(ta, tb));
        } else if (o.y < y0 || o.y > y1) continue;
        t0 = max(t0, 0.0); t1 = min(t1, tMax);
        if (t1 <= t0) continue;
        float dt = (t1 - t0) / float(FIRE_STEPS);
        vec3 glow = flameColor(0.6) * FLAME_RADIANCE * heat;
        for (int k = 0; k < FIRE_STEPS; k++) {
            vec3 q = o + d * (t0 + (float(k) + jit) * dt);
            float y = q.y - c.y, hN = clamp(y / top, 0.0, 1.0);
            float r = length(q.xz - (c.xz + lean * hN * hN));
            float sigF = 0.0, temp = 0.0, sigS = 0.0;
            if (y > -0.3 && y < Hf * 1.3) {  // flames: narrowing upward, ragged tops, rising turbulence
                vec3 nq = vec3(q.x, q.y - time * FLAME_RISE, q.z) * (2.0 / max(R, 0.3)) + seed;
                float nz = fireFbm(nq, 3);
                float u = y / Hf;
                float width = R * (1.0 - 0.55 * clamp(u, 0.0, 1.0)) * (0.75 + 0.5 * nz);
                float shape = 1.0 - smoothstep(width * 0.6, width, r);
                float tip = 1.0 - smoothstep(0.55, 1.15, u + (nz - 0.5) * 0.7);
                sigF = shape * tip * smoothstep(-0.3, 0.05, y) * FLAME_SIGMA * heat;
                temp = (1.0 - clamp(u, 0.0, 1.0)) * 0.8 + nz * 0.4 - r / max(R, 0.1) * 0.3;
            }
            if (y > Hf * 0.5) {  // smoke: spreading and thinning as it climbs, fading out at the top
                float h = y - Hf * 0.5, rs = R * 0.7 + SMOKE_SPREAD * h;
                vec3 sq = vec3(q.x - wind.x * time * 0.6, q.y - time * SMOKE_RISE, q.z - wind.y * time * 0.6) * (0.8 / max(R, 0.5));
                float nz = fireFbm(sq + seed * 1.7, 2);
                sigS = exp(-r * r / (rs * rs)) * smoothstep(0.0, Hf * 0.5, h) * (1.0 - smoothstep(0.7, 1.0, hN)) * (0.4 + 1.2 * nz) *
                       (R * 0.7 / rs) * SMOKE_SIGMA * heat;
            }
            float sig = sigF + sigS;
            if (sig <= 1e-5) continue;
            float tr = exp(-sig * dt);
            vec3 emit = flameColor(temp) * FLAME_RADIANCE * heat;
            vec3 smokeLit = (ambient + glow * exp(-max(y - Hf, 0.0) / max(Hf, 0.5)) * 0.15) * SMOKE_ALBEDO;
            L += T * (emit * sigF + smokeLit * sigS) / sig * (1.0 - tr);
            T *= tr;
            if (T < 0.01) return vec4(L, T);
        }
    }
    return vec4(L, T);
}
#endif
