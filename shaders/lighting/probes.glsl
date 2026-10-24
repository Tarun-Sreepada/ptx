// Baked interior lighting: per-building irradiance probe grids (bake = 1 volumes in the world files), each baked
// twice, lights on and lights off (probe_bake.comp). Inside a volume the GI pass reads the probes of the current
// switch state instead of tracing paths; direct light and dynamic lights stay path traced.
#ifndef PROBES_GLSL
#define PROBES_GLSL
#ifndef PI
#define PI 3.14159265359
#endif

// F.bake: x volume count, y dynamic lights (orbs, first in the light list), z flags, w lights incl. switched-off ones
#define BAKE_USE 1u  // read baked GI where it's ready

// Per volume 4 vec4: (min.xyz, spacing), (max.xyz, lights on), (dims.xyz, first probe) as uint bits, (ready off, ready on, -, -)
layout(binding = 49, std430) readonly buffer Volumes { vec4 vols[]; };
// Per volume a block of 2 * n probes (n = dims product), lights-off ones then lights-on ones; per probe 4 vec4:
// L1 SH of incoming radiance for r, g, b (L00, L1 . (x, y, z)), then (validity, -, -, -).
#ifdef PROBES_WRITE
layout(binding = 50, std430) buffer Probes { vec4 probes[]; };
#else
layout(binding = 50, std430) readonly buffer Probes { vec4 probes[]; };
#endif

uint volCount() { return F.bake.x; }
bool inVol(uint v, vec3 p) { return all(greaterThanEqual(p, vols[4u * v].xyz)) && all(lessThan(p, vols[4u * v + 1u].xyz)); }
int volumeAt(vec3 p) {
    for (uint v = 0u; v < volCount(); v++) if (inVol(v, p)) return int(v);
    return -1;
}
uint volState(uint v) { return vols[4u * v + 1u].w > 0.5 ? 1u : 0u; }
bool volReady(uint v, uint st) { return vols[4u * v + 3u][st] > 0.5; }

// Bake override: while baking volume bakeVol for state bakeState, that volume is treated as being in that state.
int bakeVol = -1;
uint bakeState = 1u;
// Emitters (light panels) inside a building whose lights are switched off don't glow.
bool emitterOff(vec3 p) {
    int v = volumeAt(p);
    if (v < 0) return false;
    return (v == bakeVol ? bakeState : volState(uint(v))) == 0u;
}

uint probeBase(uvec4 d, uint st, uint i) { return (d.w * 2u + st * d.x * d.y * d.z + i) * 4u; }

// L1 SH irradiance (Ramamoorthi & Hanrahan): E(n) = pi Y00 L00 + 2pi/3 Y1 (L1 . n)
vec3 shIrradiance(uint base, vec3 n) {
    vec4 r = probes[base], g = probes[base + 1u], b = probes[base + 2u];
    const float c0 = PI * 0.282095, c1 = 2.0 * PI / 3.0 * 0.488603;
    vec3 E = vec3(c0 * r.x + c1 * dot(r.yzw, n), c0 * g.x + c1 * dot(g.yzw, n), c0 * b.x + c1 * dot(b.yzw, n));
    return max(E, vec3(0));
}

// Baked indirect light at (x, n), demodulated like indirectImg (E / pi). False outside ready volumes or where no
// valid probe sees the point; the caller then path traces as usual.
bool bakedIndirect(vec3 x, vec3 n, out vec3 L) {
    L = vec3(0);
    if ((F.bake.z & BAKE_USE) == 0u || volCount() == 0u) return false;
    vec3 xb = x + n * 0.06;
    int vi = volumeAt(xb);
    if (vi < 0) return false;
    uint v = uint(vi), st = volState(v);
    if (!volReady(v, st)) return false;
    vec4 a = vols[4u * v];
    uvec4 d = floatBitsToUint(vols[4u * v + 2u]);
    float sp = a.w;
    vec3 g = (xb - a.xyz) / sp - 0.5;
    ivec3 g0 = ivec3(floor(g));
    vec3 f = g - vec3(g0);
    float wSum = 0.0;
    vec3 E = vec3(0);
    for (int k = 0; k < 8; k++) {
        ivec3 o = ivec3(k & 1, (k >> 1) & 1, k >> 2);
        ivec3 c = clamp(g0 + o, ivec3(0), ivec3(d.xyz) - 1);
        vec3 t = mix(1.0 - f, f, vec3(o));
        float w = t.x * t.y * t.z;
        uint base = probeBase(d, st, uint(c.x) + d.x * (uint(c.y) + d.y * uint(c.z)));
        w *= probes[base + 3u].x;  // probes stuck inside walls don't count
        // probes behind the surface are on the other side of a wall / floor: the main source of leaks
        vec3 pp = a.xyz + (vec3(c) + 0.5) * sp;
        vec3 dv = pp - x;
        w *= smoothstep(-0.1, 0.3, dot(dv, n) * inversesqrt(max(dot(dv, dv), 1e-6)));
        if (w <= 1e-5) continue;
        E += shIrradiance(base, n) * w;
        wSum += w;
    }
    if (wSum < 1e-3) return false;
    L = E / (wSum * PI);
    return true;
}
#endif
