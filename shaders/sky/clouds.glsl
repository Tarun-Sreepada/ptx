// Volumetric cloud layer: a flat slab of advected fbm noise above the world (metres, y up).
// cloudMarch() draws it in the sky (and over the sun disc), cloudShadow() dims the sun on the ground.
#ifndef CLOUDS_GLSL
#define CLOUDS_GLSL
#include "sky/cirrus.glsl"

float hash3(uvec3 c) { return float(hashu(c.x * 73856093u ^ hashu(c.y * 19349663u ^ hashu(c.z * 83492791u)))) / 4294967295.0; }
float vnoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    uvec3 c = uvec3(ivec3(i));
    const uvec2 o = uvec2(0u, 1u);
    return mix(mix(mix(hash3(c + o.xxx), hash3(c + o.yxx), f.x), mix(hash3(c + o.xyx), hash3(c + o.yyx), f.x), f.y),
               mix(mix(hash3(c + o.xxy), hash3(c + o.yxy), f.x), mix(hash3(c + o.xyy), hash3(c + o.yyy), f.x), f.y), f.z);
}

bool cloudsOn() { return F.clouds.w > 0.5 && (F.clouds.x > 0.0 || F.clouds3.x > 0.0 || F.clouds4.x > 0.0 || F.clouds4.z > 0.0); }

// x scroll applied to every sample: clouds drift with the wind
vec3 cloudWind() { return vec3(F.wind.x, 0.0, F.wind.y) * (F.camPos.w * 12.0 * F.clouds.z); }

// Three layers, each its own genus:
//  - convective layer (clouds2 base / thickness): cumulus; where a sparse low-frequency storm mask is on
//    (clouds4.x) its top rises to clouds4.y -- cumulonimbus towers, dense and dark-based, spreading out
//    into a flat anvil under the tropopause
//  - low deck (clouds3): flat, wide stratus, broken into cells (stratocumulus) by clouds3.y
//  - cirrus (clouds4.zw, clouds5): thin ice streaks, sky/cirrus.glsl
float cbMask(vec2 xz) {
    float n = vnoise3(vec3(xz * 0.00007, 0.5)) * 0.7 + vnoise3(vec3(xz * 0.00019, 2.5)) * 0.3;
    return smoothstep(1.0 - 0.45 * F.clouds4.x, 1.05 - 0.45 * F.clouds4.x, n) * step(0.001, F.clouds4.x);
}
float convTop() { return max(F.clouds2.x + F.clouds2.y, F.clouds4.x > 0.0 ? F.clouds4.y : 0.0); }

float cumulusDensity(vec3 p, bool detail) {
    float b = F.clouds2.x, top0 = b + F.clouds2.y;
    if (p.y <= b || p.y >= convTop()) return 0.0;
    vec3 q = (p + cloudWind()) * 0.00045;
    float hA = (p.y - b) / max(F.clouds4.y - b, 1.0);  // height through the full storm column
    // anvil: above ~80% of the column the storm spreads (its mask is read over a wider footprint and further downwind)
    float anvil = smoothstep(0.72, 0.88, hA) * (1.0 - smoothstep(0.97, 1.0, hA));
    vec2 sh = vec2(F.wind.x, F.wind.y) * 1500.0 * anvil;
    float cb = max(cbMask(p.xz + cloudWind().xz), cbMask((p.xz + cloudWind().xz - sh) * mix(1.0, 0.55, anvil)) * anvil);
    float top = mix(top0, F.clouds4.y, cb);
    float h = (p.y - b) / (top - b);
    if (h >= 1.0) return 0.0;
    float base = vnoise3(q) * 0.55 + vnoise3(q * 2.1 + 3.7) * 0.3 + vnoise3(q * 4.3 + 9.1) * 0.15;
    float cov = max(F.clouds.x, cb * 0.9);
    float shape = smoothstep(0.0, 0.07, h) * smoothstep(1.0, 0.35, h);  // flat-ish bottoms, rounded tops
    // towers: column narrows toward mid height, flat-topped anvil keeps full width
    shape = mix(shape, smoothstep(0.0, 0.03, h) * mix(1.0 - 0.35 * smoothstep(0.2, 0.7, h), 1.0, anvil) * smoothstep(1.0, 0.92, h), cb);
    float d = clamp((base - (1.0 - cov)) / max(cov, 0.05), 0.0, 1.0) * shape;
    if (d <= 0.0) return 0.0;
    // anvils are fibrous ice, towers billowy: detail erodes accordingly
    if (detail) d = clamp(d - (vnoise3(q * mix(11.0, 6.0, anvil * cb) + cloudWind() * 0.002) * 0.35) * (1.0 - d), 0.0, 1.0);
    return d * F.clouds.y * (1.0 + 2.0 * cb * (1.0 - anvil));  // storm cores hold far more water
}

float stratusDensity(vec3 p, bool detail) {
    float h = (p.y - F.clouds3.z) / F.clouds3.w;
    if (h <= 0.0 || h >= 1.0 || F.clouds3.x <= 0.0) return 0.0;
    vec3 q = (p + cloudWind() * 0.7) * vec3(0.00025, 0.001, 0.00025);
    float broad = vnoise3(q) * 0.65 + vnoise3(q * 2.3 + 1.3) * 0.35;
    // stratocumulus: rolls / cells about 1 km across, aligned across the wind
    vec2 w = normalize(vec2(F.wind.x, F.wind.y) + vec2(1e-4, 0.0));
    vec2 xz = (p.xz + cloudWind().xz * 0.7);
    float cells = vnoise3(vec3(dot(xz, w) * 0.0022, dot(xz, vec2(-w.y, w.x)) * 0.0011, h * 0.6)) * 0.6 +
                  vnoise3(vec3(xz * 0.004, 3.1 + h)) * 0.4;
    float lump = F.clouds3.y;
    float n = mix(broad, broad * 0.5 + cells * 0.7 - 0.1, lump);
    float cov = F.clouds3.x;
    float shape = smoothstep(0.0, 0.15, h) * smoothstep(1.0, mix(0.8, 0.45, lump), h);
    float d = clamp((n - (1.0 - cov)) / max(cov, 0.05) * 1.5, 0.0, 1.0) * shape;
    if (d <= 0.0) return 0.0;
    if (detail) d = clamp(d - vnoise3(q * 14.0) * 0.25 * (1.0 - d), 0.0, 1.0);
    return d * F.clouds.y * 0.7;
}

// extinction (1/m) at p; detail octaves only when the base shape is present
float cloudDensity(vec3 p, bool detail) { return cumulusDensity(p, detail) + stratusDensity(p, detail); }

// distances along a ray to a slab [b, top]; false if it misses
bool slabHit(vec3 ro, vec3 rd, float b, float top, out float t0, out float t1) {
    if (abs(rd.y) < 1e-4) { t0 = 0.0; t1 = (ro.y > b && ro.y < top) ? 3e4 : -1.0; return t1 > 0.0; }
    float a = (b - ro.y) / rd.y, c = (top - ro.y) / rd.y;
    t0 = max(min(a, c), 0.0); t1 = min(max(a, c), t0 + 3e4);
    return t1 > t0;
}
bool cloudSlab(vec3 ro, vec3 rd, out float t0, out float t1) { return slabHit(ro, rd, F.clouds2.x, convTop(), t0, t1); }

// fraction of sunlight reaching p through the cloud layers
float cloudShadow(vec3 p) {
    if (!cloudsOn() || F.sunDir.y <= 0.02) return 1.0;
    float t0, t1, od = 0.0;
    const int N = 5;
    if (slabHit(p, F.sunDir.xyz, F.clouds2.x, convTop(), t0, t1)) {
        t1 = min(t1, t0 + 6000.0);
        float ds = (t1 - t0) / float(N);
        for (int i = 0; i < N; i++) od += cumulusDensity(p + F.sunDir.xyz * (t0 + (float(i) + 0.5) * ds), false) * ds;
    }
    if (F.clouds3.x > 0.0 && slabHit(p, F.sunDir.xyz, F.clouds3.z, F.clouds3.z + F.clouds3.w, t0, t1)) {
        float ds = (t1 - t0) / 3.0;
        for (int i = 0; i < 3; i++) od += stratusDensity(p + F.sunDir.xyz * (t0 + (float(i) + 0.5) * ds), false) * ds;
    }
    if (F.clouds4.w > p.y) od += cirrusShadowTau(p.xz + F.sunDir.xz * (F.clouds4.w - p.y) / F.sunDir.y) / F.sunDir.y;
    return mix(1.0, exp(-od), F.clouds2.z);
}

// one volumetric layer marched over [t0, t1]; layer 0 = convective, 1 = stratus
vec3 marchLayer(int layer, vec3 ro, vec3 rd, float t0, float t1, int N, float jitter, float phase, vec3 amb, inout float T) {
    float ds = (t1 - t0) / float(N);
    vec3 s = F.sunDir.xyz, sunL = F.sunColor.rgb, L = vec3(0);
    float lb = layer == 0 ? F.clouds2.x : F.clouds3.z, lh = layer == 0 ? convTop() - F.clouds2.x : F.clouds3.w;
    float ls = (layer == 0 ? F.clouds2.y : F.clouds3.w) * 0.12;
    for (int i = 0; i < N && T > 0.01; i++) {
        vec3 p = ro + rd * (t0 + (float(i) + jitter) * ds);
        float d = layer == 0 ? cumulusDensity(p, true) : stratusDensity(p, true);
        if (d <= 0.0) continue;
        // light march toward the sun (through both volumetric layers)
        float od = 0.0;
        for (int j = 0; j < 5; j++) od += cloudDensity(p + s * (float(j) + 0.5) * ls, false);
        float beer = exp(-od * ls), powder = 1.0 - exp(-od * ls * 2.0);
        float hgt = clamp((p.y - lb) / lh, 0.0, 1.0);
        vec3 Ls = sunL * phase * mix(beer, beer * powder * 2.0, 0.5) * 2.0 + amb * mix(0.35, 1.0, hgt) / (4.0 * PI) * 4.0;
        float a = exp(-d * ds);
        L += T * Ls * (1.0 - a);   // energy-conserving integration (per unit extinction)
        T *= a;
    }
    return L;
}

// in-scattered cloud radiance along a view ray; T = transmittance through the clouds
vec3 cloudMarch(vec3 ro, vec3 rd, float jitter, out float T) {
    T = 1.0;
    if (!cloudsOn()) return vec3(0);
    vec3 s = F.sunDir.xyz;
    float mu = dot(rd, s);
    float hg1 = (1.0 - 0.64) / (4.0 * PI * pow(1.0 + 0.64 - 1.6 * mu, 1.5));     // forward silver lining
    float hg2 = (1.0 - 0.09) / (4.0 * PI * pow(1.0 + 0.09 + 0.6 * mu, 1.5));     // back scatter
    float phase = mix(hg2, hg1, 0.7);
    vec3 amb = skyAtmo(vec3(0, 1, 0)) * 1.6 + skyAtmo(normalize(vec3(rd.x, 0.05, rd.z))) * 0.4;

    // layers front to back: entry distance of each (1e9 = missed), then a three-element sort
    float a0, a1, b0, b1, ct = 1e9;
    bool hc = (F.clouds.x > 0.0 || F.clouds4.x > 0.0) && slabHit(ro, rd, F.clouds2.x, convTop(), a0, a1);
    bool hs = F.clouds3.x > 0.0 && slabHit(ro, rd, F.clouds3.z, F.clouds3.z + F.clouds3.w, b0, b1);
    if (F.clouds4.z > 0.0 && abs(rd.y) > 1e-3) { float t = (F.clouds4.w - ro.y) / rd.y; if (t > 0.0 && t < 1.5e5) ct = t; }
    vec4 ord = vec4(hc ? a0 : 1e9, hs ? b0 : 1e9, ct, 0);
    int o[3] = int[3](0, 1, 2);
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2 - i; j++)
        if (ord[o[j]] > ord[o[j + 1]]) { int x = o[j]; o[j] = o[j + 1]; o[j + 1] = x; }
    vec3 L = vec3(0);
    for (int k = 0; k < 3 && T > 0.01; k++) {
        int l = o[k];
        if (ord[l] >= 1e9) break;
        if (l == 0) L += marchLayer(0, ro, rd, a0, a1, F.clouds4.x > 0.0 ? 56 : 40, jitter, phase, amb, T);
        else if (l == 1) L += marchLayer(1, ro, rd, b0, b1, 20, jitter, phase, amb, T);
        else {
            float Tc;
            L += T * cirrusMarch(ro, rd, 2.0 * F.camUp.w / float(F.res.y), amb, Tc);
            T *= Tc;
        }
    }
    if (T < 0.01) T = 0.0;  // the march stops early once opaque: don't leave 1% for the (very bright) sun disc
    // fade into the haze near the horizon instead of a hard edge at 30 km
    float fade = smoothstep(0.0, 0.08, rd.y);
    T = mix(1.0, T, fade);
    return L * fade;
}

// The cheap stand-in, at full resolution every pixel: a few fixed samples per layer with one light lookup each, no
// jitter. It carries the sharp edges and fine shapes; cloud.comp marches the real thing at low resolution and keeps
// only the difference (march - fake), which is smooth enough to upsample. Same layers, order and fade as cloudMarch.
vec3 cloudFake(vec3 ro, vec3 rd, out float T) {
    T = 1.0;
    if (!cloudsOn()) return vec3(0);
    vec3 s = F.sunDir.xyz, sunL = F.sunColor.rgb;
    float mu = dot(rd, s);
    float phase = mix((1.0 - 0.09) / (4.0 * PI * pow(1.0 + 0.09 + 0.6 * mu, 1.5)), (1.0 - 0.64) / (4.0 * PI * pow(1.0 + 0.64 - 1.6 * mu, 1.5)), 0.7);
    vec3 amb = skyAtmo(vec3(0, 1, 0)) * 1.6 + skyAtmo(normalize(vec3(rd.x, 0.05, rd.z))) * 0.4;
    float a0, a1, b0, b1, ct = 1e9;
    bool hc = (F.clouds.x > 0.0 || F.clouds4.x > 0.0) && slabHit(ro, rd, F.clouds2.x, convTop(), a0, a1);
    bool hs = F.clouds3.x > 0.0 && slabHit(ro, rd, F.clouds3.z, F.clouds3.z + F.clouds3.w, b0, b1);
    if (F.clouds4.z > 0.0 && abs(rd.y) > 1e-3) { float t = (F.clouds4.w - ro.y) / rd.y; if (t > 0.0 && t < 1.5e5) ct = t; }
    vec4 ord = vec4(hc ? a0 : 1e9, hs ? b0 : 1e9, ct, 0);
    int o[3] = int[3](0, 1, 2);
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2 - i; j++)
        if (ord[o[j]] > ord[o[j + 1]]) { int x = o[j]; o[j] = o[j + 1]; o[j + 1] = x; }
    vec3 L = vec3(0);
    const int N = 4;
    for (int k = 0; k < 3 && T > 0.01; k++) {
        int l = o[k];
        if (ord[l] >= 1e9) break;
        if (l == 2) {
            float Tc;
            L += T * cirrusMarch(ro, rd, 2.0 * F.camUp.w / float(F.res.y), amb, Tc);
            T *= Tc;
            continue;
        }
        float t0 = l == 0 ? a0 : b0, t1 = l == 0 ? a1 : b1, ds = (t1 - t0) / float(N);
        float lb = l == 0 ? F.clouds2.x : F.clouds3.z, lh = l == 0 ? convTop() - F.clouds2.x : F.clouds3.w;
        float ls = (l == 0 ? F.clouds2.y : F.clouds3.w) * 0.3;
        for (int i = 0; i < N && T > 0.01; i++) {
            vec3 p = ro + rd * (t0 + (float(i) + 0.5) * ds);
            float d = l == 0 ? cumulusDensity(p, true) : stratusDensity(p, true);
            if (d <= 0.0) continue;
            float od = cloudDensity(p + s * ls, false) * ls * 2.0;
            float beer = exp(-od), powder = 1.0 - exp(-od * 2.0);
            float hgt = clamp((p.y - lb) / lh, 0.0, 1.0);
            vec3 Ls = sunL * phase * mix(beer, beer * powder * 2.0, 0.5) * 2.0 + amb * mix(0.35, 1.0, hgt) / (4.0 * PI) * 4.0;
            float a = exp(-d * ds);
            L += T * Ls * (1.0 - a);
            T *= a;
        }
    }
    if (T < 0.01) T = 0.0;
    float fade = smoothstep(0.0, 0.08, rd.y);
    T = mix(1.0, T, fade);
    return L * fade;
}
#endif
