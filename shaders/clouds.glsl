// Volumetric cloud layer: a flat slab of advected fbm noise above the world (metres, y up).
// cloudMarch() draws it in the sky (and over the sun disc), cloudShadow() dims the sun on the ground.
#ifndef CLOUDS_GLSL
#define CLOUDS_GLSL

float hash3(uvec3 c) { return float(hashu(c.x * 73856093u ^ hashu(c.y * 19349663u ^ hashu(c.z * 83492791u)))) / 4294967295.0; }
float vnoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    uvec3 c = uvec3(ivec3(i));
    const uvec2 o = uvec2(0u, 1u);
    return mix(mix(mix(hash3(c + o.xxx), hash3(c + o.yxx), f.x), mix(hash3(c + o.xyx), hash3(c + o.yyx), f.x), f.y),
               mix(mix(hash3(c + o.xxy), hash3(c + o.yxy), f.x), mix(hash3(c + o.xyy), hash3(c + o.yyy), f.x), f.y), f.z);
}

bool cloudsOn() { return F.clouds.w > 0.5 && F.clouds.x > 0.0; }

// x scroll applied to every sample: clouds drift with the wind
vec3 cloudWind() { return vec3(F.wind.x, 0.0, F.wind.y) * (F.camPos.w * 12.0 * F.clouds.z); }

// extinction (1/m) at p; detail octaves only when the base shape is present
float cloudDensity(vec3 p, bool detail) {
    float h = (p.y - F.clouds2.x) / F.clouds2.y;
    if (h <= 0.0 || h >= 1.0) return 0.0;
    vec3 q = (p + cloudWind()) * 0.00045;
    float base = vnoise3(q) * 0.55 + vnoise3(q * 2.1 + 3.7) * 0.3 + vnoise3(q * 4.3 + 9.1) * 0.15;
    float cov = F.clouds.x;
    float shape = smoothstep(0.0, 0.07, h) * smoothstep(1.0, 0.35, h);  // flat-ish bottoms, rounded tops
    float d = clamp((base - (1.0 - cov)) / max(cov, 0.05), 0.0, 1.0) * shape;
    if (d <= 0.0) return 0.0;
    if (detail) d = clamp(d - (vnoise3(q * 11.0 + cloudWind() * 0.002) * 0.35) * (1.0 - d), 0.0, 1.0);
    return d * F.clouds.y;
}

// distances along a ray to the slab; false if it misses
bool cloudSlab(vec3 ro, vec3 rd, out float t0, out float t1) {
    float b = F.clouds2.x, top = b + F.clouds2.y;
    if (abs(rd.y) < 1e-4) { t0 = 0.0; t1 = (ro.y > b && ro.y < top) ? 3e4 : -1.0; return t1 > 0.0; }
    float a = (b - ro.y) / rd.y, c = (top - ro.y) / rd.y;
    t0 = max(min(a, c), 0.0); t1 = min(max(a, c), t0 + 3e4);
    return t1 > t0;
}

// fraction of sunlight reaching p through the cloud layer
float cloudShadow(vec3 p) {
    if (!cloudsOn() || F.sunDir.y <= 0.02) return 1.0;
    float t0, t1;
    if (!cloudSlab(p, F.sunDir.xyz, t0, t1)) return 1.0;
    const int N = 5;
    float ds = (t1 - t0) / float(N), od = 0.0;
    for (int i = 0; i < N; i++) od += cloudDensity(p + F.sunDir.xyz * (t0 + (float(i) + 0.5) * ds), false);
    return mix(1.0, exp(-od * ds), F.clouds2.z);
}

// in-scattered cloud radiance along a view ray; T = transmittance through the clouds
vec3 cloudMarch(vec3 ro, vec3 rd, float jitter, out float T) {
    T = 1.0;
    if (!cloudsOn()) return vec3(0);
    float t0, t1;
    if (!cloudSlab(ro, rd, t0, t1)) return vec3(0);
    const int N = 40;
    float ds = (t1 - t0) / float(N);
    vec3 s = F.sunDir.xyz;
    float mu = dot(rd, s);
    float hg1 = (1.0 - 0.64) / (4.0 * PI * pow(1.0 + 0.64 - 1.6 * mu, 1.5));     // forward silver lining
    float hg2 = (1.0 - 0.09) / (4.0 * PI * pow(1.0 + 0.09 + 0.6 * mu, 1.5));     // back scatter
    float phase = mix(hg2, hg1, 0.7);
    vec3 sunL = F.sunColor.rgb;
    vec3 amb = sky(vec3(0, 1, 0)) * 1.6 + sky(normalize(vec3(rd.x, 0.05, rd.z))) * 0.4;
    vec3 L = vec3(0);
    for (int i = 0; i < N && T > 0.01; i++) {
        vec3 p = ro + rd * (t0 + (float(i) + jitter) * ds);
        float d = cloudDensity(p, true);
        if (d <= 0.0) continue;
        // light march toward the sun
        float od = 0.0, ls = F.clouds2.y * 0.12;
        for (int j = 0; j < 5; j++) od += cloudDensity(p + s * (float(j) + 0.5) * ls, false);
        float beer = exp(-od * ls), powder = 1.0 - exp(-od * ls * 2.0);
        float hgt = clamp((p.y - F.clouds2.x) / F.clouds2.y, 0.0, 1.0);
        vec3 Ls = sunL * phase * mix(beer, beer * powder * 2.0, 0.5) * 2.0 + amb * mix(0.35, 1.0, hgt) / (4.0 * PI) * 4.0;
        float a = exp(-d * ds);
        L += T * Ls * (1.0 - a);   // energy-conserving integration (per unit extinction)
        T *= a;
    }
    if (T < 0.01) T = 0.0;  // the march stops early once opaque: don't leave 1% for the (very bright) sun disc
    // fade into the haze near the horizon instead of a hard edge at 30 km
    float fade = smoothstep(0.0, 0.08, rd.y);
    T = mix(1.0, T, fade);
    return L * fade;
}
#endif
