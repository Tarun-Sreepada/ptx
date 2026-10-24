// Procedural rock material (3D, no UVs).
#ifndef MATERIALS_ROCK
#define MATERIALS_ROCK
#include "core/frame.glsl"
#include "terrain/height.glsl"
#include "terrain/albedo.glsl"

// Procedural rock material (3D, no UVs: works on every instanced, rotated rock and on fragments' fresh faces).
// Layers, as on real weathered stone:
//  - mineral grain: Voronoi crystals at ~4 mm, each a mineral (dark mica / hornblende, pink-buff feldspar, glassy
//    quartz) - the salt-and-pepper look of granite up close
//  - broad tone variation and a faint banding (foliation), quartz veins as thin pale lines
//  - relief: ridged noise (pitted, crested surface); cavities collect dark dirt, exposed crests are bleached lighter
//  - weathering: dark water streaks running down the sides, lichen rosettes (pale grey-green, some orange) and
//    moss on the upward faces, soil splashed up where the rock meets the ground
float rnoise3(vec3 p) {
    vec3 i = floor(p), f = p - i;
    f = f * f * (3.0 - 2.0 * f);
    uvec3 u = uvec3(ivec3(i));
    #define RH(o) (float(hashu(hashu(hashu(u.x + o.x) ^ (u.y + o.y)) ^ (u.z + o.z)) & 0xFFFFFFu) / 16777216.0)
    float a = mix(RH(uvec3(0, 0, 0)), RH(uvec3(1, 0, 0)), f.x), b = mix(RH(uvec3(0, 1, 0)), RH(uvec3(1, 1, 0)), f.x);
    float c = mix(RH(uvec3(0, 0, 1)), RH(uvec3(1, 0, 1)), f.x), d = mix(RH(uvec3(0, 1, 1)), RH(uvec3(1, 1, 1)), f.x);
    #undef RH
    return mix(mix(a, b, f.y), mix(c, d, f.y), f.z);
}
// Voronoi: x = distance to the nearest cell point, y = second nearest, z = nearest cell's hash
vec3 rcell3(vec3 p) {
    vec3 i = floor(p), f = p - i;
    float d1 = 8.0, d2 = 8.0, id = 0.0;
    for (int z = -1; z <= 1; z++)
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++) {
                uvec3 c = uvec3(ivec3(i) + ivec3(x, y, z));
                uint h = hashu(hashu(hashu(c.x) ^ c.y) ^ c.z);
                vec3 o = vec3(x, y, z) + vec3(h & 0x3FFu, (h >> 10) & 0x3FFu, (h >> 20) & 0x3FFu) / 1023.0;
                float d = dot(o - f, o - f);
                if (d < d1) { d2 = d1; d1 = d; id = float(hashu(h) & 0xFFFFu) / 65535.0; }
                else if (d < d2) d2 = d;
            }
    return vec3(sqrt(d1), sqrt(d2), id);
}
float rockRelief(vec3 p) {
    float r = 0.0, a = 0.5, f = 2.5;
    for (int o = 0; o < 4; o++, a *= 0.5, f *= 2.2) r += a * (1.0 - abs(2.0 * rnoise3(p * f + float(o) * 7.1) - 1.0));
    return r * 0.9;
}
vec3 rockAlbedoC(vec3 p, vec3 n, float relief) {
    // broad tone + foliation bands
    float tone = rnoise3(p * 0.9) * 0.6 + rnoise3(p * 3.7 + 5.0) * 0.4;
    float band = 0.5 + 0.5 * sin(dot(p, vec3(0.3, 5.0, 0.8)) * 3.0 + rnoise3(p * 2.0) * 4.0);
    vec3 c = mix(vec3(0.24, 0.23, 0.21), vec3(0.42, 0.40, 0.36), tone) * (0.92 + 0.08 * band);
    // mineral grain
    vec3 g = rcell3(p * 220.0);
    vec3 mineral = g.z < 0.22 ? vec3(0.05, 0.05, 0.05) : (g.z < 0.55 ? vec3(0.50, 0.42, 0.36) : (g.z < 0.85 ? vec3(0.40, 0.39, 0.37) : vec3(0.62, 0.61, 0.58)));
    c = mix(c, mineral, 0.35);
    // quartz veins
    float vn = abs(rnoise3(p * 1.3 + 17.0) - 0.5) + 0.02 * rnoise3(p * 30.0);
    c = mix(c, vec3(0.66, 0.64, 0.6), (1.0 - smoothstep(0.004, 0.012, vn)) * 0.8);
    // dirt in cavities, bleached crests
    c *= mix(0.55, 1.12, smoothstep(0.1, 0.75, relief));
    // water streaks down the sides
    float side = 1.0 - abs(n.y);
    float streak = smoothstep(0.55, 0.8, rnoise3(vec3(p.x * 9.0, p.y * 0.6, p.z * 9.0))) * side;
    c *= 1.0 - 0.35 * streak;
    // lichen rosettes on the upper faces
    vec3 lc = rcell3(p * 6.0 + 41.0);
    float up = smoothstep(0.1, 0.6, n.y);
    float lichen = (1.0 - smoothstep(0.18, 0.3, lc.x + 0.1 * rnoise3(p * 40.0))) * step(0.45, lc.z) * up;
    vec3 lcol = lc.z > 0.9 ? vec3(0.55, 0.32, 0.08) : vec3(0.50, 0.52, 0.42);
    c = mix(c, lcol * (0.85 + 0.3 * rnoise3(p * 60.0)), lichen * 0.85);
    // moss in the damp hollows of the top
    float moss = smoothstep(0.55, 0.9, n.y) * smoothstep(0.4, 0.7, rnoise3(p * 1.3 + 7.0)) * (1.2 - relief);
    return mix(c, vec3(0.10, 0.16, 0.04) * (0.8 + 0.4 * rnoise3(p * 25.0)), clamp(moss, 0.0, 1.0));
}
vec3 rockAlbedo(vec3 p, vec3 n) { return rockAlbedoC(p, n, 0.6); }  // indirect rays: no relief
vec3 rockSurface(vec3 p, inout vec3 n, out float rough, out float ao) {
    const float e = 0.004;
    float h0 = rockRelief(p);
    vec3 g = vec3(rockRelief(p + vec3(e, 0, 0)) - h0, rockRelief(p + vec3(0, e, 0)) - h0, rockRelief(p + vec3(0, 0, e)) - h0) / e;
    g -= n * dot(g, n);
    n = normalize(n - g * 0.02);
    vec3 alb = rockAlbedoC(p, n, h0);
    float moss = smoothstep(0.55, 0.9, n.y) * smoothstep(0.4, 0.7, rnoise3(p * 1.3 + 7.0));
    rough = mix(mix(0.85, 0.6, smoothstep(0.2, 0.8, h0)), 0.95, moss);  // crests worn smoother
    ao = mix(0.45, 1.0, smoothstep(0.0, 0.7, h0));  // crevices and cracks
    float above = p.y - terrainHeight(p.xz);
    float b = 1.0 - smoothstep(0.0, 0.35, above);
    alb = mix(alb, terrainAlbedo(p.xz, 0.0) * 0.8, b * 0.75);
    ao *= mix(1.0, 0.55, b);
    return alb;
}
#endif
