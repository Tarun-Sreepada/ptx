// Grass blade colour and roughness (per-blade random numbers shared with grass.comp).
#ifndef MATERIALS_GRASS
#define MATERIALS_GRASS
#include "core/frame.glsl"
#include "terrain/noise.glsl"

// Per-blade random numbers shared by shape (grass.comp) and colour (grassAlbedo).
vec4 bladeRand(uint seed) {
    uint h = hashu(seed);
    return vec4(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u), float(h >> 24)) / 255.0;
}
bool bladeDead(vec4 r)   { return r.z < F.grassD.w; }
bool bladeFlower(vec4 r) { return r.w > 0.994; }

// t = 0 at the root, 1 at the tip
vec3 grassAlbedo(uint seed, float t) {
    vec4 r = bladeRand(seed);
    vec3 root = vec3(0.04, 0.07, 0.02);
    float cv = F.vegCol.x;  // colour variation: 0 = every blade the same green
    vec3 green = mix(vec3(0.15, 0.33, 0.05), vec3(0.32, 0.42, 0.08), clamp(0.5 + (r.x - 0.5) * cv, 0.0, 1.0));
    vec3 straw = mix(vec3(0.42, 0.34, 0.16), vec3(0.56, 0.48, 0.26), r.y);
    vec3 c;
    if (bladeDead(r)) {
        c = mix(root * 1.6, straw, smoothstep(0.0, 0.6, t));
    } else {
        c = mix(root, green, smoothstep(0.0, 0.7, t));
        c = mix(c, straw, smoothstep(0.7, 1.0, t) * r.y * 0.8 * min(cv, 1.0)); // dry tips
    }
    if (bladeFlower(r) && t > 0.75) {
        vec3 petal = r.x < 0.4 ? vec3(0.85, 0.85, 0.8) : (r.x < 0.75 ? vec3(0.9, 0.75, 0.1) : vec3(0.55, 0.3, 0.75));
        c = mix(c, petal, smoothstep(0.75, 0.9, t));
    }
    return c;
}


// Per-blade surface: waxy fresh blades, rougher dry ones.
float grassRough(uint seed) {
    vec4 r = bladeRand(seed);
    return bladeDead(r) ? 0.65 : 0.24 + 0.18 * r.z;
}
#endif
