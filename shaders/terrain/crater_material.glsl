// Freshly dug ground. A crater's walls cut through the soil profile: dark organic topsoil, then iron-stained clay
// subsoil, sand and gravel, weathered rock at depth, with wandering boundaries. The floor is fallback (everything
// mixed, in clods). Around it the ejecta blanket lies over the grass: subsoil clods in clumps and rays, thinning out.
// All of it is 3D (no top-down projection), so steep walls don't streak.
#ifndef TERRAIN_CRATER_MATERIAL
#define TERRAIN_CRATER_MATERIAL
#include "core/frame.glsl"
#include "terrain/height.glsl"

float soilNoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    uvec3 c = uvec3(ivec3(i));
    #define SH(o) (float(hashu(hashu(hashu(c.x + o.x) ^ (c.y + o.y) * 0x9e3779b9u) ^ (c.z + o.z) * 0x85ebca6bu) & 0xFFFFFFu) / 16777216.0)
    float a = mix(SH(uvec3(0, 0, 0)), SH(uvec3(1, 0, 0)), f.x), b = mix(SH(uvec3(0, 1, 0)), SH(uvec3(1, 1, 0)), f.x);
    float d = mix(SH(uvec3(0, 0, 1)), SH(uvec3(1, 0, 1)), f.x), e = mix(SH(uvec3(0, 1, 1)), SH(uvec3(1, 1, 1)), f.x);
    #undef SH
    return mix(mix(a, b, f.y), mix(d, e, f.y), f.z);
}

// Layer depths below the original surface (m)
const float TOPSOIL_DEPTH = 0.25, SUBSOIL_DEPTH = 1.3, SAND_DEPTH = 3.5;

// ground: the colour the surface had before; disturbed: how much of it is freshly dug / thrown soil (0..1)
vec3 craterSoil(vec3 P, vec3 n, vec3 ground, inout float rough, out float disturbed) {
    disturbed = 0.0;
    if (F.dynInfo.z == 0u) return ground;
    CraterSum s = craterSumAt(P.xz);
    if (s.ejecta <= 0.0 && s.bowl >= 1e29) return ground;
    float cut = groundBase(P.xz) - P.y;  // > 0: dug below the original ground here (m)
    vec3 topsoil = mix(vec3(0.07, 0.055, 0.04), vec3(0.12, 0.085, 0.055), soilNoise(P * 6.0));
    vec3 subsoil = mix(vec3(0.26, 0.17, 0.095), vec3(0.36, 0.25, 0.14), soilNoise(P * 3.0 + 1.7));
    vec3 sand = mix(vec3(0.37, 0.31, 0.22), vec3(0.46, 0.40, 0.30), soilNoise(P * 9.0));
    vec3 rock = mix(vec3(0.29, 0.28, 0.26), vec3(0.42, 0.40, 0.37), soilNoise(P * 2.2 + 5.0));
    // the soil profile, its boundaries wandering a little through the ground
    float c = cut + 0.1 * (soilNoise(P * 1.3) - 0.5) + 0.05 * (soilNoise(P * 5.0 + 9.0) - 0.5);
    vec3 layer = mix(topsoil, subsoil, smoothstep(TOPSOIL_DEPTH - 0.05, TOPSOIL_DEPTH + 0.05, c));
    layer = mix(layer, sand, smoothstep(SUBSOIL_DEPTH - 0.15, SUBSOIL_DEPTH + 0.15, c));
    layer = mix(layer, rock, smoothstep(SAND_DEPTH - 0.4, SAND_DEPTH + 0.4, c));
    // pebbles and gravel weathering out of the deeper layers
    float grit = smoothstep(0.72, 0.85, soilNoise(P * 22.0)) * smoothstep(0.8, 2.0, c);
    layer = mix(layer, rock * 1.15, grit * 0.6);
    // the floor: fallback breccia, everything mixed in clods
    float floorK = smoothstep(0.78, 0.96, n.y);
    vec3 fallback = mix(subsoil, topsoil, 0.3 + 0.35 * soilNoise(P * 4.0)) * (0.85 + 0.3 * soilNoise(P * 13.0));
    vec3 cutCol = mix(layer, fallback, floorK * 0.75);
    float inCut = smoothstep(0.02, 0.12, cut);
    // ejecta blanket: subsoil clods in clumps and rays over whatever grew there
    float ang = atan(P.z - P.x * 0.37, P.x + P.z * 0.21);
    float rays = soilNoise(vec3(ang * 4.0, 0.0, 1.0));
    float clump = smoothstep(0.35, 0.75, soilNoise(vec3(P.xz * 1.5, 0.0)) * 0.5 + soilNoise(vec3(P.xz * 5.0, 3.0)) * 0.3 + rays * 0.2);
    float cover = smoothstep(0.004, 0.06, s.ejecta * (0.3 + 1.4 * clump));
    vec3 ejectaCol = mix(subsoil, topsoil, 0.3 + 0.4 * soilNoise(P * 7.0)) * (0.9 + 0.2 * soilNoise(P * 17.0));
    vec3 col = mix(ground, ejectaCol, cover * (1.0 - inCut));
    col = mix(col, cutCol, inCut);
    disturbed = max(inCut, cover);
    rough = mix(rough, 0.93, disturbed);
    return col;
}
#endif
