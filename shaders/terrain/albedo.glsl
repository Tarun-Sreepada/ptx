// Macro ground colour: grass mask, soil, stone, river bed, craters, flatten zones.
#ifndef TERRAIN_ALBEDO
#define TERRAIN_ALBEDO
#include "core/frame.glsl"
#include "terrain/height.glsl"

// 0 = bare ground, 1 = full grass: large clumps, bare patches and a winding dirt path.
float grassMask(vec2 xz) {
    float n = vnoise(xz * 0.06) * 0.65 + vnoise(xz * 0.21) * 0.35;
    float m = smoothstep(0.26, 0.42, n);
    float path = abs(xz.y - 9.0 * sin(xz.x * 0.045) - 4.0 * sin(xz.x * 0.11 + 1.0));
    float dry = 1.0;
    if (riverNear(xz, 2.0)) { vec4 r = riverInfo(xz); dry = smoothstep(r.y - 0.3, r.y + 1.2, abs(r.x)); }  // none under water
    float painted = mapPaintAt(xz).r * 2.0;                       // world map grass brush (1 = as generated)
    float buried = 1.0 - smoothstep(0.04, 0.25, snowDepth(xz));   // under snow
    buried *= 1.0 - smoothstep(0.6, 1.0, editSlope(xz));          // nothing grows on faces steeper than ~35-45 deg
    buried *= biomeGrass(xz);                                      // deserts, ice, salt, rock; none under the sea / lakes
    return clamp(m * painted + max(painted - 1.0, 0.0) * 0.8, 0.0, 1.0) * smoothstep(0.9, 1.7, path) * (1.0 - zoneCover(xz)) * dry * (1.0 - craterDisturbance(xz)) * buried;  // no grass on flattened (city) ground
}

// Blade length field (0 = short lawn, 1 = tall meadow); grass.comp scales blade height by it.
float grassHeight(vec2 xz) {
    float n = vnoise(xz * 0.045 + 31.0) * 0.7 + vnoise(xz * 0.17 + 5.0) * 0.3;
    return smoothstep(0.25, 0.75, n) * (0.4 + 0.6 * smoothstep(0.2, 0.8, grassMask(xz)));
}

// Ground colour follows grassMask (the same field that places blades): green where grass grows,
// brown soil / gray stony ground where it doesn't, with a dry fringe between.
// farGrass: 0 under real blades (shaded soil-green), 1 past the grass draw distance (looks like the field from afar)
vec3 terrainAlbedo(vec2 xz, float farGrass) {
    float mask = grassMask(xz);
    float k = vnoise(xz * 0.35) * 0.6 + vnoise(xz * 2.1) * 0.4;
    vec3 green = mix(vec3(0.10, 0.17, 0.045), vec3(0.16, 0.25, 0.06), k);
    green *= 1.0 - 0.25 * grassHeight(xz) * (1.0 - farGrass);  // tall grass shades the soil below
    green = mix(green * 0.8, mix(vec3(0.17, 0.30, 0.06), vec3(0.26, 0.33, 0.09), k), farGrass);
    vec3 soil = mix(vec3(0.19, 0.14, 0.09), vec3(0.30, 0.24, 0.16), vnoise(xz * 3.0));
    vec3 stone = mix(vec3(0.22, 0.21, 0.19), vec3(0.34, 0.33, 0.30), vnoise(xz * 4.3 + 2.0));
    vec3 bare = mix(soil, stone, smoothstep(0.45, 0.8, vnoise(xz * 0.09 + 17.0)));
    vec3 dry = vec3(0.30, 0.27, 0.13);
    float g = smoothstep(0.05, 0.6, mask);
    vec3 ground = mix(bare, green, g);
    ground = mix(ground, dry, 4.0 * g * (1.0 - g) * 0.45);
    // steep sculpted faces: bare soil and rock (lighter, greyer as it steepens)
    float sl = editSlope(xz);
    if (sl > 0.4) {
        vec3 rockC = mix(vec3(0.26, 0.24, 0.21), vec3(0.38, 0.36, 0.32), vnoise(xz * 0.9 + 3.0));
        vec3 soilC = mix(vec3(0.22, 0.17, 0.12), vec3(0.3, 0.24, 0.17), vnoise(xz * 2.3));
        ground = mix(ground, mix(soilC, rockC, smoothstep(0.8, 1.6, sl)), smoothstep(0.4, 0.9, sl));
    }
    if (riverNear(xz, 3.0)) {  // river bed: pebbly gravel in the channel, dark wet silt along the waterline
        vec4 r = riverInfo(xz);
        float d = abs(r.x);
        vec3 gravel = mix(vec3(0.16, 0.14, 0.11), vec3(0.32, 0.29, 0.24), smoothstep(0.35, 0.75, vnoise(xz * 6.0) * 0.6 + vnoise(xz * 17.0) * 0.4));
        vec3 silt = mix(vec3(0.09, 0.08, 0.06), vec3(0.15, 0.13, 0.09), vnoise(xz * 2.3));
        ground = mix(ground, silt, 1.0 - smoothstep(r.y + 0.3, r.y + 2.5, d));
        ground = mix(ground, gravel, 1.0 - smoothstep(r.y * 0.6, r.y, d));
    }
    ground = biomeAlbedo(xz, ground, g);
    return mix(ground, mix(vec3(0.23, 0.22, 0.2), vec3(0.3, 0.29, 0.26), vnoise(xz * 1.7)), smoothstep(0.5, 1.0, zoneCover(xz)));
}
#endif
