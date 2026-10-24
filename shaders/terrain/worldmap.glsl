// World map (worldmap.h): hand edits and the GPU simulation grid (snow, water, wetness).
#ifndef TERRAIN_WORLDMAP
#define TERRAIN_WORLDMAP
#include "core/frame.glsl"

// ---- world map (worldmap.h): hand edits on a 2 m grid over 4 km, and the simulation grid on the same layout
layout(binding = 90) uniform sampler2D mapHeight;   // metres added to the natural ground
layout(binding = 91) uniform sampler2D mapPaint;    // r grass x2, g trees x2, b texture index / 255, a texture weight
layout(binding = 92) uniform sampler2D mapWater;    // drawn rivers (4 m): x level (-1000 none), yz flow (m/s), w half-width
layout(binding = 93, rgba16f) uniform image2D simImg;  // x snow depth (m), y water depth (m), z wetness, w avalanche speed
vec2 mapUV(vec2 p) { return (p - F.mapP.x) / (F.mapP.y * F.mapP.z); }
bool inMap(vec2 uv) { return uv.x >= 0.0 && uv.y >= 0.0 && uv.x < 1.0 && uv.y < 1.0; }
float editHeight(vec2 p) { vec2 uv = mapUV(p); return inMap(uv) ? textureLod(mapHeight, uv, 0.0).r : 0.0; }
vec4 mapPaintAt(vec2 p) { vec2 uv = mapUV(p); return inMap(uv) ? textureLod(mapPaint, uv, 0.0) : vec4(0.5, 0.5, 0.0, 0.0); }
// drawn water: bilinear over only the neighbouring cells that hold water (a cell without stores -1000; blending that
// in made sliver lines and grid-shaped edges). The shoreline is then where this smooth level meets the ground.
vec4 mapWaterAt(vec2 p) {
    vec2 uv = mapUV(p);
    if (!inMap(uv)) return vec4(-1000.0, 0, 0, 0);
    vec2 g = uv * vec2(textureSize(mapWater, 0)) - 0.5;
    ivec2 i = ivec2(floor(g));
    vec2 f = g - vec2(i);
    ivec2 sz = textureSize(mapWater, 0) - 1;
    vec4 acc = vec4(0);
    float wsum = 0.0;
    for (int k = 0; k < 4; k++) {
        ivec2 o = ivec2(k & 1, k >> 1);
        vec4 t = texelFetch(mapWater, clamp(i + o, ivec2(0), sz), 0);
        float w = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);
        if (t.x > -999.0) { acc += t * w; wsum += w; }
    }
    return wsum > 0.0 ? acc / wsum : vec4(-1000.0, 0, 0, 0);
}
// simulation state, bilinear (storage image: filtered by hand)
vec4 simAt(vec2 p) {
    if (F.sim1.w < 0.5) return vec4(0);
    vec2 g = (p - F.mapP.x) / F.mapP.y - 0.5;
    ivec2 i = ivec2(floor(g));
    int n = int(F.mapP.z);
    if (i.x < 0 || i.y < 0 || i.x >= n - 1 || i.y >= n - 1) return vec4(0);
    vec2 f = g - vec2(i);
    return mix(mix(imageLoad(simImg, i), imageLoad(simImg, i + ivec2(1, 0)), f.x),
               mix(imageLoad(simImg, i + ivec2(0, 1)), imageLoad(simImg, i + ivec2(1, 1)), f.x), f.y);
}
// steepness of the hand-sculpted ground (tan of the slope); the natural noise terrain is gentle everywhere
float editSlope(vec2 p) {
    vec2 uv = mapUV(p);
    if (!inMap(uv)) return 0.0;
    float e = F.mapP.y;
    float hx = editHeight(p + vec2(e, 0)) - editHeight(p - vec2(e, 0)), hz = editHeight(p + vec2(0, e)) - editHeight(p - vec2(0, e));
    return length(vec2(hx, hz)) / (2.0 * e);
}
float snowDepth(vec2 p) { return F.sim1.w > 0.5 ? simAt(p).x : 0.0; }
#endif
