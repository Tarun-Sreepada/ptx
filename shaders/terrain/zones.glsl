// Flatten zones (city plots, roads): mirrors world/terrain.h.
#ifndef TERRAIN_ZONES
#define TERRAIN_ZONES
#include "core/frame.glsl"

// Flatten zones (streamed with the world cells): ground pulled to a fixed height inside a rectangle,
// smoothstep falloff around it. Count in F.counts.x. Mirrors terrainHeight() in terrain.h.
layout(binding = 36, std430) readonly buffer Zones { vec4 zones[]; };  // (minx, minz, maxx, maxz), (height, falloff, -, -)
float zoneWeight(uint i, vec2 p) {
    vec4 r = zones[2u * i];
    vec2 d = max(max(r.xy - p, p - r.zw), vec2(0.0));
    return 1.0 - smoothstep(0.0, 1.0, length(d) / max(zones[2u * i + 1u].y, 1e-3));
}
float zoneCover(vec2 p) {
    float w = 0.0;
    for (uint i = 0u; i < F.counts.x; i++) w = max(w, zoneWeight(i, p));
    return w;
}
#endif
