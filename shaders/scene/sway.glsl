// Vegetation in the wind, for the raster pass (the BVH keeps the plants at rest; lighting rays start at the rest
// position, camera.glsl loadRayPos). Worked out from the vertex's place in the plant alone, no per-vertex data:
//  - the whole plant leans with the local wind (wind map) and sways about that lean at its natural frequency
//    (tall trees slowly, reeds quickly), bending as a cantilever: deflection ~ (height / plant height)^2
//  - limbs bob with their own phase, more the further out from the trunk they reach
//  - leaves and twig ends flutter, each with its own phase
// Every term is a smooth function of position, so the mesh never tears.
#ifndef SCENE_SWAY
#define SCENE_SWAY
#include "core/frame.glsl"
#include "scene/wind.glsl"

const float SWAY_LEAN = 0.012, SWAY_LEAN2 = 0.03;  // top deflection per metre of plant: x wind, x wind^2
const float SWAY_MAX_LEAN = 0.45;                   // never more than this share of the plant's height
const float SWAY_OSC = 0.35;                        // oscillation about the lean, share of it
const float SWAY_FREQ_H = 3.0;                      // natural frequency (Hz) x height (m)
const float SWAY_BRANCH = 0.06;                     // limb bob (m) at full reach of a 10 m tree, x wind
const float SWAY_FLUTTER = 0.012;                   // leaf flutter (m), x wind
const float SWAY_CALM = 0.08;                       // a little motion even in still air
const float SWAY_CROWN = 0.35;                      // crown radius / plant height (where limbs reach full bob)

// rel: vertex - plant base (world, m); base: plant base (world); H: plant height (m); w: phase in the fraction,
// flexibility in the integer part (0 trees, more for bushes and reeds); t: time (s). Returns the world offset.
vec3 plantSway(vec3 rel, vec3 base, float H, float w, float t) {
    vec2 wind = windAt(base.xz).xy;
    float wl = length(wind), s = wl + SWAY_CALM;
    vec2 dir = wl > 1e-4 ? wind / wl : normalize(F.wind.xy + vec2(1e-4, 0.0));
    float flex = 1.0 + floor(w), ph = fract(w) * 6.2831853;
    float hN = clamp(rel.y / H, 0.0, 1.5), bend = hN * hN;
    float f0 = clamp(SWAY_FREQ_H / H, 0.15, 2.0) * 6.2831853;  // rad/s
    // lean with the wind, sway about it, and a slower, smaller swing across it
    float A = min(H * (SWAY_LEAN * s + SWAY_LEAN2 * s * s) * flex, SWAY_MAX_LEAN * H);
    float osc = sin(f0 * t + ph) + 0.35 * sin(f0 * 2.13 * t + ph * 1.7);
    vec3 along = vec3(dir.x, 0.0, dir.y), across = vec3(-dir.y, 0.0, dir.x);
    vec3 d = along * (A * (1.0 + SWAY_OSC * osc) * bend) + across * (A * 0.25 * sin(f0 * 0.71 * t + ph * 2.3) * bend);
    // limbs: the further out, the more they bob; neighbouring parts share their phase (a limb moves as one)
    float reach = length(rel.xz) / max(SWAY_CROWN * H, 0.3);
    float lp = dot(base.xz + rel.xz, vec2(0.7, 0.5)) * 0.6 + rel.y * 0.4 + ph;
    float bob = sin(f0 * 2.7 * t + lp) * SWAY_BRANCH * s * flex * min(H / 10.0, 1.5) * reach * reach;
    d += vec3(dir.x * 0.4, 1.0, dir.y * 0.4) * bob;
    // leaves: quick flutter, each its own phase, on the outer parts only
    float fp = dot(base + rel, vec3(7.3, 5.1, 6.7)) + ph;
    vec3 fl = vec3(sin(t * 11.0 + fp), sin(t * 13.7 + fp * 1.3), sin(t * 9.1 + fp * 0.7));
    d += fl * (SWAY_FLUTTER * s * sqrt(flex) * smoothstep(0.1, 0.6, reach + hN * 0.3));
    d.y -= 0.5 * dot(d.xz, d.xz) / max(rel.y, 0.5);  // the bent top drops a little: limbs keep their length
    return d;
}
#endif
