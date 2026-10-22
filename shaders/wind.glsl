// Wind map: a 2D field around the camera simulated by wind.comp (velocity xy in wind-strength units, z = pressure).
// Stored toroidally (texel = world cell mod WIND_N) so the window can follow the camera without copying.
#ifndef WIND_GLSL
#define WIND_GLSL
const int WIND_N = 256;
const float WIND_CELL = 1.0;  // metres
layout(binding = 52, rgba16f) uniform image2D windA;
layout(binding = 53, rgba16f) uniform image2D windB;

bool windInWindow(ivec2 wc) { ivec2 o = wc - F.windGrid.xy; return all(greaterThanEqual(o, ivec2(0))) && all(lessThan(o, ivec2(WIND_N))); }
ivec2 windTexel(ivec2 wc) { return ((wc % WIND_N) + WIND_N) % WIND_N; }

// bilinear sample of the current field; outside the window (or with the sim off) the global wind
vec3 windAt(vec2 xz) {
    vec3 global = vec3(F.wind.xy * F.wind.z, 0.0);
    if (F.windSim.w < 0.5) return global;
    vec2 c = xz / WIND_CELL - 0.5;
    ivec2 i = ivec2(floor(c));
    vec2 f = c - vec2(i);
    if (!windInWindow(i) || !windInWindow(i + 1)) return global;
    vec3 a = imageLoad(windA, windTexel(i)).xyz, b = imageLoad(windA, windTexel(i + ivec2(1, 0))).xyz;
    vec3 d = imageLoad(windA, windTexel(i + ivec2(0, 1))).xyz, e = imageLoad(windA, windTexel(i + 1)).xyz;
    vec3 w = mix(mix(a, b, f.x), mix(d, e, f.x), f.y);
    // keep gusts plausible for the grass: the local wind stays within ~1.6x the prevailing speed (and a floor so a
    // calm day still has some), however the pressure waves stack up
    float lim = max(length(global.xy) * 1.6, 0.15);
    float l = length(w.xy);
    if (l > lim) w.xy *= lim / l;
    return w;
}
#endif
