// The lightning flash lighting the scene: the channel (lightning.glsl segments) is a line light. Each pixel takes a
// random point along it and one shadow ray, so hills, trees and houses throw sharp, flickering shadows; it falls off
// with distance squared and is blue-white.
#ifndef LIGHTING_LIGHTNING
#define LIGHTING_LIGHTNING
#include "core/frame.glsl"

const float BOLT_INTENSITY = 1.0e6;  // radiant intensity of the channel (scene units x m^2): ~1 at a kilometre
const float BOLT_MAX_E = 20.0;       // irradiance right next to a strike (scene units)
const vec3 BOLT_COLOR = vec3(0.8, 0.85, 1.0);

vec3 lightningDirect(vec3 x, Surf sf, vec3 v, vec3 A, float u) {
    uint n = uint(F.boltInfo.x);
    if (n == 0u || F.boltInfo.y <= 0.0) return vec3(0);
    float fi = u * float(n);
    uint i = min(uint(fi), n - 1u);
    vec4 sa = F.bolt[2u * i], sb = F.bolt[2u * i + 1u];
    vec3 d = mix(sa.xyz, sb.xyz, fract(fi)) - x;
    float dist2 = max(dot(d, d), 1.0), dist = sqrt(dist2);
    vec3 l = d / dist;
    vec3 f = evalBRDF(sf, v, l);
    if (f.r + f.g + f.b <= 0.0) return vec3(0);
    float side = dot(sf.n, l) >= 0.0 ? 1.0 : -1.0;
    if (!visible(x + sf.n * (EPS * side), l, dist - 1.0)) return vec3(0);
    float E = min(BOLT_INTENSITY * sb.w * F.boltInfo.y / dist2, BOLT_MAX_E);
    return BOLT_COLOR * E * f / A;
}
#endif
