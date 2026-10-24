// ReSTIR DI / GI reservoirs.
#ifndef CORE_RESTIR
#define CORE_RESTIR
#include "core/frame.glsl"
#include "core/bindings.glsl"
#include "lighting/lights.glsl"

// ---------------------------------------------------------------- ReSTIR
struct Reservoir { uint y; float wSum; float M; float W; };

Reservoir unpackRes(vec4 v) { Reservoir r; r.y = floatBitsToUint(v.x); r.wSum = v.y; r.M = v.z; r.W = v.w; return r; }
vec4 packRes(Reservoir r) { return vec4(uintBitsToFloat(r.y), r.wSum, r.M, r.W); }

bool updateRes(inout Reservoir r, uint y, float w, float u) {
    r.wSum += w;
    if (w > 0.0 && u * r.wSum < w) { r.y = y; return true; }
    return false;
}

// Target function: unshadowed luminance of the light sample's contribution (lights.glsl).
float targetPdf(uint y, vec3 x, vec3 n, uint mat) { return lightTarget(y, x, n, mat); }

// ---------------------------------------------------------------- ReSTIR GI
// Sample = a secondary hit point x (or a far point for sky), its normal n,
// and L = radiance leaving x towards the pixel's visible point.
struct GIRes { vec3 x; vec3 n; vec3 L; float wSum; float M; float W; };

GIRes unpackGI(vec4 a, vec4 b, vec4 c) { return GIRes(a.xyz, b.xyz, c.xyz, c.w, b.w, a.w); }
GIRes loadGICur(uint i)  { return unpackGI(giCur[3u * i], giCur[3u * i + 1u], giCur[3u * i + 2u]); }
GIRes loadGIPrev(uint i) { return unpackGI(giPrev[3u * i], giPrev[3u * i + 1u], giPrev[3u * i + 2u]); }
void storeGICur(uint i, GIRes r)  { giCur[3u * i] = vec4(r.x, r.W); giCur[3u * i + 1u] = vec4(r.n, r.M); giCur[3u * i + 2u] = vec4(r.L, r.wSum); }
void storeGIPrev(uint i, GIRes r) { giPrev[3u * i] = vec4(r.x, r.W); giPrev[3u * i + 1u] = vec4(r.n, r.M); giPrev[3u * i + 2u] = vec4(r.L, r.wSum); }

// Target function at receiver (x0, n0): luminance of the cosine-weighted incoming radiance.
float giTarget(GIRes r, vec3 x0, vec3 n0) {
    return lum(r.L) * max(dot(n0, normalize(r.x - x0)), 0.0);
}

bool updateGI(inout GIRes r, GIRes s, float w, float u) {
    r.wSum += w;
    if (w > 0.0 && u * r.wSum < w) { r.x = s.x; r.n = s.n; r.L = s.L; return true; }
    return false;
}
#endif
