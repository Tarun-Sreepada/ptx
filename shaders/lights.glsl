// Many-light sampling. Light index space: [0, NL) point lights (lights[], NL = F.res.w), then [NL, NL + NT) emissive
// triangles (triLights[], NT = F.lightInfo.x). A light BVH over both (lighttree.h) picks one light in proportion to
// its estimated contribution at the shading point. A light sample packs the light index (16 bits) with the
// position on a triangle quantised to 8 + 8 bits of uv, so ReSTIR reservoirs can carry area light samples.
#ifndef LIGHTS_GLSL
#define LIGHTS_GLSL

layout(binding = 80, std430) readonly buffer TriLights { vec4 triLights[]; };  // per tri: (v0, Le.r) (v1, Le.g) (v2, Le.b) (n, area)
layout(binding = 81, std430) readonly buffer LightTree { vec4 ltree[]; };     // per node: (min, power) (max, child bits)
layout(binding = 82, std430) readonly buffer BlueNoise { vec4 bnTex[]; };      // 64 x 64, 4 channels

layout(binding = 87, std430) readonly buffer LightAlias { vec4 lalias[]; };    // per light: (keep prob, alias bits, pdf, -)

const uint LT_LEAF = 0x80000000u;
const uint NO_LIGHT = 0xFFFFFFFFu;

uint lightCount() { return F.res.w + F.lightInfo.x; }

// ---------------------------------------------------------------- blue noise
// Spatio-temporal blue noise: a 64 x 64 tile, 4 channels per texel; dimension d uses channel d % 4 of a tile offset
// by d / 4, shifted every frame by the golden ratio (keeps the blue-noise spectrum in screen space, low discrepancy
// in time).
float bn(ivec2 p, uint dim) {
    uint g = dim >> 2u;
    ivec2 q = (p + ivec2(int(g * 23u), int(g * 41u))) & 63;
    float v = bnTex[q.y * 64 + q.x][dim & 3u];
    return fract(v + float(F.res.z & 1023u) * 0.61803398875 * float(g + 1u));
}
vec2 bn2(ivec2 p, uint dim) { return vec2(bn(p, dim), bn(p, dim + 1u)); }

// ---------------------------------------------------------------- light samples
uint packLight(uint li, vec2 uv) { return li | (min(uint(uv.x * 256.0), 255u) << 16) | (min(uint(uv.y * 256.0), 255u) << 24); }
uint sampleLightIndex(uint y) { return y & 0xFFFFu; }

struct LightPoint {
    vec3 p;       // position (point light centre / point on the triangle)
    vec3 Le;      // point: intensity; triangle: emitted radiance
    vec3 n;       // triangle normal
    float area;   // 0 for point lights
    float radius; // point lights: orb radius (shadow rays stop short of it)
};

LightPoint lightPoint(uint y) {
    LightPoint l;
    uint li = sampleLightIndex(y);
    uint NL = F.res.w;
    if (li < NL) {
        vec4 a = lights[2u * li];
        l.p = a.xyz; l.radius = a.w;
        l.Le = lights[2u * li + 1u].rgb;
        l.n = vec3(0); l.area = 0.0;
        return l;
    }
    uint t = (li - NL) * 4u;
    vec4 a = triLights[t], b = triLights[t + 1u], c = triLights[t + 2u], d = triLights[t + 3u];
    vec2 u = (vec2(float((y >> 16) & 255u), float(y >> 24)) + 0.5) / 256.0;
    float su = sqrt(u.x);
    float b0 = 1.0 - su, b1 = u.y * su;  // uniform over the triangle
    l.p = a.xyz * b0 + b.xyz * b1 + c.xyz * (1.0 - b0 - b1);
    l.Le = vec3(a.w, b.w, c.w);
    l.n = d.xyz; l.area = d.w; l.radius = 0.0;
    return l;
}

// ---------------------------------------------------------------- light tree
float nodeImportance(uint i, vec3 x, vec3 n, bool twoSided) {
    vec4 a = ltree[2u * i], b = ltree[2u * i + 1u];
    vec3 c = 0.5 * (a.xyz + b.xyz), e = 0.5 * (b.xyz - a.xyz);
    vec3 d = c - x;
    if (!twoSided && dot(d, n) + dot(e, abs(n)) < 0.0) return 0.0;  // the whole box is behind the surface
    float d2 = max(dot(d, d), dot(e, e));                            // inside / next to the box: its size bounds it
    return a.w / max(d2, 1e-4);
}

// Walks the tree with one uniform number (rescaled at every level). pdf = probability of the returned light.
uint sampleLightTree(vec3 x, vec3 n, bool twoSided, float u, out float pdf) {
    pdf = 0.0;
    if (F.lightInfo.y == 0u) return NO_LIGHT;
    uint i = 0u;
    float p = 1.0;
    for (int depth = 0; depth < 48; depth++) {
        uint ch = floatBitsToUint(ltree[2u * i + 1u].w);
        if ((ch & LT_LEAF) != 0u) { pdf = p; return ch & ~LT_LEAF; }
        uint L = i + 1u, R = ch;
        float iL = nodeImportance(L, x, n, twoSided), iR = nodeImportance(R, x, n, twoSided);
        float s = iL + iR;
        if (s <= 0.0) return NO_LIGHT;
        float pL = iL / s;
        if (u < pL) { u = min(u / pL, 0.99999); p *= pL; i = L; }
        else { u = min((u - pL) / max(1.0 - pL, 1e-6), 0.99999); p *= 1.0 - pL; i = R; }
    }
    return NO_LIGHT;
}

// O(1) power-proportional pick (ReSTIR candidates: the resampling adds the spatial importance).
uint sampleLightAlias(float u, out float pdf) {
    uint n = lightCount();
    pdf = 0.0;
    if (n == 0u || F.lightInfo.y == 0u) return NO_LIGHT;
    float x = u * float(n);
    uint i = min(uint(x), n - 1u);
    vec4 e = lalias[i];
    if (x - float(i) >= e.x) i = floatBitsToUint(e.y);
    pdf = lalias[i].z;
    return pdf > 0.0 ? i : NO_LIGHT;
}

// Source pdf of a packed light sample: tree pdf (x 1 / area for a point on a triangle).
float lightSourcePdf(float treePdf, LightPoint l) { return l.area > 0.0 ? treePdf / l.area : treePdf; }

// Unshadowed luminance reaching (x, n) from light sample y (ReSTIR target function). Triangles: area measure.
float lightTarget(uint y, vec3 x, vec3 n, uint mat) {
    if (sampleLightIndex(y) >= lightCount()) return 0.0;
    LightPoint l = lightPoint(y);
    vec3 d = l.p - x;
    float d2 = max(dot(d, d), 1e-4);
    vec3 wi = d * inversesqrt(d2);
    float c = dot(n, wi);
    if (c < 0.0) c = -c * translOf(mat);
    if (l.area > 0.0) c *= abs(dot(l.n, wi));
    return lum(l.Le) * c / d2;
}

// Radiance factor of sample y toward x: L(wi) * G (area) or I / d^2 (point), without the BRDF. Also the shadow ray.
vec3 lightIncident(LightPoint l, vec3 x, out vec3 wi, out float dist) {
    vec3 d = l.p - x;
    dist = length(d);
    wi = d / max(dist, 1e-6);
    float d2 = max(dist * dist, 1e-4);
    if (l.area > 0.0) return l.Le * abs(dot(l.n, wi)) / d2;
    return l.Le / d2;
}
float shadowLength(LightPoint l, float dist) { return l.area > 0.0 ? dist * 0.999 - 2.0 * EPS : dist - l.radius; }

// Next-event estimation at a diffuse path vertex: one light from the tree, demodulated Lambert (x albedo later).
vec3 treeNEE(vec3 x, vec3 n, uint mat, inout uint s) {
    float pt;
    uint li = sampleLightTree(x, n, translOf(mat) > 0.0, rnd(s), pt);
    if (li == NO_LIGHT || pt <= 0.0) return vec3(0);
    LightPoint l = lightPoint(packLight(li, vec2(rnd(s), rnd(s))));
    vec3 wi;
    float dist;
    vec3 Li = lightIncident(l, x, wi, dist);
    float side;
    float c = lambert(n, wi, mat, side);
    if (c <= 0.0 || !visible(x + n * (EPS * side), wi, shadowLength(l, dist))) return vec3(0);
    return Li * c / (PI * lightSourcePdf(pt, l));
}

// Weight of emission found by a path that hit an emitter (the light list and next-event estimation also reach it).
// b: bounce index, w0: weight at the first hit (0 for diffuse paths: ReSTIR DI covers it; the glossy MIS share for
// glossy paths). Probe bakes: direct light from emitters is ReSTIR DI's at run time, so their first hit skips it
// too (later bounces count it: the bake's next-event estimation only has point lights). Without the full list
// (more emitters than its cap) every hit counts.
float emissionWeight(uint b, float w0) {
    if (bakeVol >= 0) return b == 0u ? 0.0 : 1.0;
    if (F.lightInfo.z == 0u) return 1.0;
    return b == 0u ? w0 : 0.0;
}
#endif
