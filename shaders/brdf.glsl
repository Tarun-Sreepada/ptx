// Surface shading model shared by every lighting pass:
//   base layer = (1 - F) Lambert diffuse + anisotropic GGX specular (+ Charlie sheen), thin-walled transmission,
//   under a GGX clearcoat that takes its Fresnel share off everything beneath.
// Lighting is stored demodulated: radiance / surfAlbedo() (see the end of this file). The denoisers filter the
// smooth quotient and the composite multiplies the albedo back in, so texture detail is never blurred.
#ifndef BRDF_GLSL
#define BRDF_GLSL
#ifndef PI
#define PI 3.14159265359
#endif

#define MAT_TERRAIN 0u
#define MAT_GRASS   1u
#define MAT_ROCK    2u

struct Surf {
    vec3 base;      // base colour (albedo for dielectrics, F0 for metals)
    vec3 n, T;      // shading normal, unit tangent (anisotropy direction)
    float rough, metal, spec, trans, cc, ccRough, sheen, aniso, ao;
    float ior;      // index of refraction (transmission)
    float transl;   // diffuse transmission: light through thin leaves / blades from behind (KHR_materials_diffuse_transmission)
    uint mat;       // MAT_TERRAIN / GRASS / ROCK / MODEL
};

Surf procSurf(vec3 base, vec3 n, uint mat) {  // terrain, grass, rocks: the old fixed lobes as material parameters
    Surf s;
    s.base = base; s.n = n; s.mat = mat;
    vec3 t = abs(n.y) < 0.99 ? cross(vec3(0, 1, 0), n) : cross(vec3(1, 0, 0), n);
    s.T = normalize(t);
    s.rough = mat == MAT_GRASS ? 0.32 : (mat == MAT_TERRAIN ? 1.0 : 0.55);
    s.spec = mat == MAT_GRASS ? 0.7 : (mat == MAT_TERRAIN ? 0.0 : 0.25);
    s.metal = 0.0; s.trans = 0.0; s.cc = 0.0; s.ccRough = 0.03; s.sheen = 0.0; s.aniso = 0.0; s.ao = 1.0; s.ior = 1.5;
    s.transl = mat == MAT_GRASS ? 0.6 : 0.0;
    return s;
}

vec3 surfF0(Surf s) { return mix(vec3(0.04 * s.spec), s.base, s.metal); }
vec3 surfDiffuseColor(Surf s) { return s.base * (1.0 - s.metal) * (1.0 - s.trans); }
// KHR_materials_anisotropy: alpha along the tangent widens with strength^2, alpha along the bitangent stays
vec2 surfAlpha(Surf s) {
    float a = max(s.rough * s.rough, 0.0025);
    return vec2(mix(a, 1.0, s.aniso * s.aniso), a);
}
float ccAlpha(Surf s) { return max(s.ccRough * s.ccRough, 0.0025); }

// ---------------------------------------------------------------- GGX (local frame: x = T, y = B, z = n)
float ggxD(vec3 h, vec2 a) {
    float k = h.x * h.x / (a.x * a.x) + h.y * h.y / (a.y * a.y) + h.z * h.z;
    return 1.0 / (PI * a.x * a.y * k * k);
}
float ggxLambda(vec3 w, vec2 a) {
    float z2 = max(w.z * w.z, 1e-8);
    return 0.5 * (-1.0 + sqrt(1.0 + (a.x * a.x * w.x * w.x + a.y * a.y * w.y * w.y) / z2));
}
float ggxG1(vec3 w, vec2 a) { return 1.0 / (1.0 + ggxLambda(w, a)); }
float ggxG2(vec3 v, vec3 l, vec2 a) { return 1.0 / (1.0 + ggxLambda(v, a) + ggxLambda(l, a)); }
vec3 fresnel(vec3 F0, float c) { return F0 + (1.0 - F0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0); }
float fresnel1(float F0, float c) { return F0 + (1.0 - F0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0); }

// Visible-normal sampling (Heitz 2018), v in the local frame (z up).
vec3 sampleVNDF(vec3 v, vec2 a, float u1, float u2) {
    vec3 vh = normalize(vec3(a.x * v.x, a.y * v.y, v.z));
    float l2 = vh.x * vh.x + vh.y * vh.y;
    vec3 t1 = l2 > 0.0 ? vec3(-vh.y, vh.x, 0.0) * inversesqrt(l2) : vec3(1, 0, 0);
    vec3 t2 = cross(vh, t1);
    float r = sqrt(u1), phi = 2.0 * PI * u2;
    float p1 = r * cos(phi), p2 = r * sin(phi);
    float sh = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - sh) * sqrt(max(1.0 - p1 * p1, 0.0)) + sh * p2;
    vec3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(vec3(a.x * nh.x, a.y * nh.y, max(nh.z, 1e-6)));
}

// Charlie sheen (Estevez & Kulla 2017) with the Neubelt visibility term, sheen roughness 0.5.
float sheenTerm(float nh, float nv, float nl) {
    const float inv = 1.0 / 0.25;
    float sin2 = max(1.0 - nh * nh, 0.0);
    float D = (2.0 + inv) * pow(sin2, 0.5 * inv) / (2.0 * PI);
    return D / (4.0 * (nl + nv - nl * nv));
}

// Average specular reflectance for a view angle (DLSS-RR integration guide appendix, Ray Tracing Gems ch. 32).
vec3 envBRDFApprox2(vec3 F0, float alpha, float NoV) {
    NoV = abs(NoV);
    vec4 X = vec4(1.0, NoV, NoV * NoV, NoV * NoV * NoV);
    vec4 Y = vec4(1.0, alpha, alpha * alpha, alpha * alpha * alpha);
    vec2 m1 = vec2(0.99044 * X.x - 1.28514 * X.y, 1.29678 * X.x - 0.755907 * X.y);
    vec3 m2 = vec3(1.0 * X.x + 2.92338 * X.y + 59.4188 * X.w, 20.3225 * X.x - 27.0302 * X.y + 222.592 * X.w,
                   121.563 * X.x + 626.13 * X.y + 316.627 * X.w);
    vec2 m3 = vec2(0.0365463 * X.x + 3.32707 * X.y, 9.0632 * X.x - 9.04756 * X.y);
    vec3 m4 = vec3(1.0 * X.x + 3.59685 * X.z - 1.36772 * X.w, 9.04401 * X.x - 16.3174 * X.z + 9.22949 * X.w,
                   5.56589 * X.x + 19.7886 * X.z - 20.2123 * X.w);
    float bias = dot(m1, Y.xy) / dot(m2, Y.xyw);
    float scale = dot(m3, Y.xy) / dot(m4, Y.xyw);
    bias *= clamp(F0.g * 50.0, 0.0, 1.0);
    return F0 * max(0.0, scale) + max(0.0, bias);
}

// Light transmitted through a blade or leaf picks up a yellow-green tint (chlorophyll passes green, absorbs blue).
const vec3 TRANSLUCENT_TINT = vec3(1.2, 1.5, 0.5);

// The lighting helpers take a material id with the translucency packed above it (bits 8..15 = translucency x 255);
// grass always has its blades' 0.6.
uint matT(uint mat, float transl) { return (mat & 0xFFu) | (uint(clamp(transl, 0.0, 1.0) * 255.0 + 0.5) << 8); }
float translOf(uint m) { return (m & 0xFFu) == MAT_GRASS ? 0.6 : float((m >> 8) & 0xFFu) / 255.0; }
// G-buffer gTan.w: index of refraction (>= 1), or minus the translucency
float packIorTransl(Surf s) { return s.transl > 0.0 ? -s.transl : s.ior; }

// f * cos(theta_l) for light from l, view v (both unit, away from the surface). Not demodulated.
// Grass blades also pass some light through from behind (diffuse only).
vec3 evalBRDF(Surf s, vec3 v, vec3 l) {
    vec3 n = s.n;
    float nl = dot(n, l), nv = max(dot(n, v), 1e-4);
    if (nl <= 0.0) return s.base * (-nl * s.transl / PI) * TRANSLUCENT_TINT;
    vec3 B = cross(n, s.T);
    vec3 h = normalize(v + l);
    vec3 hl = vec3(dot(h, s.T), dot(h, B), dot(h, n));
    vec3 vl = vec3(dot(v, s.T), dot(v, B), nv), ll = vec3(dot(l, s.T), dot(l, B), nl);
    float vh = max(dot(v, h), 0.0);
    vec3 F = fresnel(surfF0(s), vh);
    vec2 a = surfAlpha(s);
    vec3 r = (1.0 - F) * surfDiffuseColor(s) * (nl / PI);
    if (s.spec > 0.0 || s.metal > 0.0) r += F * (ggxD(hl, a) * ggxG2(vl, ll, a) / (4.0 * nv));
    if (s.sheen > 0.0) r += s.sheen * s.base * sheenTerm(hl.z, nv, nl) * nl;
    if (s.cc > 0.0) {
        float ac = ccAlpha(s);
        float Fc = fresnel1(0.04, vh);
        r *= 1.0 - s.cc * fresnel1(0.04, nv);
        r += s.cc * Fc * ggxD(hl, vec2(ac)) * ggxG2(vl, ll, vec2(ac)) / (4.0 * nv);
    }
    return r;
}

// ---------------------------------------------------------------- demodulation
// Directional albedo: the factor lighting is divided by when stored and multiplied by in the composite.
vec3 surfSpecAlbedo(Surf s, float nv) {
    vec2 a = surfAlpha(s);
    vec3 r = envBRDFApprox2(surfF0(s), sqrt(a.x * a.y), nv);
    if (s.cc > 0.0) r = r * (1.0 - s.cc * fresnel1(0.04, nv)) + s.cc * envBRDFApprox2(vec3(0.04), ccAlpha(s), nv);
    return r;
}
vec3 surfAlbedo(Surf s, float nv) {
    vec3 spec = surfSpecAlbedo(s, nv);
    float ks = max(spec.r, max(spec.g, spec.b));
    vec3 a = surfDiffuseColor(s) * (1.0 - ks) + spec + s.base * s.trans * (1.0 - ks) + s.sheen * s.base * 0.12;
    return max(a, vec3(0.01));
}
// Surfaces whose specular / clearcoat / transmission lobes get their own traced indirect sample (gi.comp);
// on the rest indirect light is reflected as if diffuse by the whole albedo.
bool surfGlossy(Surf s) { return s.metal > 0.1 || s.trans > 0.01 || s.cc > 0.01 || (s.rough < 0.3 && s.spec > 0.9); }
// MIS share of emitters seen through the glossy lobes that the glossy path (BRDF sampling) takes; light sampling
// (ReSTIR DI) takes the rest. Near-mirrors: the BRDF sample, rough lobes: the light sample.
float glossyEmitWeight(Surf s) {
    if (!surfGlossy(s)) return 0.0;
    vec2 a = surfAlpha(s);
    return 1.0 - smoothstep(0.02, 0.3, sqrt(a.x * a.y));
}
// f * cos split into the diffuse part (diff) and the rest (returned total).
vec3 evalBRDFSplit(Surf s, vec3 v, vec3 l, out vec3 diff) {
    vec3 t = evalBRDF(s, v, l);
    // the diffuse lobe alone: same Fresnel split as evalBRDF, no specular / clearcoat terms
    float nl = dot(s.n, l);
    if (nl <= 0.0) { diff = t; return t; }
    vec3 h = normalize(v + l);
    vec3 F = fresnel(surfF0(s), max(dot(v, h), 0.0));
    diff = (1.0 - F) * surfDiffuseColor(s) * (nl / PI);
    if (s.cc > 0.0) diff *= 1.0 - s.cc * fresnel1(0.04, max(dot(s.n, v), 1e-4));
    return t;
}
// Share of the albedo that the diffuse indirect estimate lights.
vec3 surfDiffuseShare(Surf s, float nv) {
    if (!surfGlossy(s)) return vec3(1);
    vec3 spec = surfSpecAlbedo(s, nv);
    float ks = max(spec.r, max(spec.g, spec.b));
    return (surfDiffuseColor(s) * (1.0 - ks) + s.sheen * s.base * 0.12) / surfAlbedo(s, nv);
}
#endif
