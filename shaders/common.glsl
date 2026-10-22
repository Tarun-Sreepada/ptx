// Shared bindings + helpers for every compute pass. All passes use one descriptor set layout.
#extension GL_EXT_ray_query : require

#define PI 3.14159265359

#define FLAG_RESTIR          1u
#define FLAG_RESTIR_TEMPORAL 2u
#define FLAG_RESTIR_SPATIAL  4u
#define FLAG_ACCUM           8u
#define FLAG_GI_RESTIR      16u
#define FLAG_GI_TEMPORAL    32u
#define FLAG_GI_SPATIAL     64u
#define FLAG_NO_GI         128u   // direct light only (A/B view of what GI adds)

layout(binding = 0) uniform accelerationStructureEXT tlas;

#include "frame.glsl"
#include "brdf.glsl"
#include "material.glsl"
#include "model.glsl"
#include "ground.glsl"
#include "wind.glsl"

layout(binding = 2, std430) buffer TerrainV { float tv[]; };            // pos.xyz, normal.xyz (written by terrain.comp)
layout(binding = 3, std430) readonly buffer TerrainI { uint ti[]; };
layout(binding = 4, std430) buffer GrassV { float gv[]; };              // pos.xyz, 7 verts per blade
layout(binding = 5, std430) readonly buffer GrassI { uint gix[]; };
layout(binding = 6, std430) buffer BladeSeed { uint bladeSeed[]; };     // stable per-blade id (for colour)
layout(binding = 7, std430) readonly buffer Lights { vec4 lights[]; };  // pos+radius, intensity rgb
layout(binding = 8, std430) buffer ResCur  { vec4 resCur[]; };
layout(binding = 9, std430) buffer ResPrev { vec4 resPrev[]; };
layout(binding = 25, std430) buffer GICur  { vec4 giCur[]; };  // 3 per pixel: ReSTIR GI reservoirs
layout(binding = 26, std430) buffer GIPrev { vec4 giPrev[]; };
layout(binding = 27, std430) buffer GrassCounter {
    uint grassCount;                      // blades emitted this frame
    uint drawIndexCount, drawInstanceCount, drawFirstIndex; // VkDrawIndexedIndirectCommand at offset 4
    int drawVertexOffset;
    uint drawFirstInstance;
};

layout(binding = 28, rgba16f) uniform image2D skyLUT;       // atmosphere radiance per unit sun irradiance
layout(binding = 29, std430) readonly buffer RockV { float rv[]; }; // pos.xyz, normal.xyz
layout(binding = 30, std430) readonly buffer RockI { uint ri[]; };
layout(binding = 10, r32f) uniform image2D gDepth;        // hit distance along the camera ray (-1 = sky)
layout(binding = 11, rgba16f) uniform image2D gNorm;      // normal, w = material
layout(binding = 12, rgba16f) uniform image2D gAlbedo;
layout(binding = 13, rgba16f) uniform image2D gEmis;       // directly visible sky / light orbs
layout(binding = 14, r32f) uniform image2D prevDepth;
layout(binding = 15, rgba16f) uniform image2D prevNorm;
layout(binding = 16, rgba16f) uniform image2D directImg;   // demodulated direct light
layout(binding = 17, rgba16f) uniform image2D indirectImg; // demodulated indirect light
layout(binding = 18, rgba16f) uniform image2D accumColor;  // rgb, a = variance
layout(binding = 19, rgba32f) uniform image2D accumMoments;// m1, m2, history length
layout(binding = 20, rgba16f) uniform image2D histColor;
layout(binding = 21, rgba32f) uniform image2D histMoments;
layout(binding = 22, rgba16f) uniform image2D pingA;
layout(binding = 23, rgba16f) uniform image2D pingB;
layout(binding = 24, rgba8) uniform writeonly image2D outImg;
// material G-buffer (render resolution): gAlbedo = base colour + ao, gMat = roughness, metalness, specular, transmission,
// gMat2 = clearcoat, clearcoat roughness, sheen, anisotropy, gTan = tangent (anisotropy direction)
layout(binding = 54, rgba16f) uniform image2D gMat;
layout(binding = 55, rgba16f) uniform image2D gMat2;
layout(binding = 56, rgba16f) uniform image2D gTan;
layout(binding = 57, rgba16f) uniform image2D specImg;    // demodulated glossy indirect (specular, clearcoat, transmission)
layout(binding = 58, rgba16f) uniform image2D hdrImg;     // the HDR scene at output resolution, before post (post.comp)
layout(binding = 60, rgba16f) uniform image2D fogImg;     // volumetric fog, half resolution: in-scatter, transmittance
layout(binding = 61, rgba16f) uniform image2D fogHist;

layout(push_constant) uniform PC { uint a, b, c, d; } pc;

// ---------------------------------------------------------------- utils
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

uint pcg(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float rnd(inout uint s) { return float(pcg(s)) * (1.0 / 4294967296.0); }
uint seedFor(ivec2 p, uint salt) {
    uint s = uint(p.x) * 1973u + uint(p.y) * 9277u + F.res.z * 26699u + salt * 104729u;
    pcg(s);
    return s;
}


void onb(vec3 n, out vec3 t, out vec3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float bb = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * bb, -s * n.x);
    b = vec3(bb, s + n.y * n.y * a, -n.y);
}
vec3 cosineSampleU(vec3 n, vec2 u) {
    float u1 = u.x, u2 = u.y;
    float r = sqrt(u1), phi = 2.0 * PI * u2;
    vec3 t, b; onb(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}
vec3 cosineSample(vec3 n, inout uint s) {
    float u1 = rnd(s), u2 = rnd(s);
    float r = sqrt(u1), phi = 2.0 * PI * u2;
    vec3 t, b; onb(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}
vec3 sampleConeU(vec3 axis, float cosMax, vec2 u) {
    float ct = mix(1.0, cosMax, u.x);
    float st = sqrt(max(0.0, 1.0 - ct * ct));
    float phi = 2.0 * PI * u.y;
    vec3 t, b; onb(axis, t, b);
    return normalize(t * (cos(phi) * st) + b * (sin(phi) * st) + axis * ct);
}
vec3 sampleCone(vec3 axis, float cosMax, inout uint s) {
    float ct = mix(1.0, cosMax, rnd(s));
    float st = sqrt(max(0.0, 1.0 - ct * ct));
    float phi = 2.0 * PI * rnd(s);
    vec3 t, b; onb(axis, t, b);
    return normalize(t * (cos(phi) * st) + b * (sin(phi) * st) + axis * ct);
}

// ---------------------------------------------------------------- camera / sky
vec3 cameraRay(ivec2 p) {
    vec2 uv = (vec2(p) + 0.5 + F.jitter.xy) / vec2(F.res.xy);
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return normalize(F.camFwd.xyz + ndc.x * F.camUp.w * F.camRight.w * F.camRight.xyz + ndc.y * F.camUp.w * F.camUp.xyz);
}

vec3 prevCameraRay(ivec2 p) {
    vec2 uv = (vec2(p) + 0.5 + F.jitter.zw) / vec2(F.res.xy);
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return normalize(F.prevFwd.xyz + ndc.x * F.prevUp.w * F.prevRight.w * F.prevRight.xyz + ndc.y * F.prevUp.w * F.prevUp.xyz);
}

// World position reconstructed from depth: xyz = position, w = distance (-1 = sky)
vec4 loadPos(ivec2 p) {
    float t = imageLoad(gDepth, p).r;
    return vec4(F.camPos.xyz + cameraRay(p) * max(t, 0.0), t);
}
vec4 loadPrevPos(ivec2 p) {
    float t = imageLoad(prevDepth, p).r;
    return vec4(F.prevCamPos.xyz + prevCameraRay(p) * max(t, 0.0), t);
}

// Continuous pixel position of world point P in the previous frame's camera.
bool prevPixel(vec3 P, out vec2 pp) {
    vec3 d = P - F.prevCamPos.xyz;
    float z = dot(d, F.prevFwd.xyz);
    if (z <= 1e-3) return false;
    float x = dot(d, F.prevRight.xyz) / (z * F.prevUp.w * F.prevRight.w);
    float y = dot(d, F.prevUp.xyz) / (z * F.prevUp.w);
    pp = vec2(x * 0.5 + 0.5, 0.5 - y * 0.5) * vec2(F.res.xy) - 0.5 - F.jitter.zw;
    return true;
}
bool inScreen(ivec2 q) { return q.x >= 0 && q.y >= 0 && q.x < int(F.res.x) && q.y < int(F.res.y); }

bool similarPrev(ivec2 q, vec3 P, vec3 n, uint mat) {
    if (!inScreen(q)) return false;
    vec4 qp = loadPrevPos(q);
    if (qp.w < 0.0) return false;
    vec4 qn = imageLoad(prevNorm, q);
    if (uint(qn.w + 0.5) != mat) return false;
    float dist = length(P - F.camPos.xyz);
    if (abs(dot(P - qp.xyz, n)) > 0.02 + 0.01 * dist) return false;
    if (dot(n, qn.xyz) < (mat == MAT_GRASS ? 0.5 : 0.9)) return false;
    return true;
}

#include "atmosphere.glsl"
#include "probes.glsl"

// Sky radiance: physical atmosphere LUT (sky.comp) scaled by sun illuminance, plus a faint night sky.
vec3 skyAtmo(vec3 d) {
    float el = asin(clamp(d.y, -1.0, 1.0));
    vec2 uv = vec2(atan(d.x, d.z) / (2.0 * PI) + 0.5, skyV(el));
    vec2 st = uv * vec2(SKY_W, SKY_H) - 0.5;
    ivec2 i0 = ivec2(floor(st));
    vec2 f = st - vec2(i0);
    ivec2 a0 = ivec2((i0.x % SKY_W + SKY_W) % SKY_W, clamp(i0.y, 0, SKY_H - 1));
    ivec2 a1 = ivec2((a0.x + 1) % SKY_W, clamp(i0.y + 1, 0, SKY_H - 1));
    vec3 c = mix(mix(imageLoad(skyLUT, a0).rgb, imageLoad(skyLUT, ivec2(a1.x, a0.y)).rgb, f.x),
                 mix(imageLoad(skyLUT, ivec2(a0.x, a1.y)).rgb, imageLoad(skyLUT, a1).rgb, f.x), f.y);
    float overcast = F.clouds.w > 0.5 ? 1.0 - 0.45 * F.clouds.x * F.clouds.x : 1.0;  // cloud cover greys out the blue sky
    return c * F.params2.z * F.sunColor.w * overcast + vec3(0.002, 0.003, 0.008);
}
// Far field: past the terrain mesh (F.terrainP extent, set by the view distance) a downward ray that missed
// everything would see black. Instead march the large-scale terrain noise (the same function, coarse terms only)
// out to 40 km, shade with the ground colour, sun + sky light (no shadows), and fade into the aerial haze.
// Cheap: only rays that leave the mesh pay, ~28 steps.
float farHeight(vec2 p) {
    return 2.2 * tnoise(p.x * 0.022 + 10.0, p.y * 0.022) + 0.7 * tnoise(p.x * 0.07, p.y * 0.07 + 5.0) - 1.5;
}
// Forest cover past the tree range, from the same forest noise the placement uses (main.cpp candidate): the far
// field gets a canopy layer (colour + ~9 m of height) so forests continue to the horizon as low-detail LOD.
float forestCover(vec2 p) {
    float f = tnoise(p.x * 0.0045 + 31.0, p.y * 0.0045 - 7.0) * 0.7 + tnoise(p.x * 0.017, p.y * 0.017 + 11.0) * 0.3;
    return clamp((smoothstep(0.5, 0.66, f) * 0.8 + 0.025) * F.water2.z * 1.6, 0.0, 1.0);  // trees per cell -> canopy cover
}
float farHeightC(vec2 p, out float cov) {
    cov = forestCover(p);
    float crowns = 0.6 + 0.4 * tnoise(p.x * 0.11, p.y * 0.11 + 3.0);  // uneven crown tops
    return farHeight(p) + cov * 9.0 * crowns;
}
float cloudShadow(vec3 x);  // clouds.glsl
vec3 farGround(vec3 o, vec3 d, out bool hit) {
    hit = false;
    float horiz = length(d.xz);
    if (d.y >= 0.0 || horiz < 1e-4) return vec3(0);
    float ext = F.terrainP.z + F.terrainP.w;
    float t = max(0.9 * ext / horiz, 0.0), tEnd = 40000.0;
    float prevT = t, prevGap = 1e30;
    for (int i = 0; i < 28 && t < tEnd; i++) {
        vec3 q = o + d * t;
        float cv;
        float gap = q.y - farHeightC(q.xz, cv);
        if (gap < 0.0) {
            t = prevT + (t - prevT) * prevGap / max(prevGap - gap, 1e-4);  // secant
            hit = true;
            break;
        }
        prevT = t; prevGap = gap;
        t = t * 1.22 + 20.0;
    }
    if (!hit) {  // the mean ground plane
        float tp = (o.y + 1.5) / -d.y;
        if (tp > tEnd * 1.5) return vec3(0);
        t = tp;
        hit = true;
    }
    vec3 P = o + d * t;
    float e = max(2.0, t * 0.004), cov, c2;
    farHeightC(P.xz, cov);
    vec3 n = normalize(vec3(farHeightC(P.xz - vec2(e, 0), c2) - farHeightC(P.xz + vec2(e, 0), c2), 2.0 * e,
                            farHeightC(P.xz - vec2(0, e), c2) - farHeightC(P.xz + vec2(0, e), c2)));
    vec3 alb = terrainAlbedo(P.xz, 1.0);
    // canopy: dark foliage, broadleaf / conifer patches as the variants are chosen; crowns self-shadow (darker
    // than the ground at the same sun angle)
    vec3 canopy = mix(vec3(0.045, 0.075, 0.03), vec3(0.025, 0.05, 0.03), step(0.5, tnoise(P.x * 0.01 + 50.0, P.y * 0.01)));
    alb = mix(alb, canopy * (0.75 + 0.5 * tnoise(P.x * 0.3, P.y * 0.3)), cov);
    // cloud shadows: the far field darkens under the same clouds as the near ground, so the illusion holds
    vec3 col = alb * (F.sunColor.rgb * max(dot(n, F.sunDir.xyz), 0.0) * cloudShadow(P) / PI + skyAtmo(vec3(0, 1, 0)) * (0.5 + 0.5 * n.y));
    // aerial perspective as composite.comp does it for geometry (sky pixels skip that pass)
    float h = max(P.y + VIEW_H, 0.0);
    vec3 T = exp(-(BR * exp(-h / HR) + vec3(BME * exp(-h / HM))) * F.params2.w * t);
    return col * T + skyAtmo(normalize(vec3(d.x, 0.03, d.z))) * (1.0 - T);
}
vec3 sky(vec3 d) {
    if (d.y < 0.0) {
        bool hit;
        vec3 g = farGround(F.camPos.xyz, d, hit);
        if (hit) return g;
    }
    return skyAtmo(d);
}

#include "clouds.glsl"
// Clear sky + sun disc, then the cloud layer over both (clouds hide the sun).
vec3 skyWithSun(vec3 d) {
    float jit = float(hashu(floatBitsToUint(d.x * 7.0 + d.z) ^ hashu(floatBitsToUint(d.y)) ^ (F.res.z * 0x9e3779b9u))) / 4294967295.0;
    float T;
    vec3 cl = cloudMarch(F.camPos.xyz, d, jit, T);
    vec3 c = sky(d) * T + cl;
    // sun disc: ~500x the sky, so even thin cloud must kill it; scale by T^3 (optical depth x3) so a cloud edge
    // reads as a bright glow rather than the bare disc on top of the cloud
    if (dot(d, F.sunDir.xyz) > F.sunDir.w)
        c += min(F.sunColor.rgb / (2.0 * PI * (1.0 - F.sunDir.w)), vec3(500.0)) * T * T * T;
    return c;
}

// ---------------------------------------------------------------- tracing
// s: the full surface (shading normal, material lobes); n / albedo repeat s.n / s.base for the diffuse-only users.
// back: hit a surface from behind; trans: glass (thin-walled transmission) the paths pass through.
struct Hit { bool hit; float t; vec3 p; vec3 n; vec3 albedo; vec3 emis; uint mat; bool back; float trans; float ior; Surf s; };

bool glassTri(uint inst, uint prim) {
    return inst >= MAT_MODEL && materials[triMaterialSlot(meshes[inst - MAT_MODEL], prim)].transmission > 0.5;
}

// true = unoccluded. Glass doesn't block (its meshes are non-opaque in the BVH, every other candidate is confirmed).
// Craters are shaded on the exact height (craterRefine) but the terrain mesh can't hold the bowl, so a shadow ray
// from inside one hits the mesh at once (or misses the rim): the hole goes dark / lit as a mesh-shaped circle. Near a
// crater the first stretch is marched against the exact height instead, and the terrain mesh (instance 0) is
// ignored there; past the band where mesh and true surface differ, the BVH takes over. Returns the stretch, or -1
// when the bowl itself blocks the ray.
// exact: march the bowl (sun shadows: sharp, worth it). Otherwise (GI, local lights, fog: soft, and ~10x more rays)
// only lift the origin onto the bowl and ignore the coarse mesh across the band, no march.
float craterSelfShadow(inout vec3 o, vec3 d, float tmax, bool exact) {
    if (F.dynInfo.z == 0u) return 0.0;
    CraterLocal L;
    // reach: a coarse mesh triangle with a vertex on a rim stands a few cm above the ground well past the bowl's
    // reach (1.8 r); points under it went pitch black, so the fix-up covers a mesh cell further (bins hold 1.9 r + 1 m)
    if (!craterLocal(o.xz, 1.0, exact, L)) return 0.0;
    float band = L.depth * 1.6 + 0.6;
    float h0 = localHeight(L, o.xz);
    if (o.y - h0 > band) return 0.0;
    // shading points can sit on the coarse mesh, under the true bowl: start from the true surface
    o.y = max(o.y, h0 + 0.03);
    float tEnd = min(min(tmax, 40.0), d.y > 0.02 ? 2.0 * band / d.y : 40.0);
    // soft rays (GI, local lights, fog): no march, but the coarse terrain mesh is ignored for the first stretch out of
    // the bowl (it can't follow it and would shut the crater in: pitch-black fill light)
    if (!exact) return 0.0;
    tEnd = min(tEnd, 12.0);  // the local ground is only trusted near the point (a sun ray leaves a bowl well within this)
    const int N = 16;
    for (int k = 1; k <= N; k++) {
        float t = tEnd * float(k) / float(N);
        vec3 q = o + d * t;
        float g = q.y - localHeight(L, q.xz);
        if (g < -0.03) return -1.0;  // a few cm of tolerance: grazing rays don't flicker on the fitted surface
        if (g > band) return t;
    }
    return tEnd;
}

bool visibleEx(vec3 o, vec3 d, float tmax, bool exact) {
    float skipTerrain = craterSelfShadow(o, d, tmax, exact);
    if (skipTerrain < 0.0) return false;
    // near a crater: the stretch out of the bowl is traced without the terrain mesh (mask 0x01 = everything but
    // terrain, which is 0x02; opaque hits can't be filtered in the candidate loop), the rest with everything
    for (int seg = skipTerrain > 0.0 ? 0 : 1; seg < 2; seg++) {
        float t0 = seg == 0 ? 0.0 : min(skipTerrain, tmax), t1 = seg == 0 ? min(skipTerrain, tmax) : tmax;
        if (t1 <= t0) continue;
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, tlas, gl_RayFlagsTerminateOnFirstHitEXT, seg == 0 ? 0x01u : 0xFFu, o, t0, d, t1);
        while (rayQueryProceedEXT(rq))
            if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT &&
                !glassTri(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false), rayQueryGetIntersectionPrimitiveIndexEXT(rq, false)))
                rayQueryConfirmIntersectionEXT(rq);
        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) return false;
    }
    return true;
}
bool visible(vec3 o, vec3 d, float tmax) { return visibleEx(o, d, tmax, false); }
bool visibleSun(vec3 o, vec3 d) { return visibleEx(o, d, 1e4, true); }  // sharp sun shadows: exact in craters

vec3 grassPos(uint i) { return vec3(gv[3u * i], gv[3u * i + 1u], gv[3u * i + 2u]); }
vec3 terrPos(uint i)  { return vec3(tv[6u * i], tv[6u * i + 1u], tv[6u * i + 2u]); }
vec3 terrNrm(uint i)  { return vec3(tv[6u * i + 3u], tv[6u * i + 4u], tv[6u * i + 5u]); }

// Pixel footprint of camera rays: cone spread angle (radians per pixel).
float pixelSpread() { return 2.0 * F.camUp.w / float(F.res.y); }

// cone: x = footprint width at the ray origin (m), y = spread angle (rad); sets the texture mip of model hits.
// full: evaluate the whole material (camera rays, mirror reflections); otherwise only what a diffuse path vertex
// needs (base colour, interpolated normal, emission, transmission) to keep the bounce loops light.
Hit traceClosest(vec3 o, vec3 d, float tmax, vec2 cone, bool full) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsOpaqueEXT, 0xFF, o, 0.0, d, tmax);
    while (rayQueryProceedEXT(rq)) {}
    Hit h;
    h.hit = false;
    h.back = false;
    h.trans = 0.0;
    h.ior = 1.5;
    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionTriangleEXT) return h;
    h.hit = true;
    h.t = rayQueryGetIntersectionTEXT(rq, true);
    h.p = o + d * h.t;
    uint inst = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
    uint prim = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    vec2 bc = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    vec3 w = vec3(1.0 - bc.x - bc.y, bc.x, bc.y);
    h.mat = min(inst, MAT_MODEL);
    h.emis = vec3(0);
    if (inst >= MAT_MODEL) {
        MeshInfo m = meshes[inst - MAT_MODEL];
        UintBuf ib = UintBuf(m.idx);
        ModelVertex a = modelVertex(m, ib.v[3u * prim]), b = modelVertex(m, ib.v[3u * prim + 1u]), c = modelVertex(m, ib.v[3u * prim + 2u]);
        vec3 nObj = a.n * w.x + b.n * w.y + c.n * w.z;
        if (dot(nObj, nObj) < 1e-12) nObj = cross(b.p - a.p, c.p - a.p);
        mat4x3 w2o = rayQueryGetIntersectionWorldToObjectEXT(rq, true);
        mat3 o2w = mat3(rayQueryGetIntersectionObjectToWorldEXT(rq, true));
        vec3 n = normalize(transpose(mat3(w2o)) * nObj);  // inverse-transpose: correct under non-uniform scale
        h.back = dot(n, d) > 0.0;
        if (h.back) n = -n;
        vec3 dpdu, dpdv;
        triDerivs(a.p, b.p, c.p, a.uv, b.uv, c.uv, dpdu, dpdv);
        // ray cone: footprint width at the hit over the triangle's world / uv area ratio
        vec3 e1 = o2w * (b.p - a.p), e2 = o2w * (c.p - a.p);
        vec2 t1 = b.uv - a.uv, t2 = c.uv - a.uv;
        float worldA = length(cross(e1, e2)), uvA = abs(t1.x * t2.y - t1.y * t2.x);
        float width = cone.x + cone.y * h.t;
        gConeLod = 0.5 * log2(max(uvA, 1e-20) / max(worldA, 1e-20)) + log2(max(width, 1e-7) / max(abs(dot(n, d)), 0.05));
        Material mt = triMaterial(m, prim);
        vec2 uv = a.uv * w.x + b.uv * w.y + c.uv * w.z;
        if (!full) {
            h.n = n;
            h.albedo = materialAlbedo(mt, uv);
            h.emis = mt.emissive;
            if (mt.emisTex != NO_TEX) h.emis *= mtex(mt.emisTex, uv).rgb;
            if (dot(h.emis, h.emis) > 0.0 && emitterOff(h.p)) h.emis = vec3(0);
            h.trans = mt.transmission;
            h.ior = mt.ior;
            h.s = procSurf(h.albedo, n, MAT_MODEL);
            h.s.transl = mt.transl;
            h.mat = matT(MAT_MODEL, mt.transl);
            return h;
        }
        SurfaceMat sm = evalMaterial(mt, uv, n, o2w * dpdu, o2w * dpdv);
        if (dot(sm.n, d) > 0.0) sm.n = normalize(sm.n - d * (dot(sm.n, d) + 0.01));  // normal map facing away
        h.n = sm.n;
        h.albedo = sm.base;
        h.emis = sm.emis;
        if (dot(h.emis, h.emis) > 0.0 && emitterOff(h.p)) h.emis = vec3(0);
        h.trans = sm.trans;
        h.ior = sm.ior;
        h.s.ior = sm.ior;
        h.s.transl = sm.transl;
        h.mat = matT(MAT_MODEL, sm.transl);
        h.s.base = sm.base; h.s.n = sm.n; h.s.T = sm.T; h.s.mat = MAT_MODEL;
        h.s.rough = sm.rough; h.s.metal = sm.metal; h.s.spec = sm.spec; h.s.trans = sm.trans;
        h.s.cc = sm.cc; h.s.ccRough = sm.ccRough; h.s.sheen = sm.sheen; h.s.aniso = sm.aniso; h.s.ao = sm.ao;
        return h;
    } else if (inst == MAT_TERRAIN) {
        uint i0 = ti[3u * prim], i1 = ti[3u * prim + 1u], i2 = ti[3u * prim + 2u];
        h.n = normalize(terrNrm(i0) * w.x + terrNrm(i1) * w.y + terrNrm(i2) * w.z);
        if (full && craterRefine(h.p, d, h.n)) h.t = dot(h.p - o, d);  // exact crater bowl (camera rays)
        float dc = length(h.p.xz - F.camPos.xz);
        h.albedo = terrainAlbedo(h.p.xz, smoothstep(F.grassA.w * 0.4, F.grassA.w, dc));
        vec3 gp = h.p;
        float gr = 1.0, gao = 1.0;
        groundSurface(gp, d, length(h.p - F.camPos.xyz), false, h.n, h.albedo, gr, gao);
        h.back = dot(h.n, d) > 0.0;
        if (h.back) h.n = -h.n;
        h.s = procSurf(h.albedo, h.n, h.mat);
        h.s.rough = gr; h.s.ao = gao; h.s.spec = 0.5;
        return h;
    } else if (inst == MAT_ROCK) {
        uint i0 = ri[3u * prim], i1 = ri[3u * prim + 1u], i2 = ri[3u * prim + 2u];
        vec3 nObj = vec3(rv[6u * i0 + 3u], rv[6u * i0 + 4u], rv[6u * i0 + 5u]) * w.x +
                    vec3(rv[6u * i1 + 3u], rv[6u * i1 + 4u], rv[6u * i1 + 5u]) * w.y +
                    vec3(rv[6u * i2 + 3u], rv[6u * i2 + 4u], rv[6u * i2 + 5u]) * w.z;
        mat4x3 o2w = rayQueryGetIntersectionObjectToWorldEXT(rq, true); // rotation * uniform scale
        h.n = normalize(mat3(o2w) * nObj);
        if (full) {
            h.back = dot(h.n, d) > 0.0;
            if (h.back) h.n = -h.n;
            float rr, rao;
            h.albedo = rockSurface(h.p, h.n, rr, rao);
            if (dot(h.n, d) > 0.0) h.n = normalize(h.n - d * (dot(h.n, d) + 0.01));
            h.s = procSurf(h.albedo, h.n, h.mat);
            h.s.rough = rr; h.s.ao = rao;
            return h;
        }
        h.albedo = rockAlbedo(h.p, h.n);
    } else {
        uint i0 = gix[3u * prim], i1 = gix[3u * prim + 1u], i2 = gix[3u * prim + 2u];
        vec3 v0 = grassPos(i0), v1 = grassPos(i1), v2 = grassPos(i2);
        h.n = normalize(cross(v1 - v0, v2 - v0));
        float t = (float((i0 % 7u) / 2u) * w.x + float((i1 % 7u) / 2u) * w.y + float((i2 % 7u) / 2u) * w.z) / 3.0;
        h.albedo = grassAlbedo(bladeSeed[prim / 5u], t);
        h.back = dot(h.n, d) > 0.0;
        if (h.back) h.n = -h.n;
        h.s = procSurf(h.albedo, h.n, h.mat);
        h.s.rough = grassRough(bladeSeed[prim / 5u]);
        return h;
    }
    h.back = dot(h.n, d) > 0.0;
    if (h.back) h.n = -h.n;
    h.s = procSurf(h.albedo, h.n, h.mat);
    return h;
}

// Diffuse bounce rays: wide cone (a Lambert lobe), so secondary hits read coarse mips.
const float DIFFUSE_SPREAD = 0.35;
Hit traceClosest(vec3 o, vec3 d, float tmax) { return traceClosest(o, d, tmax, vec2(0.0, pixelSpread()), true); }
Hit traceClosest(vec3 o, vec3 d, float tmax, vec2 cone) { return traceClosest(o, d, tmax, cone, false); }

// Diffuse cosine term; grass blades also transmit some light from behind.
// mat: material id with packed translucency (brdf.glsl matT).
float lambert(vec3 n, vec3 l, uint mat, out float side) {
    float c = dot(n, l);
    side = c >= 0.0 ? 1.0 : -1.0;
    if (c >= 0.0) return c;
    return -c * translOf(mat);
}

const float EPS = 0.004;

// Sunlight reaching x under the river's surface: absorbed along the refracted path, focused by the waves into a
// moving caustic net (two flow-advected layers half a period apart, like the surface detail in water.glsl).
float causticNet(vec2 p) {
    float a = 1.0 - abs(vnoise(p) * 2.0 - 1.0), b = 1.0 - abs(vnoise(p * 1.9 + 4.0) * 2.0 - 1.0);
    return pow(a, 6.0) + pow(b, 8.0) * 0.6;
}
vec3 underwaterSun(vec3 x) {
    float depth = waterLevelAt(x.xz) - x.y;
    if (depth <= 0.0) return vec3(1);
    vec3 rd = refract(-F.sunDir.xyz, vec3(0, 1, 0), 1.0 / 1.33);
    float path = depth / max(-rd.y, 0.2);
    vec2 e = x.xz - rd.xz * path;  // where the sunlight entered the water
    float dz = riverSlope(e.x);
    vec2 flow = vec2(1.0, dz) * inversesqrt(1.0 + dz * dz) * F.water.x * 0.9;
    float T = F.camPos.w, ph0 = fract(T / 2.0), ph1 = fract(T / 2.0 + 0.5), blend = abs(1.0 - 2.0 * ph0);
    vec2 q0 = e - flow * ph0 * 2.0, q1 = e - flow * ph1 * 2.0 + vec2(0.43, 0.71);
    float c = mix(causticNet(q0 * 1.7), causticNet(q1 * 1.7), blend);
    // sharper in shallow water, smeared out with depth; mean stays ~1 (light is redistributed, not added)
    float k = F.weather.w * exp(-0.5 * depth);
    float focus = 1.0 + k * (c * 3.2 - 0.7);
    vec3 sigma = F.waterCol.w * vec3(1.0, 0.42, 0.32) / max(F.water.y, 0.05);
    return exp(-sigma * path) * max(focus, 0.0);
}

// Demodulated sun contribution (Lo / albedo) with a soft cone-sampled shadow ray.
vec3 sunDirect(vec3 x, vec3 n, uint mat, inout uint s) {
    if (F.sunColor.r + F.sunColor.g + F.sunColor.b <= 0.0) return vec3(0);
    vec3 l = sampleCone(F.sunDir.xyz, F.sunDir.w, s);
    float side;
    float c = lambert(n, l, mat, side);
    if (c <= 0.0) return vec3(0);
    if (!visible(x + n * (EPS * side), l, 1e4)) return vec3(0);  // path vertices: approximate in craters
    return F.sunColor.rgb * cloudShadow(x) * underwaterSun(x) * c / PI * (side < 0.0 ? TRANSLUCENT_TINT : vec3(1));
}

// GGX specular lobe (the waxy sheen on grass, dull highlight on rock).
float ggxSpec(vec3 n, vec3 v, vec3 l, float rough) {
    float ndl = dot(n, l), ndv = max(dot(n, v), 1e-3);
    if (ndl <= 0.0) return 0.0;
    vec3 h = normalize(l + v);
    float a2 = rough * rough * rough * rough;
    float nh = max(dot(n, h), 0.0);
    float dd = nh * nh * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * dd * dd);
    float k = rough * rough * 0.5;
    float G = ndv / (ndv * (1.0 - k) + k) * ndl / (ndl * (1.0 - k) + k);
    float Fr = 0.04 + 0.96 * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    return D * G * Fr / (4.0 * ndv);   // already multiplied by n.l
}

// Sun at the primary hit through the full BRDF, demodulated (divided by the pixel's albedo A).
// u: 2D sample for the point on the sun disc (blue noise)
vec3 sunDirectPrimary(vec3 x, Surf sf, vec3 v, vec3 A, vec2 u) {
    if (F.sunColor.r + F.sunColor.g + F.sunColor.b <= 0.0) return vec3(0);
    vec3 l = sampleConeU(F.sunDir.xyz, F.sunDir.w, u);
    vec3 f = evalBRDF(sf, v, l);
    if (f.r + f.g + f.b <= 0.0) return vec3(0);
    float side = dot(sf.n, l) >= 0.0 ? 1.0 : -1.0;
    if (!visibleSun(x + sf.n * (EPS * side), l)) return vec3(0);
    return F.sunColor.rgb * cloudShadow(x) * underwaterSun(x) * f / A;
}

// ---------------------------------------------------------------- G-buffer surface
Surf loadSurf(ivec2 p) {
    vec4 N4 = imageLoad(gNorm, p), a = imageLoad(gAlbedo, p), m = imageLoad(gMat, p), m2 = imageLoad(gMat2, p);
    Surf s;
    s.base = a.rgb; s.ao = a.a;
    s.n = N4.xyz; s.mat = uint(N4.w + 0.5);
    vec4 t = imageLoad(gTan, p);
    s.T = t.xyz;
    s.ior = t.w >= 1.0 ? t.w : 1.5;
    s.transl = t.w < 0.0 ? -t.w : (s.mat == MAT_GRASS ? 0.6 : 0.0);
    s.rough = m.x; s.metal = m.y; s.spec = m.z; s.trans = m.w;
    s.cc = m2.x; s.ccRough = m2.y; s.sheen = m2.z; s.aniso = m2.w;
    return s;
}
void storeSurf(ivec2 p, Surf s) {
    imageStore(gAlbedo, p, vec4(s.base, s.ao));
    imageStore(gMat, p, vec4(s.rough, s.metal, s.spec, s.trans));
    imageStore(gMat2, p, vec4(s.cc, s.ccRough, s.sheen, s.aniso));
    imageStore(gTan, p, vec4(s.T, packIorTransl(s)));
}
// Demodulation albedo of pixel p (0 for sky): what lighting images are divided by and the composite multiplies by.
vec3 pixelAlbedo(ivec2 p) {
    if (imageLoad(gDepth, p).r < 0.0) return vec3(0);
    Surf s = loadSurf(p);
    return surfAlbedo(s, max(dot(s.n, -cameraRay(p)), 1e-4));
}

#include "lights.glsl"

// Volumetric fog in front of render pixel p (volfog.comp, 1/F.fogRes.x resolution, bilinear): rgb in-scatter, a transmittance.
vec4 fogAt(ivec2 p) {
    if (F.fog0.w < 0.5) return vec4(0, 0, 0, 1);
    ivec2 hs = F.fogRes.yz;
    int div = F.fogRes.x;
    vec2 c = (vec2(p) + 0.5) / float(div) - 0.5;
    ivec2 i = ivec2(floor(c));
    vec2 f = c - vec2(i);
    if (div == 1) return imageLoad(fogImg, p);
    // bilinear, but taps whose depth (at the pixel volfog.comp marched) differs from ours count less: no fog
    // bleeding across silhouettes, which shows as stair steps at low fog resolution
    float z = imageLoad(gDepth, p).r;
    z = z < 0.0 ? 1e4 : z;
    vec4 sum = vec4(0);
    float wSum = 0.0;
    for (int k = 0; k < 4; k++) {
        ivec2 o = ivec2(k & 1, k >> 1);
        ivec2 q = clamp(i + o, ivec2(0), hs - 1);
        float zq = imageLoad(gDepth, min(q * div + div / 2, ivec2(F.res.xy) - 1)).r;
        zq = zq < 0.0 ? 1e4 : zq;
        vec2 t = mix(1.0 - f, f, vec2(o));
        float w = t.x * t.y / (1e-3 + abs(zq - z) / max(z, 0.1));
        sum += imageLoad(fogImg, q) * w;
        wSum += w;
    }
    return sum / wSum;
}

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
