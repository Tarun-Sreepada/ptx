// Ray queries: visibility (with exact crater bowls) and closest hits with their full surface.
#ifndef LIGHTING_TRACE
#define LIGHTING_TRACE
#include "core/frame.glsl"
#include "core/bindings.glsl"
#include "core/util.glsl"
#include "materials/brdf.glsl"
#include "materials/model.glsl"
#include "terrain/terrain.glsl"
#include "terrain/ground.glsl"
#include "scene/wind.glsl"

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
    if (!craterUnresolved(L, o)) return 0.0;  // the bowl is in the mesh: trace it like any other ground
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
        h.s.cc = sm.cc; h.s.ccRough = sm.ccRough; h.s.sheen = sm.sheen; h.s.aniso = sm.aniso; h.s.ao = sm.ao; h.s.sss = sm.sss;
        return h;
    } else if (inst == MAT_TERRAIN) {
        uint i0 = ti[3u * prim], i1 = ti[3u * prim + 1u], i2 = ti[3u * prim + 2u];
        h.n = normalize(terrNrm(i0) * w.x + terrNrm(i1) * w.y + terrNrm(i2) * w.z);
        if (full && craterRefine(h.p, d, h.n)) h.t = dot(h.p - o, d);  // exact crater bowl (camera rays)
        float dc = length(h.p.xz - F.camPos.xz);
        h.albedo = terrainAlbedo(h.p.xz, smoothstep(F.grassA.w * 0.4, F.grassA.w, dc));
        vec3 gp = h.p;
        float gr = 1.0, gao = 1.0, dug;
        h.albedo = craterSoil(h.p, h.n, h.albedo, gr, dug);
        groundSurface(gp, d, length(h.p - F.camPos.xyz), false, h.n, h.albedo, gr, gao);
        h.back = dot(h.n, d) > 0.0;
        if (h.back) h.n = -h.n;
        h.s = procSurf(h.albedo, h.n, h.mat);
        h.s.rough = gr; h.s.ao = gao; h.s.spec = 0.5;
        if (full && F.sssP.w > 0.5 && snowDepth(h.p.xz) > 0.075) h.s.sss = 3.0;  // snow: subsurface (raster.frag)
        return h;
    } else if (inst == MAT_ROCK) {
        uint vb = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rq, true);  // variant's base vertex
        uint i0 = ri[3u * prim] + vb, i1 = ri[3u * prim + 1u] + vb, i2 = ri[3u * prim + 2u] + vb;
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
#endif
