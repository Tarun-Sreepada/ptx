// Subsurface scattering by random walk (Chiang, Kutz & Burley 2016; the "random walk" mode of Cycles / Arnold).
// Light enters the surface, scatters through the medium (exponential free flights between isotropic scattering
// events) and leaves wherever the walk crosses the surface again: on the far side of a thin part (ears, fingers,
// a candle's rim glow), or millimetres to centimetres from where it went in (soft terminator, red bleeding in skin,
// blue depth in snow). Walked from the camera's side (the reverse path), the exit point is lit by the sun and the
// pixel's chosen light. Per channel the medium is set from the surface colour by albedo inversion, so the walk
// reflects exactly the base colour in total and only moves light around.
// Thin leaves and blades use the thin-slab solution instead (brdf.glsl leafTransmit).
#ifndef SSS_GLSL
#define SSS_GLSL

// x: scattering distance (m) of the most scattering channel; yzw: mean free path per channel relative to it
vec4 sssProfile(float k) {
    int i = int(k + 0.5);
    if (i == 1) return vec4(0.0037, 1.0, 0.37, 0.19);  // skin (Jensen et al. 2001 skin1: 3.67 / 1.37 / 0.68 mm)
    if (i == 2) return vec4(0.006, 1.0, 0.78, 0.52);   // paraffin / candle wax
    if (i == 3) return vec4(0.025, 0.4, 0.8, 1.0);     // snow: ice absorbs red, blue light goes deepest
    if (i == 4) return vec4(0.0085, 1.0, 0.66, 0.46);  // marble (Jensen: 8.51 / 5.57 / 3.95 mm)
    return vec4(0.004, 1.0, 1.0, 1.0);
}
// single-scattering albedo giving multiple-scattering albedo A in a semi-infinite medium (van de Hulst, as fitted
// by Chiang et al.)
vec3 sssSingleAlbedo(vec3 A) {
    A = clamp(A, vec3(0.0), vec3(0.999));
    vec3 s = 4.09712 + 4.20863 * A - sqrt(9.59217 + 41.6808 * A + 17.7126 * A * A);
    return 1.0 - s * s;
}

// Closest hit, geometry only (interpolated normal, no material): the walk's boundary test.
bool traceGeom(vec3 o, vec3 d, float tmax, out float t, out vec3 n) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsOpaqueEXT, 0xFF, o, 0.0, d, tmax);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionTriangleEXT) return false;
    t = rayQueryGetIntersectionTEXT(rq, true);
    uint inst = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
    uint prim = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    vec2 bc = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    vec3 w = vec3(1.0 - bc.x - bc.y, bc.x, bc.y);
    if (inst >= MAT_MODEL) {
        MeshInfo m = meshes[inst - MAT_MODEL];
        UintBuf ib = UintBuf(m.idx);
        ModelVertex a = modelVertex(m, ib.v[3u * prim]), b = modelVertex(m, ib.v[3u * prim + 1u]), c = modelVertex(m, ib.v[3u * prim + 2u]);
        vec3 nObj = a.n * w.x + b.n * w.y + c.n * w.z;
        if (dot(nObj, nObj) < 1e-12) nObj = cross(b.p - a.p, c.p - a.p);
        n = normalize(transpose(mat3(rayQueryGetIntersectionWorldToObjectEXT(rq, true))) * nObj);
    } else if (inst == MAT_TERRAIN) {
        uint i0 = ti[3u * prim], i1 = ti[3u * prim + 1u], i2 = ti[3u * prim + 2u];
        n = normalize(terrNrm(i0) * w.x + terrNrm(i1) * w.y + terrNrm(i2) * w.z);
    } else if (inst == MAT_ROCK) {
        uint vb = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rq, true);
        uint i0 = ri[3u * prim] + vb, i1 = ri[3u * prim + 1u] + vb, i2 = ri[3u * prim + 2u] + vb;
        vec3 nObj = vec3(rv[6u * i0 + 3u], rv[6u * i0 + 4u], rv[6u * i0 + 5u]) * w.x +
                    vec3(rv[6u * i1 + 3u], rv[6u * i1 + 4u], rv[6u * i1 + 5u]) * w.y +
                    vec3(rv[6u * i2 + 3u], rv[6u * i2 + 4u], rv[6u * i2 + 5u]) * w.z;
        n = normalize(mat3(rayQueryGetIntersectionObjectToWorldEXT(rq, true)) * nObj);
    } else {
        uint i0 = gix[3u * prim], i1 = gix[3u * prim + 1u], i2 = gix[3u * prim + 2u];
        n = normalize(cross(grassPos(i1) - grassPos(i0), grassPos(i2) - grassPos(i0)));
    }
    return true;
}

// One walk from x (outward normal n) into the medium. Returns false if it was absorbed / left the scene; else the
// exit point, its outward normal and the path weight (per channel; its expectation over walks is the albedo A).
// maxSteps: scattering events before giving up on a deep walk: it then surfaces straight above where it got to,
// keeping its weight (the light it carries would mostly come out about there anyway, just later).
bool sssWalk(vec3 x, vec3 n, vec3 A, float prof, uint maxSteps, float scale, inout uint s, out vec3 xe, out vec3 ne, out vec3 thr) {
    vec4 pr = sssProfile(prof);
    vec3 sigT = 1.0 / max(pr.x * scale * pr.yzw, vec3(1e-6));  // extinction per m
    vec3 alpha = sssSingleAlbedo(A);
    thr = vec3(1);
    vec3 d = cosineSample(-n, s);
    vec3 o = x - n * min(2e-4, 0.05 * pr.x * scale);
    for (uint i = 0u; i < maxSteps; i++) {
        // spectral MIS: pick the channel whose free flight to sample in proportion to its throughput
        vec3 pc = thr / max(thr.r + thr.g + thr.b, 1e-20);
        float u = rnd(s);
        float sig = u < pc.r ? sigT.r : (u < pc.r + pc.g ? sigT.g : sigT.b);
        float t = -log(max(1.0 - rnd(s), 1e-7)) / sig;
        float th;
        vec3 hn;
        if (traceGeom(o, d, t, th, hn)) {  // crossed the boundary before the next event: out
            vec3 T = exp(-sigT * th);
            thr *= T / max(dot(pc, T), 1e-20);
            xe = o + d * th;
            ne = dot(hn, d) < 0.0 ? -hn : hn;  // outward = the way the walk was going
            return true;
        }
        vec3 T = exp(-sigT * t);
        thr *= alpha * sigT * T / max(dot(pc, sigT * T), 1e-20);
        o += d * t;
        // Russian roulette once the walk has gone a while
        if (i > 3u) {
            float q = clamp(max(thr.r, max(thr.g, thr.b)), 0.05, 1.0);
            if (rnd(s) > q) return false;
            thr /= q;
        }
        float z = 1.0 - 2.0 * rnd(s), ph = 2.0 * PI * rnd(s), r = sqrt(max(1.0 - z * z, 0.0));
        d = vec3(r * cos(ph), r * sin(ph), z);  // isotropic phase function
    }
    float th;
    vec3 hn;
    if (!traceGeom(o, n, 1e3, th, hn)) return false;
    xe = o + n * th;
    ne = dot(hn, n) < 0.0 ? -hn : hn;
    return true;
}
#endif
