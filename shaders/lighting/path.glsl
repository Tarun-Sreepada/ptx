// Path tracing shared by the GI pass and the probe bake.
#ifndef PATH_GLSL
#define PATH_GLSL
#include "lighting/radcache.glsl"

// One random light from [first, first + count), demodulated (multiply by the receiver's albedo).
// During a bake, lights in switched-off buildings (or in the baked one, for its lights-off bake) give nothing.
vec3 lightNEE(vec3 x, vec3 n, uint mat, uint first, uint count, inout uint s) {
    if (count == 0u) return vec3(0);
    uint li = first + min(uint(rnd(s) * float(count)), count - 1u);
    vec4 L = lights[2u * li];
    if (bakeVol >= 0 && emitterOff(L.xyz)) return vec3(0);
    vec3 d = L.xyz - x;
    float dist = length(d);
    vec3 wi = d / dist;
    float side;
    float c = lambert(n, wi, mat, side);
    if (c <= 0.0 || !visible(x + n * (EPS * side), wi, dist - L.w)) return vec3(0);
    return lights[2u * li + 1u].rgb * c / (dist * dist) * float(count) / PI;
}
vec3 lightNEE(vec3 x, vec3 n, uint mat, inout uint s) { return lightNEE(x, n, mat, 0u, F.res.w, s); }

// Bake NEE: every static light within 40 m (shadow ray each) instead of one random pick: probes are few and
// must be low-noise, and a building's own lamps are a tiny fraction of the world's lights.
vec3 lightsSum(vec3 x, vec3 n, uint mat, uint first, uint count) {
    vec3 sum = vec3(0);
    for (uint li = first; li < first + count; li++) {
        vec4 L = lights[2u * li];
        vec3 d = L.xyz - x;
        float d2 = dot(d, d);
        if (d2 > 1600.0 || emitterOff(L.xyz)) continue;
        float dist = sqrt(d2);
        vec3 wi = d / dist;
        float side;
        float c = lambert(n, wi, mat, side);
        if (c <= 0.0 || !visible(x + n * (EPS * side), wi, dist - L.w)) continue;
        sum += lights[2u * li + 1u].rgb * c / d2 / PI;
    }
    return sum;
}

// Radiance arriving at x0 from direction d (= radiance leaving the first secondary hit xs).
// Lights [lFirst, lFirst + lCount) are sampled at every vertex; the bake passes the static ones only.
// cone: footprint (width, spread) of the first ray, for texture filtering; later bounces widen it to a diffuse lobe.
// Glass (thin-walled transmission) is passed straight through, tinted by its colour, without using up a bounce.
// emitW0: weight of emission at the first hit (lights.glsl emissionWeight).
// Outside bakes, paths can end early (radcache.glsl): a hit inside a baked volume reads the probes (FLAG_GI_PROBES);
// with FLAG_GI_CACHE the first hit adds exact direct light + the cell's cached indirect light, later hits the
// cell's whole cached radiance. Vertices whose continuation was traced train the cache.
#define PATH_REC 4u
vec2 pathSunU = vec2(-1);  // >= 0: blue-noise sample for the sun at the first hit (gi.comp)
vec3 tracePath(vec3 x0, vec3 n0, vec3 d, uint B, uint lFirst, uint lCount, vec2 cone, float emitW0, inout uint s, out vec3 xs, out vec3 ns, out bool back) {
    vec3 o = x0 + n0 * EPS, nn = n0, thr = vec3(1), L = vec3(0);
    xs = x0 + d * 1e4;
    ns = -d;
    back = false;
    uint glass = 0u, nv = 0u;
    bool live = bakeVol < 0;
    bool useCache = live && (F.settings.z & FLAG_GI_CACHE) != 0u, useProbes = live && (F.settings.z & FLAG_GI_PROBES) != 0u;
    // per vertex: position, normal, radiance so far, throughput incl. its albedo, its direct light
    vec3 vP[PATH_REC], vN[PATH_REC], vL[PATH_REC], vT[PATH_REC], vD[PATH_REC];
    for (uint b = 0u; b < B; b++) {
        if (b > 0u) { d = cosineSample(nn, s); o += nn * EPS; }
        Hit h = traceClosest(o, d, 1e4, cone);
        if (!h.hit) { L += thr * sky(d); break; }
        if (h.trans > 0.5 && glass < 8u) {  // through the glass: refract in (or out, hitting a back face)
            thr *= mix(vec3(1), h.albedo, h.trans);
            vec3 r = refract(d, h.n, h.back ? h.ior : 1.0 / h.ior);
            if (dot(r, r) < 1e-6) { r = reflect(d, h.n); o = h.p + h.n * EPS; }  // total internal reflection
            else o = h.p - h.n * EPS;
            d = normalize(r);
            cone.x += cone.y * h.t;
            glass++;
            b--;
            continue;
        }
        if (dot(h.emis, h.emis) > 0.0) L += thr * h.emis * emissionWeight(b, emitW0);
        if (b == 0u) { xs = h.p; ns = h.n; back = h.back; }
        thr *= h.albedo * (b == 0u ? 1.0 : h.s.ao);
        vec3 E, Ed;
        if (useProbes && bakedIndirect(h.p, h.n, E)) {  // probes hold sun, sky, static lights: add the orbs
            L += thr * (E + lightNEE(h.p, h.n, h.mat, 0u, F.bake.y, s));
            break;
        }
        bool cached = useCache && (b > 0u || !rcTrain) && rcLookup(h.p, h.n, E, Ed);
        if (cached && b > 0u) { L += thr * E; break; }
        vec3 D = (b == 0u && pathSunU.x >= 0.0 ? sunDirect(h.p, h.n, h.mat, pathSunU) : sunDirect(h.p, h.n, h.mat, s)) + (bakeVol >= 0 ? lightsSum(h.p, h.n, h.mat, lFirst, lCount) : treeNEE(h.p, h.n, h.mat, s));
        L += thr * D;
        if (cached) { L += thr * max(E - Ed, vec3(0)); break; }
        if (nv < PATH_REC) { vP[nv] = h.p; vN[nv] = h.n; vL[nv] = L - thr * D; vT[nv] = thr; vD[nv] = D; nv++; }
        cone = vec2(cone.x + cone.y * h.t, DIFFUSE_SPREAD);
        o = h.p; nn = h.n;
    }
    if (useCache)
        for (uint k = 0u; k < nv; k++) {
            vec3 t = vT[k];
            if (min(t.r, min(t.g, t.b)) < 1e-4) continue;
            rcAdd(vP[k], vN[k], (L - vL[k]) / t, vD[k], rnd(s));
        }
    return L;
}

#endif
