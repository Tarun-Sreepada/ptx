// Exact crater surface where the terrain mesh is too coarse for a bowl. The mesh is a world-anchored clipmap
// (terrain.comp): 0.25 m vertices near the camera, twice as far apart per ring. A crater whose radius spans only a few
// vertices at its distance (a fist-sized stone's dent seen from 20 m) is shaded on the exact bowl instead: camera rays
// march it, shadow rays from inside it march it too. Bigger craters are simply in the mesh.
#ifndef TERRAIN_CRATER_SURFACE
#define TERRAIN_CRATER_SURFACE
#include "core/frame.glsl"
#include "terrain/height.glsl"

// Terrain mesh vertex spacing (m) at horizontal distance dist from where the clipmap is centred.
// F.terrainP.z = finest spacing, F.terrain2.x = levels, F.terrain2.y = cells per level side.
float terrainSpacing(float dist) {
    float half0 = 0.5 * F.terrain2.y * F.terrainP.z;
    float level = clamp(ceil(log2(max(dist, 1e-3) / half0)), 0.0, max(F.terrain2.x - 1.0, 0.0));
    return F.terrainP.z * exp2(level);
}
const float CRATER_RESOLVED = 0.15;  // a bowl is in the mesh when its radius spans 1 / this vertices or more

// Local crater surface: the ground before craters as a plane through the point (or the CPU's plane under the biggest
// crater) plus only the craters that reach it (<= CRATER_LOCAL): cheap, smooth, and the same surface for the camera
// and the shadow rays, so they agree.
const int CRATER_LOCAL = 8;
struct CraterLocal { int n; uint idx[CRATER_LOCAL]; vec2 o; float hb; vec2 g; float depth; float minR; };
// exact: fit the ground at p from the real terrain (3 evaluations: right next to a river bank or a slope the
// crater-centre plane is metres off). Otherwise the CPU's plane under the biggest crater (free, good away from banks).
bool craterLocal(vec2 p, float reach, bool exact, out CraterLocal L) {
    // the craters that shape the ground here: in a dense cluster keep the most influential (closest relative to
    // their reach), not the first in the list, so overlapping bowls all count
    L.n = 0;
    L.depth = 0.0;
    L.minR = 1e9;
    if (F.dynInfo.z == 0u) return false;
    // a bowl smaller than the mesh cells still dips the vertices around it: the pit reaches a cell diagonal out
    // (square / diamond), so that far counts as the crater's too and is put back on the exact surface
    float cell = 1.5 * terrainSpacing(length(p - F.terrainP.xy));
    reach = max(reach, cell);
    float score[CRATER_LOCAL];
    uint cs, cn;
    bool bn = craterRange(p, cs, cn);
    for (uint j = 0u; j < cn; j++) {
        uint i = craterIdx(bn, cs, j);
        vec4 c = F.craters[i];
        float sc = length(p - c.xy) / (CRATER_EJECTA_REACH * c.z + reach);
        if (sc >= 1.0) continue;
        int slot = L.n < CRATER_LOCAL ? L.n++ : -1;
        if (slot < 0) {  // full: replace the weakest if this one matters more
            int w = 0;
            for (int k = 1; k < CRATER_LOCAL; k++) if (score[k] > score[w]) w = k;
            if (score[w] <= sc) continue;
            slot = w;
        }
        L.idx[slot] = i;
        score[slot] = sc;
    }
    if (L.n == 0) return false;
    int big = 0;
    for (int k = 0; k < L.n; k++) {
        vec4 c = F.craters[L.idx[k]];
        L.depth = max(L.depth, c.w);
        if (length(p - c.xy) < 1.2 * c.z + cell) L.minR = min(L.minR, c.z);  // only bowls the point is in (or their mesh pit) decide the resolution
        if (c.z > F.craters[L.idx[big]].z) big = k;
    }
    if (exact) {
        const float e = 0.75;
        L.o = p;
        L.hb = groundBase(p);
        L.g = vec2(groundBase(p + vec2(e, 0)) - L.hb, groundBase(p + vec2(0, e)) - L.hb) / e;
        return true;
    }
    vec4 g = F.craterG[L.idx[big]];
    L.o = F.craters[L.idx[big]].xy;
    L.hb = g.x;
    L.g = g.yz;
    return true;
}
float localHeight(CraterLocal L, vec2 q) {
    CraterSum s = craterSumInit();
    for (int k = 0; k < L.n; k++) craterAcc(F.craters[L.idx[k]], F.craterG[L.idx[k]], q, s);
    return craterApply(L.hb + dot(L.g, q - L.o), s);
}
// the bowl here is too small for the mesh at this distance from the camera
bool craterUnresolved(CraterLocal L, vec3 P) { return terrainSpacing(length(P.xz - F.terrainP.xy)) > CRATER_RESOLVED * L.minR; }

bool nearCrater(vec2 p) {
    uint cs, cn;
    bool bn = craterRange(p, cs, cn);
    for (uint j = 0u; j < cn; j++) { uint i = craterIdx(bn, cs, j); if (length(p - F.craters[i].xy) < 1.2 * F.craters[i].z + 0.5) return true; }
    return false;
}
// Camera ray onto the exact bowl (P: the mesh hit, d: unit ray direction): moves P onto the bowl and gives its normal.
bool craterRefine(inout vec3 P, vec3 d, inout vec3 n) {
    if (F.dynInfo.z == 0u) return false;
    CraterLocal L;
    if (!craterLocal(P.xz, 0.5, true, L) || !craterUnresolved(L, P)) return false;
    float span = (L.depth * 1.6 + 0.5) / max(abs(d.y), 0.15);  // the true surface is within this of the mesh along the ray
    // never start behind the camera: from inside a crater the march would find the ground around / below the camera
    // first and shade that instead of the wall in front ("seeing through the earth")
    float tHit = dot(P - F.camPos.xyz, d);
    float back = min(span, max(tHit - 0.1, 0.0));
    vec3 a = P - d * back;
    if (a.y - localHeight(L, a.xz) < 0.0) return false;
    const int N = 20;
    float dt = (back + span) / float(N), t0 = 0.0;
    bool hit = false;
    for (int k = 1; k <= N; k++) {
        float t = float(k) * dt;
        vec3 q = a + d * t;
        if (q.y - localHeight(L, q.xz) <= 0.0) {
            float lo = t0, hi = t;
            for (int j = 0; j < 5; j++) {
                float m = 0.5 * (lo + hi);
                vec3 r = a + d * m;
                if (r.y - localHeight(L, r.xz) > 0.0) lo = m; else hi = m;
            }
            P = a + d * (0.5 * (lo + hi));
            hit = true;
            break;
        }
        t0 = t;
    }
    if (!hit) return false;
    const float e = 0.08;  // a little wider than a texel: smooth normals, no speckle
    n = normalize(vec3(localHeight(L, P.xz - vec2(e, 0)) - localHeight(L, P.xz + vec2(e, 0)), 2.0 * e,
                       localHeight(L, P.xz - vec2(0, e)) - localHeight(L, P.xz + vec2(0, e))));
    return true;
}
#endif
