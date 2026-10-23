// Procedural materials, shared by the ray tracer (hit shading) and the raster G-buffer pass.
#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL
float sq(float x) { return x * x; }
uint hashu(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = float(hashu(uint(int(i.x)) * 7919u + uint(int(i.y)) * 104729u)) / 4294967295.0;
    float b = float(hashu(uint(int(i.x + 1.0)) * 7919u + uint(int(i.y)) * 104729u)) / 4294967295.0;
    float c = float(hashu(uint(int(i.x)) * 7919u + uint(int(i.y + 1.0)) * 104729u)) / 4294967295.0;
    float d = float(hashu(uint(int(i.x + 1.0)) * 7919u + uint(int(i.y + 1.0)) * 104729u)) / 4294967295.0;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Infinite terrain height. Mirrors terrain.h (CPU: light/rock placement, physics heightfields).
float thash(int x, int z) {
    uint h = uint(x) * 374761393u + uint(z) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return float((h ^ (h >> 16)) & 0xFFFFFFu) / float(0xFFFFFF);
}
float tnoise(float x, float z) {
    int ix = int(floor(x)), iz = int(floor(z));
    float fx = x - float(ix), fz = z - float(iz);
    fx = fx * fx * (3.0 - 2.0 * fx);
    fz = fz * fz * (3.0 - 2.0 * fz);
    float a = thash(ix, iz), b = thash(ix + 1, iz), c = thash(ix, iz + 1), d = thash(ix + 1, iz + 1);
    return mix(mix(a, b, fx), mix(c, d, fx), fz);
}
// Flatten zones (streamed with the world cells): ground pulled to a fixed height inside a rectangle,
// smoothstep falloff around it. Count in F.counts.x. Mirrors terrainHeight() in terrain.h.
layout(binding = 36, std430) readonly buffer Zones { vec4 zones[]; };  // (minx, minz, maxx, maxz), (height, falloff, -, -)
float zoneWeight(uint i, vec2 p) {
    vec4 r = zones[2u * i];
    vec2 d = max(max(r.xy - p, p - r.zw), vec2(0.0));
    return 1.0 - smoothstep(0.0, 1.0, length(d) / max(zones[2u * i + 1u].y, 1e-3));
}
float zoneCover(vec2 p) {
    float w = 0.0;
    for (uint i = 0u; i < F.counts.x; i++) w = max(w, zoneWeight(i, p));
    return w;
}
// River channel, mirrors terrain.h. x = signed distance across the flow, y = half-width, z = depth, w = water level.
const float RIVER_Z = -58.0, RIVER_BANK = 16.0;
float riverCenter(float x) { return RIVER_Z + 14.0 * sin(0.011 * x) + 7.0 * sin(0.027 * x + 1.3); }
float riverSlope(float x) { return 0.154 * cos(0.011 * x) + 0.189 * cos(0.027 * x + 1.3); }
bool riverNear(vec2 p, float margin) { return abs(p.y - riverCenter(p.x)) <= 3.0 * F.river.x + margin; }
vec4 riverInfo(vec2 p) {
    float dz = riverSlope(p.x);
    return vec4((p.y - riverCenter(p.x)) * inversesqrt(1.0 + dz * dz), F.river.x * (0.8 + 0.45 * tnoise(p.x * 0.008 + 3.0, 7.5)),
                F.river.y * (0.45 + 1.0 * tnoise(p.x * 0.013, 2.5)), F.river.z - F.river.w * p.x);
}
float riverCarve(vec2 p, float h) {
    if (!riverNear(p, RIVER_BANK + 6.0)) return h;
    vec4 r = riverInfo(p);
    float d = abs(r.x), c = d / r.y;
    float prof = d < r.y ? r.w - r.z * (1.0 - c * c) : r.w + 0.35 * (d - r.y);
    if (F.water2.w > 0.0 && d < r.y + 2.0) {  // uneven bed (terrain.h riverCarve, gRiver.bumps)
        float n = tnoise(p.x * 0.45, p.y * 0.45) * 0.6 + tnoise(p.x * 1.3 + 7.0, p.y * 1.3) * 0.4 - 0.5;
        prof += 2.0 * F.water2.w * n * sqrt(max(1.0 - c * c, 0.0));
    }
    return mix(prof, h, smoothstep(r.y + 2.0, r.y + RIVER_BANK, d));
}
const float RIVER_ROCK_L = 2.5;
bool riverRock(int i, out vec4 rk) {  // xyz centre, w scale (terrain.h)
    if (thash(i * 7 + 3, 811) > 0.65) return false;
    bool boulder = thash(i * 5 + 1, 823) < 0.35;
    float x = (float(i) + thash(i, 853)) * RIVER_ROCK_L;
    float hc = thash(i * 3 + 2, 839) * 2.0 - 1.0;
    float c = boulder ? hc * 0.75 : (hc < 0.0 ? -1.0 : 1.0) * (0.45 + 0.5 * abs(hc)), dz = riverSlope(x);
    vec4 r = riverInfo(vec2(x, 0.0));
    float z = riverCenter(x) + c * r.y * sqrt(1.0 + dz * dz);
    float bed = r.w - r.z * (1.0 - c * c);
    float sc = boulder ? min(max(0.9 + 0.8 * thash(i, 857), (r.w - bed) * 0.9), 2.6) : 0.25 + 0.45 * thash(i, 859);
    float y = boulder ? max(bed - 0.3 * sc, r.w - 0.4 * sc) : bed - 0.3 * sc;
    float dc = c * r.y, f = 0.8 * sc, nearest = (dc - f <= 0.0 && dc + f >= 0.0) ? 0.0 : min(abs(dc - f), abs(dc + f));
    float cn = nearest / r.y;
    float minBed = nearest < r.y ? r.w - r.z * (1.0 - cn * cn) : r.w + 0.35 * (nearest - r.y);
    rk = vec4(x, min(y, minBed + 0.4 * sc), z, sc);
    return true;
}
// Impact crater (terrain.h craterCarve): parabolic bowl plus a raised rim of thrown-out soil.
// Overlapping craters don't stack (terrain.h craterAcc): the deepest bowl wins with a smooth max, rims fade out
// inside any bowl. Accumulate (D depth, R rim) over the craters, then apply.
float craterSmax(float a, float b, float k) { float h = max(k - abs(a - b), 0.0) / k; return max(a, b) + h * h * k * 0.25; }
void craterAcc(vec4 c, vec2 p, inout vec2 acc) {
    float t2 = dot(p - c.xy, p - c.xy) / (c.z * c.z);
    if (t2 > 3.24) return;
    float t = sqrt(t2);
    float d = c.w * max(1.0 - t2, 0.0);
    if (d > 0.0) acc.x = craterSmax(acc.x, d, 0.15 * c.w + 0.02);
    acc.y = max(acc.y, 0.3 * c.w * exp(-sq((t - 1.0) / 0.28)));
}
float craterApply(float h, vec2 acc) { return h - acc.x + acc.y * (1.0 - smoothstep(0.02, 0.3, acc.x)); }
// Crater bins (main.cpp buildCraterBins): a 32 x 32 grid of 16 m cells around the camera, each listing the craters
// whose reach (1.9 r + 1 m) touches it. Lookups near the camera loop over those few instead of every crater.
layout(binding = 97, std430) readonly buffer CraterBins { uint craterBins[]; };  // [2b] start, [2b+1] count; list after
const uint CB_BINS = 1024u;
bool craterRange(vec2 p, out uint s, out uint n) {
    ivec2 b = ivec2(floor(p / 16.0)) - F.craterGrid.xy;
    if (F.craterGrid.w > 0 && b.x >= 0 && b.y >= 0 && b.x < F.craterGrid.z && b.y < F.craterGrid.z) {
        uint k = uint(b.y * F.craterGrid.z + b.x);
        s = 2u * CB_BINS + craterBins[2u * k];
        n = craterBins[2u * k + 1u];
        return true;
    }
    s = 0u;
    n = F.dynInfo.z;
    return false;
}
uint craterIdx(bool binned, uint s, uint j) { return binned ? craterBins[s + j] : j; }

float craterCover(vec2 p) {  // 1 inside a crater, fading over the rim
    float k = 0.0;
    uint cs, cn;
    bool bn = craterRange(p, cs, cn);
    for (uint j = 0u; j < cn; j++) { uint i = craterIdx(bn, cs, j); k = max(k, 1.0 - smoothstep(0.9, 1.5, length(p - F.craters[i].xy) / F.craters[i].z)); }
    return k;
}
// ---- world map (worldmap.h): hand edits on a 2 m grid over 4 km, and the simulation grid on the same layout
layout(binding = 90) uniform sampler2D mapHeight;   // metres added to the natural ground
layout(binding = 91) uniform sampler2D mapPaint;    // r grass x2, g trees x2, b texture index / 255, a texture weight
layout(binding = 92) uniform sampler2D mapWater;    // drawn rivers (4 m): x level (-1000 none), yz flow (m/s), w half-width
layout(binding = 93, rgba16f) uniform image2D simImg;  // x snow depth (m), y water depth (m), z wetness, w avalanche speed
vec2 mapUV(vec2 p) { return (p - F.mapP.x) / (F.mapP.y * F.mapP.z); }
bool inMap(vec2 uv) { return uv.x >= 0.0 && uv.y >= 0.0 && uv.x < 1.0 && uv.y < 1.0; }
float editHeight(vec2 p) { vec2 uv = mapUV(p); return inMap(uv) ? textureLod(mapHeight, uv, 0.0).r : 0.0; }
vec4 mapPaintAt(vec2 p) { vec2 uv = mapUV(p); return inMap(uv) ? textureLod(mapPaint, uv, 0.0) : vec4(0.5, 0.5, 0.0, 0.0); }
// drawn water: bilinear over only the neighbouring cells that hold water (a cell without stores -1000; blending that
// in made sliver lines and grid-shaped edges). The shoreline is then where this smooth level meets the ground.
vec4 mapWaterAt(vec2 p) {
    vec2 uv = mapUV(p);
    if (!inMap(uv)) return vec4(-1000.0, 0, 0, 0);
    vec2 g = uv * vec2(textureSize(mapWater, 0)) - 0.5;
    ivec2 i = ivec2(floor(g));
    vec2 f = g - vec2(i);
    ivec2 sz = textureSize(mapWater, 0) - 1;
    vec4 acc = vec4(0);
    float wsum = 0.0;
    for (int k = 0; k < 4; k++) {
        ivec2 o = ivec2(k & 1, k >> 1);
        vec4 t = texelFetch(mapWater, clamp(i + o, ivec2(0), sz), 0);
        float w = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);
        if (t.x > -999.0) { acc += t * w; wsum += w; }
    }
    return wsum > 0.0 ? acc / wsum : vec4(-1000.0, 0, 0, 0);
}
// simulation state, bilinear (storage image: filtered by hand)
vec4 simAt(vec2 p) {
    if (F.sim1.w < 0.5) return vec4(0);
    vec2 g = (p - F.mapP.x) / F.mapP.y - 0.5;
    ivec2 i = ivec2(floor(g));
    int n = int(F.mapP.z);
    if (i.x < 0 || i.y < 0 || i.x >= n - 1 || i.y >= n - 1) return vec4(0);
    vec2 f = g - vec2(i);
    return mix(mix(imageLoad(simImg, i), imageLoad(simImg, i + ivec2(1, 0)), f.x),
               mix(imageLoad(simImg, i + ivec2(0, 1)), imageLoad(simImg, i + ivec2(1, 1)), f.x), f.y);
}
// steepness of the hand-sculpted ground (tan of the slope); the natural noise terrain is gentle everywhere
float editSlope(vec2 p) {
    vec2 uv = mapUV(p);
    if (!inMap(uv)) return 0.0;
    float e = F.mapP.y;
    float hx = editHeight(p + vec2(e, 0)) - editHeight(p - vec2(e, 0)), hz = editHeight(p + vec2(0, e)) - editHeight(p - vec2(0, e));
    return length(vec2(hx, hz)) / (2.0 * e);
}
float snowDepth(vec2 p) { return F.sim1.w > 0.5 ? simAt(p).x : 0.0; }

// the ground without snow (the simulation's floor)
float groundHeight(vec2 p) {
    float h = 2.2 * tnoise(p.x * 0.022 + 10.0, p.y * 0.022) + 0.7 * tnoise(p.x * 0.07, p.y * 0.07 + 5.0) +
              0.12 * tnoise(p.x * 0.35, p.y * 0.35) - 1.5 + editHeight(p);
    h = riverCarve(p, h);
    {
        uint cs, cn;
        bool bn = craterRange(p, cs, cn);
        vec2 acc = vec2(0);
        for (uint j = 0u; j < cn; j++) craterAcc(F.craters[craterIdx(bn, cs, j)], p, acc);
        h = craterApply(h, acc);
    }
    for (uint i = 0u; i < F.counts.x; i++) h = mix(h, zones[2u * i + 1u].x, zoneWeight(i, p));
    return h;
}
float terrainHeight(vec2 p) { return groundHeight(p) + snowDepth(p); }  // the visible surface: snow lies on top


// Local crater surface. Marching the full terrainHeight costs a loop over every crater per step, for every pixel and
// shadow ray: with a few big craters that tanks the frame. Near a crater the ground is instead the natural slope
// (a plane through the point, from 3 full evaluations) plus only the craters that reach it (<= 4): cheap, smooth,
// and the same surface for the camera and the shadow rays, so they agree.
const int CRATER_LOCAL = 8;
struct CraterLocal { int n; uint idx[CRATER_LOCAL]; vec2 o; float hb; vec2 g; float depth; };
// exact: fit the natural ground at p from the real terrain (3 evaluations: right next to a river bank or a slope the
// crater-centre plane is metres off). Otherwise the CPU's plane under the first crater (free, good away from banks).
bool craterLocal(vec2 p, float reach, bool exact, out CraterLocal L) {
    // the craters that shape the ground here: in a dense cluster keep the most influential (closest relative to
    // their reach), not the first in the list, so overlapping bowls all count
    L.n = 0;
    L.depth = 0.0;
    float score[CRATER_LOCAL];
    uint cs, cn;
    bool bn = craterRange(p, cs, cn);
    for (uint j = 0u; j < cn; j++) {
        uint i = craterIdx(bn, cs, j);
        vec4 c = F.craters[i];
        float sc = length(p - c.xy) / (1.8 * c.z + reach);
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
    int big = 0;
    for (int k = 0; k < L.n; k++) {
        L.depth = max(L.depth, F.craters[L.idx[k]].w);
        if (F.craters[L.idx[k]].z > F.craters[L.idx[big]].z) big = k;
    }
    if (L.n == 0) return false;
    if (exact) {
        const float e = 0.75;
        vec2 a0 = vec2(0), ax = vec2(0), az = vec2(0);
        for (int k = 0; k < L.n; k++) {
            vec4 c = F.craters[L.idx[k]];
            craterAcc(c, p, a0);
            craterAcc(c, p + vec2(e, 0), ax);
            craterAcc(c, p + vec2(0, e), az);
        }
        float c0 = craterApply(0.0, a0), cx = craterApply(0.0, ax), cz = craterApply(0.0, az);
        float h0 = groundHeight(p) - c0;
        L.o = p;
        L.hb = h0;
        L.g = vec2(groundHeight(p + vec2(e, 0)) - cx - h0, groundHeight(p + vec2(0, e)) - cz - h0) / e;
        return true;
    }
    // natural ground: the plane the CPU fitted under the biggest crater here (F.craterG), no terrain evaluation
    vec4 g = F.craterG[L.idx[big]];
    L.o = F.craters[L.idx[big]].xy;
    L.hb = g.x;
    L.g = g.yz;
    return true;
}
float localHeight(CraterLocal L, vec2 q) {
    float h = L.hb + dot(L.g, q - L.o);
    vec2 acc = vec2(0);
    for (int k = 0; k < L.n; k++) craterAcc(F.craters[L.idx[k]], q, acc);
    h = craterApply(h, acc);
    return h;
}

// Craters are 1-3 m bowls but the far terrain grid is metres apart, and it recentres as the camera moves: sampled by
// the mesh alone a distant crater changes shape. Inside a crater the view ray is marched against the local crater
// surface instead (P: the mesh hit, d: unit ray direction), moving P onto the bowl and giving its normal.
bool nearCrater(vec2 p) {
    uint cs, cn;
    bool bn = craterRange(p, cs, cn);
    for (uint j = 0u; j < cn; j++) { uint i = craterIdx(bn, cs, j); if (length(p - F.craters[i].xy) < 1.9 * F.craters[i].z + 0.5) return true; }
    return false;
}
bool craterRefine(inout vec3 P, vec3 d, inout vec3 n) {
    if (F.dynInfo.z == 0u) return false;
    CraterLocal L;
    if (!craterLocal(P.xz, 0.5, true, L)) return false;
    float span = (L.depth * 1.6 + 0.5) / max(abs(d.y), 0.15);  // the true surface is within this of the mesh along the ray
    // never start behind the camera: from inside a crater the march would find the ground around / below the camera
    // first and shade that instead of the wall in front ("seeing through the earth")
    float tHit = dot(P - F.camPos.xyz, d);
    // close up the terrain mesh already follows the bowl (0.5 m vertices, coarser with distance): refine only where
    // its spacing is too coarse for the smallest crater here
    float minR = 1e9;
    for (int k = 0; k < L.n; k++) minR = min(minR, F.craters[L.idx[k]].z);
    if (0.5 + 0.012 * tHit < 0.25 * minR) return false;
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

// ---- water surface (river + crater ponds), shared by the water shading, underwater light and the CPU mirror
// Crater ponds (F.craterW, main.cpp updateCraterWater): the crater whose bowl holds p (any crater, -1 = none)
int craterAt(vec2 p) {
    uint cs, cn;
    bool bn = craterRange(p, cs, cn);
    for (uint j = 0u; j < cn; j++) { uint i = craterIdx(bn, cs, j); if (length(p - F.craters[i].xy) < F.craters[i].z * 1.2) return int(i); }
    return -1;
}
int craterPondAt(vec2 p) {
    int i = craterAt(p);
    return i >= 0 && F.craterW[i].x > -1e8 ? i : -1;
}
// Water diverted into filling craters is missing downstream: flow factor at x, and the level drop there
float riverDiversion(float x, out float drop) {
    float s = 0.0;
    drop = 0.0;
    for (uint i = 0u; i < F.dynInfo.z; i++) {
        float q = F.craterW[i].z;
        if (q <= 0.0 || x < F.craters[i].x) continue;
        float k = q * exp(-(x - F.craters[i].x) / 150.0);
        s += k;
        drop += k * 0.3 * F.river.y;
    }
    return clamp(1.0 - s, 0.15, 1.0);
}
// water surface height over p, -1e9 = dry. Inside a crater only its pond counts (a crater by the river is not
// under the river's level until it has filled).
float waterLevelAt(vec2 p) {
    if (F.dynInfo.z > 0u) {
        int c = craterAt(p);
        if (c >= 0) return F.craterW[c].x;
    }
    float drawn = mapWaterAt(p).x;  // a river drawn on the world map
    if (drawn > -999.0) return drawn;
    if (F.river.x <= 0.0 || !riverNear(p, 1.0)) return -1e9;
    vec4 r = riverInfo(p);
    if (abs(r.x) > r.y + 1.0) return -1e9;
    float drop = 0.0;
    if (F.dynInfo.z > 0u) riverDiversion(p.x, drop);
    return r.w - drop + F.sim2.w;
}

// 0 = bare ground, 1 = full grass: large clumps, bare patches and a winding dirt path.
float grassMask(vec2 xz) {
    float n = vnoise(xz * 0.06) * 0.65 + vnoise(xz * 0.21) * 0.35;
    float m = smoothstep(0.26, 0.42, n);
    float path = abs(xz.y - 9.0 * sin(xz.x * 0.045) - 4.0 * sin(xz.x * 0.11 + 1.0));
    float dry = 1.0;
    if (riverNear(xz, 2.0)) { vec4 r = riverInfo(xz); dry = smoothstep(r.y - 0.3, r.y + 1.2, abs(r.x)); }  // none under water
    float painted = mapPaintAt(xz).r * 2.0;                       // world map grass brush (1 = as generated)
    float buried = 1.0 - smoothstep(0.04, 0.25, snowDepth(xz));   // under snow
    buried *= 1.0 - smoothstep(0.6, 1.0, editSlope(xz));          // nothing grows on faces steeper than ~35-45 deg
    return clamp(m * painted + max(painted - 1.0, 0.0) * 0.8, 0.0, 1.0) * smoothstep(0.9, 1.7, path) * (1.0 - zoneCover(xz)) * dry * (1.0 - craterCover(xz)) * buried;  // no grass on flattened (city) ground
}

// Blade length field (0 = short lawn, 1 = tall meadow); grass.comp scales blade height by it.
float grassHeight(vec2 xz) {
    float n = vnoise(xz * 0.045 + 31.0) * 0.7 + vnoise(xz * 0.17 + 5.0) * 0.3;
    return smoothstep(0.25, 0.75, n) * (0.4 + 0.6 * smoothstep(0.2, 0.8, grassMask(xz)));
}

// Ground colour follows grassMask (the same field that places blades): green where grass grows,
// brown soil / gray stony ground where it doesn't, with a dry fringe between.
// farGrass: 0 under real blades (shaded soil-green), 1 past the grass draw distance (looks like the field from afar)
vec3 terrainAlbedo(vec2 xz, float farGrass) {
    float mask = grassMask(xz);
    float k = vnoise(xz * 0.35) * 0.6 + vnoise(xz * 2.1) * 0.4;
    vec3 green = mix(vec3(0.10, 0.17, 0.045), vec3(0.16, 0.25, 0.06), k);
    green *= 1.0 - 0.25 * grassHeight(xz) * (1.0 - farGrass);  // tall grass shades the soil below
    green = mix(green * 0.8, mix(vec3(0.17, 0.30, 0.06), vec3(0.26, 0.33, 0.09), k), farGrass);
    vec3 soil = mix(vec3(0.19, 0.14, 0.09), vec3(0.30, 0.24, 0.16), vnoise(xz * 3.0));
    vec3 stone = mix(vec3(0.22, 0.21, 0.19), vec3(0.34, 0.33, 0.30), vnoise(xz * 4.3 + 2.0));
    vec3 bare = mix(soil, stone, smoothstep(0.45, 0.8, vnoise(xz * 0.09 + 17.0)));
    vec3 dry = vec3(0.30, 0.27, 0.13);
    float g = smoothstep(0.05, 0.6, mask);
    vec3 ground = mix(bare, green, g);
    ground = mix(ground, dry, 4.0 * g * (1.0 - g) * 0.45);
    float cr = craterCover(xz);
    // steep sculpted faces: bare soil and rock (lighter, greyer as it steepens)
    float sl = editSlope(xz);
    if (sl > 0.4) {
        vec3 rockC = mix(vec3(0.26, 0.24, 0.21), vec3(0.38, 0.36, 0.32), vnoise(xz * 0.9 + 3.0));
        vec3 soilC = mix(vec3(0.22, 0.17, 0.12), vec3(0.3, 0.24, 0.17), vnoise(xz * 2.3));
        ground = mix(ground, mix(soilC, rockC, smoothstep(0.8, 1.6, sl)), smoothstep(0.4, 0.9, sl));
    }
    if (cr > 0.0) ground = mix(ground, mix(vec3(0.15, 0.11, 0.08), vec3(0.26, 0.2, 0.14), vnoise(xz * 4.0)), cr);  // churned, fresh soil
    if (riverNear(xz, 3.0)) {  // river bed: pebbly gravel in the channel, dark wet silt along the waterline
        vec4 r = riverInfo(xz);
        float d = abs(r.x);
        vec3 gravel = mix(vec3(0.16, 0.14, 0.11), vec3(0.32, 0.29, 0.24), smoothstep(0.35, 0.75, vnoise(xz * 6.0) * 0.6 + vnoise(xz * 17.0) * 0.4));
        vec3 silt = mix(vec3(0.09, 0.08, 0.06), vec3(0.15, 0.13, 0.09), vnoise(xz * 2.3));
        ground = mix(ground, silt, 1.0 - smoothstep(r.y + 0.3, r.y + 2.5, d));
        ground = mix(ground, gravel, 1.0 - smoothstep(r.y * 0.6, r.y, d));
    }
    return mix(ground, mix(vec3(0.23, 0.22, 0.2), vec3(0.3, 0.29, 0.26), vnoise(xz * 1.7)), smoothstep(0.5, 1.0, zoneCover(xz)));
}

// Per-blade random numbers shared by shape (grass.comp) and colour (grassAlbedo).
vec4 bladeRand(uint seed) {
    uint h = hashu(seed);
    return vec4(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u), float(h >> 24)) / 255.0;
}
bool bladeDead(vec4 r)   { return r.z < F.grassD.w; }
bool bladeFlower(vec4 r) { return r.w > 0.994; }

// t = 0 at the root, 1 at the tip
vec3 grassAlbedo(uint seed, float t) {
    vec4 r = bladeRand(seed);
    vec3 root = vec3(0.04, 0.07, 0.02);
    float cv = F.vegCol.x;  // colour variation: 0 = every blade the same green
    vec3 green = mix(vec3(0.15, 0.33, 0.05), vec3(0.32, 0.42, 0.08), clamp(0.5 + (r.x - 0.5) * cv, 0.0, 1.0));
    vec3 straw = mix(vec3(0.42, 0.34, 0.16), vec3(0.56, 0.48, 0.26), r.y);
    vec3 c;
    if (bladeDead(r)) {
        c = mix(root * 1.6, straw, smoothstep(0.0, 0.6, t));
    } else {
        c = mix(root, green, smoothstep(0.0, 0.7, t));
        c = mix(c, straw, smoothstep(0.7, 1.0, t) * r.y * 0.8 * min(cv, 1.0)); // dry tips
    }
    if (bladeFlower(r) && t > 0.75) {
        vec3 petal = r.x < 0.4 ? vec3(0.85, 0.85, 0.8) : (r.x < 0.75 ? vec3(0.9, 0.75, 0.1) : vec3(0.55, 0.3, 0.75));
        c = mix(c, petal, smoothstep(0.75, 0.9, t));
    }
    return c;
}

// Procedural rock material (3D, no UVs: works on every instanced, rotated rock and on fragments' fresh faces).
// Layers, as on real weathered stone:
//  - mineral grain: Voronoi crystals at ~4 mm, each a mineral (dark mica / hornblende, pink-buff feldspar, glassy
//    quartz) - the salt-and-pepper look of granite up close
//  - broad tone variation and a faint banding (foliation), quartz veins as thin pale lines
//  - relief: ridged noise (pitted, crested surface); cavities collect dark dirt, exposed crests are bleached lighter
//  - weathering: dark water streaks running down the sides, lichen rosettes (pale grey-green, some orange) and
//    moss on the upward faces, soil splashed up where the rock meets the ground
float rnoise3(vec3 p) {
    vec3 i = floor(p), f = p - i;
    f = f * f * (3.0 - 2.0 * f);
    uvec3 u = uvec3(ivec3(i));
    #define RH(o) (float(hashu(hashu(hashu(u.x + o.x) ^ (u.y + o.y)) ^ (u.z + o.z)) & 0xFFFFFFu) / 16777216.0)
    float a = mix(RH(uvec3(0, 0, 0)), RH(uvec3(1, 0, 0)), f.x), b = mix(RH(uvec3(0, 1, 0)), RH(uvec3(1, 1, 0)), f.x);
    float c = mix(RH(uvec3(0, 0, 1)), RH(uvec3(1, 0, 1)), f.x), d = mix(RH(uvec3(0, 1, 1)), RH(uvec3(1, 1, 1)), f.x);
    #undef RH
    return mix(mix(a, b, f.y), mix(c, d, f.y), f.z);
}
// Voronoi: x = distance to the nearest cell point, y = second nearest, z = nearest cell's hash
vec3 rcell3(vec3 p) {
    vec3 i = floor(p), f = p - i;
    float d1 = 8.0, d2 = 8.0, id = 0.0;
    for (int z = -1; z <= 1; z++)
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++) {
                uvec3 c = uvec3(ivec3(i) + ivec3(x, y, z));
                uint h = hashu(hashu(hashu(c.x) ^ c.y) ^ c.z);
                vec3 o = vec3(x, y, z) + vec3(h & 0x3FFu, (h >> 10) & 0x3FFu, (h >> 20) & 0x3FFu) / 1023.0;
                float d = dot(o - f, o - f);
                if (d < d1) { d2 = d1; d1 = d; id = float(hashu(h) & 0xFFFFu) / 65535.0; }
                else if (d < d2) d2 = d;
            }
    return vec3(sqrt(d1), sqrt(d2), id);
}
float rockRelief(vec3 p) {
    float r = 0.0, a = 0.5, f = 2.5;
    for (int o = 0; o < 4; o++, a *= 0.5, f *= 2.2) r += a * (1.0 - abs(2.0 * rnoise3(p * f + float(o) * 7.1) - 1.0));
    return r * 0.9;
}
vec3 rockAlbedoC(vec3 p, vec3 n, float relief) {
    // broad tone + foliation bands
    float tone = rnoise3(p * 0.9) * 0.6 + rnoise3(p * 3.7 + 5.0) * 0.4;
    float band = 0.5 + 0.5 * sin(dot(p, vec3(0.3, 5.0, 0.8)) * 3.0 + rnoise3(p * 2.0) * 4.0);
    vec3 c = mix(vec3(0.24, 0.23, 0.21), vec3(0.42, 0.40, 0.36), tone) * (0.92 + 0.08 * band);
    // mineral grain
    vec3 g = rcell3(p * 220.0);
    vec3 mineral = g.z < 0.22 ? vec3(0.05, 0.05, 0.05) : (g.z < 0.55 ? vec3(0.50, 0.42, 0.36) : (g.z < 0.85 ? vec3(0.40, 0.39, 0.37) : vec3(0.62, 0.61, 0.58)));
    c = mix(c, mineral, 0.35);
    // quartz veins
    float vn = abs(rnoise3(p * 1.3 + 17.0) - 0.5) + 0.02 * rnoise3(p * 30.0);
    c = mix(c, vec3(0.66, 0.64, 0.6), (1.0 - smoothstep(0.004, 0.012, vn)) * 0.8);
    // dirt in cavities, bleached crests
    c *= mix(0.55, 1.12, smoothstep(0.1, 0.75, relief));
    // water streaks down the sides
    float side = 1.0 - abs(n.y);
    float streak = smoothstep(0.55, 0.8, rnoise3(vec3(p.x * 9.0, p.y * 0.6, p.z * 9.0))) * side;
    c *= 1.0 - 0.35 * streak;
    // lichen rosettes on the upper faces
    vec3 lc = rcell3(p * 6.0 + 41.0);
    float up = smoothstep(0.1, 0.6, n.y);
    float lichen = (1.0 - smoothstep(0.18, 0.3, lc.x + 0.1 * rnoise3(p * 40.0))) * step(0.45, lc.z) * up;
    vec3 lcol = lc.z > 0.9 ? vec3(0.55, 0.32, 0.08) : vec3(0.50, 0.52, 0.42);
    c = mix(c, lcol * (0.85 + 0.3 * rnoise3(p * 60.0)), lichen * 0.85);
    // moss in the damp hollows of the top
    float moss = smoothstep(0.55, 0.9, n.y) * smoothstep(0.4, 0.7, rnoise3(p * 1.3 + 7.0)) * (1.2 - relief);
    return mix(c, vec3(0.10, 0.16, 0.04) * (0.8 + 0.4 * rnoise3(p * 25.0)), clamp(moss, 0.0, 1.0));
}
vec3 rockAlbedo(vec3 p, vec3 n) { return rockAlbedoC(p, n, 0.6); }  // indirect rays: no relief
vec3 rockSurface(vec3 p, inout vec3 n, out float rough, out float ao) {
    const float e = 0.004;
    float h0 = rockRelief(p);
    vec3 g = vec3(rockRelief(p + vec3(e, 0, 0)) - h0, rockRelief(p + vec3(0, e, 0)) - h0, rockRelief(p + vec3(0, 0, e)) - h0) / e;
    g -= n * dot(g, n);
    n = normalize(n - g * 0.02);
    vec3 alb = rockAlbedoC(p, n, h0);
    float moss = smoothstep(0.55, 0.9, n.y) * smoothstep(0.4, 0.7, rnoise3(p * 1.3 + 7.0));
    rough = mix(mix(0.85, 0.6, smoothstep(0.2, 0.8, h0)), 0.95, moss);  // crests worn smoother
    ao = mix(0.45, 1.0, smoothstep(0.0, 0.7, h0));  // crevices and cracks
    float above = p.y - terrainHeight(p.xz);
    float b = 1.0 - smoothstep(0.0, 0.35, above);
    alb = mix(alb, terrainAlbedo(p.xz, 0.0) * 0.8, b * 0.75);
    ao *= mix(1.0, 0.55, b);
    return alb;
}

// Per-blade surface: waxy fresh blades, rougher dry ones.
float grassRough(uint seed) {
    vec4 r = bladeRand(seed);
    return bladeDead(r) ? 0.65 : 0.24 + 0.18 * r.z;
}


#endif
