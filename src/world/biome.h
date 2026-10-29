// Biomes, oceans, lakes and ponds on top of the natural terrain (terrain.h terrainBase calls biomeShape).
// The world is split into Voronoi climate cells (gBiome.size m apart); each cell picks a biome from a temperature /
// moisture field (+ the per-biome amount sliders) and neighbouring cells blend over gBiome.blend. Every biome reshapes
// the ground (dunes, terraced mesas cut by canyons, ridged mountains, volcanoes, karst towers, flats...) and has its
// own ground colour, grass and tree cover. Continents: a low-frequency field below a threshold is ocean (sea floor,
// beaches). Lakes / ponds: one candidate per cell, a bowl filled to a flat level with a flattened shore.
// Mirrored in shaders/terrain/biome.glsl (UBO biomeA / biomeP) - keep in sync.
#pragma once
#include <algorithm>
#include <cmath>

enum Biome {
    B_TEMPERATE, B_DESERT, B_MESA, B_SAVANNA, B_TUNDRA, B_GLACIER, B_ALPINE, B_TAIGA, B_JUNGLE, B_WETLAND, B_VOLCANIC,
    B_SALT, B_KARST, B_COUNT
};
inline const char* const BIOME_NAMES[B_COUNT] = {"Temperate", "Desert (dunes)", "Mesa / canyons", "Savanna", "Tundra",
                                                 "Glacier / snow", "Alpine", "Taiga", "Jungle", "Wetland / marsh",
                                                 "Volcanic", "Salt flats", "Karst"};
// climate each biome prefers (temperature, moisture, 0..1)
inline constexpr float BIOME_CLIMATE[B_COUNT][2] = {{0.55f, 0.55f}, {0.92f, 0.08f}, {0.8f, 0.25f}, {0.8f, 0.45f}, {0.15f, 0.4f},
                                                    {0.04f, 0.6f},  {0.3f, 0.3f},   {0.3f, 0.68f}, {0.92f, 0.92f}, {0.6f, 0.92f},
                                                    {0.65f, 0.15f}, {0.88f, 0.0f},  {0.65f, 0.75f}};
// grass and tree cover multipliers
inline constexpr float BIOME_GRASS[B_COUNT] = {1.f, 0.f, 0.08f, 0.9f, 0.55f, 0.f, 0.3f, 0.7f, 1.f, 1.f, 0.05f, 0.f, 0.8f};
inline constexpr float BIOME_TREES[B_COUNT] = {1.f, 0.f, 0.04f, 0.12f, 0.03f, 0.f, 0.15f, 1.7f, 2.3f, 0.4f, 0.04f, 0.f, 0.9f};

struct BiomeParams {
    bool on = true;
    float amount[B_COUNT] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.6f, 0.6f, 0.8f};
    float size = 1800.f, blend = 0.18f, spawn = 450.f;   // m between cell centres, blend (share of size), untouched radius
    float temp = 0.f, moist = 0.f;                        // climate offsets
    float ocean = 0.35f, oceanScale = 9000.f, seaLevel = -9.f, oceanDepth = 60.f, beach = 60.f;
    float lakes = 0.45f, lakeSize = 1.f, ponds = 0.35f;
    float duneH = 14.f, duneL = 90.f, wind = 0.6f;        // dune height, wavelength (m), wind direction (rad)
    float mesaH = 110.f, terrace = 9.f, canyonD = 90.f;
    float mountainH = 320.f, glacierH = 180.f, volcanoH = 260.f, karstH = 70.f;
    float relief = 1.f;                                   // every biome's shape
};
inline BiomeParams gBiome;

// (included from terrain.h after hash2 / vnoise / riverEdgeDist)

inline float bSstep(float a, float b, float v) { float t = std::clamp((v - a) / (b - a), 0.f, 1.f); return t * t * (3.f - 2.f * t); }
inline float bSpawnFade(float x, float z) { return bSstep(gBiome.spawn, gBiome.spawn + 500.f, sqrtf(x * x + z * z)); }

// ---- climate cells
struct BiomeCell { float px, pz, w; int b; };
inline int biomeOfCell(int cx, int cz, float px, float pz) {
    float s = gBiome.size;
    float T = vnoise(px / (3.7f * s) + 41.f, pz / (3.7f * s) - 9.f) * 0.75f + vnoise(px / (1.3f * s) - 3.f, pz / (1.3f * s) + 17.f) * 0.25f;
    float M = vnoise(px / (3.1f * s) - 77.f, pz / (3.1f * s) + 23.f) * 0.75f + vnoise(px / (1.1f * s) + 5.f, pz / (1.1f * s) - 31.f) * 0.25f;
    T = std::clamp((T - 0.5f) * 1.9f + 0.5f + gBiome.temp, 0.f, 1.f);
    M = std::clamp((M - 0.5f) * 1.9f + 0.5f + gBiome.moist, 0.f, 1.f);
    int best = 0;
    float bs = -1.f;
    for (int b = 0; b < B_COUNT; b++) {
        if (gBiome.amount[b] <= 0.f) continue;
        float dt = T - BIOME_CLIMATE[b][0], dm = M - BIOME_CLIMATE[b][1];
        float sc = gBiome.amount[b] * expf(-(dt * dt + dm * dm) / 0.06f) * (0.35f + hash2(cx * 31 + b * 7, cz * 17 + 911));
        if (sc > bs) { bs = sc; best = b; }
    }
    return best;
}
// the (up to 4) cells that matter at x, z with normalised blend weights; returns the count. Every cell centre within
// the blend width of the nearest is a candidate (5x5 search: blends up to a whole cell); the 4 strongest are kept and
// weighted by how far they beat the 5th, so a cell fades to nothing before it's dropped (no seams in the ground)
inline int biomeCells(float x, float z, BiomeCell out[4]) {
    float s = gBiome.size;
    int cx = (int)floorf(x / s), cz = (int)floorf(z / s);
    float d[25], px[25], pz[25], dmin = 1e30f;
    for (int j = 0; j < 25; j++) {
        int ix = cx + j % 5 - 2, iz = cz + j / 5 - 2;
        px[j] = (ix + 0.15f + 0.7f * hash2(ix * 3 + 1201, iz + 77)) * s;
        pz[j] = (iz + 0.15f + 0.7f * hash2(ix + 1307, iz * 3 + 19)) * s;
        d[j] = sqrtf((x - px[j]) * (x - px[j]) + (z - pz[j]) * (z - pz[j]));
        dmin = std::min(dmin, d[j]);
    }
    float bw = std::clamp(gBiome.blend, 0.f, 1.f) * s;
    bw = std::max(bw, 1.f);
    float tk[5] = {0.f, 0.f, 0.f, 0.f, 0.f};  // the 5 strongest, descending
    int jk[5] = {-1, -1, -1, -1, -1};
    for (int j = 0; j < 25; j++) {
        float t = 1.f - (d[j] - dmin) / bw;
        if (t <= tk[4]) continue;
        int i = 4;
        for (; i > 0 && tk[i - 1] < t; i--) { tk[i] = tk[i - 1]; jk[i] = jk[i - 1]; }
        tk[i] = t; jk[i] = j;
    }
    float sum = 0.f;
    int n = 0;
    for (int i = 0; i < 4 && jk[i] >= 0; i++) {
        float w = tk[i] - tk[4];
        if (w <= 0.f) break;
        int j = jk[i], ix = cx + j % 5 - 2, iz = cz + j / 5 - 2;
        out[n++] = {px[j], pz[j], w * w, biomeOfCell(ix, iz, px[j], pz[j])};
        sum += w * w;
    }
    for (int i = 0; i < n; i++) out[i].w /= sum;
    return n;
}
// weight per biome at x, z (spawn area = temperate)
inline void biomeWeights(float x, float z, float w[B_COUNT]) {
    for (int b = 0; b < B_COUNT; b++) w[b] = 0.f;
    float k = gBiome.on ? bSpawnFade(x, z) : 0.f;
    w[B_TEMPERATE] = 1.f - k;
    if (k <= 0.f) return;
    BiomeCell c[4];
    int n = biomeCells(x, z, c);
    for (int i = 0; i < n; i++) w[c[i].b] += k * c[i].w;
}

// ---- continents: metres from the coast (+ inland, - out to sea), roughly
inline float coastDist(float x, float z) {
    if (!gBiome.on || gBiome.ocean <= 0.f) return 1e9f;
    float f = 1.f / gBiome.oceanScale;
    float c = vnoise(x * f + 13.f, z * f - 29.f) * 0.6f + vnoise(x * f * 2.3f - 7.f, z * f * 2.3f + 3.f) * 0.28f +
              vnoise(x * f * 6.1f + 1.f, z * f * 6.1f + 9.f) * 0.12f;
    c += 1.f - bSstep(gBiome.spawn + 800.f, gBiome.spawn + 3500.f, sqrtf(x * x + z * z));  // land around the spawn
    return (c - gBiome.ocean * 0.75f - 0.12f) * gBiome.oceanScale * 1.4f;
}

// ---- per-biome ground shapes (h0: the natural ground with hills; cx, cz: the biome cell's centre)
inline float ridged(float x, float z, float scale, int oct) {
    float h = 0.f, a = 0.5f, f = 1.f / scale;
    for (int o = 0; o < oct; o++) {
        float n = 1.f - fabsf(2.f * vnoise(x * f + 57.3f * o, z * f - 21.1f * o) - 1.f);
        h += a * n * n;
        a *= 0.5f;
        f *= 2.05f;
    }
    return h;
}
inline float duneShape(float x, float z) {
    float cw = cosf(gBiome.wind), sw = sinf(gBiome.wind), L = std::max(gBiome.duneL, 10.f);
    float al = x * cw + z * sw, cr = -x * sw + z * cw;
    float u = al / L + 0.9f * vnoise(cr / (1.7f * L) + 3.f, al / (4.f * L)) + 0.3f * vnoise(x / (0.6f * L), z / (0.6f * L));
    float f = u - floorf(u);
    float s = f < 0.72f ? f / 0.72f : (1.f - f) / 0.28f;  // long windward slope, steep slip face
    s = s * s * (3.f - 2.f * s);
    float amp = 0.45f + 0.55f * vnoise(cr / (2.5f * L) - 11.f, al / (3.f * L) + 5.f);
    return gBiome.duneH * amp * s + 0.12f * gBiome.duneH * vnoise(x / (8.f * L), z / (8.f * L));
}
inline float terraceF(float h, float step) {
    if (step <= 0.1f) return h;
    float k = h / step, i = floorf(k), f = k - i;
    return (i + bSstep(0.7f, 1.f, f)) * step;
}
inline float biomeShapeOne(int b, float x, float z, float h0, float cx, float cz) {
    float R = gBiome.relief;
    switch (b) {
    case B_DESERT: return h0 * 0.35f + R * duneShape(x, z);
    case B_MESA: {
        float p = bSstep(0.42f, 0.58f, vnoise(x / 650.f + 3.f, z / 650.f - 8.f) * 0.7f + vnoise(x / 230.f, z / 230.f) * 0.3f);
        float r = fabsf(vnoise(x / 520.f - 19.f, z / 520.f + 4.f) - 0.5f) * 2.f;
        float canyon = gBiome.canyonD * (1.f - bSstep(0.02f, 0.11f, r));
        float m = R * (gBiome.mesaH * (0.25f + 0.75f * p) + 6.f * vnoise(x / 40.f, z / 40.f) - canyon);
        return h0 * 0.3f + terraceF(std::max(m, -gBiome.canyonD * 0.6f), gBiome.terrace);
    }
    case B_SAVANNA: return h0 * 0.5f + R * 4.f * vnoise(x / 90.f + 2.f, z / 90.f);
    case B_TUNDRA: return h0 * 0.3f + R * (1.5f * vnoise(x / 40.f, z / 40.f + 7.f) + 6.f * vnoise(x / 400.f, z / 400.f));
    case B_GLACIER: return h0 + R * gBiome.glacierH * (0.35f + 0.65f * (vnoise(x / 1500.f + 9.f, z / 1500.f) * 0.7f + ridged(x, z, 700.f, 3) * 0.3f));
    case B_ALPINE: return h0 + R * gBiome.mountainH * ridged(x, z, 950.f, 5);
    case B_TAIGA: return h0 + R * 12.f * vnoise(x / 300.f - 4.f, z / 300.f + 6.f);
    case B_JUNGLE: return h0 + R * 30.f * ridged(x, z, 260.f, 3);
    case B_WETLAND: return h0 * 0.12f - 0.6f + R * 0.8f * vnoise(x / 25.f, z / 25.f);
    case B_VOLCANIC: {
        float d = sqrtf((x - cx) * (x - cx) + (z - cz) * (z - cz)), Rc = 0.42f * gBiome.size;
        float t = std::max(1.f - d / Rc, 0.f), cone = gBiome.volcanoH * t * sqrtf(t);
        float cal = 0.07f * gBiome.size, calD = gBiome.volcanoH * 0.22f * (1.f - bSstep(cal * 0.6f, cal, d));
        return h0 * 0.5f + R * (cone - calD + 4.f * vnoise(x / 30.f + 1.f, z / 30.f) * (1.f - t));
    }
    case B_SALT: return h0 * 0.03f + 0.4f + 0.05f * vnoise(x / 8.f, z / 8.f);
    case B_KARST: {
        constexpr float C = 70.f;
        int ix = (int)floorf(x / C), iz = (int)floorf(z / C);
        float f1 = 1e9f;
        for (int j = 0; j < 9; j++) {
            int a = ix + j % 3 - 1, c = iz + j / 3 - 1;
            float px = (a + 0.2f + 0.6f * hash2(a + 4441, c)) * C, pz = (c + 0.2f + 0.6f * hash2(a, c + 6151)) * C;
            float hgt = hash2(a + 77, c + 99);
            if (hgt < 0.45f) continue;
            float dd = sqrtf((x - px) * (x - px) + (z - pz) * (z - pz)) / (C * (0.25f + 0.3f * hgt));
            f1 = std::min(f1, dd / hgt);
        }
        float tower = std::max(1.f - f1, 0.f);
        tower = powf(tower * tower * (3.f - 2.f * tower), 0.6f);  // steep sides, but a flared foot: no step the mesh shows as a wall
        return h0 + R * (gBiome.karstH * tower * (0.8f + 0.2f * vnoise(x / 9.f, z / 9.f)) + 5.f * vnoise(x / 150.f, z / 150.f));
    }
    default: return h0;
    }
}

// riverEdge: metres from the river's water (shapes fade into its valley)
inline float biomeShape(float x, float z, float h0) {
    if (!gBiome.on) return h0;
    float h = h0, k = bSpawnFade(x, z);
    if (k > 0.f) {
        BiomeCell c[4];
        int n = biomeCells(x, z, c);
        float s = 0.f;
        for (int i = 0; i < n; i++) s += c[i].w * (c[i].b == B_TEMPERATE ? h0 : biomeShapeOne(c[i].b, x, z, h0, c[i].px, c[i].pz));
        h = h0 + (s - h0) * k * bSstep(10.f, 120.f, riverEdgeDist(x, z));
    }
    float cd = coastDist(x, z);
    if (cd < gBiome.beach + 400.f) {  // beach rising out of the sea, sea floor falling away
        float sl = gBiome.seaLevel;
        float shore = cd >= 0.f ? sl + 0.5f + cd * 0.02f : sl + 0.5f - gBiome.oceanDepth * (1.f - expf(cd / 350.f));
        h = std::min(h, shore + (h - shore) * bSstep(gBiome.beach, gBiome.beach + 400.f, cd));
        if (cd < 0.f) h = std::min(h, shore);
    }
    return h;
}

// ---- lakes and ponds: one candidate per cell, wholly inside it
struct Lake { float x, z, r, depth; };
inline constexpr float LAKE_CELL = 1200.f, POND_CELL = 200.f;
// the shore reaches LAKE_REACH radii (past the shore noise's shrink), and the centre is >= 0.38 cells from the edge:
// bigger and the shore would be cut off at the cell border in a cliff
inline constexpr float LAKE_REACH = 2.f, LAKE_MAX_R = 0.38f * LAKE_CELL / LAKE_REACH;
inline bool lakeIn(int ix, int iz, bool pond, Lake& L) {
    float C = pond ? POND_CELL : LAKE_CELL, chance = pond ? gBiome.ponds : gBiome.lakes;
    int salt = pond ? 5003 : 0;
    if (!gBiome.on || chance <= 0.f) return false;
    L.x = (ix + 0.38f + 0.24f * hash2(ix + 3331 + salt, iz)) * C;
    L.z = (iz + 0.38f + 0.24f * hash2(ix, iz + 4421 + salt)) * C;
    if (sqrtf(L.x * L.x + L.z * L.z) < gBiome.spawn + 150.f || hash2(ix * 7 + 11 + salt, iz * 13 + 5) > chance * 2.5f) return false;
    float w[B_COUNT];
    biomeWeights(L.x, L.z, w);
    float like = 0.f;  // wet places hold water, deserts / mountains rarely do
    static constexpr float LAKE_LIKE[B_COUNT] = {1.f, 0.05f, 0.1f, 0.4f, 1.3f, 0.6f, 0.6f, 1.3f, 0.7f, 2.5f, 0.3f, 0.f, 0.8f};
    for (int b = 0; b < B_COUNT; b++) like += w[b] * LAKE_LIKE[b];
    if (hash2(ix * 7 + 11 + salt, iz * 13 + 5) > chance * like) return false;
    float sz = std::clamp(gBiome.lakeSize, 0.2f, 1.6f);
    L.r = pond ? (8.f + 22.f * hash2(ix + 91 + salt, iz + 3)) : std::min((60.f + 170.f * hash2(ix + 91, iz + 3)) * sz, LAKE_MAX_R);
    L.depth = pond ? 0.6f + 1.2f * hash2(ix, iz + 1717) : 4.f + 10.f * hash2(ix + 1717, iz);
    return coastDist(L.x, L.z) > L.r * 2.f;
}
// height of the ground before lakes (terrain_query.cpp) - a lake fills to just under its centre's ground
float lakeFreeHeight(float x, float z);
inline bool lakeNear(float x, float z, bool pond, Lake& L, float& dn) {
    float C = pond ? POND_CELL : LAKE_CELL;
    if (!lakeIn((int)floorf(x / C), (int)floorf(z / C), pond, L)) return false;
    dn = sqrtf((x - L.x) * (x - L.x) + (z - L.z) * (z - L.z)) / L.r;
    return dn < LAKE_REACH;
}
// the lake (else pond) reaching x, z
inline bool lakeAt(float x, float z, Lake& L, float& dn) {
    if (!gBiome.on) return false;
    return lakeNear(x, z, false, L, dn) || lakeNear(x, z, true, L, dn);
}
// bowl inside, shore flattened to the level around it (hc: the ground at the centre)
inline float lakeShape(float x, float z, float h, const Lake& L, float dn, float hc) {
    dn *= 1.f + 0.18f * (vnoise(x * 0.03f + 7.f, z * 0.03f) - 0.5f);
    if (dn < 1.f) return hc - 0.1f - L.depth * (1.f - dn * dn);  // (not min with h: downhill the rim is a dam, no step)
    return hc + (h - hc) * bSstep(1.f, 1.8f, dn);
}
inline float lakeCarve(float x, float z, float h) {
    Lake L;
    float dn;
    return lakeAt(x, z, L, dn) ? lakeShape(x, z, h, L, dn, lakeFreeHeight(L.x, L.z)) : h;
}
// standing water (sea, lakes, ponds) surface over x, z; -1e9 = none
inline float standingWaterAt(float x, float z) {
    if (!gBiome.on) return -1e9f;
    if (gBiome.ocean > 0.f && coastDist(x, z) < gBiome.beach) return gBiome.seaLevel;
    Lake L;
    float dn;
    if (lakeAt(x, z, L, dn) && dn < 1.15f) return lakeFreeHeight(L.x, L.z) - 0.4f;
    return -1e9f;
}
inline bool inStandingWater(float x, float z) {
    if (!gBiome.on) return false;
    if (gBiome.ocean > 0.f && coastDist(x, z) < 2.f) return true;
    Lake L;
    float dn;
    return lakeAt(x, z, L, dn) && dn < 1.05f;
}
inline float biomeGrassMul(float x, float z) {
    float w[B_COUNT], g = 0.f;
    biomeWeights(x, z, w);
    for (int b = 0; b < B_COUNT; b++) g += w[b] * BIOME_GRASS[b];
    return inStandingWater(x, z) ? 0.f : g;
}
// Where plants can live, from the ground itself: thin air on the heights (trees give out first, then shrubs and
// grass) and faces too steep to hold soil or roots. h: ground height (m), slope: rise over run. Mirrored in
// biome.glsl vegHabitat.
inline constexpr float TREE_LINE = 110.f, SHRUB_LINE = 150.f, VEG_LINE_FADE = 40.f;
inline constexpr float TREE_MAX_SLOPE = 0.7f, SHRUB_MAX_SLOPE = 1.f;  // ~35 deg, 45 deg
inline float vegHabitat(float h, float slope, bool tree) {
    float line = tree ? TREE_LINE : SHRUB_LINE, ms = tree ? TREE_MAX_SLOPE : SHRUB_MAX_SLOPE;
    return (1.f - bSstep(line, line + VEG_LINE_FADE, h)) * (1.f - bSstep(ms * 0.75f, ms, slope));
}
// grass / tree cover (1 = as the temperate world)
inline float biomeTreeMul(float x, float z) {
    float w[B_COUNT], t = 0.f;
    biomeWeights(x, z, w);
    for (int b = 0; b < B_COUNT; b++) t += w[b] * BIOME_TREES[b];
    return inStandingWater(x, z) ? 0.f : t;
}
