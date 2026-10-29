// Biomes, oceans, lakes and ponds (mirrors world/biome.h - keep in sync). Included from height.glsl after farRelief.
#ifndef TERRAIN_BIOME
#define TERRAIN_BIOME

const int B_TEMPERATE = 0, B_DESERT = 1, B_MESA = 2, B_SAVANNA = 3, B_TUNDRA = 4, B_GLACIER = 5, B_ALPINE = 6, B_TAIGA = 7,
          B_JUNGLE = 8, B_WETLAND = 9, B_VOLCANIC = 10, B_SALT = 11, B_KARST = 12, B_COUNT = 13;
const vec2 BIOME_CLIMATE[13] = vec2[](vec2(0.55, 0.55), vec2(0.92, 0.08), vec2(0.8, 0.25), vec2(0.8, 0.45), vec2(0.15, 0.4),
                                      vec2(0.04, 0.6), vec2(0.3, 0.3), vec2(0.3, 0.68), vec2(0.92, 0.92), vec2(0.6, 0.92),
                                      vec2(0.65, 0.15), vec2(0.88, 0.0), vec2(0.65, 0.75));
const float BIOME_GRASS[13] = float[](1.0, 0.0, 0.08, 0.9, 0.55, 0.0, 0.3, 0.7, 1.0, 1.0, 0.05, 0.0, 0.8);
const float LAKE_LIKE[13] = float[](1.0, 0.05, 0.1, 0.4, 1.3, 0.6, 0.6, 1.3, 0.7, 2.5, 0.3, 0.0, 0.8);

// F.biomeP: [0] on, size, blend, spawn  [1] temp, moist, ocean, oceanScale  [2] seaLevel, oceanDepth, beach, lakes
// [3] lakeSize, ponds, duneH, duneL  [4] wind, mesaH, terrace, canyonD  [5] mountainH, glacierH, volcanoH, karstH  [6] relief
bool biomeOn() { return F.biomeP[0].x > 0.5; }
float bAmount(int b) { return F.biomeA[b >> 2][b & 3]; }
float bSpawnFade(vec2 p) { return smoothstep(F.biomeP[0].w, F.biomeP[0].w + 500.0, length(p)); }

int biomeOfCell(int cx, int cz, vec2 c) {
    float s = F.biomeP[0].y;
    float T = tnoise(c.x / (3.7 * s) + 41.0, c.y / (3.7 * s) - 9.0) * 0.75 + tnoise(c.x / (1.3 * s) - 3.0, c.y / (1.3 * s) + 17.0) * 0.25;
    float M = tnoise(c.x / (3.1 * s) - 77.0, c.y / (3.1 * s) + 23.0) * 0.75 + tnoise(c.x / (1.1 * s) + 5.0, c.y / (1.1 * s) - 31.0) * 0.25;
    T = clamp((T - 0.5) * 1.9 + 0.5 + F.biomeP[1].x, 0.0, 1.0);
    M = clamp((M - 0.5) * 1.9 + 0.5 + F.biomeP[1].y, 0.0, 1.0);
    int best = 0;
    float bs = -1.0;
    for (int b = 0; b < B_COUNT; b++) {
        float a = bAmount(b);
        if (a <= 0.0) continue;
        vec2 d = vec2(T, M) - BIOME_CLIMATE[b];
        float sc = a * exp(-dot(d, d) / 0.06) * (0.35 + thash(cx * 31 + b * 7, cz * 17 + 911));
        if (sc > bs) { bs = sc; best = b; }
    }
    return best;
}
// up to 4 cells: xy centre, z weight, w biome (biome.h biomeCells: 5x5 search, the 4 strongest weighted by how far
// they beat the 5th, so none drops out abruptly)
int biomeCells(vec2 p, out vec4 o[4]) {
    float s = F.biomeP[0].y;
    int cx = int(floor(p.x / s)), cz = int(floor(p.y / s));
    float d[25];
    vec2 pc[25];
    float dmin = 1e30;
    for (int j = 0; j < 25; j++) {
        int ix = cx + j % 5 - 2, iz = cz + j / 5 - 2;
        pc[j] = vec2((float(ix) + 0.15 + 0.7 * thash(ix * 3 + 1201, iz + 77)) * s, (float(iz) + 0.15 + 0.7 * thash(ix + 1307, iz * 3 + 19)) * s);
        d[j] = length(p - pc[j]);
        dmin = min(dmin, d[j]);
    }
    float bw = max(clamp(F.biomeP[0].z, 0.0, 1.0) * s, 1.0);
    float tk[5] = float[5](0.0, 0.0, 0.0, 0.0, 0.0);
    int jk[5] = int[5](-1, -1, -1, -1, -1);
    for (int j = 0; j < 25; j++) {
        float t = 1.0 - (d[j] - dmin) / bw;
        if (t <= tk[4]) continue;
        int i = 4;
        for (; i > 0 && tk[i - 1] < t; i--) { tk[i] = tk[i - 1]; jk[i] = jk[i - 1]; }
        tk[i] = t; jk[i] = j;
    }
    float sum = 0.0;
    int n = 0;
    for (int i = 0; i < 4 && jk[i] >= 0; i++) {
        float w = tk[i] - tk[4];
        if (w <= 0.0) break;
        int j = jk[i], ix = cx + j % 5 - 2, iz = cz + j / 5 - 2;
        o[n++] = vec4(pc[j], w * w, float(biomeOfCell(ix, iz, pc[j])));
        sum += w * w;
    }
    for (int i = 0; i < n; i++) o[i].z /= sum;
    return n;
}
void biomeWeights(vec2 p, out float w[13]) {
    for (int b = 0; b < B_COUNT; b++) w[b] = 0.0;
    float k = biomeOn() ? bSpawnFade(p) : 0.0;
    w[B_TEMPERATE] = 1.0 - k;
    if (k <= 0.0) return;
    vec4 c[4];
    int n = biomeCells(p, c);
    for (int i = 0; i < n; i++) w[int(c[i].w)] += k * c[i].z;
}

float coastDist(vec2 p) {
    if (!biomeOn() || F.biomeP[1].z <= 0.0) return 1e9;
    float f = 1.0 / F.biomeP[1].w;
    float c = tnoise(p.x * f + 13.0, p.y * f - 29.0) * 0.6 + tnoise(p.x * f * 2.3 - 7.0, p.y * f * 2.3 + 3.0) * 0.28 +
              tnoise(p.x * f * 6.1 + 1.0, p.y * f * 6.1 + 9.0) * 0.12;
    c += 1.0 - smoothstep(F.biomeP[0].w + 800.0, F.biomeP[0].w + 3500.0, length(p));
    return (c - F.biomeP[1].z * 0.75 - 0.12) * F.biomeP[1].w * 1.4;
}

float bRidged(vec2 p, float scale, int oct) {
    float h = 0.0, a = 0.5, f = 1.0 / scale;
    for (int o = 0; o < oct; o++) {
        float n = 1.0 - abs(2.0 * tnoise(p.x * f + 57.3 * float(o), p.y * f - 21.1 * float(o)) - 1.0);
        h += a * n * n;
        a *= 0.5;
        f *= 2.05;
    }
    return h;
}
float duneShape(vec2 p) {
    float wd = F.biomeP[4].x, L = max(F.biomeP[3].w, 10.0), H = F.biomeP[3].z;
    float cw = cos(wd), sw = sin(wd);
    float al = p.x * cw + p.y * sw, cr = -p.x * sw + p.y * cw;
    float u = al / L + 0.9 * tnoise(cr / (1.7 * L) + 3.0, al / (4.0 * L)) + 0.3 * tnoise(p.x / (0.6 * L), p.y / (0.6 * L));
    float f = fract(u);
    float s = f < 0.72 ? f / 0.72 : (1.0 - f) / 0.28;
    s = s * s * (3.0 - 2.0 * s);
    float amp = 0.45 + 0.55 * tnoise(cr / (2.5 * L) - 11.0, al / (3.0 * L) + 5.0);
    return H * amp * s + 0.12 * H * tnoise(p.x / (8.0 * L), p.y / (8.0 * L));
}
float terraceF(float h, float st) {
    if (st <= 0.1) return h;
    float k = h / st, i = floor(k), f = k - i;
    return (i + smoothstep(0.7, 1.0, f)) * st;
}
float groundBase(vec2 p);  // (height.glsl)
float biomeShapeOne(int b, vec2 p, float h0, vec2 c) {
    float R = F.biomeP[6].x, x = p.x, z = p.y;
    if (b == B_DESERT) return h0 * 0.35 + R * duneShape(p);
    if (b == B_MESA) {
        float pl = smoothstep(0.42, 0.58, tnoise(x / 650.0 + 3.0, z / 650.0 - 8.0) * 0.7 + tnoise(x / 230.0, z / 230.0) * 0.3);
        float r = abs(tnoise(x / 520.0 - 19.0, z / 520.0 + 4.0) - 0.5) * 2.0;
        float cd = F.biomeP[4].w, canyon = cd * (1.0 - smoothstep(0.02, 0.11, r));
        float m = R * (F.biomeP[4].y * (0.25 + 0.75 * pl) + 6.0 * tnoise(x / 40.0, z / 40.0) - canyon);
        return h0 * 0.3 + terraceF(max(m, -cd * 0.6), F.biomeP[4].z);
    }
    if (b == B_SAVANNA) return h0 * 0.5 + R * 4.0 * tnoise(x / 90.0 + 2.0, z / 90.0);
    if (b == B_TUNDRA) return h0 * 0.3 + R * (1.5 * tnoise(x / 40.0, z / 40.0 + 7.0) + 6.0 * tnoise(x / 400.0, z / 400.0));
    if (b == B_GLACIER) return h0 + R * F.biomeP[5].y * (0.35 + 0.65 * (tnoise(x / 1500.0 + 9.0, z / 1500.0) * 0.7 + bRidged(p, 700.0, 3) * 0.3));
    if (b == B_ALPINE) return h0 + R * F.biomeP[5].x * bRidged(p, 950.0, 5);
    if (b == B_TAIGA) return h0 + R * 12.0 * tnoise(x / 300.0 - 4.0, z / 300.0 + 6.0);
    if (b == B_JUNGLE) return h0 + R * 30.0 * bRidged(p, 260.0, 3);
    if (b == B_WETLAND) return h0 * 0.12 - 0.6 + R * 0.8 * tnoise(x / 25.0, z / 25.0);
    if (b == B_VOLCANIC) {
        float d = length(p - c), Rc = 0.42 * F.biomeP[0].y, H = F.biomeP[5].z;
        float t = max(1.0 - d / Rc, 0.0), cone = H * t * sqrt(t);
        float cal = 0.07 * F.biomeP[0].y, calD = H * 0.22 * (1.0 - smoothstep(cal * 0.6, cal, d));
        return h0 * 0.5 + R * (cone - calD + 4.0 * tnoise(x / 30.0 + 1.0, z / 30.0) * (1.0 - t));
    }
    if (b == B_SALT) return h0 * 0.03 + 0.4 + 0.05 * tnoise(x / 8.0, z / 8.0);
    if (b == B_KARST) {
        const float C = 70.0;
        int ix = int(floor(x / C)), iz = int(floor(z / C));
        float f1 = 1e9;
        for (int j = 0; j < 9; j++) {
            int a = ix + j % 3 - 1, cc = iz + j / 3 - 1;
            vec2 pc = vec2((float(a) + 0.2 + 0.6 * thash(a + 4441, cc)) * C, (float(cc) + 0.2 + 0.6 * thash(a, cc + 6151)) * C);
            float hg = thash(a + 77, cc + 99);
            if (hg < 0.45) continue;
            f1 = min(f1, length(p - pc) / (C * (0.25 + 0.3 * hg)) / hg);
        }
        float tower = max(1.0 - f1, 0.0);
        tower = pow(tower * tower * (3.0 - 2.0 * tower), 0.6);  // (biome.h)
        return h0 + R * (F.biomeP[5].w * tower * (0.8 + 0.2 * tnoise(x / 9.0, z / 9.0)) + 5.0 * tnoise(x / 150.0, z / 150.0));
    }
    return h0;
}
float biomeShape(vec2 p, float h0) {
    if (!biomeOn()) return h0;
    float h = h0, k = bSpawnFade(p);
    if (k > 0.0) {
        vec4 c[4];
        int n = biomeCells(p, c);
        float s = 0.0;
        for (int i = 0; i < n; i++) { int b = int(c[i].w); s += c[i].z * (b == B_TEMPERATE ? h0 : biomeShapeOne(b, p, h0, c[i].xy)); }
        vec4 r = riverInfo(p);
        h = h0 + (s - h0) * k * smoothstep(10.0, 120.0, abs(r.x) - r.y);
    }
    float cd = coastDist(p), beach = F.biomeP[2].z;
    if (cd < beach + 400.0) {
        float sl = F.biomeP[2].x;
        float shore = cd >= 0.0 ? sl + 0.5 + cd * 0.02 : sl + 0.5 - F.biomeP[2].y * (1.0 - exp(cd / 350.0));
        h = min(h, shore + (h - shore) * smoothstep(beach, beach + 400.0, cd));
        if (cd < 0.0) h = min(h, shore);
    }
    return h;
}

// ---- lakes and ponds: xy centre, z radius, w depth
const float LAKE_CELL = 1200.0, POND_CELL = 200.0;
const float LAKE_REACH = 2.0, LAKE_MAX_R = 0.38 * LAKE_CELL / LAKE_REACH;  // (biome.h)
bool lakeIn(int ix, int iz, bool pond, out vec4 L) {
    L = vec4(0);
    float C = pond ? POND_CELL : LAKE_CELL, chance = pond ? F.biomeP[3].y : F.biomeP[2].w;
    int salt = pond ? 5003 : 0;
    if (!biomeOn() || chance <= 0.0) return false;
    L.x = (float(ix) + 0.38 + 0.24 * thash(ix + 3331 + salt, iz)) * C;
    L.y = (float(iz) + 0.38 + 0.24 * thash(ix, iz + 4421 + salt)) * C;
    if (length(L.xy) < F.biomeP[0].w + 150.0 || thash(ix * 7 + 11 + salt, iz * 13 + 5) > chance * 2.5) return false;
    float w[13];
    biomeWeights(L.xy, w);
    float like = 0.0;
    for (int b = 0; b < B_COUNT; b++) like += w[b] * LAKE_LIKE[b];
    if (thash(ix * 7 + 11 + salt, iz * 13 + 5) > chance * like) return false;
    float sz = clamp(F.biomeP[3].x, 0.2, 1.6);
    L.z = pond ? (8.0 + 22.0 * thash(ix + 91 + salt, iz + 3)) : min((60.0 + 170.0 * thash(ix + 91, iz + 3)) * sz, LAKE_MAX_R);
    L.w = pond ? 0.6 + 1.2 * thash(ix, iz + 1717) : 4.0 + 10.0 * thash(ix + 1717, iz);
    return coastDist(L.xy) > L.z * 2.0;
}
float lakeFreeHeight(vec2 p) {
    float h = 2.2 * tnoise(p.x * 0.022 + 10.0, p.y * 0.022) + 0.7 * tnoise(p.x * 0.07, p.y * 0.07 + 5.0) +
              0.12 * tnoise(p.x * 0.35, p.y * 0.35) - 1.5 + hills(p) + farRelief(p);
    return riverCarve(p, biomeShape(p, h) + editHeight(p));
}
bool lakeNear(vec2 p, bool pond, out vec4 L, out float dn) {
    float C = pond ? POND_CELL : LAKE_CELL;
    dn = 1e9;
    if (!lakeIn(int(floor(p.x / C)), int(floor(p.y / C)), pond, L)) return false;
    dn = length(p - L.xy) / L.z;
    return dn < LAKE_REACH;
}
// the lake (else pond) reaching p; the loops stay loops so the heavy calls aren't inlined twice
bool lakeAt(vec2 p, out vec4 L, out float dn) {
    L = vec4(0);
    dn = 1e9;
    if (!biomeOn()) return false;
    for (int k = 0; k < 2 + int(F.biomeP[6].w); k++)
        if (lakeNear(p, k == 1, L, dn)) return true;
    return false;
}
// the ground around a lake whose centre ground is hc (biome.h lakeShape)
float lakeShape(vec2 p, float h, vec4 L, float dn, float hc) {
    float lv = hc;  // (the shore: just above the water, 0.4 m under the centre ground)
    dn *= 1.0 + 0.18 * (tnoise(p.x * 0.03 + 7.0, p.y * 0.03) - 0.5);
    if (dn < 1.0) return lv - 0.1 - L.w * (1.0 - dn * dn);  // (biome.h)
    return lv + (h - lv) * smoothstep(1.0, 1.8, dn);
}
float standingWaterAt(vec2 p) {
    if (!biomeOn()) return -1e9;
    if (F.biomeP[1].z > 0.0 && coastDist(p) < F.biomeP[2].z) return F.biomeP[2].x;
    vec4 L;
    float dn;
    if (lakeAt(p, L, dn) && dn < 1.15) return lakeFreeHeight(L.xy) - 0.4;
    return -1e9;
}
bool inStandingWater(vec2 p) {
    if (!biomeOn()) return false;
    if (F.biomeP[1].z > 0.0 && coastDist(p) < 2.0) return true;
    vec4 L;
    float dn;
    return lakeAt(p, L, dn) && dn < 1.05;
}
float biomeGrass(vec2 p) {
    float w[13];
    biomeWeights(p, w);
    float g = 0.0;
    for (int b = 0; b < B_COUNT; b++) g += w[b] * BIOME_GRASS[b];
    return inStandingWater(p) ? 0.0 : g;
}

// where plants can live: below the tree / shrub line, on faces that hold soil (biome.h vegHabitat)
const float TREE_LINE = 110.0, SHRUB_LINE = 150.0, VEG_LINE_FADE = 40.0, TREE_MAX_SLOPE = 0.7, SHRUB_MAX_SLOPE = 1.0;
float vegHabitat(float h, float slope, bool tree) {
    float line = tree ? TREE_LINE : SHRUB_LINE, ms = tree ? TREE_MAX_SLOPE : SHRUB_MAX_SLOPE;
    return (1.0 - smoothstep(line, line + VEG_LINE_FADE, h)) * (1.0 - smoothstep(ms * 0.75, ms, slope));
}
// ground colour per biome (h: ground height, base: the temperate colour, mask: its grass cover)
vec3 biomeAlbedo(vec2 p, vec3 base, float mask) {
    if (!biomeOn()) return base;
    float w[13];
    biomeWeights(p, w);
    float n1 = tnoise(p.x * 0.31, p.y * 0.31), n2 = tnoise(p.x * 2.3 + 5.0, p.y * 2.3), n3 = tnoise(p.x * 0.05 - 3.0, p.y * 0.05);
    float cd = coastDist(p);
    float h = w[B_MESA] > 0.001 || cd < F.biomeP[2].z + 30.0 ? groundBase(p) : 0.0;  // (only what needs it)
    vec3 col = base * w[B_TEMPERATE];
    for (int b = 1; b < B_COUNT; b++) {
        if (w[b] <= 0.001) continue;
        vec3 c;
        if (b == B_DESERT) c = mix(vec3(0.56, 0.42, 0.26), vec3(0.70, 0.55, 0.36), n1 * 0.6 + n2 * 0.4);
        else if (b == B_MESA) {
            float band = fract(h / 7.0 + 0.25 * n3);
            c = band < 0.33 ? vec3(0.52, 0.26, 0.15) : band < 0.6 ? vec3(0.66, 0.42, 0.25) : band < 0.8 ? vec3(0.74, 0.60, 0.44) : vec3(0.45, 0.22, 0.14);
            c *= 0.85 + 0.3 * n2;
        } else if (b == B_SAVANNA) c = mix(vec3(0.36, 0.30, 0.14), vec3(0.50, 0.42, 0.20), n1) * mix(0.8, 1.0, mask);
        else if (b == B_TUNDRA) c = mix(vec3(0.22, 0.22, 0.15), vec3(0.36, 0.33, 0.24), n1 * 0.5 + n2 * 0.5) * (0.85 + 0.15 * step(0.5, fract(n3 * 9.0)));
        else if (b == B_GLACIER) c = mix(vec3(0.62, 0.74, 0.86), vec3(0.88, 0.92, 0.96), smoothstep(0.2, 0.6, abs(tnoise(p.x * 0.02, p.y * 0.1) - 0.5) * 4.0));
        else if (b == B_ALPINE) c = mix(vec3(0.30, 0.29, 0.27), vec3(0.45, 0.44, 0.41), n1 * 0.5 + n2 * 0.5);
        else if (b == B_TAIGA) c = mix(base * vec3(0.7, 0.8, 0.75), vec3(0.17, 0.14, 0.09), 1.0 - mask);
        else if (b == B_JUNGLE) c = mix(base * vec3(0.6, 1.05, 0.7), vec3(0.13, 0.09, 0.05), (1.0 - mask) * 0.7);
        else if (b == B_WETLAND) c = mix(vec3(0.07, 0.07, 0.045), base * vec3(0.8, 0.95, 0.7), mask * n1);
        else if (b == B_VOLCANIC) c = mix(vec3(0.04, 0.04, 0.045), vec3(0.14, 0.11, 0.10), n1 * 0.4 + n2 * 0.6);
        else if (b == B_SALT) {
            vec2 g = p * 0.35;  // hex-ish crust plates with darker cracks
            float cr = smoothstep(0.03, 0.08, min(abs(fract(g.x + 0.5 * floor(g.y)) - 0.5), abs(fract(g.y) - 0.5)));
            c = mix(vec3(0.62, 0.60, 0.56), vec3(0.90, 0.89, 0.85), cr * (0.8 + 0.2 * n2));
        } else c = mix(vec3(0.50, 0.49, 0.45), base, mask * 0.8);  // karst: grey limestone, green on the flats
        col += c * w[b];
    }
    if (cd < F.biomeP[2].z + 30.0 && h < F.biomeP[2].x + 4.0)  // beach sand, sea floor
        col = mix(col, mix(vec3(0.40, 0.35, 0.26), vec3(0.68, 0.60, 0.44), smoothstep(-3.0, 0.5, h - F.biomeP[2].x)) * (0.9 + 0.2 * n2),
                  1.0 - smoothstep(F.biomeP[2].z, F.biomeP[2].z + 30.0, cd));
    return col;
}
#endif
