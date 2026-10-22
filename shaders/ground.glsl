// High resolution ground: generated texture sets (texgen.cpp) layered on the procedural terrain colour.
// Each set = albedo (a = stone mask) + detail (normal xz, height, AO). Bare/stony ground and soil (under grass)
// are blended by the same grassMask that places the blades. Close to the camera the height map drives
// parallax occlusion mapping so stones read as raised, not painted.
#ifndef GROUND_GLSL
#define GROUND_GLSL
#include "material.glsl"
#include "model.glsl"

// ---------------------------------------------------------------- fractal ground
// Generated per pixel from world position, so it never repeats and has detail at any distance: an eroded fBm
// (value noise with analytic derivatives, each octave damped by the slope so far -> worn, natural relief), domain
// warped, plus pebbles and grit from cellular noise at two scales. Octaves are added until they're finer than the
// pixel footprint (the last one faded in), so it stays alias-free: millimetres underfoot, a few octaves far off.
float fhash(vec2 i) { return float(hashu(uint(int(i.x)) * 1597334677u ^ uint(int(i.y)) * 3812015801u)) / 4294967295.0; }
vec3 noiseD(vec2 x) {  // value, d/dx, d/dy
    vec2 i = floor(x), f = x - i;
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0), du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    float a = fhash(i), b = fhash(i + vec2(1, 0)), c = fhash(i + vec2(0, 1)), d = fhash(i + vec2(1, 1));
    float k1 = b - a, k2 = c - a, k3 = a - b - c + d;
    return vec3(a + k1 * u.x + k2 * u.y + k3 * u.x * u.y, du * (vec2(k1, k2) + k3 * u.yx));
}
// cellular: distance to the nearest feature point (cell size 1), its offset, and its cell's hash
vec4 cellular(vec2 x) {
    vec2 i = floor(x), f = x - i;
    float best = 9.0, id = 0.0;
    vec2 off = vec2(0);
    for (int y = -1; y <= 1; y++)
        for (int xx = -1; xx <= 1; xx++) {
            vec2 g = vec2(xx, y), c = i + g;
            vec2 o = vec2(fhash(c), fhash(c + 17.31)) * 0.8 + 0.1;
            vec2 r = g + o - f;
            float d = dot(r, r);
            if (d < best) { best = d; off = r; id = fhash(c + 5.7); }
        }
    return vec4(sqrt(best), off, id);
}
struct FractalG { float h; vec2 grad; float cavity, stone; vec3 stoneCol; };
// pix: pixel footprint (m) at P; bare: 0 grassy soil .. 1 stony ground
FractalG fractalGround(vec2 p, float pix, float bare, vec3 soil) {
    FractalG g;
    // domain warp: large-scale distortion so features don't line up
    vec2 q = p + 1.5 * vec2(noiseD(p * 0.21).x, noiseD(p * 0.21 + 7.3).x);
    float amp = 0.15, freq = 0.5, h = 0.0, hn = 0.0, ampSum = 0.0;  // amp: metres (15 cm bumps at the 2 m scale)
    vec2 dsum = vec2(0), grad = vec2(0);
    float finest = max(F.fractal.w, 1e-4);
    mat2 rot = mat2(0.8, -0.6, 0.6, 0.8);
    vec2 x = q * freq;
    for (int o = 0; o < 14; o++) {
        float cell = 1.0 / freq;                         // feature size of this octave (m)
        if (cell < max(2.0 * pix, finest)) break;
        float fade = smoothstep(2.0 * pix, 4.0 * pix, cell);  // the last resolvable octave fades in
        vec3 n = noiseD(x);
        dsum += n.yz;
        float w = amp * fade / (1.0 + 0.15 * dot(dsum, dsum));  // eroded: steep parts get a little less fine detail
        h += w * (n.x - 0.5);
        hn += amp * fade * n.x;
        ampSum += amp * fade;
        grad += w * n.yz * freq;
        amp *= 0.56;  // a touch rougher than 1/f: real soil gets grittier toward the small end
        freq *= 2.03;
        x = rot * x * 2.03 + vec2(1.7, 9.2);
    }
    g.h = ampSum > 0.0 ? hn / ampSum : 0.5;           // normalised 0..1 for colour variation
    g.grad = grad;
    g.cavity = 1.0 - smoothstep(0.3, 0.55, g.h);       // low spots collect shade and fine dirt  // low spots collect shade and fine dirt
    // pebbles (6 cm) and grit (1.8 cm), fewer where grass grows, only where they cover a few pixels
    g.stone = 0.0;
    g.stoneCol = vec3(0);
    for (int k = 0; k < 2; k++) {
        float cs = k == 0 ? 0.06 : 0.018;
        float vis = smoothstep(1.5 * pix, 4.0 * pix, cs);
        if (vis <= 0.0) continue;
        vec4 c = cellular(q / cs + float(k) * 31.7);
        // stones gather in patches (washed together, weathered out of the soil), grit is sparse
        float patchK = smoothstep(0.35, 0.8, noiseD(p * (k == 0 ? 0.35 : 0.9) + float(k) * 5.0).x);
        float r = (0.22 + 0.2 * c.w), dens = F.fractal.z * mix(0.15, 1.0, bare) * patchK * (k == 0 ? 0.6 : 0.25);
        if (fract(c.w * 13.7) > dens) continue;
        float m = (1.0 - smoothstep(r * 0.75, r, c.x)) * vis;
        if (m <= g.stone) continue;
        // rounded top: the slope of a spherical cap pushes the normal out from the pebble's centre
        float t = clamp(c.x / r, 0.0, 0.98);
        grad += (-c.yz / max(c.x, 1e-4)) * (t / sqrt(1.0 - t * t)) * 0.6 * m / max(F.fractal.y, 0.05);  // (relief scales it back)
        g.stone = m;
        // stone colour: the local soil's hue, greyed and a little lighter, varying stone to stone
        float tint = fract(c.w * 71.3);
        vec3 grey = vec3(dot(soil, vec3(0.3, 0.5, 0.2)));
        g.stoneCol = mix(soil, grey, 0.5 + 0.3 * tint) * (0.8 + 0.35 * fract(c.w * 5.1)) * (1.0 - 0.25 * float(k));
    }
    g.grad = grad;
    return g;
}

struct GroundSample { vec4 alb; vec4 det; vec3 avg; };

GroundSample groundFetch(uint ia, uint id, vec2 uv, float lod) {
    GroundSample g;
    g.alb = textureLod(textures[nonuniformEXT(ia)], uv, lod);
    g.det = textureLod(textures[nonuniformEXT(id)], uv, lod);
    g.avg = textureLod(textures[nonuniformEXT(ia)], uv, 16.0).rgb;  // 1x1 mip: the set's mean colour
    return g;
}
float groundHeight(vec2 uv, float lod, float soilW) {
    float a = textureLod(textures[nonuniformEXT(F.groundTex.y)], uv, lod).b * F.groundP.z;
    float b = textureLod(textures[nonuniformEXT(F.groundTex.w)], uv, lod).b * F.groundP.w;
    return mix(a, b, soilW);
}

// P: shaded point, v: unit direction camera -> P, dist: camera distance (sets the mip), pom: parallax pass.
// rough / ao: the ground's roughness (soil matte, stones smoother on their worn tops) and cavity occlusion.
// Flatten zones with a ground texture (zones[2i+1].z = bindless slot + 1, .w = metres per repeat) replace the
// plain paving colour inside the zone.
void zoneTexture(vec3 P, vec3 v, float dist, vec3 n, inout vec3 albedo) {
    for (uint i = 0u; i < F.counts.x; i++) {
        vec4 q = zones[2u * i + 1u];
        if (q.z < 0.5) continue;
        float w = smoothstep(0.5, 1.0, zoneWeight(i, P.xz));
        if (w <= 0.0) continue;
        uint s = uint(q.z + 0.5) - 1u;
        float tile = max(q.w, 0.05);
        float pix = dist * 2.0 * F.camUp.w / float(F.res.y) / max(abs(dot(n, v)), 0.2);
        float lod = max(log2(max(pix, 1e-6) * float(textureSize(textures[nonuniformEXT(s)], 0).x) / tile), 0.0);
        albedo = mix(albedo, textureLod(textures[nonuniformEXT(s)], P.xz / tile, lod).rgb, w);
    }
}

// World map: a painted ground texture (palette in F.mapPal), then what the simulation left there: wet ground
// (darker, glossier) and snow (white, matte, smoothed relief). Applied after the detail layers below.
void mapSurface(vec3 P, vec3 v, float dist, inout vec3 n, inout vec3 albedo, inout float rough) {
    vec4 pt = mapPaintAt(P.xz);
    uint idx = uint(pt.b * 255.0 + 0.5);
    if (idx > 0u && idx < 8u && pt.a > 0.01) {
        uint slot1 = F.mapPal[idx / 4u][idx % 4u];
        if (slot1 > 0u) {
            uint s = slot1 - 1u;
            float tile = max(F.mapPalTile[idx / 4u][idx % 4u], 0.05);
            float pix = dist * 2.0 * F.camUp.w / float(F.res.y) / max(abs(dot(n, v)), 0.2);
            float lod = max(log2(max(pix, 1e-6) * float(textureSize(textures[nonuniformEXT(s)], 0).x) / tile), 0.0);
            albedo = mix(albedo, textureLod(textures[nonuniformEXT(s)], P.xz / tile, lod).rgb, smoothstep(0.0, 1.0, pt.a));
        }
    }
    if (F.sim1.w > 0.5) {
        vec4 sm = simAt(P.xz);
        float wet = max(sm.z, smoothstep(0.0, 0.03, sm.y));
        albedo *= mix(1.0, 0.55, wet);
        rough = mix(rough, 0.3, wet * 0.6);
        // standing water (runoff gathered in a dip / crater): a puddle, flat and mirror-smooth, the ground under it dark
        float puddle = smoothstep(0.008, 0.03, sm.y);
        if (puddle > 0.0) {
            albedo *= mix(1.0, 0.6, puddle);
            rough = mix(rough, 0.02, puddle);
            n = normalize(mix(n, vec3(0, 1, 0), puddle));
        }
        float snow = smoothstep(0.01, 0.12, sm.x);
        if (snow > 0.0) {
            float sparkle = 0.92 + 0.08 * vnoise(P.xz * 23.0);
            albedo = mix(albedo, vec3(0.86, 0.88, 0.92) * sparkle, snow);
            rough = mix(rough, 0.65, snow);
            n = normalize(mix(n, vec3(0, 1, 0), 0.4 * snow));  // fresh snow softens the ground's relief
        }
    }
}

void groundSurfaceDetail(inout vec3 P, vec3 v, float dist, bool pom, inout vec3 n, inout vec3 albedo, inout float rough, inout float ao);
void groundSurface(inout vec3 P, vec3 v, float dist, bool pom, inout vec3 n, inout vec3 albedo, inout float rough, inout float ao) {
    zoneTexture(P, v, dist, n, albedo);
    groundSurfaceDetail(P, v, dist, pom, n, albedo, rough, ao);
    mapSurface(P, v, dist, n, albedo, rough);
}
void groundSurfaceDetail(inout vec3 P, vec3 v, float dist, bool pom, inout vec3 n, inout vec3 albedo, inout float rough, inout float ao) {
    if (F.groundS.w < 0.5) return;
    float zone = zoneCover(P.xz);
    if (zone > 0.99) return;
    float tile = F.groundP.x, texel = tile / F.groundP.y;
    float cosv = max(abs(dot(n, v)), 0.2);
    float pix = dist * 2.0 * F.camUp.w / float(F.res.y) / cosv;
    float lod = max(log2(max(pix, 1e-6) / texel) - 0.5, 0.0);
    float soilW = smoothstep(0.05, 0.6, grassMask(P.xz));
    uint fractalMode = uint(F.fractal.x + 0.5);
    if (fractalMode == 1u) {  // fractal only: no textures, no repeat
        // triplanar: top projection on flat ground; on slopes (sculpted hills, crater walls) also the planes facing x
        // and z, blended by the normal, so the pattern keeps its scale instead of stretching down the face
        vec3 tw3 = pow(abs(n), vec3(6.0));
        tw3 /= tw3.x + tw3.y + tw3.z;
        FractalG fg = fractalGround(P.xz, pix, 1.0 - soilW, albedo);
        vec3 G = vec3(fg.grad.x, 0.0, fg.grad.y) * tw3.y;  // world-space relief gradient
        float fh = fg.h * tw3.y, fcav = fg.cavity * tw3.y, fst = fg.stone * tw3.y;
        vec3 fsc = fg.stoneCol * fg.stone * tw3.y;
        if (tw3.x > 0.02) {
            FractalG sx = fractalGround(P.zy + vec2(37.0, 0.0), pix, 1.0, albedo);  // plane facing x: (z, y)
            G += vec3(0.0, sx.grad.y, sx.grad.x) * tw3.x;
            fh += sx.h * tw3.x; fcav += sx.cavity * tw3.x; fst += sx.stone * tw3.x; fsc += sx.stoneCol * sx.stone * tw3.x;
        }
        if (tw3.z > 0.02) {
            FractalG sz = fractalGround(P.xy + vec2(0.0, 53.0), pix, 1.0, albedo);  // plane facing z: (x, y)
            G += vec3(sz.grad.x, sz.grad.y, 0.0) * tw3.z;
            fh += sz.h * tw3.z; fcav += sz.cavity * tw3.z; fst += sz.stone * tw3.z; fsc += sz.stoneCol * sz.stone * tw3.z;
        }
        fg.h = fh; fg.cavity = fcav;
        fg.stoneCol = fst > 1e-4 ? fsc / fst : albedo;
        fg.stone = fst;
        float k = 1.0 - zone;
        // mottling: damper / drier, finer / coarser patches at 0.5-3 m, plus the relief's own light / dark (3D, so
        // it doesn't stretch either)
        float mott = noiseD(P.xz * 0.37 + 3.1 + P.y * 0.29).x * 0.6 + noiseD(P.xz * 1.9 + P.y * 1.3).x * 0.4;
        float varC = (0.72 + 0.56 * fg.h) * (0.8 + 0.4 * mott);
        vec3 col = albedo * varC * mix(1.0, 0.8, fg.cavity);
        col = mix(col, fg.stoneCol, fg.stone);
        albedo = mix(albedo, col, k);
        // bump: tilt the normal against the relief gradient's part along the surface
        vec3 Gt = (G - dot(G, n) * n) * F.fractal.y * k;
        n = normalize(n - Gt);
        rough = mix(rough, mix(0.95, 0.6, fg.stone), k);
        ao = mix(ao, mix(1.0, 0.7, fg.cavity * (1.0 - fg.stone)), k);
        return;
    }

    // parallax occlusion: march the view ray down through the height volume (world-space xz offsets)
    if (pom && F.groundS.y > 0.5 && dist < F.groundS.z && -v.y > 0.05) {
        float maxH = mix(F.groundP.z, F.groundP.w, soilW);
        vec2 stepXZ = v.xz / max(-v.y, 0.15);  // xz travelled per metre of depth
        int N = int(mix(32.0, 8.0, -v.y));
        float dd = maxH / float(N), depth = 0.0, prevGap = 0.0;
        vec2 xz = P.xz;
        float fade = 1.0 - smoothstep(F.groundS.z * 0.6, F.groundS.z, dist);
        for (int i = 0; i < N; i++) {
            float surf = maxH - groundHeight(xz / tile, lod, soilW);  // depth of the surface below the top plane
            float gap = surf - depth;
            if (gap <= 0.0) {
                float t = prevGap / max(prevGap - gap, 1e-6);  // refine between the last two steps
                xz -= stepXZ * dd * (1.0 - t);
                break;
            }
            prevGap = gap;
            depth += dd;
            xz += stepXZ * dd;
        }
        P.xz = mix(P.xz, xz, fade);
    }

    vec2 uv = P.xz / tile;
    // triplanar weights: straight-down projection on flat ground, side projections on slopes (no vertical smear)
    vec3 tw = pow(abs(n), vec3(4.0));
    tw /= tw.x + tw.y + tw.z;
    // second, rotated and scaled lookup blended by low-frequency noise breaks up the tile repetition
    vec2 uv2 = mat2(0.8, -0.6, 0.6, 0.8) * uv * 0.73 + vec2(0.37, 0.71);
    float w2 = smoothstep(0.3, 0.7, vnoise(P.xz * 0.35 + 11.0));
    vec4 alb = vec4(0), det = vec4(0);
    vec3 avg = vec3(0);
    for (int s = 0; s < 2; s++) {
        float ws = s == 0 ? 1.0 - soilW : soilW;
        if (ws < 0.01) continue;
        uint ia = s == 0 ? F.groundTex.x : F.groundTex.z, id = s == 0 ? F.groundTex.y : F.groundTex.w;
        GroundSample a = groundFetch(ia, id, uv, lod), b = groundFetch(ia, id, uv2, lod);
        vec4 ta = mix(a.alb, b.alb, w2), td = mix(a.det, b.det, w2);
        if (tw.y < 0.98) {  // steep ground (sculpted hills, crater walls): side projections wrap the texture over it
            GroundSample sx = groundFetch(ia, id, P.zy / tile, lod), sz = groundFetch(ia, id, P.xy / tile + 0.5, lod);
            ta = ta * tw.y + sx.alb * tw.x + sz.alb * tw.z;
            td = td * tw.y + sx.det * tw.x + sz.det * tw.z;
        }
        alb += ta * ws;
        det += td * ws;
        avg += a.avg * ws;
    }
    if (fractalMode == 2u) {  // hybrid: the textures, with fractal variation breaking up their repeat and finer detail
        FractalG fg = fractalGround(P.xz, pix, 1.0 - soilW, albedo);
        float near = 1.0 - smoothstep(8.0, 25.0, dist);
        alb.rgb *= 0.82 + 0.36 * fg.h;
        det.xy += -fg.grad * 0.5 * F.fractal.y * near;
        alb.rgb = mix(alb.rgb, fg.stoneCol, fg.stone * near);
    }
    float k = 1.0 - zone;
    // colour: the macro terrain colour (green / brown / river / craters) carries the large scale,
    // the texture adds detail around its own mean; stones keep their own colour
    vec3 detailCol = albedo * alb.rgb / max(avg, vec3(0.02));
    albedo = mix(albedo, mix(detailCol, alb.rgb, alb.a * 0.7), k);
    vec2 dn = (det.xy * 2.0 - 1.0) * F.groundS.x * k * tw.y;  // detail normals are top-projected: only where that dominates
    n = normalize(vec3(n.x + dn.x, n.y, n.z + dn.y));
    float stone = alb.a;
    rough = mix(rough, mix(0.96, mix(0.82, 0.55, det.b), stone), k);
    ao = mix(ao, mix(1.0, det.a, 0.85), k);
}
#endif
