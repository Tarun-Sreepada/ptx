// Layered ground material generator (vtgen.comp fills the ground virtual texture with it, groundvt.glsl reads it).
// The material is built scale by scale, each layer a real feature size rather than one self-similar fractal:
//   macro   (0.5-10 m)  moisture / organic patches, stony vs soft patches: hue and value shifts
//   relief  (3 m-0.5 mm) eroded gradient-noise fBm, domain warped
//   stones  (14 / 4.5 / 1.4 cm) shaped, part-buried rocks of several rock types sitting on the relief at their scale
//   soil    (mm) crumb lumps and crevices, light / dark grit grains, drying cracks, leaf litter fragments
// Every layer fades out once its features get finer than ~2 texels (t), so each clipmap level is its own
// band-limited version of the same surface and the levels blend without aliasing.
// Positions are doubles: lattice coordinates are wrapped mod 4096 cells before going to float, so millimetre
// detail stays exact kilometres from the origin.
#ifndef GROUNDGEN_GLSL
#define GROUNDGEN_GLSL
#extension GL_ARB_gpu_shader_fp64 : require

vec2 lat(dvec2 p, double fr) {  // p * fr, wrapped to [0, 4096)
    dvec2 x = p * fr;
    return vec2(x - floor(x * (1.0LF / 4096.0LF)) * 4096.0LF);
}
uint ghash(ivec2 i, uint s) { return hashu((uint(i.x & 4095) | (uint(i.y & 4095) << 12)) ^ (s * 0x9E3779B9u)); }
float g01(uint h) { return float(h & 0xFFFFFFu) * (1.0 / 16777215.0); }
vec2 ggrad(ivec2 i, uint s) { float a = g01(ghash(i, s)) * 6.2831853; return vec2(cos(a), sin(a)); }
// gradient noise with analytic derivatives: value (about -0.7..0.7), d/dx, d/dy (per lattice cell)
vec3 gnoise(vec2 x, uint s) {
    ivec2 i = ivec2(floor(x));
    vec2 f = x - floor(x);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0), du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    vec2 ga = ggrad(i, s), gb = ggrad(i + ivec2(1, 0), s), gc = ggrad(i + ivec2(0, 1), s), gd = ggrad(i + ivec2(1, 1), s);
    float va = dot(ga, f), vb = dot(gb, f - vec2(1, 0)), vc = dot(gc, f - vec2(0, 1)), vd = dot(gd, f - vec2(1, 1));
    float k = va - vb - vc + vd;
    float v = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k;
    vec2 d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) + du * (u.yx * k + vec2(vb, vc) - va);
    return vec3(v, d);
}
float gfbm(dvec2 p, double fr, int oct, uint s) {  // 0..1-ish
    float v = 0.0, a = 0.5;
    for (int o = 0; o < oct; o++) { v += a * gnoise(lat(p, fr), s + uint(o)).x; fr *= 2.03LF; a *= 0.5; }
    return clamp(v * 1.1 + 0.5, 0.0, 1.0);
}
// cellular: F1, F2 (cell units) and the nearest cell's hash
vec3 gcell(vec2 x, uint s) {
    ivec2 i = ivec2(floor(x));
    vec2 f = x - floor(x);
    float d1 = 9.0, d2 = 9.0;
    uint id = 0u;
    for (int y = -1; y <= 1; y++)
        for (int xx = -1; xx <= 1; xx++) {
            ivec2 c = i + ivec2(xx, y);
            uint h = ghash(c, s);
            vec2 r = vec2(xx, y) + vec2(g01(h), g01(hashu(h))) * 0.8 + 0.1 - f;
            float d = dot(r, r);
            if (d < d1) { d2 = d1; d1 = d; id = h; } else if (d < d2) d2 = d;
        }
    return vec3(sqrt(d1), sqrt(d2), uintBitsToFloat(id));
}

struct GroundGen {
    float h;      // relief height (m)
    vec3 rel;     // colour relative to the terrain's macro colour (1 = unchanged)
    float stone;  // stone coverage 0..1
    float rough;
    float ao;
};

const vec3 SOIL_REF = vec3(0.30, 0.24, 0.18);  // stones' absolute colours are divided by this to go relative
vec3 rockColour(uint type, float grain, float speck, vec2 lv, float n) {
    if (type == 0u) {  // granite: grey with black and white flecks
        vec3 c = vec3(0.42, 0.40, 0.38) * (0.85 + 0.3 * grain);
        return mix(c, speck > 0.5 ? vec3(0.78, 0.76, 0.72) : vec3(0.07), abs(speck - 0.5) > 0.36 ? 1.0 : 0.0);
    }
    if (type == 1u) return vec3(0.12, 0.12, 0.13) * (1.0 - 0.45 * smoothstep(0.62, 0.8, grain));  // basalt, vesicles
    if (type == 2u) return vec3(0.56, 0.44, 0.30) * (0.86 + 0.14 * sin((lv.x * 0.6 + lv.y) * 8.0 + n * 5.0));  // sandstone bands
    if (type == 3u) return vec3(0.74, 0.71, 0.66) * (0.92 + 0.12 * grain);  // quartzite
    if (type == 4u) return vec3(0.47, 0.28, 0.20) * (0.82 + 0.3 * grain);   // red sandstone / jasper
    return vec3(0.60, 0.58, 0.52) * (0.85 + 0.2 * grain) * (1.0 - 0.3 * smoothstep(0.7, 0.85, speck));  // limestone, pitted
}

// p: world xz (m), t: texel / footprint size (m), pebbles: stone amount, warpK: macro warp
GroundGen groundGen(dvec2 p, float t, float pebbles, float warpK) {
    GroundGen g;
    // ---- macro: warp, moisture, organics, stony patches
    vec2 w = vec2(gnoise(lat(p, 0.17LF), 1u).x, gnoise(lat(p, 0.17LF), 2u).x) * 0.7
           + vec2(gnoise(lat(p, 0.9LF), 3u).x, gnoise(lat(p, 0.9LF), 4u).x) * 0.12;
    dvec2 pw = p + dvec2(w * warpK);
    float moist = gfbm(pw, 0.11LF, 3, 10u), organic = gfbm(pw, 0.33LF, 3, 20u);
    float stony = smoothstep(0.35, 0.7, gfbm(pw, 0.45LF, 2, 30u));
    float dry = 1.0 - smoothstep(0.35, 0.6, moist);

    // ---- relief: eroded fBm, gradient noise, wavelength 3.3 m down to 2 texels
    const float CLS[3] = float[3](0.14, 0.045, 0.014);  // stone class cell sizes (m)
    float hAt[3] = float[3](0.0, 0.0, 0.0);
    float h = 0.0, amp = 0.09, hFine = 0.0, aFine = 0.0, hn = 0.0, an = 0.0;
    double fr = 0.3LF;
    vec2 dsum = vec2(0);
    for (int o = 0; o < 20; o++) {
        float wl = float(1.0LF / fr);
        if (wl < 2.0 * t) break;
        for (int c = 0; c < 3; c++) if (wl >= CLS[c]) hAt[c] = h;
        float fade = smoothstep(2.0 * t, 4.0 * t, wl);
        vec3 n = gnoise(lat(pw, fr), 100u + uint(o));
        dsum += n.yz;
        float a = amp * fade / (1.0 + 0.2 * dot(dsum, dsum));  // eroded: less fine detail on steep bits
        h += a * n.x;
        hn += amp * fade * n.x; an += amp * fade;
        if (wl < 0.12) { hFine += a * n.x; aFine += amp * fade; }
        amp *= wl > 0.2 ? 0.5 : 0.58;  // grittier toward the small end
        fr *= 1.93LF;
    }
    for (int c = 0; c < 3; c++) if (CLS[c] < 2.0 * t) hAt[c] = h;
    float cav = aFine > 0.0 ? clamp(-hFine / (aFine * 0.5), 0.0, 1.0) : 0.0;
    float hn01 = an > 0.0 ? clamp(hn / an * 0.8 + 0.5, 0.0, 1.0) : 0.5;

    vec3 rel = mix(vec3(1.10, 1.07, 1.02), vec3(0.74, 0.70, 0.68), moist);
    rel *= mix(vec3(1.0), vec3(0.78, 0.68, 0.56), organic * 0.7);
    rel *= 0.88 + 0.24 * hn01;
    float rough = mix(0.97, 0.8, moist), ao = 1.0;

    // ---- soil micro layers
    {   // crumbs: 7 mm lumps, dark crevices between them
        float vis = smoothstep(2.0 * t, 6.0 * t, 0.007);
        if (vis > 0.0) {
            vec3 c = gcell(lat(pw, 1.0LF / 0.007LF), 200u);
            float crev = 1.0 - smoothstep(0.0, 0.3, c.y - c.x);
            h += (0.0016 * (1.0 - c.x * c.x) - 0.0022 * crev) * vis;
            rel *= (1.0 - 0.38 * crev * vis) * (1.0 + 0.12 * (g01(floatBitsToUint(c.z)) - 0.5) * vis);
            ao *= 1.0 - 0.4 * crev * vis;
        }
    }
    {   // clods: 2.5 cm lumps of soil, each a little lighter / darker, redder / greyer
        // crumbly patches between smoother, compacted ones; only some cells are a raised clod
        float vis = smoothstep(2.0 * t, 6.0 * t, 0.025) * smoothstep(0.35, 0.75, gfbm(pw, 0.7LF, 2, 208u));
        if (vis > 0.0) {
            vec2 q = lat(pw, 1.0LF / 0.025LF);
            vec3 c = gcell(q + vec2(gnoise(q * 1.7, 205u).x, gnoise(q * 1.7, 206u).x) * 0.3, 207u);
            uint id = floatBitsToUint(c.z);
            float raised = smoothstep(0.35, 0.6, g01(hashu(id ^ 0x5bd1e995u)));
            float crev = (1.0 - smoothstep(0.0, 0.12 + 0.12 * g01(id >> 4), c.y - c.x)) * (0.3 + 0.7 * raised);
            float lump = (1.0 - c.x * c.x) * raised;
            h += (0.0035 * lump - 0.003 * crev) * vis * (1.0 - 0.5 * moist);
            vec3 tint = vec3(1.0) + (vec3(g01(id), g01(id >> 8), g01(id >> 16)) - 0.5) * vec3(0.10, 0.08, 0.06) + (g01(hashu(id)) - 0.5) * 0.22;
            rel *= mix(vec3(1.0), tint * (1.0 - 0.3 * crev), vis);
            ao *= 1.0 - 0.3 * crev * vis;
        }
    }
    {   // drying cracks: cell edges of a 30 cm warped Voronoi, only in dry patches
        float wid = 0.0035 * smoothstep(0.2, 0.8, dry);
        float vis = smoothstep(0.5 * t, 2.0 * t, wid);
        if (vis > 0.0) {
            vec2 q = lat(pw, 1.0LF / 0.3LF);
            q += vec2(gnoise(q * 2.3, 210u).x, gnoise(q * 2.3, 211u).x) * 0.25;
            vec3 c = gcell(q, 212u);
            float e = (c.y - c.x) * 0.3;  // edge distance (m)
            float crack = (1.0 - smoothstep(0.4 * wid, wid, e)) * vis;
            h -= 0.006 * crack;
            rel *= 1.0 - 0.55 * crack;
            ao *= 1.0 - 0.6 * crack;
        }
    }
    {   // leaf litter: thin elongated fragments in organic patches
        float cs = 0.03, vis = smoothstep(1.5 * t, 4.0 * t, 0.004);
        if (vis > 0.0) {
            vec2 q = lat(pw, 1.0LF / double(cs));
            ivec2 ci = ivec2(floor(q));
            vec2 f = q - floor(q);
            for (int y = -1; y <= 1; y++)
                for (int x = -1; x <= 1; x++) {
                    uint hh = ghash(ci + ivec2(x, y), 220u);
                    if (g01(hh) > 0.35 * smoothstep(0.3, 0.8, organic)) continue;
                    uint h2 = hashu(hh), h3 = hashu(h2);
                    vec2 r = (vec2(x, y) + vec2(g01(h2), g01(h3)) - f) * cs;
                    float a = g01(hashu(h3)) * 6.2831853;
                    vec2 lv = vec2(dot(r, vec2(cos(a), sin(a))), dot(r, vec2(-sin(a), cos(a))));
                    float L = 0.006 + 0.008 * g01(hh >> 3), Wd = L * (0.25 + 0.2 * g01(h2 >> 5));
                    float d = length(lv / vec2(L, Wd));
                    if (d < 1.0) {
                        float m = (1.0 - smoothstep(0.75, 1.0, d)) * vis;
                        vec3 lc = mix(vec3(0.33, 0.20, 0.10), vec3(0.50, 0.36, 0.18), g01(h3 >> 7));
                        rel = mix(rel, lc / SOIL_REF, m);
                        h += 0.0006 * m;
                        rough = mix(rough, 0.85, m);
                    }
                }
        }
    }
    {   // grit: 1.6 mm grains, some pale quartz sand (a little glossy), some dark organic specks
        float vis = smoothstep(1.5 * t, 4.0 * t, 0.0016);
        if (vis > 0.0) {
            vec3 c = gcell(lat(p, 1.0LF / 0.0016LF), 230u);
            uint id = floatBitsToUint(c.z);
            float r = g01(id), m = (1.0 - smoothstep(0.25, 0.42, c.x)) * vis;
            if (r < 0.2) { rel = mix(rel, rel * 1.6 + 0.15, m); rough = mix(rough, 0.45, m); h += 0.0003 * m; }
            else if (r < 0.36) rel = mix(rel, rel * 0.5, m);
        }
    }
    rel *= 1.0 - 0.3 * cav;
    ao *= 1.0 - 0.35 * cav;

    // ---- stones: three size classes, big to small, composited by height
    float stoneM = 0.0;
    const float BASE[3] = float[3](0.16, 0.30, 0.42);
    for (int k = 0; k < 3; k++) {
        float cs = CLS[k];
        float vis = smoothstep(1.2 * t, 3.0 * t, cs * 0.3);
        float dens = min(1.0, BASE[k] * pebbles * 1.7 * (0.25 + 0.75 * stony));
        // stones too small for this texel: their mean colour over the area they'd cover, so levels agree
        rel = mix(rel, vec3(0.48, 0.42, 0.37) / SOIL_REF, (1.0 - vis) * dens * 0.25);
        vec2 q = lat(pw, 1.0LF / double(cs));
        ivec2 ci = ivec2(floor(q));
        vec2 f = q - floor(q);
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++) {
                uint h0 = ghash(ci + ivec2(x, y), 300u + uint(k));
                if (g01(h0) > dens) continue;
                uint h1 = hashu(h0), h2 = hashu(h1), h3 = hashu(h2), h4 = hashu(h3);
                float R = cs * (0.22 + 0.22 * g01(h1));
                vec2 v = (vec2(x, y) + vec2(g01(h2), g01(h3)) * 0.7 + 0.15 - f) * cs;
                float a = g01(h4) * 6.2831853, aspect = 0.6 + 0.4 * g01(h1 >> 8);
                vec2 lv = vec2(dot(v, vec2(cos(a), sin(a))), dot(v, vec2(-sin(a), cos(a)))) / vec2(R, R * aspect);
                float d = length(lv);
                if (d > 1.4) continue;
                float th = atan(lv.y, lv.x);
                d /= 1.0 + 0.11 * sin(3.0 * th + g01(h2 >> 4) * 6.28) + 0.06 * sin(5.0 * th + g01(h3 >> 4) * 6.28);
                float hr = mix(0.3, 0.7, g01(h4 >> 6)) * R, bury = mix(0.15, 0.45, g01(h0 >> 9));
                if (d >= 1.0) {  // contact shadow on the soil around a stone
                    ao *= 1.0 - 0.4 * (1.0 - smoothstep(1.0, 1.35, d)) * vis;
                    continue;
                }
                float z = hr * pow(max(1.0 - d * d, 0.0), mix(0.45, 0.8, g01(h1 >> 12)));
                float grain = gnoise(lv * 9.0 + g01(h0 >> 4) * 50.0, 400u).x * 0.5 + 0.5;
                float top = hAt[k] + (z - bury * hr + R * 0.03 * (grain - 0.5)) * vis;
                if (top <= h) continue;
                h = top;
                uint type = (h0 >> 20) % 6u;
                float speck = g01(ghash(ivec2(floor(lv * 14.0)) + ivec2(h1 & 255u), 401u));
                float fine = smoothstep(1.5 * t, 4.0 * t, R / 14.0);
                vec3 col = rockColour(type, mix(0.5, grain, fine), mix(0.5, speck, fine), lv, gnoise(lv * 2.0, 402u).x);
                col *= 0.88 + 0.24 * g01(h2 >> 10);
                float zr = z / max(hr, 1e-6);
                col = mix(col, SOIL_REF * rel, 0.45 * (1.0 - zr) * (1.0 - zr));  // dusty where it meets the soil
                if (k == 0 && g01(h3 >> 11) < 0.3) {  // lichen on some cobbles
                    float l = smoothstep(0.55, 0.65, gnoise(lv * 3.0, 403u).x * 0.5 + 0.5) * zr;
                    col = mix(col, (h3 & 1u) == 1u ? vec3(0.55, 0.58, 0.45) : vec3(0.62, 0.40, 0.12), l * 0.8);
                }
                float m = vis * (1.0 - smoothstep(0.85, 1.0, d));
                rel = mix(rel, col / SOIL_REF, m);
                rough = mix(rough, mix(0.75, 0.5, zr), m);  // worn, smoother tops
                ao = mix(ao, 1.0, m * zr);
                stoneM = max(stoneM, m);
            }
    }
    g.h = h; g.rel = rel; g.stone = stoneM; g.rough = rough; g.ao = clamp(ao, 0.0, 1.0);
    return g;
}
#endif
