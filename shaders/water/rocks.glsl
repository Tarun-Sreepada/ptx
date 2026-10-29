// Rocks in the current: pillow, wake, flow deflection, foam.
#ifndef WATER_ROCKS
#define WATER_ROCKS
#include "core/frame.glsl"
#include "water/flow.glsl"
#include "water/splashes.glsl"

// River rocks near p (terrain.h layout): c, flow direction, waterline radius R (0 = under water), top depth sub, scale
struct RockW { vec2 c, dir; float R, sub, sc, u; };  // u: own flow speed past it (thrown rocks), < 0 = the river's
const int MAX_NEAR_ROCKS = 16;  // up to 9 placed + thrown ones in reach
int nearRocks(vec2 p, float level, out RockW rk[MAX_NEAR_ROCKS]) {
    int n = 0, ci = int(floor(p.x / RIVER_ROCK_L));
    int up = int(ceil(F.water2.x / RIVER_ROCK_L)) + 1;  // wakes trail downstream (+x): look as far upstream as they reach
    // only the channels this point could be near (rocks lie within their channel)
    vec4 chs[4];
    int nch = riverChannels(p.x, chs);
    for (int kc = 0; kc < nch; kc++) {
    if (abs(p.y - chs[kc].x) > 3.0 * F.river.x * chs[kc].z + 8.0) continue;
    for (int i = ci - up; i <= ci + 1 && n < MAX_NEAR_ROCKS; i++) {
        vec4 r;
        float dz;
        if (!riverRock(i, kc, r, dz)) continue;
        RockW k;
        k.sub = level - (r.y + 0.72 * r.w);
        if (k.sub > 1.0 * r.w + 0.3) continue;  // too deep to show on the surface
        float hh = (level - r.y) / (0.72 * r.w);
        k.c = r.xz;
        k.dir = vec2(1.0, dz) * inversesqrt(1.0 + dz * dz);
        k.sc = r.w;
        k.u = -1.0;
        k.R = k.sub < 0.0 ? 0.95 * r.w * sqrt(clamp(1.0 - hh * hh, 0.05, 1.0)) : 0.0;
        rk[n++] = k;
    }
    }
    // thrown rocks lying in the water act like placed ones (tumbling ones too, as an approximation)
    for (uint i = 0u; i < F.dynInfo.x && n < MAX_NEAR_ROCKS; i++) {
        vec4 r = F.dynRocks[i];
        vec2 dd = p - r.xz;
        if (dot(dd, dd) > sq(6.0 * r.w + 3.0)) continue;
        RockW k;
        k.sub = level - (r.y + 0.72 * r.w);
        if (k.sub > r.w + 0.3 || r.y - 0.6 * r.w > level) continue;  // deep under, or still in the air
        float dz = riverSlope(r.x), hh = (level - r.y) / (0.72 * r.w);
        k.c = r.xz;
        // the water moves past a moving rock at (current - rock velocity): the wake trails along that, so a rock
        // ploughing across or against the current drags its foam behind it, a resting one leaves it downstream
        vec2 cur = vec2(1.0, dz) * inversesqrt(1.0 + dz * dz) * max(F.water.x * 1.3, 0.05);
        vec2 rel = cur - F.dynRockV[i].xz;
        float rl = length(rel);
        k.dir = rl > 1e-3 ? rel / rl : vec2(1.0, dz) * inversesqrt(1.0 + dz * dz);
        k.u = min(rl, 6.0);
        if (k.sub > 0.0) {  // fully under water: only the current passes over it (its own tumbling doesn't reach the top)
            k.dir = vec2(1.0, dz) * inversesqrt(1.0 + dz * dz);
            k.u = -1.0;
        }
        k.sc = r.w;
        k.R = k.sub < 0.0 ? 0.95 * r.w * sqrt(clamp(1.0 - hh * hh, 0.05, 1.0)) : 0.0;
        rk[n++] = k;
    }
    return n;
}
// Surface height around one rock (m). Breaking the surface: a pillow against the upstream face, a trough in the
// wake. Submerged: a boil just downstream of the stone, then a decaying stationary wave train.
float rockHeight(vec2 p, RockW k, float U) {
    if (k.u >= 0.0) U = k.u;
    vec2 rel = p - k.c;
    float a = dot(rel, k.dir), b = dot(rel, vec2(-k.dir.y, k.dir.x)), r = length(rel);
    if (k.R > 0.0) {
        float R = k.R, wd = R * sqrt(1.0 + 0.3 * max(a, 0.0) / R);
        float pillow = 0.12 * U * R * exp(-sq((r - R) / (0.3 * R + 0.1))) * smoothstep(0.2, -0.7, a / max(r, 1e-3));
        float wake = exp(-sq(b / wd)) * wakeTail(a, R);
        return pillow - 0.06 * U * wake;
    }
    float Rb = k.sc, e = exp(-2.0 * k.sub), ab = a - 0.6 * Rb;
    float boil = exp(-(ab * ab + b * b) / (Rb * Rb));
    float train = cos(2.6 * ab / Rb) * exp(-max(ab, 0.0) / (3.0 * Rb)) * exp(-b * b / (Rb * Rb)) * smoothstep(0.0, Rb, ab);
    return e * U * (0.08 * boil + 0.04 * train);
}
// Flow deflection (potential flow around a cylinder), foam, turbulence and impact splashes of one rock.
void rockFlow(vec2 p, RockW k, float U, float T, inout vec2 v, inout float foam, inout float turb, inout vec2 g) {
    if (k.u >= 0.0) U = k.u;
    vec2 rel = p - k.c, perp = vec2(-k.dir.y, k.dir.x);
    float a = dot(rel, k.dir), b = dot(rel, perp), r = max(length(rel), 1e-3);
    if (k.R > 0.0) {
        float R = k.R, R2 = R * R, r4 = r * r * r * r;
        if (r > R) v += (k.dir * (-U * R2 * (a * a - b * b) / r4) + perp * (-U * 2.0 * R2 * a * b / r4)) * min(1.0, r4 / (R2 * R2));
        float wd = R * sqrt(1.0 + 0.3 * max(a, 0.0) / R), tail = wakeTail(a, R);
        float wake = exp(-sq(b / wd)) * tail;
        float ring = exp(-sq((r - R) / (0.2 + 0.3 * R)));
        foam += ring * (0.45 + 0.4 * U) + wake * (0.2 + 0.5 * U) + exp(-sq((abs(b) - wd) * 2.5)) * tail * 0.8 * U;  // + shear-line streaks
        turb += 1.5 * wake + ring;
        // water slapping the upstream face: a splash every second or two
        float h0 = whash(k.c * 3.1), Tp = 0.9 + 1.4 * h0, age = mod(T + h0 * 7.3, Tp);
        if (age < 1.0 && F.water.w > 0.0) foam += splashRing(p - (k.c - k.dir * R), age, F.water.w * min(U, 2.0) * min(R, 1.2), g);
    } else {
        // a submerged stone only boils the surface when it's close under it (and in a real current); deeper it's calm
        float Rb = k.sc, ab = a - 0.6 * Rb, boil = exp(-(ab * ab + b * b) / (Rb * Rb));
        float shallow = exp(-6.0 * k.sub / max(Rb, 0.1)) * smoothstep(0.3, 0.8, U);
        foam += shallow * boil * 0.6 * U;
        turb += exp(-3.0 * k.sub / max(Rb, 0.1)) * boil;
    }
}

// Thrown rocks hitting the water (spray droplets themselves are CPU particles, see emitSplash). The surface really
// moves here: a cavity opens under the impact, a crown rim rises around it and collapses, then wave-particle
#endif
