// Bank splashes and impact splashes on the river surface (cavity, crown, Worthington jet, ring waves, foam).
#ifndef WATER_SPLASHES
#define WATER_SPLASHES
#include "core/frame.glsl"
#include "water/flow.glsl"

// One wave-particle style splash: expanding ring (slope added to g) and a foam burst, fading with age (s).
float splashRing(vec2 dv, float age, float amp, inout vec2 g) {
    float dist = length(dv) + 1e-4, r = 0.25 + 1.1 * age, x = dist - r, a = exp(-2.5 * age) * amp;
    g += 0.025 * a * exp(-16.0 * x * x) * (-32.0 * x * cos(14.0 * x) - 14.0 * sin(14.0 * x)) * dv / dist;
    return a * (1.0 - smoothstep(0.0, r + 0.4, dist)) * 0.9 + a * exp(-20.0 * x * x) * 0.6;
}

// Bank splashes: per 3 m stretch of the pixel's bank, an event every few seconds (if its dice roll hits).
// Adds the ring's slope to g and returns foam.
float splashes(vec2 p, Flow f, float T, inout vec2 g) {
    float d = abs(f.r.x);
    if (F.water.w <= 0.0 || d < f.r.y - 3.5 || d > f.r.y) return 0.0;
    float side = f.r.x >= 0.0 ? 1.0 : -1.0;
    float curv = (riverSlope(p.x + 1.0) - riverSlope(p.x - 1.0)) * 0.5;  // main centreline z'' (side channels: about the same)
    float outer = clamp(-side * curv * 400.0, 0.0, 1.0);
    float foam = 0.0;
    for (int j = -1; j <= 1; j++) {
        float cell = floor(p.x / 3.0) + float(j);
        vec2 key = vec2(cell, side * 17.0 + 3.0);
        float Tp = 2.2 + 2.5 * whash(key + vec2(0, 41)), tt = T + whash(key + vec2(0, 93)) * Tp;
        float epoch = floor(tt / Tp), age = tt - epoch * Tp;
        if (age > 1.6 || whash(key + vec2(epoch * 7.0, 11)) > F.water.w * (0.3 + 0.6 * outer) * min(F.water.x, 1.5)) continue;
        float cx = (cell + 0.2 + 0.6 * whash(key + vec2(0, 7))) * 3.0;
        // the channel this point is in (main, braid or tributary), at the splash's x
        vec4 chs[4];
        int nch = riverChannels(cx, chs), kb = 0;
        for (int k = 1; k < nch; k++) if (abs(p.y - chs[k].x) < abs(p.y - chs[kb].x)) kb = k;
        if (!riverWet(chs[kb])) continue;
        vec4 rc = riverInfoC(vec2(cx, chs[kb].x), chs[kb]);
        float dz = chs[kb].y, inv = inversesqrt(1.0 + dz * dz);
        vec2 c = vec2(cx, chs[kb].x) + vec2(-dz, 1.0) * inv * side * (rc.y - 0.35) + vec2(1.0, dz) * inv * (f.speed * 0.35 * age);
        foam += splashRing(p - c, age, 1.0, g);
    }
    return foam;
}

// rings spread at a few speeds (capillary-gravity waves disperse); all of it carried downstream by the current.
// dropHeight is the displacement (m), dropFoam the white water: crown, ring crests and a drifting bubble patch.
float dropImpact(uint i) { return F.splashEv[2u * i].w; }
// Impact geometry from size sc and entry speed U: cavity radius grows with the Froude number, it collapses in
// ~sqrt(R / g) (the jet rises then), the crown reaches ~U^2 / 2g capped by the cavity scale.
struct Drop { vec2 c; float age, dist, sc, e, Rc, tc, crown; };
bool dropLocal(uint i, vec2 p, float T, out Drop d) {
    vec4 e = F.splashEv[2u * i];
    d.age = T - e.z;
    d.sc = F.splashEv[2u * i + 1u].x;
    if (e.z < 0.0 || d.age < 0.0 || d.age > 40.0) return false;
    float U = max(F.splashEv[2u * i + 1u].y, 0.5), Fr = U / sqrt(9.81 * max(d.sc, 0.05));
    d.e = e.w;
    d.Rc = 0.5 * d.sc * (1.0 + 0.35 * sqrt(Fr)) + 0.05;
    d.tc = 1.6 * sqrt(d.Rc / 9.81) * (1.0 + 0.2 * sqrt(Fr));
    d.crown = min(U * U / 19.62, 1.2 * d.Rc) * 0.5;
    // the whole disturbance drifts with the current; it slows as it spreads (it is carried by the mean flow)
    d.c = e.xy + riverFlow(e.xy).v * d.age;
    d.dist = length(p - d.c) + 1e-4;
    // fastest group (longest wave ~4 Rc) and the spreading bubble patch bound the reach
    float cg = 0.5 * sqrt(9.81 * 4.0 * d.Rc / (2.0 * PI)) + 0.3;
    return d.dist < max(cg * d.age * 1.3 + d.Rc + 1.0, 3.0 * sqrt(sq(d.Rc + 0.3) + 0.04 * d.age) + 0.5);
}
// capillary-gravity dispersion, deep water: w^2 = g k + (sigma / rho) k^3
float cgPhase(float k) { return sqrt(9.81 / k + 7.3e-5 * k); }
float cgGroup(float k) { return (9.81 + 3.0 * 7.3e-5 * k * k) / (2.0 * sqrt(9.81 * k + 7.3e-5 * k * k * k)); }
float dropHeight(vec2 p, float T) {
    float h = 0.0;
    Drop d;
    for (uint i = 0u; i < F.dynInfo.y; i++) {
        if (!dropLocal(i, p, T, d)) continue;
        float e = min(d.e, 2.0), x = d.dist / d.Rc;
        // cavity opens and refills over tc (overshooting into the jet after)
        float ph = d.age / d.tc;
        h -= e * d.Rc * 0.6 * sin(PI * min(ph, 1.0)) * exp(-x * x * 1.5);
        // crown: a rim thrown out and up while the cavity opens, then falling back
        float rim = d.Rc * (1.0 + 0.8 * min(ph, 1.5));
        h += e * d.crown * sin(PI * clamp(ph / 1.2, 0.0, 1.0)) * exp(-sq((d.dist - rim) / (0.2 * d.Rc + 0.03)));
        // Worthington jet: the collapse throws a column up, it falls in ~sqrt(2 crown / g) * 2
        float tj = (d.age - d.tc) / (0.8 * d.tc + 0.15);
        if (tj > 0.0 && tj < 1.0) h += e * 1.4 * d.crown * sin(PI * tj) * exp(-sq(x / 0.3));
        // ring waves from the collapse: three wavelengths around the cavity scale, each packet travelling at its
        // group speed with crests moving through it at the phase speed. Amplitude falls as 1/sqrt(r) (energy
        // spreading on the surface) times viscous / turbulent damping; short waves die fastest.
        float ta = max(d.age - 0.5 * d.tc, 0.0);
        for (int k = 0; k < 3; k++) {
            float lam = 4.0 * d.Rc * pow(0.45, float(k)) + 0.04, kw = 2.0 * PI / lam;
            float cg = cgGroup(kw), cp = cgPhase(kw);
            float front = d.Rc + cg * ta, wd = lam * (1.2 + 0.4 * ta * kw * 0.05);
            float env = exp(-sq((d.dist - front) / wd)) * smoothstep(0.0, d.Rc + 0.3, d.dist);
            float amp = e * 0.25 * d.Rc * sqrt(d.Rc / (d.Rc + cg * ta)) * exp(-ta * (0.08 + 0.02 * kw)) * (1.0 - 0.3 * float(k));
            h += amp * env * cos(kw * (d.dist - d.Rc - cp * ta));
        }
    }
    return h;
}
// White water: the crown and jet sheets fall back as a bubble patch around the impact, plus crest foam on the
// first, steepest ring. The patch keeps its foam mass while it spreads (diffusion, 1/area) and the bubbles
// burst slowly (half-life ~8 s), so it lingers and drifts off with the current instead of blinking out.
float dropFoam(vec2 p, float T) {
    float foam = 0.0;
    Drop d;
    for (uint i = 0u; i < F.dynInfo.y; i++) {
        if (!dropLocal(i, p, T, d)) continue;
        float e = min(d.e, 2.0);
        float R0 = d.Rc * 1.4 + 0.15, R = sqrt(R0 * R0 + 0.04 * d.age);
        float settle = smoothstep(0.0, d.tc * 1.5, d.age);  // the patch forms as the splash falls back
        // break the disc up: the mass sits in streaks and clumps that stretch as it spreads
        vec2 q = (p - d.c) / R;
        float lumpy = 0.6 + 0.8 * vnoise(q * 2.5 + float(i) * 7.3);
        foam += e * 1.2 * sq(R0 / R) * exp(-d.age / 12.0) * exp(-dot(q, q) * 1.6) * lumpy * settle;
        // churned-up centre right after the collapse
        foam += e * 1.5 * exp(-sq(d.dist / R0)) * exp(-d.age / (2.0 * d.tc + 0.5)) * smoothstep(0.0, d.tc, d.age);
        // crest foam on the leading ring, gone in a second or two
        float lam = 4.0 * d.Rc + 0.04, kw = 2.0 * PI / lam, ta = max(d.age - 0.5 * d.tc, 0.0);
        float front = d.Rc + cgGroup(kw) * ta;
        foam += e * 0.5 * exp(-sq((d.dist - front) / (0.4 * lam))) * exp(-1.2 * ta) * step(0.0, d.age - 0.5 * d.tc);
    }
    return foam;
}
#endif
