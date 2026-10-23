// River surface, shaded in screen space over the already lit river bed (composite / RR input), after denoising.
// Published ideas, approximated for speed:
//  - flow: an analytic velocity field instead of a shallow-water solve. Fast laminar core, a shear layer toward
//    the banks, and faster water where the channel narrows / shallows (continuity: u ~ Q / (width * depth)).
//  - surface detail: flow-mapped normals (Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010 course): two noise
//    layers advected along the flow with phases half a period apart, crossfaded so neither stretches for long.
//    Plus a few wind ripples with the deep-water dispersion w^2 = g k (Tessendorf, "Simulating Ocean Water").
//  - bank splashes: short-lived expanding rings with foam, wave-particle style (Yuksel, House & Keyser,
//    "Wave Particles", SIGGRAPH 2007), more of them on the outer bank of bends where the current hits the bank.
//  - optics: Fresnel reflection (one traced ray, reusing the lit screen pixel when the hit is on screen),
//    refraction by a screen-space offset, Beer-Lambert absorption + in-scatter over the water path length.
//  - foam: shallow water (anything under the surface, so rocks and banks both get it), shear and splashes,
//    advected with the flow and thresholded against slow noise so it comes and goes.
// The includer defines vec3 waterScene(ivec2 q): lit radiance of pixel q without aerial perspective.
#ifndef WATER_GLSL
#define WATER_GLSL

const float WATER_PERIOD = 2.0;  // flow-map reset period (s)

struct Flow { vec2 v; float speed, shear; vec4 r; };  // r = riverInfo
Flow riverFlow(vec2 p) {
    Flow f;
    f.r = riverInfo(p);
    float drop;
    float divK = F.dynInfo.z > 0u ? riverDiversion(p.x, drop) : 1.0;
    float dz = riverSlope(p.x);
    vec2 dir = vec2(1.0, dz) * inversesqrt(1.0 + dz * dz);
    float c = min(abs(f.r.x) / f.r.y, 1.0);
    // more water after rain runs faster (the swell over the channel depth, roughly)
    float U = F.water.x * 1.3 * (F.river.x * F.river.y) / (f.r.y * f.r.z) * divK * (1.0 + 1.6 * F.sim2.w / max(F.river.y, 0.3));
    float k = max(1.0 - c * c, 0.02);
    f.speed = U * pow(k, 0.35) * step(c, 0.999);
    f.v = dir * f.speed;
    f.shear = min(U * 0.7 * c * pow(k, -0.65) / f.r.y, 4.0);  // |du/dn|
    return f;
}

float whash(vec2 i) { return float(hashu(uint(int(i.x)) * 7919u + uint(int(i.y)) * 104729u)) / 4294967295.0; }
// value noise with analytic gradient: x = value, yz = d/dp
vec3 vnoiseD(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f), du = 6.0 * f * (1.0 - f);
    float a = whash(i), b = whash(i + vec2(1, 0)), c = whash(i + vec2(0, 1)), d = whash(i + vec2(1, 1));
    float k1 = b - a, k2 = c - a, k3 = a - b - c + d;
    return vec3(a + k1 * u.x + k2 * u.y + k3 * u.x * u.y, du * (vec2(k1, k2) + k3 * u.yx));
}
vec2 waveGrad(vec2 p) {
    return 0.5 * 1.1 * vnoiseD(p * 1.1).yz + 0.25 * 2.7 * vnoiseD(p * 2.7 + 5.3).yz + 0.12 * 6.3 * vnoiseD(p * 6.3 - 2.1).yz;
}
float caustics(vec2 p) {
    float a = 1.0 - abs(vnoise(p) * 2.0 - 1.0), b = 1.0 - abs(vnoise(p * 1.9 + 4.0) * 2.0 - 1.0);
    return pow(a, 6.0) + pow(b, 8.0) * 0.6;
}
float foamNoise(vec2 p) { return vnoise(p * 2.3) * 0.6 + vnoise(p * 7.1) * 0.4; }

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
    float curv = -0.154 * 0.011 * sin(0.011 * p.x) - 0.189 * 0.027 * sin(0.027 * p.x + 1.3);  // centreline z''
    float outer = clamp(-side * curv * 400.0, 0.0, 1.0);
    float foam = 0.0;
    for (int j = -1; j <= 1; j++) {
        float cell = floor(p.x / 3.0) + float(j);
        vec2 key = vec2(cell, side * 17.0 + 3.0);
        float Tp = 2.2 + 2.5 * whash(key + vec2(0, 41)), tt = T + whash(key + vec2(0, 93)) * Tp;
        float epoch = floor(tt / Tp), age = tt - epoch * Tp;
        if (age > 1.6 || whash(key + vec2(epoch * 7.0, 11)) > F.water.w * (0.3 + 0.6 * outer) * min(F.water.x, 1.5)) continue;
        float cx = (cell + 0.2 + 0.6 * whash(key + vec2(0, 7))) * 3.0;
        vec4 rc = riverInfo(vec2(cx, riverCenter(cx)));
        float dz = riverSlope(cx), inv = inversesqrt(1.0 + dz * dz);
        vec2 c = vec2(cx, riverCenter(cx)) + vec2(-dz, 1.0) * inv * side * (rc.y - 0.35) + vec2(1.0, dz) * inv * (f.speed * 0.35 * age);
        foam += splashRing(p - c, age, 1.0, g);
    }
    return foam;
}

// Foam trail length (water2.x): wakes fade out smoothly before it instead of being cut off
float wakeFade(float a) { return 1.0 - smoothstep(F.water2.x * 0.3, F.water2.x, a); }
// Wake strength along the trail: decays over a good part of the trail length (bubbles take a while to pop),
// not a fixed few metres, so the white water thins out gradually all the way down
float wakeTail(float a, float R) { return smoothstep(-0.3 * R, R, a) * exp(-max(a, 0.0) / (0.3 * F.water2.x + 2.0 * R + 1.0)) * wakeFade(a); }

// River rocks near p (terrain.h layout): c, flow direction, waterline radius R (0 = under water), top depth sub, scale
struct RockW { vec2 c, dir; float R, sub, sc, u; };  // u: own flow speed past it (thrown rocks), < 0 = the river's
const int MAX_NEAR_ROCKS = 16;  // up to 9 placed + thrown ones in reach
int nearRocks(vec2 p, float level, out RockW rk[MAX_NEAR_ROCKS]) {
    int n = 0, ci = int(floor(p.x / RIVER_ROCK_L));
    int up = int(ceil(F.water2.x / RIVER_ROCK_L)) + 1;  // wakes trail downstream (+x): look as far upstream as they reach
    for (int i = ci - up; i <= ci + 1 && n < MAX_NEAR_ROCKS; i++) {
        vec4 r;
        if (!riverRock(i, r)) continue;
        RockW k;
        k.sub = level - (r.y + 0.72 * r.w);
        if (k.sub > 1.0 * r.w + 0.3) continue;  // too deep to show on the surface
        float dz = riverSlope(r.x), hh = (level - r.y) / (0.72 * r.w);
        k.c = r.xz;
        k.dir = vec2(1.0, dz) * inversesqrt(1.0 + dz * dz);
        k.sc = r.w;
        k.u = -1.0;
        k.R = k.sub < 0.0 ? 0.95 * r.w * sqrt(clamp(1.0 - hh * hh, 0.05, 1.0)) : 0.0;
        rk[n++] = k;
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

bool curPixel(vec3 P, out ivec2 q) {
    vec3 d = P - F.camPos.xyz;
    float z = dot(d, F.camFwd.xyz);
    if (z <= 1e-3) return false;
    float x = dot(d, F.camRight.xyz) / (z * F.camUp.w * F.camRight.w), y = dot(d, F.camUp.xyz) / (z * F.camUp.w);
    q = ivec2(floor(vec2(x * 0.5 + 0.5, 0.5 - y * 0.5) * vec2(F.res.xy) - F.jitter.xy));
    return inScreen(q);
}

struct Water { bool hit; float t; vec3 p, n, col; float foam, fres; };

// p: pixel, d: camera ray, P: G-buffer position (w = distance, -1 = sky), scene: waterScene(p)
Water shadeWater(ivec2 p, vec3 d, vec4 P, vec3 scene) {
    Water w;
    w.hit = false;
    if (P.w < 0.0 || d.y > -1e-4) return w;
    // a river drawn on the world map (its own level and flow) where the main river isn't
    vec4 mw = mapWaterAt(P.xz);
    bool drawn = mw.x > P.y + 0.003;
    if (!drawn && (F.river.x <= 0.0 || !riverNear(P.xz, RIVER_BANK))) return w;
    vec3 o = F.camPos.xyz;
    vec4 rb = riverInfo(P.xz);
    float drop = 0.0;
    if (F.dynInfo.z > 0u) riverDiversion(P.x, drop);
    int crater = F.dynInfo.z > 0u ? craterAt(P.xz) : -1;
    int pond = crater >= 0 && F.craterW[crater].x > -1e8 ? crater : -1;
    if (crater >= 0 && pond < 0) return w;  // a dry crater (not fed by the river)
    float t;
    if (drawn && pond < 0) {
        t = (o.y - mw.x) / -d.y;
    } else if (pond >= 0) {  // a crater filling from the river: its own (rising) level
        float lv = F.craterW[pond].x;
        if (lv <= P.y) return w;
        t = (o.y - lv) / -d.y;
    } else {
        drop -= F.sim2.w;  // rain swells the river: the surface rises (onto the banks), the bed doesn't
        if (rb.w - drop <= P.y || abs(rb.x) > rb.y + RIVER_BANK) return w;  // the bed under this pixel is dry
        // the surface is a plane per x (level drops slowly downstream): two fixed-point steps land on it
        t = (o.y - (rb.w - drop)) / -d.y;
        t = (o.y - (F.river.z - F.river.w * (o.x + d.x * t) - drop)) / -d.y;
    }
    // splash displacement: walk the ray onto the moving surface (fixed point, damped)
    if (F.dynInfo.y > 0u && t > 0.0 && pond < 0 && !drawn)
        for (int i = 0; i < 3; i++) {
            vec3 S = o + d * t;
            float tn = (o.y - (F.river.z - F.river.w * S.x + dropHeight(S.xz, F.camPos.w))) / -d.y;
            t = mix(t, tn, 0.7);
        }
    if (t <= 0.0 || t > P.w) return w;
    w.hit = true;
    w.t = t;
    w.p = o + d * t;
    vec2 xz = w.p.xz;
    float T = F.camPos.w;
    Flow f = riverFlow(xz);
    if (drawn && pond < 0) {  // the drawn river's flow; no rocks / bank splashes from the main river's layout
        f.v = mw.yz;
        f.speed = length(mw.yz);
        f.shear = 0.3 + 0.5 * f.speed / max(mw.w, 1.0);
        f.r = vec4(1e3, mw.w, 1.0, mw.x);
    }
    if (pond >= 0 && F.craterW[pond].y < 1.0) {  // pouring in: flow toward the bowl's centre, churned
        vec2 toC = F.craters[pond].xy - xz;
        float r = length(toC), sp = F.craterW[pond].w * (0.4 + 0.6 * min(r / F.craters[pond].z, 1.0));
        f.v = toC / max(r, 1e-3) * sp;
        f.speed = sp;
        f.shear = 1.5 * (1.0 - F.craterW[pond].y);
    }
    float thick = max(w.p.y - P.y, 0.0);
    float edge = 1.0 - smoothstep(0.0, 0.15, thick);
    float turb = 0.25 * f.shear + 0.8 * edge;

    // rocks: deflect the advection velocity, add their surface shape, foam and turbulence
    vec2 vAdv = f.v, rg = vec2(0);
    float rockFoam = 0.0;
    if (abs(f.r.x) < f.r.y + 0.5) {
        RockW rk[MAX_NEAR_ROCKS];
        int nr = nearRocks(xz, f.r.w, rk);
        float U = max(f.speed, 0.3 * F.water.x), h0 = 0.0, hx = 0.0, hz = 0.0;
        const float e = 0.04;
        for (int i = 0; i < nr; i++) {
            rockFlow(xz, rk[i], U, T, vAdv, rockFoam, turb, rg);
            h0 += rockHeight(xz, rk[i], U);
            hx += rockHeight(xz + vec2(e, 0), rk[i], U);
            hz += rockHeight(xz + vec2(0, e), rk[i], U);
        }
        rg += vec2(hx - h0, hz - h0) / e;
    }
    turb = clamp(turb, 0.0, 1.0);

    // flow-mapped detail: two advected layers half a period apart
    float ph0 = fract(T / WATER_PERIOD), ph1 = fract(T / WATER_PERIOD + 0.5), blend = abs(1.0 - 2.0 * ph0);
    vec2 q0 = xz - vAdv * (ph0 * WATER_PERIOD), q1 = xz - vAdv * (ph1 * WATER_PERIOD) + vec2(0.43, 0.71);
    vec2 g = mix(waveGrad(q0), waveGrad(q1), blend) * (0.012 + 0.05 * turb + 0.012 * f.speed);
    // wind ripples, deep-water dispersion
    vec3 lw = windAt(xz);  // local wind map: gust fronts roughen the water as they pass
    float wa = atan(lw.y, lw.x), ws = length(lw.xy);
    for (int i = 0; i < 4; i++) {
        float lam = 0.9 * pow(0.6, float(i)), k = 2.0 * PI / lam, ang = wa + (float(i) - 1.5) * 0.55;
        vec2 dw = vec2(cos(ang), sin(ang));
        g += dw * (0.03 * lam * ws * k * cos(k * dot(dw, xz) - sqrt(9.81 * k) * T + float(i) * 1.7)) * 0.1;
    }
    g += rg;
    float splashFoam = splashes(xz, f, T, g) + rockFoam;
    if (F.dynInfo.y > 0u) {
        const float e = 0.03;
        float h0 = dropHeight(xz, T);
        g += vec2(dropHeight(xz + vec2(e, 0), T) - h0, dropHeight(xz + vec2(0, e), T) - h0) / e;
        splashFoam += dropFoam(xz, T);
    }
    vec3 n = normalize(vec3(-g.x, 1.0, -g.y));
    w.n = n;

    vec3 v = -d;
    float nv = max(dot(n, v), 0.0);
    w.fres = 0.02 + 0.98 * pow(1.0 - nv, 5.0);
    float vis = visible(w.p + vec3(0, 0.03, 0), F.sunDir.xyz, 1e4) ? cloudShadow(w.p) : 0.0;
    vec3 skyUp = sky(vec3(0, 1, 0));
    vec3 amb = F.sunColor.rgb * vis * max(F.sunDir.y, 0.0) * (0.25 / PI) + skyUp;  // in-scatter: mostly forward, little comes back up

    // reflection
    vec3 r = reflect(d, n);
    if (r.y < 0.01) r = normalize(vec3(r.x, 0.01, r.z));
    // Fully ray traced reflection, the same at every view angle (no screen-space reuse): the reflected surface gets
    // sun (shadowed), one light from the light list, sky light (its ambient occlusion) and its own emission.
    Hit h = traceClosest(w.p + vec3(0, 0.02, 0), r, 1e4, vec2(pixelSpread() * t, pixelSpread() + 0.05));
    vec3 refl;
    ivec2 q;
    if (!h.hit) refl = skyWithSun(r);
    else {
        uint s = seedFor(p, 91u);
        vec3 skyAmb = sky(normalize(h.n + vec3(0, 1, 0))) * (0.55 + 0.45 * h.n.y) * h.s.ao;
        refl = h.albedo * (sunDirect(h.p, h.n, h.mat, s) + treeNEE(h.p, h.n, h.mat, s) + skyAmb) + h.emis;
        // aerial perspective along the reflected ray
        if (F.params2.w > 0.0) {
            float hh = max(h.p.y + VIEW_H, 0.0);
            vec3 T = exp(-(BR * exp(-hh / HR) + vec3(BME * exp(-hh / HM))) * F.params2.w * h.t);
            refl = refl * T + sky(normalize(vec3(r.x, max(r.y, 0.03), r.z))) * (1.0 - T);
        }
    }

    // refraction: follow the bent ray roughly as deep as the water is, take that pixel if it is under water
    vec3 rd = refract(d, n, 0.75);
    vec3 under = scene;
    float path = P.w - t;
    if (curPixel(w.p + rd * (thick / max(-rd.y, 0.3)), q)) {
        vec4 Q = loadPos(q);
        if (Q.w > t && Q.y < w.p.y - 0.01) { under = waterScene(q); path = length(Q.xyz - w.p); }
    }
    vec3 sigma = F.waterCol.w * vec3(1.0, 0.42, 0.32) / max(F.water.y, 0.05);
    // (the bed's caustics are part of its sun lighting now: common.glsl underwaterSun)
    vec3 Tr = exp(-sigma * path);
    under = under * Tr + F.waterCol.rgb * amb * (1.0 - Tr);

    float rough = 0.06 + 0.25 * turb;
    vec3 spec = min(F.sunColor.rgb * vis * ggxSpec(n, v, F.sunDir.xyz, rough), F.sunColor.rgb * 30.0);
    vec3 col = mix(under, refl, w.fres) + spec;

    // foam
    float amount = F.water.z * (0.55 * edge + 0.06 * f.shear * f.speed + splashFoam);
    amount *= 0.55 + 0.9 * vnoise(xz * 0.15 + vec2(-T * 0.12, T * 0.07));  // comes and goes
    float fn = mix(foamNoise(q0), foamNoise(q1 + vec2(3.1, 1.7)), blend);
    // coverage never reaches 100%: foam stays a lacy pattern, denser where there's more of it
    // sqrt: a little foam already shows as sparse flecks (a linear map put it under the noise's range, so a trail
    // vanished at once when it thinned); the fade-in keeps the last of it from popping out
    float am = clamp(amount, 0.0, 1.0), cover = 0.75 * sqrt(am);
    float foam = smoothstep(0.92 - cover, 1.12 - cover, fn) * (0.55 + 0.45 * smoothstep(0.5, 0.9, fn)) * smoothstep(0.0, 0.2, am);
    foam = max(foam, (1.0 - smoothstep(0.0, 0.035, thick)) * smoothstep(0.35, 0.6, fn) * 0.8 * min(F.water.z, 1.0));  // broken contact line
    w.foam = foam;
    vec3 foamCol = vec3(0.7) * (F.sunColor.rgb * vis * max(dot(n, F.sunDir.xyz), 0.0) / PI + skyUp * 0.9);
    col = mix(col, foamCol, foam);
    w.col = mix(scene, col, smoothstep(0.0, 0.03, thick));
    return w;
}

// The camera under a water surface (river, crater pond, drawn river): everything is seen through the water, absorbed
// and scattered with distance (the murk colour), dimmer with depth; the surface above isn't shaded from below.
vec3 underwaterView(vec3 col, float dist) {
    float lv = waterLevelAt(F.camPos.xz);
    float depth = lv - F.camPos.y;
    if (depth <= 0.02) return col;
    vec3 sigma = vec3(0.45, 0.12, 0.08) * max(F.waterCol.w, 0.05) * 3.0;  // red goes first
    vec3 T = exp(-sigma * min(dist, 200.0));
    vec3 lit = F.sunColor.rgb * max(F.sunDir.y, 0.0) * 0.05 + sky(vec3(0, 1, 0)) * 0.3;
    vec3 murk = F.waterCol.rgb * lit * exp(-0.15 * depth);
    return col * T + murk * (1.0 - T);
}
#endif
