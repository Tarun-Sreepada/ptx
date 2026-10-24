// Water shading: surface, reflection, refraction, absorption, foam; the view from under water.
#ifndef WATER_SHADE
#define WATER_SHADE
#include "core/frame.glsl"
#include "water/flow.glsl"
#include "water/splashes.glsl"
#include "water/rocks.glsl"

bool curPixel(vec3 P, out ivec2 q) {
    vec3 d = P - F.camPos.xyz;
    float z = dot(d, F.camFwd.xyz);
    if (z <= 1e-3) return false;
    float x = dot(d, F.camRight.xyz) / (z * F.camUp.w * F.camRight.w), y = dot(d, F.camUp.xyz) / (z * F.camUp.w);
    q = ivec2(floor(vec2(x * 0.5 + 0.5, 0.5 - y * 0.5) * vec2(F.res.xy) - F.jitter.xy));
    return inScreen(q);
}

struct Water { bool hit; float t; vec3 p, n, col; float foam, fres; };
const float ICE_MARGIN = 15.0;  // m: water freezes as the snow line comes down to this far above it

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

    // ice: water freezes when the snow line is below it. Ponds freeze over; rivers only along their slow, shallow
    // edges. A flat sheet: clear and dark where it's young, white with frost and trapped bubbles, cracked.
    float freeze = smoothstep(w.p.y + ICE_MARGIN, w.p.y - 5.0, F.sim0.x);
    if (freeze > 0.0) {
        float cover = pond >= 0 ? freeze : freeze * (1.0 - smoothstep(0.1, 0.5, thick)) * (1.0 - smoothstep(0.3, 1.0, f.speed));
        cover *= smoothstep(0.25, 0.45, cover + 0.2 * (vnoise(xz * 0.7) - 0.5));  // the sheet's ragged edge
        if (cover > 0.0) {
            float crack = 1.0 - smoothstep(0.0, 0.035, abs(vnoise(xz * 1.3) - 0.5));
            float frost = smoothstep(0.45, 0.8, vnoise(xz * 2.1 + 9.0) * 0.7 + vnoise(xz * 9.0) * 0.3);
            vec3 albedo = mix(vec3(0.5, 0.64, 0.75), vec3(0.85, 0.9, 0.95), frost) * (1.0 - 0.35 * crack);
            vec3 iceSpec = min(F.sunColor.rgb * vis * ggxSpec(vec3(0, 1, 0), v, F.sunDir.xyz, 0.08), F.sunColor.rgb * 20.0);
            vec3 ice = albedo * (F.sunColor.rgb * vis * max(F.sunDir.y, 0.0) / PI + skyUp) + (refl * 0.25 + iceSpec * 0.4) * (1.0 - frost);
            col = mix(col, ice, cover);
            w.foam *= 1.0 - cover;
        }
    }
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
