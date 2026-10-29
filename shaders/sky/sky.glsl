// Sky radiance: the atmosphere LUT, night sky, lightning glow, and the far-field terrain past the mesh.
#ifndef SKY_SKY
#define SKY_SKY
#include "core/frame.glsl"
#include "core/bindings.glsl"
#include "core/util.glsl"
#include "terrain/terrain.glsl"
#include "sky/atmosphere.glsl"
#include "lighting/probes.glsl"
#include "sky/night.glsl"
#include "sky/lightning.glsl"

// Sky radiance: physical atmosphere LUT (sky.comp: sun- and moonlight scattered), plus the night background.
vec3 skyAtmo(vec3 d) {
    float el = asin(clamp(d.y, -1.0, 1.0));
    vec2 uv = vec2(atan(d.x, d.z) / (2.0 * PI) + 0.5, skyV(el));
    vec2 st = uv * vec2(SKY_W, SKY_H) - 0.5;
    ivec2 i0 = ivec2(floor(st));
    vec2 f = st - vec2(i0);
    ivec2 a0 = ivec2((i0.x % SKY_W + SKY_W) % SKY_W, clamp(i0.y, 0, SKY_H - 1));
    ivec2 a1 = ivec2((a0.x + 1) % SKY_W, clamp(i0.y + 1, 0, SKY_H - 1));
    vec3 c = mix(mix(imageLoad(skyLUT, a0).rgb, imageLoad(skyLUT, ivec2(a1.x, a0.y)).rgb, f.x),
                 mix(imageLoad(skyLUT, ivec2(a0.x, a1.y)).rgb, imageLoad(skyLUT, a1).rgb, f.x), f.y);
    float overcast = F.clouds.w > 0.5 ? 1.0 - 0.45 * F.clouds.x * F.clouds.x : 1.0;  // cloud cover greys out the blue sky
    return c * F.sunColor.w * overcast + nightAmbient(d) * (F.clouds.w > 0.5 ? 1.0 - 0.8 * F.clouds.x : 1.0) + lightningSky(d);
}
// Far field: past the terrain mesh (F.terrainP extent, set by the view distance) a downward ray that missed
// everything would see black. Instead march the large-scale terrain noise (the same function, coarse terms only)
// out to 40 km, shade with the ground colour, sun + sky light (no shadows), and fade into the aerial haze.
// Cheap: only rays that leave the mesh pay, ~28 steps.
const int FAR_STEPS = 96;  // most rays end well before: the steps stretch where the ground is far below
const float FAR_GROWTH = 1.16;  // each step this much further: fine enough not to step over the far ridges
float farHeight(vec2 p) {
    return biomeShape(p, 2.2 * tnoise(p.x * 0.022 + 10.0, p.y * 0.022) + 0.7 * tnoise(p.x * 0.07, p.y * 0.07 + 5.0) - 1.5 + hills(p) + farRelief(p));
}
// Forest cover past the tree range, from the same forest noise the placement uses (main.cpp candidate): the far
// field gets a canopy layer (colour + ~9 m of height) so forests continue to the horizon as low-detail LOD.
float forestCover(vec2 p) {
    float f = tnoise(p.x * 0.0045 + 31.0, p.y * 0.0045 - 7.0) * 0.7 + tnoise(p.x * 0.017, p.y * 0.017 + 11.0) * 0.3;
    return clamp((smoothstep(0.5, 0.66, f) * 0.8 + 0.025) * F.water2.z * 1.6, 0.0, 1.0);  // trees per cell -> canopy cover
}
float farHeightC(vec2 p, out float cov) {
    cov = forestCover(p);
    float crowns = 0.6 + 0.4 * tnoise(p.x * 0.11, p.y * 0.11 + 3.0);  // uneven crown tops
    return farHeight(p) + cov * 9.0 * crowns;
}
float cloudShadow(vec3 x);  // clouds.glsl
// highest the far field gets (m): the horizon mountains + the base noise + canopy
const float FAR_TOP = RELIEF_HEIGHT * 1.1 + 15.0;
float farHitT;  // distance along the ray of farGround's last hit
vec3 farGround(vec3 o, vec3 d, out bool hit) {
    hit = false;
    float horiz = length(d.xz);
    if (horiz < 1e-4) return vec3(0);
    float ext = F.terrainP.w;  // the terrain mesh reaches this far from its centre
    float t = max(0.9 * ext / horiz, 0.0), tEnd = 40000.0;
    // looking up: the peaks past the view distance still rise above the eye; a ray already above every summit
    // where the far field starts can't meet one
    if (d.y >= 0.0) {
        if (o.y + d.y * t > FAR_TOP) return vec3(0);
        tEnd = min(tEnd, (FAR_TOP - o.y) / max(d.y, 1e-6));
    }
    // the sea past the mesh: its surface is the floor of the march, so the ocean runs on to the horizon
    bool ocean = biomeOn() && F.biomeP[1].z > 0.0;
    float sea = ocean ? F.biomeP[2].x : -1e9;
    float prevT = t, prevGap = 1e30;
    for (int i = 0; i < FAR_STEPS && t < tEnd; i++) {
        vec3 q = o + d * t;
        float cv;
        float gap = q.y - max(farHeightC(q.xz, cv), sea);
        if (gap < 0.0) {
            t = prevT + (t - prevT) * prevGap / max(prevGap - gap, 1e-4);  // secant
            hit = true;
            break;
        }
        prevT = t; prevGap = gap;
        // step by the height above the ground (the relief is never steeper than ~1:1), so near a ridge the samples
        // close in on it instead of jumping over it: fixed geometric steps hopped across crests at a spacing tied to
        // the camera, and the far mountains changed shape as it moved
        t += clamp(gap * 0.8, 4.0 + t * 0.002, t * (FAR_GROWTH - 1.0) + 20.0);
    }
    if (!hit) {  // the mean ground plane
        if (d.y >= 0.0) return vec3(0);  // passed over every peak: sky
        float tp = (o.y - max(-1.5, sea)) / -d.y;
        if (tp > tEnd * 1.5) return vec3(0);
        t = tp;
        hit = true;
    }
    farHitT = t;
    vec3 P = o + d * t;
    float e = max(2.0, t * 0.004), cov, c2;
    // aerial perspective as composite.comp does it for geometry (sky pixels skip that pass)
    float h = max(P.y + VIEW_H, 0.0);
    vec3 T = exp(-(BR * exp(-h / HR) + vec3(BME * exp(-h / HM))) * F.params2.w * t);
    vec3 haze = skyAtmo(normalize(vec3(d.x, 0.03, d.z)));
    if (farHeightC(P.xz, cov) < sea) {  // open water: deep colour under a Fresnel sky reflection + the sun's glint
        vec2 rp = P.xz * 0.02 + F.wind.xy * F.camPos.w * 0.05;
        vec3 n = normalize(vec3((tnoise(rp.x, rp.y) - 0.5) * 0.08, 1.0, (tnoise(rp.x + 31.0, rp.y - 17.0) - 0.5) * 0.08));
        vec3 r = reflect(d, n);
        r.y = abs(r.y);
        float fr = 0.02 + 0.98 * pow(1.0 - max(dot(-d, n), 0.0), 5.0);
        vec3 deep = vec3(0.004, 0.022, 0.035) * (F.sunColor.rgb * max(F.sunDir.y, 0.0) + skyAtmo(vec3(0, 1, 0)));
        vec3 glint = F.sunColor.rgb * pow(max(dot(r, F.sunDir.xyz), 0.0), 400.0) * 40.0 * cloudShadow(P);
        vec3 col = mix(deep, skyAtmo(r) + glint, fr);
        return col * T + haze * (1.0 - T);
    }
    vec3 n = normalize(vec3(farHeightC(P.xz - vec2(e, 0), c2) - farHeightC(P.xz + vec2(e, 0), c2), 2.0 * e,
                            farHeightC(P.xz - vec2(0, e), c2) - farHeightC(P.xz + vec2(0, e), c2)));
    vec3 alb = terrainAlbedo(P.xz, 1.0);
    // canopy: dark foliage, broadleaf / conifer patches as the variants are chosen; crowns self-shadow (darker
    // than the ground at the same sun angle)
    vec3 canopy = mix(vec3(0.045, 0.075, 0.03), vec3(0.025, 0.05, 0.03), step(0.5, tnoise(P.x * 0.01 + 50.0, P.y * 0.01)));
    alb = mix(alb, canopy * (0.75 + 0.5 * tnoise(P.x * 0.3, P.y * 0.3)), cov);
    // the mountains: bare rock on the steep faces, snow above the snow line where it can lie
    alb = mix(alb, vec3(0.30, 0.29, 0.27) * (0.8 + 0.4 * tnoise(P.x * 0.02, P.y * 0.02)), smoothstep(0.85, 0.62, n.y) * (1.0 - 0.7 * cov));
    alb = mix(alb, vec3(0.82, 0.85, 0.9), smoothstep(F.sim0.x - 40.0, F.sim0.x + 60.0, P.y) * smoothstep(0.5, 0.75, n.y));
    // cloud shadows: the far field darkens under the same clouds as the near ground, so the illusion holds
    vec3 col = alb * (F.sunColor.rgb * max(dot(n, F.sunDir.xyz), 0.0) * cloudShadow(P) / PI + skyAtmo(vec3(0, 1, 0)) * (0.5 + 0.5 * n.y));
    return col * T + haze * (1.0 - T);
}
vec3 sky(vec3 d) {
    if (d.y < 0.3) {  // (above that no ray from below the summits reaches one)
        bool hit;
        vec3 g = farGround(F.camPos.xyz, d, hit);
        if (hit) return g;
    }
    return skyAtmo(d);
}
#endif
