// Night sky: moon (phase-lit disc, earthshine), stars, the Milky Way and airglow, in physical luminance (cd/m^2)
// scaled to scene units by F.night0.w (night light boost / scene luminance per unit, main.cpp).
// The sky dome's frame: the celestial pole leans north (-z) at the observer's latitude; stars turn with the sun
// (F.night1.x = the sun's angle about the pole), so moving the sun turns the sky.
#ifndef NIGHT_GLSL
#define NIGHT_GLSL

vec3 celestialPole() { float l = F.night1.y; return vec3(0.0, sin(l), -cos(l)); }
// direction -> (right ascension-like angle, declination), fixed to the stars
vec2 celestial(vec3 d) {
    vec3 P = celestialPole();
    vec3 e1 = normalize(vec3(0.0, cos(F.night1.y), sin(F.night1.y)));  // the equator due south
    vec3 e2 = cross(P, e1);                                              // east
    float a = atan(dot(d, e2), dot(d, e1)) - F.night1.x;
    return vec2(a, asin(clamp(dot(d, P), -1.0, 1.0)));
}
// relative airmass toward elevation y (Kasten & Young-ish, finite at the horizon)
float airmass(float y) { return 1.0 / (max(y, 0.0) + 0.15 * pow(max(y, 0.0) * 57.3 + 3.885, -1.253)); }

// Smooth night background: airglow (bright toward the horizon, van Rhijn; the 557.7 nm oxygen line, greenish),
// integrated starlight and zodiacal light. Cheap: the sky light for GI and fog. ~2e-4 cd/m^2 at the zenith.
vec3 nightAmbient(vec3 d) {
    if (F.night0.w <= 0.0) return vec3(0);
    float y = max(d.y, 0.0);
    float c2 = 1.0 - y * y;
    float vr = 1.0 / sqrt(1.0 - 0.972 * c2);              // van Rhijn: R / (R + 90 km) squared
    float ext = exp(-0.25 * min(airmass(y), 20.0));       // ~0.25 mag per airmass
    vec3 glow = vec3(0.55, 1.0, 0.5) * 6e-5 * vr * F.night0.z;
    vec3 stars = vec3(0.85, 0.9, 1.0) * 8e-5 * F.night0.x;
    vec3 c = (glow + stars) * ext + vec3(0.6, 0.7, 1.0) * 1.5e-5 * F.night0.x;  // + scattered starlight near the horizon
    return c * F.night0.w * (d.y < 0.0 ? 0.3 : 1.0);
}

float nHash(vec2 c, float s) { return float(hashu(uint(int(c.x) + 7919 * int(s)) * 2654435761u ^ hashu(uint(int(c.y)) * 40503u + uint(s)))) / 4294967295.0; }

// Stars: three layers of cells on the celestial sphere (bright rare ones coarse, faint common ones fine), one star
// per occupied cell; drawn as a ~1 px spot carrying the star's whole flux. Colour from a hashed temperature,
// dimmed and twinkling more toward the horizon (more air).
vec3 starField(vec3 d, float pix) {
    vec2 rc = celestial(d);
    vec3 sum = vec3(0);
    float y = max(d.y, 0.0);
    float am = min(airmass(y), 30.0);
    float twinkleK = F.night1.z * clamp((am - 1.0) * 0.15, 0.05, 0.8);
    for (int layer = 0; layer < 3; layer++) {
        float res = layer == 0 ? 40.0 : (layer == 1 ? 120.0 : 360.0);
        // cells of roughly equal area: rows in sin(dec), columns in angle
        vec2 g = vec2(rc.x / (2.0 * PI) * res * 2.0, (sin(rc.y) * 0.5 + 0.5) * res);
        vec2 cell = floor(g);
        for (int k = 0; k < 4; k++) {
            vec2 c = cell + vec2(k & 1, k >> 1) - 0.5 + step(0.5, fract(g)) - 0.5;
            float h = nHash(c, float(layer) * 13.0 + 1.0);
            if (h > 0.5) continue;
            vec2 sp = c + vec2(nHash(c, float(layer) * 13.0 + 2.0), nHash(c, float(layer) * 13.0 + 3.0));
            float ra = sp.x / (res * 2.0) * 2.0 * PI + F.night1.x, sd = clamp(sp.y / res * 2.0 - 1.0, -1.0, 1.0);
            float dec = asin(sd);
            vec3 P = celestialPole(), e1 = normalize(vec3(0.0, cos(F.night1.y), sin(F.night1.y))), e2 = cross(P, e1);
            vec3 sdir = cos(dec) * (cos(ra) * e1 + sin(ra) * e2) + sin(dec) * P;
            float ang = acos(clamp(dot(sdir, d), -1.0, 1.0));
            if (ang > 2.5 * pix) continue;
            // magnitude: brighter layers hold the bright stars; illuminance 2.5e-6 lux at magnitude 0
            float mag = (layer == 0 ? 0.5 : (layer == 1 ? 3.0 : 5.0)) + 2.5 * nHash(c, float(layer) * 13.0 + 4.0);
            float E = 2.5e-6 * pow(10.0, -0.4 * mag);
            float t = nHash(c, float(layer) * 13.0 + 5.0);
            vec3 col = t < 0.15 ? vec3(1.0, 0.72, 0.5) : (t < 0.45 ? vec3(1.0, 0.9, 0.78) : (t < 0.85 ? vec3(0.95, 0.97, 1.0) : vec3(0.75, 0.85, 1.0)));
            float tw = 1.0 + twinkleK * sin(F.camPos.w * (9.0 + 13.0 * t) + h * 40.0) * sin(F.camPos.w * 23.0 * (0.5 + h));
            float spot = exp(-ang * ang / (0.5 * pix * pix)) / (PI * 0.5 * pix * pix);  // normalised: flux / solid angle
            sum += col * E * spot * tw;
        }
    }
    return sum * exp(-0.25 * am) * F.night0.x;
}

// Milky Way: a band along the galactic equator, brightest toward the core (Sagittarius), with dark dust lanes.
vec3 milkyWay(vec3 d) {
    vec2 rc = celestial(d);
    vec3 P = celestialPole(), e1 = normalize(vec3(0.0, cos(F.night1.y), sin(F.night1.y))), e2 = cross(P, e1);
    float ra = rc.x, dec = rc.y;
    vec3 cdir = cos(dec) * (cos(ra) * e1 + sin(ra) * e2) + sin(dec) * P;
    // galactic frame: the plane tilted 62.9 deg to the equator
    vec3 gp = normalize(0.46 * P + 0.89 * e1 - 0.0 * e2);   // north galactic pole
    vec3 gc = normalize(cross(gp, e2));                      // toward the core
    float b = asin(clamp(dot(cdir, gp), -1.0, 1.0));
    float l = atan(dot(cdir, cross(gp, gc)), dot(cdir, gc));
    float core = exp(-l * l / 0.35) * exp(-b * b / 0.03);
    float band = exp(-b * b / 0.012) * (0.5 + 0.5 * cos(l * 0.5));
    vec2 q = vec2(l * 6.0, b * 30.0);
    float clump = vnoise(q * 1.7) * 0.6 + vnoise(q * 5.1 + 3.0) * 0.4;
    float dust = smoothstep(0.45, 0.7, vnoise(vec2(l * 9.0, b * 70.0)) * 0.7 + vnoise(vec2(l * 23.0, b * 160.0)) * 0.3) * exp(-b * b / 0.004);
    float L = (band * (0.6 + 0.8 * clump) + core * 2.5) * (1.0 - 0.75 * dust);
    float y = max(d.y, 0.0);
    return vec3(1.0, 0.95, 0.85) * 1.5e-4 * L * exp(-0.25 * min(airmass(y), 30.0)) * F.night0.y;
}

// The moon disc as seen through the atmosphere: lit by the sun on the side facing it (phase), maria darker,
// limb a little darker (Lommel-Seeliger regolith), earthshine on the night side. Returns radiance (scene units),
// 0 off the disc. T3: cloud transmittance cubed, as the sun disc.
vec3 moonDisc(vec3 d) {
    vec3 m = F.moon0.xyz;
    float c = dot(d, m);
    if (c < F.moon0.w || m.y < -0.02) return vec3(0);
    float sinR = sqrt(max(1.0 - F.moon0.w * F.moon0.w, 1e-12));
    vec3 o = (d - m * c) / sinR;   // disc coordinates, |o| < 1
    float r2 = dot(o, o);
    if (r2 >= 1.0) return vec3(0);
    vec3 N = o - m * sqrt(1.0 - r2);  // the moon's surface normal (facing us = -m)
    float mu0 = max(dot(N, F.sun2.xyz), 0.0), mu = max(-dot(N, m), 1e-3);
    // maria: large dark lunar seas fixed on the disc (tidally locked: same face always)
    vec3 t1 = normalize(cross(m, vec3(0, 1, 0))), t2 = cross(t1, m);
    vec2 uv = vec2(dot(o, t1), dot(o, t2));
    float maria = smoothstep(0.45, 0.65, vnoise(uv * 2.2 + 5.0) * 0.7 + vnoise(uv * 6.0 + 1.0) * 0.3);
    float craters = vnoise(uv * 25.0) * 0.15;
    float alb = 0.12 * (1.0 - 0.45 * maria) * (0.92 + craters);
    vec3 sunI = vec3(F.sun2.w) * vec3(1.0, 0.96, 0.9);
    vec3 L = sunI * alb / PI * 2.0 * mu0 / (mu0 + mu);      // Lommel-Seeliger
    L += sunI * alb / PI * 2e-4 * (1.0 - F.moon1.y);       // earthshine on the unlit part
    vec3 Tm = transmittanceToSun(vec3(0.0, RG + VIEW_H, 0.0), m);
    return L * Tm * F.moon1.w;                               // x night light boost, like the moonlight
}
#endif
