// Rainbow. Sunlight refracted into a raindrop, reflected once off its back and refracted out again leaves at
// ~42 degrees from the antisolar point (the primary bow, red outside); after two reflections at ~51 degrees (the
// secondary, ~40% as bright, colours reversed). Light from the drops at smaller angles brightens the sky inside the
// primary bow; between the bows less comes back (Alexander's dark band). Big drops give narrow, vivid bands; small
// ones broad, pale ones (a fogbow is almost white). It only shows over the rain between the eye and what's behind
// it: up to the rain shaft distance F.rainbow.z. F.rainbow: x strength, y drop radius (mm), z distance (m), w on.
#ifndef SKY_RAINBOW
#define SKY_RAINBOW
#include "core/frame.glsl"

const float BOW1_VIOLET = 40.6, BOW1_RED = 42.3;  // degrees from the antisolar point
const float BOW2_RED = 50.4, BOW2_VIOLET = 53.6;
const float BOW_BRIGHTNESS = 0.03;                 // of the sunlight, primary bow at full strength
const float BOW2_SHARE = 0.43;
const float BOW_INSIDE = 0.012;                    // extra sky brightness just inside the primary

// colour across a band, t = 0 violet .. 1 red (sums to roughly white over the band)
vec3 bowSpectrum(float t) {
    vec3 c = clamp(vec3(1.6 * t - 0.5, 1.0 - 2.4 * abs(t - 0.55), 1.0 - 2.0 * t), 0.0, 1.0);
    c.r += 0.35 * max(0.25 - t, 0.0) * 4.0;  // violet is blue with a little red
    return c * c;
}

// width (degrees) the sun's disc and the drop size smear each band over
float bowBlur(float rMM) { return 0.5 + 0.35 / max(rMM, 0.03); }

vec3 bowBand(float th, float a, float b, float blur, float sat) {
    float lo = min(a, b), hi = max(a, b);
    float inside = smoothstep(lo - blur, lo + blur, th) * (1.0 - smoothstep(hi - blur, hi + blur, th));
    if (inside <= 0.0) return vec3(0);
    float t = clamp((th - a) / (b - a), 0.0, 1.0);  // 0 at a (violet), 1 at b (red)
    return mix(vec3(0.33), bowSpectrum(t), sat) * inside;
}

// light the rain adds along view direction d in front of a surface depth metres away (1e9 = sky)
vec3 rainbow(vec3 d, float depth) {
    if (F.rainbow.w < 0.5 || F.rainbow.x <= 0.0) return vec3(0);
    vec3 sunL = F.sunColor.rgb;
    if (sunL.r + sunL.g + sunL.b <= 0.0) return vec3(0);
    float th = degrees(acos(clamp(dot(d, -F.sunDir.xyz), -1.0, 1.0)));
    if (th > BOW2_VIOLET + 4.0) return vec3(0);
    float rMM = F.rainbow.y, blur = bowBlur(rMM), sat = smoothstep(0.03, 0.4, rMM);
    vec3 c = bowBand(th, BOW1_VIOLET, BOW1_RED, blur, sat) + BOW2_SHARE * bowBand(th, BOW2_VIOLET, BOW2_RED, blur * 1.3, sat);
    c += vec3(BOW_INSIDE / BOW_BRIGHTNESS) * smoothstep(BOW1_VIOLET - 12.0, BOW1_VIOLET, th) * (1.0 - step(BOW1_VIOLET, th));
    float rainPath = clamp(depth / F.rainbow.z, 0.0, 1.0);  // how much rain is in front of the surface
    // sunlit rain: the clouds' shadow halfway along it
    vec3 mid = F.camPos.xyz + d * min(depth, F.rainbow.z) * 0.5;
    return c * sunL * (BOW_BRIGHTNESS * F.rainbow.x * rainPath * cloudShadow(mid));
}
#endif
