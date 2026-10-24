// Night vision (Purkinje shift). Below a few cd/m^2 the rods take over from the cones: colour fades, and what's
// left shifts toward blue (rods peak near 507 nm, so reds go dark first and blues stay brightest). The pixel's
// luminance in cd/m^2 (F.purk.y per scene unit, without the night boost) picks the blend: cone vision above
// PURKINJE_PHOTOPIC, rod vision below PURKINJE_SCOTOPIC, mesopic in between (blended in log luminance).
#ifndef POST_PURKINJE
#define POST_PURKINJE
#include "core/frame.glsl"

const float PURKINJE_PHOTOPIC = 3.0, PURKINJE_SCOTOPIC = 0.003;  // cd/m^2
const vec3 ROD_TINT = vec3(0.70, 0.86, 1.25);  // how rod vision looks on a display: bluish grey
const float ROD_GAIN = 0.39;                   // scotopic / photopic luminance of daylight white: keeps white as bright

vec3 purkinje(vec3 c) {
    if (F.purk.x < 0.5) return c;
    // CIE XYZ of the linear Rec.709 colour, scotopic luminance after Larson, Rushmeier & Piatko 1997
    float X = dot(c, vec3(0.4124, 0.3576, 0.1805)), Y = dot(c, vec3(0.2126, 0.7152, 0.0722)), Z = dot(c, vec3(0.0193, 0.1192, 0.9505));
    if (Y <= 0.0 || X <= 1e-9) return c;
    float Ys = max(Y * (1.33 * (1.0 + (Y + Z) / X) - 1.68), 0.0);
    float L = Y * F.purk.y;  // cd/m^2
    float k = clamp((log2(max(L, 1e-9)) - log2(PURKINJE_SCOTOPIC)) / (log2(PURKINJE_PHOTOPIC) - log2(PURKINJE_SCOTOPIC)), 0.0, 1.0);
    k = k * k * (3.0 - 2.0 * k);
    vec3 rod = Ys * ROD_GAIN * ROD_TINT / dot(ROD_TINT, vec3(0.2126, 0.7152, 0.0722));
    return mix(rod, c, k);
}
#endif
