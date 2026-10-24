// What a ray that leaves the scene sees: sky, clouds, stars, sun and moon discs, lightning.
#ifndef SKY_SKY_WITH_SUN
#define SKY_SKY_WITH_SUN
#include "core/frame.glsl"
#include "sky/sky.glsl"
#include "sky/clouds.glsl"

// Clear sky + sun disc, then the cloud layer over both (clouds hide the sun).
vec3 skyWithSun(vec3 d) {
    float jit = float(hashu(floatBitsToUint(d.x * 7.0 + d.z) ^ hashu(floatBitsToUint(d.y)) ^ (F.res.z * 0x9e3779b9u))) / 4294967295.0;
    float T;
    vec3 cl = cloudMarch(F.camPos.xyz, d, jit, T);
    vec3 c = sky(d) * T + cl;
    // stars and the Milky Way behind everything (fade out under the day sky by themselves)
    if (d.y > 0.0 && F.night0.w > 0.0) c += (starField(d, 2.0 * F.camUp.w / float(F.res.y)) + milkyWay(d)) * F.night0.w * T;
    // sun disc: ~500x the sky, so even thin cloud must kill it; scale by T^3 (optical depth x3) so a cloud edge
    // reads as a bright glow rather than the bare disc on top of the cloud. At night the key light is the moon.
    if (F.night1.w < 0.5 && dot(d, F.sunDir.xyz) > F.sunDir.w)
        c += min(F.sunColor.rgb / (2.0 * PI * (1.0 - F.sunDir.w)), vec3(500.0)) * T * T * T;
    c += min(moonDisc(d), vec3(500.0)) * T * T * T;
    c += lightningBolt(d, 1e9);
    return c;
}
#endif
