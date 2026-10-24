// Lightning (main.cpp updateLightning): a branching channel from the cloud base to the ground, drawn as a glowing
// line where the view passes it, and the flash it throws over the sky and the clouds around it (the scene light
// of the flash itself is a point light in the light list, so it casts shadows).
#ifndef LIGHTNING_GLSL
#define LIGHTNING_GLSL

// Sky / cloud glow of the flash: strongest around the flash centre, a dimmer sheet over the whole sky.
vec3 lightningSky(vec3 d) {
    if (F.boltInfo.y <= 0.0) return vec3(0);
    vec3 to = F.boltPos.xyz - F.camPos.xyz;
    float c = dot(d, normalize(to));
    float k = exp((c - 1.0) / 0.12) + 0.12 * max(d.y + 0.2, 0.0);
    return vec3(0.75, 0.8, 1.0) * F.boltInfo.z * F.boltInfo.y * k;
}

// The channel itself along the view ray d (from the camera), up to distance tMax (scene depth).
vec3 lightningBolt(vec3 d, float tMax) {
    uint n = uint(F.boltInfo.x);
    if (n == 0u || F.boltInfo.y <= 0.0) return vec3(0);
    vec3 o = F.camPos.xyz;
    float pix = 2.0 * F.camUp.w / float(F.res.y);
    float glow = 0.0;
    for (uint i = 0u; i < n; i++) {
        vec4 A = F.bolt[2u * i], B = F.bolt[2u * i + 1u];
        vec3 a = A.xyz - o, ab = B.xyz - A.xyz;
        // closest approach between the ray (t d) and the segment (a + s ab)
        float dd = dot(ab, ab), db = dot(d, ab), da = dot(d, a), ad = dot(a, ab);
        float den = dd - db * db;
        float s = den > 1e-6 ? clamp((db * da - ad) / den, 0.0, 1.0) : 0.0;
        vec3 q = a + ab * s;
        float t = dot(q, d);
        if (t <= 0.0 || t > tMax) continue;
        float ang = length(q - d * t) / t;
        float w = max(A.w / t, 0.7 * pix);  // the channel is a few cm thick: at least ~1 px
        float core = exp(-ang * ang / (w * w)) * (A.w / t) / w;  // thinner than a pixel: same flux, spread
        float halo = exp(-ang / (25.0 * pix)) * 0.02;
        glow = max(glow, (core + halo) * B.w);
    }
    return vec3(0.8, 0.85, 1.0) * glow * F.boltInfo.y * 400.0;
}
#endif
