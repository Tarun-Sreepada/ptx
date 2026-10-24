// G-buffer surface load / store and the demodulation albedo.
#ifndef CORE_GBUFFER
#define CORE_GBUFFER
#include "core/frame.glsl"
#include "core/bindings.glsl"
#include "core/camera.glsl"
#include "materials/brdf.glsl"

// ---------------------------------------------------------------- G-buffer surface
Surf loadSurf(ivec2 p) {
    vec4 N4 = imageLoad(gNorm, p), a = imageLoad(gAlbedo, p), m = imageLoad(gMat, p), m2 = imageLoad(gMat2, p);
    Surf s;
    s.base = a.rgb; s.ao = a.a;
    s.n = N4.xyz; s.mat = uint(N4.w + 0.5);
    vec4 t = imageLoad(gTan, p);
    s.T = t.xyz;
    s.ior = t.w >= 1.0 ? t.w : 1.5;
    s.transl = t.w < 0.0 ? -t.w : (s.mat == MAT_GRASS ? 0.6 : 0.0);
    s.rough = m.x; s.metal = m.y; s.spec = m.z; s.trans = m.w;
    s.cc = m2.x; s.ccRough = m2.y; s.aniso = m2.w;
    s.sheen = max(m2.z, 0.0); s.sss = max(-m2.z, 0.0);  // gMat2.z: sheen, or minus the subsurface profile
    return s;
}
void storeSurf(ivec2 p, Surf s) {
    imageStore(gAlbedo, p, vec4(s.base, s.ao));
    imageStore(gMat, p, vec4(s.rough, s.metal, s.spec, s.trans));
    imageStore(gMat2, p, vec4(s.cc, s.ccRough, s.sss > 0.0 ? -s.sss : s.sheen, s.aniso));
    imageStore(gTan, p, vec4(s.T, packIorTransl(s)));
}
// Demodulation albedo of pixel p (0 for sky): what lighting images are divided by and the composite multiplies by.
vec3 pixelAlbedo(ivec2 p) {
    if (imageLoad(gDepth, p).r < 0.0) return vec3(0);
    Surf s = loadSurf(p);
    return surfAlbedo(s, max(dot(s.n, -cameraRay(p)), 1e-4));
}
#endif
