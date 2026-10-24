// Octahedral tree impostors, shared by the bake (impostor_bake.comp) and the raster pass (raster_impostor.*).
// Each tree variant is one layer of a 1024^2 atlas: IMP_G x IMP_G frames of IMP_RES^2 px, one per view
// direction on the upper hemisphere (hemi-octahedral layout, frame (i, j) = direction impDir(i, j)).
// A frame is an orthographic capture of the variant's bounding sphere (centre C, radius R) looking along -dir;
// texels store albedo (sqrt-encoded, a = coverage) and the local-space normal + depth from the frame plane.
#ifndef IMPOSTOR_GLSL
#define IMPOSTOR_GLSL
#define IMP_G 8
#define IMP_RES 128
#define IMP_ATLAS (IMP_G * IMP_RES)

// Per instance (3 vec4): (pos.xyz, scale), (yaw, variant, fade, -), (C.xyz, R)
layout(binding = 44, std430) readonly buffer ImpInst { vec4 impInst[]; };

vec2 hemiOctEncode(vec3 d) {
    d.y = max(d.y, 0.0);
    d /= abs(d.x) + abs(d.y) + abs(d.z);
    return vec2(d.x + d.z, d.z - d.x);
}
vec3 hemiOctDecode(vec2 e) {
    float x = (e.x - e.y) * 0.5, z = (e.x + e.y) * 0.5;
    return normalize(vec3(x, 1.0 - abs(x) - abs(z), z));
}
vec3 impDir(ivec2 f) { return hemiOctDecode(vec2(f) / float(IMP_G - 1) * 2.0 - 1.0); }
void impBasis(vec3 d, out vec3 r, out vec3 u) {
    vec3 up0 = abs(d.y) > 0.99 ? vec3(0, 0, 1) : vec3(0, 1, 0);
    r = normalize(cross(up0, d));
    u = cross(d, r);
}
// tree yaw: local -> world rotation (matches the CPU instance transform {{c,0,s},{0,1,0},{-s,0,c}})
vec3 rotY(vec3 v, float c, float s)    { return vec3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z); }
vec3 invRotY(vec3 v, float c, float s) { return vec3(c * v.x - s * v.z, v.y, s * v.x + c * v.z); }

// Per-pixel threshold for the mesh <-> impostor crossfade; both sides use the same value so they tile exactly.
float fadeNoise(vec2 px, uint frame) {
    px += float(frame % 64u) * 5.588238;
    return fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715))));
}
#endif
