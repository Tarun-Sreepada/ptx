// Ground virtual texture: a clipmap of VT_LEVELS layers around the camera, each 1024^2 texels, texel size
// F.vtP.x * 2^level (0.49 mm .. 4 m, the coarsest reaching ~2 km). Tiles are generated on demand by vtgen.comp (main.cpp updateGroundVT) and
// stored toroidally (world texel i lives at i mod 1024), so the REPEAT sampler addresses them directly.
// F.vtRect[l] is the world xz rectangle whose texels are all resident at level l (shrunk a texel for bilinear).
#ifndef GROUNDVT_GLSL
#define GROUNDVT_GLSL
layout(binding = 99) uniform sampler2DArray vtTexA;   // rgb sqrt(rel / 2), a stone coverage
layout(binding = 100) uniform sampler2DArray vtTexB;  // slope x, z (s / (1 + |s|)), roughness, AO
layout(binding = 101) uniform sampler2DArray vtTexH;  // height: 0.5 + h * 2 (m)

struct VtSample { vec3 rel; float stone; vec2 slope; float rough, ao, h, w; bool ok; };  // w: coverage 0..1

bool vtIn(int l, vec2 p) { vec4 r = F.vtRect[l]; return all(greaterThan(p, r.xy)) && all(lessThan(p, r.zw)); }
vec2 vtSlope(vec2 e) { e = clamp(e * 2.0 - 1.0, -0.97, 0.97); return e / (1.0 - abs(e)); }
void vtAdd(inout VtSample s, int l, vec2 p, float w) {
    vec3 uv = vec3(p / (F.vtP.x * exp2(float(l)) * F.vtP.y), float(l));
    vec4 a = textureLod(vtTexA, uv, 0.0), b = textureLod(vtTexB, uv, 0.0);
    s.rel += a.rgb * a.rgb * 2.0 * w;
    s.stone += a.a * w;
    s.slope += vtSlope(b.xy) * w;
    s.rough += b.z * w;
    s.ao += b.w * w;
    s.h += (textureLod(vtTexH, uv, 0.0).r - 0.5) * 0.5 * w;
}
// pix: footprint (m) of the shaded pixel. Picks the level whose texel matches it and blends the next level up by
// the fractional part, like trilinear mip filtering. Near the edge of a level's resident rectangle (where it ends,
// or where newly generated tiles join it) its weight fades into the next coarser level over two tiles, so there is
// no seam and nothing pops; past the coarsest level's reach the detail fades out (s.w < 1).
VtSample vtFetch(vec2 p, float pix) {
    VtSample s;
    s.rel = vec3(0); s.stone = 0.0; s.slope = vec2(0); s.rough = 0.0; s.ao = 0.0; s.h = 0.0; s.ok = false; s.w = 0.0;
    if (F.vtP.w < 0.5) return s;
    int L = int(F.vtP.z);
    float lv = max(log2(max(pix, 1e-7) / F.vtP.x), 0.0);
    int l0 = int(lv);
    float f = fract(lv), carry = 1.0 - f, next = f;
    int taken = 0;
    for (int l = l0; l < L && taken < 3 && carry + next > 0.001; l++) {
        float want = carry;
        carry = next;
        next = 0.0;
        if (want < 0.001) continue;
        vec4 r = F.vtRect[l];
        float e = min(min(p.x - r.x, r.z - p.x), min(p.y - r.y, r.w - p.y));
        float we = smoothstep(0.0, 128.0 * F.vtP.x * exp2(float(l)), e);
        float take = want * we;
        if (take > 0.001) { vtAdd(s, l, p, take); s.w += take; taken++; }
        carry += want - take;
    }
    if (s.w < 0.001) return s;
    float inv = 1.0 / s.w;
    s.rel *= inv; s.stone *= inv; s.slope *= inv; s.rough *= inv; s.ao *= inv; s.h *= inv;
    s.ok = true;
    return s;
}
float vtHeight(vec2 p, int l) { return (textureLod(vtTexH, vec3(p / (F.vtP.x * exp2(float(l)) * F.vtP.y), float(l)), 0.0).r - 0.5) * 0.5; }
int vtLevelFor(vec2 p, float pix) {
    int L = int(F.vtP.z), l = int(max(log2(max(pix, 1e-7) / F.vtP.x), 0.0));
    while (l < L && !vtIn(l, p)) l++;
    return l;
}
#endif
