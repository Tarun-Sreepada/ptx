// Streamed glTF models: bindless meshes (buffer device addresses), material table and texture array.
#ifndef MODEL_GLSL
#define MODEL_GLSL
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_nonuniform_qualifier : require

#define MAT_MODEL 3u        // instance custom index >= MAT_MODEL: mesh slot = index - MAT_MODEL
#define MAX_TEXTURES 4096
#define NO_TEX 0xFFFFFFFFu

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer FloatBuf { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer UintBuf { uint v[]; };

struct MeshInfo { uvec2 vtx; uvec2 idx; uvec2 mat; uvec2 pad; };  // vtx: pos3 nrm3 uv2, mat: material slot per triangle
// glTF metal-rough + KHR clearcoat / sheen / anisotropy / transmission / ior / specular (main.cpp GpuMat).
// nrmTex: tangent-space normal (rgb) + height (a, when heightScale > 0); ormTex: g roughness, b metalness; occTex: r.
struct Material {
    vec4 color;
    vec3 emissive; uint tex;
    float rough, metal, normalScale; uint nrmTex;
    uint ormTex, emisTex, occTex; float occStrength;
    float heightScale, clearcoat, ccRough, sheen;
    vec3 sheenColor; float aniso;
    float anisoRot, transmission, ior, specular;
    float transl, pad0, pad1, pad2;  // diffuse transmission
};

layout(binding = 32) uniform sampler2D textures[MAX_TEXTURES];
layout(binding = 33, std430) readonly buffer Meshes { MeshInfo meshes[]; };
layout(binding = 34, std430) readonly buffer Materials { Material materials[]; };
layout(binding = 35, std430) readonly buffer ModelXf { vec4 modelXf[]; };  // raster instances: 3 rows + (mesh slot, -, -, -)

struct ModelVertex { vec3 p; vec3 n; vec2 uv; };
ModelVertex modelVertex(MeshInfo m, uint i) {
    FloatBuf b = FloatBuf(m.vtx);
    uint o = i * 8u;
    return ModelVertex(vec3(b.v[o], b.v[o + 1u], b.v[o + 2u]), vec3(b.v[o + 3u], b.v[o + 4u], b.v[o + 5u]), vec2(b.v[o + 6u], b.v[o + 7u]));
}
uint triMaterialSlot(MeshInfo m, uint prim) { return UintBuf(m.mat).v[prim]; }
Material triMaterial(MeshInfo m, uint prim) { return materials[triMaterialSlot(m, prim)]; }

// Texture filtering. Raster: the fragment's own uv gradients (set before sampling, so a parallax-shifted uv
// keeps the original footprint). Ray hits: ray cone (Akenine-Moller et al., RTG ch. 20): gConeLod is the
// footprint in uv units, log2; each texture adds half the log2 of its texel count.
#ifdef IMPLICIT_LOD
vec2 gUvDx, gUvDy;
vec4 mtex(uint i, vec2 uv) { return textureGrad(textures[nonuniformEXT(i)], uv, gUvDx, gUvDy); }
#else
float gConeLod = -32.0;
vec4 mtex(uint i, vec2 uv) {
    vec2 sz = vec2(textureSize(textures[nonuniformEXT(i)], 0));
    return textureLod(textures[nonuniformEXT(i)], uv, max(gConeLod + 0.5 * log2(sz.x * sz.y), 0.0));
}
#endif

// Everything the shading needs at a point of a model surface. T: unit tangent (anisotropy direction).
struct SurfaceMat {
    vec3 base, n, T, emis;
    float rough, metal, spec, trans, cc, ccRough, sheen, aniso, ao, alpha, ior, transl;
};

// n: interpolated (geometric side already chosen) normal; dpdu, dpdv: position derivatives along the uv axes.
SurfaceMat evalMaterial(Material mt, vec2 uv, vec3 n, vec3 dpdu, vec3 dpdv) {
    SurfaceMat s;
    vec4 bc = mt.color;
    if (mt.tex != NO_TEX) bc *= mtex(mt.tex, uv);
    s.base = bc.rgb;
    s.alpha = bc.a;
    s.rough = mt.rough;
    s.metal = mt.metal;
    if (mt.ormTex != NO_TEX) { vec4 o = mtex(mt.ormTex, uv); s.rough *= o.g; s.metal *= o.b; }
    s.ao = 1.0;
    if (mt.occTex != NO_TEX) s.ao = 1.0 + mt.occStrength * (mtex(mt.occTex, uv).r - 1.0);
    // tangent frame from the uv parametrisation (glTF convention: bitangent toward -v)
    vec3 T = dpdu - n * dot(n, dpdu);
    if (dot(T, T) < 1e-12) { vec3 a = abs(n.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0); T = cross(a, n); }
    T = normalize(T);
    vec3 B = cross(n, T) * (dot(cross(n, T), -dpdv) < 0.0 ? -1.0 : 1.0);
    s.n = n;
    if (mt.nrmTex != NO_TEX) {
        vec3 tn = mtex(mt.nrmTex, uv).xyz * 2.0 - 1.0;
        tn.xy *= mt.normalScale;
        vec3 pn = normalize(T * tn.x + B * tn.y + n * max(tn.z, 1e-3));
        if (all(equal(pn, pn))) s.n = pn;
    }
    float ca = cos(mt.anisoRot), sa = sin(mt.anisoRot);
    vec3 Ta = T * ca + B * sa;
    s.T = normalize(Ta - s.n * dot(s.n, Ta));
    s.emis = mt.emissive;
    if (mt.emisTex != NO_TEX) s.emis *= mtex(mt.emisTex, uv).rgb;
    // dielectric F0 = ((ior - 1) / (ior + 1))^2, in units of the default 0.04, times the specular factor
    float f0 = (mt.ior - 1.0) / (mt.ior + 1.0);
    s.spec = mt.specular * f0 * f0 / 0.04;
    s.trans = mt.transmission;
    s.ior = mt.ior;
    s.transl = mt.transl;
    s.cc = mt.clearcoat;
    s.ccRough = mt.ccRough;
    s.sheen = mt.sheen;
    s.aniso = mt.aniso;
    return s;
}

vec3 materialAlbedo(Material mt, vec2 uv) {
    vec3 c = mt.color.rgb;
    if (mt.tex != NO_TEX) c *= mtex(mt.tex, uv).rgb;
    return c;
}

// Position derivatives along u and v for a triangle (object space; the caller transforms them).
void triDerivs(vec3 p0, vec3 p1, vec3 p2, vec2 t0, vec2 t1, vec2 t2, out vec3 dpdu, out vec3 dpdv) {
    vec3 e1 = p1 - p0, e2 = p2 - p0;
    vec2 d1 = t1 - t0, d2 = t2 - t0;
    float det = d1.x * d2.y - d1.y * d2.x;
    if (abs(det) < 1e-12) { dpdu = vec3(0); dpdv = vec3(0); return; }
    float r = 1.0 / det;
    dpdu = (e1 * d2.y - e2 * d1.y) * r;
    dpdv = (e2 * d1.x - e1 * d2.x) * r;
}
#endif
