#version 460
#extension GL_GOOGLE_include_directive : require
#define IMPLICIT_LOD
#include "frame.glsl"
#include "model.glsl"
#include "impostor.glsl"
#include "probes.glsl"

// G-buffer for model triangles; matches what traceClosest() returns for a camera ray (same tangent frame and
// material evaluation), plus parallax occlusion mapping of materials with a height map.
layout(location = 0) in vec3 oP;
layout(location = 1) in vec3 oN;
layout(location = 2) in vec2 oUV;
layout(location = 3) flat in uint oSlot;
layout(location = 4) flat in float oFade;
layout(location = 5) flat in uint oInst;
layout(location = 6) in vec3 oPrev;

layout(location = 0) out float outDepth;
layout(location = 1) out vec4 outNorm;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outEmis;
layout(location = 4) out vec4 outMat;
layout(location = 5) out vec4 outMat2;
layout(location = 6) out vec4 outTan;
layout(location = 7) out vec4 outVel;

const float POM_DIST = 30.0;  // parallax fades out by this distance (m)

void main() {
    if (oFade > 0.0 && fadeNoise(gl_FragCoord.xy, F.res.z) < oFade) discard;  // impostor draws these pixels
    vec3 n = normalize(oN);
    if (dot(n, n) < 0.5) n = normalize(cross(dFdx(oP), dFdy(oP)));
    vec3 v = oP - F.camPos.xyz;
    if (dot(n, v) > 0.0) n = -n;
    MeshInfo mi = meshes[oSlot];
    uint prim = uint(gl_PrimitiveID);
    Material mt = triMaterial(mi, prim);

    // exact per-triangle uv derivatives, in world space (same as the ray tracer)
    UintBuf ib = UintBuf(mi.idx);
    ModelVertex a = modelVertex(mi, ib.v[3u * prim]), b = modelVertex(mi, ib.v[3u * prim + 1u]), c = modelVertex(mi, ib.v[3u * prim + 2u]);
    uint xb = oInst * 4u;
    mat3 lin = transpose(mat3(modelXf[xb].xyz, modelXf[xb + 1u].xyz, modelXf[xb + 2u].xyz));
    vec3 dpdu, dpdv;
    triDerivs(a.p, b.p, c.p, a.uv, b.uv, c.uv, dpdu, dpdv);
    dpdu = lin * dpdu;
    dpdv = lin * dpdv;
    gUvDx = dFdx(oUV);
    gUvDy = dFdy(oUV);

    // parallax occlusion: march the view ray down through the height field (height 1 = the polygon surface)
    vec2 uv = oUV;
    float dist = length(v), extra = 0.0;
    vec3 dv = v / dist;
    float cosv = -dot(dv, n);
    if (mt.heightScale > 0.0 && mt.nrmTex != NO_TEX && dist < POM_DIST && cosv > 0.05) {
        float fade = 1.0 - smoothstep(POM_DIST * 0.6, POM_DIST, dist);
        vec3 lateral = (dv + n * cosv) / max(cosv, 0.2);  // world offset per metre of depth
        float g00 = dot(dpdu, dpdu), g01 = dot(dpdu, dpdv), g11 = dot(dpdv, dpdv), det = g00 * g11 - g01 * g01;
        if (det > 1e-20) {
            vec2 r = vec2(dot(lateral, dpdu), dot(lateral, dpdv));
            vec2 uvPerDepth = vec2(g11 * r.x - g01 * r.y, g00 * r.y - g01 * r.x) / det;
            float H = mt.heightScale * fade;
            int N = int(mix(32.0, 8.0, cosv));
            float dd = H / float(N), depth = 0.0, prevGap = 0.0;
            for (int i = 0; i < N; i++) {
                vec2 q = oUV + uvPerDepth * depth;
                float gap = (1.0 - mtex(mt.nrmTex, q).a) * H - depth;  // surface depth below the polygon minus ray depth
                if (gap <= 0.0) {
                    float t = prevGap / max(prevGap - gap, 1e-6);
                    depth -= dd * (1.0 - t);
                    break;
                }
                prevGap = gap;
                depth += dd;
            }
            uv = oUV + uvPerDepth * depth;
            extra = depth / max(cosv, 0.2);
        }
    }

    SurfaceMat sm = evalMaterial(mt, uv, n, dpdu, dpdv);
    if (dot(sm.n, dv) > 0.0) sm.n = normalize(sm.n - dv * (dot(sm.n, dv) + 0.01));
    outDepth = dist + extra;
    outNorm = vec4(sm.n, float(MAT_MODEL));
    outAlbedo = vec4(sm.base, sm.ao);
    outEmis = vec4(dot(sm.emis, sm.emis) > 0.0 && emitterOff(oP) ? vec3(0) : sm.emis, 0.0);
    outMat = vec4(sm.rough, sm.metal, sm.spec, sm.trans);
    outMat2 = vec4(sm.cc, sm.ccRough, sm.sss > 0.0 ? -sm.sss : sm.sheen, sm.aniso);
    outTan = vec4(sm.T, sm.transl > 0.0 ? -sm.transl : sm.ior);
    outVel = vec4(oPrev - oP, 0.0);
}
