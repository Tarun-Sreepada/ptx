#version 460
#extension GL_GOOGLE_include_directive : require
#include "frame.glsl"
#include "ground.glsl"
#include "brdf.glsl"

// Raster G-buffer: writes exactly what primary.comp would write for a traced camera ray.
layout(location = 0) in vec3 oP;
layout(location = 1) in vec3 oN;
layout(location = 2) flat in uint oMat;
layout(location = 3) in float oT;
layout(location = 4) flat in uint oBlade;
layout(location = 5) in vec3 oPrev;

layout(location = 0) out float outDepth;
layout(location = 1) out vec4 outNorm;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outEmis;
layout(location = 4) out vec4 outMat;
layout(location = 5) out vec4 outMat2;
layout(location = 6) out vec4 outTan;
layout(location = 7) out vec4 outVel;

void main() {
    vec3 n = oMat == 1u ? normalize(cross(dFdx(oP), dFdy(oP))) : normalize(oN);
    vec3 Pw = oP;
    vec3 v = Pw - F.camPos.xyz;
    if (oMat == 0u && craterRefine(Pw, normalize(v), n)) v = Pw - F.camPos.xyz;  // exact crater bowl
    if (dot(n, v) > 0.0) n = -n;
    outDepth = length(v);
    float rough = 1.0, ao = 1.0;
    vec3 alb;
    if (oMat == 0u) {
        alb = terrainAlbedo(Pw.xz, smoothstep(F.grassA.w * 0.4, F.grassA.w, length(Pw.xz - F.camPos.xz)));
        vec3 P = Pw;
        groundSurface(P, v / outDepth, outDepth, true, n, alb, rough, ao);
    } else if (oMat == 1u) {
        alb = grassAlbedo(oBlade, oT);
        rough = grassRough(oBlade);
    } else {
        alb = rockSurface(Pw, n, rough, ao);
        if (dot(n, v) > 0.0) n = normalize(n - v / outDepth * (dot(n, v / outDepth) + 0.01));
    }
    outNorm = vec4(n, float(oMat));
    outAlbedo = vec4(alb, ao);
    outEmis = vec4(0.0);
    Surf sf = procSurf(alb, n, oMat);
    sf.rough = rough;
    if (oMat == 0u) sf.spec = 0.5;  // soil / stone: dielectric, F0 ~0.02
    outMat = vec4(sf.rough, sf.metal, sf.spec, sf.trans);
    outMat2 = vec4(0.0);
    outTan = vec4(sf.T, packIorTransl(sf));
    outVel = vec4(oPrev - oP, 0.0);
}
