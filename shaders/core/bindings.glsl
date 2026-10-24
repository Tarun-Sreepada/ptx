// Descriptor bindings every compute pass shares (one set layout, gpu/pipelines.cpp).
#ifndef CORE_BINDINGS
#define CORE_BINDINGS
#include "core/frame.glsl"

layout(binding = 0) uniform accelerationStructureEXT tlas;
layout(binding = 2, std430) buffer TerrainV { float tv[]; };            // pos.xyz, normal.xyz (written by terrain.comp)
layout(binding = 3, std430) buffer TerrainI { uint ti[]; };  // written by terrain.comp (clipmap cells, degenerate under finer levels)
layout(binding = 4, std430) buffer GrassV { float gv[]; };              // pos.xyz, 7 verts per blade
layout(binding = 5, std430) readonly buffer GrassI { uint gix[]; };
layout(binding = 6, std430) buffer BladeSeed { uint bladeSeed[]; };     // stable per-blade id (for colour)
layout(binding = 7, std430) readonly buffer Lights { vec4 lights[]; };  // pos+radius, intensity rgb
layout(binding = 8, std430) buffer ResCur  { vec4 resCur[]; };
layout(binding = 9, std430) buffer ResPrev { vec4 resPrev[]; };
layout(binding = 25, std430) buffer GICur  { vec4 giCur[]; };  // 3 per pixel: ReSTIR GI reservoirs
layout(binding = 26, std430) buffer GIPrev { vec4 giPrev[]; };
layout(binding = 27, std430) buffer GrassCounter {
    uint grassCount;                      // blades emitted this frame
    uint drawIndexCount, drawInstanceCount, drawFirstIndex; // VkDrawIndexedIndirectCommand at offset 4
    int drawVertexOffset;
    uint drawFirstInstance;
};

layout(binding = 28, rgba32f) uniform image2D skyLUT;       // atmosphere radiance (sun + moon light scattered), scene units
layout(binding = 29, std430) readonly buffer RockV { float rv[]; }; // pos.xyz, normal.xyz
layout(binding = 30, std430) readonly buffer RockI { uint ri[]; };
layout(binding = 10, r32f) uniform image2D gDepth;        // hit distance along the camera ray (-1 = sky)
layout(binding = 11, rgba16f) uniform image2D gNorm;      // normal, w = material
layout(binding = 12, rgba16f) uniform image2D gAlbedo;
layout(binding = 13, rgba16f) uniform image2D gEmis;       // directly visible sky / light orbs
layout(binding = 14, r32f) uniform image2D prevDepth;
layout(binding = 15, rgba16f) uniform image2D prevNorm;
layout(binding = 16, rgba16f) uniform image2D directImg;   // demodulated direct light
layout(binding = 17, rgba16f) uniform image2D indirectImg; // demodulated indirect light
layout(binding = 18, rgba16f) uniform image2D accumColor;  // rgb, a = variance
layout(binding = 19, rgba32f) uniform image2D accumMoments;// m1, m2, history length
layout(binding = 20, rgba16f) uniform image2D histColor;
layout(binding = 21, rgba32f) uniform image2D histMoments;
layout(binding = 22, rgba16f) uniform image2D pingA;
layout(binding = 23, rgba16f) uniform image2D pingB;
layout(binding = 24, rgba8) uniform writeonly image2D outImg;
// material G-buffer (render resolution): gAlbedo = base colour + ao, gMat = roughness, metalness, specular, transmission,
// gMat2 = clearcoat, clearcoat roughness, sheen, anisotropy, gTan = tangent (anisotropy direction)
layout(binding = 54, rgba16f) uniform image2D gMat;
layout(binding = 55, rgba16f) uniform image2D gMat2;
layout(binding = 56, rgba16f) uniform image2D gTan;
layout(binding = 57, rgba16f) uniform image2D specImg;    // demodulated glossy indirect (specular, clearcoat, transmission)
layout(binding = 58, rgba16f) uniform image2D hdrImg;     // the HDR scene at output resolution, before post (post.comp)
layout(binding = 60, rgba16f) uniform image2D fogImg;     // volumetric fog, half resolution: in-scatter, transmittance
layout(binding = 61, rgba16f) uniform image2D fogHist;
layout(binding = 62, rgba16f) uniform image2D gVel;       // G-buffer velocity: xyz = where the surface was last frame - where it is (m)

layout(push_constant) uniform PC { uint a, b, c, d; } pc;
#endif
