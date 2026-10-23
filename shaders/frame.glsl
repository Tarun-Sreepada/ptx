// Per-frame uniforms, shared by compute and raster shaders.
#ifndef FRAME_GLSL
#define FRAME_GLSL
layout(binding = 1, std140) uniform FrameUBO {
    vec4 camPos;     // xyz, w = time
    vec4 camRight;   // w = aspect
    vec4 camUp;      // w = tan(fov/2)
    vec4 camFwd;
    vec4 prevCamPos;
    vec4 prevRight;
    vec4 prevUp;
    vec4 prevFwd;
    vec4 sunDir;     // xyz toward sun, w = cos(sun angular radius)
    vec4 sunColor;   // rgb = sun irradiance, w = sky intensity
    vec4 wind;       // xy = direction (world xz), z = strength, w = speed
    vec4 params;     // x exposure, y history max, z firefly clamp, w spatial radius (px)
    vec4 params2;
    uvec4 res;       // width, height, frame index, light count
    uvec4 settings;  // rays on first bounce, bounces, flags, view mode
    uvec4 counts;    // unused, GI spatial samples, ReSTIR candidates, spatial samples
    vec4 grassA;     // x cell size (m), y length scale, z LOD start distance, w draw distance
    vec4 grassB;     // x tan(half horizontal fov + margin), y tan(half vertical fov + margin), z keep radius, w off-screen density
    ivec4 grassGrid; // xy first grid cell, z cells per side, w blade capacity this frame
    vec4 terrainP;   // xy grid centre, z inner extent, w outer extent
    vec4 jitter;     // xy this frame's sub-pixel camera jitter (render pixels, +y down), zw last frame's (0 without DLSS)
    uvec4 bake;      // baked interior GI, see probes.glsl
    vec4 river;      // half-width, depth, water level at x = 0, level drop per m downstream (terrain.h RiverParams)
    vec4 water;      // x flow strength, y clarity (absorption scale), z foam amount, w bank splashes
    vec4 water2;     // x foam trail length behind rocks (m), y tree placement range (m), z tree density, w riverbed bumps (m)
    vec4 waterCol;   // rgb in-scatter colour (murk), w absorption per m (scaled per channel inside water.glsl)
    vec4 dynRocks[256];   // thrown rocks / fragments (MAX_THROWN): centre xyz, w = size (rock scale equivalent)
    vec4 dynRockV[256];   // their velocity (m/s)
    vec4 splashEv[128];   // water impacts (MAX_SPLASH_EV = 64): [2i] x, z, start time, impact; [2i+1] x = size (m)
    vec4 craters[256];     // ground craters (MAX_CRATERS): x, z, radius, depth (terrain.h Crater)
    uvec4 dynInfo;        // x thrown rocks, y splash events, z craters
    vec4 clouds;          // x coverage 0..1, y extinction per m, z drift speed, w enabled
    vec4 clouds2;         // x base altitude (m), y thickness (m), z shadow strength
    uvec4 groundTex;      // texture slots: bare albedo, bare detail, soil albedo, soil detail (ground.glsl)
    vec4 groundP;         // x tile size (m), y texels per side, z bare max height (m), w soil max height (m)
    vec4 groundS;         // x normal strength, y parallax on, z parallax distance (m), w textures loaded
    ivec4 windGrid;       // xy wind-map window origin (cells), zw last frame's origin (wind.glsl)
    vec4 windSim;         // x substep dt, y gust front strength, z front direction spread (rad), w sim on
    vec4 windSim2;        // x wave speed^2, y pressure damping, z relaxation to target, w sim time
    uvec4 lightInfo;      // x emissive triangle lights, y light tree nodes, z all emitters are in the light list, w decals
    vec4 post0;           // x exposure compensation (EV), y auto exposure on, z adaptation speed brighter, w darker (1/s)
    vec4 post1;           // x bloom strength, y lens flare, z vignette, w chromatic aberration (px at the edge)
    vec4 post2;           // x film grain, y sharpen, z tonemapper (0 AgX, 1 AgX punchy, 2 ACES, 3 linear), w frame dt (s)
    vec4 post3;           // x focus distance (m), y aperture (lens diameter m), z motion blur (shutter fraction), w DoF on
    vec4 post4;           // x display (0 SDR, 1 HDR10 PQ, 2 scRGB), y paper white (nits), z peak (nits), w output aspect
    vec4 fog0;            // x density (1/m at the ground), y height falloff (1/m), z phase anisotropy g, w enabled
    vec4 fog1;            // x local lights in fog, y max distance (m), z noise amount, w temporal blend
    vec4 weather;         // x wetness 0..1, y rain 0..1, z puddles 0..1, w caustics strength
    vec4 decals[256];     // decals (MAX_DECALS = 128): [2i] centre xyz, radius; [2i+1] normal xyz, kind + age / 1000
    uvec4 fxInfo;         // x dust puffs
    vec4 dust[64];        // dust clouds (MAX_DUST = 32): [2i] centre xyz, radius; [2i+1] x density, y age (s), z rise speed
    vec4 craterW[256];    // crater ponds fed by the river (main.cpp updateCraterWater): x water level (-1e9 = none),
                          // y fill 0..1, z share of the river's flow diverted into it, w inflow speed (m/s)
    vec4 craterG[256];    // natural ground under each crater (main.cpp): x height at the centre, yz slope (dh/dx, dh/dz)
    uvec4 mapPal[2];      // world map ground textures: bindless slot + 1 (0 = none), palette index 0..7
    vec4 mapPalTile[2];   // metres per repeat
    vec4 mapP;            // x origin (m), y cell (m), z cells per side, w -
    vec4 sim0;            // x snow line (m), y precipitation 0..1, z melt rate, w river search range (m)
    vec4 sim1;            // x angle of repose (tan), y avalanche mobility, z sim step (s), w sim on
    vec4 sim2;            // x snowfall (m/s at full precipitation), y wetness decay (1/s), z water mobility, w river swell from rain (m)
    vec4 simBrush;        // editor snow brush: x, z, radius (0 = off), metres per second (negative removes)
    vec4 fractal;         // ground detail (ground.glsl): x mode (0 textures, 1 fractal, 2 hybrid, 3 layered virtual texture), y relief, z pebbles, w finest scale (m)
    ivec4 craterGrid;     // crater bins (material.glsl): xy first bin (16 m), z bins per side, w valid
    ivec4 fogRes;         // volumetric fog: x resolution divisor (1 full .. 4 quarter), yz grid size (volfog.comp)
    vec4 clouds3;         // stratus deck: x coverage, y lumpiness (stratus -> stratocumulus), z base (m), w thickness (m)
    vec4 clouds4;         // x cumulonimbus amount, y cumulonimbus top (m), z cirrus amount, w cirrus altitude (m)
    vec4 clouds5;         // cirrus: x strand length, y strand fineness, z puffiness (strands -> tufts)
    vec4 grassC;          // x clumping, y clump size (m), z blade width scale, w curvature
    vec4 grassD;          // x twist, y outward tilt, z height variation, w dead blade fraction
    vec4 ground2;         // x macro warp, y mottling strength, z mottling scale, w dark spots
    vec4 ground3;         // x dark spot size, y grit, z cavity shading, w tile breakup
    vec4 vegCol;          // x grass colour variation, y ground colour variation
    vec4 vtP;             // ground virtual texture (groundvt.glsl): x finest texel (m), y texels per side, z levels, w on
    vec4 vtRect[14];      // per level: resident world xz rectangle (min xz, max xz)
} F;

// Camera projection matching cameraRay(): reverse-Z (near plane 0.05 -> depth 1, infinity -> 0).
vec4 project(vec3 P) {
    vec3 d = P - F.camPos.xyz;
    float z = dot(d, F.camFwd.xyz);
    float x = dot(d, F.camRight.xyz) / (F.camUp.w * F.camRight.w);
    float y = dot(d, F.camUp.xyz) / F.camUp.w;
    // sub-pixel jitter matching cameraRay(), which samples pixel p at p + 0.5 + jitter: move the image the
    // opposite way so that point lands on the pixel centre (NDC offset scaled by w to survive the divide)
    vec2 j = 2.0 * F.jitter.xy / vec2(F.res.xy) * z;
    return vec4(x - j.x, -y - j.y, 0.05, z);
}
#endif
