#pragma once
#include <cstdint>
#include "world/terrain.h"
#include "core/limits.h"

struct UBO {
    float camPos[4], camRight[4], camUp[4], camFwd[4];
    float prevCamPos[4], prevRight[4], prevUp[4], prevFwd[4];
    float sunDir[4], sunColor[4], wind[4], params[4], params2[4];
    uint32_t res[4], settings[4], counts[4];
    float grassA[4], grassB[4];
    int32_t grassGrid[4];
    float terrainP[4];
    float jitter[4];
    uint32_t bake[4];  // probes.glsl: volumes, dynamic lights, flags, lights incl. switched-off ones
    float river[4], water[4], water2[4], waterCol[4];  // water.glsl
    float dynRocks[MAX_THROWN][4];          // thrown rocks / fragments: centre, size (rock-mesh scale equivalent)
    float dynRockV[MAX_THROWN][4];          // their velocity (m/s)
    float splashEv[128][4];                 // water impacts: (x, z, time, impact), (size, speed, -, -)
    float craters[MAX_CRATERS][4];          // terrain.h Crater
    uint32_t dynInfo[4];                    // rocks, splash events, craters
    float clouds[4], clouds2[4];            // clouds.glsl
    uint32_t groundTex[4];                  // ground.glsl
    float groundP[4], groundS[4];
    int32_t windGrid[4];                    // wind.glsl
    float windSim[4], windSim2[4];
    uint32_t lightInfo[4];                  // tri lights, light tree nodes, all emitters listed, decals
    float post0[4], post1[4], post2[4], post3[4], post4[4];  // frame.glsl: exposure, bloom / lens, tonemap, DoF / motion blur, display
    float fog0[4], fog1[4];                 // volumetric fog
    float weather[4];                       // wetness, rain, puddles, caustics
    float decals[2 * MAX_DECALS][4];
    uint32_t fxInfo[4];                     // dust puffs
    float dust[2 * MAX_DUST][4];
    float craterW[MAX_CRATERS][4];          // frame.glsl craterW
    float craterG[MAX_CRATERS][4];          // frame.glsl craterG
    uint32_t mapPal[8];                     // world map ground textures: bindless slot + 1 (0 = none), index 1..7
    float mapPalTile[8];                    // metres per repeat
    float mapP[4];                          // x origin, y cell size, z cells per side, w -
    float sim0[4];                          // x snow line (m), y precipitation 0..1, z melt rate (m/s at +10 m below the line), w river search range (m)
    float sim1[4];                          // x angle of repose (tan), y avalanche mobility, z sim step (s), w sim on
    float sim2[4];                          // x snowfall (m/s at full precipitation), y wetness decay (1/s), z water mobility, w -
    float simBrush[4];                      // editor snow brush: x, z, radius (0 = off), m/s
    float fractal[4];                       // ground detail: mode (0 textures, 1 fractal, 2 hybrid), relief, pebbles, finest scale (m)
    int32_t craterGrid[4];                  // crater bins: first bin xz (16 m), bins per side, valid
    int32_t fogRes[4];                      // volumetric fog: resolution divisor, grid width, height
    float clouds3[4], clouds4[4];           // clouds.glsl: stratus, cumulonimbus, cirrus
    float clouds5[4];                       // cirrus: strand length, fineness, puffiness
    float grassC[4], grassD[4];             // grass.comp: clumping / shape, variation
    float ground2[4], ground3[4];           // ground.glsl: noise, dark spots, grit
    float vegCol[4];                        // x grass colour variation, y ground colour variation
    float vtP[4];                           // ground virtual texture: finest texel (m), texels per side, levels, on
    float vtRect[14][4];                    // per level: resident world xz rectangle
    float sssP[4];                          // subsurface: on, max walk steps, distance scale, snow
    float wet2[4];                          // soaked-in water, drips from cover, rivulets
    float sun2[4], moon0[4], moon1[4], night0[4], night1[4], purk[4];  // night.glsl
    float bolt[96][4];                      // lightning.glsl segments
    float boltInfo[4], boltPos[4];
    float rainbow[4];                       // composite.comp
    float fogBank[4], fogBank2[4];          // volfog.comp
    float terrain2[4];                      // terrain clipmap: levels, cells per level side, morph band (cells)
    float fires[2 * MAX_FIRES][4];          // fire.glsl: [2i] base xyz, radius; [2i+1] heat, seed, flame height, smoke height
    float fireInfo[4];                      // x fires
    float pushers[2 * MAX_PUSHERS][4];      // grass.comp: [2i] xyz, radius; [2i+1] x flattening 0..1
    float pushInfo[4];                      // x pushers
};
static_assert(sizeof(UBO) <= 65536, "the frame UBO must fit maxUniformBufferRange");
