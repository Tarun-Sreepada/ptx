// Static GPU resources of the scene that aren't streamed: terrain / grass / rock geometry and their acceleration
// structures, the sky / wind / world-map / virtual-texture images, the light and model tables, the TLAS.
#pragma once
#include <cstdint>
#include <vulkan/vulkan.h>
#include "gpu/vk.h"

extern Buffer terrainV, terrainI, grassV, grassI, bladeBuf, lightBuf, uboBuf, instBuf, grassCounter, countReadback, rockV, rockI, rockXf;
extern Image skyImg;
extern Image windImg[2];
extern Accel terrainAS, grassAS, rockAS[16], tlas;
extern VkAccelerationStructureGeometryKHR grassGeom, terrainGeom, rockGeom[16], tlasGeom;
extern uint32_t terrainTris, rockTris, rockVerts, numRocks;

// CPU half of buildScene (rock meshes, terrain / grass index lists): runs on the job pool at startup, no GPU.
void prepareSceneGeometry();
// Creates every static buffer / image (prepareSceneGeometry must have finished).
void buildScene();
void createAccels();
void cmdBuildTerrainBLAS(VkCommandBuffer cb);
void cmdBuildGrassBLAS(VkCommandBuffer cb);
void cmdBuildTLAS(VkCommandBuffer cb);
