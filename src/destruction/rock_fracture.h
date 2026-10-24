#pragma once
#include <vector>
#include <cstdint>
#include "physics/fracture.h"
#include "world/world.h"
#include "core/math.h"

struct FractureReq { uint32_t body; V3 contact, dir; float En; };

extern std::vector<FractureReq> pendingFractures;

extern fracture::Poly rockPoly[16];

TextureData rockTexture();

float fractureThreshold(float radius);
