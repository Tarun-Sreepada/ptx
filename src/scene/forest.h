#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "world/world.h"

void loadForest(const std::string& worldDir);

MeshData mergeAsset(const ModelData& a, uint32_t matBase);

void assetTrunk(const MeshData& m, std::vector<float>& cv, std::vector<uint32_t>& ci);
