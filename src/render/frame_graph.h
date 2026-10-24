#pragma once
#include <cstdint>
#include <imgui.h>
#include "gpu/rendergraph.h"

extern rg::Graph graph;

extern bool asyncCompute;

extern bool asyncFog;

void buildFrameGraph(uint32_t si, ImDrawData* dd);
