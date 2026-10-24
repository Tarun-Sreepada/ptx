#pragma once
#include <cstdint>

void frustumPlanes(float planes[5][4]);

void setPrevXf(uint32_t k, const float m[3][4], uint32_t prevSlot);

void finishPrevXf();

void cullModels();
