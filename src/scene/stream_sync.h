#pragma once
#include <string>
#include <cstdint>

extern int loadRadius;

uint32_t zoneTexSlot(const std::string& name);

void onStreamChanged(bool moveOnly = false);
