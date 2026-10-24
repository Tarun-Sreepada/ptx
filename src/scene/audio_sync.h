#pragma once
#include <vector>
#include <cstdint>

extern std::vector<uint32_t> audioMeshRetire;

void syncAudioInstances();

void updateAudio(float dt);
