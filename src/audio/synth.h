// Built-in sounds, synthesized at startup (mono, SAMPLE_RATE): footsteps, hum, wind, thunder, impacts (modal
// synthesis), splashes and fire, and the sounds of the world around: birdsong, crickets, a river, leaves, an owl.
#pragma once
#include <cstdint>
#include <vector>

namespace audio::synth {
inline constexpr int SAMPLE_RATE = 48000;
std::vector<float> footstep(uint32_t seed, float cutoff, float thump, float len);
std::vector<float> hum();
std::vector<float> wind();
std::vector<float> thunder(uint32_t seed, bool near);
std::vector<float> impact(uint32_t seed, int kind);  // kind: 0 stone, 1 wood, 2 dirt, 3 metal
std::vector<float> splash(uint32_t seed, bool big);
std::vector<float> fire();
std::vector<float> bird(uint32_t seed, int kind);    // kind 0..3: four singers
std::vector<float> crickets();
std::vector<float> river();
std::vector<float> leaves();
std::vector<float> owl();
}  // namespace audio::synth
