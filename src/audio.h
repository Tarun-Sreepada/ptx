// Ray-traced spatial audio: Steam Audio (phonon) simulates against the same world geometry the renderer
// streams (simplified with meshoptimizer), miniaudio drives the output device.
//   direct path: distance attenuation, air absorption, volumetric occlusion, frequency-dependent transmission
//   reflections: real-time ray-traced impulse responses (1st-order ambisonics), shared convolution mixer
//   output:      HRTF binaural on stereo / headphones
// Threads: main thread (this API), simulation thread (rays), audio device thread (DSP). The API is
// main-thread only; it never blocks on simulation.
#pragma once
#include <cstdint>

#include "world.h"

namespace audio {
enum Material : uint8_t { MAT_GENERIC, MAT_CONCRETE, MAT_WOOD, MAT_GLASS, MAT_METAL, MAT_GROUND, MAT_COUNT };

struct Settings {
    float masterGain = 1.f;
    bool occlusion = true, transmission = true, reflections = true;
    int reflectionRays = 1024, bounces = 6;
    float reflectionHz = 15.f;  // reflection IRs update rate (direct paths run every tick, ~60 Hz)
    float reverbGain = 1.f;
};

bool init();
void shutdown();
Settings& settings();

// ---- geometry (meshes are simplified for acoustics; instances reference them)
uint32_t meshCreate(const float* verts, uint32_t strideFloats, uint32_t nverts, const uint32_t* idx, uint32_t ntris, Material mat);
void meshRelease(uint32_t mesh);
uint32_t instanceAdd(uint32_t mesh, const Xf& xf);
void instanceRemove(uint32_t inst);
// ground patch around the listener (world space, replaces the previous one)
void setTerrain(const float* verts, uint32_t nverts, const uint32_t* idx, uint32_t ntris);

// ---- listener / sounds / emitters
void setListener(const float pos[3], const float fwd[3], const float up[3]);
uint32_t soundLoad(const char* path);  // any format miniaudio decodes; mono-mixed, resampled
enum BuiltinSound : uint32_t {
    SND_FOOTSTEP0, SND_FOOTSTEP1, SND_FOOTSTEP2, SND_FOOTSTEP3, SND_LAND, SND_HUM, SND_WIND,
    SND_THUNDER_NEAR, SND_THUNDER_FAR,                       // a close crack + rumble, a distant low roll
    SND_IMPACT_STONE, SND_IMPACT_WOOD = SND_IMPACT_STONE + 3, // impacts: 3 variants each (modal synthesis), pitch by size
    SND_IMPACT_DIRT = SND_IMPACT_WOOD + 3, SND_IMPACT_METAL = SND_IMPACT_DIRT + 3,
    SND_SPLASH = SND_IMPACT_METAL + 3,                        // 2 variants: small plop, big splash
    SND_FIRE = SND_SPLASH + 2,                                // crackling fire (loop)
    SND_BUILTIN_COUNT
};
// Impact sound for a collision: material kind (0 stone, 1 wood, 2 dirt, 3 metal), kinetic energy dissipated (J) and
// the body's size (m): louder with energy (~ sqrt), lower pitched for bigger bodies. Rate-limited per position.
void impact(int kind, const float pos[3], float energyJ, float size);
uint32_t emitterPlay(uint32_t sound, const float pos[3], float gain, bool loop, float pitch = 1.f);  // UINT32_MAX if none free
void emitterSetPos(uint32_t e, const float pos[3]);
void emitterSetGain(uint32_t e, float gain);
void emitterStop(uint32_t e);
bool emitterAlive(uint32_t e);
void setAmbience(float windGain);  // non-spatial bed

struct Stats { uint32_t emitters, meshes, instances, sceneTris; float simMs, dspMs; bool running; };
Stats stats();
}  // namespace audio
