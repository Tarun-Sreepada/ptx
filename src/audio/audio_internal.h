// Audio internals shared by its parts: constants, the emitter slots, the Steam Audio objects and the threads' state.
//   audio.cpp       the API (main thread)        mixer.cpp     the device callback: DSP per emitter, binaural, reverb
//   geometry.cpp    acoustic meshes / instances  simulation.cpp the simulation thread: direct paths, reflections
#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include <miniaudio.h>
#include <phonon.h>

#include "audio/audio.h"
#include "audio/synth.h"

namespace audio {
inline constexpr int RATE = synth::SAMPLE_RATE, FRAME = 1024;  // Steam Audio's convolution is tuned for 1024-sample frames
inline constexpr int MAX_EMITTERS = 64;
inline constexpr size_t MAX_SOUNDS = 1024;
inline constexpr int AMBI_ORDER = 1, AMBI_CH = (AMBI_ORDER + 1) * (AMBI_ORDER + 1);
inline constexpr float IR_SECONDS = 1.0f;
inline constexpr int IR_SIZE = (int)(IR_SECONDS * RATE);

// absorption (low/mid/high), scattering, transmission (low/mid/high) — values from Steam Audio's presets
inline const IPLMaterial kMaterials[MAT_COUNT] = {
    {{0.10f, 0.20f, 0.30f}, 0.05f, {0.100f, 0.050f, 0.030f}},  // generic
    {{0.05f, 0.07f, 0.08f}, 0.05f, {0.015f, 0.002f, 0.001f}},  // concrete
    {{0.11f, 0.07f, 0.06f}, 0.05f, {0.070f, 0.014f, 0.005f}},  // wood
    {{0.06f, 0.03f, 0.02f}, 0.05f, {0.060f, 0.044f, 0.011f}},  // glass
    {{0.20f, 0.07f, 0.06f}, 0.05f, {0.200f, 0.025f, 0.010f}},  // metal
    {{0.60f, 0.70f, 0.80f}, 0.40f, {0.000f, 0.000f, 0.000f}},  // ground / grass / soil
};

struct Sound { std::vector<float> s; };  // mono, RATE; immutable once registered

enum SlotState : int { FREE, PENDING, LIVE, DEAD };
struct Emitter {
    std::atomic<int> state{FREE};
    std::atomic<float> px{0}, py{0}, pz{0}, gain{1};
    std::atomic<bool> stop{false};
    const Sound* sound = nullptr;
    bool loop = false;
    float pitch = 1.f;
    double cursor = 0;  // audio thread
    IPLSource src = nullptr;
    IPLDirectEffect direct = nullptr;
    IPLBinauralEffect binaural = nullptr;
    IPLReflectionEffect reflect = nullptr;
};

struct SubMesh { IPLScene scene = nullptr; IPLStaticMesh mesh = nullptr; uint32_t tris = 0; };

extern IPLContext ctx;
extern IPLHRTF hrtf;
extern IPLEmbreeDevice embree;
extern IPLScene scene;
extern IPLSimulator sim;
extern IPLReflectionMixer mixer;
extern IPLAmbisonicsDecodeEffect ambiDecode;
extern IPLSceneType sceneType;
extern IPLAudioSettings as;
extern ma_device device;
extern bool deviceOk, ok;

extern Settings S;
extern std::vector<Sound> sounds;  // index = sound id; only appended from the main thread before use
extern Emitter emitters[MAX_EMITTERS];

// listener snapshot shared with the sim and audio threads
extern std::mutex lisMtx;
extern IPLCoordinateSpace3 listener;
inline IPLCoordinateSpace3 getListener() { std::lock_guard l(lisMtx); return listener; }

// scene edits are queued by the main thread and applied by the simulation thread between runs
extern std::mutex cmdMtx;
extern std::vector<std::function<void()>> cmds;
inline void post(std::function<void()> f) { std::lock_guard l(cmdMtx); cmds.push_back(std::move(f)); }

// sim-thread owned
extern std::vector<SubMesh> meshes;
struct Inst { IPLInstancedMesh m = nullptr; uint32_t tris = 0; };
extern std::vector<Inst> instances;
extern SubMesh terrain;
extern uint32_t sceneTris;
// main-thread handle allocation
extern uint32_t nextMesh, nextInst;
extern std::vector<uint32_t> freeMesh, freeInst;
extern uint32_t liveMeshes, liveInsts;

extern std::thread simThread;
extern std::atomic<bool> quit;
extern std::atomic<float> simMs, dspMs, windGain;
extern IPLAudioBuffer inBuf, dirBuf, binBuf, reflBuf, ambiBuf, mixBuf, decBuf;  // DSP scratch (device thread)

void dataCallback(ma_device*, void* output, const void*, ma_uint32 frames);  // mixer.cpp
void releaseEmitter(Emitter& e);                                             // simulation.cpp
void simMain();
}  // namespace audio
