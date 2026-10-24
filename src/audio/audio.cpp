// The audio API (main thread): init / shutdown, the listener, sounds and emitters, impacts.
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#include "audio/audio_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

namespace audio {
IPLContext ctx;
IPLHRTF hrtf;
IPLEmbreeDevice embree;
IPLScene scene;
IPLSimulator sim;
IPLReflectionMixer mixer;
IPLAmbisonicsDecodeEffect ambiDecode;
IPLSceneType sceneType = IPL_SCENETYPE_DEFAULT;
IPLAudioSettings as{RATE, FRAME};
ma_device device;
bool deviceOk = false, ok = false;
Settings S;
std::vector<Sound> sounds;  // index = sound id; only appended from the main thread before use
Emitter emitters[MAX_EMITTERS];
std::mutex lisMtx;
IPLCoordinateSpace3 listener{{1, 0, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 0}};
std::mutex cmdMtx;
std::vector<std::function<void()>> cmds;
std::vector<SubMesh> meshes;
std::vector<Inst> instances;
SubMesh terrain;
uint32_t sceneTris = 0;
uint32_t nextMesh = 0, nextInst = 0;
std::vector<uint32_t> freeMesh, freeInst;
uint32_t liveMeshes = 0, liveInsts = 0;
std::thread simThread;
std::atomic<bool> quit{false};
std::atomic<float> simMs{0}, dspMs{0}, windGain{0};
IPLAudioBuffer inBuf, dirBuf, binBuf, reflBuf, ambiBuf, mixBuf, decBuf;

namespace {
void allocBuf(int ch, IPLAudioBuffer& b) { iplAudioBufferAllocate(ctx, ch, FRAME, &b); }
}  // namespace

// ---------------------------------------------------------------- API
bool init() {
    IPLContextSettings cs{};
    cs.version = STEAMAUDIO_VERSION;
    cs.simdLevel = IPL_SIMDLEVEL_AVX2;
    if (iplContextCreate(&cs, &ctx) != IPL_STATUS_SUCCESS) { fprintf(stderr, "audio: Steam Audio context failed\n"); return false; }
    IPLHRTFSettings hs{};
    hs.type = IPL_HRTFTYPE_DEFAULT;
    hs.volume = 1.f;
    hs.normType = IPL_HRTFNORMTYPE_NONE;
    iplHRTFCreate(ctx, &as, &hs, &hrtf);

    IPLEmbreeDeviceSettings es{};
    if (iplEmbreeDeviceCreate(ctx, &es, &embree) == IPL_STATUS_SUCCESS) sceneType = IPL_SCENETYPE_EMBREE;
    IPLSceneSettings scs{};
    scs.type = sceneType;
    scs.embreeDevice = embree;
    iplSceneCreate(ctx, &scs, &scene);

    IPLSimulationSettings ss{};
    ss.flags = (IPLSimulationFlags)(IPL_SIMULATIONFLAGS_DIRECT | IPL_SIMULATIONFLAGS_REFLECTIONS);
    ss.sceneType = sceneType;
    ss.reflectionType = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;
    ss.maxNumOcclusionSamples = 16;
    ss.maxNumRays = 4096;
    ss.numDiffuseSamples = 32;
    ss.maxDuration = IR_SECONDS;
    ss.maxOrder = AMBI_ORDER;
    ss.maxNumSources = MAX_EMITTERS;
    ss.numThreads = 2;
    ss.samplingRate = RATE;
    ss.frameSize = FRAME;
    if (iplSimulatorCreate(ctx, &ss, &sim) != IPL_STATUS_SUCCESS) { fprintf(stderr, "audio: simulator failed\n"); return false; }
    iplSimulatorSetScene(sim, scene);
    iplSimulatorCommit(sim);

    IPLReflectionEffectSettings rs{IPL_REFLECTIONEFFECTTYPE_CONVOLUTION, IR_SIZE, AMBI_CH};
    iplReflectionMixerCreate(ctx, &as, &rs, &mixer);
    IPLAmbisonicsDecodeEffectSettings ads{};
    ads.speakerLayout.type = IPL_SPEAKERLAYOUTTYPE_STEREO;
    ads.hrtf = hrtf;
    ads.maxOrder = AMBI_ORDER;
    iplAmbisonicsDecodeEffectCreate(ctx, &as, &ads, &ambiDecode);
    allocBuf(1, inBuf); allocBuf(1, dirBuf); allocBuf(2, binBuf); allocBuf(AMBI_CH, reflBuf);
    allocBuf(AMBI_CH, ambiBuf); allocBuf(2, mixBuf); allocBuf(2, decBuf);

    sounds.reserve(MAX_SOUNDS);  // never reallocates: emitters on the audio thread hold pointers into it
    sounds.resize(SND_BUILTIN_COUNT);
    for (uint32_t i = 0; i < 4; i++) sounds[SND_FOOTSTEP0 + i].s = synth::footstep(11 + i * 7, 1800.f + 500.f * i, 0.5f, 0.2f);
    sounds[SND_LAND].s = synth::footstep(99, 1200.f, 1.2f, 0.35f);
    sounds[SND_HUM].s = synth::hum();
    sounds[SND_WIND].s = synth::wind();
    sounds[SND_THUNDER_NEAR].s = synth::thunder(5, true);
    sounds[SND_THUNDER_FAR].s = synth::thunder(9, false);
    for (int k = 0; k < 4; k++)
        for (int v = 0; v < 3; v++) sounds[SND_IMPACT_STONE + k * 3 + v].s = synth::impact(100 + k * 17 + v * 5, k);
    sounds[SND_SPLASH].s = synth::splash(41, false);
    sounds[SND_SPLASH + 1].s = synth::splash(43, true);
    sounds[SND_FIRE].s = synth::fire();
    for (int k = 0; k < 4; k++) sounds[SND_BIRD + k].s = synth::bird(61 + 13 * k, k);
    sounds[SND_CRICKETS].s = synth::crickets();
    sounds[SND_RIVER].s = synth::river();
    sounds[SND_LEAVES].s = synth::leaves();
    sounds[SND_OWL].s = synth::owl();

    simThread = std::thread(simMain);

    ma_device_config dc = ma_device_config_init(ma_device_type_playback);
    dc.playback.format = ma_format_f32;
    dc.playback.channels = 2;
    dc.sampleRate = RATE;
    dc.periodSizeInFrames = FRAME / 2;
    dc.dataCallback = dataCallback;
    deviceOk = ma_device_init(nullptr, &dc, &device) == MA_SUCCESS && ma_device_start(&device) == MA_SUCCESS;
    if (!deviceOk) fprintf(stderr, "audio: no output device, running silent\n");
    printf("audio: Steam Audio %s scene, %d Hz, %d-sample frames, %s\n", sceneType == IPL_SCENETYPE_EMBREE ? "Embree" : "built-in", RATE,
           FRAME, deviceOk ? device.playback.name : "no device");
    ok = true;
    return true;
}

void shutdown() {
    if (!ok) return;
    if (deviceOk) ma_device_uninit(&device);
    quit = true;
    simThread.join();
    for (auto& e : emitters)
        if (e.state.load() != FREE && e.src) releaseEmitter(e);
    {
        std::lock_guard l(cmdMtx);
        cmds.clear();
    }
    for (auto& i : instances) if (i.m) iplInstancedMeshRelease(&i.m);
    for (auto& m : meshes) { if (m.mesh) iplStaticMeshRelease(&m.mesh); if (m.scene) iplSceneRelease(&m.scene); }
    if (terrain.mesh) iplStaticMeshRelease(&terrain.mesh);
    for (auto* b : {&inBuf, &dirBuf, &binBuf, &reflBuf, &ambiBuf, &mixBuf, &decBuf}) iplAudioBufferFree(ctx, b);
    iplAmbisonicsDecodeEffectRelease(&ambiDecode);
    iplReflectionMixerRelease(&mixer);
    iplSimulatorRelease(&sim);
    iplSceneRelease(&scene);
    if (embree) iplEmbreeDeviceRelease(&embree);
    iplHRTFRelease(&hrtf);
    iplContextRelease(&ctx);
    ok = false;
}

Settings& settings() { return S; }

void setListener(const float p[3], const float f[3], const float u[3]) {
    IPLVector3 ahead{f[0], f[1], f[2]}, up{u[0], u[1], u[2]};
    IPLVector3 right{ahead.y * up.z - ahead.z * up.y, ahead.z * up.x - ahead.x * up.z, ahead.x * up.y - ahead.y * up.x};
    std::lock_guard l(lisMtx);
    listener = {right, up, ahead, {p[0], p[1], p[2]}};
}

uint32_t soundLoad(const char* path) {
    ma_decoder_config dc = ma_decoder_config_init(ma_format_f32, 1, RATE);
    float* frames = nullptr;
    ma_uint64 n = 0;
    if (ma_decode_file(path, &dc, &n, (void**)&frames) != MA_SUCCESS) { fprintf(stderr, "audio: cannot decode %s\n", path); return UINT32_MAX; }
    Sound s;
    s.s.assign(frames, frames + n);
    ma_free(frames, nullptr);
    if (s.s.size() < 2 || sounds.size() >= MAX_SOUNDS) return UINT32_MAX;
    sounds.push_back(std::move(s));
    return (uint32_t)sounds.size() - 1;
}

uint32_t emitterPlay(uint32_t sound, const float pos[3], float gain, bool loop, float pitch) {
    if (!ok || sound >= sounds.size()) return UINT32_MAX;
    for (uint32_t i = 0; i < MAX_EMITTERS; i++) {
        Emitter& e = emitters[i];
        if (e.state.load(std::memory_order_acquire) != FREE) continue;
        e.sound = &sounds[sound];
        e.loop = loop;
        e.pitch = pitch;
        e.cursor = 0;
        e.stop = false;
        e.px = pos[0]; e.py = pos[1]; e.pz = pos[2];
        e.gain = gain;
        e.src = nullptr;
        e.state.store(PENDING, std::memory_order_release);
        return i;
    }
    return UINT32_MAX;
}

void emitterSetPos(uint32_t i, const float p[3]) {
    if (i >= MAX_EMITTERS) return;
    emitters[i].px = p[0]; emitters[i].py = p[1]; emitters[i].pz = p[2];
}
void emitterSetGain(uint32_t i, float g) { if (i < MAX_EMITTERS) emitters[i].gain = g; }
void emitterStop(uint32_t i) { if (i < MAX_EMITTERS) emitters[i].stop = true; }
bool emitterAlive(uint32_t i) {
    if (i >= MAX_EMITTERS) return false;
    int s = emitters[i].state.load();
    return (s == PENDING || s == LIVE) && !emitters[i].stop.load();
}
void setAmbience(float g) { windGain = g; }

void impact(int kind, const float pos[3], float E, float size) {
    if (E < 0.5f) return;
    // a body rattling on the ground fires contact after contact: one sound per spot per 80 ms
    static std::vector<std::array<float, 4>> recent;  // x, y, z, time
    static auto t0 = std::chrono::steady_clock::now();
    float now = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
    std::erase_if(recent, [&](const std::array<float, 4>& r) { return now - r[3] > 0.08f; });
    for (auto& r : recent) {
        float dx = r[0] - pos[0], dy = r[1] - pos[1], dz = r[2] - pos[2];
        if (dx * dx + dy * dy + dz * dz < 0.25f) return;
    }
    recent.push_back({pos[0], pos[1], pos[2], now});
    kind = std::clamp(kind, 0, 3);
    static uint32_t rot = 0;
    uint32_t snd = SND_IMPACT_STONE + kind * 3 + (rot++ % 3);
    // loudness ~ sqrt of the energy (sound power ~ a small fraction of the impact energy), ~1 at 200 J
    float gain = std::clamp(sqrtf(E / 200.f), 0.04f, 2.5f);
    // bigger bodies ring lower (modal frequencies ~ 1 / size), within a musical-ish range
    float pitch = std::clamp(powf(0.25f / std::max(size, 0.02f), 0.5f), 0.45f, 2.2f);
    emitterPlay(snd, pos, gain, false, pitch);
}

Stats stats() {
    Stats s{};
    for (auto& e : emitters) s.emitters += e.state.load() == LIVE;
    s.meshes = liveMeshes;
    s.instances = liveInsts;
    s.sceneTris = sceneTris + terrain.tris;
    s.simMs = simMs.load();
    s.dspMs = dspMs.load();
    s.running = ok && deviceOk;
    return s;
}
}  // namespace audio
