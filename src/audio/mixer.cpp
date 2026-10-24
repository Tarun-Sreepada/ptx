// The output device callback: every live emitter through its direct path, binaural rendering and reflections.
#include "audio/audio_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "core/prof.h"

namespace audio {
namespace {
// ---------------------------------------------------------------- device thread
float fifo[FRAME * 2 * 2];  // interleaved stereo, at most one block + leftover
int fifoCount = 0;          // frames

void renderBlock(float* outInterleaved) {
    PROF_ZONE("audio block");
    auto t0 = std::chrono::steady_clock::now();
    IPLCoordinateSpace3 L = getListener();
    for (int c = 0; c < 2; c++) std::fill_n(mixBuf.data[c], FRAME, 0.f);
    bool anyRefl = false;
    for (auto& e : emitters) {
        if (e.state.load(std::memory_order_acquire) != LIVE) continue;
        if (e.stop.load()) { e.state.store(DEAD, std::memory_order_release); continue; }
        // source samples (pitch via linear interpolation)
        const auto& s = e.sound->s;
        float g = e.gain.load(), *in = inBuf.data[0];
        bool ended = false;
        for (int i = 0; i < FRAME; i++) {
            if (e.cursor >= s.size() - 1) {
                if (e.loop) e.cursor = fmod(e.cursor, (double)(s.size() - 1));
                else { std::fill(in + i, in + FRAME, 0.f); ended = true; break; }
            }
            size_t k = (size_t)e.cursor;
            float f = (float)(e.cursor - k);
            in[i] = (s[k] + (s[k + 1] - s[k]) * f) * g;
            e.cursor += e.pitch;
        }
        IPLSimulationOutputs out{};
        iplSourceGetOutputs(e.src, (IPLSimulationFlags)(IPL_SIMULATIONFLAGS_DIRECT | IPL_SIMULATIONFLAGS_REFLECTIONS), &out);
        // direct path
        IPLDirectEffectParams& dp = out.direct;
        dp.flags = (IPLDirectEffectFlags)(IPL_DIRECTEFFECTFLAGS_APPLYDISTANCEATTENUATION | IPL_DIRECTEFFECTFLAGS_APPLYAIRABSORPTION |
                                          (S.occlusion ? IPL_DIRECTEFFECTFLAGS_APPLYOCCLUSION : 0) |
                                          (S.transmission ? IPL_DIRECTEFFECTFLAGS_APPLYTRANSMISSION : 0));
        dp.transmissionType = IPL_TRANSMISSIONTYPE_FREQDEPENDENT;
        iplDirectEffectApply(e.direct, &dp, &inBuf, &dirBuf);
        IPLVector3 sp{e.px.load(), e.py.load(), e.pz.load()};
        IPLBinauralEffectParams bp{};
        bp.direction = iplCalculateRelativeDirection(ctx, sp, L.origin, L.ahead, L.up);
        bp.interpolation = IPL_HRTFINTERPOLATION_BILINEAR;
        bp.spatialBlend = 1.f;
        bp.hrtf = hrtf;
        iplBinauralEffectApply(e.binaural, &bp, &dirBuf, &binBuf);
        iplAudioBufferMix(ctx, &binBuf, &mixBuf);
        // ray-traced reflections, accumulated into the shared mixer (one inverse FFT for all sources)
        if (S.reflections && out.reflections.ir) {
            IPLReflectionEffectParams& rp = out.reflections;
            rp.type = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;
            rp.numChannels = AMBI_CH;
            rp.irSize = IR_SIZE;
            iplReflectionEffectApply(e.reflect, &rp, &inBuf, &reflBuf, mixer);
            anyRefl = true;
        }
        if (ended && !e.loop) e.state.store(DEAD, std::memory_order_release);
    }
    if (anyRefl) {
        IPLReflectionEffectParams rp{};
        rp.type = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;
        rp.numChannels = AMBI_CH;
        rp.irSize = IR_SIZE;
        iplReflectionMixerApply(mixer, &rp, &ambiBuf);
        IPLAmbisonicsDecodeEffectParams ap{};
        ap.order = AMBI_ORDER;
        ap.hrtf = hrtf;
        ap.orientation = L;
        ap.binaural = IPL_TRUE;
        iplAmbisonicsDecodeEffectApply(ambiDecode, &ap, &ambiBuf, &decBuf);
        for (int c = 0; c < 2; c++)
            for (int i = 0; i < FRAME; i++) mixBuf.data[c][i] += decBuf.data[c][i] * S.reverbGain;
    }
    // wind bed: the same loop read at two offsets so left/right decorrelate
    float wg = windGain.load();
    if (wg > 0.f && sounds.size() > SND_WIND) {
        static double wc = 0;
        const auto& w = sounds[SND_WIND].s;
        size_t n = w.size();
        for (int i = 0; i < FRAME; i++) {
            size_t k = (size_t)wc % n;
            mixBuf.data[0][i] += w[k] * wg;
            mixBuf.data[1][i] += w[(k + n / 3) % n] * wg;
            wc += 1;
        }
    }
    float mg = S.masterGain;
    for (int i = 0; i < FRAME; i++) {
        // soft clip keeps transient pile-ups (many footsteps + reverb) from wrapping
        outInterleaved[i * 2] = tanhf(mixBuf.data[0][i] * mg);
        outInterleaved[i * 2 + 1] = tanhf(mixBuf.data[1][i] * mg);
    }
    float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    dspMs.store(dspMs.load() * 0.95f + ms * 0.05f);
}

}  // namespace

void dataCallback(ma_device*, void* output, const void*, ma_uint32 frames) {
    float* out = (float*)output;
    while (frames > 0) {
        if (fifoCount == 0) { renderBlock(fifo); fifoCount = FRAME; }
        int take = std::min<int>(frames, fifoCount);
        int off = FRAME - fifoCount;
        memcpy(out, fifo + off * 2, take * 2 * sizeof(float));
        out += take * 2;
        frames -= take;
        fifoCount -= take;
    }
}

}  // namespace audio
