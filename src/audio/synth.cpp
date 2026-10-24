#include "audio/synth.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace audio::synth {
std::vector<float> footstep(uint32_t seed, float cutoff, float thump, float len) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> s((size_t)(len * SAMPLE_RATE));
    float lp = 0, a = 1.f - expf(-2.f * 3.14159265f * cutoff / SAMPLE_RATE);
    for (size_t i = 0; i < s.size(); i++) {
        float t = (float)i / SAMPLE_RATE;
        lp += (U(rng) - lp) * a;
        float crunch = lp * expf(-t / 0.035f) * (1.f + 0.6f * U(rng) * expf(-t / 0.01f));
        float body = sinf(2.f * 3.14159265f * (70.f + 40.f * expf(-t / 0.02f)) * t) * expf(-t / 0.05f) * thump;
        s[i] = (crunch * 1.6f + body) * 0.5f;
    }
    return s;
}

std::vector<float> hum() {
    std::vector<float> s(2 * SAMPLE_RATE);  // 2 s, every partial completes whole cycles so the loop is seamless
    for (size_t i = 0; i < s.size(); i++) {
        float t = (float)i / SAMPLE_RATE, w = 2.f * 3.14159265f;
        float trem = 0.8f + 0.2f * sinf(w * 1.5f * t);
        s[i] = trem * (0.5f * sinf(w * 110 * t) + 0.3f * sinf(w * 220 * t) + 0.12f * sinf(w * 330.5f * t) + 0.05f * sinf(w * 661 * t)) * 0.4f;
    }
    return s;
}

std::vector<float> wind() {
    const size_t n = 8 * SAMPLE_RATE;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> s(n);
    float lp1 = 0, lp2 = 0;
    for (size_t i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        float gust = 0.55f + 0.45f * sinf(2.f * 3.14159265f * t / 8.f) * sinf(2.f * 3.14159265f * t * 3.f / 8.f);
        float fc = 250.f + 500.f * gust, a = 1.f - expf(-2.f * 3.14159265f * fc / SAMPLE_RATE);
        lp1 += (U(rng) - lp1) * a;
        lp2 += (lp1 - lp2) * a;
        s[i] = lp2 * gust * 1.5f;
    }
    const size_t xf = SAMPLE_RATE / 2;  // crossfade the ends so the loop has no click
    for (size_t i = 0; i < xf; i++) {
        float k = (float)i / xf;
        s[i] = s[i] * k + s[n - xf + i] * (1.f - k);
    }
    s.resize(n - xf);
    return s;
}

// Thunder: a near strike starts with the tearing crack of the channel (broadband, sharp), then rolls; a distant one
// is only the low roll (air absorbs the highs), arriving softly and swelling as sound from further along the
// channel comes in.
std::vector<float> thunder(uint32_t seed, bool near) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f), U01(0.f, 1.f);
    const float len = near ? 7.f : 9.f;
    std::vector<float> s((size_t)(len * SAMPLE_RATE));
    float lp1 = 0, lp2 = 0, hp = 0, lpc = 0;
    // swells: a few random bumps of rumble
    float bt[6], ba[6], bw[6];
    for (int k = 0; k < 6; k++) { bt[k] = (near ? 0.3f : 0.6f) + U01(rng) * (len * 0.55f); ba[k] = 0.4f + 0.6f * U01(rng); bw[k] = 0.3f + 0.8f * U01(rng); }
    float fc = near ? 220.f : 110.f, a = 1.f - expf(-2.f * 3.14159265f * fc / SAMPLE_RATE), ac = 1.f - expf(-2.f * 3.14159265f * 3000.f / SAMPLE_RATE);
    for (size_t i = 0; i < s.size(); i++) {
        float t = (float)i / SAMPLE_RATE, n = U(rng);
        lp1 += (n - lp1) * a;
        lp2 += (lp1 - lp2) * a;
        float env = (1.f - expf(-t / (near ? 0.05f : 0.4f))) * expf(-t / (near ? 2.2f : 2.8f));
        float sw = 0.35f;
        for (int k = 0; k < 6; k++) sw += ba[k] * expf(-(t - bt[k]) * (t - bt[k]) / (bw[k] * bw[k]));
        float v = lp2 * 6.f * env * sw;
        if (near) {  // the crack: band-limited noise burst with sharp attack, a few ms of crackles after
            lpc += (n - lpc) * ac;
            hp = n - lpc;
            float crack = expf(-t / 0.08f) * (t < 0.002f ? t / 0.002f : 1.f);
            float rip = U01(rng) < 0.002f * expf(-t / 0.4f) ? 1.f : 0.f;
            v += (lpc * 1.2f + hp * 0.4f) * crack + rip * U(rng) * 0.6f;
        }
        s[i] = std::clamp(v, -1.f, 1.f) * 0.8f;
    }
    return s;
}

// Impacts by modal synthesis: a few damped resonances of the struck body (frequency ratios and decay per material)
// over a short noise burst for the contact itself.
std::vector<float> impact(uint32_t seed, int kind) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f), U01(0.f, 1.f);
    struct M { float f0, ratio[5], t60, noiseLen, noiseFc, body; };
    static const M mats[4] = {
        {1100.f, {1.f, 2.32f, 3.91f, 5.43f, 7.2f}, 0.06f, 0.012f, 4000.f, 0.0f},  // stone: sharp click, short ring
        {380.f, {1.f, 2.74f, 5.1f, 7.6f, 0.f}, 0.07f, 0.02f, 1800.f, 0.4f},      // wood: hollow knock
        {0.f, {0.f}, 0.0f, 0.12f, 500.f, 1.0f},                                    // dirt: dull thud, crumbling grit
        {520.f, {1.f, 2.76f, 5.40f, 8.93f, 13.3f}, 1.1f, 0.006f, 6000.f, 0.f},    // metal: clang, long ring
    };
    const M& m = mats[kind];
    float len = kind == 3 ? 1.8f : (kind == 2 ? 0.45f : 0.4f);
    std::vector<float> s((size_t)(len * SAMPLE_RATE));
    float jit = 0.85f + 0.3f * U01(rng);
    float amp[5], ph[5];
    for (int k = 0; k < 5; k++) { amp[k] = (k == 0 ? 1.f : 0.6f / (k + 0.5f)) * (0.6f + 0.8f * U01(rng)); ph[k] = U01(rng) * 6.2832f; }
    float lp = 0, lp2 = 0, a = 1.f - expf(-2.f * 3.14159265f * m.noiseFc / SAMPLE_RATE);
    for (size_t i = 0; i < s.size(); i++) {
        float t = (float)i / SAMPLE_RATE, v = 0.f;
        if (m.f0 > 0.f)
            for (int k = 0; k < 5; k++) {
                if (m.ratio[k] <= 0.f) continue;
                float f = m.f0 * jit * m.ratio[k];
                float dec = m.t60 / (1.f + 0.6f * k) / 6.9f;  // higher modes die faster
                v += amp[k] * sinf(6.2832f * f * t + ph[k]) * expf(-t / dec);
            }
        lp += (U(rng) - lp) * a;
        lp2 += (lp - lp2) * a;
        float nz = (kind == 2 ? lp2 * 3.f : lp) * expf(-t / m.noiseLen) * (t < 0.0015f ? t / 0.0015f : 1.f);
        if (kind == 2 && U01(rng) < 0.004f * expf(-t / 0.15f)) nz += U(rng) * 0.4f;  // grit settling
        float body = m.body * sinf(6.2832f * (70.f + 60.f * expf(-t / 0.02f)) * t) * expf(-t / 0.06f);
        s[i] = std::clamp((v * 0.5f + nz * 1.4f + body) * 0.6f, -1.f, 1.f);
    }
    return s;
}

// Water: the plop (cavity collapse ringing a bubble: a fast upward chirp) over a broadband splash of droplets.
std::vector<float> splash(uint32_t seed, bool big) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f), U01(0.f, 1.f);
    float len = big ? 1.4f : 0.5f;
    std::vector<float> s((size_t)(len * SAMPLE_RATE));
    float lp = 0, hp = 0, a = 1.f - expf(-2.f * 3.14159265f * 2500.f / SAMPLE_RATE);
    int nb = big ? 14 : 4;
    std::vector<float> bt(nb), bf(nb);
    for (int k = 0; k < nb; k++) { bt[k] = U01(rng) * len * 0.5f + (k == 0 ? 0.f : 0.03f); bf[k] = 400.f + 1600.f * U01(rng) * (big ? 0.7f : 1.f); }
    for (size_t i = 0; i < s.size(); i++) {
        float t = (float)i / SAMPLE_RATE, n = U(rng);
        lp += (n - lp) * a;
        hp = n - lp;
        float env = expf(-t / (big ? 0.35f : 0.12f)) * (t < 0.003f ? t / 0.003f : 1.f);
        float v = (lp * 0.8f + hp * 0.5f) * env;
        for (int k = 0; k < nb; k++) {
            float u = t - bt[k];
            if (u < 0.f || u > 0.08f) continue;
            float f = bf[k] * (1.f + 8.f * u);  // bubble resonance rising as it collapses
            v += 0.35f * sinf(6.2832f * f * u) * expf(-u / 0.018f);
        }
        s[i] = std::clamp(v, -1.f, 1.f) * 0.7f;
    }
    return s;
}

// Fire: the roar of burning gas (low noise, flickering) with the pops and crackles of sap and splitting wood.
std::vector<float> fire() {
    const size_t n = 6 * SAMPLE_RATE;
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> U(-1.f, 1.f), U01(0.f, 1.f);
    std::vector<float> s(n);
    float lp1 = 0, lp2 = 0, a = 1.f - expf(-2.f * 3.14159265f * 300.f / SAMPLE_RATE), fl = 0.5f;
    float pop = 0.f, popF = 0.f, popT = 0.f;
    for (size_t i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        lp1 += (U(rng) - lp1) * a;
        lp2 += (lp1 - lp2) * a;
        fl += (U01(rng) - fl) * 0.0005f;
        float v = lp2 * 3.f * (0.6f + 0.8f * fl);
        if (U01(rng) < 0.0006f) { pop = 0.3f + 0.7f * U01(rng); popF = 1500.f + 4000.f * U01(rng); popT = t; }
        float u = t - popT;
        v += pop * U(rng) * expf(-u / 0.004f) + pop * 0.3f * sinf(6.2832f * popF * u) * expf(-u / 0.01f);
        s[i] = std::clamp(v, -1.f, 1.f) * 0.7f;
    }
    const size_t xf = SAMPLE_RATE / 2;
    for (size_t i = 0; i < xf; i++) { float k = (float)i / xf; s[i] = s[i] * k + s[n - xf + i] * (1.f - k); }
    s.resize(n - xf);
    return s;
}

namespace {
constexpr float TWO_PI = 6.2831853f;

// one whistled note: frequency sliding f0 -> f1 with vibrato, a smooth (Hann) envelope, a little second harmonic
void note(std::vector<float>& s, float start, float dur, float f0, float f1, float vibHz, float vibDepth, float amp) {
    size_t i0 = (size_t)(start * SAMPLE_RATE), n = (size_t)(dur * SAMPLE_RATE);
    float ph = 0.f;
    for (size_t i = 0; i < n && i0 + i < s.size(); i++) {
        float u = (float)i / (float)n, t = (float)i / SAMPLE_RATE;
        float f = f0 + (f1 - f0) * u;
        f *= 1.f + vibDepth * sinf(TWO_PI * vibHz * t);
        ph += TWO_PI * f / SAMPLE_RATE;
        float env = sinf(3.14159265f * u);
        env *= env;
        s[i0 + i] += amp * env * (sinf(ph) + 0.15f * sinf(2.f * ph));
    }
}

// the ends of a loop crossfaded together (no click), xf samples long
void loopSeam(std::vector<float>& s, size_t xf) {
    size_t n = s.size();
    for (size_t i = 0; i < xf; i++) { float k = (float)i / xf; s[i] = s[i] * k + s[n - xf + i] * (1.f - k); }
    s.resize(n - xf);
}
}  // namespace

// Songbird phrases, 2-6 kHz whistles with fast sweeps: kind 0 a warbler's varied run of notes, 1 a clear two-note
// "fee-bee", 2 a fast trill ending in a flourish, 3 a few quick down-chirps.
std::vector<float> bird(uint32_t seed, int kind) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::vector<float> s((size_t)(2.4f * SAMPLE_RATE), 0.f);
    float t = 0.05f;
    switch (kind) {
    case 0:
        for (int k = 0, n = 7 + (int)(U(rng) * 4); k < n; k++) {
            float d = 0.06f + 0.09f * U(rng), f = 2600.f + 3000.f * U(rng);
            note(s, t, d, f, f * (0.7f + 0.6f * U(rng)), 25.f + 30.f * U(rng), 0.03f * U(rng), 0.45f);
            t += d + 0.02f + 0.05f * U(rng);
        }
        break;
    case 1:
        note(s, t, 0.38f, 4100.f, 4050.f, 6.f, 0.004f, 0.5f);
        note(s, t + 0.48f, 0.32f, 3450.f, 3300.f, 6.f, 0.004f, 0.45f);
        break;
    case 2: {
        float f = 4600.f + 800.f * U(rng);
        for (int k = 0; k < 18; k++) { note(s, t, 0.032f, f * 1.08f, f * 0.94f, 0.f, 0.f, 0.4f); t += 0.055f; }
        note(s, t + 0.04f, 0.16f, f * 0.8f, f * 1.15f, 30.f, 0.02f, 0.45f);
        break;
    }
    default:
        for (int k = 0, n = 2 + (int)(U(rng) * 3); k < n; k++) {
            note(s, t, 0.045f, 6200.f + 600.f * U(rng), 3000.f + 500.f * U(rng), 0.f, 0.f, 0.5f);
            t += 0.18f + 0.25f * U(rng);
        }
        break;
    }
    return s;
}

// Field crickets on a warm night: a 4.5 kHz carrier in chirps of three pulses, ~2.5 chirps a second, and a second
// cricket a little higher and slower further off. 4 s: both rhythms close exactly, so the loop is seamless.
std::vector<float> crickets() {
    std::vector<float> s((size_t)(4.f * SAMPLE_RATE), 0.f);
    auto chirps = [&](float period, float f, float amp) {
        for (float c = 0.f; c + period <= 4.f + 1e-4f; c += period)
            for (int p = 0; p < 3; p++) note(s, c + p * 0.024f, 0.015f, f, f * 0.99f, 0.f, 0.f, amp);
    };
    chirps(0.4f, 4500.f, 0.35f);
    chirps(4.f / 12.f, 4950.f, 0.18f);
    return s;
}

// A river: the rush of moving water (noise in a slowly wandering band around 400-1200 Hz) and the soft plinks of
// air pockets (short damped sines rising in pitch). 8 s loop.
std::vector<float> river() {
    const size_t n = 8 * SAMPLE_RATE;
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> U(-1.f, 1.f), U01(0.f, 1.f);
    std::vector<float> s(n);
    float lp1 = 0, lp2 = 0, hp = 0, prev = 0;
    float bubbleT = -1.f, bubbleF = 0.f, bubbleA = 0.f;
    for (size_t i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        float fc = 800.f + 400.f * sinf(TWO_PI * t / 8.f) * sinf(TWO_PI * t * 3.f / 8.f), a = 1.f - expf(-TWO_PI * fc / SAMPLE_RATE);
        lp1 += (U(rng) - lp1) * a;
        lp2 += (lp1 - lp2) * a;
        hp = 0.995f * (hp + lp2 - prev);  // take out the rumble below ~40 Hz
        prev = lp2;
        float v = hp * 2.2f;
        if (U01(rng) < 0.0009f) { bubbleT = t; bubbleF = 500.f + 1500.f * U01(rng); bubbleA = 0.1f + 0.2f * U01(rng); }
        float u = t - bubbleT;
        if (u >= 0.f && u < 0.06f) v += bubbleA * sinf(TWO_PI * bubbleF * (1.f + 6.f * u) * u) * expf(-u / 0.015f);
        s[i] = std::clamp(v, -1.f, 1.f) * 0.6f;
    }
    loopSeam(s, SAMPLE_RATE / 2);
    return s;
}

// Leaves in the wind: crisp high noise (2-8 kHz) swelling with the gusts, with fast flutters. 8 s loop.
std::vector<float> leaves() {
    const size_t n = 8 * SAMPLE_RATE;
    std::mt19937 rng(37);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> s(n);
    float lp = 0, slow = 0, prev = 0;
    float a = 1.f - expf(-TWO_PI * 7000.f / SAMPLE_RATE);
    for (size_t i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        lp += (U(rng) - lp) * a;
        float hi = lp - prev;  // differentiated: the highs of the rustle
        prev = lp;
        slow += (U(rng) - slow) * 0.0004f;
        float gust = 0.5f + 0.5f * sinf(TWO_PI * t / 8.f + 1.f) * sinf(TWO_PI * t * 5.f / 8.f);
        float flutter = 0.6f + 0.4f * fabsf(slow) * 8.f;
        s[i] = std::clamp(hi * 3.f * gust * flutter, -1.f, 1.f) * 0.5f;
    }
    loopSeam(s, SAMPLE_RATE / 2);
    return s;
}

// A tawny owl: a breathy hoot, a pause, then a long wavering "hoo-oo-oo" sliding down (~400 Hz).
std::vector<float> owl() {
    std::mt19937 rng(41);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> s((size_t)(3.2f * SAMPLE_RATE), 0.f);
    note(s, 0.05f, 0.45f, 420.f, 400.f, 5.f, 0.01f, 0.5f);
    note(s, 1.3f, 1.4f, 430.f, 360.f, 9.f, 0.035f, 0.5f);
    float lp = 0.f;  // breath
    for (size_t i = 0; i < s.size(); i++) {
        lp += (U(rng) - lp) * 0.05f;
        s[i] += lp * 0.25f * fabsf(s[i]);
    }
    return s;
}
}  // namespace audio::synth
