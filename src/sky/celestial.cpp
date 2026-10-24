#include "sky/celestial.h"
#include <algorithm>
#include <cmath>
#include "app/settings.h"
#include "core/math.h"

// Sun transmittance through the atmosphere at the viewer (mirrors atmosphere.glsl), for the sun colour.
V3 sunTransmittance(V3 sd) {
    const double RG = 6360e3, RT = 6460e3;
    double py = RG + 300.0;
    double b = py * sd.y, c = py * py - RG * RG, disc = b * b - c;
    if (disc >= 0 && -b - sqrt(disc) > 0) return {0, 0, 0};
    double L = -b + sqrt(b * b - (py * py - RT * RT)), ds = L / 32;
    double od[3] = {0, 0, 0};
    for (int i = 0; i < 32; i++) {
        double t = (i + 0.5) * ds, x = sd.x * t, y = py + sd.y * t, z = sd.z * t;
        double h = sqrt(x * x + y * y + z * z) - RG;
        double r = exp(-h / 8000.0), m = exp(-h / 1200.0), o = std::max(0.0, 1.0 - fabs(h - 25000.0) / 15000.0);
        const double br[3] = {5.802e-6, 13.558e-6, 33.1e-6}, bo[3] = {0.650e-6, 1.881e-6, 0.085e-6};
        for (int k = 0; k < 3; k++) od[k] += br[k] * r + 4.40e-6 * m + bo[k] * o;
    }
    return {(float)exp(-od[0] * ds), (float)exp(-od[1] * ds), (float)exp(-od[2] * ds)};
}

V3 sunDirection() {
    float el = S.sunElev * 3.14159265f / 180.f, az = S.sunAzim * 3.14159265f / 180.f;
    return {cosf(el) * sinf(az), sinf(el), cosf(el) * cosf(az)};
}

// The sky turns about the celestial pole, which leans north (-z) at the latitude (night.glsl celestialPole).
V3 celestialPole() { float l = S.latitude * 3.14159265f / 180.f; return {0.f, sinf(l), -cosf(l)}; }

// The moon sits where the sun was (phase x 1 day) earlier in its daily turn: rotated eastward about the pole by the
// elongation (0 new, 180 deg full: opposite the sun).
V3 moonDirection() { return normalize(rotateAbout(sunDirection(), celestialPole(), S.moonPhase * 2.f * 3.14159265f)); }

// Moonlight at full moon is ~2.2e-6 of sunlight (0.25 lux / 110 klux); fainter away from full by the lunar phase
// law (Allen: magnitude + 0.026 |a| + 4e-9 a^4, a = phase angle in degrees).
float moonlightScale() {
    float a = fabsf(S.moonPhase - 0.5f) * 360.f;
    return 2.2e-6f * powf(10.f, -0.4f * (0.026f * a + 4e-9f * a * a * a * a));
}

bool keyIsMoon() {
    V3 sd = sunDirection(), md = moonDirection();
    auto lum = [](V3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; };
    float sunL = lum(sunTransmittance(sd)) * S.sunIntensity;
    float moonL = lum(sunTransmittance(md)) * S.sunIntensity * moonlightScale() * S.nightBoost;
    return md.y > 0.f && moonL > sunL;
}
