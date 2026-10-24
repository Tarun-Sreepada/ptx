// Leaves through the year for broadleaf species: fresh, lighter and sparser in spring as they open, the species'
// own colour in summer, then yellow, orange or red in autumn (which, depends on the species) as they thin out, and
// bare branches in winter. Conifers keep their needles.
#include "vegetation/trees.h"

#include <algorithm>
#include <cmath>

namespace trees {
namespace {
constexpr float SPRING = 80.f, SUMMER = 140.f, AUTUMN = 250.f, WINTER = 320.f;  // day of year each begins
constexpr float BUCKET = 10.f;                                                   // days the leaves stay the same
const float AUTUMN_COLOR[4][3] = {{0.55f, 0.40f, 0.05f}, {0.62f, 0.24f, 0.04f}, {0.48f, 0.08f, 0.04f}, {0.32f, 0.18f, 0.06f}};

float smooth01(float x) { x = std::clamp(x, 0.f, 1.f); return x * x * (3.f - 2.f * x); }
}  // namespace

float seasonBucketDay(float d) { return d < 0.f ? -1.f : floorf(d / BUCKET) * BUCKET + 0.5f * BUCKET; }

TreeParams seasonal(const TreeParams& P, int species) {
    if (seasonDay < 0.f || conifer(P) || P.leafDensity <= 0.f) return P;
    TreeParams q = P;
    float d = seasonDay;
    if (d >= SPRING && d < SUMMER) {  // opening: lighter, yellower green, filling out
        float k = 1.f - smooth01((d - SPRING) / (SUMMER - SPRING));
        const float spring[3] = {1.3f, 1.3f, 0.75f};
        for (int c = 0; c < 3; c++) q.leafColor[c] = P.leafColor[c] * (1.f + (spring[c] - 1.f) * k);
        q.leafDensity = P.leafDensity * (1.f - 0.5f * k);
    } else if (d >= AUTUMN && d < WINTER) {  // turning, then falling
        float k = (d - AUTUMN) / (WINTER - AUTUMN), c0 = smooth01(k / 0.6f);
        const float* au = AUTUMN_COLOR[species % 4];
        for (int c = 0; c < 3; c++) q.leafColor[c] = P.leafColor[c] + (au[c] - P.leafColor[c]) * c0;
        q.leafDensity = P.leafDensity * (1.f - 0.85f * smooth01((k - 0.5f) / 0.5f));
    } else if (d >= WINTER || d < SPRING) {
        q.leafDensity = 0.f;  // bare
    }
    return q;
}
}  // namespace trees
