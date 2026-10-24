#include "characters/skel.h"

#include <algorithm>
#include <cmath>

namespace skel {
namespace {
// the key interval holding t: k such that times[k] <= t < times[k + 1], and the fraction in it
void findKey(const std::vector<float>& ts, float t, size_t& k, float& f) {
    if (t <= ts.front()) { k = 0; f = 0; return; }
    if (t >= ts.back()) { k = ts.size() - 1; f = 0; return; }
    k = (size_t)(std::upper_bound(ts.begin(), ts.end(), t) - ts.begin()) - 1;
    float d = ts[k + 1] - ts[k];
    f = d > 0 ? (t - ts[k]) / d : 0.f;
}
void nlerp(const float* a, const float* b, float w, float* o) {
    float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3], s = d < 0 ? -1.f : 1.f, l = 0;  // shortest arc
    float q[4];
    for (int i = 0; i < 4; i++) { q[i] = a[i] + (s * b[i] - a[i]) * w; l += q[i] * q[i]; }
    l = l > 0 ? 1.f / sqrtf(l) : 0.f;
    for (int i = 0; i < 4; i++) o[i] = q[i] * l;
}
}  // namespace

void restPose(const SkinData& s, Pose& out) { out = s.rest; }

void sample(const ClipData& c, float t, bool loop, Pose& p) {
    if (loop && c.duration > 0) { t = fmodf(t, c.duration); if (t < 0) t += c.duration; }
    for (const ClipChannel& ch : c.ch) {
        if (ch.t.empty() || (size_t)ch.joint * 10 >= p.size()) continue;
        int n = ch.path == 1 ? 4 : 3;
        float* dst = &p[ch.joint * 10 + (ch.path == 0 ? 0 : ch.path == 1 ? 3 : 7)];
        size_t k;
        float f;
        findKey(ch.t, t, k, f);
        const float* a = &ch.v[k * n];
        if (ch.step || f == 0.f || k + 1 >= ch.t.size()) { std::copy(a, a + n, dst); continue; }
        const float* b = &ch.v[(k + 1) * n];
        if (n == 4) nlerp(a, b, f, dst);
        else for (int i = 0; i < 3; i++) dst[i] = a[i] + (b[i] - a[i]) * f;
    }
}

void blend(const Pose& a, const Pose& b, float w, Pose& out) {
    out.resize(a.size());
    for (size_t j = 0; j + 9 < a.size() && j + 9 < b.size(); j += 10) {
        for (int i : {0, 1, 2, 7, 8, 9}) out[j + i] = a[j + i] + (b[j + i] - a[j + i]) * w;
        nlerp(&a[j + 3], &b[j + 3], w, &out[j + 3]);
    }
}

Xf localXf(const float* r) {
    float x = r[3], y = r[4], z = r[5], w = r[6];
    float R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
                     {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
                     {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
    Xf o;
    for (int i = 0; i < 3; i++) {
        for (int k = 0; k < 3; k++) o.m[i][k] = R[i][k] * r[7 + k];
        o.m[i][3] = r[i];
    }
    return o;
}

void globals(const SkinData& s, const Pose& p, std::vector<Xf>& out) {
    out.resize(s.parent.size());
    for (int j : s.order) {
        Xf l = localXf(&p[j * 10]);
        out[j] = s.parent[j] >= 0 ? xfMul(out[s.parent[j]], l) : xfMul(s.rootXf[j], l);
    }
}

void skinMatrices(const SkinData& s, const std::vector<Xf>& g, float* out) {
    for (size_t j = 0; j < g.size(); j++) {
        Xf m = xfMul(g[j], s.invBind[j]);
        std::copy(&m.m[0][0], &m.m[0][0] + 12, out + j * 12);
    }
}

Xf inverse(const Xf& a) {
    const auto& m = a.m;
    float c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1], c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2], c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    float det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02, id = fabsf(det) > 1e-20f ? 1.f / det : 0.f;
    Xf r;
    r.m[0][0] = c00 * id; r.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id; r.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id;
    r.m[1][0] = c01 * id; r.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id; r.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id;
    r.m[2][0] = c02 * id; r.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id; r.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id;
    for (int i = 0; i < 3; i++) r.m[i][3] = -(r.m[i][0] * m[0][3] + r.m[i][1] * m[1][3] + r.m[i][2] * m[2][3]);
    return r;
}
}  // namespace skel
