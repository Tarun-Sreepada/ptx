#include "lighttree.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace lighttree {

static uint32_t buildRange(const std::vector<Prim>& p, uint32_t* idx, uint32_t n, std::vector<Node>& out) {
    uint32_t me = (uint32_t)out.size();
    out.push_back({});
    Node nd{};
    for (int k = 0; k < 3; k++) { nd.mn[k] = 1e30f; nd.mx[k] = -1e30f; }
    float cmn[3] = {1e30f, 1e30f, 1e30f}, cmx[3] = {-1e30f, -1e30f, -1e30f};
    for (uint32_t i = 0; i < n; i++) {
        const Prim& q = p[idx[i]];
        nd.power += q.power;
        for (int k = 0; k < 3; k++) {
            nd.mn[k] = std::min(nd.mn[k], q.mn[k]);
            nd.mx[k] = std::max(nd.mx[k], q.mx[k]);
            float c = 0.5f * (q.mn[k] + q.mx[k]);
            cmn[k] = std::min(cmn[k], c);
            cmx[k] = std::max(cmx[k], c);
        }
    }
    if (n == 1) {
        nd.child = LEAF | idx[0];
        out[me] = nd;
        return me;
    }
    // median split of the centroids along the widest axis
    int ax = 0;
    for (int k = 1; k < 3; k++) if (cmx[k] - cmn[k] > cmx[ax] - cmn[ax]) ax = k;
    uint32_t half = n / 2;
    std::nth_element(idx, idx + half, idx + n, [&](uint32_t a, uint32_t b) {
        return p[a].mn[ax] + p[a].mx[ax] < p[b].mn[ax] + p[b].mx[ax];
    });
    buildRange(p, idx, half, out);
    nd.child = buildRange(p, idx + half, n - half, out);
    out[me] = nd;
    return me;
}

void build(const std::vector<Prim>& prims, std::vector<Node>& out) {
    out.clear();
    if (prims.empty()) return;
    out.reserve(prims.size() * 2);
    std::vector<uint32_t> idx(prims.size());
    std::iota(idx.begin(), idx.end(), 0u);
    buildRange(prims, idx.data(), (uint32_t)idx.size(), out);
}

void buildAlias(const std::vector<Prim>& prims, std::vector<AliasEntry>& out) {
    size_t n = prims.size();
    out.assign(n, {});
    if (!n) return;
    double total = 0;
    for (auto& p : prims) total += std::max(p.power, 0.f);
    std::vector<double> q(n);
    std::vector<uint32_t> small, large;
    for (size_t i = 0; i < n; i++) {
        double pw = total > 0 ? std::max(prims[i].power, 0.f) / total : 1.0 / n;
        out[i].pdf = (float)pw;
        q[i] = pw * n;
        (q[i] < 1.0 ? small : large).push_back((uint32_t)i);
    }
    while (!small.empty() && !large.empty()) {
        uint32_t a = small.back(); small.pop_back();
        uint32_t b = large.back();
        out[a].prob = (float)q[a];
        out[a].alias = b;
        q[b] -= 1.0 - q[a];
        if (q[b] < 1.0) { large.pop_back(); small.push_back(b); }
    }
    for (uint32_t i : large) { out[i].prob = 1.f; out[i].alias = i; }
    for (uint32_t i : small) { out[i].prob = 1.f; out[i].alias = i; }
}

// Void-and-cluster (Ulichney 1993) on a torus with a Gaussian energy kernel.
static void voidCluster(int S, uint32_t seed, std::vector<float>& rankOut) {
    const int N = S * S;
    const float sigma = 1.5f;
    std::vector<float> kern(N);
    for (int y = 0; y < S; y++)
        for (int x = 0; x < S; x++) {
            int dx = std::min(x, S - x), dy = std::min(y, S - y);
            kern[y * S + x] = expf(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        }
    std::vector<uint8_t> on(N, 0);
    std::vector<float> E(N, 0.f);
    auto splat = [&](int i, float s) {
        int px = i % S, py = i / S;
        for (int y = 0; y < S; y++) {
            const float* kr = &kern[((y - py + S) % S) * S];
            float* er = &E[y * S];
            for (int x = 0; x < S; x++) er[x] += s * kr[(x - px + S) % S];
        }
    };
    auto tightest = [&]() { int b = -1; for (int i = 0; i < N; i++) if (on[i] && (b < 0 || E[i] > E[b])) b = i; return b; };
    auto largestVoid = [&]() { int b = -1; for (int i = 0; i < N; i++) if (!on[i] && (b < 0 || E[i] < E[b])) b = i; return b; };
    std::mt19937 rng(seed);
    int ones = N / 10;
    for (int k = 0; k < ones;) {
        int i = (int)(rng() % N);
        if (!on[i]) { on[i] = 1; splat(i, 1.f); k++; }
    }
    for (int it = 0; it < 4 * N; it++) {  // relax the initial pattern
        int c = tightest();
        on[c] = 0; splat(c, -1.f);
        int v = largestVoid();
        on[v] = 1; splat(v, 1.f);
        if (v == c) break;
    }
    std::vector<uint8_t> init = on;
    std::vector<float> initE = E;
    std::vector<int> rank(N, -1);
    for (int r = ones - 1; r >= 0; r--) {  // phase 1: remove clusters
        int c = tightest();
        on[c] = 0; splat(c, -1.f);
        rank[c] = r;
    }
    on = init;
    E = initE;
    for (int r = ones; r < N; r++) {  // phases 2 + 3: fill voids
        int v = largestVoid();
        on[v] = 1; splat(v, 1.f);
        rank[v] = r;
    }
    rankOut.resize(N);
    for (int i = 0; i < N; i++) rankOut[i] = (rank[i] + 0.5f) / N;
}

void blueNoise(int size, std::vector<float>& out) {
    out.assign((size_t)size * size * 4, 0.f);
    std::vector<float> ch;
    for (int c = 0; c < 4; c++) {
        voidCluster(size, 0x9e3779b9u * (c + 1), ch);
        for (int i = 0; i < size * size; i++) out[(size_t)i * 4 + c] = ch[i];
    }
}

}  // namespace lighttree
