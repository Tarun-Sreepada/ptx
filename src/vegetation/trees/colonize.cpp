// Space colonization (Runions, Lane & Prusinkiewicz 2007): the crown is a cloud of attraction points (the space a
// tree competes for, inside its crown envelope); every step each point pulls the nearest branch node within reach,
// each pulled node grows one segment toward the average pull, and points a branch reaches are used up. Branches
// fill the crown evenly, avoid each other and fork where space opens, which reads as a grown tree rather than a
// rule-built one. Thickness follows the pipe model (a parent's cross-section ~ the sum of its children's), leaves
// sit on the outer twigs. The result is the same Plan as the rule-based generator (chains of segments, parents,
// foliage clusters), so the meshes and the wood mechanics work on it unchanged.
#include "vegetation/trees/growth.h"

#include <unordered_map>

namespace trees {
namespace {
constexpr int MAX_NODES = 6000;
constexpr int MAX_STEPS = 220;
constexpr float SEGMENT = 0.025f;      // node spacing / tree height
constexpr float REACH = 0.22f;         // influence radius / tree height
constexpr float KILL = 2.f;            // kill distance / node spacing
constexpr float PIPE_EXP = 2.5f;       // pipe model: r^n of a parent = sum of its children's
constexpr int MERGE = 3;               // nodes per mesh / beam segment
constexpr float TWIG_LEAVES = 3.f;     // the last this many nodes of a twig carry leaf clusters
constexpr int ATTRACTORS = 2600;       // at leaf density 1

struct Node { V p; int parent; float r = 0.f; std::vector<int> kids; V pull{0, 0, 0}; int pulls = 0; };

float shape(int s, float r) {  // crown envelope (growth.cpp shapeRatio)
    r = std::clamp(r, 0.f, 1.f);
    switch (s) {
    case CROWN_CONICAL: return 0.2f + 0.8f * r;
    case CROWN_SPHERICAL: return 0.2f + 0.8f * sinf(PI * r);
    case CROWN_HEMISPHERICAL: return 0.2f + 0.8f * sinf(0.5f * PI * r);
    case CROWN_CYLINDRICAL: return 1.f;
    case CROWN_TAPERED: return 0.5f + 0.5f * r;
    case CROWN_FLAME: return std::max(0.1f, r <= 0.7f ? r / 0.7f : (1.f - r) / 0.3f);
    default: return 1.f - 0.8f * r;
    }
}

struct Grid {  // nodes by cell, for "nearest node within reach"
    float cell;
    std::unordered_map<uint64_t, std::vector<int>> cells;
    uint64_t key(int x, int y, int z) const { return ((uint64_t)(uint32_t)(x + 4096) << 42) | ((uint64_t)(uint32_t)(y + 4096) << 21) | (uint32_t)(z + 4096); }
    void add(V p, int i) { cells[key((int)floorf(p.x / cell), (int)floorf(p.y / cell), (int)floorf(p.z / cell))].push_back(i); }
    int nearest(V p, const std::vector<Node>& nodes, float maxD, float& d2out) const {
        int cx = (int)floorf(p.x / cell), cy = (int)floorf(p.y / cell), cz = (int)floorf(p.z / cell), best = -1;
        float bd = maxD * maxD;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++) {
                    auto it = cells.find(key(cx + dx, cy + dy, cz + dz));
                    if (it == cells.end()) continue;
                    for (int i : it->second) {
                        V d = nodes[i].p - p;
                        float d2 = dot(d, d);
                        if (d2 < bd) { bd = d2; best = i; }
                    }
                }
        d2out = bd;
        return best;
    }
};
}  // namespace

Plan colonizeTree(const TreeParams& P0, uint32_t seed, uint32_t leafMat, uint32_t woodMat) {
    TreeParams P = P0;
    Rng rng{seed * 2654435761u + 7u};
    P.height *= rng.in(0.88f, 1.12f);
    float H = P.height, boleY = std::clamp(P.bole, 0.05f, 0.9f) * H;
    float D = SEGMENT * H, reach = REACH * H, kill = KILL * D;
    Plan p;
    p.leafMat = leafMat;
    p.woodMat = woodMat;
    p.leafWidth = P.leafWidth;
    p.droop = P.leafDroop;
    p.upturn = P.upturn;
    p.needles = conifer(P);
    p.flare = P.flare;
    p.wave = P.trunkWave;
    p.seed = (float)(seed % 997u);
    V lean = V{rng.in(-1.f, 1.f), 0, rng.in(-1.f, 1.f)} * (P.lean * boleY);
    V top = V{0, boleY, 0} + lean;
    p.trunk = {{0, -0.3f, 0}, top, P.trunkRadius * 1.15f, P.trunkRadius * 0.85f, 10, -1, -1};

    // the crown's space: points in the envelope (radius by height from the crown shape), a little lumpy
    float crownR = (H - boleY) * P.lengthRatio * 0.9f + 0.5f;
    std::vector<V> pts;
    int want = (int)(ATTRACTORS * std::sqrt(std::max(P.leafDensity, 0.2f)));
    for (int tries = 0; (int)pts.size() < want && tries < want * 20; tries++) {
        V q{rng.in(-crownR, crownR), rng.in(boleY + 0.5f, H), rng.in(-crownR, crownR)};
        float ratio = (H - q.y) / std::max(H - boleY, 0.5f), R = crownR * shape(P.crown, ratio);
        R *= 0.85f + 0.3f * noise3(q * (1.5f / std::max(crownR, 1.f)) + V{p.seed, 0, 0});
        if (q.x * q.x + q.z * q.z <= R * R) pts.push_back(q + lean * ((q.y - boleY) / std::max(H, 1.f)));
    }

    std::vector<Node> nodes;
    Grid grid{reach};
    nodes.push_back({top, -1});
    grid.add(top, 0);
    // the leader climbs straight until the crown's points are within reach
    auto nearAny = [&](V q) { for (const V& a : pts) { V d = a - q; if (dot(d, d) < reach * reach) return true; } return false; };
    V up = norm(V{0, 1, 0} + lean * (1.f / std::max(boleY, 0.5f)));
    while (!nearAny(nodes.back().p) && nodes.back().p.y < H && (int)nodes.size() < 64) {
        Node n{nodes.back().p + up * D, (int)nodes.size() - 1};
        nodes[n.parent].kids.push_back((int)nodes.size());
        grid.add(n.p, (int)nodes.size());
        nodes.push_back(n);
    }
    // grow
    V tropism = V{0, 1, 0} * (P.upturn * 0.35f) - V{0, 1, 0} * (P.gravity * 0.25f);
    std::vector<uint8_t> alive(pts.size(), 1);
    for (int step = 0; step < MAX_STEPS && (int)nodes.size() < MAX_NODES; step++) {
        for (Node& n : nodes) { n.pull = V{0, 0, 0}; n.pulls = 0; }
        int live = 0;
        for (size_t i = 0; i < pts.size(); i++) {
            if (!alive[i]) continue;
            float d2;
            int k = grid.nearest(pts[i], nodes, reach, d2);
            if (k < 0) continue;
            if (d2 < kill * kill) { alive[i] = 0; continue; }  // reached: this space is taken
            live++;
            nodes[k].pull = nodes[k].pull + norm(pts[i] - nodes[k].p);
            nodes[k].pulls++;
        }
        if (!live) break;
        size_t before = nodes.size();
        for (size_t k = 0; k < before && nodes.size() < (size_t)MAX_NODES; k++) {
            if (!nodes[k].pulls) continue;
            V dir = norm(norm(nodes[k].pull) + tropism + randUnit(rng) * (P.gnarl * 0.35f));  // the average pull, bent by light / weight
            V q = nodes[k].p + dir * D;
            Node n{q, (int)k};
            nodes[k].kids.push_back((int)nodes.size());
            grid.add(q, (int)nodes.size());
            nodes.push_back(n);
        }
        if (nodes.size() == before) break;
    }

    // pipe model thickness, tips up (children always come after their parents)
    float tipR = std::max(0.006f, P.trunkRadius * 0.03f);
    for (size_t i = nodes.size(); i-- > 0;) {
        Node& n = nodes[i];
        if (n.kids.empty()) { n.r = tipR; continue; }
        float s = 0.f;
        for (int c : n.kids) s += powf(nodes[c].r, PIPE_EXP);
        n.r = powf(s, 1.f / PIPE_EXP);
    }
    float k = P.trunkRadius * 0.85f / std::max(nodes[0].r, 1e-4f);  // the stem meets the trunk's top
    for (Node& n : nodes) n.r = std::max(n.r * k, tipR * 0.6f);

    // chains: the thickest child carries a limb on, the others start limbs of their own
    struct Start { int node; int parentSeg; float parentR; V parentDir; int level; };
    std::vector<Start> todo{{0, -1, 0.f, V{0, 1, 0}, 0}};
    std::vector<int> segOfNode(nodes.size(), -1);  // the segment a node lies on (branches hang off it)
    int leafN = std::max(2, (int)lroundf(P.leavesPerCluster * P.leafDensity));
    while (!todo.empty()) {
        Start st = todo.back();
        todo.pop_back();
        std::vector<int> path{st.node};
        for (int n = st.node;;) {
            const Node& nd = nodes[n];
            if (nd.kids.empty()) break;
            int best = nd.kids[0];
            for (int c : nd.kids) if (nodes[c].r > nodes[best].r) best = c;
            for (int c : nd.kids) if (c != best) todo.push_back({c, -2 - n, nd.r, norm(nodes[best].p - nd.p), st.level + 1});  // parent: resolved below
            path.push_back(best);
            n = best;
        }
        if (path.size() < 2) continue;
        int chain = (int)p.chains.size();
        Chain ch;
        float r0 = nodes[path[0]].r;
        ch.sides = st.level == 0 ? 8 : (r0 > 0.06f ? 6 : (r0 > 0.015f ? 5 : 3));
        ch.parentR = st.parentR;
        ch.parentDir = st.parentDir;
        p.chains.push_back(ch);
        // the branch hangs off the segment its first node lies on
        int parentSeg = st.parentSeg <= -2 ? segOfNode[-2 - st.parentSeg] : st.parentSeg;
        int prevSeg = -1;
        for (size_t i = 0; i + 1 < path.size();) {
            size_t j = std::min(i + MERGE, path.size() - 1);
            const Node &a = nodes[path[i]], &b = nodes[path[j]];
            p.branches.push_back({a.p, b.p, a.r, b.r, ch.sides, prevSeg >= 0 ? prevSeg : parentSeg, std::min(st.level, 3), chain});
            prevSeg = (int)p.branches.size() - 1;
            p.chains[chain].segs.push_back(prevSeg);
            for (size_t q = i; q <= j; q++) segOfNode[path[q]] = prevSeg;
            i = j;
        }
        // leaves on the outer, thin part of the limb
        if (P.leafDensity > 0.f && nodes[path.back()].r < tipR * 4.f) {
            int nl = std::min((int)path.size() - 1, (int)TWIG_LEAVES);
            for (int q = 0; q < nl; q++) {
                int ni = path[std::max((int)path.size() - 1 - q * 2, 0)];
                const Node& n = nodes[ni];
                V dir = n.parent >= 0 ? norm(n.p - nodes[n.parent].p) : V{0, 1, 0};
                p.leaves.push_back({F_LEAVES, n.p, dir, P.leafSpread * (q == 0 ? 1.f : 0.75f), P.leafSize, leafN, rng.in(0.f, 1000.f), segOfNode[ni]});
            }
        }
    }
    fitFoliageBudget(p);
    return p;
}
}  // namespace trees
