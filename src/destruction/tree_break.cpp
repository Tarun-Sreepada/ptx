#include "destruction/tree_break.h"
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "physics/physics.h"
#include "vegetation/trees.h"
#include "physics/fracture.h"
#include "world/world.h"
#include "core/limits.h"
#include "core/math.h"
#include "core/random.h"
#include "destruction/thrown.h"
#include "destruction/tree_damage.h"
#include "gpu/memory.h"
#include "render/targets.h"
#include "scene/models.h"
#include "scene/vegetation.h"
#include "sim/debris.h"
#include "sim/effects.h"

std::vector<TreeBreakReq> pendingTreeBreaks;

// Break a placed tree where it was hit (at): trees::breakApart cuts it into a stump, which stays as a static body,
// and logs / branches / crowns, which become wood pieces on the thrown list (they float, roll, knock other trees
// and break walls). severity >= 1 grows with the blow: more trunk breaks, more limbs and crowns torn loose, more
// splinters and leaves. The impulse goes into the piece nearest the hit, the rest get the tree's share of it.
void knockTree(const TreeInst& t, V3 impulse, V3 at, float severity) {
    if (brokenTrees.count(treeKey(t.ix, t.iz))) return;
    for (auto& q : pendingTreeBreaks) if (q.t.ix == t.ix && q.t.iz == t.iz) return;
    pendingTreeBreaks.push_back({t, impulse, at, severity});
}

// Imported tree (variant v >= trees::VARIANTS) broken at local height cutY: triangles are sorted by centroid into
// the stump (below the cut) and the trunk + crown above, which splits once more at mid height on a hard blow.
// Mass ~ a real hardwood of that height (~1 t for 10 m), shared by height.
static void assetBreak(int v, float cutY, float severity, std::vector<trees::Piece>& out) {
    const MeshData& m = treeModel.data.meshes[v];
    float y0 = m.bmin[1], H = std::max(m.bmax[1] - m.bmin[1], 0.5f);
    cutY = std::clamp(cutY, y0 + 0.25f, y0 + 0.5f * H);
    float cut2 = severity > 2.f ? cutY + 0.5f * (m.bmax[1] - cutY) : 1e30f;
    float total = 1000.f * powf(H / 10.f, 2.5f);
    auto piece = [&](float lo, float hi, int kind) {
        trees::Piece pc;
        pc.kind = kind;
        std::vector<uint32_t> remap(m.verts.size() / 8, UINT32_MAX);
        for (size_t t = 0; t * 3 + 2 < m.idx.size(); t++) {
            const uint32_t* tri = &m.idx[t * 3];
            float cy = (m.verts[tri[0] * 8 + 1] + m.verts[tri[1] * 8 + 1] + m.verts[tri[2] * 8 + 1]) / 3.f;
            if (cy < lo || cy >= hi) continue;
            for (int k = 0; k < 3; k++) {
                uint32_t& r = remap[tri[k]];
                if (r == UINT32_MAX) {
                    r = (uint32_t)pc.mesh.verts.size() / 8;
                    pc.mesh.verts.insert(pc.mesh.verts.end(), &m.verts[tri[k] * 8], &m.verts[tri[k] * 8 + 8]);
                }
                pc.mesh.idx.push_back(r);
            }
            pc.mesh.triMat.push_back(t < m.triMat.size() ? m.triMat[t] : 0);
        }
        pc.mass = total * (std::min(hi, m.bmax[1]) - std::max(lo, y0)) / H;
        if (!pc.mesh.idx.empty()) out.push_back(std::move(pc));
    };
    piece(-1e30f, cutY, trees::PIECE_STUMP);
    piece(cutY, cut2, trees::PIECE_LOG);
    if (cut2 < 1e30f) piece(cut2, 1e30f, trees::PIECE_LEAVES);
}

// Only the materials the pieces use, and their textures (an imported tree's are large), remapped into md.
void takeTreeMaterials(Model& md, std::vector<trees::Piece>& pcs) {
    std::map<uint32_t, uint32_t> mats;
    std::map<int, int> texs;
    for (auto& pc : pcs)
        for (uint32_t& tm : pc.mesh.triMat) {
            auto [it, fresh] = mats.try_emplace(tm, (uint32_t)md.data.materials.size());
            if (fresh) {
                MaterialData mt = treeModel.data.materials[tm];
                for (int* tx : {&mt.tex, &mt.normalTex, &mt.ormTex, &mt.occTex, &mt.emisTex}) {
                    if (*tx < 0) continue;
                    auto [jt, nt] = texs.try_emplace(*tx, (int)md.data.textures.size());
                    if (nt) md.data.textures.push_back(treeModel.data.textures[*tx]);
                    *tx = jt->second;
                }
                md.data.materials.push_back(mt);
            }
            tm = it->second;
        }
}

void breakTreeNow(const TreeInst& t, V3 impulse, V3 at, float severity) {
    addDecal(at, impulse * -1.f, 0.3f, 2);  // splintered wood where it was hit
    addDust(at, 0.8f, 0.25f, 0.2f);         // a snapping trunk throws bark dust
    uint64_t key = treeKey(t.ix, t.iz);
    if (!brokenTrees.insert(key).second) return;
    placeDirty = true;
    auto it = treeBodies.find(key);
    if (it != treeBodies.end()) { physics::removeBodies(&it->second, 1); treeBodies.erase(it); }
    std::vector<trees::Piece> pcs;
    std::vector<float> shed;  // leaves shaken loose (tree-local positions)
    assetBreak((int)t.v, (at.y - t.p.y) / t.scale, severity, pcs);

    auto fs = std::make_unique<FragSet>();
    Model& md = fs->model;
    md.path = "<tree pieces>";
    md.state = 1;
    takeTreeMaterials(md, pcs);
    float c = cosf(t.yaw), sn = sinf(t.yaw), s3 = t.scale * t.scale * t.scale;
    struct Spawn { size_t mesh; Xf xf; float mass, r; int kind; std::vector<float> col; std::vector<uint32_t> colIdx; };
    std::vector<Spawn> spawns;
    float total = 0.f;
    for (auto& pc : pcs) {
        MeshData& m = pc.mesh;
        if (m.idx.empty()) continue;
        // into metres, centred on its bounding box
        V3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
        for (size_t i = 0; i < m.verts.size(); i += 8) {
            for (int k = 0; k < 3; k++) m.verts[i + k] *= t.scale;
            V3 q{m.verts[i], m.verts[i + 1], m.verts[i + 2]};
            lo = {std::min(lo.x, q.x), std::min(lo.y, q.y), std::min(lo.z, q.z)};
            hi = {std::max(hi.x, q.x), std::max(hi.y, q.y), std::max(hi.z, q.z)};
        }
        V3 ctr = (lo + hi) * 0.5f;
        float r = 0.f;
        for (size_t i = 0; i < m.verts.size(); i += 8) {
            m.verts[i] -= ctr.x; m.verts[i + 1] -= ctr.y; m.verts[i + 2] -= ctr.z;
            r = std::max(r, m.verts[i] * m.verts[i] + m.verts[i + 1] * m.verts[i + 1] + m.verts[i + 2] * m.verts[i + 2]);
        }
        for (int k = 0; k < 3; k++) { m.bmin[k] = (&lo.x)[k] - (&ctr.x)[k]; m.bmax[k] = (&hi.x)[k] - (&ctr.x)[k]; }
        // same yaw as the placed instance ({c, 0, sn}, {0, 1, 0}, {-sn, 0, c})
        V3 cw = t.p + V3{c * ctr.x + sn * ctr.z, ctr.y, -sn * ctr.x + c * ctr.z};
        Xf px = {{{c, 0, sn, cw.x}, {0, 1, 0, cw.y}, {-sn, 0, c, cw.z}}};
        md.data.meshes.push_back(std::move(m));
        for (size_t i = 0; i < pc.colVerts.size(); i += 3)  // the standing tree's own collider, same frame as its mesh
            for (int k = 0; k < 3; k++) pc.colVerts[i + k] = pc.colVerts[i + k] * t.scale - (&ctr.x)[k];
        spawns.push_back({md.data.meshes.size() - 1, px, pc.mass * s3, sqrtf(r), pc.kind, std::move(pc.colVerts), std::move(pc.colIdx)});
        if (pc.kind != trees::PIECE_STUMP) total += pc.mass * s3;
    }
    if (spawns.empty()) return;
    uploadModel(md);
    FragSet* set = fs.get();
    fragSets.push_back(std::move(fs));
    V3 vBase = impulse * (0.5f / std::max(total, 1.f));
    float vb = sqrtf(dot(vBase, vBase));
    if (vb > 15.f) vBase = vBase * (15.f / vb);
    uint32_t hitBody = UINT32_MAX;
    float hitD = 1e30f;
    for (auto& sp : spawns) {
        const MeshData& mesh = set->model.data.meshes[sp.mesh];
        uint32_t nv = (uint32_t)mesh.verts.size() / 8;
        // hulls from at most ~4k points (an imported tree piece can have millions)
        std::vector<float> sub;
        const float* hp = mesh.verts.data();
        uint32_t hs = 8, hn = nv;
        if (nv > 4096) {
            for (uint32_t i = 0; i < nv; i += nv / 4096) sub.insert(sub.end(), &mesh.verts[(size_t)i * 8], &mesh.verts[(size_t)i * 8 + 3]);
            hp = sub.data(); hs = 3; hn = (uint32_t)sub.size() / 3;
        }
        void* shape = !sp.colIdx.empty() ? physics::meshShape(sp.col.data(), 3, (uint32_t)sp.col.size() / 3, sp.colIdx.data(), (uint32_t)sp.colIdx.size() / 3)
                                         : physics::convexShape(hp, hs, hn);
        if (!shape) continue;
        if (sp.kind == trees::PIECE_STUMP) {
            if (stumps.size() >= (size_t)MAX_FALLEN) dropStump(0);
            Stump st{set, (uint32_t)sp.mesh, 0, shape, sp.xf};
            physics::addStatic(&st.shape, &st.xf, 1, &st.body);
            set->live++;
            stumps.push_back(st);
            continue;
        }
        // the hull is mostly air around a crown: pick the density that gives the piece its real mass
        fracture::Poly hull = fracture::fromPoints(hp, hs, hn, 26);
        float dens = std::clamp(sp.mass / std::max(fracture::volume(hull), 1e-3f), 5.f, 900.f);
        V3 v = vBase + rndDir() * (0.8f * (std::min(severity, 6.f) - 1.f) * rnd01());
        V3 w = rndDir() * (0.4f * std::min(severity, 5.f) * rnd01());
        if (thrown.size() >= (size_t)MAX_THROWN) dropThrown(0);
        Thrown f;
        f.sc = 1.f;
        f.radius = sp.r;
        f.frag = set;
        f.fragMesh = (uint32_t)sp.mesh;
        f.shape = shape;
        f.body = physics::addDynamicPose(shape, sp.xf, &v.x, &w.x, dens);
        f.xf = sp.xf;
        f.cooldown = 0.4f;  // pieces settling off the stump aren't impacts
        f.density = dens;
        f.wood = true;
        memcpy(f.v, &v.x, 12);
        set->live++;
        V3 d = V3{sp.xf.m[0][3], sp.xf.m[1][3], sp.xf.m[2][3]} - at;
        if (dot(d, d) < hitD) { hitD = dot(d, d); hitBody = f.body; }
        thrown.push_back(std::move(f));
    }
    if (hitBody != UINT32_MAX) {
        V3 J = impulse * 0.5f;
        physics::addImpulse(hitBody, &J.x, &at.x);
    }
    if (set->live == 0) {
        auto keep = std::shared_ptr<FragSet>(fragSets.back().release());
        fragSets.pop_back();
        retireNext.push_back([keep]() { if (keep->model.gpu) releaseModel(keep->model); });
    }
    float sev = std::min(severity, 4.f);
    V3 cut{t.p.x, at.y, t.p.z};
    for (int i = 0; i < (int)(20.f * sev); i++)  // splinters from the snap
        addDebris(DEB_WOOD, cut + rndDir() * (0.3f * t.scale), rndDir() * (2.f + 4.f * rnd01()) + V3{0, 2.f, 0}, 0.04f + 0.08f * rnd01());
    V3 dir = normalize(impulse + V3{0, 1e-3f, 0});
    // leaves shaken out of the crown, from where they actually hung (at most ~600, picked evenly)
    size_t nShed = shed.size() / 3, stepL = std::max<size_t>(1, nShed / 600);
    for (size_t i = 0; i < nShed; i += stepL) {
        V3 l{shed[i * 3] * t.scale, shed[i * 3 + 1] * t.scale, shed[i * 3 + 2] * t.scale};
        V3 q = t.p + V3{c * l.x + sn * l.z, l.y, -sn * l.x + c * l.z};
        addDebris(DEB_LEAF, q, rndDir() * (0.5f + 1.5f * rnd01()) + dir * (1.5f * rnd01() * sev), 0.035f + 0.03f * rnd01(), 0.6f * rnd01());
    }
    if (t.v >= (uint32_t)trees::VARIANTS)
        for (int i = 0; i < (int)(60.f * sev); i++) {  // imported trees: no leaf positions, shake from the crown's bounds
            V3 q = t.p + V3{0, (4.f + 4.f * rnd01()) * t.scale, 0} + rndDir() * (2.5f * t.scale * sqrtf(rnd01()));
            addDebris(DEB_LEAF, q, rndDir() * (1.f + 2.f * rnd01()) + dir * (2.f * rnd01() * sev), 0.035f + 0.03f * rnd01(), 0.15f * rnd01());
        }
}
