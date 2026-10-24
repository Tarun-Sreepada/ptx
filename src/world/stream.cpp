// World streaming: cells and shared assets load on worker threads (parse, decode, physics shapes),
// the main thread uploads them (callback), then activates a cell once it and every asset it uses is resident.
// Terrain collision is streamed separately as heightfield tiles around the viewer, rebuilt when zones change.
#include <condition_variable>
#include <cstring>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "physics/physics.h"
#include "core/prof.h"
#include "world/texcomp.h"
#include "world/world.h"

namespace {
using Key = std::pair<int, int>;

struct Cell {
    std::shared_ptr<Model> own;
    std::vector<std::shared_ptr<Model>> deps;
    bool depsRequested = false, active = false, broken = false;
    std::vector<Placed> placed;
    std::vector<uint32_t> bodies;
    std::vector<WorldLight> lights;
    std::vector<Zone> zones;
    std::vector<ProbeVolume> volumes;
    std::vector<stream::WorldProp> props;
};

struct Tile {  // terrain collision
    int state = 0;  // 0 building, 1 ready
    uint32_t zoneVersion = 0;
    void* shape = nullptr;
    uint32_t body = 0xFFFFFFFF;
};

constexpr int TILE_N = 129;  // samples per side, 0.5 m spacing over one 64 m cell
constexpr int TILE_RADIUS = 1;

std::string worldDir;
std::set<Key> cellFiles;
std::map<Key, Cell> cells;
std::unordered_map<std::string, std::shared_ptr<Model>> models;
std::map<Key, std::shared_ptr<Tile>> tiles;
std::vector<WorldLight> allLights;
std::vector<Zone> allZones;
std::vector<ProbeVolume> allVolumes;
std::vector<stream::WorldProp> allProps;
bool editChanged = false;  // an editor call changed the placed set: next update() reports it
uint32_t zoneVersion = 1;
std::unordered_map<std::string, std::shared_ptr<Model>> overrides;  // stream::overridePlaced, by placedKey

std::mutex mtx;  // guards Model::state and Tile::state/shape
std::condition_variable cv;
std::deque<std::function<void()>> jobs;
std::vector<std::thread> workers;
bool quitting = false;

void enqueue(std::function<void()> f) {
    {
        std::lock_guard<std::mutex> l(mtx);
        jobs.push_back(std::move(f));
    }
    cv.notify_one();
}

void workerMain() {
    prof::setThreadName("stream loader");
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> l(mtx);
            cv.wait(l, [] { return quitting || !jobs.empty(); });
            if (quitting) return;
            job = std::move(jobs.front());
            jobs.pop_front();
        }
        PROF_ZONE("stream job");
        job();
    }
}

std::shared_ptr<Model> request(const std::string& path) {
    auto it = models.find(path);
    if (it != models.end()) { it->second->refs++; return it->second; }
    auto m = std::make_shared<Model>();
    m->path = path;
    m->refs = 1;
    models[path] = m;
    enqueue([m] {
        ModelData d;
        std::string err;
        bool ok;
        {
            PROF_ZONE("loadGltf");
            ok = loadGltf(m->path, d, err);
        }
        if (ok) {
            std::vector<bool> collide(d.meshes.size(), false);
            for (auto& in : d.instances)
                if (in.mesh >= 0 && in.collide) collide[in.mesh] = true;
            // assets: meshes collide unless every node using them says otherwise; unplaced meshes still get shapes
            for (size_t i = 0; i < d.meshes.size(); i++) {
                auto& md = d.meshes[i];
                bool used = false;
                for (auto& in : d.instances) used |= in.mesh == (int)i;
                if ((collide[i] || !used) && !md.idx.empty())
                    md.shape = physics::meshShape(md.verts.data(), 8, (uint32_t)(md.verts.size() / 8), md.idx.data(), (uint32_t)(md.idx.size() / 3));
            }
            size_t bytes = 0;
            for (auto& md : d.meshes) bytes += md.verts.size() * 4 + md.idx.size() * 8;
            for (auto& t : d.textures) {
                texcomp::prepare(t);  // BC7 + mips, from the texcache (encoded the first time)
                bytes += t.bc7.empty() ? t.rgba.size() * 4 / 3 : t.bc7.size();
            }
            m->bytes = bytes;
        } else {
            fprintf(stderr, "stream: %s: %s\n", m->path.c_str(), err.c_str());
        }
        std::lock_guard<std::mutex> l(mtx);
        m->data = std::move(d);
        m->state = ok ? 1 : 2;
    });
    return m;
}

int modelState(const Model& m) {
    std::lock_guard<std::mutex> l(mtx);
    return m.state;
}

void unref(std::shared_ptr<Model>& m) {
    if (m) m->refs--;
    m.reset();
}

void activate(Cell& c) {
    const ModelData& own = c.own->data;
    std::vector<void*> shapes;
    std::vector<Xf> xfs;
    auto addModel = [&](Model* m, const Xf& parent, bool collide) {
        for (auto& in : m->data.instances) {
            if (in.mesh < 0) continue;
            Xf w = xfMul(parent, in.xf);
            Placed pl{m, in.mesh, w};
            if (!overrides.empty()) {
                auto ov = overrides.find(stream::placedKey(pl));
                if (ov != overrides.end()) pl = {ov->second.get(), 0, w};
            }
            c.placed.push_back(pl);
            void* sh = pl.model->data.meshes[pl.mesh].shape;
            if (collide && in.collide && sh) { shapes.push_back(sh); xfs.push_back(w); }
        }
        for (auto& L : m->data.lights) {
            WorldLight wl;
            xfPoint(parent, L.pos, wl.pos);
            memcpy(wl.color, L.color, sizeof wl.color);
            c.lights.push_back(wl);
        }
    };
    addModel(c.own.get(), xfIdentity(), true);
    for (auto& in : own.instances) {
        if (in.mesh >= 0) continue;
        for (auto& d : c.deps)
            if (d->path == assetFile(worldDir, in.asset) && modelState(*d) == 1) addModel(d.get(), in.xf, in.collide);
    }
    c.zones = own.zones;
    c.volumes = own.volumes;
    for (auto& pd : own.props) c.props.push_back({pd.kind, pd.variant, pd.xf});
    c.bodies.resize(shapes.size());
    if (!shapes.empty()) physics::addStatic(shapes.data(), xfs.data(), shapes.size(), c.bodies.data());
    c.active = true;
}

void deactivate(Cell& c) {
    if (!c.bodies.empty()) physics::removeBodies(c.bodies.data(), c.bodies.size());
    c.bodies.clear();
    c.placed.clear();
    c.lights.clear();
    c.zones.clear();
    c.volumes.clear();
    c.props.clear();
    c.active = false;
}

void updateTiles(float x, float z) {
    int cx = (int)floorf(x / stream::CELL), cz = (int)floorf(z / stream::CELL);
    // drop far tiles (keep ones still building until their job finishes)
    for (auto it = tiles.begin(); it != tiles.end();) {
        auto& t = it->second;
        bool far = std::max(abs(it->first.first - cx), abs(it->first.second - cz)) > TILE_RADIUS + 1;
        std::lock_guard<std::mutex> l(mtx);
        if (far && t->state == 1) {
            if (t->body != 0xFFFFFFFF) physics::removeBodies(&t->body, 1);
            physics::releaseShape(t->shape);
            it = tiles.erase(it);
        } else ++it;
    }
    for (int dz = -TILE_RADIUS; dz <= TILE_RADIUS; dz++)
        for (int dx = -TILE_RADIUS; dx <= TILE_RADIUS; dx++) {
            Key k{cx + dx, cz + dz};
            auto& t = tiles[k];
            if (!t) t = std::make_shared<Tile>();
            std::unique_lock<std::mutex> l(mtx);
            if (t->state == 1 && t->shape && t->body == 0xFFFFFFFF) {  // freshly built: add body
                Xf id = xfIdentity();
                l.unlock();
                physics::addStatic(&t->shape, &id, 1, &t->body);
                l.lock();
            }
            if (t->zoneVersion != zoneVersion && (t->state == 1 || t->zoneVersion == 0)) {
                // (re)build with a snapshot of the current zones
                if (t->body != 0xFFFFFFFF) {
                    uint32_t b = t->body;
                    t->body = 0xFFFFFFFF;
                    l.unlock();
                    physics::removeBodies(&b, 1);
                    l.lock();
                }
                void* old = t->shape;
                t->shape = nullptr;
                t->state = 0;
                t->zoneVersion = zoneVersion;
                l.unlock();
                physics::releaseShape(old);
                std::vector<Zone> zs = allZones;
                enqueue([t, k, zs] {
                    std::vector<float> h(TILE_N * TILE_N);
                    float x0 = k.first * stream::CELL, z0 = k.second * stream::CELL;
                    terrainGrid(h.data(), TILE_N, x0, z0, stream::CELL / (TILE_N - 1), zs.data(), (int)zs.size());
                    void* s = physics::heightShape(h.data(), TILE_N, x0, z0, stream::CELL / (TILE_N - 1));
                    std::lock_guard<std::mutex> l2(mtx);
                    t->shape = s;
                    t->state = 1;
                });
            }
        }
}
}  // namespace

namespace stream {

void init(const std::string& dir, int threads) {
    worldDir = dir;
    std::error_code ec;
    for (auto& e : std::filesystem::directory_iterator(dir + "cells", ec)) {
        int x, z;
        if (e.path().extension() == ".glb" && sscanf(e.path().stem().c_str(), "%d_%d", &x, &z) == 2) cellFiles.insert({x, z});
    }
    printf("stream: %zu cells in %scells\n", cellFiles.size(), dir.c_str());
    for (int i = 0; i < threads; i++) workers.emplace_back(workerMain);
}

void shutdown() {
    {
        std::lock_guard<std::mutex> l(mtx);
        quitting = true;
    }
    cv.notify_all();
    for (auto& w : workers) w.join();
    workers.clear();
    for (auto& [k, c] : cells) deactivate(c);
    cells.clear();
    for (auto& [k, t] : tiles) {
        if (t->body != 0xFFFFFFFF) physics::removeBodies(&t->body, 1);
        physics::releaseShape(t->shape);
    }
    tiles.clear();
    for (auto& [p, m] : models)
        for (auto& md : m->data.meshes) physics::releaseShape(md.shape);
    models.clear();
}

bool update(float x, float y, float z, int radius, const std::function<bool(Model&)>& upload, const std::function<void(Model&)>& release) {
    (void)y;
    bool changed = false;
    int cx = (int)floorf(x / CELL), cz = (int)floorf(z / CELL);
    auto dist = [&](const Key& k) { return std::max(abs(k.first - cx), abs(k.second - cz)); };

    // request missing cells nearest-first
    std::vector<Key> want;
    for (auto& k : cellFiles)
        if (dist(k) <= radius && !cells.count(k)) want.push_back(k);
    std::sort(want.begin(), want.end(), [&](const Key& a, const Key& b) {
        float da = hypotf(a.first + 0.5f - x / CELL, a.second + 0.5f - z / CELL), db = hypotf(b.first + 0.5f - x / CELL, b.second + 0.5f - z / CELL);
        return da < db;
    });
    for (auto& k : want) cells[k].own = request(worldDir + "cells/" + std::to_string(k.first) + "_" + std::to_string(k.second) + ".glb");

    bool budget = true;
    for (auto it = cells.begin(); it != cells.end();) {
        Cell& c = it->second;
        if (dist(it->first) > radius + 1) {  // hysteresis: unload one ring further out than we load
            if (c.active) { deactivate(c); changed = true; }
            unref(c.own);
            for (auto& d : c.deps) unref(d);
            it = cells.erase(it);
            continue;
        }
        ++it;
        if (c.active || c.broken) continue;
        int st = modelState(*c.own);
        if (st == 2) { c.broken = true; continue; }
        if (st != 1) continue;
        if (!c.depsRequested) {
            std::set<std::string> names;
            for (auto& in : c.own->data.instances)
                if (in.mesh < 0) names.insert(in.asset);
            for (auto& n : names) c.deps.push_back(request(assetFile(worldDir, n)));
            c.depsRequested = true;
        }
        bool ready = true;
        for (auto& d : c.deps) ready &= modelState(*d) != 0;
        if (!ready) continue;
        // upload own + deps (failed assets are skipped)
        auto up = [&](Model& m) {
            if (m.gpu || modelState(m) != 1) return true;
            if (!budget) return false;
            budget = upload(m);
            return m.gpu != nullptr;
        };
        bool all = up(*c.own);
        for (auto& d : c.deps) all &= up(*d);
        if (!all) continue;
        activate(c);
        changed = true;
    }

    // free models nobody references (ones still loading are freed once their job lands)
    for (auto it = models.begin(); it != models.end();) {
        Model& m = *it->second;
        if (m.refs <= 0 && modelState(m) != 0) {
            if (m.gpu) release(m);
            for (auto& md : m.data.meshes) physics::releaseShape(md.shape);
            it = models.erase(it);
        } else ++it;
    }

    changed |= editChanged;
    editChanged = false;
    if (changed) {
        allProps.clear();
        for (auto& [k, c] : cells) allProps.insert(allProps.end(), c.props.begin(), c.props.end());
        allLights.clear();
        allZones.clear();
        allVolumes.clear();
        for (auto& [k, c] : cells) {
            allLights.insert(allLights.end(), c.lights.begin(), c.lights.end());
            allZones.insert(allZones.end(), c.zones.begin(), c.zones.end());
            allVolumes.insert(allVolumes.end(), c.volumes.begin(), c.volumes.end());
        }
        zoneVersion++;
    }
    updateTiles(x, z);
    return changed;
}

void forEachPlaced(const std::function<void(const Placed&)>& f) {
    for (auto& [k, c] : cells)
        for (auto& p : c.placed) f(p);
}
void terrainChanged() { zoneVersion++; }

std::string placedKey(const Placed& p) {
    if (p.model->path.rfind("#dmg ", 0) == 0) return p.model->path;  // already a damaged copy
    char b[96];
    snprintf(b, sizeof b, "#%d@%.2f,%.2f,%.2f", p.mesh, p.xf.m[0][3], p.xf.m[1][3], p.xf.m[2][3]);
    return "#dmg " + p.model->path + b;
}

std::shared_ptr<Model> overridePlaced(const std::string& key, std::shared_ptr<Model> with) {
    with->path = key;
    std::shared_ptr<Model> old;
    auto it = overrides.find(key);
    if (it != overrides.end()) old = it->second;
    overrides[key] = with;
    // re-place (and re-collide) every active cell that shows it
    for (auto& [k, c] : cells) {
        if (!c.active) continue;
        bool has = false;
        for (auto& p : c.placed) has |= placedKey(p) == key;
        if (has) { deactivate(c); activate(c); }
    }
    return old;
}

std::shared_ptr<Model> cellModel(int cx, int cz) {
    auto it = cells.find({cx, cz});
    if (it == cells.end() || !it->second.own || modelState(*it->second.own) != 1) return nullptr;
    return it->second.own;
}

void setCellModel(int cx, int cz, std::shared_ptr<Model> m, std::vector<std::shared_ptr<Model>>& freed) {
    Key k{cx, cz};
    cellFiles.insert(k);
    Cell& c = cells[k];
    if (c.active) deactivate(c);
    std::string path = worldDir + "cells/" + std::to_string(cx) + "_" + std::to_string(cz) + ".glb";
    // damage stays on a mesh the rebuild didn't change (same slot, place and geometry): walls don't heal when
    // the editor rebuilds the cell; edited or moved meshes lose theirs
    std::shared_ptr<Model> old = c.own;
    std::unordered_set<std::string> keep;
    if (old && !overrides.empty()) {
        m->path = path;
        for (auto& in : m->data.instances) {
            if (in.mesh < 0) continue;
            Placed pl{m.get(), in.mesh, in.xf};
            std::string key = stream::placedKey(pl);
            if (!overrides.count(key)) continue;
            for (auto& oin : old->data.instances) {
                if (oin.mesh < 0 || (size_t)oin.mesh >= old->data.meshes.size()) continue;
                if (stream::placedKey(Placed{old.get(), oin.mesh, oin.xf}) != key) continue;
                const MeshData &a = old->data.meshes[oin.mesh], &b = m->data.meshes[in.mesh];
                if (a.verts == b.verts && a.idx == b.idx) keep.insert(key);
            }
        }
    }
    if (c.own && c.own != m) {
        auto it = models.find(c.own->path);
        if (it != models.end() && it->second == c.own) models.erase(it);
        freed.push_back(c.own);
    }
    for (auto& d : c.deps) unref(d);
    c.deps.clear();
    c.depsRequested = false;
    c.broken = false;
    m->path = path;
    m->refs = 1;
    m->state = 1;
    models[path] = m;
    c.own = m;
    std::string pre = "#dmg " + path + "#";
    for (auto it = overrides.begin(); it != overrides.end();)
        if (it->first.rfind(pre, 0) == 0 && !keep.count(it->first)) { freed.push_back(it->second); it = overrides.erase(it); } else ++it;
    editChanged = true;
}

void refreshCell(int cx, int cz) {
    auto it = cells.find({cx, cz});
    if (it == cells.end() || !it->second.active) return;
    deactivate(it->second);
    activate(it->second);
    editChanged = true;
}

Model* findModel(const std::string& path) {
    auto it = models.find(path);
    return it != models.end() && modelState(*it->second) == 1 ? it->second.get() : nullptr;
}

const std::vector<WorldProp>& props() { return allProps; }

void clearOverrides(std::vector<std::shared_ptr<Model>>& freed) {
    if (overrides.empty()) return;
    for (auto& [k, m] : overrides) freed.push_back(m);
    overrides.clear();
    for (auto& [k, c] : cells)
        if (c.active) { deactivate(c); activate(c); }
}
const std::vector<WorldLight>& lights() { return allLights; }
const std::vector<Zone>& zones() { return allZones; }
const std::vector<ProbeVolume>& volumes() { return allVolumes; }

Stats stats() {
    Stats s{};
    s.cellsKnown = (int)cellFiles.size();
    s.cellsLoaded = (int)cells.size();
    for (auto& [k, c] : cells) s.cellsActive += c.active;
    s.models = (int)models.size();
    for (auto& [p, m] : models) s.loading += modelState(*m) == 0;
    s.terrainTiles = (int)tiles.size();
    return s;
}

}  // namespace stream
