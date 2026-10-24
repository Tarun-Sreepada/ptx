// Edited cells: imported from their streamed glb, rebuilt into a model and pushed live; undo; the selection.
#include "editor/editor_internal.h"

namespace editor {

// ---------------------------------------------------------------- cells
ECell importCell(int cx, int cz, const std::shared_ptr<Model>& live) {
    ECell c;
    c.cx = cx; c.cz = cz;
    c.live = live;
    if (!live) return c;
    const ModelData& d = live->data;
    c.meshes = d.meshes;
    for (auto& m : c.meshes) m.shape = nullptr;  // belong to the live model
    c.mats = d.materials;
    c.texs = d.textures;
    for (auto& in : d.instances) {
        Item it;
        it.name = in.name;
        it.collide = in.collide;
        decompose(in.xf, it.pos, it.rot, it.scale);
        const char* ex = in.extras.c_str();
        std::string prim;
        if (!in.asset.empty()) {
            it.kind = K_ASSET;
            it.asset = in.asset;
        } else if (extraStr(ex, "prim", prim)) {
            it.kind = K_PRIM;
            it.shape = prim == "cylinder" ? CYLINDER : BOX;
            it.size[0] = extraNum(ex, "sx", 1); it.size[1] = extraNum(ex, "sy", 1); it.size[2] = extraNum(ex, "sz", 1);
            it.sides = (int)extraNum(ex, "sides", 24);
            extraStr(ex, "mat", it.mat);
            it.uvScale = extraNum(ex, "uv", 1);
            it.tint[0] = extraNum(ex, "tr", 1); it.tint[1] = extraNum(ex, "tg", 1); it.tint[2] = extraNum(ex, "tb", 1);
            it.strength = extraNum(ex, "strength", 1);
            for (float& s : it.scale) s = 1.f;
        } else {
            it.kind = K_MESH;
            it.mesh = in.mesh;
            it.extras = in.extras;
            it.strength = in.mesh >= 0 ? d.meshes[in.mesh].strength : 1.f;
        }
        c.items.push_back(it);
    }
    for (auto& L : d.lights) {
        Item it;
        it.kind = K_LIGHT;
        memcpy(it.pos, L.pos, 12);
        it.intensity = std::max({L.color[0], L.color[1], L.color[2], 1e-6f});
        for (int k = 0; k < 3; k++) it.color[k] = L.color[k] / it.intensity;
        c.items.push_back(it);
    }
    for (auto& v : d.volumes) {
        Item it;
        it.kind = K_VOLUME;
        it.name = v.name;
        for (int k = 0; k < 3; k++) { it.pos[k] = (v.mn[k] + v.mx[k]) * 0.5f; it.scale[k] = (v.mx[k] - v.mn[k]) * 0.5f; }
        c.items.push_back(it);
    }
    for (auto& z : d.zones) {
        Item it;
        it.kind = K_ZONE;
        it.pos[0] = (z.minx + z.maxx) * 0.5f; it.pos[1] = z.height; it.pos[2] = (z.minz + z.maxz) * 0.5f;
        it.scale[0] = (z.maxx - z.minx) * 0.5f; it.scale[1] = 0.5f; it.scale[2] = (z.maxz - z.minz) * 0.5f;
        it.falloff = z.falloff;
        it.zoneMat = z.mat;
        it.zoneTile = z.tile;
        c.items.push_back(it);
    }
    for (auto& p : d.props) {
        Item it;
        it.kind = p.kind == PROP_TREE ? K_TREE : K_ROCK;
        it.variant = p.variant;
        decompose(p.xf, it.pos, it.rot, it.scale);
        c.items.push_back(it);
    }
    return c;
}

// cell (cx, cz) for editing: imported from the stream once loaded; a cell with no file starts empty
ECell* open(int cx, int cz) {
    auto it = cells.find({cx, cz});
    if (it != cells.end()) return &it->second;
    auto live = stream::cellModel(cx, cz);
    if (!live && std::filesystem::exists(cellPath(cx, cz))) return nullptr;  // still loading (or out of range)
    return &(cells[{cx, cz}] = importCell(cx, cz, live));
}

// The cell's items -> a model: the cell's own meshes still in use (materials and textures compacted), a mesh per
// primitive (materials per material + tint, the library texture once per name), instances, lights, volumes, zones,
// props. Each item's slot (instance / light / volume / zone / prop index) is recorded for in-place moves.
void build(ECell& c, ModelData& d, bool shapes) {
    d = {};
    std::map<int, int> meshMap, matMap, texMap;
    auto useTex = [&](int t) {
        if (t < 0 || t >= (int)c.texs.size()) return -1;
        auto [e, fresh] = texMap.emplace(t, (int)d.textures.size());
        if (fresh) d.textures.push_back(c.texs[t]);
        return e->second;
    };
    auto useMat = [&](uint32_t m) {
        auto [e, fresh] = matMap.emplace((int)m, (int)d.materials.size());
        if (fresh) {
            MaterialData md = m < c.mats.size() ? c.mats[m] : MaterialData{};
            forEachTex(md, useTex);
            d.materials.push_back(md);
        }
        return (uint32_t)e->second;
    };
    std::map<std::string, int> libTex, primMat;
    auto primMaterial = [&](const Item& it) {
        std::string key = it.mat + fmt("|%g|%g|%g", it.tint[0], it.tint[1], it.tint[2]);
        auto pm = primMat.find(key);
        if (pm != primMat.end()) return (uint32_t)pm->second;
        auto lt = libTex.find(it.mat);
        if (lt == libTex.end()) {  // albedo, normal + height, ORM: three consecutive textures
            lt = libTex.emplace(it.mat, (int)d.textures.size()).first;
            const matlib::Maps& mp = matlib::maps(it.mat);
            d.textures.push_back(matlib::texture(it.mat));
            d.textures.push_back(mp.normal);
            d.textures.push_back(mp.orm);
        }
        MaterialData md;
        memcpy(md.color, it.tint, 12);
        md.tex = lt->second;
        md.normalTex = lt->second + 1;
        md.ormTex = md.occTex = lt->second + 2;
        md.rough = md.metal = 1.f;  // factors x the ORM texture
        md.heightScale = matlib::maps(it.mat).heightScale;
        md.sss = sssProfileOf(it.mat);  // wax, marble
        primMat[key] = (int)d.materials.size();
        d.materials.push_back(md);
        return (uint32_t)d.materials.size() - 1;
    };
    for (auto& it : c.items) {
        it.slot = -1;
        switch (it.kind) {
        case K_PRIM: {
            MeshData m;
            genPrim(it, primMaterial(it), m);
            InstanceData in;
            in.mesh = (int)d.meshes.size();
            d.meshes.push_back(std::move(m));
            in.xf = itemXf(it);
            in.collide = it.collide;
            in.name = it.name;
            in.extras = primExtras(it);
            it.slot = (int)d.instances.size();
            d.instances.push_back(in);
            break;
        }
        case K_MESH: {
            if (it.mesh < 0 || it.mesh >= (int)c.meshes.size()) break;
            auto [e, fresh] = meshMap.emplace(it.mesh, (int)d.meshes.size());
            if (fresh) {
                MeshData m = c.meshes[it.mesh];
                for (auto& t : m.triMat) t = useMat(t);
                m.shape = nullptr;
                d.meshes.push_back(std::move(m));
            }
            d.meshes[e->second].strength = it.strength;
            InstanceData in;
            in.mesh = e->second;
            in.xf = itemXf(it);
            in.collide = it.collide;
            in.name = it.name;
            in.extras = it.strength != 1.f || !it.extras.empty() ? setKey(it.extras, "strength", it.strength) : std::string();
            if (!it.collide) in.extras = setKey(in.extras, "collide", 0);
            it.slot = (int)d.instances.size();
            d.instances.push_back(in);
            break;
        }
        case K_ASSET: {
            InstanceData in;
            in.asset = it.asset;
            in.xf = itemXf(it);
            in.collide = it.collide;
            in.name = it.name;
            it.slot = (int)d.instances.size();
            d.instances.push_back(in);
            break;
        }
        case K_LIGHT: {
            LightData L;
            memcpy(L.pos, it.pos, 12);
            for (int k = 0; k < 3; k++) L.color[k] = it.color[k] * it.intensity;
            it.slot = (int)d.lights.size();
            d.lights.push_back(L);
            break;
        }
        case K_VOLUME: {
            ProbeVolume v;
            for (int k = 0; k < 3; k++) { v.mn[k] = it.pos[k] - fabsf(it.scale[k]); v.mx[k] = it.pos[k] + fabsf(it.scale[k]); }
            v.name = it.name.empty() ? fmt("vol_%d_%d_%zu", c.cx, c.cz, d.volumes.size()) : it.name;
            it.slot = (int)d.volumes.size();
            d.volumes.push_back(v);
            break;
        }
        case K_ZONE: {
            Zone z{it.pos[0] - fabsf(it.scale[0]), it.pos[2] - fabsf(it.scale[2]), it.pos[0] + fabsf(it.scale[0]), it.pos[2] + fabsf(it.scale[2]), it.pos[1], it.falloff, it.zoneMat, it.zoneTile};
            it.slot = (int)d.zones.size();
            d.zones.push_back(z);
            break;
        }
        case K_TREE: case K_ROCK: {
            PropData p;
            p.kind = it.kind == K_TREE ? PROP_TREE : PROP_ROCK;
            p.variant = it.variant;
            float r[3] = {0, it.rot[1], 0}, s[3] = {it.scale[0], it.scale[0], it.scale[0]};
            p.xf = compose(it.pos, r, s);
            it.slot = (int)d.props.size();
            d.props.push_back(p);
            break;
        }
        default: break;
        }
    }
    if (d.materials.empty()) d.materials.push_back(MaterialData{});
    if (!shapes) return;
    std::vector<bool> coll(d.meshes.size(), false);
    for (auto& in : d.instances) if (in.mesh >= 0 && in.collide) coll[in.mesh] = true;
    for (size_t i = 0; i < d.meshes.size(); i++) {
        auto& m = d.meshes[i];
        if (coll[i] && !m.idx.empty())
            m.shape = physics::meshShape(m.verts.data(), 8, (uint32_t)(m.verts.size() / 8), m.idx.data(), (uint32_t)(m.idx.size() / 3));
    }
}

void freeModel(std::shared_ptr<Model> m) { host.retire(std::move(m)); }

bool pushLive(ECell& c) {
    auto m = std::make_shared<Model>();
    build(c, m->data, true);
    m->state = 1;
    for (auto& md : m->data.meshes) m->bytes += md.verts.size() * 4 + md.idx.size() * 8;
    if (!host.upload(*m)) {
        for (auto& md : m->data.meshes) if (md.shape) physics::releaseShape(md.shape);
        status = "upload failed (out of mesh / material / texture slots)";
        return false;
    }
    std::vector<std::shared_ptr<Model>> freed;
    stream::setCellModel(c.cx, c.cz, m, freed);
    for (auto& f : freed) freeModel(f);
    c.live = m;
    lastRebuild = now();
    return true;
}

// move what a slot points at in the live model, then re-place the cell
bool moveInPlace(ECell& c, const Item& it) {
    if (!c.live || it.slot < 0 || stream::cellModel(c.cx, c.cz) != c.live) return false;
    ModelData& d = c.live->data;
    switch (it.kind) {
    case K_PRIM: case K_MESH: case K_ASSET:
        if (it.slot >= (int)d.instances.size()) return false;
        d.instances[it.slot].xf = itemXf(it);
        break;
    case K_LIGHT:
        if (it.slot >= (int)d.lights.size()) return false;
        memcpy(d.lights[it.slot].pos, it.pos, 12);
        break;
    case K_VOLUME:
        if (it.slot >= (int)d.volumes.size()) return false;
        for (int k = 0; k < 3; k++) { d.volumes[it.slot].mn[k] = it.pos[k] - fabsf(it.scale[k]); d.volumes[it.slot].mx[k] = it.pos[k] + fabsf(it.scale[k]); }
        break;
    case K_ZONE: {
        if (it.slot >= (int)d.zones.size()) return false;
        Zone& z = d.zones[it.slot];
        z = {it.pos[0] - fabsf(it.scale[0]), it.pos[2] - fabsf(it.scale[2]), it.pos[0] + fabsf(it.scale[0]), it.pos[2] + fabsf(it.scale[2]), it.pos[1], it.falloff, it.zoneMat, it.zoneTile};
        break;
    }
    case K_TREE: case K_ROCK: {
        if (it.slot >= (int)d.props.size()) return false;
        float r[3] = {0, it.rot[1], 0}, s[3] = {it.scale[0], it.scale[0], it.scale[0]};
        d.props[it.slot].xf = compose(it.pos, r, s);
        break;
    }
    default: return false;
    }
    return true;
}

Item* selected() {
    if (sel.idx < 0) return nullptr;
    auto it = cells.find({sel.cx, sel.cz});
    if (it == cells.end() || sel.idx >= (int)it->second.items.size()) { sel.idx = -1; return nullptr; }
    return &it->second.items[sel.idx];
}

ECell* selCell() {
    auto it = cells.find({sel.cx, sel.cz});
    return it == cells.end() ? nullptr : &it->second;
}

void pushUndo(ECell& c) {
    c.undo.push_back(c.items);
    if (c.undo.size() > 64) c.undo.erase(c.undo.begin());
}

void changed(ECell& c, bool geometry) {
    c.modified = c.dirty = true;
    if (geometry) c.rebuild = true;
    else c.refresh = true;
}

// an item that moved out of its cell's square goes to the cell it's in now
void rehome(ECell& c, int idx) {
    Item& it = c.items[idx];
    int nx = cellOf(it.pos[0]), nz = cellOf(it.pos[2]);
    if (it.kind == K_MESH || (nx == c.cx && nz == c.cz)) return;  // imported meshes stay with their cell's mesh data
    ECell* d = open(nx, nz);
    if (!d) return;
    pushUndo(*d);
    d->items.push_back(it);
    c.items.erase(c.items.begin() + idx);
    changed(c, true);
    changed(*d, true);
    if (sel.cx == c.cx && sel.cz == c.cz && sel.idx == idx) sel = {nx, nz, (int)d->items.size() - 1};
}

}  // namespace editor
