// Asset library: importing glTF / glb into world/assets (simplified), thumbnails, the background worker.
#include "editor/editor_internal.h"

namespace editor {

// Simplify every mesh to ~keep of its triangles, per material run (so triMat stays valid), then drop unused vertices.
// Fewer triangles = faster hulls / cutting when a tree breaks, smaller BLAS, quicker loads.
void simplify(ModelData& m, float keep) {
    for (MeshData& md : m.meshes) {
        size_t nv = md.verts.size() / 8;
        if (!nv || md.idx.size() < 300) continue;
        std::vector<uint32_t> idx, mats;
        for (size_t t0 = 0; t0 < md.triMat.size();) {
            size_t t1 = t0;
            while (t1 < md.triMat.size() && md.triMat[t1] == md.triMat[t0]) t1++;
            size_t n = (t1 - t0) * 3, target = std::max<size_t>(3, (size_t)(n * keep) / 3 * 3);
            std::vector<uint32_t> out(n);
            float err = 0;
            size_t got = meshopt_simplify(out.data(), &md.idx[t0 * 3], n, md.verts.data(), nv, 8 * sizeof(float), target, 0.08f, 0, &err);
            if (got > target * 1.3f)  // leaf cards and other disconnected bits barely simplify: collapse them spatially
                got = meshopt_simplifySloppy(out.data(), &md.idx[t0 * 3], n, md.verts.data(), nv, 8 * sizeof(float), target, 0.08f, &err);
            idx.insert(idx.end(), out.begin(), out.begin() + got);
            mats.insert(mats.end(), got / 3, md.triMat[t0]);
            t0 = t1;
        }
        std::vector<float> verts(md.verts.size());
        size_t used = meshopt_optimizeVertexFetch(verts.data(), idx.data(), idx.size(), md.verts.data(), nv, 8 * sizeof(float));
        verts.resize(used * 8);
        md.verts.swap(verts);
        md.idx.swap(idx);
        md.triMat.swap(mats);
    }
}

size_t triCount(const ModelData& m) { size_t n = 0; for (auto& md : m.meshes) n += md.idx.size() / 3; return n; }

std::string lower(std::string s) { for (auto& ch : s) ch = (char)tolower((unsigned char)ch); return s; }

std::string assetName(const std::filesystem::path& p) {
    std::string n = p.stem().string();
    for (auto& ch : n) if (!isalnum((unsigned char)ch) && ch != '_' && ch != '-') ch = '_';
    return n.empty() ? "asset" : n;
}

// Software-rasterized 3/4 view (lambert, base colour texture, alpha cutout), 2x supersampled, transparent background.
std::vector<uint8_t> renderThumb(const ModelData& m) {
    const int S = THUMB * 2;
    const float yaw = 35 * DEG, pitch = 22 * DEG;
    float fwd[3] = {-sinf(yaw) * cosf(pitch), -sinf(pitch), -cosf(yaw) * cosf(pitch)};
    float rt[3] = {cosf(yaw), 0, -sinf(yaw)};
    float up[3] = {rt[1] * fwd[2] - rt[2] * fwd[1], rt[2] * fwd[0] - rt[0] * fwd[2], rt[0] * fwd[1] - rt[1] * fwd[0]};
    auto dot = [](const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    auto xfp = [](const Xf& x, const float* p, float* o) {
        for (int i = 0; i < 3; i++) o[i] = x.m[i][0] * p[0] + x.m[i][1] * p[1] + x.m[i][2] * p[2] + x.m[i][3];
    };
    // fit: the projected extent of every vertex
    float lo[2] = {1e30f, 1e30f}, hi[2] = {-1e30f, -1e30f};
    for (auto& in : m.instances) {
        if (in.mesh < 0 || in.mesh >= (int)m.meshes.size()) continue;
        const auto& v = m.meshes[in.mesh].verts;
        for (size_t i = 0; i + 8 <= v.size(); i += 8) {
            float w[3];
            xfp(in.xf, &v[i], w);
            float sx = dot(w, rt), sy = dot(w, up);
            lo[0] = std::min(lo[0], sx); hi[0] = std::max(hi[0], sx);
            lo[1] = std::min(lo[1], sy); hi[1] = std::max(hi[1], sy);
        }
    }
    if (lo[0] > hi[0]) return {};
    float ext = std::max(hi[0] - lo[0], hi[1] - lo[1]) * 1.08f + 1e-6f, cx = 0.5f * (lo[0] + hi[0]), cy = 0.5f * (lo[1] + hi[1]);
    std::vector<float> zb((size_t)S * S, 1e30f), col((size_t)S * S * 3, 0.f);
    std::vector<uint8_t> cov((size_t)S * S, 0);
    float L[3] = {0.45f, 0.8f, 0.35f};
    float ll = sqrtf(dot(L, L));
    for (float& c : L) c /= ll;
    for (auto& in : m.instances) {
        if (in.mesh < 0 || in.mesh >= (int)m.meshes.size()) continue;
        const MeshData& md = m.meshes[in.mesh];
        for (size_t t = 0; t * 3 + 2 < md.idx.size(); t++) {
            float w[3][3], P[3][3], uv[3][2];
            for (int k = 0; k < 3; k++) {
                const float* v = &md.verts[(size_t)md.idx[t * 3 + k] * 8];
                xfp(in.xf, v, w[k]);
                P[k][0] = ((dot(w[k], rt) - cx) / ext + 0.5f) * S;
                P[k][1] = (0.5f - (dot(w[k], up) - cy) / ext) * S;
                P[k][2] = dot(w[k], fwd);
                uv[k][0] = v[6]; uv[k][1] = v[7];
            }
            float area = (P[1][0] - P[0][0]) * (P[2][1] - P[0][1]) - (P[2][0] - P[0][0]) * (P[1][1] - P[0][1]);
            if (fabsf(area) < 1e-9f) continue;
            float e1[3], e2[3], n[3];
            for (int k = 0; k < 3; k++) { e1[k] = w[1][k] - w[0][k]; e2[k] = w[2][k] - w[0][k]; }
            n[0] = e1[1] * e2[2] - e1[2] * e2[1]; n[1] = e1[2] * e2[0] - e1[0] * e2[2]; n[2] = e1[0] * e2[1] - e1[1] * e2[0];
            float nl = sqrtf(dot(n, n));
            if (nl < 1e-20f) continue;
            for (float& c : n) c /= nl;
            if (dot(n, fwd) > 0) for (float& c : n) c = -c;  // two-sided (leaves)
            float shade = 0.35f + 0.65f * std::max(0.f, dot(n, L));
            int mi = t < md.triMat.size() ? (int)md.triMat[t] : -1;
            const MaterialData* mt = mi >= 0 && mi < (int)m.materials.size() ? &m.materials[mi] : nullptr;
            const TextureData* tx = mt && mt->tex >= 0 && mt->tex < (int)m.textures.size() && !m.textures[mt->tex].rgba.empty() ? &m.textures[mt->tex] : nullptr;
            int x0 = std::max(0, (int)floorf(std::min({P[0][0], P[1][0], P[2][0]}))), x1 = std::min(S - 1, (int)ceilf(std::max({P[0][0], P[1][0], P[2][0]})));
            int y0 = std::max(0, (int)floorf(std::min({P[0][1], P[1][1], P[2][1]}))), y1 = std::min(S - 1, (int)ceilf(std::max({P[0][1], P[1][1], P[2][1]})));
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) {
                    float px = x + 0.5f, py = y + 0.5f;
                    float b0 = ((P[1][0] - px) * (P[2][1] - py) - (P[2][0] - px) * (P[1][1] - py)) / area;
                    float b1 = ((P[2][0] - px) * (P[0][1] - py) - (P[0][0] - px) * (P[2][1] - py)) / area;
                    float b2 = 1.f - b0 - b1;
                    if (b0 < 0 || b1 < 0 || b2 < 0) continue;
                    float z = b0 * P[0][2] + b1 * P[1][2] + b2 * P[2][2];
                    size_t o = (size_t)y * S + x;
                    if (z >= zb[o]) continue;
                    float c[4] = {1, 1, 1, 1};
                    if (mt) for (int k = 0; k < 4; k++) c[k] = mt->color[k];
                    if (tx) {
                        float u = b0 * uv[0][0] + b1 * uv[1][0] + b2 * uv[2][0], v = b0 * uv[0][1] + b1 * uv[1][1] + b2 * uv[2][1];
                        u -= floorf(u); v -= floorf(v);
                        const uint8_t* s = &tx->rgba[((size_t)std::min(tx->h - 1, (uint32_t)(v * tx->h)) * tx->w + std::min(tx->w - 1, (uint32_t)(u * tx->w))) * 4];
                        for (int k = 0; k < 4; k++) c[k] *= s[k] / 255.f;
                    }
                    if (c[3] < 0.5f) continue;  // cutout
                    zb[o] = z;
                    cov[o] = 1;
                    for (int k = 0; k < 3; k++) col[o * 3 + k] = std::min(1.f, c[k] * shade);
                }
        }
    }
    std::vector<uint8_t> out((size_t)THUMB * THUMB * 4);
    for (int y = 0; y < THUMB; y++)
        for (int x = 0; x < THUMB; x++) {
            float c[3] = {}, a = 0;
            for (int j = 0; j < 2; j++)
                for (int i = 0; i < 2; i++) {
                    size_t o = (size_t)(y * 2 + j) * S + x * 2 + i;
                    if (!cov[o]) continue;
                    a += 0.25f;
                    for (int k = 0; k < 3; k++) c[k] += col[o * 3 + k];
                }
            uint8_t* d = &out[((size_t)y * THUMB + x) * 4];
            for (int k = 0; k < 3; k++) d[k] = (uint8_t)(a > 0 ? std::min(255.f, c[k] / (a * 4) * 255.f + 0.5f) : 0);
            d[3] = (uint8_t)(a * 255.f + 0.5f);
        }
    return out;
}

std::string thumbPath(const std::string& name) { return host.worldDir + "assets/.thumbs/" + name + ".png"; }

// worker side: make (or load the cached) thumbnail of assets/<name>.glb
// footprint radius in xz about the model origin (scatter spacing)
float footprint(const ModelData& m) {
    float r = 0;
    for (auto& in : m.instances) {
        if (in.mesh < 0 || in.mesh >= (int)m.meshes.size()) continue;
        float a[3], b[3];
        xfAabb(in.xf, m.meshes[in.mesh].bmin, m.meshes[in.mesh].bmax, a, b);
        r = std::max({r, -a[0], b[0], -a[2], b[2]});
    }
    return r;
}

std::vector<uint8_t> makeThumb(const std::string& name, const ModelData* loaded, float& foot) {
    std::error_code ec;
    std::string glb = assetFile(host.worldDir, name), tp = thumbPath(name);
    if (!loaded && std::filesystem::exists(tp, ec) && std::filesystem::last_write_time(tp, ec) >= std::filesystem::last_write_time(glb, ec)) {
        if (FILE* f = fopen((tp + ".txt").c_str(), "r")) { if (fscanf(f, "%f", &foot) != 1) foot = 0; fclose(f); }
        int w, h, ch;
        if (stbi_uc* px = stbi_load(tp.c_str(), &w, &h, &ch, 4)) {
            std::vector<uint8_t> r;
            if (w == THUMB && h == THUMB) r.assign(px, px + (size_t)w * h * 4);
            stbi_image_free(px);
            if (!r.empty() && foot > 0) return r;
        }
    }
    ModelData m;
    std::string err;
    if (!loaded) { if (!loadGltf(glb, m, err)) return {}; loaded = &m; }
    std::vector<uint8_t> px = renderThumb(*loaded);
    foot = footprint(*loaded);
    if (!px.empty()) {
        std::filesystem::create_directories(std::filesystem::path(tp).parent_path(), ec);
        stbi_write_png(tp.c_str(), THUMB, THUMB, 4, px.data(), THUMB * 4);
        if (FILE* f = fopen((tp + ".txt").c_str(), "w")) { fprintf(f, "%g\n", foot); fclose(f); }
    }
    return px;
}

// glTF materials -> PBR sets in <world>/textures/<name>[_i]/: albedo.png, normal.png, orm.png (occlusion r,
// roughness g, metalness b), usable on primitives and flatten zones. Returns how many were written.
int extractMaterials(const ModelData& m, const std::string& name, std::string& names) {
    int n = 0;
    std::error_code ec;
    for (size_t i = 0; i < m.materials.size(); i++) {
        const MaterialData& mt = m.materials[i];
        auto tex = [&](int k) -> const TextureData* { return k >= 0 && k < (int)m.textures.size() && !m.textures[k].rgba.empty() ? &m.textures[k] : nullptr; };
        const TextureData* alb = tex(mt.tex);
        if (!alb) continue;
        std::string set = m.materials.size() == 1 ? name : fmt("%s_%zu", name.c_str(), i);
        std::string dir = host.worldDir + "textures/" + set + "/";
        std::filesystem::create_directories(dir, ec);
        stbi_write_png((dir + "albedo.png").c_str(), alb->w, alb->h, 4, alb->rgba.data(), alb->w * 4);
        if (const TextureData* nt = tex(mt.normalTex)) stbi_write_png((dir + "normal.png").c_str(), nt->w, nt->h, 4, nt->rgba.data(), nt->w * 4);
        else std::filesystem::remove(dir + "normal.png", ec);
        const TextureData *mr = tex(mt.ormTex), *oc = tex(mt.occTex);
        if (mr || oc) {
            const TextureData* b = mr ? mr : oc;
            std::vector<uint8_t> px((size_t)b->w * b->h * 4);
            for (size_t p = 0; p < (size_t)b->w * b->h; p++) {
                px[p * 4 + 0] = oc && oc->w == b->w && oc->h == b->h ? oc->rgba[p * 4] : 255;
                px[p * 4 + 1] = mr ? mr->rgba[p * 4 + 1] : (uint8_t)(mt.rough * 255.f);
                px[p * 4 + 2] = mr ? mr->rgba[p * 4 + 2] : (uint8_t)(mt.metal * 255.f);
                px[p * 4 + 3] = 255;
            }
            stbi_write_png((dir + "orm.png").c_str(), b->w, b->h, 4, px.data(), b->w * 4);
        } else std::filesystem::remove(dir + "orm.png", ec);
        names += (n ? ", set:" : "set:") + set;
        n++;
    }
    return n;
}

Done runJob(const Job& j) {
    Done d;
    if (!j.import) { d.name = j.path; d.rgba = makeThumb(j.path, nullptr, d.foot); return d; }
    std::filesystem::path src(j.path);
    std::string ext = lower(src.extension().string());
    std::error_code ec;
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        std::filesystem::create_directories(host.worldDir + "textures", ec);
        std::filesystem::copy_file(src, host.worldDir + "textures/" + src.filename().string(), std::filesystem::copy_options::overwrite_existing, ec);
        d.msg = ec ? "texture import failed: " + ec.message() : "texture " + src.filename().string() + " added";
        d.rescan = !ec;
        return d;
    }
    // .gltf (with its .bin + textures) or .glb -> one self-contained assets/<name>.glb
    d.name = j.as.empty() ? assetName(src) : j.as;
    ModelData m;
    std::string err, dst = host.worldDir + "assets/" + d.name + ".glb";
    std::filesystem::create_directories(host.worldDir + "assets", ec);
    if (!loadGltf(j.path, m, err)) { d.msg = "import " + src.filename().string() + " failed: " + err; d.name.clear(); return d; }
    if (j.materials) {  // only its materials, into the texture library
        std::string names;
        int n = extractMaterials(m, d.name, names);
        d.msg = n ? "materials added: " + names : "no textured materials in " + src.filename().string();
        d.rescan = n > 0;
        d.name.clear();
        return d;
    }
    // a texture pack (a plane or swatch carrying a material): the material is what's wanted
    if (triCount(m) <= 2048 && m.materials.size() <= 2) {
        std::string names;
        if (extractMaterials(m, d.name, names)) {
            d.msg = "texture pack: materials added (" + names + "); pick them under Material / Ground texture";
            d.rescan = true;
            d.name.clear();
            return d;
        }
    }
    size_t tris = triCount(m);
    if (j.keep < 0.999f) simplify(m, j.keep);
    bool ok = ext == ".glb" && j.keep >= 0.999f ? (std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing, ec), !ec)
                                                : writeGlb(dst, m, err);
    if (!ok) { d.msg = "import " + src.filename().string() + " failed: " + (ec ? ec.message() : err); d.name.clear(); return d; }
    d.rgba = makeThumb(d.name, &m, d.foot);
    d.msg = j.keep < 0.999f ? fmt("imported %s: %zu -> %zu triangles", d.name.c_str(), tris, triCount(m)) : fmt("imported %s (%zu triangles)", d.name.c_str(), tris);
    return d;
}

void queueJob(Job j) {
    std::lock_guard lk(jobMu);
    jobQ.push_back(std::move(j));
    jobsLeft++;
    if (!workerUp) {
        workerUp = true;
        std::thread([] {
            for (;;) {
                Job j;
                {
                    std::unique_lock lk(jobMu);
                    jobCv.wait(lk, [] { return !jobQ.empty(); });
                    j = jobQ.front();
                    jobQ.pop_front();
                }
                Done d = runJob(j);
                std::lock_guard lk(jobMu);
                jobDone.push_back(std::move(d));
                jobsLeft--;
            }
        }).detach();
    }
    jobCv.notify_one();
}

void scanAssets() {
    assetNames.clear();
    assetDirs.clear();
    std::error_code ec;
    // every .glb / .gltf under assets/, nested folders too (a .gltf is used in place with its .bin and textures).
    // Names are paths relative to assets/; top-level .glb files keep their bare name (what older cells store).
    const std::filesystem::path root = host.worldDir + "assets/";
    for (auto it = std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::follow_directory_symlink, ec);
         it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        const auto& p = it->path();
        if (it->is_directory(ec)) {
            if (p.filename().string().rfind('.', 0) == 0) { it.disable_recursion_pending(); continue; }  // .thumbs
            assetDirs.push_back(std::filesystem::relative(p, root, ec).generic_string());
            continue;
        }
        std::string e = lower(p.extension().string());
        if (e != ".glb" && e != ".gltf") continue;
        std::string rel = std::filesystem::relative(p, root, ec).generic_string();
        if (e == ".glb" && p.parent_path() == root.parent_path()) rel = p.stem().string();
        assetNames.push_back(rel);
    }
    std::sort(assetNames.begin(), assetNames.end());
    std::sort(assetDirs.begin(), assetDirs.end());
    if (!host.thumbnail) return;
    for (auto& n : assetNames)
        if (Thumb& t = thumbs[n]; !t.tex && !t.queued) { t.queued = true; queueJob({false, n}); }
}

void pollJobs() {
    std::vector<Done> done;
    std::vector<std::string> picked;
    { std::lock_guard lk(jobMu); done.swap(jobDone); picked.swap(pickedPaths); }
    if (!picked.empty()) importPaths(picked);
    bool rescan = false;
    for (auto& d : done) {
        if (!d.msg.empty()) status = d.msg;
        if (d.rescan) matlib::rescan();
        if (d.name.empty()) continue;
        rescan |= !std::count(assetNames.begin(), assetNames.end(), d.name);
        Thumb& t = thumbs[d.name];
        t.queued = true;  // made (or failed): don't retry every scan
        t.foot = d.foot;
        if (!d.rgba.empty() && host.thumbnail) t.tex = host.thumbnail(d.rgba.data(), THUMB, THUMB);
    }
    if (rescan) scanAssets();
}

void importPaths(const std::vector<std::string>& paths) {
    int n = 0;
    std::error_code ec;
    auto take = [&](const std::filesystem::path& p) {
        std::string e = lower(p.extension().string());
        if (e == ".gltf" || e == ".glb" || e == ".png" || e == ".jpg" || e == ".jpeg") { queueJob({true, p.string(), importKeep}); n++; }
    };
    for (auto& s : paths) {
        std::filesystem::path p(s);
        if (std::filesystem::is_directory(p, ec)) {
            // a folder: its models (a model's own texture folder is not imported as world textures)
            for (auto it = std::filesystem::recursive_directory_iterator(p, ec); it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
                std::string e = lower(it->path().extension().string());
                if (e == ".gltf" || e == ".glb") take(it->path());
            }
        } else take(p);
    }
    status = n ? fmt("importing %d file%s...", n, n == 1 ? "" : "s") : "nothing to import (drop .gltf / .glb files, folders or PNG / JPG)";
}

// the point under the mouse (items, then terrain), false when it looks at the sky
bool mousePoint(float out[3]) {
    float d[3];
    mouseRay(d);
    float t = 1e30f;
    Hit h = pick(view.pos, d);
    if (h.idx >= 0) t = h.t;
    float tt = rayTerrain(view.pos, d);
    if (tt >= 0) t = std::min(t, tt);
    if (t > 400.f) return false;
    for (int k = 0; k < 3; k++) out[k] = view.pos[k] + d[k] * t;
    return true;
}

void addAsset(const std::string& name, const float* at) {
    Item it;
    it.kind = K_ASSET;
    it.asset = it.name = name;
    if (!at) { addItem(it); return; }
    it.pos[0] = snapv(at[0], grid); it.pos[1] = at[1]; it.pos[2] = snapv(at[2], grid);
    it.rot[1] = camYawSnapped();
    addItem(it, false);
}

void openPicker(bool folders) {
    if (pickerOpen.exchange(true)) return;
    std::thread([folders] {
        std::string cmd = folders
            ? "kdialog --title 'Import asset folder' --getexistingdirectory ~ 2>/dev/null || zenity --file-selection --directory --title='Import asset folder' 2>/dev/null"
            : "kdialog --title 'Import assets' --multiple --separate-output --getopenfilename ~ 'Models and textures (*.gltf *.glb *.png *.jpg *.jpeg)' 2>/dev/null"
              " || zenity --file-selection --multiple --separator='\n' --title='Import assets' --file-filter='Models and textures | *.gltf *.glb *.png *.jpg *.jpeg' 2>/dev/null";
        std::vector<std::string> paths;
        if (FILE* f = popen(cmd.c_str(), "r")) {
            char line[4096];
            while (fgets(line, sizeof line, f)) {
                std::string l(line);
                while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
                if (!l.empty()) paths.push_back(l);
            }
            pclose(f);
        }
        if (!paths.empty()) { std::lock_guard lk(jobMu); pickedPaths.insert(pickedPaths.end(), paths.begin(), paths.end()); }
        pickerOpen = false;
    }).detach();
}

void openFolder(const std::string& dir) {
    std::string cmd = "xdg-open '" + dir + "' >/dev/null 2>&1 &";
    if (system(cmd.c_str()) != 0) status = "couldn't open " + dir;
}

}  // namespace editor
