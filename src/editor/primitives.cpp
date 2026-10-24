// Breakable primitives (boxes, cylinders) as meshes, and the glTF extras that keep their parameters.
#include "editor/editor_internal.h"

namespace editor {

// ---------------------------------------------------------------- primitives
// Box from y = 0 up, centred in x / z; per-face planar UVs in metres / uvScale. Cylinder: n-sided prism, smooth
// sides, flat caps. Both convex, so breakStatic shatters them as one piece each.
void genPrim(const Item& it, uint32_t mat, MeshData& m) {
    m = {};
    auto vtx = [&](float x, float y, float z, float nx, float ny, float nz, float u, float v) {
        m.verts.insert(m.verts.end(), {x, y, z, nx, ny, nz, u / it.uvScale, v / it.uvScale});
        return (uint32_t)(m.verts.size() / 8 - 1);
    };
    auto quad = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        m.idx.insert(m.idx.end(), {a, b, c, a, c, d});
        m.triMat.insert(m.triMat.end(), {mat, mat});
    };
    float hx = it.size[0] * 0.5f, hy = it.size[1], hz = it.size[2] * 0.5f;
    if (it.shape == BOX) {
        // each face: normal, then 4 corners CCW from outside, with their (u, v)
        struct F { float n[3]; float c[4][3]; int va; };
        const F faces[6] = {
            {{1, 0, 0}, {{hx, 0, hz}, {hx, 0, -hz}, {hx, hy, -hz}, {hx, hy, hz}}, 1},
            {{-1, 0, 0}, {{-hx, 0, -hz}, {-hx, 0, hz}, {-hx, hy, hz}, {-hx, hy, -hz}}, 1},
            {{0, 1, 0}, {{-hx, hy, hz}, {hx, hy, hz}, {hx, hy, -hz}, {-hx, hy, -hz}}, 2},
            {{0, -1, 0}, {{-hx, 0, -hz}, {hx, 0, -hz}, {hx, 0, hz}, {-hx, 0, hz}}, 2},
            {{0, 0, 1}, {{-hx, 0, hz}, {hx, 0, hz}, {hx, hy, hz}, {-hx, hy, hz}}, 1},
            {{0, 0, -1}, {{hx, 0, -hz}, {-hx, 0, -hz}, {-hx, hy, -hz}, {hx, hy, -hz}}, 1}};
        for (auto& f : faces) {
            uint32_t v[4];
            for (int k = 0; k < 4; k++) {
                const float* c = f.c[k];
                // u runs left to right seen from outside, v up the wall (texture upright, not mirrored)
                float u = f.n[0] > 0 ? -c[2] : f.n[0] < 0 ? c[2] : f.n[2] > 0 ? c[0] : f.n[2] < 0 ? -c[0] : c[0];
                v[k] = vtx(c[0], c[1], c[2], f.n[0], f.n[1], f.n[2], u, f.va == 1 ? c[1] : c[2]);
            }
            quad(v[0], v[1], v[2], v[3]);
        }
    } else {
        int n = std::clamp(it.sides, 3, 128);
        float rx = hx, rz = hz, circ = 3.14159f * (rx + rz);
        uint32_t base = (uint32_t)(m.verts.size() / 8);
        for (int i = 0; i <= n; i++) {
            float a = 6.2831853f * i / n, c = cosf(a), s = sinf(a);
            float nx = c / std::max(rx, 1e-4f), nz = s / std::max(rz, 1e-4f), l = sqrtf(nx * nx + nz * nz);
            float u = circ * i / n;
            vtx(c * rx, 0, s * rz, nx / l, 0, nz / l, u, 0);
            vtx(c * rx, hy, s * rz, nx / l, 0, nz / l, u, hy);
        }
        for (int i = 0; i < n; i++) {
            uint32_t a = base + 2 * i;
            quad(a, a + 1, a + 3, a + 2);
        }
        for (int cap = 0; cap < 2; cap++) {
            float y = cap ? hy : 0.f, ny = cap ? 1.f : -1.f;
            uint32_t c = vtx(0, y, 0, 0, ny, 0, 0, 0), first = (uint32_t)(m.verts.size() / 8);
            for (int i = 0; i < n; i++) {
                float a = 6.2831853f * i / n;
                vtx(cosf(a) * rx, y, sinf(a) * rz, 0, ny, 0, cosf(a) * rx, sinf(a) * rz);
            }
            for (int i = 0; i < n; i++) {
                uint32_t p = first + i, q = first + (i + 1) % n;
                if (cap) m.idx.insert(m.idx.end(), {c, q, p});
                else m.idx.insert(m.idx.end(), {c, p, q});
                m.triMat.push_back(mat);
            }
        }
    }
    for (int k = 0; k < 3; k++) { m.bmin[k] = 1e30f; m.bmax[k] = -1e30f; }
    for (size_t i = 0; i < m.verts.size(); i += 8)
        for (int k = 0; k < 3; k++) { m.bmin[k] = std::min(m.bmin[k], m.verts[i + k]); m.bmax[k] = std::max(m.bmax[k], m.verts[i + k]); }
    m.strength = it.strength;
    m.breakKind = breakKindOf(it.mat);
}

std::string fmt(const char* f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

std::string primExtras(const Item& it) {
    return fmt("{\"prim\":\"%s\",\"sx\":%g,\"sy\":%g,\"sz\":%g,\"sides\":%d,\"mat\":\"%s\",\"uv\":%g,\"tr\":%g,\"tg\":%g,\"tb\":%g,\"strength\":%g%s}",
               it.shape == BOX ? "box" : "cylinder", it.size[0], it.size[1], it.size[2], it.sides, it.mat.c_str(), it.uvScale, it.tint[0],
               it.tint[1], it.tint[2], it.strength, it.collide ? "" : ",\"collide\":0");
}

// set / replace one numeric key in a flat extras object
std::string setKey(std::string json, const char* key, float v) {
    std::string k = std::string("\"") + key + "\"";
    std::string val = fmt("%g", v);
    size_t p = json.find(k);
    if (json.empty() || json[0] != '{') json = "{}";
    if (p == std::string::npos) {
        json.insert(1, k + ":" + val + (json.size() > 2 ? "," : ""));
        return json;
    }
    size_t c = json.find(':', p + k.size());
    size_t e = json.find_first_of(",}", c + 1);
    json.replace(c + 1, e - c - 1, val);
    return json;
}

}  // namespace editor
