// Procedural buildings from breakable primitives (the editor's boxes): a footprint, storeys, walls with windows on
// every side and a door at the front, floor slabs, an inner wall with a doorway on each storey, a pitched (gable) or
// flat roof, a chimney, a light per storey and a bake volume. Each wall is cut into pieces around its openings
// (piers between them, the sill below, the lintel above), so the breach code takes walls apart piece by piece the way
// masonry fails. The style picks the materials.
#include "editor/editor_internal.h"

namespace editor {
namespace {
constexpr float SLAB = 0.2f;              // floor slab thickness (m)
constexpr float DOOR_W = 1.1f, DOOR_H = 2.2f;
constexpr float INNER_DOOR_W = 0.9f, INNER_DOOR_H = 2.1f;
constexpr float WINDOW_MARGIN = 0.8f;     // m of wall kept solid at each corner
constexpr float ROOF_PITCH = 35.f;        // degrees
constexpr float EAVES = 0.4f;             // roof overhang (m)
constexpr float ROOF_THICK = 0.18f;
constexpr int GABLE_STEPS = 4;            // the gable triangle as stacked courses
constexpr float LIGHT_INTENSITY = 300.f;

struct Style { const char* name; const char* wall; const char* trim; const char* floor; const char* roof; };
const Style STYLES[] = {
    {"Brick cottage", "brick", "concrete", "planks", "roof"},
    {"Plaster town house", "plaster", "stone", "concrete", "tiles"},
    {"Timber cabin", "planks", "planks", "planks", "roof"},
    {"Concrete block", "concrete", "concrete", "concrete", "asphalt"},
    {"Stone farmhouse", "stone", "stone", "planks", "roof"},
};

struct Opening { float x, w, sill, h; };  // along the wall: centre, width; from the storey floor: sill, height

// A wall along local x from -L/2 to L/2, thickness T, one storey high (y0 .. y0 + H), cut around its openings.
// emit(sizeX, sizeY, centreX, centreY): one piece in the wall's own frame.
template <class Emit> void wall(float L, float H, std::vector<Opening> ops, Emit emit) {
    std::sort(ops.begin(), ops.end(), [](const Opening& a, const Opening& b) { return a.x < b.x; });
    float x = -0.5f * L;
    for (const Opening& o : ops) {
        float a = o.x - 0.5f * o.w, b = o.x + 0.5f * o.w;
        if (a - x > 0.02f) emit(a - x, H, 0.5f * (x + a), 0.5f * H);         // pier
        if (o.sill > 0.02f) emit(o.w, o.sill, o.x, 0.5f * o.sill);           // below the opening
        float top = o.sill + o.h;
        if (H - top > 0.02f) emit(o.w, H - top, o.x, 0.5f * (H + top));     // lintel over it
        x = b;
    }
    if (0.5f * L - x > 0.02f) emit(0.5f * L - x, H, 0.5f * (x + 0.5f * L), 0.5f * H);
}

// windows spread evenly along a wall of length L (none if it's too short), optionally a door in the middle
std::vector<Opening> openings(const BuildingParams& P, float L, bool door) {
    std::vector<Opening> ops;
    int n = std::max(0, (int)floorf((L - 2.f * WINDOW_MARGIN + P.windowSpacing - P.windowW) / P.windowSpacing));
    float span = (n - 1) * P.windowSpacing;
    for (int i = 0; i < n; i++) {
        float x = -0.5f * span + i * P.windowSpacing;
        if (door && fabsf(x) < 0.5f * (DOOR_W + P.windowW) + 0.2f) continue;  // the door takes this place
        ops.push_back({x, P.windowW, P.windowSill, P.windowH});
    }
    if (door) ops.push_back({0.f, DOOR_W, 0.f, DOOR_H});
    return ops;
}
}  // namespace

const char* buildingStyleName(int i) { return i >= 0 && i < (int)(sizeof STYLES / sizeof STYLES[0]) ? STYLES[i].name : ""; }
int buildingStyleCount() { return (int)(sizeof STYLES / sizeof STYLES[0]); }

// The panel: style, size, storeys, roof; builds at the placement point facing the camera (snapped to 90 degrees).
void buildingPanel() {
    static BuildingParams P;
    if (!ImGui::TreeNode("Building generator")) return;
    if (ImGui::BeginCombo("Style", buildingStyleName(P.style))) {
        for (int i = 0; i < buildingStyleCount(); i++)
            if (ImGui::Selectable(buildingStyleName(i), i == P.style)) P.style = i;
        ImGui::EndCombo();
    }
    ImGui::SliderInt("Storeys", &P.storeys, 1, 5);
    ImGui::SliderFloat("Width", &P.width, 4.f, 24.f, "%.1f m");
    ImGui::SliderFloat("Depth", &P.depth, 4.f, 16.f, "%.1f m");
    ImGui::SliderFloat("Storey height", &P.storeyH, 2.6f, 4.f, "%.1f m");
    ImGui::SliderFloat("Window spacing", &P.windowSpacing, 1.6f, 5.f, "%.1f m");
    ImGui::Checkbox("Pitched roof", &P.gable);
    if (ImGui::Button("Build here")) {
        float p[3];
        placePoint(p);
        p[0] = snapv(p[0], 0.5f); p[2] = snapv(p[2], 0.5f);
        std::vector<Item> parts;
        generateBuilding(P, p, camYawSnapped(), parts);
        if (ECell* c = open(cellOf(p[0]), cellOf(p[2]))) {
            pushUndo(*c);
            for (auto& it : parts) c->items.push_back(it);
            changed(*c, true);
            sel = {c->cx, c->cz, (int)c->items.size() - (int)parts.size()};
            status = fmt("building added: %d pieces", (int)parts.size());
        } else status = "that cell is still loading";
    }
    ImGui::TreePop();
}

void generateBuilding(const BuildingParams& P, const float at[3], float yawDeg, std::vector<Item>& out) {
    const Style& st = STYLES[std::clamp(P.style, 0, buildingStyleCount() - 1)];
    float yaw = yawDeg * DEG, W = P.width, D = P.depth, T = P.wall, Hs = P.storeyH;
    // a part placed in the building's frame (x across, z deep, front at -z), rotated by extraYaw about its centre
    auto put = [&](Item it, float x, float y, float z, float extraYaw = 0.f, float pitch = 0.f) {
        it.pos[0] = at[0] + cosf(yaw) * x + sinf(yaw) * z;
        it.pos[2] = at[2] - sinf(yaw) * x + cosf(yaw) * z;
        it.pos[1] = at[1] + y;
        it.rot[0] += pitch;
        it.rot[1] += yawDeg + extraYaw;
        out.push_back(it);
    };
    put(prim(BOX, W + 0.6f, 0.3f, D + 0.6f, st.trim, "foundation"), 0, -0.1f, 0);
    for (int s = 0; s < P.storeys; s++) {
        float y0 = 0.2f + s * Hs, wallH = Hs - (s + 1 < P.storeys ? SLAB : 0.f);
        // outer walls: front / back span the width, the sides fit between them
        for (int side = 0; side < 4; side++) {
            bool alongX = side < 2;
            float L = alongX ? W : D - 2.f * T;
            bool door = side == 0 && s == 0;
            auto ops = openings(P, L, door);
            float off = alongX ? (side == 0 ? -1.f : 1.f) * (0.5f * D - 0.5f * T) : (side == 2 ? -1.f : 1.f) * (0.5f * W - 0.5f * T);
            const char* nm = side == 0 ? "front wall" : side == 1 ? "back wall" : "side wall";
            wall(L, wallH, ops, [&](float sx, float sy, float cx, float cy) {
                Item it = prim(BOX, sx, sy, T, st.wall, nm);
                if (alongX) put(it, cx, y0 + cy - 0.5f * sy, off);
                else put(it, off, y0 + cy - 0.5f * sy, cx, 90.f);
            });
        }
        // an inner wall across the building with a doorway, a third of the way in
        float ix = -W * 0.5f + W * 0.38f;
        std::vector<Opening> inner{{(s & 1 ? 0.25f : -0.25f) * (D - 2.f * T), INNER_DOOR_W, 0.f, INNER_DOOR_H}};
        wall(D - 2.f * T, wallH, inner, [&](float sx, float sy, float cx, float cy) {
            put(prim(BOX, sx, sy, T * 0.6f, st.wall, "inner wall"), ix, y0 + cy - 0.5f * sy, cx, 90.f);
        });
        // the floor above
        if (s + 1 < P.storeys) put(prim(BOX, W - 2.f * T, SLAB, D - 2.f * T, st.floor, "floor"), 0, y0 + wallH, 0);
        Item L;
        L.kind = K_LIGHT;
        L.intensity = LIGHT_INTENSITY;
        put(L, W * 0.15f, y0 + wallH - 0.4f, 0);
    }
    float top = 0.2f + P.storeys * Hs;
    if (P.gable) {  // two pitched planes meeting over the ridge (along x), stepped gable ends filling the triangle
        float p = ROOF_PITCH * DEG, half = 0.5f * D + EAVES, rise = 0.5f * D * tanf(p), slope = half / cosf(p);
        for (int s = -1; s <= 1; s += 2)
            put(prim(BOX, W + 2.f * EAVES, ROOF_THICK, slope, st.roof, "roof"), 0, top + 0.5f * rise - 0.5f * EAVES * tanf(p), s * 0.5f * half,
                0.f, s * ROOF_PITCH);  // front (s -1) rises toward +z, back falls
        for (int s = -1; s <= 1; s += 2)
            for (int k = 0; k < GABLE_STEPS; k++) {
                float h = rise / GABLE_STEPS, w = (D - 2.f * T) * (1.f - (k + 0.5f) / GABLE_STEPS);
                put(prim(BOX, w, h, T, st.wall, "gable"), s * (0.5f * W - 0.5f * T), top + k * h, 0, 90.f);
            }
        put(prim(BOX, 0.6f, rise + 1.2f, 0.6f, "brick", "chimney"), W * 0.3f, top + 0.2f, D * 0.15f);
    } else {  // flat roof with a parapet
        put(prim(BOX, W, SLAB, D, st.roof, "roof"), 0, top, 0);
        for (int s = -1; s <= 1; s += 2) {
            put(prim(BOX, W, 0.6f, T, st.trim, "parapet"), 0, top + SLAB, s * (0.5f * D - 0.5f * T));
            put(prim(BOX, D - 2.f * T, 0.6f, T, st.trim, "parapet"), s * (0.5f * W - 0.5f * T), top + SLAB, 0, 90.f);
        }
    }
    // a bake volume around it (axis aligned), so its interior gets baked light
    Item V;
    V.kind = K_VOLUME;
    V.name = fmt("building_%d", (int)now() % 100000);
    float hx = 0.5f * W + 0.3f, hz = 0.5f * D + 0.3f, hy = 0.5f * (top + 0.2f);
    V.scale[0] = fabsf(cosf(yaw)) * hx + fabsf(sinf(yaw)) * hz;
    V.scale[1] = hy;
    V.scale[2] = fabsf(sinf(yaw)) * hx + fabsf(cosf(yaw)) * hz;
    V.pos[0] = at[0]; V.pos[1] = at[1] + hy; V.pos[2] = at[2];
    out.push_back(V);
}
}  // namespace editor
