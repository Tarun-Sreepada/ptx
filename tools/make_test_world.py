# Generates a small test world for ptx with Blender, the same way you'd author a real one.
#   blender -b --python tools/make_test_world.py -- world/
# Shows the conventions from src/world.h:
#   * one collection per cell, exported in WORLD coordinates to world/cells/<x>_<z>.glb (cells are 64 m)
#   * reusable assets in their own collection at the origin -> world/assets/<name>.glb
#   * Empties with custom properties: asset="<name>" (instance an asset), flatten=1 (+ falloff) (flatten terrain;
#     put the Empty a few cm BELOW your ground mesh or the terrain z-fights with it)
#   * point lights (exported as KHR_lights_punctual), emissive materials
import math
import os
import sys

import bpy
import numpy as np

out = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "world/"
os.makedirs(os.path.join(out, "cells"), exist_ok=True)
os.makedirs(os.path.join(out, "assets"), exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def collection(name):
    c = bpy.data.collections.new(name)
    scene.collection.children.link(c)
    return c


def B(x, y, z):  # engine (Y up, glTF) -> Blender (Z up); the glTF exporter converts back
    return (x, -z, y)


def checker(name, size, c0, c1, cells=8):
    path = os.path.join(out, "assets", f"_{name}.png")
    yy, xx = np.mgrid[0:size, 0:size]
    m = ((xx * cells // size + yy * cells // size) % 2)[..., None]
    noise = np.random.default_rng(1).uniform(0.9, 1.0, (size, size, 1))
    rgb = (np.where(m, np.array(c0), np.array(c1)) * noise).astype(np.float32)
    px = np.concatenate([rgb, np.ones((size, size, 1), np.float32)], axis=2)
    img = bpy.data.images.new(name, size, size)
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    return img


def material(name, color, image=None, emit=None, strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = 0.7
    if image:
        t = m.node_tree.nodes.new("ShaderNodeTexImage")
        t.image = image
        m.node_tree.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*emit, 1)
        bsdf.inputs["Emission Strength"].default_value = strength
    return m


def boxes(name, bxs, mat, col):
    """One mesh object from engine-space boxes [(min, max), ...], box-projected UVs at 1 unit = 1 m."""
    verts, faces, uvs = [], [], []
    for mn, mx in bxs:
        x0, y0, z0 = B(mn[0], mn[1], mx[2])
        x1, y1, z1 = B(mx[0], mx[1], mn[2])
        c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        b = len(verts)
        verts += c
        for f, axis in (((0, 3, 2, 1), 2), ((4, 5, 6, 7), 2), ((0, 1, 5, 4), 1), ((2, 3, 7, 6), 1), ((1, 2, 6, 5), 0), ((3, 0, 4, 7), 0)):
            faces.append(tuple(b + i for i in f))
            a, bb = [(1, 2), (0, 2), (0, 1)][axis]
            uvs.append([(c[i][a], c[i][bb]) for i in f])
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    uv = me.uv_layers.new()
    for p, fu in zip(me.polygons, uvs):
        for li, u in zip(p.loop_indices, fu):
            uv.data[li].uv = u
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    return ob


def empty(name, pos, col, scale=1.0, **props):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = "CUBE"
    e.location = B(*pos)
    e.scale = (scale, scale, scale) if not isinstance(scale, tuple) else scale
    for k, v in props.items():
        e[k] = v
    col.objects.link(e)
    return e


def point_light(name, pos, col, watts, color=(1.0, 0.85, 0.65)):
    ld = bpy.data.lights.new(name, "POINT")
    ld.energy = watts
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    ob.location = B(*pos)
    col.objects.link(ob)


wood = material("wood", (1, 1, 1), checker("wood", 256, (0.45, 0.3, 0.18), (0.38, 0.25, 0.15)))
plaster = material("plaster", (0.78, 0.76, 0.72))
concrete = material("concrete", (0.5, 0.5, 0.48))
paving = material("paving", (1, 1, 1), checker("paving", 256, (0.42, 0.41, 0.4), (0.35, 0.34, 0.33), 16))
crate_m = material("crate", (1, 1, 1), checker("crate", 128, (0.6, 0.45, 0.25), (0.5, 0.36, 0.2), 4))
panel = material("panel", (0.9, 0.9, 0.9), emit=(1.0, 0.9, 0.75), strength=4.0)
pole_m = material("pole", (0.15, 0.15, 0.16))
lamp_m = material("lamp", (1, 1, 1), emit=(1.0, 0.8, 0.5), strength=6.0)


def building(prefix, col, x0, z0, x1, z1, floors, H=3.2, T=0.2):
    """Walkable building: slabs with stair holes, switchback stairs, door, window openings, a light per floor."""
    slab = 0.15
    flight_a = (x1 - T - 2.2, x1 - T - 0.2)  # stair x ranges
    flight_b = (x1 - T - 4.4, x1 - T - 2.4)
    sz0, sz1 = z0 + 2.0, z0 + 2.0 + 4.9      # stair run along z
    floor_boxes, wall_boxes, stair_boxes, panel_boxes = [], [], [], []
    for k in range(floors + 1):
        y = k * H
        top = y + slab
        if k == 0:
            floor_boxes.append(((x0, y, z0), (x1, top, z1)))
        else:  # hole over the flight arriving from below
            hx = flight_a if (k - 1) % 2 == 0 else flight_b
            hx = (hx[0] - 0.1, hx[1] + 0.1)
            floor_boxes += [((x0, y, z0), (hx[0], top, z1)), ((hx[1], y, z0), (x1, top, z1)),
                            ((hx[0], y, z0), (hx[1], top, sz0)), ((hx[0], y, sz1), (hx[1], top, z1))]
            if k == floors:
                floor_boxes[-4:] = [((x0, y, z0), (x1, top, z1))]  # roof: closed
        if k == floors:
            break
        # walls with openings (door on the ground floor south side, windows elsewhere)
        wy0, wy1 = top, (k + 1) * H
        mid = (x0 + x1) / 2
        for (a, b, zz) in ((x0, x1, z0), (x0, x1, z1 - T)):
            south = zz == z0
            o0, o1 = (mid - 1.0, mid + 1.0) if (south and k == 0) else (mid - 1.5, mid + 1.5)
            sill = wy0 if (south and k == 0) else wy0 + 0.9
            head = wy0 + 2.2
            wall_boxes += [((a, wy0, zz), (o0, wy1, zz + T)), ((o1, wy0, zz), (b, wy1, zz + T)), ((o0, head, zz), (o1, wy1, zz + T))]
            if sill > wy0:
                wall_boxes.append(((o0, wy0, zz), (o1, sill, zz + T)))
        zm = (z0 + z1) / 2
        for xx in (x0, x1 - T):
            wall_boxes += [((xx, wy0, z0 + T), (xx + T, wy1, zm - 1.2)), ((xx, wy0, zm + 1.2), (xx + T, wy1, z1 - T)),
                           ((xx, wy0, zm - 1.2), (xx + T, wy0 + 0.9, zm + 1.2)), ((xx, wy0 + 2.2, zm - 1.2), (xx + T, wy1, zm + 1.2))]
        # stair flight up to the next floor: 18 solid steps, alternating direction
        fx = flight_a if k % 2 == 0 else flight_b
        n, rise, run = 18, H / 18, (sz1 - sz0) / 18
        for i in range(n):
            if k % 2 == 0:
                za, zb = sz0 + i * run, sz0 + (i + 1) * run
            else:
                za, zb = sz1 - (i + 1) * run, sz1 - i * run
            stair_boxes.append(((fx[0], top, za), (fx[1], top + (i + 1) * rise, zb)))
        # ceiling light panel + point light
        cx, cz = (x0 + x1) / 2 - 2, (z0 + z1) / 2 + 1
        panel_boxes.append(((cx - 0.6, wy1 - 0.04, cz - 0.6), (cx + 0.6, wy1, cz + 0.6)))
        point_light(f"{prefix}_light{k}", (cx, wy1 - 0.3, cz), col, 60)
    # interior probe volume: baked lights-on / lights-off irradiance, and the light switch for this building
    empty(f"{prefix}_bake", ((x0 + x1) / 2, floors * H / 2 + slab, (z0 + z1) / 2), col,
          scale=((x1 - x0) / 2, (z1 - z0) / 2, floors * H / 2), bake=1)
    boxes(f"{prefix}_floors", floor_boxes, wood, col)
    boxes(f"{prefix}_walls", wall_boxes, plaster, col)
    boxes(f"{prefix}_stairs", stair_boxes, concrete, col)
    o = boxes(f"{prefix}_panels", panel_boxes, panel, col)
    o["collide"] = 0


# ---- assets (local space, origin at the base)
assets = collection("assets_crate")
boxes("crate", [((-0.5, 0, -0.5), (0.5, 1, 0.5))], crate_m, assets)
lamp = collection("assets_lamp_post")
boxes("pole", [((-0.08, 0, -0.08), (0.08, 4.2, 0.08))], pole_m, lamp)
o = boxes("head", [((-0.25, 4.2, -0.25), (0.25, 4.45, 0.25))], lamp_m, lamp)
point_light("lamp", (0, 4.0, 0), lamp, 150)

# ---- cell 0_0: plaza (flattened to y=0), 3-storey building, crates, lamp posts
c00 = collection("cell_0_0")
empty("zone_plaza", (32, -0.1, 32), c00, scale=(30, 30, 1), flatten=1, falloff=12.0)
boxes("plaza", [((2, -0.3, 2), (62, 0.0, 62))], paving, c00)
building("bA", c00, 14, 20, 30, 34, floors=3)
for i, (x, z, r) in enumerate([(8, 12, 0), (9.2, 12.3, 25), (8.5, 13.4, 50), (20, 25, 10), (22, 28, 0)]):
    e = empty(f"crate{i}", (x, 0.0 if i < 3 else 0.15, z), c00, asset="crate")
    e.rotation_euler = (0, 0, math.radians(r))
for i, (x, z) in enumerate([(6, 6), (32, 6), (58, 6), (6, 58), (58, 58), (40, 30)]):
    empty(f"lamp{i}", (x, 0, z), c00, asset="lamp_post")

# ---- cell 1_0: second plaza + 2-storey building
c10 = collection("cell_1_0")
empty("zone_b", (96, -0.1, 30), c10, scale=(26, 22, 1), flatten=1, falloff=12.0)
boxes("plaza_b", [((72, -0.3, 10), (120, 0.0, 50))], paving, c10)
building("bB", c10, 80, 16, 92, 28, floors=2)
for i, (x, z) in enumerate([(74, 12), (118, 12), (118, 48)]):
    empty(f"lampb{i}", (x, 0, z), c10, asset="lamp_post")


def export(col, path):
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[col.name]
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_active_collection=True, export_extras=True, export_lights=True)


export(assets, os.path.join(out, "assets", "crate.glb"))
export(lamp, os.path.join(out, "assets", "lamp_post.glb"))
export(c00, os.path.join(out, "cells", "0_0.glb"))
export(c10, os.path.join(out, "cells", "1_0.glb"))
print("wrote test world to", out)
