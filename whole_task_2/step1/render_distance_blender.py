"""Render the metric Step1.3 distance diagram (Blender entry point).

The arm posture and stand are illustrations. The bunny, tool length, processing
point, station distance and window are drawn in the configured metre scale.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robots_config"))
from arm_visuals import box, camera, cylinder, illustrated_arm, light, line, material, setup, sphere


def dimension(name, start, finish, offset, mat, radius=.0012):
    start, finish, offset = Vector(start), Vector(finish), Vector(offset)
    a, b = start+offset, finish+offset
    objects = [line(name+" extension", [start, a], radius*.7, mat),
               line(name+" extension", [finish, b], radius*.7, mat)]
    for i in range(20):
        objects.append(line(name+" dash", [a+(b-a)*(i/20), a+(b-a)*((i+.63)/20)], radius, mat))
    tick = offset.normalized()*radius*7
    for endpoint in (a, b):
        objects.append(line(name+" tick", [endpoint-tick, endpoint+tick], radius, mat))
    return objects, (a+b)/2


def pixels(scene, cam, point):
    value = world_to_camera_view(scene, cam, Vector(point))
    return [value.x*scene.render.resolution_x, (1-value.y)*scene.render.resolution_y]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--python-site-packages", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    sys.path.insert(0, str(args.python_site_packages))
    import numpy as np
    context = json.loads((args.data / "context.json").read_text())
    config = json.loads(Path(context["config_snapshot"]).read_text())
    data = np.load(args.data / "scene.npz")
    scene = setup(width=1500, height=1100, samples=48)
    points, tip, target, horizontal, lateral = illustrated_arm(config)
    for name in ["workpiece", "assumed work window", "tool body", "tool nozzle", "contact tool face"]:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    robot_objects = list(scene.objects)
    axis = Vector(config["task_constraint"]["working_direction_world"]).normalized()
    normal = -axis
    root = Vector(config["feasibility"]["base_pose_world"]["position_m"])
    up = Vector((0, 0, 1))

    rotation = data["rotation_world_object"]
    translation = data["translation_world_object_m"]
    vertices = data["vertices_object"] @ rotation.T+translation
    faces = data["faces"]
    body = material("bunny porcelain", (.79, .82, .84), .05, roughness=.55)
    green = material("broader ear working area", (.22, .72, .60), .03, roughness=.55)
    dark_green = material("ear region outline", (.08, .39, .32), roughness=.6)
    mesh = bpy.data.meshes.new("original 5000-face bunny")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    bunny = bpy.data.objects.new("bunny / screening alignment", mesh)
    bpy.context.collection.objects.link(bunny)
    mesh.materials.append(body)
    mesh.materials.append(green)
    selected = set(data["broader_ear_source_face_ids"].tolist())
    for face in mesh.polygons:
        face.material_index = int(face.index in selected)
        face.use_smooth = True
    edges = {}
    for face_id in selected:
        face = faces[face_id]
        for a, b in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(a), int(b))))
            edges[key] = edges.get(key, 0)+1
    for (a, b), count in edges.items():
        if count == 1:
            line("ear patch boundary", [vertices[a], vertices[b]], .00045, dark_green)

    # A slender illustrative probe makes the current zero-gap setting visible.
    metal = material("probe steel", (.42, .48, .53), .65, roughness=.28)
    orange = material("contact marker", (.91, .39, .13), roughness=.55)
    flange = points[-1]
    length = (tip-flange).length
    cylinder("probe mount", flange, flange+axis*length*.40, .012, metal, .009)
    cylinder("probe shaft", flange+axis*length*.40, tip, .005, metal, .0015)
    sphere("zero gap contact", tip, .0022, orange)

    # Display support only: no grasp, collision or support optimization is claimed.
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    stand_top = max(.035, float(lower[2])-.0005)
    stand_center = (lower+upper)/2
    stand_center[2] = stand_top/2
    box("display stand / schematic", stand_center,
        (max(.12, float(upper[0]-lower[0])*.88), max(.12, float(upper[1]-lower[1])*.88), stand_top),
        material("display stand", (.92, .93, .94), roughness=.65), radius=.006)

    blue = material("metric distances", (.22, .43, .62), roughness=.65)
    base_dimensions, base_midpoint = dimension("base to workpoint", root, target, -lateral*.23, blue)
    base_label = base_midpoint-up*.045-lateral*.012
    tool_dimensions, tool_midpoint = dimension("tool length", flange, tip, -lateral*.035, blue, radius=.00065)
    for obj in tool_dimensions:
        obj.hide_render = True

    first = lateral
    second = normal.cross(first).normalized()
    radius = context["window_radius_m"]
    window_objects = []
    for i in range(36):
        angles = np.linspace(2*math.pi*i/36, 2*math.pi*(i+.60)/36, 5)
        ring = [target+radius*(math.cos(a)*first+math.sin(a)*second) for a in angles]
        window_objects.append(line("task window boundary", ring, .00075, blue))
    window_label = target+first*.115-second*.080

    light("key", (-1, -2, 3), (0, -.2, .4), 180, 3)
    light("fill", (2, 1, 2), (0, -.3, .4), 110, 2.5)
    bpy.context.view_layer.update()
    extent = [obj.matrix_world @ Vector(corner) for obj in scene.objects
              if obj.type in ("MESH", "CURVE") and not obj.hide_render for corner in obj.bound_box]
    extent += [base_label-up*.04, window_label+up*.04]
    view = (-lateral+.36*up-.42*horizontal).normalized()
    cam = camera(extent, view, 1.16)
    bpy.context.view_layer.update()
    projections = {"wide": {"base_distance": pixels(scene, cam, base_label),
                             "window": pixels(scene, cam, window_label),
                             "contact": pixels(scene, cam, tip)}}
    scene.render.filepath = str(args.data / "wide.png")
    bpy.ops.render.render(write_still=True)

    for obj in base_dimensions+window_objects:
        obj.hide_render = True
    for obj in tool_dimensions:
        obj.hide_render = False
    # Isolate the tool in the inset so the schematic wrist cannot hide its length.
    for obj in robot_objects:
        obj.hide_render = True
    cylinder("flange / detail", flange-axis*.012, flange, .017, metal)
    bpy.data.objects["display stand / schematic"].hide_render = True
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1000
    # Crop to the actual upper ear and probe at unchanged physical scale.
    extent = [Vector(vertex) for vertex in vertices[data["vertices_object"][:, 1] >= .14]]
    extent += [flange+up*.025, tip, tool_midpoint-lateral*.05]
    close_view = (-lateral+.55*up-.85*horizontal).normalized()
    cam = camera(extent, close_view, 1.27)
    bpy.context.view_layer.update()
    projections["closeup"] = {"tool_length": pixels(scene, cam, tool_midpoint-lateral*.037),
                               "contact": pixels(scene, cam, tip)}
    scene.render.filepath = str(args.data / "closeup.png")
    bpy.ops.render.render(write_still=True)
    (args.data / "projections.json").write_text(json.dumps(projections, indent=2)+"\n")
    print("Distance diagram rendered at configured metric scale", flush=True)


if __name__ == "__main__":
    main()
