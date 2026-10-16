"""Render the fixed processing robot and its opposed work areas in Blender."""
from pathlib import Path
import argparse
import json
import math
import sys

import bpy
from mathutils import Matrix, Vector

IDEA = Path(__file__).resolve().parents[1]
DATA = IDEA / "data/robot_access"
OUT = IDEA / "vis/robot_access"


def material(name, rgb, metallic=0., roughness=.4, alpha=1.):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    rgba = tuple((c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4) for c in rgb) + (alpha,)
    mat.diffuse_color = rgba
    node = mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = rgba
    node.inputs["Metallic"].default_value = metallic
    node.inputs["Roughness"].default_value = roughness
    node.inputs["Alpha"].default_value = alpha
    if alpha < 1:
        mat.blend_method = "HASHED"
        mat.use_screen_refraction = True
        mat.show_transparent_back = False
    return mat


def load_mesh(name, mat, smooth=False):
    vertices, faces = [], []
    for line in (DATA / (name + ".obj")).read_text().splitlines():
        if line.startswith("v "):
            vertices.append(tuple(float(v) for v in line.split()[1:4]))
        elif line.startswith("f "):
            faces.append(tuple(int(v.split("/")[0]) - 1 for v in line.split()[1:]))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(mat)
    for face in mesh.polygons:
        face.use_smooth = smooth
    return obj


def cylinder(name, a, b, radius, mat, radius2=None):
    a, b = Vector(a), Vector(b)
    delta = b - a
    if radius2 is None:
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius, depth=delta.length)
    else:
        bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=radius, radius2=radius2, depth=delta.length)
    obj = bpy.context.object
    obj.name = name
    obj.location = (a + b) / 2
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(mat)
    obj.data.use_auto_smooth = True
    for p in obj.data.polygons:
        p.use_smooth = len(p.vertices) == 4
    bevel = obj.modifiers.new("Soft machined edges", "BEVEL")
    bevel.width, bevel.segments = min(radius*.08, .0015), 3
    obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    return obj


def box(name, position, size, mat, bevel=.005):
    bpy.ops.mesh.primitive_cube_add(size=1, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    obj.data.use_auto_smooth = True
    modifier = obj.modifiers.new("Rounded edges", "BEVEL")
    modifier.width, modifier.segments = bevel, 4
    obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    return obj


def area(name, position, target, power, size):
    light = bpy.data.lights.new(name, "AREA")
    light.energy, light.shape, light.size = power, "DISK", size
    obj = bpy.data.objects.new(name, light)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat("-Z", "Y").to_euler()


def camera(target, direction, width):
    obj = bpy.context.scene.camera
    obj.location = Vector(target) + Vector(direction).normalized()*3
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat("-Z", "Y").to_euler()
    obj.data.ortho_scale = width
    return dict(target=target, direction=direction, width_m=width)


def fit_overview(direction):
    forward = Vector(direction).normalized()
    right = Vector((0, 0, 1)).cross(forward).normalized()
    up = forward.cross(right)
    points = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and not obj.hide_render:
            points.extend(obj.matrix_world @ Vector(p) for p in obj.bound_box)
    x = [p.dot(right) for p in points]
    y = [p.dot(up) for p in points]
    center = right*((min(x)+max(x))/2) + up*((min(y)+max(y))/2)
    width = max(max(x)-min(x), (max(y)-min(y))*1.6)*1.12
    return camera(tuple(center), direction, width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA/"scene.json").read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24 if args.preview else 128
    scene.cycles.use_denoising = False
    scene.eevee.taa_render_samples = 32 if args.preview else 128
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance, scene.eevee.gtao_factor = .025, 1.1
    scene.eevee.use_soft_shadows = True
    scene.render.resolution_x, scene.render.resolution_y = (1200, 750) if args.preview else (2400, 1500)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .8
    # Expose the rendered objects before adding exact white around them.
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layers = nodes.new("CompositorNodeRLayers")
    exposure = nodes.new("CompositorNodeExposure")
    exposure.inputs[1].default_value = -2.
    over = nodes.new("CompositorNodeAlphaOver")
    over.inputs[1].default_value = (1, 1, 1, 1)
    output = nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(layers.outputs["Image"], exposure.inputs[0])
    scene.node_tree.links.new(exposure.outputs[0], over.inputs[2])
    scene.node_tree.links.new(over.outputs[0], output.inputs[0])

    white = material("Robot ceramic white", (.88, .90, .92), .08)
    black = material("Robot charcoal joints", (.16, .19, .22), .3)
    silver = material("Spindle brushed alloy", (.60, .66, .72), .72, .28)
    body_mat = material("B workpiece gray", (.66, .69, .72), .13)
    blue = material("Saved rigid fixture blue", (.12, .47, .72), .12)
    green = material("Front working area pose17", (.14, .71, .40))
    amber = material("Opposite working area pose13", (1., .48, .11))
    red = material("Blocked straight tool access", (.85, .12, .13))
    pedestal = material("Pedestal", (.91, .92, .93), .1)
    load_mesh("robot_white", white, True)
    load_mesh("robot_black", black, True)
    body = load_mesh("object", body_mat, True)
    fixture = load_mesh("fixture", blue)
    load_mesh("front_patch", green)
    load_mesh("back_patch", amber)

    H = Matrix(manifest["tool_hand_transform"])
    def local(z):
        return H @ Vector((0, 0, z))
    cylinder("Spindle mount", local(-.018), local(.006), .032, black)
    cylinder("Processing spindle", local(.006), local(.088), .023, silver)
    cylinder("Dark spindle collar", local(.088), local(.113), .016, black)
    cylinder("Tool collet", local(.113), local(.139), .012, silver, .005)
    cylinder("Fine processing bit", local(.139), local(.155), .0022, silver)
    # A small square work stand supports the unchanged saved fixture.
    all_vertices = [v.co for v in fixture.data.vertices]
    low = Vector(tuple(min(v[k] for v in all_vertices) for k in range(3)))
    high = Vector(tuple(max(v[k] for v in all_vertices) for k in range(3)))
    cx, cy = (low.x+high.x)/2, (low.y+high.y)/2
    box("Work stand top", (cx, cy, .5875), (.24, .23, .025), pedestal)
    box("Work stand column", (cx, cy, .2825), (.095, .095, .565), pedestal)
    box("Work stand base", (cx, cy, .014), (.27, .25, .028), pedestal)
    floor = material("White studio floor", (1., 1., 1.), roughness=.8)
    bpy.ops.mesh.primitive_plane_add(size=200)
    bpy.context.object.data.materials.append(floor)
    bpy.context.object.hide_render = True
    area("Large key", (-.4, -1.7, 2.3), (.4, 0, .5), 100, 2.)
    area("Soft fill", (1.2, 1.3, 1.8), (.4, 0, .5), 80, 2.)
    area("Rim", (-1., .7, 1.6), (.4, 0, .5), 65, 1.5)
    data = bpy.data.cameras.new("Camera")
    obj = bpy.data.objects.new("Camera", data)
    bpy.context.collection.objects.link(obj)
    data.type = "ORTHO"
    data.clip_start, data.clip_end = .01, 100
    scene.camera = obj

    # The overlay's dashed line is the actual tested straight approach to the
    # rear work area. A cross sits at its first intersection with the body.
    annotations = []
    a = Vector(manifest["blocked_segment_start"])
    b = Vector(manifest["rear_contact"])
    length = (b-a).length
    direction = (b-a).normalized()
    for step in range(math.ceil(length/.012)):
        lo, hi = step*.012, min(step*.012+.006, length)
        if lo < length:
            annotations.append(cylinder("Dashed rear access", a+direction*lo, a+direction*hi, .00085, red))
    hit = Vector(manifest["blocked_segment_hit"])
    diagram_view = Vector((-.10, -1., .58)).normalized()
    diagram_right = Vector((0., 0., 1.)).cross(diagram_view).normalized()
    diagram_up = diagram_view.cross(diagram_right)
    for sign in (-1, 1):
        u = (diagram_right + sign*diagram_up)*.008
        # An orthographic screen marker: move toward the camera without
        # changing its projection, so the complete cross stays visible.
        center = hit + diagram_view*.15
        annotations.append(cylinder("Blocked cross", center-u, center+u, .00125, red))
    for annotation in annotations:
        annotation.hide_render = True

    scenes = [
        ("01_robot_working", (.34, 0, .445), (-.38, -1.55, .83), 1.38, False),
        ("02_front_work_area", (.662, 0, .705), (-.60, -1., .53), .48, False),
        ("03_back_work_area", (.675, 0, .705), (.75, -1., .45), .50, False),
        ("04_blocked_access", (.65, 0, .70), (-.10, -1., .58), .53, True),
    ]
    camera_path = DATA/("preview_cameras.json" if args.preview else "cameras.json")
    cameras = json.loads(camera_path.read_text()) if args.only and camera_path.exists() else {}
    for name, target, direction, width, annotate in scenes:
        if args.only and name not in args.only:
            continue
        for annotation in annotations:
            annotation.hide_render = not annotate
        body.data.materials[0] = (material("Workpiece explanatory transparency", (.66,.69,.72), alpha=.25)
                                  if annotate else body_mat)
        fixture.data.materials[0] = (material("Fixture explanatory transparency", (.12,.47,.72), alpha=.18)
                                     if annotate else blue)
        bpy.context.view_layer.update()
        cameras[name] = fit_overview(direction) if name == "01_robot_working" else camera(target, direction, width)
        scene.render.filepath = str((DATA if args.preview else OUT)/(name + ("_preview.png" if args.preview else ".png")))
        bpy.ops.render.render(write_still=True)
        print("ROBOT ACCESS RENDERED", name, flush=True)
    camera_path.write_text(json.dumps(cameras, indent=2)+"\n")
    if not args.preview:
        body.data.materials[0] = body_mat
        fixture.data.materials[0] = blue
        for annotation in annotations:
            annotation.hide_render = True
        fit_overview((-.38, -1.55, .83))
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(DATA/"robot_access.blend"))


if __name__ == "__main__":
    main()
