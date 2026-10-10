"""Render the immutable idea geometry/motion with Blender's Eevee renderer.

Run with blender -b --python this_file -- --mode pose_following_wrap.
The renderer applies only saved rigid transforms; it never modifies a fixture.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import time

import bpy
from mathutils import Matrix, Vector

IDEA = Path(__file__).resolve().parents[1]
DISPLAY_SCALE = 10.


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pose_following_wrap", "fixed_seat_reuse"), required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def linear(color):
    rgb = [int(color[n:n + 2], 16) / 255. for n in (1, 3, 5)]
    return tuple(v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb)


def material(name, color, roughness=.45, metallic=0.):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*linear(color), 1.)
    m.use_nodes = True
    node = m.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = m.diffuse_color
    node.inputs["Roughness"].default_value = roughness
    node.inputs["Metallic"].default_value = metallic
    return m


def load_mesh(path, name, mat, smooth=False):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(x) * DISPLAY_SCALE for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:]])
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    for polygon in data.polygons:
        polygon.use_smooth = smooth
    return obj, [Vector(v) / DISPLAY_SCALE for v in vertices]


def place(obj, transform):
    T = Matrix(transform)
    T.translation *= DISPLAY_SCALE
    obj.matrix_world = T


def area(name, location, power, size, target):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.shape, data.size = power, "DISK", size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def camera_fit(scene, frames, object_vertices, support_vertices, mode):
    # Focus on the fixture and seated object, rather than reserving empty air
    # for the complete extraction distance. Objects enter/leave the close-up.
    # Every placement of the empty rotating fixture remains inside the view.
    camera_vector = Vector((1.15, -1.65, 1.15)).normalized()
    right = Vector((0., 0., 1.)).cross(camera_vector).normalized()
    up = camera_vector.cross(right)
    points = []
    for row in frames[::6]:
        T = Matrix(row["support_transform"])
        points.extend(T @ v for v in support_vertices)
        if row["object_visible"] and row["lift_m"] == 0.:
            T = Matrix(row["object_transform"])
            points.extend(T @ v for v in object_vertices)
    xy = [(v.dot(right) * DISPLAY_SCALE, v.dot(up) * DISPLAY_SCALE) for v in points]
    lo = [min(v[k] for v in xy) for k in (0, 1)]
    hi = [max(v[k] for v in xy) for k in (0, 1)]
    center_2d = [(a + b) / 2 for a, b in zip(lo, hi)]
    center = right * center_2d[0] + up * center_2d[1]
    target = center
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    # ortho_scale specifies horizontal width. Keep the entire fixture and both
    # seated object poses visible, with a small margin instead of empty air.
    ratio = scene.render.resolution_x / scene.render.resolution_y
    camera_data.ortho_scale = max(hi[0] - lo[0], (hi[1] - lo[1]) * ratio) * 1.08
    camera.location = target + camera_vector * 15.
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.clip_start, camera_data.clip_end = .01, 100.
    scene.camera = camera
    return dict(location=list(camera.location), target=list(target),
                ortho_scale=camera_data.ortho_scale, direction=list(camera_vector),
                framing="Close-up of all fixture placements and both seated poses; the moving object enters and exits the view")


def main():
    args = arguments()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 32
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = .18
    scene.eevee.gtao_factor = 1.2
    scene.eevee.use_soft_shadows = True
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -1.5
    scene.view_settings.gamma = 1.
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.78, .82, .9, 1.)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .5
    blue = material("Blue rigid support", "#2A91D2", .38, .12)
    gray = material("Gray B object", "#A5ADB5", .5, .08)
    fixture, fixture_vertices = load_mesh(IDEA / f"data/{args.mode}.obj", "Shared rigid fixture", blue)
    obj, object_vertices = load_mesh(IDEA / "data/object_local.obj", "B object", gray, smooth=True)
    # No floor plane, gradient or grid is rendered. The transparent film is
    # composited on exact white during publication; ground checks stay physical.
    area("Key", (4., -5., 8.), 1300., 5., (0., 0., 1.))
    area("Fill", (-5., -1., 4.), 750., 5., (0., 0., 1.))
    area("Rim", (1., 5., 6.), 1100., 4., (0., 0., 1.))
    motion = json.loads((IDEA / f"data/{args.mode}_motion.json").read_text())
    frames = motion["frames"]
    camera = camera_fit(scene, frames, object_vertices, fixture_vertices, args.mode)
    out = IDEA / "data/frames" / args.mode
    out.mkdir(parents=True, exist_ok=True)
    chosen = ([0, int(3.5 * motion["fps"]), int(7.75 * motion["fps"]), int(12.25 * motion["fps"])]
              if args.preview else range(args.start, args.end or len(frames)))
    start = time.monotonic()
    for n in chosen:
        row = frames[n]
        place(fixture, row["support_transform"])
        place(obj, row["object_transform"])
        obj.hide_render = not row["object_visible"]
        scene.render.filepath = str(out / f"{n:05d}.png")
        bpy.ops.render.render(write_still=True)
        if n % 24 == 0 or args.preview:
            print("IDEA FRAME", args.mode, n, row["stage"], "elapsed", round(time.monotonic() - start, 1), flush=True)
    (IDEA / f"data/{args.mode}_camera.json").write_text(json.dumps(camera, indent=2) + "\n")
    print("RENDER COMPLETE", args.mode, flush=True)


if __name__ == "__main__":
    main()
