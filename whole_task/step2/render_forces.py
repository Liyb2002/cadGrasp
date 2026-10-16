"""Render saved Step2.2 processing-force directions and the fixed-axis needle.

The numerical load outputs are read only. Arrows terminate at saved surface
samples and have equal display lengths; their directions come from force_push_mg.
Run with the repository Python; it invokes Blender for the four metric scenes.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
COLORS = ("#e89732", "#3ba48e", "#4c8dcc", "#a379bd")
VIEW = (-1.3, 1.0, .75)
ARROW_LENGTH = .025


def prepare(output_root, object_name):
    import numpy as np
    from whole_task.step1.common import sha256, write_json
    from whole_task.step2.loads_adapter import load_task

    base = Path(output_root)/object_name/"step2"
    pose_path = base/"step2.1/object_poses.json"
    poses = json.loads(pose_path.read_text())
    config_path = base.parent/"step1/step1.1/config.json"
    config = json.loads(config_path.read_text())
    gap = float(config["task_constraint"]["tcp_to_surface_distance_m"]["nominal"])
    length = float(config["feasibility"]["tcp_transform"]["translation_flange_m"][2])
    view = np.asarray(VIEW)/np.linalg.norm(VIEW)
    right = np.cross([0., 0., 1.], view); right /= np.linalg.norm(right)
    up = np.cross(view, right)
    basis = np.column_stack((right, up))
    reference = np.asarray(poses["fixed_station"]["reference_surface_point_world_m"])
    scenes, framing, hashes = [], [], {}
    for path in (pose_path, config_path, base/"step2.2/loads.json"):
        hashes[str(path.resolve())] = sha256(path)
    for i, pose in enumerate(poses["poses"]):
        task = load_task(pose["id"], output_root, object_name)
        mesh, axis = task.domain.mesh, task.domain.axis
        sample_path = base/"step2.2"/pose["id"]/"samples.npz"
        with np.load(sample_path) as z:
            points, forces = z["pt_m"].copy(), z["force_push_mg"].copy()
            source_faces = z["source_face_id"].copy()
            screened = z["geometrically_screened"].copy()
        directions = forces/np.linalg.norm(forces, axis=1)[:, None]
        np.testing.assert_allclose(directions, np.tile(axis, (len(points), 1)), atol=1e-13, rtol=0)
        # Choose readable representatives, without changing the load family.
        pool = np.arange(0, len(points), 8)
        normals = mesh.face_normals[source_faces[pool]]
        pool = pool[(normals @ view) > .12]
        origins = points[pool]+1e-5*mesh.face_normals[source_faces[pool]]
        visible = ~mesh.ray.intersects_any(origins, np.tile(view, (len(pool), 1)))
        pool = pool[visible]
        if not len(pool):
            raise ValueError(f"No visible samples in {pose['id']}")
        target_data = pose.get("display_target")
        if target_data:
            target = np.asarray(target_data["point_world_m"])
        else:
            candidates = pool[screened[pool]]
            if not len(candidates):
                raise ValueError(f"No screened needle position in {pose['id']}")
            target = points[candidates[np.argmin(np.linalg.norm(points[candidates]-reference, axis=1))]]
        # Avoid drawing force arrows directly underneath the illustrated shaft.
        xy = points[pool] @ basis
        delta = points[pool]-target
        needle_distance = np.linalg.norm(delta-(delta @ axis)[:, None]*axis, axis=1)
        keep = needle_distance > .006
        pool, xy = pool[keep], xy[keep]
        desired = 18 if i == 0 else 28
        selected = [int(np.argmin(np.linalg.norm(xy-xy.mean(axis=0), axis=1)))]
        distances = np.linalg.norm(xy-xy[selected[0]], axis=1)
        for _ in range(min(desired, len(pool))-1):
            j = int(np.argmax(distances))
            if distances[j] < .004:
                break
            selected.append(j)
            distances = np.minimum(distances, np.linalg.norm(xy-xy[j], axis=1))
        ids = pool[selected]
        heads = points[ids]
        tails = heads-ARROW_LENGTH*directions[ids]
        np.testing.assert_allclose((heads-tails)/ARROW_LENGTH, directions[ids], atol=1e-13, rtol=0)
        tip = target-gap*axis
        flange = tip-length*axis
        framing.extend(mesh.vertices.tolist())
        framing.extend(tails.tolist())
        framing.extend((flange-.016*axis+np.array([x,y,z])*.015).tolist()
                       for x in (-1,1) for y in (-1,1) for z in (-1,1))
        scenes.append(dict(pose=pose["id"], color=COLORS[i], vertices=mesh.vertices.tolist(),
            faces=mesh.faces.tolist(), work_face_ids=task.domain.work_ids.tolist(),
            force_axis_world=axis.tolist(), arrow_sample_ids=ids.tolist(),
            arrow_tails_world_m=tails.tolist(), arrow_heads_world_m=heads.tolist(),
            arrow_processing_forces_N=(forces[ids]*task.domain.weight_N).tolist(),
            needle_surface_point_world_m=target.tolist(), needle_tip_world_m=tip.tolist(),
            needle_flange_world_m=flange.tolist(), needle_length_m=length,
            selected_sample_direction_max_error=float(np.max(np.abs(directions[ids]-axis)))))
        hashes[str(sample_path.resolve())] = sha256(sample_path)
    out = base/"step2.2/force_directions"
    out.mkdir(exist_ok=True)
    data = dict(scenes=scenes, frame_points=framing, view_world=VIEW,
        arrow_display_length_m=ARROW_LENGTH, arrow_heads_are_saved_surface_points=True,
        arrow_semantics="processing force ON the object, from force_push_mg; uniform display length",
        sample_selection="visible and spaced representatives for drawing only; loads unmodified",
        needle_semantics="one scan position per pose; fixed direction, translating tool",
        needle_body_geometry="schematic; axis, TCP and flange-to-TCP length from current inputs",
        same_world_camera_and_scale=True, force_magnitudes_visualized=False,
        input_sha256=hashes, renderer_sha256=sha256(__file__))
    write_json(out/"scene.json", data)
    return out


def render_blender(scene_path):
    import bpy
    from mathutils import Vector
    sys.path.insert(0, str(ROOT/"robots_config"))
    from arm_visuals import setup, material, cylinder, sphere, camera, light

    data = json.loads(Path(scene_path).read_text())
    def rgb(color):
        return tuple(int(color[j:j+2], 16)/255 for j in (1, 3, 5))
    for row in data["scenes"]:
        scene = setup(width=1200, height=1050, samples=64)
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 8
        mesh = bpy.data.meshes.new("current world-pose bunny")
        mesh.from_pydata(row["vertices"], [], row["faces"]); mesh.update()
        obj = bpy.data.objects.new("bunny", mesh); bpy.context.collection.objects.link(obj)
        mesh.materials.append(material("bunny porcelain", (.55, .58, .61), roughness=.7))
        mesh.materials.append(material("selected working surface", rgb(row["color"]), roughness=.7))
        selected = set(row["work_face_ids"])
        for polygon in mesh.polygons:
            polygon.material_index = int(polygon.index in selected)
            polygon.use_smooth = False
        red = material("processing force arrows", (.88, .035, .025), roughness=.48)
        axis = Vector(row["force_axis_world"])
        for tail, head in zip(row["arrow_tails_world_m"], row["arrow_heads_world_m"]):
            tail, head = Vector(tail), Vector(head)
            neck = head-axis*.0065
            cylinder("force shaft", tail, neck, .00065, red, bevel=False)
            cylinder("force arrowhead at surface sample", neck, head, .0022, red, 0, bevel=False)
        steel = material("needle steel", (.31, .36, .42), metallic=.75, roughness=.3)
        dark = material("tool holder", (.15, .19, .25), metallic=.35, roughness=.4)
        gold = material("tool collar", (.88, .54, .16), metallic=.55, roughness=.35)
        tip, flange = Vector(row["needle_tip_world_m"]), Vector(row["needle_flange_world_m"])
        length = row["needle_length_m"]
        cylinder("flange", flange-axis*.012, flange, .013, dark)
        cylinder("holder", flange, flange+axis*(length*.35), .009, dark, .006)
        cylinder("collar", flange+axis*(length*.32), flange+axis*(length*.39), .0068, gold)
        cylinder("needle", flange+axis*(length*.39), tip-axis*.009, .0032, steel, .0015)
        cylinder("needle point", tip-axis*.009, tip, .0015, steel, 0, bevel=False)
        sphere("contact point", tip, .0013, red)
        light("key", (-.7, .5, 1.3), (0, -.5, .3), 30, 1.0)
        light("fill", (.8, -.1, .9), (0, -.5, .3), 18, 1.1)
        light("rim", (.1, -1.4, 1.), (0, -.5, .3), 22, 1.)
        camera(data["frame_points"], data["view_world"], margin=1.12)
        assert not any(obj.type == "FONT" for obj in scene.objects)
        scene.render.filepath = str(Path(scene_path).parent/f"{row['pose']}.png")
        bpy.ops.render.render(write_still=True)


def main():
    from PIL import Image
    from whole_task.step1.common import DEFAULT_OUTPUT, sha256, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object", default="B")
    args = parser.parse_args()
    out = prepare(args.output_root, args.object)
    with (out/"render.log").open("w") as log:
        subprocess.run(["blender", "--background", "--threads", "8", "--python-exit-code", "1",
            "--python", str(Path(__file__).resolve()), "--", "--blender-scene", str(out/"scene.json")],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads((out/"scene.json").read_text())
    sheet = Image.new("RGB", (2400, 2100), "white")
    for i, row in enumerate(data["scenes"]):
        with Image.open(out/f"{row['pose']}.png") as tile:
            sheet.paste(tile, ((i%2)*1200, (i//2)*1050))
    destination = out.parent/"force_directions.png"
    sheet.save(destination)
    for path, expected in data["input_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Load input changed while rendering: {path}")
    write_json(out/"visualization.json", dict(status="rendered", image=str(destination),
        image_sha256=sha256(destination), scene_sha256=sha256(out/"scene.json"),
        poses=[dict(pose=row["pose"], arrow_count=len(row["arrow_sample_ids"]),
                    maximum_direction_error=row["selected_sample_direction_max_error"])
               for row in data["scenes"]], all_load_inputs_unchanged=True))
    print(destination, flush=True)


if __name__ == "__main__":
    if "--blender-scene" in sys.argv:
        render_blender(sys.argv[sys.argv.index("--blender-scene")+1])
    else:
        main()
