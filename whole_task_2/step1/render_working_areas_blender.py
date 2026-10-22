"""Text-free work-area closeups and metric robot-context views.

Robot joint postures and display stands are schematic, without IK certification.
Object alignments, colored source faces, tool geometry and workspace sphere
come from the saved Step1.3 candidates. This script does not optimize poses.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"robots_config"))
from arm_visuals import box, camera, cylinder, light, line, material, setup, sphere


def bunny_mesh(vertices, faces, selected, color):
    mesh = bpy.data.meshes.new("source bunny")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new("bunny", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(material("bunny porcelain", (.78, .81, .83), roughness=.60))
    mesh.materials.append(material("working area", color, roughness=.60))
    for polygon in mesh.polygons:
        polygon.material_index = int(polygon.index in selected)
        # Show the actual geometric facets used by the normal checks.
        polygon.use_smooth = False
    return obj


def lights():
    light("key", (-1, -2, 3), (0, -.2, .4), 180, 3)
    light("fill", (2, 1, 2), (0, -.3, .4), 110, 2.5)


def dashed_sphere(center, radius):
    center = Vector(center)
    ink = material("workspace sphere", (.37, .51, .64), roughness=1)
    rings = []
    for degrees in (-50, 0, 50):
        latitude = math.radians(degrees)
        rings.append((center+Vector((0, 0, radius*math.sin(latitude))),
                      Vector((1, 0, 0)), Vector((0, 1, 0)), radius*math.cos(latitude)))
    for degrees in (0, 60, 120):
        longitude = math.radians(degrees)
        rings.append((center, Vector((math.cos(longitude), math.sin(longitude), 0)),
                      Vector((0, 0, 1)), radius))
    for origin, first, second, ring_radius in rings:
        for dash in range(72):
            start = 2*math.pi*dash/72
            finish = 2*math.pi*(dash+.48)/72
            points = [origin+ring_radius*(math.cos(a)*first+math.sin(a)*second)
                      for a in (start, (start+finish)/2, finish)]
            line("workspace sphere dash", points, .00035, ink)


def fixed_work_cone(workspace):
    """Finite work cone ending at the first solid ring.

    Dashed forward extensions are context only and cannot supply work faces.
    This is a station work volume, not a spray fan or a moving flange diagram.
    """
    cone = workspace["spatial_cone"]
    origin = Vector(cone["apex_world_m"])
    forward = Vector(cone["axis_world"]).normalized()
    half_angle_deg = cone["half_angle_deg"]
    length = cone["reference_axial_distance_m"]
    end_length = length+workspace["usable_radius_m"]
    helper = Vector(tuple(1 if i == min(range(3), key=lambda j: abs(forward[j])) else 0
                          for i in range(3)))
    first = forward.cross(helper).normalized()
    second = forward.cross(first).normalized()
    center = origin+forward*length
    radius = length*math.tan(math.radians(half_angle_deg))
    ink = material("fixed spatial work cone", (.85, .56, .18), roughness=1)
    ring = [center+radius*(math.cos(i*2*math.pi/96)*first+math.sin(i*2*math.pi/96)*second)
            for i in range(97)]
    line("work cone reference rim", ring, .00033, ink)
    end_ring = [origin+(point-origin)*(end_length/length) for point in ring]
    sphere("fixed work cone apex", origin, .0022, ink)
    for i in (0, 12, 24, 36, 48, 60, 72, 84):
        line("fixed work cone edge", [origin, ring[i]], .00030, ink)
        for dash in range(12):
            line("work cone forward extension", [ring[i]+(end_ring[i]-ring[i])*dash/12,
                                                   ring[i]+(end_ring[i]-ring[i])*(dash+.55)/12], .00028, ink)
    for i in range(0, 96, 2):
        line("work cone extension rim", end_ring[i:i+2], .00028, ink)
    for i in range(8):
        line("fixed world direction dash", [origin+forward*length*(i/8),
                                             origin+forward*length*((i+.5)/8)], .00025, ink)
    return [origin, *end_ring]


def render(path):
    assert not any(obj.type == "FONT" for obj in bpy.context.scene.objects)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-site-packages", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    sys.path.insert(0, str(args.python_site_packages))
    import numpy as np
    metadata = json.loads(args.context.read_text())
    config = metadata["config"]
    report = metadata["report"]
    arrays = np.load(args.arrays)
    faces = arrays["faces"]
    vertices = arrays["vertices_object"]
    body_up = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    scene_metadata = []
    for ordinal, example in enumerate(report["working_areas"]):
        candidate_id = example["candidate_index"]
        ids = np.array(example["face_ids"])
        selected = set(ids.tolist())
        raw_ids = np.asarray(example["raw_face_ids"])
        color = metadata["colors"][ordinal % len(metadata["colors"])]

        setup(width=1200, height=1200, samples=48)
        close_vertices = vertices @ body_up.T
        bunny_mesh(close_vertices, faces, selected, color)
        lights()
        view = body_up @ np.asarray(example["normal_cap_axis_object"])
        camera(close_vertices, view, 1.13)
        bpy.context.view_layer.update()
        render(args.output/f"{ordinal:02d}_area.png")

        rotation = arrays["candidate_rotation_world_object"][candidate_id]
        translation = arrays["candidate_translation_world_object_m"][candidate_id]
        world_vertices = vertices @ rotation.T+translation
        offsets = arrays["candidate_tool_offsets"]
        outward = arrays["candidate_outward_tool_object"][offsets[candidate_id]:offsets[candidate_id+1]] @ rotation.T
        tool_face_ids = arrays["candidate_tool_face_ids"][offsets[candidate_id]:offsets[candidate_id+1]]
        assert np.array_equal(tool_face_ids, raw_ids)
        centers = world_vertices[faces[raw_ids]].mean(axis=1)
        areas = arrays["face_area_m2"][raw_ids]
        area_center = np.average(centers, weights=areas, axis=0)
        nominal_outward = -np.asarray(config["task_constraint"]["working_direction_world"])
        score = np.linalg.norm(centers-area_center, axis=1)/.06+.4*(1-outward @ nominal_outward)
        contact_index = int(score.argmin())
        display_target = example.get("display_target")
        if display_target:
            contact_index = int(np.flatnonzero(raw_ids == display_target["source_face_id"])[0])
        target = centers[contact_index]
        drawing_config = copy.deepcopy(config)
        drawing_config["task_constraint"]["world_position_constraint"]["center_world_m"] = target.tolist()
        drawing_config["task_constraint"]["working_direction_world"] = (-outward[contact_index]).tolist()

        setup(width=1600, height=1600, samples=48)
        bunny_mesh(world_vertices, faces, selected, color)
        axis = Vector(drawing_config["task_constraint"]["working_direction_world"])
        assert axis.angle(Vector(config["task_constraint"]["working_direction_world"])) < 1e-6
        length = float(config["feasibility"]["tcp_transform"]["translation_flange_m"][2])
        gap = float(config["task_constraint"]["tcp_to_surface_distance_m"]["nominal"])
        tip = Vector(target)-axis*gap
        flange = tip-axis*length
        metal = material("tool steel", (.43, .49, .54), metallic=.65, roughness=.30)
        cylinder("tool flange / schematic", flange-axis*.012, flange, .018, metal)
        cylinder("tool mount", flange, flange+axis*length*.40, .012, metal, .009)
        cylinder("tool shaft", flange+axis*length*.40, tip, .005, metal, .0015)
        sphere("tool tip", tip, .0018, metal)
        lower = world_vertices.min(axis=0)
        upper = world_vertices.max(axis=0)
        top = max(.035, float(lower[2])-.0005)
        center = (lower+upper)/2
        table_material = material("display table", (.89, .91, .93), roughness=.65)
        tabletop_dimensions = (.50, .40, .024)
        center[2] = top-tabletop_dimensions[2]/2
        box("enlarged tabletop / schematic", center, tabletop_dimensions,
            table_material, radius=.004)
        pedestal_height = max(.01, top-tabletop_dimensions[2])
        center[2] = pedestal_height/2
        box("table pedestal / schematic", center, (.12, .12, pedestal_height), table_material, radius=.004)
        workspace = report["workspace"]
        work_center = np.asarray(workspace["center_world_m"])
        radius = workspace["usable_radius_m"]
        configured_axis = config["task_constraint"]["working_direction_world"]
        configured_half_angle = config["task_constraint"]["cone_half_angle_deg"]
        cone_points = fixed_work_cone(workspace)
        lights()
        # Frame the object, fixed cone and tool. The work ball is not drawn.
        bpy.context.view_layer.update()
        extent = [Vector(point) for point in world_vertices]
        extent.extend(flange+Vector((x, y, z))*.022
                      for x in (-1, 1) for y in (-1, 1) for z in (-1, 1))
        extent.extend([tip, flange-axis*.012])
        extent.extend(cone_points)
        cam = camera(extent, (-1.5, .65, .45), 1.13)
        bpy.context.view_layer.update()
        assert not any(obj.name.startswith("workspace sphere") for obj in bpy.context.scene.objects)
        render(args.output/f"{ordinal:02d}_robot.png")
        scene_metadata.append(dict(candidate_index=candidate_id,
                                   aimed_site=display_target["site"] if display_target else None,
                                   contact_source_face_id=int(raw_ids[contact_index]),
                                   contact_surface_world_m=target.tolist(),
                                   displayed_flange_world_m=list(flange),
                                   displayed_tcp_world_m=list(tip),
                                   displayed_tool_direction_world=list(axis),
                                   tool_direction_fixed_for_entire_region=True,
                                   surface_normal_half_angle_deg=report["angular_budget"]["full_normal_cap_half_angle_deg"],
                                   one_color_includes_selected_region_and_initialization_fills=True,
                                   work_ball_drawn=False,
                                   surface_shading="flat_matches_source_face_normals",
                                   sphere_center_world_m=work_center.tolist(), sphere_radius_m=radius,
                                   configured_tool_direction_world=configured_axis,
                                   tool_direction_cone_half_angle_deg=configured_half_angle,
                                   reserved_tool_direction_half_angle_deg=report["angular_budget"]["working_tool_cone_half_angle_deg"],
                                   spatial_cone_apex_world_m=workspace["spatial_cone"]["apex_world_m"],
                                   spatial_cone_axis_world=workspace["spatial_cone"]["axis_world"],
                                   spatial_cone_half_angle_deg=configured_half_angle,
                                   spatial_cone_usable_half_angle_deg=workspace["spatial_cone"]["usable_half_angle_deg"],
                                   spatial_cone_reference_distance_m=workspace["spatial_cone"]["reference_axial_distance_m"],
                                   spatial_cone_maximum_work_depth_m=workspace["spatial_cone"]["maximum_axial_distance_m"],
                                   dashed_extension_selectable=False,
                                   spatial_cone_fixed_across_all_candidates=True,
                                   displayed_tool_deviation_deg=math.degrees(axis.angle(Vector(configured_axis))),
                                   direction_cone_semantics="first_solid_ring_finite_work_cone_dashed_extension_not_selectable",
                                   view_scope="fixed_cone_and_tool_end_without_ball",
                                   camera_orthographic_width_m=cam.data.ortho_scale,
                                   tabletop_dimensions_m=list(tabletop_dimensions),
                                   robot_posture="not_shown_only_schematic_tool_end",
                                   support="display_table_only_not_a_fixture_design"))
        print(f"Rendered area / robot pair {ordinal+1}", flush=True)
    (args.output/"render_metadata.json").write_text(json.dumps(scene_metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
