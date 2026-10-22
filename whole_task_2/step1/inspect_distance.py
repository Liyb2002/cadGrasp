"""Explain Step1.3 distances and the artificial compact-ear size restriction.

Adds diagnostic figures without changing the four saved working regions.
Run: .venv/bin/python -m whole_task_2.step1.inspect_distance
"""
import argparse
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from robots_config.constraints import surface_point_allowed, tool_direction_allowed_at_surface
from .common import DEFAULT_OUTPUT, ROOT, load_shape, sha256, write_json
from .render import COLORS, _draw_shape
from .step1_3 import (_adjacency, _clipped_visibility, _component, _seed_faces,
                      _visible_faces, approach_directions, requested_region_mask)
from .work_disks import clip_surface_disk


def screen_component(mesh, site_mask, seed, center, axis, config, budget,
                     neighbors, check_angle=True, check_depth=True):
    window = config["task_constraint"]["world_position_constraint"]
    available = site_mask.copy()
    if check_angle:
        available &= mesh.face_normals @ axis >= np.cos(np.radians(
            budget["working_normal_cap_half_angle_deg"]))-1e-12
    delta = mesh.vertices-center
    depth = delta @ axis
    if check_depth:
        available &= np.max(np.abs(depth[mesh.faces]), axis=1) <= window["normal_tolerance_m"]+1e-12
    radial = np.linalg.norm(delta-depth[:, None]*axis, axis=1)
    radius = .95*window["radius_m"]
    available &= radial[mesh.faces].max(axis=1) <= radius+1e-12
    ids = _component(seed, available, neighbors)
    if not len(ids):
        return ids
    directions = approach_directions(mesh.face_normals[ids], axis,
                                      budget["working_tool_cone_half_angle_deg"])
    visible = _visible_faces(mesh, ids, directions)
    available[:] = False
    available[ids[visible]] = True
    return _component(seed, available, neighbors)


def upright_alignment(axis, world_normal):
    rotation = Rotation.align_vectors([world_normal], [axis])[0].as_matrix()
    current_up = rotation @ np.array([0., 1., 0.])
    current_up -= (current_up @ world_normal)*world_normal
    desired_up = np.array([0., 0., 1.])
    desired_up -= (desired_up @ world_normal)*world_normal
    current_up /= np.linalg.norm(current_up)
    desired_up /= np.linalg.norm(desired_up)
    roll = np.arctan2(world_normal @ np.cross(current_up, desired_up), current_up @ desired_up)
    return Rotation.from_rotvec(world_normal*roll).as_matrix() @ rotation


def _font(size, bold=False):
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf", size)


def _label(draw, position, text, color="#315e83", size=26, anchor="mm"):
    bounds = draw.textbbox(position, text, font=_font(size, True), anchor=anchor)
    draw.rounded_rectangle((bounds[0]-8, bounds[1]-5, bounds[2]+8, bounds[3]+5),
                           radius=7, fill="white")
    draw.text(position, text, fill=color, font=_font(size, True), anchor=anchor)


def coverage_figure(mesh, saved, ear_index, same_ids, candidate_ids, ear, candidate, output):
    selected = saved["selected_triangles_object"]
    outside = saved["unselected_triangles_object"]
    old_labels = np.where(saved["selected_triangle_region_id"] == ear_index, 0, -1)
    original_display = dict(triangles=np.concatenate([outside, selected]),
                            labels=np.r_[np.full(len(outside), -1), old_labels])
    fig = plt.figure(figsize=(18, 9), facecolor="white")
    fig.text(.04, .94, "Why was the ear working area so small?", fontsize=26, weight="bold", color="#25364b")
    titles = ["Saved compact region", "Same pose, natural boundary", "Different ear-side alignment"]
    values = [ear["area_m2"], mesh.area_faces[same_ids].sum(), candidate["area_m2"]]
    for i in range(3):
        fig.text(.045+i/3, .86, titles[i], fontsize=16, color=COLORS[i], weight="bold")
        fig.text(.045+i/3, .815, f"{values[i]*1e4:.2f} cm²", fontsize=22, color=COLORS[i])
        ax = fig.add_axes([i/3, .16, 1/3, .64], projection="3d")
        if i == 0:
            display = original_display
        else:
            ids = same_ids if i == 1 else candidate_ids
            labels = np.full(len(mesh.faces), -1)
            labels[ids] = i
            display = dict(triangles=mesh.triangles, labels=labels)
        _draw_shape(ax, mesh, np.full(len(mesh.faces), -1), display=display)
        normal = np.array(ear["window_normal_object"] if i < 2 else candidate["normal_object"])
        ax.view_init(elev=np.degrees(np.arcsin(normal[1])),
                     azim=np.degrees(np.arctan2(normal[0], normal[2])), vertical_axis="y")
    fig.text(.045, .10, "All cases keep the same 583 mm base-to-window distance, 25° direction cone and ±5 mm window depth.",
             fontsize=12, color="#25364b")
    fig.text(.045, .055, "Panel 2 removes the near-circle requirement. Panel 3 also changes the local alignment within the same upper ear.",
             fontsize=11, color="#687687")
    fig.text(.045, .02, "Geometric candidates only; a complete tool path and actual half-ear coverage have not been certified.",
             fontsize=10, color="#687687")
    fig.savefig(output / "ear_coverage_comparison.png", dpi=150, facecolor="white")
    plt.close(fig)


def compose_distance_figure(output, report):
    data = output / "distance_data"
    projections = json.loads((data / "projections.json").read_text())
    wide = Image.open(data / "wide.png").convert("RGB")
    close = Image.open(data / "closeup.png").convert("RGB")
    for raw, key in [(wide, "wide"), (close, "closeup")]:
        draw = ImageDraw.Draw(raw)
        coordinates = projections[key]
        if key == "wide":
            _label(draw, coordinates["base_distance"], "583 mm · base to ear center", size=25)
            _label(draw, coordinates["window"], "Ø 200 mm task window", size=21)
        else:
            _label(draw, coordinates["tool_length"], "120 mm tool", size=24)
            point = coordinates["contact"]
            text_point = [min(raw.width-280, point[0]+160), min(raw.height-55, point[1]+85)]
            draw.line([tuple(point), tuple(text_point)], fill="#c66e2f", width=3)
            _label(draw, text_point, "0 mm · contact tip", color="#c66e2f", size=23)
    poster = Image.new("RGB", (2400, 1560), "white")
    draw = ImageDraw.Draw(poster)
    draw.text((95, 45), "STEP 1.3  /  ROBOT–BUNNY DISTANCES", font=_font(24), fill="#687687")
    draw.text((95, 95), "A larger ear patch at the same workstation", font=_font(45, True), fill="#25364b")
    draw.text((95, 160), "KUKA LBR Med 14 R820 · 25° tool direction cone · metric scale", font=_font(27), fill="#687687")
    poster.paste(wide.resize((1500, 1100), Image.Resampling.LANCZOS), (0, 240))
    draw.text((1550, 245), "Current tool / ear detail", font=_font(27, True), fill="#25364b")
    poster.paste(close.resize((850, 710), Image.Resampling.LANCZOS), (1500, 290))
    reference = Image.open(data / "idea_reference.png").convert("RGB")
    poster.paste(reference.resize((600, 375), Image.Resampling.LANCZOS), (1600, 1030))
    draw.text((1535, 1410), "Idea video: Panda, 52 mm spray gap", font=_font(25), fill="#687687")
    draw.text((95, 1370), f'Broader ear candidate: {report["broader_ear_candidate"]["area_m2"]*1e4:.2f} cm²',
              font=_font(31, True), fill="#278c83")
    draw.text((95, 1420), "583 mm is a base-to-workpoint distance; 0 mm is a tool-to-surface gap.",
              font=_font(24), fill="#25364b")
    draw.text((95, 1470), "Robot posture and display stand are schematic. Working pose, support and continuous motion remain to be solved.",
              font=_font(21), fill="#687687")
    poster.save(output / "distance_overview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    stage = args.output_root / "B/step1"
    output = stage / "step1.3"
    data = output / "distance_data"
    data.mkdir(exist_ok=True)
    config_path = stage / "step1.1/config.json"
    config = json.loads(config_path.read_text())
    report_path = output / "working_areas.json"
    base = json.loads(report_path.read_text())
    if sha256(config_path) != base["config_sha256"] or sha256(ROOT / "objects/B/mesh.stl") != base["shape_sha256"]:
        raise ValueError("Step1.3 inputs changed; refresh the working regions first")
    saved = np.load(output / "working_areas.npz")
    mesh = load_shape(ROOT / "objects/B/mesh.stl")
    ear_index = next(i for i, value in enumerate(base["working_areas"]) if value["name"] == "ear")
    ear = base["working_areas"][ear_index]
    neighbors = _adjacency(mesh)
    budget = base["angular_budget"]
    site = requested_region_mask(mesh, ear["requested_site"])
    center = np.array(ear["center_object_m"])
    axis = np.array(ear["normal_cap_axis_object"])
    seed = ear["seed_face_id"]
    same_ids = screen_component(mesh, site, seed, center, axis, config, budget, neighbors)
    no_depth = screen_component(mesh, site, seed, center, axis, config, budget, neighbors, check_depth=False)
    no_angle = screen_component(mesh, site, seed, center, axis, config, budget, neighbors, check_angle=False)
    height = float(ear["requested_site"]["bounds_object_m"][0][1])
    component = _component(seed, mesh.triangles_center[:, 1] >= height, neighbors)
    upper_ear = np.zeros(len(mesh.faces), dtype=bool)
    upper_ear[component] = True
    best = None
    for candidate_seed in _seed_faces(mesh, upper_ear):
        candidate_center = mesh.triangles_center[candidate_seed]
        candidate_axis = mesh.face_normals[candidate_seed]
        ids = screen_component(mesh, upper_ear, candidate_seed, candidate_center, candidate_axis,
                               config, budget, neighbors)
        area = float(mesh.area_faces[ids].sum())
        if not len(ids) or (best is not None and area <= best["area_m2"]):
            continue
        geometry = clip_surface_disk(mesh, ids, candidate_center, candidate_axis,
                                      .95*base["work_window"]["radius_m"], minimum_fill=0)
        if geometry is None:
            continue
        directions = approach_directions(mesh.face_normals[geometry["source_face_ids"]], candidate_axis,
                                          budget["working_tool_cone_half_angle_deg"])
        if not _clipped_visibility(mesh, geometry, directions):
            continue
        best = dict(area_m2=geometry["area_m2"], seed_face_id=int(candidate_seed),
                     center_object_m=candidate_center.tolist(), normal_object=candidate_axis.tolist(),
                     source_face_ids=ids.tolist(), geometry=geometry)
    if best is None:
        raise ValueError("No broader connected ear candidate found")
    task = config["task_constraint"]
    world_normal = np.asarray(base["work_window"]["outward_normal_world"])
    world_center = np.asarray(base["work_window"]["center_world_m"])
    world_axis = np.asarray(task["working_direction_world"])
    if not np.allclose(world_normal, -world_axis):
        raise ValueError("This diagnostic expects the selected aligned-normal 04 workstation")
    rotation = upright_alignment(np.array(best["normal_object"]), world_normal)
    translation = world_center-rotation @ best["center_object_m"]
    world_points = best["geometry"]["triangles"].reshape(-1, 3) @ rotation.T+translation
    if not all(surface_point_allowed(config, point) for point in world_points):
        raise ValueError("Broader ear candidate left the configured world window")
    directions = approach_directions(mesh.face_normals[best["geometry"]["source_face_ids"]],
                                      np.array(best["normal_object"]), budget["working_tool_cone_half_angle_deg"])
    if not all(tool_direction_allowed_at_surface(config, d, n) for d, n in zip(
            -directions @ rotation.T, mesh.face_normals[best["geometry"]["source_face_ids"]] @ rotation.T)):
        raise ValueError("Broader ear candidate violated direction constraints")
    np.savez_compressed(data / "scene.npz", vertices_object=mesh.vertices, faces=mesh.faces,
                        broader_ear_source_face_ids=np.array(best["source_face_ids"]),
                        same_pose_connected_source_face_ids=same_ids,
                        broader_ear_triangles_object=best["geometry"]["triangles"],
                        broader_ear_barycentric=best["geometry"]["barycentric"],
                        broader_ear_triangle_source_face_ids=best["geometry"]["source_face_ids"],
                        broader_ear_approach_object=directions,
                        rotation_world_object=rotation, translation_world_object_m=translation)
    video = ROOT / "whole_task_2/idea/robot_access_spray.mp4"
    motion_path = ROOT / "whole_task_2/assets/robot_access_spray/motion.json"
    motion = json.loads(motion_path.read_text())
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "0", "-i", str(video),
                    "-frames:v", "1", str(data / "idea_reference.png")], check=True)
    public_candidate = {key: value for key, value in best.items() if key != "geometry"}
    public_candidate.update(rotation_world_object=rotation.tolist(), translation_world_object_m=translation.tolist(),
                            upper_ear_height_threshold_m=height,
                            upper_ear_component_area_m2=float(mesh.area_faces[component].sum()),
                            fraction_of_this_upper_ear_component=best["area_m2"]/float(mesh.area_faces[component].sum()),
                            exact_half_of_entire_ear_certified=False,
                            status="supplemental_geometric_candidate_not_a_replacement_of_the_four_saved_regions")
    report = dict(schema_version=1, config_id=config["id"], config_snapshot=str(config_path.resolve()),
        base_to_processing_center_m=base["work_window"]["base_to_center_distance_m"],
        tool_length_m=config["feasibility"]["tcp_transform"]["translation_flange_m"][2],
        tcp_to_surface_distance_m=task["tcp_to_surface_distance_m"]["nominal"],
        window_radius_m=base["work_window"]["radius_m"],
        ear_ablation=dict(saved_compact_area_m2=ear["area_m2"],
                          same_pose_without_compactness_m2=float(mesh.area_faces[same_ids].sum()),
                          same_pose_without_depth_limit_m2=float(mesh.area_faces[no_depth].sum()),
                          same_pose_without_direction_limit_m2=float(mesh.area_faces[no_angle].sum())),
        broader_ear_candidate=public_candidate,
        video_reference=dict(video=str(video.relative_to(ROOT)),
                             tcp_to_surface_distance_m=motion["standoff_m"],
                             robot="Franka Panda", scope="different historical station; image reference only"),
        distance_semantics="base to processing reference point, distinct from tool-tip standoff",
        geometric_window_coverage_does_not_predict_distance_dependent_ik_reachability=True,
        drawing_scope="Configuration-scale schematic robot posture and display stand; exact configured distances. No continuous motion or support certificate.",
        input_sha256={str(path.resolve()): sha256(path) for path in [config_path, report_path,
            output / "working_areas.npz", ROOT / "objects/B/mesh.stl", video, motion_path]},
        source_sha256={name: sha256(Path(__file__).parent / name)
                       for name in ["inspect_distance.py", "render_distance_blender.py"]})
    write_json(data / "context.json", report)
    coverage_figure(mesh, saved, ear_index, same_ids, np.array(best["source_face_ids"]), ear, best, output)
    log = data / "render.log"
    with log.open("w") as stream:
        subprocess.run(["blender", "-b", "-t", "8", "--factory-startup", "--python-exit-code", "1",
                        "--python", str(Path(__file__).with_name("render_distance_blender.py")),
                        "--", "--data", str(data.resolve()),
                        "--python-site-packages", str(Path(np.__file__).parent.parent)],
                        stdout=stream, stderr=subprocess.STDOUT, check=True)
    compose_distance_figure(output, report)
    report["artifact_sha256"] = {name: sha256(output / name) for name in [
        "distance_overview.png", "ear_coverage_comparison.png", "distance_data/scene.npz",
        "distance_data/context.json", "distance_data/wide.png", "distance_data/closeup.png",
        "distance_data/projections.json", "distance_data/idea_reference.png"]}
    write_json(output / "distance_analysis.json", report)
    print(json.dumps(dict(same_pose_area_cm2=report["ear_ablation"]["same_pose_without_compactness_m2"]*1e4,
                          broader_candidate_area_cm2=best["area_m2"]*1e4,
                          distance_figure=str(output / "distance_overview.png")), indent=2), flush=True)


if __name__ == "__main__":
    main()
