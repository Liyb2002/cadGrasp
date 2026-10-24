"""Render stage results from the actual input and selected mesh triangles."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image

from .common import sha256


COLORS = ["#ed8b42", "#299c95", "#547fc2", "#a278b5", "#ccaa3f", "#c9687f"]
INK = "#25364b"
MUTED = "#687687"


def _text(fig, x, y, text, size=12, color=INK, **kwargs):
    fig.text(x, y, text, fontsize=size, color=color,
             fontfamily="DejaVu Sans", **kwargs)


def _cone_diagram(ax, config):
    axis = np.array(config["task_constraint"]["working_direction_world"])
    alpha = np.radians(config["task_constraint"]["cone_half_angle_deg"])
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    t, p = np.meshgrid(np.linspace(0, alpha, 25), np.linspace(0, 2*np.pi, 60))
    cap = (np.cos(t)[..., None] * axis
           + np.sin(t)[..., None] * (np.cos(p)[..., None]*first
                                     + np.sin(p)[..., None]*second))
    ax.plot_surface(*np.moveaxis(cap, -1, 0), color=COLORS[0], alpha=.65)
    ax.quiver(0, 0, 0, *axis, color=COLORS[0], linewidth=3)
    for i, label in enumerate("XYZ"):
        vector = np.eye(3)[i]
        ax.quiver(0, 0, 0, *vector, color="#b4bfca", linewidth=1)
        ax.text(*(vector*1.12), label, color=MUTED)
    ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2))
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def render_configuration(config, source, destination):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    _text(fig, .055, .935, "STEP 1.1  /  USER CONFIGURATION", 12, MUTED)
    _text(fig, .055, .875, config["name"], 27, weight="bold")
    # Use an existing arm illustration only when it is bound to this exact config.
    picture = None
    manifest = source.parent / "figures/manifest.json"
    if manifest.is_file() and config.get("figure"):
        entries = json.loads(manifest.read_text())["entries"]
        for entry in entries:
            candidate = source.parent / entry["figure"]
            if (entry["config_sha256"] == sha256(source) and candidate.is_file()
                    and entry["figure_sha256"] == sha256(candidate)):
                picture = Image.open(candidate).convert("RGB")
                # The current figure has a title above its 1000-pixel illustration.
                picture = picture.crop((0, 120, picture.width, picture.height))
                break
    if picture is not None:
        ax = fig.add_axes([.0, .125, .68, .70])
        ax.imshow(picture)
        ax.set_axis_off()
    else:
        ax = fig.add_axes([.03, .14, .60, .65], projection="3d")
        _cone_diagram(ax, config)
    task = config["task_constraint"]
    spec = config["robot_specification"]
    x = .705
    _text(fig, x, .735, "TOOL DIRECTION", 11, MUTED)
    _text(fig, x, .675, f'±{task["cone_half_angle_deg"]:g}°', 36, COLORS[0], weight="bold")
    _text(fig, x, .602, "Surface incidence tolerance", 11, MUTED)
    _text(fig, x, .555, f'±{task["surface_incidence_half_angle_deg"]:g}°', 24)
    axis = task["working_direction_world"]
    _text(fig, x, .47, "Fixed world axis · tool → surface", 10, MUTED)
    _text(fig, x, .425, "[" + ", ".join(f"{v:.3f}" for v in axis) + "]", 13)
    _text(fig, x, .335, f'{spec["dof"]} joints  ·  nominal reach {spec["reach_m"]:g} m', 12)
    window = task.get("world_position_constraint")
    if window is not None:
        _text(fig, x, .275, f'Work window: Ø {2000*window["radius_m"]:g} mm', 12)
        _text(fig, x, .23, f'Base → window center: {1000*window["base_to_center_distance_m"]:.0f} mm', 11)
        gap = task["tcp_to_surface_distance_m"]
        _text(fig, x, .185, f'TCP → surface: {1000*gap["nominal"]:g} mm', 11)
        length = config["feasibility"]["tcp_transform"]["translation_flange_m"][2]
        _text(fig, x, .14, f'Tool length: {1000*length:g} mm', 11)
    else:
        _text(fig, x, .265, "Input snapshot saved as config.json", 10, MUTED)
    _text(fig, .055, .07, "An example user input, selected from robots_config/", 12)
    _text(fig, .055, .035, "Planning is reserved for Step 1.2. Arm geometry is illustrative.", 10, MUTED)
    fig.savefig(destination, dpi=150, facecolor="white")
    plt.close(fig)


def _draw_shape(ax, mesh, labels, focus=None, display=None):
    triangles = mesh.triangles if display is None else display["triangles"]
    face_labels = labels if display is None else display["labels"]
    rgb = np.tile(colors.to_rgb("#dce1e6"), (len(triangles), 1))
    for label in np.unique(face_labels[face_labels >= 0]):
        if focus is None or label == focus:
            rgb[face_labels == label] = colors.to_rgb(COLORS[int(label) % len(COLORS)])
    light = np.array([.35, .6, .72])
    light /= np.linalg.norm(light)
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    illumination = .70 + .30*np.clip(normals @ light, 0, 1)
    rgb *= illumination[:, None]
    collection = Poly3DCollection(triangles, facecolors=rgb,
                                 edgecolors=(.2, .25, .3, .10), linewidths=.08)
    ax.add_collection3d(collection)
    center = mesh.bounds.mean(axis=0)
    radius = max(mesh.extents)*.52
    ax.set(xlim=(center[0]-radius, center[0]+radius),
           ylim=(center[1]-radius, center[1]+radius),
           zlim=(center[2]-radius, center[2]+radius))
    ax.set_box_aspect((1, 1, 1), zoom=1.23)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def render_working_areas(mesh, labels, report, output, display=None):
    areas = report["working_areas"]
    task = report["angular_budget"]
    inset = 100*task["inset_fraction"]
    named_sites = report.get("region_request_source") is not None
    fig = plt.figure(figsize=(18, 12) if named_sites else (18, 10), facecolor="white")
    _text(fig, .055, .94, "STEP 1.3  /  CANDIDATE WORKING AREAS", 12, MUTED)
    _text(fig, .055, .895, f'B · {len(areas)} surface working areas', 27, weight="bold")
    _text(fig, .055, .855, f'{report["robot_name"]}  ·  fixed workstation  ·  {inset:g}% radius and angle margin', 13)
    if named_sites:
        for i, area in enumerate(areas):
            column, row = i % 2, i // 2
            x = .055+.50*column
            y = .805-.37*row
            color = COLORS[i % len(COLORS)]
            _text(fig, x, y, f'A{i+1:02d} · {area["label_en"]}', 20, color, weight="bold")
            _text(fig, x, y-.027,
                  f'Surface {area["area_m2"]*1e4:.1f} cm²  ·  envelope Ø {area["radius_m"]*2000:.1f} mm', 11)
            ax = fig.add_axes([.005+.50*column, .475-.37*row, .49, .29], projection="3d")
            _draw_shape(ax, mesh, labels, focus=i, display=display)
            site = area.get("requested_site") or {}
            axis = np.array(site.get("display_view_object", area["window_normal_object"]), dtype=float)
            axis /= np.linalg.norm(axis)
            ax.view_init(elev=np.degrees(np.arcsin(np.clip(axis[1], -1, 1))),
                         azim=np.degrees(np.arctan2(axis[0], axis[2])), vertical_axis="y")
    else:
        for i, azimuth in enumerate((-50, 130)):
            ax = fig.add_axes([.0+.39*i, .18, .42, .63], projection="3d")
            _draw_shape(ax, mesh, labels, display=display)
            ax.view_init(elev=18, azim=azimuth, vertical_axis="y")
            _text(fig, .16+.39*i, .165, f'VIEW {i+1}', 10, MUTED)
        for i, area in enumerate(areas):
            y = .73-i*min(.135, .51/max(len(areas), 1))
            color = COLORS[i % len(COLORS)]
            _text(fig, .825, y, f'A{i+1:02d}', 19, color, weight="bold")
            _text(fig, .825, y-.035, f'{area["area_m2"]*1e4:.1f} cm²  ·  {100*area["area_fraction"]:.1f}%', 11)
            _text(fig, .825, y-.065, f'Envelope Ø {2000*area["radius_m"]:.1f} mm', 10, MUTED)
    _text(fig, .055, .073 if named_sites else .095,
          f'Window radius: {1000*report["work_window"]["radius_m"]:g} → '
          f'{1000*report["work_window"]["radius_m"]*(1-report["spatial_inset_fraction"]):g} mm  |  '
          f'Base → processing center: {1000*report["work_window"]["base_to_center_distance_m"]:.0f} mm', 12)
    _text(fig, .055, .044 if named_sites else .052, "Curved-surface boundaries may depart from a circle. All four candidates use the same fixed world window.", 11, MUTED)
    _text(fig, .055, .02 if named_sites else .022, "Object coordinates · viewing directions only · working poses and robot planning remain pending.", 10, MUTED)
    fig.savefig(output / "working_areas.png", dpi=150, facecolor="white")
    plt.close(fig)
    for i, area in enumerate(areas):
        fig = plt.figure(figsize=(12, 8), facecolor="white")
        color = COLORS[i % len(COLORS)]
        _text(fig, .055, .93, f'STEP 1.3  /  A{i+1:02d}', 13, color, weight="bold")
        _text(fig, .055, .875, f'{area["label_en"]} · working area on B', 25, weight="bold")
        ax = fig.add_axes([.0, .10, .72, .73], projection="3d")
        _draw_shape(ax, mesh, labels, focus=i, display=display)
        site = area.get("requested_site") or {}
        axis = np.array(site.get("display_view_object", area["window_normal_object"]), dtype=float)
        axis /= np.linalg.norm(axis)
        elevation = np.degrees(np.arcsin(np.clip(axis[1], -1, 1)))
        azimuth = np.degrees(np.arctan2(axis[0], axis[2]))
        ax.view_init(elev=elevation, azim=azimuth, vertical_axis="y")
        _text(fig, .72, .69, f'{area["area_m2"]*1e4:.1f} cm²', 26, color, weight="bold")
        _text(fig, .72, .625, f'{100*area["area_fraction"]:.1f}% of surface', 13)
        _text(fig, .72, .54, f'Envelope Ø {2000*area["radius_m"]:.1f} mm', 16)
        _text(fig, .72, .485, f'{inset:g}% radius + angle inset', 12)
        _text(fig, .72, .425, f'Station center: {1000*area["base_to_processing_center_distance_m"]:.0f} mm', 11)
        _text(fig, .72, .36, "Pose: to be optimized", 12)
        _text(fig, .055, .05, "Original object geometry · a viewing direction is not a selected working pose", 11, MUTED)
        fig.savefig(output / f"area_{i+1:02d}.png", dpi=150, facecolor="white")
        plt.close(fig)
