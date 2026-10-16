"""Text-free views of the actual saved cone-carved support in each pose."""
import math
import numpy as np
import trimesh
from PIL import Image


COLORS = ("#e89732", "#3ba48e", "#4c8dcc", "#a379bd")


def transformed(mesh, transform):
    result = mesh.copy()
    result.apply_transform(transform)
    return result


def render(mesh, support, states, out):
    from work_access_render import Panel, render_scene
    count = len(states)
    columns = min(2, count)
    size = (780, 700)
    sheet = Image.new("RGB", (columns*size[0], math.ceil(count/columns)*size[1]), "white")
    for i, (_, (task, transform, _)) in enumerate(states.items()):
        body, shell = transformed(mesh, transform), transformed(support, transform)
        work = trimesh.Trimesh(body.vertices.copy(), body.faces[task.domain.work_ids], process=False)
        # Raise the drawing surface only 1 um to avoid depth ties; no geometry
        # used by the mechanics/Boolean operations is moved.
        work.vertices += 1e-6*body.vertex_normals
        vertices = np.vstack((body.vertices, shell.vertices))
        # Look from the fixed needle side so every current work opening is
        # visible, while keeping the same world camera for all four poses.
        panel = Panel(None, size=size, points=vertices, elevation=38, azimuth=80)
        tile = render_scene(panel, [(body, "#b8b8b8"), (shell, "#2a91d2"), (work, COLORS[i%len(COLORS)])])
        sheet.paste(tile, ((i%columns)*size[0], (i//columns)*size[1]))
    sheet.save(out/"step3.1/overview.png")

    return dict(onscreen_text=False, actual_saved_support_used=True, working_surface_display_offset_m=1e-6,
                pose_order=list(states), camera_world=dict(elevation_deg=38, azimuth_deg=80),
                separate_work_exclusion_stage=False)
