"""Render one task montage, labelled only with w_i and surface-area shares."""
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from whole_task_2.step1.common import ROOT, sha256, load_shape
from whole_task_2.step1.reach_video import Scene, tool_frame
from whole_task_2.step1.find_tasks_figure import (
    VIEW, GOLD, Opacity, cone, dashed, tube, tubes, tabletop, task_outline, WORKSPACE_COLOR)
from whole_task_2.step1.workspace import box_edges

HERE = Path(__file__).resolve().parent
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
ITALIC_FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf'


def task_label(draw, width, baseline, index, area_fraction):
    """Only w with a true subscript and the assigned surface-area percentage."""
    w_font = ImageFont.truetype(ITALIC_FONT, 43)
    sub_font = ImageFont.truetype(FONT, 25)
    percent_font = ImageFont.truetype(FONT, 35)
    subscript = str(index)
    percentage = f'{100*area_fraction:.2f}%'
    w_width = draw.textlength('w', font=w_font)
    sub_width = draw.textlength(subscript, font=sub_font)
    percent_width = draw.textlength(percentage, font=percent_font)
    x = (width-w_width-sub_width-26-percent_width)/2
    draw.text((x, baseline), 'w', font=w_font, fill='#27313b', anchor='ls')
    x += w_width
    draw.text((x, baseline+9), subscript, font=sub_font, fill='#27313b', anchor='ls')
    draw.text((x+sub_width+26, baseline), percentage, font=percent_font,
              fill='#27313b', anchor='ls')


def render(scene_path, output_path):
    mesh = load_shape(ROOT/'objects/B/mesh.stl')
    with np.load(scene_path) as z:
        poses, owner, axes = z['poses'], z['face_task'], z['tool_axis_object']
        setup = json.loads(str(z['setup_json']))
        unlimited = np.isinf(float(z['translation_mm']))
        if str(z['mesh_sha256']) != sha256(ROOT/'objects/B/mesh.stl'):
            raise ValueError('Scene data belongs to a different source mesh')
    if owner.shape != (len(mesh.faces),) or set(np.unique(owner)) != set(range(len(poses))):
        raise ValueError('Every source face and task must have an assignment')
    count = len(poses)
    size = (930, 970)
    shift = np.asarray(setup['work_point_m'])
    direction = np.asarray(setup['working_direction_world'])
    contacts = []
    boxes,fit = [],[]
    local_poses = poses.copy(); local_poses[:, :3, 3] -= shift
    for k, W in enumerate(poses):
        vertices = mesh.vertices@W[:3,:3].T+W[:3,3]
        bounds = (np.array([vertices.min(0),vertices.max(0)]) if unlimited else
                  np.asarray(setup['surface_workspace']['bounds_world_m']))
        center = bounds.mean(0)
        box = np.array(box_edges(bounds,shift)).reshape(-1,3)
        boxes.append(box); fit.append(box)
        ids = np.flatnonzero(owner == k)
        points = mesh.triangles_center[ids] @ W[:3, :3].T+W[:3, 3]
        score = np.linalg.norm(points-center, axis=1)
        f = int(ids[np.argmin(score)])
        axis = W[:3, :3] @ axes[k, f]
        tip = mesh.triangles_center[f] @ W[:3, :3].T+W[:3, 3]-shift
        contacts.append((f, tip, axis))
        fit.append(mesh.vertices @ local_poses[k, :3, :3].T+local_poses[k, :3, 3])
        root = tip-.1*axis
        head_vertices, _ = tube(root, tip-.17*axis, .02)
        fit.append(head_vertices)
        fit.append(np.array(cone(root, direction, .1, 25.)).reshape(-1, 3))
    fit = np.vstack(fit)
    panels = []
    for k, W in enumerate(local_poses):
        scene = Scene(mesh, setup, *size, arm=False)
        scene.view = VIEW.copy(); scene.fit(fit, 1.07)
        scene.ctx.line_width = 2.3
        r = scene.r
        mask = owner == k
        colors = np.tile(r.GRAY, (len(mesh.faces), 1)); colors[mask] = r.PINK
        scene.workspace_wireframe = (boxes[k], WORKSPACE_COLOR)
        _, tip, axis = contacts[k]
        root = tip-.1*axis
        wire = cone(root, direction, .1, 25.)
        geometry = tubes(dashed(wire), .0004)
        shape = r.Shape(scene.ctx, scene.shader, *geometry, GOLD, metallic=.1, roughness=.5)
        tools = [(part, tool_frame(tip, axis)) for part in scene.tools[False]]
        compositor = Opacity(scene)
        try:
            data = compositor.draw(W, colors, tabletop(scene, setup, shift)+tools,
                                   None, [], .3, [(shape, np.eye(4))], .60,
                                   segments=task_outline(mesh, mask, W))
            picture = Image.frombytes('RGB', size, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        finally:
            compositor.close(); scene.close()
        panel = Image.new('RGB', (size[0], size[1]+72), 'white')
        panel.paste(picture, (0, 0))
        draw = ImageDraw.Draw(panel)
        task_label(draw, size[0], size[1]+44, k+1,float(mesh.area_faces[mask].sum()/mesh.area))
        panels.append(panel)
    columns = min(3 if count <= 6 else 4, count); rows = math.ceil(count/columns)
    gap = 24
    width, height = panels[0].size
    composite = Image.new('RGB', (columns*width+(columns+1)*gap, rows*height+(rows+1)*gap), 'white')
    for k, panel in enumerate(panels):
        composite.paste(panel, (gap+(k % columns)*(width+gap), gap+(k//columns)*(height+gap)))
    composite.save(output_path)
    print('RENDERED',str(output_path),count,'poses',flush=True)
    return output_path
