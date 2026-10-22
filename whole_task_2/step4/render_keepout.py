"""idea/keepout.png, no text: every task's keep-out, needle and head (orange) and KUKA (violet).

Per nonempty exclusive task, one station view with the whole keep-out, at most three per row.
The bunny is gray with
the task pink; the KUKA holds the needle and head on the task's central certified face at the
joints Step1 certified there; the gold cone is the max tilt.  The keep-outs are drawn at 45%
opacity over the same picture without them, so what lies inside stays visible.
The orange head extension is shortened to 70% for display only.
"""
import math
from pathlib import Path
import numpy as np
from PIL import Image

from whole_task_2.operations.geometry import moved_mesh
from whole_task_2.step1.common import ROOT, sha256, write_json
from whole_task_2.step1.reach_render import contact
from whole_task_2.step1.reach_video import STATION_VIEW, Scene, work_arm
from whole_task_2.step3.render_tool_membranes import image
from whole_task_2.step3.tool_membranes import load_model, task_partition
from whole_task_2.step3.tool_keepout import keepout
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step5.video.prepare import matrices
from .keepout import folders, load

TOOL = (.93, .52, .18); ARM = (.47, .40, .78); OPACITY = .45
HEAD_EXTENSION_SCALE = .70


def picture(model, i, meshes, size):
    r = model.records[i]; P = np.asarray(r['pose']); _, arm, _ = meshes
    tool = G.mesh(keepout(model.source, r['certified_face_ids'], r['axes'], model.setup,
                          model.reach*HEAD_EXTENSION_SCALE, model.policy['tool_polygon_sides']))
    faces = np.asarray(r['certified_face_ids']); f = contact(model.source, faces, P); k = int(np.flatnonzero(faces == f)[0])
    tip = model.source.triangles_center[f]@P[:3, :3].T+P[:3, 3]; axis = P[:3, :3]@r['axes'][k]
    links = matrices(work_arm(model.setup).forward(r['joints'][k][3], geometry=True)[2])   # test point 3: the centroid
    placed = moved_mesh(model.source, P); bunny = placed.vertices
    scene = Scene(placed, model.setup, size, size, arm=True)              # image() draws the scene body in place
    try:
        g = scene.r; colors = np.tile(g.GRAY, (len(model.source.faces), 1)); colors[r['working_face_ids']] = g.PINK
        shapes = [(g.Shape(scene.ctx, scene.shader, m.vertices, m.faces, color, metallic=.05, roughness=.7), P)
                  for m, color in ((tool, TOOL), (arm, ARM))]
        points = np.vstack([bunny, tool.vertices@P[:3, :3].T+P[:3, 3], arm.vertices@P[:3, :3].T+P[:3, 3]])
        scene.view = STATION_VIEW; scene.fit(points, 1.04); tools = [(tip, axis, False)]
        plain = image(scene, colors, tools, links, ()); covered = image(scene, colors, tools, links, shapes)
        return Image.blend(plain, covered, OPACITY)
    finally: scene.close()


def render(object_name='B', output=None, size=620, gap=40):
    # The original station keeps its idea figure; a run of a robots_config station (B+<config>) its own.
    output = output or (ROOT/'whole_task_2/idea/keepout.png' if '+' not in object_name else ROOT/'whole_task_2/output'/object_name/'step4/keepout.png')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    source, _ = folders(object_name); model, _, _ = load_model(source); saved = load(object_name)
    n = len(model.ids); columns = min(3, n); tilew = size+gap
    canvas = Image.new('RGB', (columns*tilew+gap, math.ceil(n/columns)*(size+gap)+gap), 'white')
    for i, name in enumerate(model.ids):
        whole = picture(model, i, saved[name], size)
        x = gap+(i % columns)*tilew; y = gap+(i//columns)*(size+gap)
        canvas.paste(whole, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output)
    _, folder = folders(object_name)
    write_json(folder/'figure.json', dict(file=str(output), sha256=sha256(output),
        source_sha256=sha256(__file__), keepout_report_sha256=sha256(folder/'report.json'),
        step3_report_sha256=sha256(source/'report.json'), task_ids=model.ids,
        task_partition=task_partition(model.records, model.source),
        image_size_px=list(canvas.size), opacity=OPACITY, head_extension_scale=HEAD_EXTENSION_SCALE))
    print('KEEPOUT FIGURE', output, flush=True)
    return output


if __name__ == '__main__':
    render()
