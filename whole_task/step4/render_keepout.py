"""idea/keepout.png, no text: every task's keep-out, needle and head (orange) and KUKA (violet).

Per task, one small station view with the whole keep-out. The six tasks form a 3-by-2 grid.
The bunny is gray with
the task pink; the KUKA holds the needle and head on the task's central certified face at the
joints Step1 certified there; the gold cone is the max tilt.  The keep-outs are drawn at 45%
opacity over the same picture without them, so what lies inside stays visible.
The orange head extension is shortened to 70% for display only.
"""
import math
import numpy as np
from PIL import Image

from whole_task.operations.geometry import moved_mesh
from whole_task.step1.common import ROOT,sha256,write_json
from whole_task.step1.find_tasks_figure import workspace_box,WORKSPACE_COLOR
from whole_task.step1.reach_render import contact
from whole_task.step1.reach_video import STATION_VIEW, Scene, kuka
from whole_task.step3.render_tool_membranes import image
from whole_task.step3.tool_keepout import keepout
from whole_task.step3 import independent_geometry as G
from whole_task.step5.video.prepare import matrices
from .keepout import folders, load, load_task_model

TOOL = (.93, .52, .18); ARM = (.47, .40, .78); OPACITY = .45
HEAD_EXTENSION_SCALE = .70


def picture(model, i, meshes, size):
    r = model.records[i]; P = np.asarray(r['pose']); _, arm, _ = meshes
    tool = G.mesh(keepout(model.source, r['certified_face_ids'], r['axes'], model.setup,
                          model.reach*HEAD_EXTENSION_SCALE, model.policy['tool_polygon_sides']))
    faces = np.asarray(r['certified_face_ids']); f = contact(model.source, faces, P); k = int(np.flatnonzero(faces == f)[0])
    tip = model.source.triangles_center[f]@P[:3, :3].T+P[:3, 3]; axis = P[:3, :3]@r['axes'][k]
    links = matrices(kuka(model.setup).forward(r['joints'][k][3], geometry=True)[2])   # test point 3: the centroid
    placed = moved_mesh(model.source, P); bunny = placed.vertices
    scene = Scene(placed, model.setup, size, size, arm=True)              # image() draws the scene body in place
    try:
        g = scene.r; colors = np.tile(g.GRAY, (len(model.source.faces), 1)); colors[r['working_face_ids']] = g.PINK
        workspace_box(scene,model.setup)
        shapes = [(g.Shape(scene.ctx, scene.shader, m.vertices, m.faces, color, metallic=.05, roughness=.7), P)
                  for m, color in ((tool, TOOL), (arm, ARM))]
        points = np.vstack([bunny, tool.vertices@P[:3, :3].T+P[:3, 3], arm.vertices@P[:3, :3].T+P[:3, 3]])
        scene.view = STATION_VIEW; scene.fit(points, 1.04); tools = [(tip, axis, False)]
        plain = image(scene, colors, tools, links, ()); covered = image(scene, colors, tools, links, shapes)
        return Image.blend(plain, covered, OPACITY)
    finally: scene.close()


def render(object_name='B', output=ROOT/'whole_task/idea/keepout.png', size=620, gap=40):
    _, folder = folders(object_name); model = load_task_model(object_name); saved = load(object_name)
    n = len(model.ids); columns = min(3, n); tilew = size+gap
    canvas = Image.new('RGB', (columns*tilew+gap, math.ceil(n/columns)*(size+gap)+gap), 'white')
    for i, name in enumerate(model.ids):
        whole = picture(model, i, saved[name], size)
        x = gap+(i % columns)*tilew; y = gap+(i//columns)*(size+gap)
        canvas.paste(whole, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output)
    write_json(folder/'figure.json',dict(file=str(output),sha256=sha256(output),
        source_sha256=sha256(__file__),report_sha256=sha256(folder/'report.json'),
        plan_sha256=sha256(ROOT/'whole_task/data'/object_name/'step1_reach/plan.json'),
        task_ids=model.ids,task_face_counts=[len(r['working_face_ids']) for r in model.records],
        surface_workspace=model.setup['surface_workspace'],workspace_box_color=list(WORKSPACE_COLOR),
        workspace_box_edges=12,keepout_opacity=OPACITY,head_extension_display_scale=HEAD_EXTENSION_SCALE,
        width=canvas.width,height=canvas.height))
    print('KEEPOUT FIGURE', output, flush=True)
    return output


if __name__ == '__main__':
    render()
