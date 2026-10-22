"""One Step3 initialization image: all six tasks on one common base.

Top: six placed bunnies with their own cleaned task regions in pink.
Bottom: the same complete base without the bunnies or tools. No text.
"""
import itertools
import numpy as np
from PIL import Image
from whole_task_2.operations.geometry import moved_mesh
from whole_task_2.step1.reach_render import contact
from whole_task_2.step1.reach_video import GOLD, Scene, STATION_VIEW, tool_frame
from .render_independent import BLUE

def camera(R, a):
    """The diagonal of the six sides (object rotation R) on the needle's side, nearest the station view."""
    diagonals = [R@np.array(g)/np.sqrt(3) for g in itertools.product((1, -1), repeat=3)]
    return max((v for v in diagonals if v@a < 0), key=lambda v: v@STATION_VIEW)


def image(scene, colors, tools, links, shapes, segments=()):
    return Image.frombytes('RGB', scene.size, scene.draw(np.eye(4), colors, tools, links, segments, shapes)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def cone(scene, tip):
    """Step1's gold max-tilt cone at tip, as line pairs: the rim and eight sides about the nominal direction."""
    T = tool_frame(tip, scene.nominal); rim = np.stack([scene.rim[:-1], scene.rim[1:]], 1).reshape(-1, 3)
    return np.vstack([rim, scene.sides])@T[:3, :3].T+T[:3, 3]


def render_connected(model, q, joined, shifts, folder, width=4200, height=1050, view=(0., 1., .35)):
    """initialization.png, no text: all six membranes on one base plate, joined.
    Top: every bunny in its seat with the needle and head on its central certified face and Step1's gold
    max-tilt cone at the tip; bottom: the support alone.  The camera looks level along the row from the
    needle's side (+y), a little from above, so the plate lies flat across the picture.  Vertices are split at sharp edges for drawing only, so the plate shades flat."""
    shown = joined.smooth_shaded
    scene = Scene(shown, model.setup, width, height)                     # the joined support is the body
    scene.ctx.line_width = 3*height/560                                  # Step1's 3 px lines on its 560 px pictures
    try:
        g = scene.r; seated = []; tools = []; cones = []
        for i, shift in enumerate(shifts):
            r = model.records[i]; P = q.poses[i]; M = np.eye(4); M[:3, 3] = shift
            bunny = g.Shape(scene.ctx, scene.shader, model.source.vertices, model.source.faces, g.GRAY, dynamic=True, metallic=.03, roughness=.63)
            colors = np.tile(g.GRAY, (len(model.source.faces), 1)); colors[r['working_face_ids']] = g.PINK; bunny.paint(colors)
            faces = np.asarray(r['certified_face_ids']); f = contact(model.source, faces, P); k = int(np.flatnonzero(faces == f)[0])
            tip = model.source.triangles_center[f]@P[:3, :3].T+P[:3, 3]+shift; axis = P[:3, :3]@r['axes'][k]
            seated.append((bunny, M@P)); tools += [(shape, tool_frame(tip, axis)) for shape in scene.tools[False]]
            cones.append(cone(scene, tip))
        blue = np.tile(BLUE, (len(shown.faces), 1)); scene.view = np.asarray(view)/np.linalg.norm(view)
        L = model.setup['needle_length_m']
        tops = [world[:3, 3]+world[:3, 2]*(L+.07) for _, world in tools[::2]]
        scene.fit(np.vstack([joined.vertices, *[moved_mesh(model.source, W).vertices for _, W in seated], tops, *cones]), 1.04)
        rows = [image(scene, blue, [], None, seated+tools, [(np.vstack(cones), GOLD)]), image(scene, blue, [], None, [])]
    finally: scene.close()
    canvas = Image.new('RGB', (width, 2*height), 'white')
    for k, row in enumerate(rows): canvas.paste(row, (0, k*height))
    canvas.save(folder/'initialization.png')
