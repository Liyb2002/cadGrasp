"""Two needs side by side, both vertical: vertical.py's upright bunny at the same place, the KUKA needle straight
down over the top of the vertical through the centre of mass, one camera; no text.

Left: a big work area (the upward faces within BIG of that top point) and pushes anywhere in the gold cone (CONE
about the needle, drawn at CONE_OPACITY like the other idea figures): the forces go every way, so the bottom is
big: an eggshell under the bunny (the convex hull of the bunny below DEPTH above its lowest point, grown by
THICK and rounded, up to that height, minus the bunny swept straight up) standing on LEGS legs that leave its underside
halfway out from the vertical through the centre of mass and land SPLAY further out on round pads.  The bunny
sits DEPTH deep in it, held every way but up.
Right: a small work area (SMALL) pushed at one point straight down: one thin pillar under it, vertical.py's.

Each support reports its contact on the bunny and how far from straight up the bunny could still move
(slanted.freedom).  Checked on dense samples: the arm links against bunny, supports and table; the needle and
head against bunny and supports, with its axis inside neither.

Run from the repository root:
  PYTHONPATH=.:whole_task_2/idea/easy_task/code .venv/bin/python -B whole_task_2/idea/easy_task/code/contrast_vertical.py
"""
import json
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import Delaunay
import manifold3d as md

from whole_task_2.step1.common import ROOT, load_shape
from whole_task_2.step1.reach_video import Scene, tool_frame, GOLD
from whole_task_2.step1.find_tasks_figure import task_outline, OUTLINE_WIDTH_PX, CONE_OPACITY, Opacity, cone, tubes
from whole_task_2.step1.find_tasks_figure import dashed as dashes
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step4.combine import exact_sweep
from vertical import HERE, LIFT, RADIUS, GAP, DOWN, TABLE, FACE, TURN, first_hit, spin, holder
from slanted import INFLATE, freedom, inside

BIG = .05           # m: the big work area's radius about the top point
SMALL = .008        # m: the small work area's radius
CONE = 35.          # deg: the pushes on the big area, about the needle
DEPTH = .03         # m: how far up the bunny the eggshell reaches, from its lowest point
THICK = .004        # m: the eggshell
SMOOTH = 6          # the eggshell's hull is rounded (manifold smooth_out) and each edge split this many times
LEGS = 3            # legs from the eggshell to the table
LEG = .004          # m: a leg's radius
SPLAY = .03         # m: how far out past where it leaves the shell a leg lands
PAD = (.01, .003)   # m: a foot pad's radius and height
SIZE = (1000, 1250) # px: one panel
GAP_PX = 24


def eggshell(m, axis, phase):
    """The convex hull of the bunny below DEPTH above its lowest point, grown by THICK, up to that height, with
    LEGS legs from its underside (halfway out from axis, LEGS directions from phase) down to pads SPLAY further
    out on the table, all minus the bunny (grown by INFLATE) swept straight up."""
    top = m.bounds[0, 2]+DEPTH; part = m.vertices[m.vertices[:, 2] < top]
    ball = trimesh.creation.icosphere(2, THICK).vertices
    shell = md.Manifold.hull_points((part[:, None]+ball).reshape(-1, 3)/G.SCALE).smooth_out(90.).refine(SMOOTH)
    shell = shell.trim_by_plane([0, 0, -1.], -top/G.SCALE)
    under = G.mesh(shell); parts = [shell]
    for a in phase+2*np.pi*np.arange(LEGS)/LEGS:
        u = np.r_[np.cos(a), np.sin(a)]; out = .5*((part[:, :2]-axis)@u).max()
        start = np.r_[axis+out*u, first_hit(under, [[*(axis+out*u), -1.]], -DOWN)[0]]
        foot = np.r_[axis+(out+SPLAY)*u, 0.]
        leg = trimesh.creation.capsule(np.linalg.norm(foot-start), LEG, count=[24, 24])
        leg.apply_transform(tool_frame((start+foot)/2, (foot-start)/np.linalg.norm(foot-start)))
        pad = trimesh.creation.cylinder(PAD[0], segment=[foot, foot+[0, 0, PAD[1]]], sections=48)
        parts += [G.solid(leg), G.solid(pad)]
    grown = trimesh.Trimesh(m.vertices+INFLATE*m.vertex_normals, m.faces, process=False)
    body = G.union(parts).trim_by_plane([0, 0, 1.], 0.)-exact_sweep(grown, -DOWN, DEPTH+.1)
    return G.mesh(max(body.decompose(), key=lambda p: p.volume()))


def main():
    source = load_shape(ROOT/'objects/B/mesh.stl')
    setup = json.loads((ROOT/'whole_task_2/data/B+03_kuka/step1_reach/plan.json').read_text())['setup']
    setup['working_direction_world'] = DOWN.tolist()
    station = np.r_[setup['work_point_m'][:2], setup['table_height_m']]
    setup['base_position_world_m'] = (np.asarray(setup['base_position_world_m'])-station).tolist()
    reach = -np.asarray(setup['base_position_world_m'])[:2]; camera = np.degrees(np.arctan2(reach[1], reach[0]))+90+TURN
    view = np.r_[np.cos(np.radians(camera)), np.sin(np.radians(camera)), .45]; view /= np.linalg.norm(view)

    W = np.eye(4); W[:3, :3] = spin(camera-FACE)@[[1, 0, 0], [0, 0, -1], [0, 1, 0]]   # original +Y up
    m = source.copy(); m.apply_transform(W)
    shift = -m.center_mass; shift[2] = LIFT-m.bounds[0, 2]; W[:3, 3] = shift; m.apply_translation(shift)
    com = m.center_mass
    top = np.r_[com[:2], first_hit(m, [[*com[:2], com[2]+1]], DOWN)[0]]
    tip = top-GAP*DOWN; up = m.face_normals[:, 2] > 0; distance = np.linalg.norm(m.triangles_center-top, axis=1)
    ring = np.linspace(0, 2*np.pi, 33)[:-1]; ring = com[:2]+RADIUS*np.c_[np.cos(ring), np.sin(ring)]
    seat = first_hit(m, np.c_[ring, np.full(len(ring), com[2]-1)], -DOWN)
    pillar = trimesh.creation.cylinder(RADIUS, segment=[[*com[:2], 0.], [*com[:2], seat.max()+1e-4]], sections=96)
    shell = eggshell(m, com[:2], np.radians(camera+60))         # two legs in front, one behind
    table = trimesh.creation.box(bounds=[[-TABLE, -TABLE, -.015], [TABLE, TABLE, 0.]])

    L, H, N = setup['needle_length_m'], setup['head_radius_m'], setup['needle_radius_m']
    solids = np.vstack([trimesh.sample.sample_surface_even(x, n, seed=0)[0] for x, n in ((m, 60000), (shell, 40000))])
    axis = tip-np.linspace(0, L+.07, 171)[:, None]*DOWN
    if (inside(solids, tip, tip-L*DOWN, N).any() or inside(solids, tip-L*DOWN, tip-(L+.07)*DOWN, H).any()
            or m.contains(axis).any() or shell.contains(axis).any()):
        raise ValueError('The needle hits the bunny or the eggshell')
    arm, links, q, _ = holder(setup, tip)
    samples = np.vstack([solids, *(trimesh.sample.sample_surface_even(x, n, seed=0)[0] for x, n in ((pillar, 4000), (table, 20000)))])
    hits = [i for i in range(1, 8)
            if Delaunay(arm.collisions[i]).find_simplex((samples-links[i][:3, 3])@links[i][:3, :3]).max() >= 0]
    if hits or shell.contains(trimesh.sample.sample_surface_even(m, 60000, seed=0)[0]).any():
        raise ValueError(f'KUKA links {hits} hit something, or the eggshell cuts the bunny')

    wire = cone(tip-L*DOWN, DOWN, L, CONE)
    sides = [(up & (distance < BIG), shell, True), (up & (distance < SMALL), pillar, False)]
    fit = np.vstack([m.vertices, table.vertices, shell.vertices, tip-(L+.07)*DOWN, *(p for s in wire for p in s)])
    panels = []
    for area, support, gold in sides:
        scene = Scene(source, setup, *SIZE, arm=True)
        try:
            r = scene.r
            scene.view = view; scene.ctx.line_width = OUTLINE_WIDTH_PX*scene.studio.size[0]/SIZE[0]; scene.fit(fit, 1.04)
            shapes = [(r.Shape(scene.ctx, scene.shader, table.vertices, table.faces, (.80, .81, .83), roughness=.85), np.eye(4)),
                      (r.Shape(scene.ctx, scene.shader, support.vertices, support.faces, BLUE, metallic=.05, roughness=.5), np.eye(4))]
            shapes += [(part, tool_frame(tip, DOWN)) for part in scene.tools[False]]
            cones = [(r.Shape(scene.ctx, scene.shader, *tubes(dashes(wire), .0005), GOLD, metallic=.1, roughness=.5), np.eye(4))]
            colors = np.tile(r.GRAY, (len(source.faces), 1)); colors[area] = r.PINK
            compositor = Opacity(scene)
            try: data = compositor.draw(W, colors, shapes, links, [], cones=cones if gold else None,
                                        cone_opacity=CONE_OPACITY, segments=task_outline(m, area, np.eye(4)))
            finally: compositor.close()
        finally: scene.close()
        panels.append(Image.frombytes('RGB', SIZE, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    out = Image.new('RGB', (2*SIZE[0]+3*GAP_PX, SIZE[1]+2*GAP_PX), 'white')
    for k, p in enumerate(panels): out.paste(p, (GAP_PX+k*(SIZE[0]+GAP_PX), GAP_PX))
    out.save(HERE/'contrast_vertical.png')
    print('RENDERED', HERE/'contrast_vertical.png', f'cone {CONE:.0f} deg')
    for label, (area, support, _) in zip(('big area', 'small area'), sides):
        held, angle = freedom(support, m, DOWN)
        print(f'  {label}: work area {100*m.area_faces[area].sum()/m.area:.1f}%, support {1e6*support.volume:.1f} cm3,',
              f'watertight {support.is_watertight}, contact {1e4*held:.2f} cm2, free up to {angle:.0f} deg from straight up')


if __name__ == '__main__': main()
