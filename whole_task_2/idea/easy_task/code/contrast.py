"""Two needs side by side: one bunny pose (slanted.py's NAME), one needle station, one camera; no text.

Left: a big work area (the faces within BIG of P that face the needle) worked from many places, each pushing
along its own inward surface normal (the black needle on the KUKA at P, transparent ones at LOADS more places
spread farthest-first over the part of the area facing the camera): the forces go every way, so the support is big, slanted.py's slot for
this pose alone.  Right: a small work area (PATCH) pushed at one point along the needle line: the support is
one thin round-ended rod on that line, slanted.py's rod.

Each support reports its contact on the bunny and how far from the way back along the needle the bunny could
still move (slanted.freedom): the slot holds it every way but back, the rod only against the push.  Checked
on dense samples: the arm links against bunny, supports and table; every needle and head against bunny and
supports, with its axis inside neither the bunny nor the slot and its head above the table; the transparent
needles push level or down.

Run from the repository root:  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/easy_task/code/contrast.py
"""
import json
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import Delaunay

from whole_task_2.step1.common import ROOT, load_shape
from whole_task_2.step1.reach_video import Scene, tool_frame
from whole_task_2.step1.find_tasks_figure import task_outline, OUTLINE_WIDTH_PX, Opacity
from whole_task_2.step3.render_independent import BLUE
from slanted import (HERE, SLOPE, LIFT, RADIUS, PATCH, GAP, TURN, RISE, SIZE, GAP_PX, SLOT, SLOT_BACK,
                     poses, holder, block, freedom, inside)

NAME = '+Y'         # the pose
BIG = .045          # m: the big work area's radius about P
LOADS = 5           # transparent needles over the big area
SMOOTH = .006       # m: a place's push is the mean inward normal of the faces this close


def places(m, area, P, count, clear):
    """count places over the area, farthest-first from P, each with its smoothed inward normal, whose needle
    and head pass clear(tip, push)."""
    centres = m.triangles_center[area]; chosen = [P]; out = []
    order = np.argsort(-np.linalg.norm(centres-P, axis=1)); tried = np.zeros(len(centres), bool)
    while len(out) < count and not tried.all():
        gap = np.min([np.linalg.norm(centres-c, axis=1) for c in chosen], axis=0); gap[tried] = -1
        k = int(np.argmax(gap)); tried[k] = True
        near = np.linalg.norm(m.triangles_center-centres[k], axis=1) < SMOOTH
        push = -(m.face_normals[near]*m.area_faces[near, None]).sum(0); push /= np.linalg.norm(push)
        tip = centres[k]-GAP*push
        if clear(tip, push): chosen.append(centres[k]); out.append((tip, push))
    return out


def main():
    source = load_shape(ROOT/'objects/B/mesh.stl')
    setup = json.loads((ROOT/'whole_task_2/data/B+03_kuka/step1_reach/plan.json').read_text())['setup']
    station = np.r_[setup['work_point_m'][:2], setup['table_height_m']]
    setup['base_position_world_m'] = (np.asarray(setup['base_position_world_m'])-station).tolist()
    reach = -np.asarray(setup['base_position_world_m'])[:2]; reach /= np.linalg.norm(reach)
    slope = np.radians(SLOPE); d = np.r_[np.cos(slope)*reach, -np.sin(slope)]
    setup['working_direction_world'] = d.tolist()
    camera = np.degrees(np.arctan2(reach[1], reach[0]))+90+TURN
    view = np.r_[np.cos(np.radians(camera)), np.sin(np.radians(camera)), RISE]; view /= np.linalg.norm(view)

    _, R, entry, exit = poses(source, d)[0][NAME]
    chord = np.linalg.norm(exit-entry)
    S = np.r_[(chord+GAP)*d[:2], LIFT-((source.vertices-exit)@R[2]).min()]
    tip = S-(chord+GAP)*d; P = tip+GAP*d; foot = S+(S[2]/np.sin(slope)+RADIUS)*d
    W = np.eye(4); W[:3, :3] = R; W[:3, 3] = S-R@exit; m = source.copy(); m.apply_transform(W)
    facing = m.face_normals@d < 0; distance = np.linalg.norm(m.triangles_center-P, axis=1)
    rod = trimesh.creation.capsule(np.linalg.norm(foot-S), RADIUS, count=[48, 48]); rod.apply_transform(tool_frame((S+foot)/2, d))
    slot = block(S, d, [m], SLOT, SLOT_BACK, True)
    xy = np.vstack([m.vertices[:, :2], slot.vertices[:, :2], foot[:2]])
    table = trimesh.creation.box(bounds=[[*(xy.min(0)-.02), -.015], [*(xy.max(0)+.02), 0.]])

    L, H, N = setup['needle_length_m'], setup['head_radius_m'], setup['needle_radius_m']
    solids = np.vstack([trimesh.sample.sample_surface_even(x, n, seed=0)[0] for x, n in ((m, 60000), (slot, 40000), (rod, 6000))])
    def clear(t, a):
        """Pushing level or down, needle and head off every surface, their axis inside no solid, the head above
        the table."""
        axis = t-np.linspace(0, L+.07, 171)[:, None]*a
        return not (a[2] > 0 or inside(solids, t, t-L*a, N).any() or inside(solids, t-L*a, t-(L+.07)*a, H).any()
                    or axis[:, 2].min() < H or m.contains(axis).any() or slot.contains(axis).any())
    if not clear(tip, d): raise ValueError('The needle at P hits the bunny or a support')
    loads = places(m, facing & (distance < BIG) & (m.face_normals@view > .2), P, LOADS, clear)
    arm, links, q, _ = holder(setup, tip, d)
    samples = np.vstack([solids, trimesh.sample.sample_surface_even(table, 20000, seed=0)[0]])
    hits = [i for i in range(1, 8)
            if Delaunay(arm.collisions[i]).find_simplex((samples-links[i][:3, 3])@links[i][:3, :3]).max() >= 0]
    if hits or slot.contains(trimesh.sample.sample_surface_even(m, 60000, seed=0)[0]).any():
        raise ValueError(f'KUKA links {hits} hit something, or the slot cuts the bunny')

    sides = [(facing & (distance < BIG), slot, loads), (facing & (distance < PATCH), rod, [])]
    fit = np.vstack([m.vertices, table.vertices, slot.vertices, tip-(L+.07)*d, *(t-(L+.07)*a for t, a in loads)])
    panels = []
    for area, support, ghosts in sides:
        scene = Scene(source, setup, *SIZE, arm=True)
        try:
            r = scene.r
            scene.view = view; scene.ctx.line_width = OUTLINE_WIDTH_PX*scene.studio.size[0]/SIZE[0]; scene.fit(fit, 1.04)
            shapes = [(r.Shape(scene.ctx, scene.shader, table.vertices, table.faces, (.80, .81, .83), roughness=.85), np.eye(4)),
                      (r.Shape(scene.ctx, scene.shader, support.vertices, support.faces, BLUE, metallic=.05, roughness=.5), np.eye(4))]
            shapes += [(part, tool_frame(tip, d)) for part in scene.tools[False]]
            ghost = [(part, tool_frame(t, a)) for t, a in ghosts for part in scene.tools[True]]
            colors = np.tile(r.GRAY, (len(source.faces), 1)); colors[area] = r.PINK
            compositor = Opacity(scene)
            try: data = compositor.draw(W, colors, shapes, links, ghost, segments=task_outline(m, area, np.eye(4)))
            finally: compositor.close()
        finally: scene.close()
        panels.append(Image.frombytes('RGB', SIZE, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    out = Image.new('RGB', (2*SIZE[0]+3*GAP_PX, SIZE[1]+2*GAP_PX), 'white')
    for k, p in enumerate(panels): out.paste(p, (GAP_PX+k*(SIZE[0]+GAP_PX), GAP_PX))
    out.save(HERE/'contrast.png')
    turn = [np.degrees(np.arccos(np.clip(a@d, -1, 1))) for _, a in loads]
    print('RENDERED', HERE/'contrast.png', f'pose {NAME}; {len(loads)} more pushes, {min(turn):.0f}-{max(turn):.0f} deg off the needle line')
    for label, (area, support, _) in zip(('big area', 'small area'), sides):
        held, angle = freedom(support, m, d)
        print(f'  {label}: work area {100*m.area_faces[area].sum()/m.area:.1f}%, support {1e6*support.volume:.1f} cm3,',
              f'contact {1e4*held:.2f} cm2, free up to {angle:.0f} deg from the way back')


if __name__ == '__main__': main()
