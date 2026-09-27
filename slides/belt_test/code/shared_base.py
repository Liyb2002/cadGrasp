"""Three fixed docking sockets on one connected ground frame."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, Point

import fixture_geometry as F
from geometry_utils import tube, resample
import draw as figure

OUT = Path(__file__).resolve().parent.parent
# Stagger along Y so dock 2's diagonal approach does not cross dock 1's post.
# Object envelopes may overlap: only one object occupies the base at a time.
# Selected using the complete three-dock sequence, including unused sockets.
STATIONS = (np.array([.02, .25, 0.]), np.array([.02, .145, 0.]), np.array([.02, .355, 0.]))


def build():
    original, meta = F.build()
    cases = []
    for c, offset in zip(original, STATIONS):
        c = c.copy()
        for name in ('object', 'contact', 'channel', 'slider_detail'):
            c[name] = c[name].copy().apply_translation(offset)
        c['transform'] = c['transform'].copy()
        c['transform'][:3, 3] += offset
        c['port'] = c['port'] + offset
        c['points'] = c['points'] + offset
        cases.append(c)
    cloud = np.vstack([c['object'].vertices for c in cases] + [c['port'][None, :] for c in cases])
    footprint = MultiPoint(cloud[:, :2]).convex_hull.buffer(.035)
    xy = resample(footprint, 96)
    parts = tube(np.c_[xy, np.full(len(xy), .005)], .005, True)
    for c in cases:
        x, y, d = c['basis'].T
        mount = c['port'] - d*.0315 - y*.004
        back = mount - d*.020 + x*.025
        # Keep the local post outside its own workpiece footprint. Expanding
        # the common hull must not reroute a post through the blue module.
        local = MultiPoint(np.vstack([c['object'].vertices[:, :2], c['port'][None, :2]])).convex_hull.buffer(.023)
        edge = local.exterior.interpolate(local.exterior.project(Point(back[:2])))
        foot = np.array([edge.x, edge.y, .005])
        elbow = np.array([edge.x, edge.y, max(.018, back[2]-.010)])
        outer = footprint.exterior.interpolate(footprint.exterior.project(edge))
        ground_anchor = np.array([outer.x, outer.y, .005])
        parts += [c['channel']] + tube(np.array([ground_anchor, foot, elbow, back, mount]), .0045)
    base = F.joined(parts)
    if len(F.solid(base).decompose()) != 1:
        raise RuntimeError('Shared base must be one connected solid')
    for c in cases:
        c['base'] = base
        c['base_layout'] = 'shared_ground_frame_three_sockets'
        if c['pose_kind'] == 'illustrative_tilt':
            # The patch was selected against a local preliminary base. Recheck
            # its view rays against ALL sockets on the final common base.
            view = F.VIEW/np.linalg.norm(F.VIEW)
            origins = c['object'].triangles_center[c['work_ids']] + view*1e-6
            directions = np.tile(view, (len(origins), 1))
            blocked = base.ray.intersects_any(origins, directions) | c['contact'].ray.intersects_any(origins, directions)
            if blocked.any():
                raise RuntimeError('A shared-base socket obscures the illustrative working patch')
            c['work_area'] = dict(c['work_area'], selection_fixture='local preliminary base',
                                 shared_fixture_face_center_view_rays_clear=True)
    return cases, base, meta


def draw(cases, base):
    S = figure.S
    empty = cases[0]['object'].copy()
    empty.faces = np.empty((0, 3), int)
    empty.vertices = np.empty((0, 3))
    canvas = Image.new('RGB', (2000, 2250), 'white')
    ink = ImageDraw.Draw(canvas)
    S.text(ink, (55, 60), 'One fixed base. Three sockets. One reusable contact module.', 43, S.INK, 'lm')
    S.text(ink, (55, 120), 'Unused sockets stay on the common base; the object and blue module move together.', 30, S.MUTED, 'lm')
    cloud = np.vstack([base.vertices] + [c['object'].vertices for c in cases] + [c['contact'].vertices for c in cases])
    titles = ('Shared base with three sockets', 'Dock 1: pose 2', 'Dock 2: pose 2 + 25° tilt', 'Dock 3: pose 4')
    for i in range(4):
        c = cases[max(0, i-1)]
        domain = SimpleNamespace(mesh=empty if i == 0 else c['object'], work_ids=np.array([], int) if i == 0 else c['work_ids'])
        parts = [(base, figure.ORANGE)] + ([] if i == 0 else [(c['contact'], figure.BLUE)])
        # Use identical framing in every panel to make the stationary base clear.
        camera_domain = SimpleNamespace(mesh=cases[0]['object'])
        cam = S.camera(camera_domain, size=1000, extra=cloud, ground_points=cloud, view=F.VIEW)
        ground = trimesh.Trimesh(vertices=S.floor(camera_domain, cloud).reshape(-1, 3), faces=np.arange(6).reshape(2, 3), process=False)
        pic, _, _ = S.render(domain, parts=[(ground, S.FLOOR)]+parts, cam=cam, ground=False)
        pen = ImageDraw.Draw(pic)
        S.text(pen, (35, 45), titles[i], 34, S.INK, 'lm')
        for j, socket in enumerate(cases):
            location = cam.project(socket['port'] + np.array([0., 0., .040]))[:2]
            S.text(pen, location, str(j+1), 34, S.INK)
        canvas.paste(pic, ((i%2)*1000, 160+(i//2)*1000))
    S.text(ink, (55, 2200), 'Geometry concept; grasping, task loads and retention require validation.', 27, S.MUTED, 'lm')
    path = OUT/'shared_base.png'
    canvas.save(path)
    return path


if __name__ == '__main__':
    cases, base, meta = build()
    print(draw(cases, base))
