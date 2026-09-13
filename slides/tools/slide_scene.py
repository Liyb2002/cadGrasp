"""Shared presentation scene: saved B/pose_2 geometry, Y-up, compact floor.

Only read baseline inputs. Rendering never invokes a baseline stage or rebuilds
the setup catalogue. Colours, camera and floor match head_total_force.png.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

SLIDES = Path(__file__).resolve().parents[1]
BASE = SLIDES / 'baseline_algo'
CASE = BASE / 'output/B/pose_2'
sys.path.insert(0, str(BASE))
from step1.needs import ContinuousNeeds, demand
from step2_local_support import render as R

PAPER = (255, 255, 255)
GREY = (184, 189, 191)
GREEN = (163, 186, 157)
FLOOR = (230, 232, 231)
ORANGE = (226, 126, 49)
BLUE = (45, 124, 177)
FRAME = (106, 138, 157)
RED = (184, 49, 53)
INK = '#151515'
MUTED = '#686868'
VIEW = np.array([.8, .12, -1.])
FLOOR_MARGIN = .13
CAMERA_MARGIN = 1.32
LOAD_FACE = 213


def load():
    domain = ContinuousNeeds.read(CASE / 'step_1_needs/needs.json')
    assert domain.data['object'] == 'B' and domain.data['pose_id'] == 'pose_2'
    assert np.allclose(domain.gravity, [0., -1., 0.])
    assert LOAD_FACE in domain.work_ids
    return domain


def samples(domain=None):
    domain = load() if domain is None else domain
    data = json.loads((CASE / 'step_1_needs/samples.json').read_text())
    digest = hashlib.sha256((CASE/'step_1_needs/needs.json').read_bytes()).hexdigest()
    assert data['provenance']['physical_domain_sha256'] == digest
    assert np.allclose(data['moment_origin_m'], domain.com, atol=1e-12, rtol=0)
    q = np.asarray(data['pt_m'])
    force = np.asarray(data['force_push_mg'])
    wrench = np.asarray(data['need_wrench'])
    assert np.allclose(wrench, demand(q, force, domain.com), atol=1e-12, rtol=0)
    return dict(q=q, push=force, wrench=wrench, seed=data['seed'])


def floor(domain, points=()):
    """A finite Y=0 rectangle, enlarged only to contain depicted floor data."""
    cloud = domain.mesh.vertices[:, [0, 2]]
    if len(points):
        cloud = np.vstack([cloud, np.asarray(points).reshape(-1, 3)[:, [0, 2]]])
    margin = FLOOR_MARGIN * float(domain.mesh.extents.max())
    low, high = cloud.min(0)-margin, cloud.max(0)+margin
    corners = np.array([[low[0], 0., low[1]], [high[0], 0., low[1]],
                        [high[0], 0., high[1]], [low[0], 0., high[1]]])
    return corners[[[0, 1, 2], [0, 2, 3]]]


@dataclass
class Camera:
    focus: np.ndarray
    basis: np.ndarray
    width: float
    size: int

    def project(self, points):
        return R.project(np.asarray(points), self.focus, self.basis, self.width, self.size)


def camera(domain, size=1100, extra=(), ground_points=()):
    cloud = [domain.mesh.vertices, floor(domain, ground_points).reshape(-1, 3)]
    if len(extra):
        cloud.append(np.asarray(extra).reshape(-1, 3))
    basis = R.axes(VIEW)
    bounds = np.array([(np.vstack(cloud)@basis.T).min(0), (np.vstack(cloud)@basis.T).max(0)])
    return Camera(bounds.mean(0)@basis, basis,
                  CAMERA_MARGIN*float(np.max(bounds[1, :2]-bounds[0, :2])), size)


def render(domain, *, size=1100, parts=(), patches=(), cam=None, ground_points=(), ground=True):
    """Depth-tested geometry; patches are actual surface triangles, not decals."""
    if cam is None:
        extra = np.concatenate([m.vertices for m, _ in parts]) if parts else ()
        cam = camera(domain, size, extra, ground_points)
    triangles = [floor(domain, ground_points)] if ground else []
    colors = [np.tile(FLOOR, (2, 1))] if ground else []
    triangles.append(domain.mesh.triangles)
    colors.append(np.tile(GREY, (len(domain.mesh.faces), 1)))
    colors[-1][domain.work_ids] = GREEN
    for mesh, color in parts:
        triangles.append(mesh.triangles)
        colors.append(np.tile(color, (len(mesh.faces), 1)))
    patch_start = sum(len(t) for t in triangles)
    for tri, color in patches:
        triangles.append(np.asarray(tri))
        colors.append(np.tile(color, (len(tri), 1)))
    count = sum(len(t) for t in triangles)
    overlay = np.arange(patch_start, count) if patches else None
    R.PAPER = PAPER
    image, ids = R.raster(np.concatenate(triangles), np.concatenate(colors),
                          cam.focus, cam.basis, cam.width, cam.size,
                          overlay=overlay, unlit=[0, 1] if ground else [])
    return image, cam, ids


def arrow(draw, start, end, color=RED, width=9, head=24):
    a, b = np.asarray(start, float), np.asarray(end, float)
    vector = b-a
    length = np.linalg.norm(vector)
    if length < 1e-8:
        raise ValueError('Arrow is parallel to the camera')
    u = vector/length
    side = np.array([-u[1], u[0]])
    draw.line([tuple(a), tuple(b-.55*head*u)], fill=color, width=width)
    draw.polygon([tuple(b), tuple(b-head*u+.5*head*side),
                  tuple(b-head*u-.5*head*side)], fill=color)


def text(draw, xy, message, size=28, color=INK, anchor='mm'):
    draw.text(xy, message, font=R.font(size), fill=color, anchor=anchor)


def applied_force(image, cam, domain, label=True):
    point = domain.mesh.triangles_center[LOAD_FACE]
    end = cam.project(point)[:2]
    start = cam.project(point+[0., .045, 0.])[:2]
    draw = ImageDraw.Draw(image)
    arrow(draw, start, end)
    if label:
        text(draw, start+[-22, 24], 'Applied force', 25, RED, 'rm')
    return point


def record(path, **fields):
    source = CASE / 'step_1_needs/needs.json'
    result = dict(object='B', pose='pose_2', coordinate_system='Y-up; floor y=0',
                  camera_vector=VIEW.tolist(), floor_margin_fraction=FLOOR_MARGIN,
                  camera_margin=CAMERA_MARGIN,
                  source=str(source.relative_to(SLIDES)),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), **fields)
    Path(path).write_text(json.dumps(result, indent=2)+'\n')
