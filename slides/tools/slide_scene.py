"""Shared presentation scenes: saved case geometry, Y-up, compact floor.

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
PRESENTATION_CASES = (('B', 'pose_2'), ('B', 'pose_3'),
                      ('A1-f', 'pose_2'), ('A1-f', 'pose_3'), ('C5', 'pose_2'))


def case_path(domain):
    return BASE/'output'/domain.data['object']/domain.data['pose_id']


def case_label(domain):
    return f"{domain.data['object']} / {domain.data['pose_id'].replace('_', ' ')}"


def load(name='B', pose='pose_2'):
    domain = ContinuousNeeds.read(BASE/'output'/name/pose/'step_1_needs/needs.json')
    assert domain.data['object'] == name and domain.data['pose_id'] == pose
    assert np.allclose(domain.gravity, [0., -1., 0.])
    return domain


def samples(domain=None):
    domain = load() if domain is None else domain
    case = case_path(domain)
    data = json.loads((case / 'step_1_needs/samples.json').read_text())
    digest = hashlib.sha256((case/'step_1_needs/needs.json').read_bytes()).hexdigest()
    assert data['provenance']['physical_domain_sha256'] == digest
    assert np.allclose(data['moment_origin_m'], domain.com, atol=1e-12, rtol=0)
    q = np.asarray(data['pt_m'])
    force = np.asarray(data['force_push_mg'])
    wrench = np.asarray(data['need_wrench'])
    assert np.allclose(wrench, demand(q, force, domain.com), atol=1e-12, rtol=0)
    return dict(q=q, push=force, wrench=wrench, seed=data['seed'],
                faces=domain.work_ids[np.asarray(data['work_face_index'], int)])


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


def camera(domain, size=1100, extra=(), ground_points=(), view=None):
    cloud = [domain.mesh.vertices, floor(domain, ground_points).reshape(-1, 3)]
    if len(extra):
        cloud.append(np.asarray(extra).reshape(-1, 3))
    basis = R.axes(VIEW if view is None else view)
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


def applied_force(image, cam, domain, label=True, point=None, direction=None, length=None):
    point = domain.mesh.triangles_center[LOAD_FACE] if point is None else np.asarray(point)
    direction = np.array([0., -1., 0.]) if direction is None else np.asarray(direction)
    length = .045 if length is None else length
    end = cam.project(point)[:2]
    start = cam.project(point-length*direction)[:2]
    draw = ImageDraw.Draw(image)
    arrow(draw, start, end)
    if label:
        text(draw, start+[-22, 24], 'Applied force', 25, RED, 'rm')
    return point


def record(path, domain=None, **fields):
    domain = load() if domain is None else domain
    source = case_path(domain) / 'step_1_needs/needs.json'
    result = dict(object=domain.data['object'], pose=domain.data['pose_id'], coordinate_system='Y-up; floor y=0',
                  camera_vector=VIEW.tolist(), floor_margin_fraction=FLOOR_MARGIN,
                  camera_margin=CAMERA_MARGIN,
                  source=str(source.relative_to(SLIDES)),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), **fields)
    Path(path).write_text(json.dumps(result, indent=2)+'\n')


def gallery(path, pages, title):
    """Five readable, equally sized panels in two rows, without blank cases."""
    cell_w, cell_h = 750, 725
    page = Image.new('RGB', (3*cell_w, 2*cell_h+100), PAPER)
    draw = ImageDraw.Draw(page)
    text(draw, (page.width/2, 50), title, 43)
    for i, panel in enumerate(pages):
        thumb = panel.copy()
        thumb.thumbnail((cell_w, cell_h))
        row, column = divmod(i, 3)
        offset = cell_w//2 if row == 1 and len(pages) == 5 else 0
        page.paste(thumb, (column*cell_w+offset+(cell_w-thumb.width)//2,
                           100+row*cell_h+(cell_h-thumb.height)//2))
    page.save(path)


def working_forces(domain, cam, count=12):
    """Select visible, spatially spread real process loads for an illustration."""
    sample = samples(domain)
    _, _, face_ids = render(domain, cam=cam)
    uv = cam.project(sample['q'])[:, :2]
    pixels = np.rint(uv).astype(int)
    inside = ((pixels >= 0) & (pixels < cam.size)).all(axis=1)
    candidates = np.flatnonzero(inside)
    x, y = pixels[candidates].T
    candidates = candidates[face_ids[y, x] == sample['faces'][candidates]+2]
    magnitude = np.linalg.norm(sample['push'], axis=1)
    directions = sample['push']/np.maximum(magnitude[:, None], 1e-20)
    projected_length = np.linalg.norm(directions@cam.basis[:2].T, axis=1)
    candidates = candidates[(magnitude[candidates] > .15) & (projected_length[candidates] > .3)]
    if not len(candidates):
        raise ValueError(f'No visible work loads for {case_label(domain)}; choose a surface-facing camera')
    features = np.c_[uv[candidates]/cam.size, .025*directions[candidates]]
    chosen = [int(np.argmin(uv[candidates, 1]))]
    nearest = np.linalg.norm(features-features[chosen[0]], axis=1)
    for _ in range(min(count, len(candidates))-1):
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(nearest, np.linalg.norm(features-features[index], axis=1))
    ids = candidates[chosen]
    return dict(sample_indices=ids.tolist(), points=sample['q'][ids],
                directions=directions[ids], force_push_mg=sample['push'][ids])


def force_parts(domain, cam, example):
    """Depth-tested 3-D arrows; every tip lands on its real working surface."""
    import trimesh
    extent = float(domain.mesh.extents.max())
    parts = []
    for q, direction in zip(example['points'], example['directions']):
        out = -direction
        length = .22*extent/max(.55, np.linalg.norm(direction@cam.basis[:2].T))
        tip = q+1e-5*extent*out
        tail = tip+length*out
        neck = tip+.024*extent*out
        stem = trimesh.creation.cylinder(radius=.0022*extent, segment=[tail, neck], sections=8)
        screen = direction@cam.basis[:2].T
        screen /= np.linalg.norm(screen)
        along = screen@cam.basis[:2]
        across = np.array([-screen[1], screen[0]])@cam.basis[:2]
        head = trimesh.Trimesh(vertices=[tip, tip-.027*extent*along+.011*extent*across,
                                        tip-.027*extent*along-.011*extent*across],
                               faces=[[0, 1, 2]], process=False)
        parts.extend([(stem, RED), (head, RED)])
    return parts
