"""Text-free isometric views of final fixtures and all original ground demands."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import trimesh

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / 'slides/baseline_algo'
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))
from step4_connect_support.clean_render import Renderer

# These utilities only build raster input; they never run the reuse experiment.
CPU_PATH = ROOT / 'slides/reuse/code/render_cpu.py'
spec = importlib.util.spec_from_file_location('step51_cpu_pieces', CPU_PATH)
CPU = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CPU)

BLUE = '#2A91D2'
GRAY = '#A5ADB5'
COLORS = ['#df8236', '#895eab', '#299584', '#c75278', '#ab912c',
          '#5875bf', '#bc6456', '#649142', '#af64a0', '#398fba']


def camera_for(points):
    camera = np.array([1., -1., 1.]) / np.sqrt(3)
    right = np.array([1., 1., 0.]) / np.sqrt(2)
    basis = np.array([right, np.cross(camera, right), camera])
    p = np.asarray(points) @ basis.T
    low, high = p.min(axis=0), p.max(axis=0)
    focus = ((low + high) / 2) @ basis
    span = max(float(np.max(high[:2] - low[:2])) * 1.16, .001)
    return focus, basis, span


def piece(mesh, color, opacity=1.):
    result = CPU.piece(dict(v=mesh.vertices, f=mesh.faces), color)
    result['opacity'] = opacity
    return result


def points_piece(points, color, camera, size, half_size_pixels=.60):
    """Include every saved point as a camera-facing marker, without decimation."""
    _, basis, span = camera
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    offsets = corners @ basis[:2] * half_size_pixels * span / size
    squares = np.asarray(points)[:, None, :] + offsets
    triangles = squares[:, [[0, 1, 2], [0, 2, 3]], :].reshape(-1, 3, 3)
    return dict(triangles=triangles,
                normals=np.broadcast_to(basis[2], triangles.shape),
                colors=np.broadcast_to(CPU.rgb(color), triangles.shape),
                bias=2e-6, unlit=True)


def floor_piece(points, color='#eef1f4', opacity=.16):
    xy = np.asarray(points)[:, :2]
    low, high = xy.min(axis=0) - .006, xy.max(axis=0) + .006
    vertices = np.array([[*low, 0.], [high[0], low[1], 0.],
                         [*high, 0.], [low[0], high[1], 0.]])
    result = piece(trimesh.Trimesh(vertices, [[0, 1, 2], [0, 2, 3]], process=False), color, opacity)
    result['unlit'] = True
    return result


def render_pose(renderer, body, fixture, cloud, color, path, size=900):
    camera = camera_for(np.vstack([body.vertices, fixture.vertices, cloud]))
    parts = [floor_piece(cloud), piece(fixture, BLUE),
             piece(body, GRAY, .28), points_piece(cloud, color, camera, size)]
    image = renderer.render([CPU.placed(p) for p in parts], camera, size)
    image.save(path)
    return image, camera


def render_common(renderer, fixture, clouds, colors, path, size=1200):
    """Different world floors stay different planes in the one fixture frame."""
    camera = camera_for(np.vstack([fixture.vertices, *clouds]))
    parts = [CPU.placed(piece(fixture, BLUE))]
    for cloud, color in zip(clouds, colors):
        parts.append(CPU.placed(points_piece(cloud, color, camera, size)))
    image = renderer.render(parts, camera, size)
    image.save(path)
    return camera


def sheet(images, path):
    columns = min(3, len(images))
    width = images[0].width
    height = images[0].height
    rows = (len(images) + columns - 1) // columns
    image = Image.new('RGB', (width * columns, height * rows), 'white')
    for k, tile in enumerate(images):
        image.paste(tile, ((k % columns) * width, (k // columns) * height))
    image.save(path)
