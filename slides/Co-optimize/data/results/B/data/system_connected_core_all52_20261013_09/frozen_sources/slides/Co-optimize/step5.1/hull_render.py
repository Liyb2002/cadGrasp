"""Step5.1 views with the exact saved convex polygon around each demand cloud."""
from __future__ import annotations

import numpy as np

import step51_render as V


def outline_piece(vertices, color, camera, size, half_width_pixels=1.35):
    """Draw straight, closed hull edges with the existing depth-aware renderer.

    The centerlines are the actual hull edges in their own floor plane. Thin
    camera-facing ribbons give a constant screen width, just like point markers.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError('A finite ordered convex polygon in 3D is required')
    _, basis, span = camera
    ends = np.roll(vertices, -1, axis=0)
    projected_edges = (ends-vertices) @ basis[:2].T
    length = np.linalg.norm(projected_edges, axis=1)
    perpendicular = np.zeros_like(projected_edges)
    visible = length > np.finfo(float).eps*span
    perpendicular[visible] = np.c_[-projected_edges[visible, 1], projected_edges[visible, 0]]/length[visible, None]
    offsets = perpendicular @ basis[:2] * half_width_pixels*span/size
    corners = np.stack([vertices+offsets, ends+offsets, ends-offsets, vertices-offsets], axis=1)
    edges = corners[:, [[0, 1, 2], [0, 2, 3]], :].reshape(-1, 3, 3)
    previous = np.roll(offsets, 1, axis=0)
    joins = np.concatenate([np.stack([vertices, vertices+previous, vertices+offsets], axis=1),
                            np.stack([vertices, vertices-previous, vertices-offsets], axis=1)])
    triangles = np.concatenate([edges, joins])
    return dict(triangles=triangles, normals=np.broadcast_to(basis[2], triangles.shape),
                colors=np.broadcast_to(V.CPU.rgb(color)*.62, triangles.shape),
                bias=4e-6, unlit=True)


def render_pose(renderer, body, fixture, cloud, color, path, *, hull, size=900):
    camera = V.camera_for(np.vstack([body.vertices, fixture.vertices, cloud]))
    parts = [V.floor_piece(cloud), V.piece(fixture, V.BLUE),
             V.piece(body, V.GRAY, .28), V.points_piece(cloud, color, camera, size),
             outline_piece(hull, color, camera, size)]
    image = renderer.render([V.CPU.placed(p) for p in parts], camera, size)
    image.save(path)
    return image, camera


def render_common(renderer, fixture, clouds, colors, path, *, hulls, size=1200):
    """Each hull follows its cloud's actual plane in the common fixture frame."""
    camera = V.camera_for(np.vstack([fixture.vertices, *clouds]))
    parts = [V.CPU.placed(V.piece(fixture, V.BLUE))]
    for cloud, color, hull in zip(clouds, colors, hulls, strict=True):
        parts.append(V.CPU.placed(V.points_piece(cloud, color, camera, size)))
        parts.append(V.CPU.placed(outline_piece(hull, color, camera, size)))
    image = renderer.render(parts, camera, size)
    image.save(path)
    return camera
