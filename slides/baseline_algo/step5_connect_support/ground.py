"""Solid construction and sweep primitives owned by Step 5."""
import numpy as np
from scipy.spatial import ConvexHull
import trimesh
import manifold3d as manifold
from step2_local_support import insertion as D

def frame(bearing_deg):
    angle = np.deg2rad(bearing_deg)
    a = np.array([np.cos(angle), 0., np.sin(angle)])
    return np.array([a, -np.cross([0, 1, 0], a), [0., 1., 0.]])

def box(low, high, basis):
    mesh = trimesh.creation.box(extents=np.asarray(high)-low)
    mesh.apply_translation((np.asarray(high)+low)/2)
    mesh.vertices = mesh.vertices @ basis
    if np.linalg.det(basis) < 0: mesh.faces = mesh.faces[:, [0, 2, 1]]
    return mesh

def hull_coverage(points, contacts, tolerance):
    hull = ConvexHull(contacts)
    values = points @ hull.equations[:, :2].T + hull.equations[:, 2]
    return np.all(values <= tolerance, axis=1), hull

def swept_pieces(pieces, direction, length):
    return [D.engine.hull_mesh(np.vstack([p.vertices, p.vertices-length*direction]))
            for p in pieces]

def solid64(mesh,origin,scale):
    """Preserve the float64 clipped geometry through manifold input and output."""
    result=manifold.Manifold(manifold.Mesh64(np.asarray((mesh.vertices-origin)/scale,np.float64),
                                            np.asarray(mesh.faces,np.uint64)))
    if result.status()!=manifold.Error.NoError:
        raise ValueError(f'Manifold rejected mesh: {result.status()}')
    return result

def intersection_volume(mesh, piece, scale):
    return intersection_volumes(mesh,[piece],scale)[0]

def intersection_volumes(mesh,pieces,scale):
    # Work near unit scale for mesh boolean precision, then return m^3.
    origin = mesh.bounds.mean(axis=0)
    obstacle=solid64(mesh,origin,scale)
    return [abs(float((obstacle^solid64(piece,origin,scale)).volume()))*scale**3 for piece in pieces]
