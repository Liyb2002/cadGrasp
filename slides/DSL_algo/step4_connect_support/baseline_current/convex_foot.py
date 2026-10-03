"""One complete convex landing and one loft for each head/floor pair."""
import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from step4_connect_support.baseline_current import build_coupled_saddle as S
from step2_local_support import geometry as G


def hull_xy(polygons):
    hull = MultiPoint(np.vstack(polygons)).convex_hull
    if hull.geom_type != 'Polygon' or hull.area < 1e-6:
        raise ValueError('Convex landing has insufficient area')
    return np.asarray(hull.exterior.coords)[:-1]


def landing(body, basis, offset):
    mesh = S.unpack(body)
    world = S.local_to_world(mesh.vertices, basis, offset)
    faces = mesh.faces[np.all(np.abs(world[mesh.faces, 2]) < 1e-9, axis=1)]
    polygons = [Polygon(world[f, :2]) for f in faces]
    return unary_union([p for p in polygons if p.area > 1e-14])


def loft(patch, xy, basis, offset, carve, connected_piece):
    """Rebuild, never append spokes. The complete 3 mm sole must survive.

    Requiring the whole sole to survive excludes footprints spanning an object
    corridor. A connected body alone would not exclude multiple ground fingers.
    """
    pad = np.c_[xy, np.zeros(len(xy))] @ basis + offset
    terminal = np.vstack([pad, pad + .003 * basis[2]])
    required = S.solid(G.hull_mesh(terminal))
    envelope = S.solid(G.hull_mesh(np.vstack([patch['v'], patch['root'], terminal])))
    proposed = carve(envelope) + patch['original']
    connected = connected_piece(proposed, [patch['original'], required])
    if connected is None:
        return None
    # Require one continuous footprint, including any root touching the plane.
    ground = landing(connected, basis, offset)
    if ground.geom_type != 'Polygon' or ground.area < 1e-6:
        return None
    missing = Polygon(xy).difference(ground.buffer(1e-10)).area
    if missing > 1e-12:
        return None
    return connected
