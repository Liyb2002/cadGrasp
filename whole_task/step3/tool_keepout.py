"""The tool Step1 certified, as the keep-out of one task: needle and head along every face's own axis.

Step1 certifies face f of a task along one axis d (object frame; the needle
points along d onto the surface, so the tool lies at p - s d).  Wherever the tip
is on the face, the needle fills s in [0, L] within the needle radius r of its
line, and beyond s = L the head clears radius R (Step1's head test).  The tool
over the whole face therefore lies in two convex pieces:
  needle: the triangle widened by r across d, swept over s in [0, L];
  head:   the triangle widened by R across d, swept over s in [L, L + reach].
The circles are replaced by circumscribed polygons, so each piece contains the
tool.  reach is relative to the object: long enough that the head piece leaves
everything a fixture around the object can hold.
"""
import numpy as np
import manifold3d as md

from whole_task.common.force_cone import tangent_frame
from . import independent_geometry as G


def disc(d, radius, sides):
    """Polygon around a circle of `radius` across d (vertices on the circumscribed polygon)."""
    u, v = tangent_frame(d); phi = 2*np.pi*np.arange(sides)/sides
    return radius/np.cos(np.pi/sides)*(np.cos(phi)[:, None]*u+np.sin(phi)[:, None]*v)


def pieces(triangles, axes, length, needle_radius, head_radius, reach, sides=16):
    """Point sets whose convex hulls are the needle and head pieces of each face (object frame)."""
    out = []
    for T, d in zip(np.asarray(triangles, float), np.asarray(axes, float)):
        d = d/np.linalg.norm(d)
        for radius, s0, s1 in ((needle_radius, 0., length), (head_radius, length, length+reach)):
            ring = (T[:, None, :]+disc(d, radius, sides)[None]).reshape(-1, 3)
            out.append(np.vstack([ring-s0*d, ring-s1*d]))
    return out


def keepout(mesh, faces, axes, setup, reach, sides=16):
    """Union of every certified face's needle and head pieces, as one solid in the object frame."""
    parts = pieces(mesh.triangles[np.asarray(faces, int)], axes, setup['needle_length_m'],
                   setup['needle_radius_m'], setup['head_radius_m'], reach, sides)
    return G.union([md.Manifold.hull_points(p/G.SCALE) for p in parts])
