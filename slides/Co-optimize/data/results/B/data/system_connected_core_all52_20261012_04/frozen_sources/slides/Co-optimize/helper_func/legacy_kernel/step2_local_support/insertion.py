"""Reusable horizontal insertion sets and actual contact-head sweep geometry."""
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull

from step3_scheculer import contacts as I
from step2_local_support import geometry as G, surface

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'obj_supp'))
from insert_trajectory import angles as engine

MOTION = dict(
    frame='Step 1 world coordinates; +X=0 deg, +Y=90 deg',
    trajectory='Head(s)=Head_final-s*a; s decreases from L to 0',
    direction='a=(cos(theta), sin(theta), 0); degrees modulo 360',
    rotation_allowed=False, floor_checked=False, connectors_checked=False,
    obstacle='Complete object mesh; all selected heads translate together',
    boundary_convention='Closed certified intervals at the stated geometry tolerance; unresolved bands are excluded')


def contains(directions, angle):
    angle = float(angle) % 360
    return (any(a <= angle <= b or (angle == 0 and b == 360)
                for a, b in directions['intervals_deg'])
            or angle in directions['isolated_angles_deg'])


def normalize(intervals=(), isolated=()):
    """Closed circular sets. Never bridge a positive gap, however narrow."""
    merged = []
    points = list(isolated)
    for low, high in sorted(intervals):
        low, high = float(low), float(high)
        if not (0 <= low <= high <= 360):
            raise ValueError('Split wrapped intervals at 0/360 before normalization')
        if low == high:
            points.append(low)
        elif merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    result = dict(intervals_deg=merged, isolated_angles_deg=[])
    result['isolated_angles_deg'] = [a for a in sorted(set(float(x) % 360 for x in points))
                                     if not contains(result, a)]
    return result


def full():
    return normalize([[0., 360.]])


def intersect(first, second):
    def pieces(value):
        result = list(value['intervals_deg']) + [[a, a] for a in value['isolated_angles_deg']]
        # The endpoints 360 and 0 are the same physical direction.
        if contains(value, 0):
            result.append([0., 0.])
        return result
    return normalize([[max(a, c), min(b, d)] for a, b in pieces(first)
                      for c, d in pieces(second) if max(a, c) <= min(b, d)])


def nonempty(value):
    return bool(value['intervals_deg'] or value['isolated_angles_deg'])


def representative(value):
    if value['intervals_deg']:
        low, high = max(value['intervals_deg'], key=lambda pair: pair[1]-pair[0])
        angle = (low+high)/2
    elif value['isolated_angles_deg']:
        angle = value['isolated_angles_deg'][0]
    else:
        return None
    return dict(angle_deg=angle, vector=engine.direction(angle).tolist())


def summary(value):
    return dict(has_certified_direction=nonempty(value),
                interval_width_deg=sum(b-a for a, b in value['intervals_deg']),
                isolated_direction_count=len(value['isolated_angles_deg']),
                representative=representative(value))


def from_analysis(analysis):
    return normalize(analysis['clear_intervals_deg'],
                     [r['angle_deg'] for r in analysis['isolated_direction_checks'] if r['clear']])


def signature(contact, depth):
    digest = hashlib.sha256()
    for key in sorted(contact):
        value = np.ascontiguousarray(contact[key])
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    digest.update(float(depth).hex().encode())
    return digest.hexdigest()


def code_hashes():
    return I.hashes([Path(__file__), Path(engine.__file__), Path(G.__file__),
                     Path(surface.__file__), Path(I.__file__)])


class Analyzer:
    def __init__(self, mesh, depth):
        self.mesh, self.depth = mesh, depth
        self.offsets, self.valid = G.vertex_offsets(mesh, depth)

    def heads(self, contact):
        """Same continuous offset skin as insert_trajectory/study.build_heads.

        Preserve tiny clipped cells with that experiment's normalized hull builder.
        Shared vertex offsets are computed once for the entire catalogue.
        """
        heads = []
        mesh = self.mesh
        for face in np.unique(contact['source_faces']):
            face = int(face)
            which = contact['source_faces'] == face
            if not self.valid[mesh.faces[face]].all():
                raise ValueError('Invalid outward skin')
            vertices = np.unique(contact['triangles_m'][which].reshape(-1, 3), axis=0)
            source = mesh.triangles[face]
            x = source[1]-source[0]
            x /= np.linalg.norm(x)
            basis = np.array([x, np.cross(mesh.face_normals[face], x)])
            polygon = vertices[ConvexHull((vertices-source[0])@basis.T).vertices]
            np.testing.assert_allclose(surface.area(polygon), contact['triangle_areas_m2'][which].sum(),
                                       rtol=1e-7, atol=1e-24)
            heads.append(engine.hull_mesh(G.head_cell(mesh, polygon, face, self.offsets)))
        return heads

    def analyze(self, contact):
        heads = self.heads(contact)
        study = engine.AngleStudy(self.mesh, heads)
        analysis = study.classify(self.mesh.face_normals[np.unique(contact['source_faces'])])
        directions = from_analysis(analysis)
        return dict(candidate_id=contact['candidate_id'], candidate_index=contact['candidate_index'],
                    geometry_signature=signature(contact, self.depth), radius_m=contact['radius_m'],
                    center_m=contact['center_m'].tolist(), area_m2=I.area([contact]),
                    head_cells=len(heads), certified_directions=directions,
                    **summary(directions), analysis=analysis)
