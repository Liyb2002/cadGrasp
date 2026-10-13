"""Shared fixture transforms and offline presentation, independent of a design recipe."""
from pathlib import Path
import numpy as np
from step1.needs import ROOT
from step2_local_support import geometry as G
def cells_for(contact, domain, offsets):
    return [G.head_cell(domain.mesh, contact['triangles_m'][contact['source_faces'] == f, 1], int(f), offsets)
            for f in np.unique(contact['source_faces'])]


def frame(direction, inverted=False):
    x = np.asarray(direction, float)
    z = np.array([0., 0., -1. if inverted else 1.])
    return np.column_stack([x, np.cross(z, x), z])


def local_to_world(points, basis, offset):
    return (points-offset)@basis.T


def pack(mesh):
    return dict(vertices=mesh.vertices.ravel().tolist(), faces=mesh.faces.ravel().tolist())


def export_viewer(out, mesh, visual, report, problems, bases, offsets, colors):
    from step4_connect_support.refresh_shared_geometry_view import head_regions, write_viewer
    data = dict(fixture=pack(mesh), parts=[dict(id=label, color=colors.get(label, '#98b2c0'), **pack(m)) for label, m in visual],
        poses=[dict(name=p.pose, object=pack(p.domain.mesh), work_faces=p.domain.work_ids.tolist(),
                    rotation=b.tolist(), translation=(-b@o).tolist()) for p, b, o in zip(problems, bases, offsets)],
        dimensions_mm=(mesh.extents*1000).tolist(), report=report)
    data['head_regions'] = head_regions(report, bases, offsets)
    write_viewer(out, data)

