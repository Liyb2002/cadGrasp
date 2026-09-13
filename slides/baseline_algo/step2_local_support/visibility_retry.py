"""Conservative recovery of nearly coincident possible-shadow unions.

An outer shadow may grow, but must not lose an occluder. Always-shadowed
regions retain the original construction. Recorded provenance makes this
opt-in recovery reproducible without changing other running cases.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import json
import runpy
import sys
import numpy as np
from shapely.geometry import MultiPoint, GeometryCollection
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent.parent


def outer_union(shapes, margin):
    if not shapes:
        return GeometryCollection()
    # Expand each input before overlay: expanding a damaged union is too late.
    expanded = [s.buffer(margin, join_style='mitre') for s in shapes]
    result = unary_union(expanded)
    if any(not result.covers(s) for s in shapes):
        # The convex cover is a valid upper bound, including disconnected gaps.
        # Build from input coordinates, independent of the possibly damaged union.
        result = MultiPoint(np.vstack([np.asarray(s.exterior.coords) if s.geom_type == 'Polygon'
                                      else np.asarray(s.coords) for s in shapes])).convex_hull.buffer(margin, join_style='mitre')
    return result


def shadows(occluders, slopes, source):
    from step2_local_support import visibility as V
    possible = []; certain = []
    directions = np.c_[slopes, np.ones(len(slopes))]
    source_buffer = source.buffer(V.POSITION_MARGIN, join_style='mitre')
    for occluder in occluders:
        vertices, normal = occluder[:2]
        projected = vertices[:, None, :2]-vertices[:, None, 2:]*slopes[None]
        envelope = MultiPoint(projected.reshape(-1, 2)).convex_hull
        if not envelope.intersects(source_buffer): continue
        possible.append(envelope)
        dots = directions@normal
        if not (dots.min() > 1e-10 or dots.max() < -1e-10): continue
        if len(occluder) > 2:
            vertices = occluder[2]
            if len(vertices) < 3: continue
            projected = vertices[:, None, :2]-vertices[:, None, 2:]*slopes[None]
        common = None
        for index in range(len(slopes)):
            shadow = MultiPoint(projected[:, index]).convex_hull
            common = shadow if common is None else common.intersection(shadow)
            if common.is_empty or common.area == 0: break
        if common is not None and common.area > 0: certain.append(common)
    return outer_union(possible, V.POSITION_MARGIN), unary_union(certain) if certain else GeometryCollection()


def install():
    from step2_local_support import visibility as V, work_volume as W
    V.shadows = shadows
    original = W.WorkVolume.export
    def export(self, out, needs_path):
        self.visibility['possible_shadow_union_recovery'] = dict(
            method='Expand each possible shadow before union; check input containment; convex input cover on failure',
            normalized_input_expansion=V.POSITION_MARGIN, always_shadow_unchanged=True)
        path = original(self, out, needs_path)
        report = json.loads(path.read_text())
        report['provenance']['code'].update(W.I.hashes([Path(__file__)]))
        W.I.save(path, report)
        return path
    W.WorkVolume.export = export


if __name__ == '__main__':
    stage = Path(sys.argv[1]).resolve()
    if HERE not in stage.parents: raise ValueError('Stage must be inside baseline_algo')
    install()
    sys.argv = [str(stage)]+sys.argv[2:]
    runpy.run_path(str(stage), run_name='__main__')
