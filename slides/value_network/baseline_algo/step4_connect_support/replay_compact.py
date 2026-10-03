"""Independently reload exported compact solids and replay physical geometry.

No construction or placement optimization is run. All original Step0 samples
and frozen Step3 contact/root geometry remain the authorities.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import MultiPoint
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step4_connect_support import boxed_support as F, build_coupled_saddle as S
from step4_connect_support import deterministic_space as D, working_surface as W
from step4_connect_support.run_boxed_batch import protected_hashes
from step3_scheculer import contacts as I
from step5_evaluate import evaluate as E, render_bbox as VIS


def replay(group, source='boxed_support'):
    out = group/'step4/data'/source
    report = json.loads((out/'report.json').read_text())
    assert report['passed'] and report['complete']
    case, _ = F.read_case(group, out)
    mesh = trimesh.load(out/'shape.obj', force='mesh', process=False)
    bases = np.asarray(report['placement']['bases'])
    offsets = np.asarray(report['placement']['offsets'])
    directions = np.asarray(report['placement']['directions'])
    # Fresh continuous sweeps: do not reuse the constructor's cache/certificate.
    sweeps = []
    for task, direction, basis, offset, detail in zip(case.tasks, directions, bases, offsets, report['sweep_conditioning']):
        swept = S.swept_solid(task.domain.mesh, -S.SWEEP_LENGTH*direction, fan_in=detail['boolean_fan_in'])
        sweeps.append(trimesh.Trimesh(swept.vertices@basis+offset, swept.faces, process=False))
    with contextlib.redirect_stdout(io.StringIO()):
        checks, _ = S.verify(case.tasks, case.groups, case.support_seeds,
            directions, bases, offsets, mesh, sweeps, check_equilibrium=False)
    coverage = []
    for check, demands in zip(checks, case.demands):
        ground = MultiPoint(check['actual_ground_hull_xy_m']).convex_hull
        missing = MultiPoint(demands).convex_hull.difference(ground.buffer(1e-10)).area
        coverage.append(dict(pose=check['pose'], original_point_count=len(demands),
                             uncovered_area_m2=float(missing), passed=ground.area > 0 and missing <= 1e-12))
    working = [W.check(trimesh.Trimesh((mesh.vertices-o)@b.T, mesh.faces, process=False), task)
               for task, b, o in zip(case.tasks, bases, offsets)]
    actual = D.measure(case, mesh, bases, offsets)
    np.testing.assert_allclose(actual['extents_mm'], report['space_budget']['extents_mm'], atol=1e-6, rtol=0)
    parts = S.solid(mesh).decompose()
    solid_count = sum(p.volume()*S.SCALE**3 > 1e-16 for p in parts)
    boxes = F.box_checks(mesh, report['result']['box'], bases, offsets, case.poses)
    passed = (mesh.is_watertight and mesh.is_winding_consistent and solid_count == 1
              and all(c['passed'] for c in coverage) and all(c['passed'] for c in working)
              and all(c['all_inside'] for c in boxes))
    result = dict(passed=bool(passed), scope='step4_geometry_only', fresh_continuous_sweeps=True,
        exported_mesh_sha256=I.sha256(out/'shape.obj'), checks=checks, ground_coverage=coverage,
        working_surfaces=working, box_checks=boxes, space_budget=actual,
        watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
        connected_material_components=solid_count, step3_passed=case.schedule['passed'])
    footprint, _, supports = E.measure([t.domain.mesh.vertices for t in case.tasks], mesh.vertices, bases, offsets)
    result['step5_footprint_metrics'] = footprint
    if passed:
        VIS.draw(out/'footprint.png', [t.domain.mesh for t in case.tasks], supports, mesh.faces, footprint)
    I.save(out/'replay.json', result)
    assert passed, group.name
    print('REPLAY', group.name, 'passed', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    parser.add_argument('--groups', nargs='+')
    parser.add_argument('--source', choices=['boxed_support', 'growing_support'], default='boxed_support')
    args = parser.parse_args()
    root = I.OUTPUTS/args.object
    groups = ([root/name for name in args.groups if name != 'pose1+3'] if args.groups else
              sorted(p.parent.parent.parent.parent for p in root.glob(f'pose*+*/step4/data/{args.source}/report.json')
                     if p.parent.parent.parent.parent.name != 'pose1+3'))
    protected = protected_hashes(root)
    results = {group.name: replay(group,args.source)['passed'] for group in groups}
    assert protected == protected_hashes(root)
    print('REPLAY COMPLETE', len(results), 'protected inputs unchanged', flush=True)
