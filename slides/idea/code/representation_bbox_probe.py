"""Read-only check of the proposed occupancy lower bound on saved final models.

The bound uses actual object positions and saved system-ground pressure hulls.
It performs no force solve, Oracle query, Boolean, construction or optimization.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
COOPT = ROOT/'slides/Co-optimize'
sys.path.insert(0, str(COOPT/'evaluation'))
from metrics import bbox, transformed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=COOPT/'data/results/B/data'
        /'evaluation_normal_frontier_more10_20261011_01/summary.json')
    parser.add_argument('--output', type=Path,
        default=Path(__file__).resolve().parents[1]/'data/representation_bbox_probe.json')
    args = parser.parse_args()
    protected = {}

    def protect(path):
        path = path.resolve()
        protected[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    report = json.loads(protect(args.evaluation).read_text())
    if not report['complete'] or report['metric_version'] != 'static_occupancy_v2_overlapping_objects':
        raise ValueError('A completed current static Evaluation is required')
    mesh_path = protect(ROOT/'objects'/report['object']/'mesh.stl')
    body = trimesh.load(mesh_path, force='mesh', process=False)
    rows = []
    for case in report['cases']:
        final_path = ROOT/case['step52_source']/'data/report.json'
        final = json.loads(protect(final_path).read_text())
        if not final['complete'] or not final['passed']:
            raise ValueError('A completed accepted construction is required')
        step51 = ROOT/final['step51_source']/'data'
        files = sorted(step51.glob('pose_*.npz'))
        if len(files) != case['pose_count']:
            raise ValueError('Pressure data must cover every active object pose')
        clouds = []
        for path in files:
            with np.load(protect(path)) as arrays:
                clouds.append(transformed(body.vertices, arrays['T_world_object']))
                ground = arrays['floor_demands_world_m'][arrays['hull_indices']]
                if not np.allclose(ground[:, 2], 0., atol=1e-12, rtol=0):
                    raise ValueError('Required pressure hull must use its actual world floor')
                clouds.append(ground)
        bound = bbox(clouds)
        actual = case['with_full_support']
        tolerance = 1e-7  # Numerical comparison tolerance, not a geometric clearance.
        if (np.any(np.asarray(bound['min_m']) < np.asarray(actual['min_m'])-tolerance)
                or np.any(np.asarray(bound['max_m']) > np.asarray(actual['max_m'])+tolerance)):
            raise ValueError('Required object/ground extrema lie outside the saved complete box')
        rows.append(dict(id=case['id'], lower_bbx=bound,
            nominal_bbx_litres=case['with_step4_support']['volume_litres'],
            completed_bbx_litres=actual['volume_litres'],
            lower_to_completed_ratio=bound['volume_litres']/actual['volume_litres'],
            lower_box_contained_in_completed_box=True))
    for path, expected in protected.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Saved input changed during this read-only probe')
    output = dict(complete=True, scope='read-only lower-bound audit against saved completed Evaluation',
        cases=rows, group_count=len(rows), force_solver_run=False, oracle_query_run=False,
        boolean_run=False, construction_run=False, optimization_run=False,
        input_hashes_match=True, inputs=protected,
        bound_definition='world AABB of actual objects and all required system-ground pressure hulls',
        lower_bound_is_completed_model=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    for row in rows:
        print(row['id'], 'lower', round(row['lower_bbx']['volume_litres'], 2),
              'completed', round(row['completed_bbx_litres'], 2), 'L', flush=True)
    print('SAVED OCCUPANCY LOWER-BOUND CHECK PASSED', len(rows), 'cases')


if __name__ == '__main__':
    main()
