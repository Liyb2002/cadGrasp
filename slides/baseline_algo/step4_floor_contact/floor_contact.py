"""Whole-assembly ground-demand entry point.

The independent-foot helper functions below are retained for old regressions;
the current build/read/schema are imported from whole_assembly before main."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ContinuousNeeds, OBJECTS, OUTPUTS, ROOT, demand, sha256
from step1.cases import pose_name
from step3_scheculer import contacts as I, verification as V, enclosure as E
from step3_scheculer.completion import read_passed, IncompleteSchedule
from step4_floor_contact import equilibrium as Q, footprints as P

STAGE = 'step4_floor_contact'
SCHEMA = 'fixed_independent_footprints_v3'


def output_folder(name):
    return OUTPUTS/name/pose_name()/STAGE


def search_load_paths(name):
    return sorted((OUTPUTS/name/pose_name()/'step3_scheculer/search_loads').glob('after_round_*.json'))


def supplemental_loads(domain, paths):
    loads = {}
    for path in paths:
        for case in json.loads(path.read_text())['counterexamples']:
            evaluated = domain.evaluate(case['work_face_index'], case['u'], case['v'],
                case['theta_rad'], case['phi_rad'], magnitude_mg=case['magnitude_mg'])
            if not bool(evaluated['reachable']):
                raise ValueError('A supplementary load is outside the current reachable domain')
            value = np.asarray(evaluated['need_wrench'], float)
            np.testing.assert_allclose(value, case['need_wrench'], atol=1e-13, rtol=0)
            loads[tuple(value)] = value
    return np.asarray(list(loads.values()), float).reshape(-1, 6)


def array_solution(prefix, result, solver):
    return {prefix+'_basis_indices': np.asarray(solver.bases, int).reshape(-1, solver.matrix.shape[0]),
            prefix+'_assignment': result['assignment'], prefix+'_coefficients_mg': result['weights']}


def summarize(result):
    return {k: v for k, v in result.items() if k not in ('assignment', 'weights')}


def load_label(index, samples):
    if index == 0:
        return dict(kind='zero_process_force', magnitude_mg=0.)
    if index <= samples['count']:
        i = index-1
        return dict(kind='stored_reachable_sample', sample_index=i,
                    pt_m=samples['pt_m'][i], force_push_mg=samples['force_push_mg'][i],
                    work_face_index=samples['work_face_index'][i], parameters=samples['parameters'][i])
    return dict(kind='validated_step3_counterexample', index=index-1-samples['count'])


def continuous_check(problem, solver):
    attempts = []
    lower, upper, vertices = V.outer_box(problem)
    tests = [('axis_aligned_outer_box', vertices, dict(lower_wrench=lower.tolist(), upper_wrench=upper.tolist())),
             ('triangle_tangent_cap_outer_polytope', None, dict(sides=8, bands=1)),
             ('triangle_tangent_cap_outer_polytope', None, dict(sides=16, bands=2))]
    for label, loads, settings in tests:
        if loads is None:
            loads = E.targets(problem, settings['sides'], settings['bands'])[:, :6]/problem.scale
        result = solver.solve(Q.padded_targets(loads, problem.scale, solver.matrix.shape[0]), certified=True)
        attempts.append(dict(method=label, target_count=len(loads), **settings, **summarize(result)))
        if result['passed']:
            return dict(status='verified', method=label, attempts=attempts,
                scope='All work-face positions, entire local 30-degree cap including occluded rays, and magnitude 0–0.5mg; same fixed footprints.',
                coefficient_entry_guard=1e-10, entry_error_rule='1e-10 * (1 + absolute entry), in conditioned coordinates'), loads, result
    return dict(status='unresolved', attempts=attempts,
        reason='Sufficient outer-domain/coefficient tests did not certify coverage; failed outer vertices are not reachable counterexamples.'), None, None


# Current one-body entry; shared geometry and regression helpers stay available.
from step4_floor_contact.whole_assembly import build, read, SCHEMA


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    for name in parser.parse_args().objects or OBJECTS:
        build(name)
