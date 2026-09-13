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


def read(name):
    out = output_folder(name)
    status = json.loads((out/'status.json').read_text())
    if not status.get('complete') or status.get('floor_contact_sha256') != sha256(out/'floor_contact.json'):
        raise RuntimeError('Step 4 is unfinished or its result has been superseded')
    report = I.check_report(out/'floor_contact.json')
    if report.get('schema') != SCHEMA:
        raise RuntimeError('Legacy whole-assembly hull; rebuild Step 4 with independent footprints')
    current = [str(p.relative_to(ROOT)) for p in search_load_paths(name)]
    if report['supplemental_load_sources'] != current:
        raise RuntimeError('New validation loads require rebuilding the independent footprints')
    return report


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


def build(name):
    started = time.monotonic()
    root = OUTPUTS/name/pose_name(); out = output_folder(name)
    out.mkdir(parents=True, exist_ok=True)
    I.save(out/'status.json', dict(complete=False, status='designing_independent_footprints'))
    source = root/'step_1_needs'
    domain = ContinuousNeeds.read(source/'needs.json')
    samples = json.loads((source/'samples.json').read_text())
    assert samples['provenance']['physical_domain_sha256'] == sha256(source/'needs.json')
    sampled = np.asarray(samples['need_wrench'])
    np.testing.assert_allclose(sampled, demand(samples['pt_m'], samples['force_push_mg'], domain.com, domain.gravity), atol=1e-13, rtol=0)
    extra_paths = search_load_paths(name)
    extra = supplemental_loads(domain, extra_paths)
    loads = np.vstack([np.r_[-domain.gravity, np.zeros(3)], sampled, extra])
    snapshot = ROOT/domain.data['provenance']['setup_snapshot']
    with np.load(snapshot) as z: pivot = z['floor_contact_m'].copy().reshape(3)
    contact_path = root/'step3_scheculer/final_contacts.npz'
    contacts = I.read_contacts(contact_path) if contact_path.exists() else []
    inputs = [source/'needs.json', source/'samples.json', snapshot]+extra_paths
    inputs += [p for p in [contact_path, root/'step3_scheculer/schedule.json', root/'step3_scheculer/status.json'] if p.exists()]
    arrays = dict(load_wrenches=loads, original_pivot_m=pivot, moment_origin_m=domain.com,
                  sampled_load_count=np.array(len(sampled)), supplemental_load_count=np.array(len(extra)))
    record = dict(object=name, pose=pose_name(), stage=STAGE, schema=SCHEMA, complete=True,
        role='Fixed four-pad geometry per Step 3 contact; shared reactions are checked without moving feet.',
        sample_count=len(sampled), load_count=len(loads), zero_process_force_included=True,
        supplemental_load_count=len(extra), supplemental_load_sources=[str(p.relative_to(ROOT)) for p in extra_paths],
        selected_ids=[c['candidate_id'] for c in contacts], contact_count=len(contacts),
        contacts_fixed=True, contact_geometry_sha256=sha256(contact_path) if contact_path.exists() else None,
        ground_footprints=P.fixed_layout(contacts, domain.mesh), shape_designed=bool(contacts), trajectory_selected=False,
        footprint_template=dict(P.FIXED_TEMPLATE), footprint_layout_depends_on_loads=False,
        foot_positions_frozen_for_step5=True, connectors_may_move_feet=False,
        sampled_independent_equilibrium_verified=False, continuous_domain_coverage_proved=False,
        physical_supports_verified=False, connections_constructed=False,
        assumptions=dict(support_mass='zero', anchoring=False, inter_support_force_transfer=False,
            workpiece_floor='original point, vertical reaction only',
            workpiece_support='nonnegative inward-normal forces, no upper bound',
            floor_friction='sufficient; successful designs report a finite sufficient Coulomb coefficient'),
        arrays_file='floor_contact.npz', scope='Bearing regions and per-body static equilibrium only. Solid connections, work-volume clearance, strength and installation belong to Step 5.')
    def finish():
        # Even an incomplete force search retains fixed, explicitly uncertified
        # foot candidates so Step 5 can report each construction independently.
        record['footprint_layout_available'] = len(record['ground_footprints']) == len(contacts) and bool(contacts)
        record['footprint_certification'] = ('continuous' if record['continuous_domain_coverage_proved'] else
            'sampled_only' if record['sampled_independent_equilibrium_verified'] else 'unverified_candidate')
        if 'pressure_centers_xz_m' not in arrays:
            arrays['pressure_centers_xz_m'] = np.full((0,len(contacts),2),np.nan)
        if contact_path.exists():
            assert sha256(contact_path) == record['contact_geometry_sha256']
        np.savez_compressed(out/'floor_contact.npz', **arrays)
        record.update(elapsed_seconds=time.monotonic()-started,
            provenance=dict(inputs=I.hashes(inputs), code=I.hashes([Path(__file__), Path(Q.__file__), Path(P.__file__), Path(V.__file__), Path(E.__file__), Path(I.__file__)])),
            artifacts={'floor_contact.npz': sha256(out/'floor_contact.npz')})
        I.save(out/'floor_contact.json', record)
        I.save(out/'status.json', dict(complete=True, status=record['status'], floor_contact_sha256=sha256(out/'floor_contact.json')))
        print(name, pose_name(), 'Step 4:', record['status'], flush=True)
        return record
    try:
        read_passed(name)
    except IncompleteSchedule as error:
        record.update(status='upstream_step3_not_verified', reason=str(error))
        return finish()
    problem = V.C.Problem(name)
    scale = problem.scale
    points, normals, owners = Q.contact_rays(domain, contacts, pivot)
    arrays.update(contact_points_m=points, contact_normals=normals, contact_owners=owners, scale=scale)
    relaxed = Q.relaxed_matrix(points, normals, owners, domain.com, scale, len(contacts))
    solver = Q.BatchSolver(relaxed)
    targets = Q.padded_targets(loads, scale, relaxed.shape[0])
    result = solver.solve(targets)
    record['necessary_vertical_test'] = summarize(result)
    if not result['passed']:
        failure = result['diagnostics'][0]; index = failure['index']
        separator = Q.exact_relaxed_separator(relaxed, targets[index], points, normals, owners, domain.com, scale, len(contacts), loads[index])
        failure = dict(failure, load=load_label(index, samples), need_wrench=loads[index].tolist(), exact_separator=separator)
        control = linprog(np.ones(len(points)), A_eq=relaxed[:6, :len(points)],
                          b_eq=targets[index, :6], bounds=(0, None), method='highs', options=Q.OPTIONS)
        if control.success and control.x.min() >= 0 and np.max(np.abs(relaxed[:6, :len(points)]@control.x-targets[index, :6])) <= 2e-8:
            raw = Q.wrench(points, normals, domain.com)
            shares = np.array([control.x[owners == j]@raw[owners == j] for j in range(len(contacts))])
            failure['workpiece_only_allocation'] = dict(support_wrenches=shares.tolist(),
                required_floor_normal_mg=shares[:, 1].tolist(),
                residual_conditioned=float(np.max(np.abs(relaxed[:6, :len(points)]@control.x-targets[index, :6]))),
                scope='One diagnostic workpiece solution; infeasibility of all reallocations is tested separately.')
            arrays['diagnostic_contact_coefficients_mg'] = control.x
            arrays['diagnostic_support_wrenches'] = shares
        arrays.update(failing_load_index=np.array(index), relaxed_equilibrium_matrix=relaxed)
        record.update(status='independent_supports_infeasible' if separator else 'independent_supports_infeasible_numeric' if failure['status'] == 'infeasible_numeric' else 'equilibrium_unresolved',
            failure=failure, reason='The fixed heads fail even with arbitrary per-support floor moments and horizontal forces; only floor normal >= 0 is imposed.' if separator or failure['status'] == 'infeasible_numeric' else 'Numerical feasibility was not resolved.')
        return finish()
    attempts = []; accepted = None
    feet = record['ground_footprints']
    geometry = P.check(feet, domain.mesh)
    assert geometry['passed']
    for mu in (1., 4., 16., 64.):
        matrix, ground = Q.grounded_matrix(points, normals, owners, domain.com, scale, feet, mu)
        actual = Q.BatchSolver(matrix)
        checked = actual.solve(Q.padded_targets(loads, scale, matrix.shape[0]))
        attempts.append(dict(sufficient_friction_coefficient=mu, **summarize(checked)))
        if checked['passed']:
            accepted = feet, geometry, mu, actual, checked, ground
            break
    record['footprint_search'] = dict(attempts=attempts, global_optimum_claimed=False,
        geometry_attempt_count=1, feet_moved=False,
        rule='One fixed four-pad template; only reactions and finite sufficient friction coefficients are searched.')
    if not accepted:
        record.update(status='fixed_footprint_equilibrium_unresolved',
                      reason='The fixed feet did not pass the finite friction menu. Geometry is retained for independent Step 5 construction; no global impossibility proof.')
        return finish()
    feet, geometry, mu, actual, checked, ground = accepted
    shares = Q.support_wrenches(checked, actual, points, normals, owners, domain.com, len(contacts))
    cop, loaded = Q.pressure_centers(shares, domain.com)
    arrays.update(support_wrenches=shares, pressure_centers_xz_m=cop, loaded_support_mask=loaded,
                  equilibrium_matrix=actual.matrix, ground_points_m=ground['points_m'],
                  ground_forces=ground['forces'], ground_owners=ground['owners'])
    continuous, outer_loads, proof = continuous_check(problem, actual)
    if proof is not None:
        arrays['continuous_outer_load_wrenches'] = outer_loads
        arrays.update(array_solution('continuous', proof, actual))
        outer_shares = Q.support_wrenches(proof, actual, points, normals, owners, domain.com, len(contacts))
        arrays['continuous_pressure_centers_xz_m'] = Q.pressure_centers(outer_shares, domain.com)[0]
    arrays.update(array_solution('sample', checked, actual))
    record.update(status='independent_footprints_verified' if continuous['status'] == 'verified' else 'sampled_footprints_verified_continuous_unresolved',
        ground_footprints=feet, shape_designed=True,
        sufficient_friction_coefficient=mu, footprint_geometry_check=geometry,
        sampled_independent_equilibrium_verified=True, continuous_verification=continuous,
        continuous_domain_coverage_proved=continuous['status'] == 'verified',
        ground_bearing_equilibrium_verified=continuous['status'] == 'verified',
        pressure_center_scope='Saved feasible allocations, not every possible allocation; continuous outer-vertex allocations are included when certified. Step 5 must preserve every pad exactly.')
    return finish()


# Current one-body entry; earlier independent-body helpers remain for regressions.
from step4_floor_contact.whole_assembly import build, read, SCHEMA


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    for name in parser.parse_args().objects or OBJECTS:
        build(name)
