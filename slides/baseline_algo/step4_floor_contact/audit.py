"""Audit current whole-assembly floor demand; retain legacy force-replay helpers."""
import argparse
from fractions import Fraction as FQ
from pathlib import Path
import sys
import numpy as np
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step4_floor_contact import floor_contact as F


def moments(points, forces, origin):
    r = points-origin
    return np.stack([r[..., 1]*forces[..., 2]-r[..., 2]*forces[..., 1],
                     r[..., 2]*forces[..., 0]-r[..., 0]*forces[..., 2],
                     r[..., 0]*forces[..., 1]-r[..., 1]*forces[..., 0]], axis=-1)


def replay(arrays, prefix, loads, count, friction):
    assignment = arrays[prefix+'_assignment']; bases = arrays[prefix+'_basis_indices']
    coefficients = arrays[prefix+'_coefficients_mg']
    assert len(assignment) == len(loads) and np.all(assignment >= 0)
    assert np.min(coefficients) >= -2e-9
    p, n, owner = arrays['contact_points_m'], arrays['contact_normals'], arrays['contact_owners']
    ground_p, ground_f, ground_owner = arrays['ground_points_m'], arrays['ground_forces'], arrays['ground_owners']
    origin, scale = arrays['moment_origin_m'], arrays['scale']
    # Reconstruct every column by force and moment accumulation, independently
    # of the block-matrix assembly used by the solver.
    columns = np.zeros((len(p)+len(ground_p), count+1, 6))
    head = np.c_[n, moments(p, n, origin)]
    columns[:len(p), 0] = head
    for i in range(len(p)):
        if owner[i] >= 0: columns[i, owner[i]+1] = -head[i]
    ground = np.c_[ground_f, moments(ground_p, ground_f, origin)]
    for i in range(len(ground_p)): columns[len(p)+i, ground_owner[i]+1] = ground[i]
    assert np.all(ground_f[:, 1] > 0)
    assert np.all(np.linalg.norm(COORD.floor(ground_f), axis=1) <= friction*ground_f[:, 1]+1e-12)
    if 'equilibrium_matrix' in arrays:
        np.testing.assert_allclose((columns*scale).reshape(len(columns), -1).T,
                                   arrays['equilibrium_matrix'], atol=1e-13, rtol=1e-13)
    maximum = 0.
    for k, basis in enumerate(bases):
        ids = np.flatnonzero(assignment == k)
        if not len(ids): continue
        residual = np.einsum('ni,ibc->nbc', coefficients[ids], columns[basis])
        residual[:, 0] -= loads[ids]
        maximum = max(maximum, float(np.max(np.abs(residual*scale))))
    assert maximum <= 5e-8, maximum
    return dict(load_count=len(loads), maximum_body_equilibrium_residual_conditioned=maximum,
                minimum_reaction_coefficient_mg=float(coefficients.min()),
                every_workpiece_and_support_checked=True)


def replay_separator(arrays, proof):
    dual = list(map(FQ, proof['dual_exact']))
    origin = list(map(lambda v: FQ(float(v)), arrays['moment_origin_m']))
    scores = []
    for p, n, owner in zip(arrays['contact_points_m'], arrays['contact_normals'], arrays['contact_owners']):
        n = [FQ(float(x)) for x in n]
        r = [FQ(float(x))-c for x, c in zip(p, origin)]
        g = n+[r[1]*n[2]-r[2]*n[1], r[2]*n[0]-r[0]*n[2], r[0]*n[1]-r[1]*n[0]]
        value = sum(a*b for a, b in zip(dual[:6], g))
        if owner >= 0: value += dual[6+owner]*n[1]
        scores.append(value)
    scores += [-v for v in dual[6:]]
    load = arrays['load_wrenches'][int(arrays['failing_load_index'])]
    debt = sum(h*FQ(float(b)) for h, b in zip(dual[:6], load))
    assert min(scores) >= 0 and debt < 0
    assert str(min(scores)) == proof['min_generator_dot_exact']
    assert str(debt) == proof['target_dot_exact']
    return dict(exact_separation_replayed=True, generators_checked=len(scores), target_dot_exact=str(debt))


def run(name):
    report = F.read(name); out = F.output_folder(name)
    arrays = F.I.load_npz(out/'floor_contact.npz')
    contact_path = out.parent/'step3_scheculer/final_contacts.npz'
    if contact_path.exists(): assert F.sha256(contact_path) == report['contact_geometry_sha256']
    assert not report['connections_constructed'] and not report['trajectory_selected']
    assert 'required_hull_xz_m' not in arrays and 'floor_demands_xz_m' not in arrays
    results = dict(contacts_unchanged=True, shared_ground_hull_absent=True)
    assert report['foot_positions_frozen_for_step5'] and not report['connectors_may_move_feet']
    if report['ground_footprints']:
        domain = F.ContinuousNeeds.read(out.parent/'step_1_needs/needs.json')
        assert [p['candidate_id'] for p in report['ground_footprints']] == report['selected_ids']
        assert all(p['fixed_for_step5'] and p['height_m'] > 0 for p in report['ground_footprints'])
        contacts = F.I.read_contacts(contact_path)
        assert report['ground_footprints'] == F.P.fixed_layout(contacts, domain.mesh)
        assert report['footprint_template'] == F.P.FIXED_TEMPLATE
        assert report['footprint_layout_depends_on_loads'] is False
        results['fixed_footprint_geometry'] = F.P.check(report['ground_footprints'], domain.mesh)
        assert results['fixed_footprint_geometry']['passed']
    if 'contact_points_m' in arrays:
        from step4_floor_contact import equilibrium as Q
        domain = F.ContinuousNeeds.read(out.parent/'step_1_needs/needs.json')
        contacts = F.I.read_contacts(contact_path)
        p, n, owner = Q.contact_rays(domain, contacts, arrays['original_pivot_m'])
        for key, rebuilt in [('contact_points_m', p), ('contact_normals', n), ('contact_owners', owner)]:
            np.testing.assert_array_equal(arrays[key], rebuilt)
        results['contact_rays_rebuilt_from_step3_surfaces'] = True
    if report['sampled_independent_equilibrium_verified']:
        results['footprint_geometry'] = F.P.check(report['ground_footprints'], domain.mesh)
        assert results['footprint_geometry']['passed']
        mu = report['sufficient_friction_coefficient']
        results['samples'] = replay(arrays, 'sample', arrays['load_wrenches'], report['contact_count'], mu)
        for j, foot in enumerate(report['ground_footprints']):
            allowed = Polygon(foot['hull_xz_m']).buffer(1e-7)
            finite = arrays['pressure_centers_xz_m'][:, j]
            if 'continuous_pressure_centers_xz_m' in arrays:
                finite = np.vstack([finite, arrays['continuous_pressure_centers_xz_m'][:, j]])
            finite = finite[np.isfinite(finite).all(axis=1)]
            assert all(allowed.covers(Point(p)) for p in finite)
            ground_points = arrays['ground_points_m'][arrays['ground_owners'] == j]
            corners = np.concatenate(foot['pads_xz_m'])
            assert np.all(ground_points[:, 1] == 0)
            assert all(np.any(np.all(corners == COORD.floor(p), axis=1)) for p in ground_points)
        if report['continuous_domain_coverage_proved']:
            results['continuous'] = replay(arrays, 'continuous', arrays['continuous_outer_load_wrenches'], report['contact_count'], mu)
            problem = F.V.C.Problem(name)
            attempt = report['continuous_verification']['attempts'][-1]
            rebuilt = (F.V.outer_box(problem)[2] if attempt['method'] == 'axis_aligned_outer_box' else
                       F.E.targets(problem, attempt['sides'], attempt['bands'])/problem.scale)
            np.testing.assert_array_equal(rebuilt, arrays['continuous_outer_load_wrenches'])
            targets = Q.padded_targets(rebuilt, arrays['scale'], arrays['equilibrium_matrix'].shape[0])
            for k, basis in enumerate(arrays['continuous_basis_indices']):
                rows = np.flatnonzero(arrays['continuous_assignment'] == k)
                if len(rows):
                    passed, _, _ = Q.membership(arrays['equilibrium_matrix'], basis, targets[rows], certified=True)
                    assert np.all(passed)
            results['continuous']['outer_domain_and_coefficient_bounds_rebuilt'] = True
    proof = report.get('failure', {}).get('exact_separator')
    if proof: results['infeasibility'] = replay_separator(arrays, proof)
    result = dict(object=name, complete=True, passed=True, design_status=report['status'], checks=results,
                  audit_pass_does_not_mean_design_feasible=True,
                  provenance=dict(inputs=F.I.hashes([out/'floor_contact.json', out/'floor_contact.npz']),
                                  code=F.I.hashes([Path(__file__)])))
    F.I.save(out/'audit.json', result)
    print(name, 'Step 4 evidence audit passed:', report['status'], flush=True)
    return result


# Current one-body entry; earlier independent-body helpers remain for regressions.
from step4_floor_contact.whole_assembly import audit as run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('objects', nargs='*')
    for name in parser.parse_args().objects or F.OBJECTS: run(name)
