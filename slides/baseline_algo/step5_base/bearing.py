"""Shared head/base equilibrium; used during base selection and final replay.

The virtual massless support already includes the fixed heads and candidate
floor contacts. Connecting them changes collision geometry, not these equations.
"""
from time import perf_counter
import numpy as np
from scipy.spatial import ConvexHull
from step4_floor_contact import equilibrium as Q

def bearing_rays(domain, contacts, pivot, friction):
    """Use the same sufficient floor friction at the workpiece pivot and base."""
    points,normals,owners=Q.contact_rays(domain,contacts,pivot)
    assert owners[0] == -1 and np.count_nonzero(owners<0) == 1
    rays=np.array([[friction,0,1],[-friction,0,1],[0,friction,1],[0,-friction,1]],float)
    return (np.vstack([np.tile(points[0],(4,1)),points[1:]]),
            np.vstack([rays,normals[1:]]),np.r_[np.full(4,-1,int),np.zeros(len(owners)-1,int)])



def floor_vertices(base):
    """Extreme vertices of ACTUAL coplanar contact material, never its pivot.

    At common friction, [f, (p-c) x f] is affine in p. Every omitted floor ray
    is a convex combination of equal-force rays at these retained vertices.
    Thus their nonnegative wrench cones agree, including yaw and tangential
    friction. This does not fill the holes with physical material.
    """
    xy = np.unique(np.concatenate(base['pads_xy_m']), axis=0)
    return xy[ConvexHull(xy).vertices]


def grounded_matrix(points, normals, owners, origin, scale, feet, friction):
    reduced = [dict(foot, pads_xy_m=[floor_vertices(foot)]) for foot in feet]
    return Q.grounded_matrix(points, normals, owners, origin, scale, reduced, friction)


def seed_probes(loads, count=48):
    """A cheap rejection screen; acceptance always evaluates every load."""
    loads = np.asarray(loads)
    if not len(loads):
        return []
    ids = np.r_[np.linspace(0, len(loads)-1, min(count, len(loads)), dtype=int),
                np.argmin(loads, axis=0), np.argmax(loads, axis=0)]
    return np.unique(ids).tolist()

def bearing(domain, contacts, floor, base, friction_values=(1., 4., 16., 64., 256., 1024., 4096., 16384., 65536.), witnesses=None):
    """Continue the friction menu after either sampled OR continuous failure."""
    scale = np.r_[np.ones(3), np.ones(3)/domain.mesh.extents.max()]
    report = dict(sampled_passed=False, continuous_passed=False, body_count=1, reactions_shared=True,
        attempts=[], original_object_floor_friction_included=True, finite_friction_menu_is_sufficient_not_complete=True,
        assumption='floor friction is sufficiently large; coefficient is a numerical witness, not a material input',
        search_exhaustion_means='unresolved under the sufficient-friction assumption, not a proof of infeasibility')
    arrays = {}
    witnesses = set() if witnesses is None else witnesses
    for mu in friction_values:
        points,normals,owners=bearing_rays(domain,contacts,floor['original_pivot_m'],mu)
        matrix, ground = grounded_matrix(points, normals, owners, domain.com, scale, [base], mu)
        solver = Q.BatchSolver(matrix)
        # Use the SAME certificate standard on remembered continuous loads.
        # The old non-certified probe repeatedly accepted the precise boundary
        # load whose full continuous certificate had just failed.
        began = perf_counter()
        probe_failed = False
        for group, key, certified in [('continuous', 'continuous_outer_load_wrenches', True),
                                      ('sample', 'load_wrenches', False)]:
            indices = sorted({i for g, i in witnesses if g == group} |
                             set(seed_probes(floor[key])))
            if not indices:
                continue
            probe = solver.solve(Q.padded_targets(floor[key][indices], scale, 12),
                                 certified=certified)
            if not probe['passed']:
                for diagnostic in probe['diagnostics']:
                    witnesses.add((group, indices[int(diagnostic['index'])]))
                report['attempts'].append(dict(friction=mu, sampled_passed=False, continuous_passed=False,
                    probe_only=True, probe_group=group, certified_probe=certified,
                    probe_load_indices=indices, probe_diagnostics=probe['diagnostics'],
                    elapsed_s=perf_counter()-began, lp_count=solver.lp_count))
                probe_failed = True
                break
        if probe_failed:
            continue
        # Continuous certification is usually the rejecting condition. Do it
        # before the 32769 sampled loads, while retaining both on acceptance.
        continuous = solver.solve(Q.padded_targets(floor['continuous_outer_load_wrenches'], scale, 12), certified=True)
        sampled = solver.solve(Q.padded_targets(floor['load_wrenches'], scale, 12)) if continuous['passed'] else None
        report['attempts'].append(dict(friction=mu, sampled_passed=bool(sampled and sampled['passed']),
            sampled_diagnostics=sampled['diagnostics'] if sampled else [], continuous_passed=continuous['passed'],
            continuous_diagnostics=continuous['diagnostics'], elapsed_s=perf_counter()-began, lp_count=solver.lp_count))
        for group,value in [('sample',sampled),('continuous',continuous)]:
            if value is not None:
                for diagnostic in value['diagnostics']:
                    witnesses.add((group,int(diagnostic['index'])))
        if sampled is None or not sampled['passed']: continue
        report.update(sampled_passed=True, continuous_passed=continuous['passed'], sufficient_friction_coefficient=mu)
        arrays = dict(contact_points_m=points, contact_normals=normals, contact_owners=owners,
            ground_points_m=ground['points_m'], ground_forces=ground['forces'], ground_owners=ground['owners'],
            moment_origin_m=domain.com, scale=scale, equilibrium_matrix=matrix)
        for prefix, value in [('sample', sampled), ('continuous', continuous)]:
            arrays[prefix+'_assignment'] = value['assignment']
            arrays[prefix+'_coefficients_mg'] = value['weights']
            arrays[prefix+'_basis_indices'] = np.asarray(solver.bases, int).reshape(-1, 12)
        if continuous['passed']: break
    return report, arrays

