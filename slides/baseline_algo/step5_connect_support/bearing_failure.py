"""Illustrate an actual rejected load without inventing a falling trajectory."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linprog
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.cases import selected_pose, pose_name
from step5_connect_support import belt_assembly as B, visual_details as V
from step3_scheculer import contacts as I


def reaction_check(points, normals, owners, com, scale, target):
    """Maximize support compression over all workpiece-equilibrium allocations.

    This necessary test allows any footprint, floor moment and shear at the
    support. A strictly negative maximum is a numerical uplift diagnosis.
    It is separate from the actual finite-footprint acceptance test.
    """
    raw = B.Q.wrench(points, normals, com)
    matrix = (raw*scale).T
    vertical = np.where(owners >= 0, normals[:, 1], 0.)
    result = linprog(-vertical, A_eq=matrix, b_eq=target*scale,
                     bounds=(0, None), method='highs', options=B.Q.OPTIONS)
    report = dict(maximum_normal_lp_status=int(result.status), numerical_diagnostic=True,
                  exact_infeasibility_certificate=False,
                  support_self_weight_included=False)
    if result.success:
        weights = result.x
        maximum = float(vertical@weights)
        report.update(maximum_support_floor_normal_mg=maximum,
                      uplift_required_even_with_unrestricted_support_footprint=maximum < -B.Q.TOL)
    else:
        fallback = linprog(np.ones(len(points)), A_eq=matrix, b_eq=target*scale,
                           bounds=(0, None), method='highs', options=B.Q.OPTIONS)
        if not fallback.success:
            report.update(object_only_reaction_found=False)
            return report, None
        weights = fallback.x
    residual = float(np.max(np.abs(matrix@weights-target*scale)))
    if residual > B.Q.TOL or weights.min() < -B.Q.TOL:
        report.update(object_only_reaction_found=False, residual=residual)
        return report, None
    head = weights[owners >= 0]@raw[owners >= 0]
    report.update(object_only_reaction_found=True, residual=residual,
                  head_resultant_on_workpiece=head.tolist(),
                  head_resultant_on_support=(-head).tolist(),
                  required_floor_normal_for_shown_allocation_mg=float(head[1]),
                  allocation_scope='One workpiece-equilibrium allocation; not a feasible support equilibrium.')
    return report, weights


def run(name):
    out = B.OUTPUTS/name/pose_name()/B.STAGE
    report = I.check_report(out/'connection.json')
    if report['passed'] or not report['geometry_constructed'] or report['bearing'].get('continuous_passed'):
        return None
    domain, contacts, schedule, floor, directions, work, paths = B.A.read_inputs(name)
    attempt = report['bearing']['attempts'][-1]
    continuous = bool(attempt['sampled_passed'])
    diagnostics = attempt['continuous_diagnostics'] if continuous else attempt['sampled_diagnostics']
    if not diagnostics:
        return None
    index = diagnostics[0]['index']
    key = 'continuous_outer_load_wrenches' if continuous else 'load_wrenches'
    target = floor[key][index]
    points, normals, owners = B.bearing_rays(domain, contacts, floor['original_pivot_m'], attempt['friction'])
    scale = np.r_[np.ones(3), np.ones(3)/domain.mesh.extents.max()]
    check, weights = reaction_check(points, normals, owners, domain.com, scale, target)
    result = dict(object=name, pose=pose_name(), complete=True, load_array=key, load_index=int(index),
                  tested_friction=attempt['friction'], original_diagnostic=diagnostics[0],
                  continuous_outer_vertex_not_necessarily_a_physical_load=continuous,
                  target_wrench=target.tolist(), reaction_check=check,
                  trajectory_passed=report['trajectory_verified'],
                  scope='Static failed-load diagnostic; no dynamic fall or new acceptance claim.')
    data = I.load_npz(out/'geometry.npz')
    size = 820
    view = V.camera(domain, [(dict(candidate_id='assembly'), data)])
    panel = V.scene(domain, contacts, [(dict(candidate_id='assembly'), data)], size,
                    view=view, xray=True)
    ink = ImageDraw.Draw(panel)
    def arrow(point, force, color):
        length = np.linalg.norm(force)
        if length < 1e-12:
            return
        tip = point+.2*domain.mesh.extents.max()*force/length
        a, b = V.R.project(np.array([point, tip]), *[view[0], view[2], view[1], size])[:, :2]
        V.arrow(ink, a, b, color=color, width=7)
    arrow(domain.com, np.array([0., -1., 0.]), '#303b3e')
    if weights is not None:
        arrow(np.mean([c['center_m'] for c in contacts], axis=0),
              np.asarray(check['head_resultant_on_support'])[:3], '#b93535')
    page = Image.new('RGB', (1660, 1090), V.R.PAPER)
    page.paste(panel, (0, 95))
    ink = ImageDraw.Draw(page)
    ink.text((24, 18), f'{name} / {pose_name()} / BEARING FAILURE', font=V.R.font(32), fill=V.R.INK)
    ink.text((24, 61), 'Actual constructed shape; arrows show directions, not scaled force magnitudes.', font=V.R.font(21), fill=V.MUTED)
    lines = [f"Rejected {'outer-envelope' if continuous else 'physical'} load #{index}",
             f"Solver: {diagnostics[0]['status']}", f"Tested floor friction: {attempt['friction']:g}", '',
             'Black: gravity on the workpiece', 'Red: head resultant acting on the support',
             '(shown at mean contact location; moment retained in JSON)', '',
             'Support weight is zero in the current model.',
             'The floor may push up; it cannot pull down.']
    if weights is not None:
        force = np.asarray(check['head_resultant_on_support'])[:3]
        lines += ['', f'Resultant on support / mg: {np.array2string(force, precision=4)}',
                  f"Required floor normal / mg: {check['required_floor_normal_for_shown_allocation_mg']:.6g}"]
    maximum = check.get('maximum_support_floor_normal_mg')
    if maximum is not None:
        lines += [f'Maximum possible floor normal / mg: {maximum:.6g}']
    uplift = check.get('uplift_required_even_with_unrestricted_support_footprint')
    lines += ['', 'UPLIFT: even the largest normal is negative.' if uplift else
              'See actual-footprint equilibrium diagnostic in connection.json.',
              'No unanchored footprint passes this numerical necessary test.' if uplift else
              'This panel does not prove all footprints impossible.', '',
              'Geometry insertion passed.' if report['trajectory_verified'] else 'Geometry insertion not verified.',
              'The video is an insertion path, not a stability certificate.']
    for i, line in enumerate(lines):
        ink.text((845, 110+33*i), line, font=V.R.font(21),
                 fill='#a33336' if line.startswith(('UPLIFT', 'Required floor', 'Maximum possible')) else V.R.INK)
    ink.text((24, 940), 'Head forces used by Step3 must also balance the support itself; a floor-demand hull alone is insufficient.', font=V.R.font(23), fill='#a33336')
    ink.text((24, 985), 'This is a numerical static diagnosis with the saved heads, load and friction; no physical fall is simulated.', font=V.R.font(21), fill=V.MUTED)
    page.save(out/'bearing_failure.png')
    np.savez_compressed(out/'bearing_failure.npz', points_m=points, normals=normals, owners=owners,
                        target_wrench=target, coefficients_mg=weights if weights is not None else np.array([]))
    result.update(provenance=dict(inputs=I.hashes(paths+[out/'connection.json', out/'geometry.npz']),
                                 code=I.hashes([Path(__file__), Path(B.__file__), Path(B.Q.__file__), Path(V.__file__)])),
                  artifacts={filename: I.sha256(out/filename) for filename in
                             ('bearing_failure.png', 'bearing_failure.npz')})
    I.save(out/'bearing_failure.json', result)
    print(name, pose_name(), check, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--pose')
    args = parser.parse_args()
    with selected_pose(args.pose):
        run(args.name)
