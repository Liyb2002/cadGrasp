"""Step4.2: retain a maximal connected free solid inside the Step4.1 box.

Fixed box, finite placements/directions, no material-volume or surface-area cost.
Saved public Step4 models are preserved; new candidates live in diagnostic data.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import manifold3d as md
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
from shapely.geometry import MultiPoint
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
from step4_connect_support.baseline_current import space_budget as B, build_coupled_saddle as S
from step4_connect_support.baseline_current import head_registration as H, zero_thickness_heads as Z
from step4_connect_support.baseline_current import process_access as ACCESS, working_surface as SURFACE
from step2_local_support import geometry as G, withdrawal as W

VOLUME_TOL_M3 = 8e-14


def union(parts):
    return md.Manifold.batch_boolean(parts, md.OpType.Add)


def transform(value, basis, offset):
    return value.transform(np.c_[np.asarray(basis).T, np.asarray(offset)/S.SCALE])


def bounded_space(bounds, bases, offsets):
    """Intersect all installed 3D box constraints and all floor halfspaces."""
    bases, offsets = np.asarray(bases, float), np.asarray(offsets, float)
    if bases.ndim != 3 or bases.shape[1:] != (3, 3) or offsets.shape != (len(bases), 3) or not len(bases):
        raise ValueError('Every pose needs one basis and offset')
    if (not np.isfinite(bases).all() or not np.isfinite(offsets).all()
            or not np.allclose(bases@bases.transpose(0, 2, 1), np.eye(3), atol=1e-12, rtol=0)
            or not np.allclose(np.linalg.det(bases), 1, atol=1e-12, rtol=0)):
        raise ValueError('Expected finite proper rigid placements')
    lo, hi = np.asarray(bounds['min_m']), np.asarray(bounds['max_m'])
    if np.any(hi <= lo):
        raise ValueError('A solid budget needs positive XYZ extents')
    # Build the convex intersection once from plane inequalities. Repeated
    # Boolean intersections of near-coplanar rotated boxes produced nonfinite
    # vertices in Manifold; the LP also distinguishes an empty intersection
    # from an unresolved geometry operation.
    planes = []
    for basis, offset in zip(bases, offsets):
        for axis, normal in enumerate(basis):
            planes.append(np.r_[normal, -hi[axis]-normal@offset])
            planes.append(np.r_[-normal, lo[axis]+normal@offset])
        planes.append(np.r_[-basis[2], basis[2]@offset])
    planes = np.asarray(planes)
    interior = linprog([0, 0, 0, -1], A_ub=np.c_[planes[:, :3], np.ones(len(planes))],
        b_ub=-planes[:, 3], bounds=[(None, None)]*3+[(0, None)], method='highs',
        options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    if interior.status == 2 or (interior.success and interior.x[3] <= 1e-10):
        return md.Manifold()
    if not interior.success:
        raise RuntimeError(f'Allowed-space interior LP unresolved: {interior.message}')
    vertices = HalfspaceIntersection(planes, interior.x[:3]).intersections
    if np.max(vertices@planes[:, :3].T+planes[:, 3]) > 1e-9:
        raise RuntimeError('Allowed-space vertex violates a box/floor plane')
    return S.solid(G.hull_mesh(vertices))


def maximal_component(allowed, forbidden, roots):
    """Keep all allowed material; select a component containing every root.

    Roots restore exact contact through conservative non-contact padding.
    Their raw sweep collisions are independently checked by the caller.
    """
    missing = [abs(float((root-allowed).volume()))*S.SCALE**3 for root in roots]
    if max(missing, default=0) > VOLUME_TOL_M3:
        return None, dict(reason='mandatory_roots_outside_box_or_floor', missing_root_volumes_m3=missing)
    full = (allowed-forbidden)+union(roots)
    if full.status() != md.Error.NoError:
        raise RuntimeError(f'Free-space Boolean unresolved: {full.status()}')
    boundaries = full.decompose()
    # Decompose splits SURFACE components, including inward cavity shells.
    # Selecting a positive outer shell alone would silently fill its cavities.
    # Reassemble each material component with all enclosed negative shells.
    tiny_m3 = 1e-16
    parts = [p for p in boundaries if p.volume()*S.SCALE**3 > tiny_m3]
    cavities = []
    for shell in boundaries:
        if shell.volume()*S.SCALE**3 < -tiny_m3:
            mesh = S.unpack(shell)
            mesh.invert()
            cavities.append(S.solid(mesh))
    for outer in parts:
        enclosed = [c for c in cavities if abs(float((c-outer).volume()))*S.SCALE**3 <= VOLUME_TOL_M3]
        part = outer-union(enclosed) if enclosed else outer
        loss = [abs(float((root-part).volume()))*S.SCALE**3 for root in roots]
        if max(loss, default=0) <= VOLUME_TOL_M3:
            return part, dict(reason='all_roots_in_one_free_component',
                free_component_count=len(parts), boundary_component_count=len(boundaries),
                preserved_cavity_count=len(enclosed), missing_root_volumes_m3=loss)
    return None, dict(reason='roots_in_different_free_components', free_component_count=len(parts),
        conservative_padding_m=S.RELIEF, general_infeasibility_claim=False)


def read_case(group, out):
    poses, objects, faces, demands, paths = B.load_group(group)
    tasks, contacts, catalogues, menus, reports = [], [], [], [], []
    for pose in poses:
        folder = group/'step3_scheculer/independent_poses_floor2mm'/pose
        report_path = folder/'step3_scheculer/schedule.json'
        report = I.check_report(report_path)
        if report.get('head_model') != Z.MODEL or report.get('floor_poses') != poses:
            raise ValueError('Expected this group\'s exact surface-head inputs')
        task = read_task(group.parent.name, pose, folder=folder/'step_1_needs')
        index = poses.index(pose)
        np.testing.assert_allclose(task.domain.mesh.vertices, objects[index], atol=1e-12, rtol=0)
        contact_path = report_path.parent/f'final_contacts_{pose}.npz'
        metadata_path = folder/'step2_local_support'/f'candidates_{pose}.json'
        catalogue = json.loads(metadata_path.read_text())['direction_catalogue']['vectors']
        tasks.append(task); contacts.append(I.read_contacts(contact_path)); reports.append(report)
        catalogues.append(np.asarray(catalogue))
        menus.append(report['result']['geometry']['per_pose'][0]['common_direction_ids'])
        paths.extend([report_path, contact_path, metadata_path]+list(task.inputs))
    case = SimpleNamespace(name=group.parent.name, poses=poses, pair=group, output=out,
        tasks=tasks, groups=contacts, heads=[[list(c['triangles_m']) for c in g] for g in contacts],
        schedule=dict(passed=all(r['result']['passed'] for r in reports),
            covered_counts=[r['result']['covered_counts'][0] for r in reports]), paths=paths,
        catalogues=catalogues, menus=menus, demands=[d[:, :2] for d in demands])
    Z.prepare(case, S.RELIEF)
    case.root_solids = [union([S.solid(G.hull_mesh(v)) for cells in row for v in cells])
                        for row in case.support_seeds]
    return case, B.analyze(objects, demands, poses)


def placements(case):
    saved = case.pair/'step4/data/report.json'
    if saved.exists():
        report = json.loads(saved.read_text())
        if report['poses'] != case.poses:
            raise ValueError('Saved placement pose order differs')
        case.paths.append(saved)
        yield 'saved_seating', np.asarray(report['placement']['bases']), np.asarray(report['placement']['offsets'])
    bases, offsets = H.fixed_placements(case.tasks)
    yield 'common_object_registration', bases, offsets


def box_checks(mesh, bounds, bases, offsets, poses):
    return [dict(pose=p, **B.inclusion((mesh.vertices-o)@b.T, bounds))
            for p, b, o in zip(poses, bases, offsets)]


def draw(path, case, mesh, bases, offsets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(6*len(case.poses), 6), facecolor='white')
    installed = [(mesh.vertices-o)@b.T for b, o in zip(bases, offsets)]
    cloud = np.vstack(installed+[t.domain.mesh.vertices for t in case.tasks])
    focus = (cloud.min(0)+cloud.max(0))/2; radius = np.ptp(cloud, axis=0).max()*.6
    for k, (task, vertices) in enumerate(zip(case.tasks, installed), 1):
        ax = fig.add_subplot(1, len(case.poses), k, projection='3d')
        ax.add_collection3d(Poly3DCollection(vertices[mesh.faces], facecolors='#b4b9bd', edgecolors='#b4b9bd', linewidths=0,
            shade=True, lightsource=LightSource(azdeg=315, altdeg=45)))
        ax.add_collection3d(Poly3DCollection(task.domain.mesh.triangles, facecolors='#77acd0', edgecolors='none', alpha=.22))
        for index, contact in enumerate(case.groups[k-1]):
            ax.add_collection3d(Poly3DCollection(contact['triangles_m'],
                facecolors=plt.get_cmap('tab10')(index), edgecolors='none'))
        ax.set(xlim=(focus[0]-radius, focus[0]+radius), ylim=(focus[1]-radius, focus[1]+radius),
               zlim=(focus[2]-radius, focus[2]+radius))
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        direction = -np.asarray(case.preview_directions[k-1]) if hasattr(case, 'preview_directions') else np.array([.57, -.82, 0])
        ax.view_init(elev=25, azim=float(np.rad2deg(np.arctan2(direction[1], direction[0]))))
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run(group, max_direction_sets=16):
    out = group/'step4/data/boxed_support'
    out.mkdir(parents=True, exist_ok=True)
    # Only replace this experimental entry's own artifacts, never public Step4.
    for filename in ('shape.obj', 'overview.png', 'geometry_certificate.npz'):
        (out/filename).unlink(missing_ok=True)
    I.save(out/'report.json', dict(complete=False, status='preparing', constructed=False, passed=False))
    case, space = read_case(group, out)
    # Complete the placement iterator before hashing all dependencies.
    layouts = list(placements(case))
    before = I.hashes(case.paths)
    report = dict(complete=False, schema='step4_2_fixed_box_free_solid_v1', object=case.name,
        poses=case.poses, constructed=False, passed=False, attempts=[],
        space_budget=space['object_and_demands_box'], box_expanded_beyond_step4_1=False,
        material_volume_objective=False, surface_area_objective=False,
        passed_scope='step4_geometry_only', force_torque_authority='step3',
        step3_passed=case.schedule['passed'], step3_covered_counts=case.schedule['covered_counts'],
        step3_verdict_modified=False, max_direction_sets_per_placement=max_direction_sets,
        general_infeasibility_claim=False)
    bounds = report['space_budget']
    sweep_cache = {}
    for name, bases, offsets in layouts:
        placement = dict(bases=bases.tolist(), offsets=offsets.tolist())
        allowed = bounded_space(bounds, bases, offsets)
        if allowed.status() != md.Error.NoError or allowed.is_empty():
            report['attempts'].append(dict(layout=name,
                reason='empty_full_dimensional_allowed_space' if allowed.status() == md.Error.NoError else 'allowed_space_boolean_unresolved',
                boolean_status=str(allowed.status()), general_infeasibility_claim=False))
            continue
        roots = [transform(r, b, o) for r, b, o in zip(case.root_solids, bases, offsets)]
        losses = [abs(float((r-allowed).volume()))*S.SCALE**3 for r in roots]
        if max(losses, default=0) > VOLUME_TOL_M3:
            report['attempts'].append(dict(layout=name, placement=placement,
                reason='mandatory_roots_outside_box_or_floor', missing_root_volumes_m3=losses,
                root_box_checks=box_checks(S.unpack(union(roots)), bounds, bases, offsets, case.poses)))
            continue
        try:
            guard = ACCESS.Guard(case, placement)
        except ACCESS.AccessRejected as error:
            report['attempts'].append(dict(layout=name, reason='contact_or_root_on_working_face', check=error.access_report))
            continue
        # Bound finite search; no claim that all possible exit directions were tested.
        menus = [[i for i in menu if abs(cat[i, 2]) <= 1e-12]
                 for menu, cat in zip(case.menus, case.catalogues)]
        if any(not menu for menu in menus):
            report['attempts'].append(dict(layout=name, reason='no_certified_horizontal_direction',
                menu_sizes=[len(menu) for menu in menus]))
            continue
        for ids in itertools.islice(itertools.product(*menus), max_direction_sets):
            row = dict(layout=name, direction_ids=list(ids), constructed=False)
            report['attempts'].append(row)
            directions = np.array([cat[i] for cat, i in zip(case.catalogues, ids)])
            all_cells = [v@b+o for seed, b, o in zip(case.support_seeds, bases, offsets)
                         for cells in seed for v in cells]
            checks = [W.Analyzer(t.domain.mesh, .01*t.domain.mesh.extents.max(),
                dict(vectors=[d.tolist()])).test([SimpleNamespace(vertices=(v-o)@b.T) for v in all_cells], d)
                for t, b, o, d in zip(case.tasks, bases, offsets, directions)]
            if not all(c['clear'] for c in checks):
                row.update(reason='whole_root_exit_blocked', root_exit_checks=checks)
                continue
            sweeps = []
            for k, (task, ident, direction, basis, offset) in enumerate(zip(case.tasks, ids, directions, bases, offsets)):
                key = (k, ident)
                if key not in sweep_cache:
                    sweep_cache[key] = S.swept_solid(task.domain.mesh, -S.SWEEP_LENGTH*direction)
                raw = sweep_cache[key]
                sweeps.append(trimesh.Trimesh(raw.vertices@basis+offset, raw.faces, process=False))
            pad = md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
            forbidden = union([S.solid(sweep).minkowski_sum(pad) for sweep in sweeps])
            try:
                full, detail = maximal_component(allowed, forbidden, roots)
            except RuntimeError as error:
                # A failed padded Boolean is not a geometry impossibility.
                # Try the unpadded continuous sweeps and retain the original
                # error; full working-surface and exit checks remain mandatory.
                row.update(padded_boolean_error=str(error), fallback_noncontact_relief_m=0.,
                    manufacturing_clearance_certified=False)
                forbidden = union([S.solid(sweep) for sweep in sweeps])
                try:
                    full, detail = maximal_component(allowed, forbidden, roots)
                except RuntimeError as fallback_error:
                    row.update(reason='geometry_boolean_unresolved', error=str(fallback_error),
                        general_infeasibility_claim=False)
                    continue
            row.update(detail)
            if full is None:
                continue
            mesh = S.unpack(full)
            if not (mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0):
                row['reason'] = 'invalid_boolean_mesh'
                continue
            row.update(constructed=True, volume_cm3=float(mesh.volume*1e6))
            try:
                verification, certificate = S.verify(case.tasks, case.groups, case.support_seeds,
                    directions, bases, offsets, mesh, sweeps, check_equilibrium=False)
            except RuntimeError as error:
                if not hasattr(error, 'checks'):
                    raise
                row.update(reason='full_geometry_failed', checks=error.checks)
                continue
            coverage = []
            for check, demand in zip(verification, case.demands):
                hull = MultiPoint(check['actual_ground_hull_xy_m']).convex_hull
                loss = float(MultiPoint(demand).convex_hull.difference(hull.buffer(1e-10)).area)
                coverage.append(dict(pose=check['pose'], uncovered_area_m2=loss, passed=hull.area > 0 and loss <= 1e-12))
            work = guard.verify(mesh)
            boxes = box_checks(mesh, bounds, bases, offsets, case.poses)
            passed = all(c['passed'] for c in coverage) and work['passed'] and all(c['all_inside'] for c in boxes)
            row.update(passed=passed, checks=verification, ground_coverage=coverage,
                working_surface=work, box_checks=boxes,
                reason='fixed_box_geometry_passed' if passed else 'ground_work_surface_or_box_failed')
            if not report['constructed'] or passed:
                mesh.export(out/'shape.obj')
                np.savez_compressed(out/'geometry_certificate.npz', **certificate)
                draw(out/'overview.png', case, mesh, bases, offsets)
                report.update(constructed=True, passed=passed, result=row,
                    placement=dict(**placement, directions=directions.tolist(), direction_ids=list(ids)),
                    volume_cm3=row['volume_cm3'])
            if passed:
                break
        if report['passed']:
            break
    report.update(complete=True, status='fixed_box_geometry_passed' if report['passed'] else 'finite_fixed_box_search_failed',
        provenance=dict(inputs=before, code=I.hashes([Path(__file__), Path(B.__file__), Path(S.__file__),
            Path(S.swept_solid.__code__.co_filename), Path(Z.__file__), Path(H.__file__),
            Path(G.__file__), Path(W.__file__), Path(SURFACE.__file__)]+ACCESS.sources())),
        artifacts={p.name:I.sha256(p) for p in out.iterdir() if p.name in ('shape.obj', 'overview.png', 'geometry_certificate.npz')})
    if I.hashes(case.paths) != before:
        raise RuntimeError('Boxed construction changed upstream inputs')
    I.save(out/'report.json', report)
    print('BOXED SUPPORT', group.name, report['status'], 'constructed', report['constructed'], flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    parser.add_argument('--groups', nargs='+', default=['pose3+6'])
    parser.add_argument('--max-direction-sets', type=int, default=16)
    args = parser.parse_args()
    if args.max_direction_sets < 1:
        parser.error('Positive direction-set budget required')
    for group in args.groups:
        target = I.OUTPUTS/args.object/group
        try:
            run(target, args.max_direction_sets)
        except (RuntimeError, ValueError) as error:
            I.save(target/'step4/data/boxed_support/report.json', dict(complete=True,
                schema='step4_2_fixed_box_free_solid_v1', status='pipeline_error',
                constructed=False, passed=False, error=f'{type(error).__name__}: {error}',
                general_infeasibility_claim=False))
            print('BOXED SUPPORT ERROR', group, str(error), flush=True)
