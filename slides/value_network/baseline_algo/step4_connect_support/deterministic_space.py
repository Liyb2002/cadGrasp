"""Deterministic compact workstation envelope search with frozen Step3 heads.

Tighten rigid-group placements, grow only the necessary spatial envelope, and
carve continuous exits from that envelope. No material-volume minimization.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import time

import manifold3d as md
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog
from shapely.geometry import MultiPoint
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step4_connect_support import boxed_support as F, space_budget as B
from step4_connect_support import build_coupled_saddle as S, process_access as ACCESS
from step2_local_support import geometry as G
from step3_scheculer import contacts as I


def measure(case, mesh, bases, offsets):
    return B.box([t.domain.mesh.vertices for t in case.tasks]+
                 [(mesh.vertices-o)@b.T for b, o in zip(bases, offsets)])


def enlarged(bounds, margin_m):
    lo = np.asarray(bounds['min_m'])-margin_m
    hi = np.asarray(bounds['max_m'])+margin_m
    lo[2] = max(0., lo[2])
    return B.box([np.array([lo, hi])])


def export_exact_obj(mesh, path):
    """OBJ positions round-trip bit-for-bit; fixed decimal places do not.

    Almost coplanar Boolean boundaries require preserving the actual doubles,
    including coordinates close to zero. Python repr uses shortest exact
    floating-point round-trip notation, which OBJ readers accept.
    """
    with Path(path).open('w') as stream:
        stream.write('# CAD support; coordinates in metres; exact float64 round trip\n')
        for vertex in mesh.vertices:
            stream.write('v '+' '.join(repr(float(x)) for x in vertex)+'\n')
        for face in mesh.faces:
            stream.write('f '+' '.join(str(int(x)+1) for x in face)+'\n')
    restored = trimesh.load(path, force='mesh', process=False)
    if not (np.array_equal(restored.vertices, mesh.vertices) and np.array_equal(restored.faces, mesh.faces)):
        raise RuntimeError('Exported OBJ did not preserve the exact accepted geometry')


class Search:
    def __init__(self, group):
        self.group = group
        self.out = group/'step4/data/boxed_support'
        self.out.mkdir(parents=True, exist_ok=True)
        self.case, self.space = F.read_case(group, self.out)
        self.original_box = self.space['object_and_demands_box']
        self.saved_path = group/'step4/data/report.json'
        self.saved = json.loads(self.saved_path.read_text())
        self.case.paths.append(self.saved_path)
        self.before = I.hashes(self.case.paths)
        self.bases = np.asarray(self.saved['placement']['bases'])
        self.offsets = np.asarray(self.saved['placement']['offsets'])
        self.directions = np.asarray(self.saved['placement']['directions'])
        self.root_points = [np.vstack([v for cells in row for v in cells]) for row in self.case.support_seeds]
        self.patch_points = [np.vstack([c['triangles_m'].reshape(-1, 3) for c in row]) for row in self.case.groups]
        self.mandatory = []
        for root, xy in zip(self.root_points, self.case.demands):
            outline = xy[ConvexHull(xy).vertices]
            self.mandatory.append(np.vstack([root, np.c_[outline, np.zeros(len(outline))]]))
        self.raw, self.padded, self.sweep_meshes = [], [], []
        self.sweep_conditioning = []
        self.screen_count, self.boolean_count = 0, 0
        self.attempts, self.optimization = [], []
        self.placed_geometry = {}

    def precompute(self):
        pad = md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
        cache = self.out/'sweeps'; cache.mkdir(exist_ok=True)
        for k, (task, direction) in enumerate(zip(self.case.tasks, self.directions)):
            if abs(direction[2]) > 1e-12 and not getattr(self.case,'object_exit_mode',False):
                raise ValueError('Saved direction must be horizontal')
            numerical_attempts = []
            for fan_in in (64, 8, 16, 2):
                mesh, raw, padded, detail = self.make_sweep(task, direction, pad, cache, fan_in)
                missing = abs(float((raw-padded).volume()))*S.SCALE**3
                numerical_attempts.append(dict(boolean_fan_in=fan_in, exclusion_missing_sweep_m3=missing))
                if missing <= F.VOLUME_TOL_M3:
                    break
            else:
                raise RuntimeError(f'Continuous sweep expansion unresolved: {task.pose}: {numerical_attempts}')
            own = abs(float((raw^self.case.root_solids[k]).volume()))*S.SCALE**3
            if own > F.VOLUME_TOL_M3:
                raise RuntimeError(f'Saved exit blocked by generated owner root: {task.pose}: {own}')
            detail.update(boolean_fan_in=fan_in, numerical_attempts=numerical_attempts)
            self.sweep_conditioning.append(detail)
            self.raw.append(raw); self.padded.append(padded); self.sweep_meshes.append(mesh)

    def make_sweep(self, task, direction, pad, cache, fan_in):
        # Every ordering includes exactly the same original solid and all
        # leading-face prisms. Retry order changes numerical conditioning,
        # not geometry, clearance or the final acceptance threshold.
        key = hashlib.sha256(task.domain.mesh.vertices.tobytes()+task.domain.mesh.faces.tobytes()
            +direction.tobytes()+str(S.SWEEP_LENGTH).encode()+importlib.metadata.version('manifold3d').encode()
            +str(fan_in).encode()
            +I.sha256(Path(S.swept_solid.__code__.co_filename)).encode()).hexdigest()
        path = cache/(key+'.npz')
        if path.exists():
            with np.load(path) as z:
                mesh = trimesh.Trimesh(z['v'], z['f'], process=False)
        else:
            mesh = S.swept_solid(task.domain.mesh, -S.SWEEP_LENGTH*direction, fan_in=fan_in)
            np.savez_compressed(path, v=mesh.vertices, f=mesh.faces)
        raw = S.solid(mesh)
        # Minkowski expansion of an oriented cavity shell in Manifold can
        # leave an inverted residual. Conservatively fill enclosed sweep
        # cavities in the CONSTRUCTION exclusion before expanding it.
        # The original unmodified sweep remains final exit authority.
        pieces = raw.decompose()
        threshold_m3 = 1e-16
        outer = [p for p in pieces if p.volume()*S.SCALE**3 > threshold_m3]
        if not outer:
            raise RuntimeError('No positive outer sweep boundary')
        conditioned = F.union(outer)
        detail = dict(pose=task.pose,
            construction_policy='fill oriented sweep cavities; final validation uses original sweep',
            filled_negative_shell_volume_m3=float(sum(-p.volume()*S.SCALE**3 for p in pieces
                if p.volume()*S.SCALE**3 < -threshold_m3)),
            tiny_shell_absolute_volume_m3=float(sum(abs(p.volume())*S.SCALE**3 for p in pieces
                if abs(p.volume())*S.SCALE**3 <= threshold_m3)),
            tiny_shell_threshold_m3=threshold_m3, original_sweep_modified=False)
        padded = conditioned.minkowski_sum(pad)
        if padded.status() != md.Error.NoError:
            raise RuntimeError('Padded sweep Boolean unresolved')
        return mesh, raw, padded, detail

    def envelope(self, bases, offsets):
        fixture = np.vstack([p@b+o for p, b, o in zip(self.mandatory, bases, offsets)])
        return B.box([t.domain.mesh.vertices for t in self.case.tasks]+
                     [(fixture-o)@b.T for b, o in zip(bases, offsets)])

    def screen(self, bases, offsets):
        self.screen_count += 1
        fixture = [p@b+o for p, b, o in zip(self.mandatory, bases, offsets)]
        patches = [p@b+o for p, b, o in zip(self.patch_points, bases, offsets)]
        for b, o in zip(bases, offsets):
            if min(float(((p-o)@b.T)[:, 2].min()) for p in fixture) < -1e-9:
                return False
            if min(float(((p-o)@b.T)[:, 2].min()) for p in patches) < .002-1e-9:
                return False
        for k, (b, o) in enumerate(zip(bases, offsets)):
            for j, (bj, oj, root) in enumerate(zip(bases, offsets, self.case.root_solids)):
                if k == j:
                    continue
                relative = F.transform(root, bj@b.T, (oj-o)@b.T)
                overlap = abs(float((relative^self.padded[k]).volume()))*S.SCALE**3
                self.boolean_count += 1
                if overlap > F.VOLUME_TOL_M3:
                    return False
        return True

    def tighten(self, bases, offsets):
        """Fixed best-improvement coordinate moves; deterministic tie order."""
        bases, offsets = bases.copy(), offsets.copy()
        states = [(self.envelope(bases, offsets), bases.copy(), offsets.copy())]
        for iteration in range(3):
            proposal = self.joint_translation(bases, offsets)
            if proposal is None:
                break
            old = self.envelope(bases, offsets)['box_volume_cm3']
            accepted = None
            for fraction in (1., .5, .25, .125, .0625):
                trial = offsets+fraction*(proposal-offsets)
                value = self.envelope(bases, trial)['box_volume_cm3']
                if value < old-1e-8 and self.screen(bases, trial):
                    accepted = trial
                    self.optimization.append(dict(method='joint_translation_lp', iteration=iteration,
                        fraction=fraction, before_box_volume_cm3=old, after_box_volume_cm3=value))
                    break
            if accepted is None:
                break
            offsets = accepted
            states.append((self.envelope(bases, offsets), bases.copy(), offsets.copy()))
        for step_mm, yaw_deg in ((12, 12), (6, 6), (3, 3), (1.5, 1.5)):
            for iteration in range(12):
                old = self.envelope(bases, offsets)['box_volume_cm3']
                candidates = []
                for k in range(1, len(bases)):
                    for axis in range(3):
                        for sign in (-1, 1):
                            trial = offsets.copy(); trial[k, axis] += sign*step_mm/1000
                            bounds = self.envelope(bases, trial)
                            if bounds['box_volume_cm3'] < old-1e-8:
                                candidates.append((bounds['box_volume_cm3'], k, axis, sign, bases.copy(), trial))
                    for sign in (-1, 1):
                        rotated = bases.copy(); trial = offsets.copy()
                        center = self.root_points[k].mean(axis=0)
                        pivot = center@bases[k]+offsets[k]
                        rotated[k] = Rotation.from_euler('z', sign*yaw_deg, degrees=True).as_matrix().T@bases[k]
                        trial[k] = pivot-center@rotated[k]
                        bounds = self.envelope(rotated, trial)
                        if bounds['box_volume_cm3'] < old-1e-8:
                            candidates.append((bounds['box_volume_cm3'], k, 3, sign, rotated, trial))
                accepted = None
                for proposal in sorted(candidates, key=lambda p:p[:4]):
                    if self.screen(proposal[4], proposal[5]):
                        accepted = proposal
                        break
                if accepted is None:
                    break
                bases, offsets = accepted[4], accepted[5]
                self.optimization.append(dict(step_mm=step_mm, yaw_deg=yaw_deg, iteration=iteration,
                    before_box_volume_cm3=old, after_box_volume_cm3=accepted[0],
                    moved_pose=self.case.poses[accepted[1]], axis=accepted[2], sign=accepted[3]))
            states.append((self.envelope(bases, offsets), bases.copy(), offsets.copy()))
        return sorted(states, key=lambda p:p[0]['box_volume_cm3'])

    def joint_translation(self, bases, offsets):
        """Move all groups together under box, floor and exit-separation planes.

        Minimizes the first-order change of XYZ box volume. Exact sweep checks
        accept each proposal; no bounding-plane approximation certifies a solid.
        """
        count = len(bases); dimension = 3*count+6
        rows, rhs = [], []
        def difference(j, k, vector):
            row = np.zeros(dimension)
            row[3*j:3*j+3] += vector
            row[3*k:3*k+3] -= vector
            return row
        for k, bk in enumerate(bases):
            for j, bj in enumerate(bases):
                native = self.mandatory[j]@bj@bk.T
                for axis, vector in enumerate(bk):
                    row = -difference(j, k, vector); row[3*count+axis] = 1
                    rows.append(row); rhs.append(float(native[:, axis].min()))
                    row = difference(j, k, vector); row[3*count+3+axis] = -1
                    rows.append(row); rhs.append(float(-native[:, axis].max()))
                if j == k:
                    continue
                rows.append(-difference(j, k, bk[2])); rhs.append(float(native[:, 2].min())-1e-6)
                patch = self.patch_points[j]@bj@bk.T
                rows.append(-difference(j, k, bk[2])); rhs.append(float(patch[:, 2].min())-.002)
                planes = ConvexHull(self.sweep_meshes[k].vertices).equations
                # A root can occupy a concavity outside the exact sweep but
                # inside its hull. Such roots have no convex separator here;
                # the subsequent exact Boolean screen still checks them.
                for cells in self.case.support_seeds[j]:
                    cloud = np.vstack(cells)@bj@bk.T
                    installed = cloud+(offsets[j]-offsets[k])@bk.T
                    margins = (installed@planes[:, :3].T+planes[:, 3]).min(axis=0)
                    padded = S.RELIEF*np.abs(planes[:, :3]).sum(axis=1)
                    index = int(np.argmax(margins-padded))
                    if margins[index] <= padded[index]+1e-9:
                        continue
                    normal, intercept = planes[index, :3], planes[index, 3]
                    rows.append(-difference(j, k, normal@bk))
                    rhs.append(float((cloud@normal).min()+intercept-padded[index]-1e-7))
        current = self.envelope(bases, offsets)
        extents = np.asarray(current['extents_mm'])/1000
        weights = np.prod(extents)/extents
        weights /= weights.max()
        objective = np.r_[np.zeros(3*count), -weights, weights]
        bound = [(None, None)]*(3*count)
        bound[:3] = [(float(v), float(v)) for v in offsets[0]]
        objects = B.box([t.domain.mesh.vertices for t in self.case.tasks])
        bound += [(None, v) for v in objects['min_m']]+[(v, None) for v in objects['max_m']]
        result = linprog(objective, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bound,
            method='highs', options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
        return result.x[:3*count].reshape(count, 3) if result.success else None

    def installed_geometry(self, bases, offsets):
        key = bases.tobytes()+offsets.tobytes()
        if key in self.placed_geometry:
            return self.placed_geometry[key]
        roots = [F.union([S.solid(G.hull_mesh(v@b+o)) for cells in row for v in cells])
                 for row, b, o in zip(self.case.support_seeds, bases, offsets)]
        sweeps = [trimesh.Trimesh(m.vertices@b+o, m.faces, process=False)
                  for m, b, o in zip(self.sweep_meshes, bases, offsets)]
        pad = md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
        exclusions = []
        for swept in sweeps:
            raw = S.solid(swept)
            outer = [part for part in raw.decompose() if part.volume()*S.SCALE**3 > 1e-16]
            expanded = F.union(outer).minkowski_sum(pad)
            exclusions.append(expanded)
        forbidden = F.union(exclusions)
        result = roots, sweeps, forbidden
        self.placed_geometry[key] = result
        return result

    def construct(self, bases, offsets, bounds):
        row = dict(box=bounds, passed=False, constructed=False)
        self.attempts.append(row)
        self.case.output = self.out
        try:
            guard = ACCESS.Guard(self.case, dict(bases=bases, offsets=offsets))
        except ACCESS.AccessRejected as error:
            row.update(reason='working_surface_preflight', check=error.access_report)
            return None
        allowed = F.bounded_space(bounds, bases, offsets)
        if allowed.is_empty():
            row['reason'] = 'empty_allowed_space'
            return None
        roots, sweeps, forbidden = self.installed_geometry(bases, offsets)
        full, detail = F.maximal_component(allowed, forbidden, roots)
        row.update(detail)
        if full is None:
            return None
        mesh = S.unpack(full)
        row.update(constructed=True, material_volume_cm3=float(mesh.volume*1e6))
        if not (mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0):
            row['reason'] = 'invalid_topology'
            return None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                checks, certificate = S.verify(self.case.tasks, self.case.groups, self.case.support_seeds,
                    self.directions, bases, offsets, mesh, sweeps, check_equilibrium=False)
        except RuntimeError as error:
            if not hasattr(error, 'checks'):
                raise
            row.update(reason='full_geometry_failed', checks=error.checks)
            print('GEOMETRY RESIDUALS', [(c['pose'], c['withdrawal']['complete_sweep_overlap_m3'],
                c['maximum_missing_head_cell_volume_m3']) for c in error.checks], flush=True)
            return None
        coverage = []
        for check, demand in zip(checks, self.case.demands):
            actual = MultiPoint(check['actual_ground_hull_xy_m']).convex_hull
            loss = float(MultiPoint(demand).convex_hull.difference(actual.buffer(1e-10)).area)
            coverage.append(dict(pose=check['pose'], uncovered_area_m2=loss, passed=actual.area > 0 and loss <= 1e-12))
        working = guard.verify(mesh)
        boxes = F.box_checks(mesh, bounds, bases, offsets, self.case.poses)
        passed = working['passed'] and all(r['passed'] for r in coverage) and all(r['all_inside'] for r in boxes)
        row.update(passed=passed, reason='geometry_passed' if passed else 'ground_work_or_box_failed',
            checks=checks, ground_coverage=coverage, working_surface=working, box_checks=boxes)
        if passed:
            return mesh, certificate, row
        return None

    def run(self):
        began = time.monotonic()
        I.save(self.out/'report.json', dict(complete=False, constructed=False, passed=False,
            schema='deterministic_workstation_space_v1', status='searching', object=self.case.name, poses=self.case.poses))
        self.precompute()
        self.initial_screen = self.screen(self.bases, self.offsets)
        states = self.tighten(self.bases, self.offsets) if self.initial_screen else [
            (self.envelope(self.bases, self.offsets), self.bases, self.offsets)]
        winner = None
        for lower, bases, offsets in states:
            for margin_mm in (1, 2, 4, 8, 16, 32):
                bounds = enlarged(lower, margin_mm/1000)
                if winner is not None and bounds['box_volume_cm3'] >= winner[0]:
                    break
                print('SPACE TRY', self.group.name, round(bounds['box_volume_cm3'], 1), 'cm3', 'margin', margin_mm, flush=True)
                result = self.construct(bases, offsets, bounds)
                if result is None:
                    print('SPACE REJECT', self.group.name, self.attempts[-1]['reason'], flush=True)
                if result is not None:
                    mesh, certificate, check = result
                    actual = measure(self.case, mesh, bases, offsets)
                    winner = actual['box_volume_cm3'], mesh, certificate, check, bases, offsets, actual
                    # Refine the successful box with deterministic bisection.
                    low_mm, high_mm = 0., float(margin_mm)
                    for _ in range(3):
                        middle = (low_mm+high_mm)/2
                        refined = enlarged(lower, middle/1000)
                        trial = self.construct(bases, offsets, refined)
                        if trial is None:
                            low_mm = middle
                        else:
                            high_mm = middle
                            m, c, chk = trial
                            actual = measure(self.case, m, bases, offsets)
                            if actual['box_volume_cm3'] < winner[0]:
                                winner = actual['box_volume_cm3'], m, c, chk, bases, offsets, actual
                    break
            if winner is not None:
                break
        old_mesh = trimesh.load(self.group/'step4/shape.obj', force='mesh', process=False)
        old_box = measure(self.case, old_mesh, self.bases, self.offsets)
        report = dict(complete=True, schema='deterministic_workstation_space_v1', object=self.case.name,
            poses=self.case.poses, constructed=winner is not None, passed=winner is not None,
            status='compact_space_geometry_passed' if winner is not None else 'deterministic_space_search_failed',
            step3_passed=self.case.schedule['passed'], step3_covered_counts=self.case.schedule['covered_counts'],
            step3_verdict_modified=False, passed_scope='step4_geometry_only',
            force_torque_authority='step3', objective='Minimum aggregate workstation XYZ bounding-box volume',
            material_volume_objective=False, surface_area_objective=False, deterministic=True, random_sampling=False,
            global_optimality_claim=False, general_infeasibility_claim=False,
            ideal_box=self.original_box, previous_box=old_box, initial_screen_passed=self.initial_screen,
            attempts=self.attempts, optimization=self.optimization,
            sweep_conditioning=self.sweep_conditioning,
            screen_count=self.screen_count, screen_boolean_count=self.boolean_count,
            search_seconds=time.monotonic()-began, artifacts={})
        if winner is not None:
            _, mesh, certificate, check, bases, offsets, actual = winner
            export_exact_obj(mesh, self.out/'shape.obj')
            np.savez_compressed(self.out/'geometry_certificate.npz', **certificate)
            self.case.preview_directions = self.directions
            F.draw(self.out/'overview.png', self.case, mesh, bases, offsets)
            report.update(result=check, space_budget=actual, volume_cm3=float(mesh.volume*1e6),
                export_roundtrip_bit_exact=True,
                placement=dict(bases=bases.tolist(), offsets=offsets.tolist(), directions=self.directions.tolist()),
                box_volume_reduction_percent=100*(1-actual['box_volume_cm3']/old_box['box_volume_cm3']),
                xy_area_reduction_percent=100*(1-actual['xy_area_cm2']/old_box['xy_area_cm2']),
                artifacts={p:I.sha256(self.out/p) for p in ('shape.obj','overview.png','geometry_certificate.npz')})
        report['provenance'] = dict(manifold3d_version=importlib.metadata.version('manifold3d'),
            inputs=self.before, code=I.hashes([Path(__file__), Path(F.__file__), Path(B.__file__),
            Path(S.__file__), Path(S.swept_solid.__code__.co_filename)]+ACCESS.sources()))
        if I.hashes(self.case.paths) != self.before:
            raise RuntimeError('Frozen upstream inputs changed')
        I.save(self.out/'report.json', report)
        print('SPACE RESULT', self.group.name, report['status'], 'seconds', round(report['search_seconds'], 2), flush=True)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    parser.add_argument('--groups', nargs='+', default=['pose3+6'])
    args = parser.parse_args()
    for group in args.groups:
        Search(I.OUTPUTS/args.object/group).run()
