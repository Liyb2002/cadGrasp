"""Score each Step 2 contact against the same stored Step 1 paired samples.

One nonnegative reaction vector must satisfy all six equilibrium equations.
Build a supply cone once per circle and classify its demands in NumPy batches.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent
sys.path.insert(0, str(BASELINE))
from step2_local_support import circles as P
from step3_scheculer.stage_imports import load_stage
from step3_scheculer import floor_support as F
from step3_scheculer import passive_support as U
W = load_stage('score', 'wrench_cone')
from step1.needs import OUTPUTS, OBJECTS, ROOT, demand, sha256
from step1.cases import pose_name

FEASIBILITY_TOL = 2e-9  # force / mg; moment / (mg * largest bbox side)


def context(name):
    domain, data, report = P.read(name)
    source = ROOT/domain.data['provenance']['setup_snapshot']
    assert sha256(source) == domain.data['provenance']['setup_snapshot_sha256']
    with np.load(source) as z:
        floor = z['floor_contact_m'].copy()
    scale = np.r_[np.ones(3), np.ones(3)/domain.mesh.extents.max()]
    assert domain.k == .5 and np.array_equal(domain.gravity, [0, 0, -1])
    return domain, data, report, SimpleNamespace(scale=scale), floor


def read_samples(name, domain):
    folder = OUTPUTS/name/pose_name()/'step_1_needs'
    samples = json.loads((folder/'samples.json').read_text())
    assert samples['object'] == name
    assert samples['provenance']['physical_domain_sha256'] == sha256(folder/'needs.json')
    assert samples['pair_order'] == ['Fx', 'Fy', 'Fz', 'tau_x', 'tau_y', 'tau_z']
    assert samples['magnitude_range_mg'] == [0., domain.k]
    assert samples['moment_origin_m'] == domain.com.tolist()
    count = samples['count']
    assert count > 0 and samples['weight_per_sample'] == 1./count
    targets = np.asarray(samples['need_wrench'], float)
    points = np.asarray(samples['pt_m'], float)
    forces = np.asarray(samples['force_push_mg'], float)
    assert targets.shape == (count, 6) and points.shape == forces.shape == (count, 3)
    assert np.isfinite(targets).all()
    np.testing.assert_allclose(targets, demand(points, forces, domain.com, domain.gravity), atol=1e-13)
    return samples, targets


def columns(domain, data, index, floor, scale):
    assert data.valid[index], 'Rejected Step 2 geometry cannot contribute reactions'
    a, b = data.offsets[index:index+2]
    q = data.triangles[a:b].reshape(-1, 3)
    n = np.repeat(-domain.mesh.face_normals[data.source_faces[a:b]], 3, axis=0)
    full = np.vstack([U.heads(np.c_[n, np.cross(q-domain.com, n)], scale),
                      U.floor(F.columns(floor, domain.com), scale)])
    ids = np.unique(np.round(full, 13), axis=0, return_index=True)[1]
    return np.ascontiguousarray(full[np.sort(ids)])


def sample_separator(full, targets):
    """Optional fast rejection of this finite batch, never a continuum claim.

    Add the worst stored target to a small separating LP until its plane rejects
    every target. If that fails, the complete cone classifier handles the batch.
    """
    targets = U.target(targets, full.shape[1])
    dimension = full.shape[1]
    if not len(targets):
        return None
    active = [targets.mean(axis=0)]
    for _ in range(16):
        constraints = np.vstack([np.c_[full, np.zeros(len(full))],
                                 np.c_[-np.asarray(active), np.ones(len(active))]])
        result = linprog(np.r_[np.zeros(dimension), -1.], A_ub=constraints,
                         b_ub=np.zeros(len(constraints)),
                         bounds=[(-1, 1)]*dimension+[(0, None)], method='highs')
        if not result.success or result.x[-1] < 1e-8:
            return None
        y = result.x[:dimension]/np.linalg.norm(result.x[:dimension])
        margins = targets@y
        if margins.min() > 1e-8 and (full@y).max() <= 1e-12:
            return dict(vector=y.tolist(), minimum_over_samples=float(margins.min()),
                        maximum_over_supply_generators=float((full@y).max()))
        active.append(targets[margins.argmin()])
    return None


def halfspace_membership(H, targets, tolerance=FEASIBILITY_TOL):
    """All inequalities must hold on the SAME six-dimensional target."""
    targets = U.target(targets, H.shape[1])
    accepted = np.ones(len(targets), bool)
    if not len(H):
        return accepted
    # Test discriminating facets first, then only multiply surviving targets.
    pilot = targets[np.linspace(0, len(targets)-1, min(128, len(targets)), dtype=int)]
    order = np.argsort(-np.count_nonzero(H@pilot.T > tolerance, axis=1), kind='stable')
    for start in range(0, len(order), 64):
        ids = np.flatnonzero(accepted)
        if not len(ids):
            break
        scores = targets[ids]@H[order[start:start+64]].T
        accepted[ids] = np.all(scores <= tolerance, axis=1)
    return accepted


def classify(full, targets):
    """Finite-sample feasibility, including cones of dimension below six."""
    targets = U.target(targets, full.shape[1])
    _, singular, vt = np.linalg.svd(full, full_matrices=False)
    rank = int((singular > singular[0]*1e-10).sum())
    info = dict(supply_rank=rank, singular_values=singular.tolist())
    if rank < full.shape[1]:
        basis = vt[:rank].T
        error = np.max(np.abs(targets-targets@basis@basis.T), axis=1)
        possible = np.flatnonzero(error <= FEASIBILITY_TOL)
        accepted = np.zeros(len(targets), bool)
        for i in possible:
            accepted[i] = W.solve(full, targets[i]) is not None
        info.update(method='span_then_joint_lp', samples_in_supply_span=len(possible))
        return accepted, info, None
    separator = sample_separator(full, targets)
    if separator:
        info.update(method='sample_separator', separator=separator)
        return np.zeros(len(targets), bool), info, None
    H = W.cone(full)
    accepted = halfspace_membership(H, targets)
    info.update(method='joint_cone_halfspaces', facets=len(H))
    return accepted, info, H


def verify_classification(full, targets, accepted):
    """Independent nonnegative LPs for both verdicts and generated reactions."""
    targets = U.target(targets, full.shape[1])
    ids = list(np.linspace(0, len(targets)-1, 4, dtype=int))
    for verdict in (True, False):
        group = np.flatnonzero(accepted == verdict)
        if len(group):
            ids.extend(group[np.linspace(0, len(group)-1, min(3, len(group)), dtype=int)])
    records = []
    for i in np.unique(ids):
        witness = W.solve(full, targets[i])
        if (witness is not None) != bool(accepted[i]):
            raise RuntimeError(f'Sample {i}: batch cone and independent joint LP disagree')
        passive = None
        if witness is not None and full.shape[1] == 7 and targets[i, -1] == 0:
            ids = np.asarray(witness['indices'], int)
            exact = witness.get('precise')
            weights = np.array(exact['coefficients'] if exact else witness['coefficients'], float)
            used = np.asarray(exact['indices'] if exact else ids, int)
            slack = np.all(full[used, :6] == 0, axis=1) & (full[used, 6] == -1)
            head_z = float(weights[~slack] @ full[used[~slack], 6])
            normal = float(weights[slack].sum())
            if head_z < -2e-8 or abs(head_z-normal) > 2e-8:
                raise RuntimeError('Accepted sample violates the shared no-uplift equation')
            passive = dict(head_force_on_workpiece_z_mg=head_z,
                workpiece_force_on_support_z_mg=-head_z, support_floor_normal_mg=normal,
                passed=True, original_workpiece_floor_force_excluded=True)
        records.append(dict(sample_index=int(i), feasible=bool(accepted[i]), witness=witness,
                            passive_support=passive))
    # Known feasible combinations also test cones with zero sampled coverage.
    weights = 1+np.cos(np.outer(np.arange(1, 3), np.arange(1, len(full)+1)))
    generated = (weights@full)/weights.sum(axis=1)[:, None]
    for target in generated:
        if W.solve(full, target) is None:
            raise RuntimeError('A nonnegative generated wrench failed its joint LP')
    return dict(sample_checks=records, generated_wrench_checks=len(generated), disagreements=0)


# The same score operation is used in every scheduler round, including round 1.
from step3_scheculer import contacts as I
J = load_stage('score', 'joint_samples')
OUTPUT_NAME = 'step3.1_score_candidate'


def code_hashes():
    return I.hashes([Path(__file__), Path(W.__file__), Path(J.__file__), Path(I.__file__), Path(F.__file__), Path(U.__file__),
                     BASELINE/'step3_scheculer/stage_imports.py'])


class Problem:
    def __init__(self, name):
        self.name = name
        self.domain, self.data, self.geometry, scaling, self.floor = context(name)
        self.scale = scaling.scale
        self.samples, targets = read_samples(name, self.domain)
        self.targets = np.ascontiguousarray(targets*self.scale)
        self.random_sample_count = len(self.targets)
        self.counterexamples = []
        root = OUTPUTS/name/pose_name()
        self.inputs = [root/'step_1_needs/needs.json', root/'step_1_needs/samples.json',
                       root/'step2_local_support/circles.json', root/'step2_local_support/circles.npz',
                       ROOT/self.domain.data['provenance']['setup_snapshot']]
        self.floor_columns = U.floor(F.columns(self.floor, self.domain.com), self.scale)

    def candidate(self, index):
        assert self.data.valid[index]
        a, b = self.data.offsets[index:index+2]
        return dict(triangles_m=self.data.triangles[a:b], source_faces=self.data.source_faces[a:b],
            triangle_areas_m2=self.data.triangle_areas[a:b], center_m=self.data.centers_m[index],
            center_face=int(self.data.center_faces[index]), radius_m=float(self.data.radius_m[index]),
            candidate_index=int(index), candidate_id=self.geometry['patches'][index]['id'])

    def supply(self, contacts):
        groups = [self.floor_columns]
        for patch in contacts:
            points = patch['triangles_m'].reshape(-1, 3)
            normal = np.repeat(-self.domain.mesh.face_normals[patch['source_faces']], 3, axis=0)
            groups.append(U.heads(np.c_[normal, np.cross(points-self.domain.com, normal)], self.scale))
        return I.merge_columns(*groups)

    def load_state(self, path=None):
        if path is None:
            mask, _ = J.classify(self.floor_columns, self.targets)
            return [], mask, []
        path = Path(path)
        meta = I.check_report(path)
        assert meta['object'] == self.name and meta['sample_count'] <= len(self.targets)
        patches = I.read_contacts(path.parent/'contacts.npz')
        mask = I.load_npz(path.parent/'sample_coverage.npz')['adjusted']
        assert mask.shape == (meta['sample_count'],) and int(mask.sum()) == meta['covered_count']
        assert [p['candidate_index'] for p in patches] == meta['selected_indices']
        for patch in patches:
            index = patch['candidate_index']
            assert self.data.valid[index]
            np.testing.assert_array_equal(patch['center_m'], self.data.centers_m[index])
            np.testing.assert_allclose(patch['triangle_areas_m2'], P.areas(patch['triangles_m']), rtol=1e-12)
        if len(mask) < len(self.targets):
            # A continuous-domain counterexample extends the requirement list.
            # Keep old geometry; recompute feasibility of the added loads.
            extra, _ = J.classify(self.supply(patches), self.targets[len(mask):])
            mask = np.r_[mask, extra]
        return patches, mask, [path, path.parent/'contacts.npz', path.parent/'sample_coverage.npz']

    def add_counterexample(self, case, path, write=True):
        evaluated = self.domain.evaluate(case['work_face_index'], case['u'], case['v'],
            case['theta_rad'], case['phi_rad'], magnitude_mg=case['magnitude_mg'])
        if not bool(evaluated['reachable']):
            raise ValueError('Counterexample is outside the reachable load domain')
        np.testing.assert_allclose(evaluated['need_wrench'], case['need_wrench'], atol=1e-13)
        self.targets = np.vstack([self.targets, evaluated['need_wrench']*self.scale])
        self.counterexamples.append(case)
        if write:
            I.save(path, dict(object=self.name, random_sample_count=self.random_sample_count,
                              counterexamples=self.counterexamples, total_search_loads=len(self.targets)))
        self.inputs.append(Path(path))


def gravity_check(full, domain, scale):
    """Hard per-design test, independent of the gradually covered work loads."""
    gravity = np.r_[-domain.gravity, np.zeros(3)]*scale
    witness = W.solve(full, gravity)
    return dict(passed=witness is not None, witness=witness,
                load='zero_process_force', no_uplift_equation_included=True,
                scope='Object equilibrium and shared support no-uplift; finite-footprint tipping not certified.')


def run(name, round_number=1, state_path=None, problem=None):
    tick = time.monotonic()
    problem = problem or Problem(name)
    fixed, base, state_inputs = problem.load_state(state_path)
    assert len(fixed)+1 == round_number
    already = {p['candidate_index'] for p in fixed}
    out = I.folder(name, OUTPUT_NAME, round_number)
    out.mkdir(parents=True, exist_ok=True)
    I.save(out/'status.json', dict(object=name, complete=False, status='scoring'))
    fixed_full = problem.supply(fixed)
    masks = np.zeros((len(problem.data.valid), len(base)), bool)
    rows = []
    for index, info in enumerate(problem.geometry['patches']):
        row = dict(index=index, id=info['id'], center_m=problem.data.centers_m[index].tolist(),
            radius_m=float(problem.data.radius_m[index]), area_m2=float(info['area_m2']),
            covered_count=None, covered_percent=None, gain_count=None, gain_percentage_points=None)
        if index in already:
            row['status'] = 'already_selected'
        elif not problem.data.valid[index]:
            row['status'] = 'rejected_geometry'
        else:
            full = I.merge_columns(fixed_full, columns(problem.domain, problem.data, index, problem.floor, problem.scale))
            hard = gravity_check(full, problem.domain, problem.scale)
            row['hard_feasibility'] = hard
            if not hard['passed']:
                row['status'] = 'rejected_rest_equilibrium'
                rows.append(row)
                continue
            masks[index], classifier = J.classify(full, problem.targets, known_covered=base)
            verification = verify_classification(full, problem.targets, masks[index])
            count = int(masks[index].sum())
            row.update(status='scored_joint', covered_count=count, covered_percent=100*count/len(base),
                gain_count=count-int(base.sum()), gain_percentage_points=100*(count-int(base.sum()))/len(base),
                classifier=classifier, verification=verification)
        rows.append(row)
        if (index+1) % 20 == 0:
            print(name, 'round', round_number, 'Step 3.1', index+1, '/', len(rows)+len(problem.data.valid)-index-1,
                  'seconds', round(time.monotonic()-tick, 1), flush=True)
    np.savez_compressed(out/'sample_coverage.npz', covered=masks, base=base,
        eligible=np.array([r['status'] == 'scored_joint' for r in rows]),
        insertion_eligible=np.array([i not in already and problem.data.valid[i] for i in range(len(rows))]))
    fields = ['id', 'index', 'status', 'covered_count', 'covered_percent', 'gain_count', 'gain_percentage_points', 'area_m2']
    with (out/'contributions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    result = dict(object=name, round=round_number, complete=True, candidate_count=len(rows),
        definition='joint coverage with all previously optimized contacts and the original floor point',
        original_floor_reaction_model=F.description(),
        passive_support_constraint=U.description(),
        hard_filter='common_head_direction_then_zero_process_force_equilibrium_with_shared_no_uplift',
        selected_indices=sorted(already), fixed_area_m2=I.area(fixed),
        base_covered_count=int(base.sum()), sample_count=len(base), sample_seed=problem.samples['seed'],
        random_sample_count=problem.random_sample_count, validation_load_count=len(problem.counterexamples),
        contributions=rows, reactions_reoptimized=True,
        provenance=dict(inputs=I.hashes(problem.inputs+state_inputs), code=code_hashes()),
        artifacts={f: sha256(out/f) for f in ['sample_coverage.npz', 'contributions.csv']},
        elapsed_seconds=time.monotonic()-tick)
    I.save(out/'contributions.json', result)
    I.save(out/'status.json', dict(object=name, complete=True, status='scored',
                                  contributions_sha256=sha256(out/'contributions.json')))
    return result


def read(name, round_number=1):
    return I.check_report(I.folder(name, OUTPUT_NAME, round_number)/'contributions.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--state', type=Path)
    args = parser.parse_args()
    for name in args.objects or OBJECTS:
        run(name, args.round, args.state)
