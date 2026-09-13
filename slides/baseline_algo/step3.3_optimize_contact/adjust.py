"""Optimize only this round's contact; preserve every previously fixed geometry."""
import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.stage_imports import load_stage
C = load_stage('score', 'contribution')
M = load_stage('select', 'choose')
size_search = load_stage('optimize', 'size_search')
insertion_limit = load_stage('optimize', 'insertion_limit')
area_limit = load_stage('optimize', 'area_limit')
maximize_efficiency = size_search.maximize_efficiency
efficiency = size_search.efficiency
triangulate = size_search.triangulate
patch_columns = size_search.patch_columns
EFFICIENCY_REL_TOL = size_search.EFFICIENCY_REL_TOL
RADIUS_REL_TOL = size_search.RADIUS_REL_TOL
from step2_local_support import circles as P, connectivity
from step2_local_support import withdrawal as D
from step2_local_support import work_volume as W, work_clearance as WC
from step3_scheculer import contacts as I

OUTPUT_NAME = 'step3.3_optimize_contact'


def code_hashes():
    paths = [Path(__file__), Path(size_search.__file__), Path(M.__file__), Path(P.__file__),
             Path(P.S.__file__), Path(P.G.__file__), Path(connectivity.__file__),
             Path(insertion_limit.__file__), Path(area_limit.__file__),
             Path(W.__file__), Path(W.V.__file__), Path(WC.__file__)]
    return {**C.code_hashes(), **D.code_hashes(), **I.hashes(paths)}


class SizeProblem:
    def __init__(self, name, round_number=1, problem=None):
        self.name, self.round_number = name, round_number
        self.problem = problem or C.Problem(name)
        self.domain, self.data, self.geometry_report = self.problem.domain, self.problem.data, self.problem.geometry
        self.scale, self.floor = self.problem.scale, self.problem.floor
        self.samples, self.targets = self.problem.samples, self.problem.targets
        self.selection = M.read(name, round_number)
        self.winner = self.selection['winner']
        if self.winner is None:
            raise ValueError('No selected candidate to optimize')
        source = I.folder(name, M.OUTPUT_NAME, round_number)
        before = I.read_contacts(source/'contacts_before_optimization.npz')
        self.fixed, selected = before[:-1], before[-1]
        self.fixed_area = I.area(self.fixed)
        self.fixed_full = self.problem.supply(self.fixed)
        self.index = selected['candidate_index']
        self.center, self.seed, self.initial_radius = selected['center_m'], selected['center_face'], selected['radius_m']
        assert self.index == self.winner['index'] and len(before) == round_number
        masks = I.load_npz(source/'sample_coverage.npz')
        self.base_mask, self.initial_mask = masks['base'], masks['selected']
        self.base_count = int(self.base_mask.sum())
        self.input_paths = self.problem.inputs+[source/'selection.json', source/'selected_contact.npz',
            source/'contacts_before_optimization.npz', source/'sample_coverage.npz']
        _, polygons = P.eligible_polygons(self.domain)
        self.surface = P.SurfaceCircles(self.domain.mesh, polygons)
        self.pool = self.surface.pool(self.seed)
        self.clearance = P.LocalClearance(self.domain.mesh, self.geometry_report['normal_depth_m'])
        volume_folder = C.OUTPUTS/name/P.pose_name()/W.STAGE
        if P.POLICY.ENFORCE_PROCESS_ACCESS:
            self.input_paths += [volume_folder/f for f in ['work_volume.json','work_volume.npz','work_volume_audit.json']]
        if not hasattr(self.problem, 'work_volume') or not P.POLICY.ENFORCE_PROCESS_ACCESS:
            self.problem.work_volume = W.WorkVolume.read(volume_folder/'work_volume.json') if P.POLICY.ENFORCE_PROCESS_ACCESS else None
        self.work_clearance = WC.ContactClearance(self.domain.mesh, self.geometry_report['normal_depth_m'],
                                                 self.problem.work_volume, self.clearance.offsets)
        self.work_safe_radius = 0.
        self.tolerance = max(self.initial_radius*RADIUS_REL_TOL, 64*self.surface.tol)
        numerical_minimum = max(self.initial_radius*1e-6, 64*self.surface.tol)
        self.minimum_area = area_limit.MIN_AREA_FRACTION*float(self.domain.mesh.area)
        if any(I.area([p]) <= self.minimum_area for p in self.fixed):
            raise ValueError('A fixed contact is below the current area requirement; rerun Step 3')
        vertices = np.concatenate([polygons[f] for f in self.pool])
        self.cap = float(np.linalg.norm(vertices-self.center, axis=1).max())*(1+1e-9)
        self.cache = {}
        # A fixed-center smaller circle and its affine joined skin are subsets
        # of this checked head. Reuse that containment proof during shrinkage.
        if not self.geometry(self.initial_radius)['row']['geometry_valid']:
            raise ValueError('Selected initial head violates current clearance; rebuild Step 2')
        self.minimum_radius, self.area_constraint = area_limit.radius_floor(
            numerical_minimum, self.initial_radius, self.tolerance,
            lambda radius: self.geometry(radius)['row']['area_m2'], self.minimum_area)
        self.area_constraint['total_object_area_m2'] = float(self.domain.mesh.area)
        self.allowed_directions=None
        filter_path=I.folder(name,'step3_scheculer',round_number)/'candidate_filter.json'
        if str(filter_path.relative_to(C.ROOT)) in self.selection['provenance']['inputs']:
            filtering=I.check_report(filter_path)
            assert filtering['eligible'][self.index]
            assert filtering['mode']==D.MODE
            self.allowed_directions=filtering['common_before_selection']
            self.direction_catalogue=filtering['direction_catalogue']
        self.direction_analyzer=None
        self.direction_cache={}

    def insertion_directions(self,radius):
        if radius not in self.direction_cache:
            contact=self.contact(radius)
            if self.direction_analyzer is None:
                self.direction_analyzer=D.Analyzer(self.domain.mesh,self.geometry_report['normal_depth_m'],self.direction_catalogue)
            record=self.direction_analyzer.analyze(contact,self.allowed_directions,stop_after_first=True)
            self.direction_cache[radius]=record['certified_directions']
        return self.direction_cache[radius]

    def insertion_radius_cap(self,maximum):
        if self.allowed_directions is None:
            return maximum,dict(enforced=False,reason='Standalone selection has no scheduler direction filter')
        radius,record=insertion_limit.radius_cap(self.initial_radius,maximum,self.tolerance,
            lambda r:D.nonempty(self.insertion_directions(r)))
        return radius,dict(record,enforced=True,allowed_common_directions=self.allowed_directions,
                           unconstrained_geometry_radius_m=maximum,
                           scope='Conservative certified radius interval for nested contact areas; unresolved directions are excluded')

    def geometry(self, radius):
        radius = float(radius)
        if radius not in self.cache:
            patch, area = self.surface.at_radius(self.center, self.seed, radius, self.pool)
            checked = self.clearance.check(patch)
            if checked['valid']:
                if self.work_clearance.work is None:
                    work_check = P.POLICY.skipped_access_check()
                elif radius <= self.work_safe_radius:
                    work_check = dict(passed=True, classification='clear', method='nested_head_containment',
                        enclosing_checked_radius_m=self.work_safe_radius,
                        scope='Same center, fixed source-face polygons and fixed affine vertex offsets')
                else:
                    work_check = self.work_clearance.check(patch)
                    if work_check['passed']:
                        self.work_safe_radius = radius
                checked = WC.apply_constraint(checked, work_check)
            self.cache[radius] = dict(patch=patch, row=dict(radius_m=radius, area_m2=area,
                total_area_m2=self.fixed_area+area, fixed_area_m2=self.fixed_area,
                object_area_percent=100*area/self.domain.mesh.area,
                geometry_valid=checked['valid'], geometry_status=checked['status'], geometry_check=checked))
        return self.cache[radius]

    def contact(self, radius):
        triangles, faces = triangulate(self.geometry(radius)['patch'])
        return dict(triangles_m=triangles, source_faces=faces, triangle_areas_m2=P.areas(triangles),
            center_m=self.center, center_face=self.seed, radius_m=float(radius),
            candidate_index=self.index, candidate_id=self.winner['id'])

    def score(self, radius):
        entry = self.geometry(radius)
        if not entry['row']['area_m2'] > self.minimum_area:
            raise ValueError('Contact area must exceed 0.5% of total object surface area')
        if not entry['row']['geometry_valid']:
            raise ValueError(f'Rejected size {radius}: {entry["row"]["geometry_status"]}')
        if 'mask' not in entry:
            full = I.merge_columns(self.fixed_full, patch_columns(self.domain, entry['patch'], self.floor, self.scale))
            hard=C.gravity_check(full,self.domain,self.scale)
            if not hard['passed']:
                raise ValueError('Size violates the hard rest-equilibrium/no-uplift constraint')
            known = self.base_mask
            smaller = [(r, e['mask']) for r, e in self.cache.items() if r <= radius and 'mask' in e]
            if smaller:
                known = max(smaller, key=lambda item: item[0])[1]
            mask, info = C.J.classify(full, self.targets, known_covered=known)
            for other_radius, other in self.cache.items():
                if 'mask' in other:
                    low, high = (mask, other['mask']) if radius < other_radius else (other['mask'], mask)
                    if np.any(low & ~high):
                        raise RuntimeError('Nested contact coverage decreased')
            count = int(mask.sum())
            entry.update(full=full, mask=mask,hard_feasibility=hard)
            entry['row'].update(covered_count=count, covered_percent=100*count/len(mask),
                gain_count=count-self.base_count, classifier=info,
                efficiency=(count/len(mask))*self.domain.mesh.area/entry['row']['total_area_m2'],
                coverage_per_m2=(count/len(mask))/entry['row']['total_area_m2'])
        return entry

    def maximum_radius(self):
        low, high = self.initial_radius, self.cap
        assert self.geometry(low)['row']['geometry_valid']
        if self.geometry(high)['row']['geometry_valid']:
            return high, dict(reason='entire_center_connected_pool', lower_valid_radius_m=high,
                              upper_invalid_radius_m=None, iterations=0)
        iterations = 0
        while high-low > self.tolerance:
            middle = (low+high)/2
            if self.geometry(middle)['row']['geometry_valid']:
                low = middle
            else:
                high = middle
            iterations += 1
        return low, dict(reason=self.geometry(high)['row']['geometry_status'], lower_valid_radius_m=low,
                         upper_invalid_radius_m=high, iterations=iterations)

    def verify(self, radius):
        entry = self.score(radius)
        contact = self.contact(radius)
        hard=C.gravity_check(self.problem.supply(self.fixed+[contact]),self.domain,self.scale)
        assert hard['passed'], 'Exported contacts fail rest equilibrium'
        _, topology = connectivity.components(self.domain.mesh, contact['triangles_m'], contact['source_faces'])
        assert topology['components'] == 1
        spread = P.normal_spread(self.domain.mesh, np.unique(contact['source_faces']))
        assert spread['wrap_limit_satisfied']
        assert not set(contact['source_faces']) & set(self.domain.work_ids)
        triangles=contact['triangles_m']
        perimeter=np.linalg.norm(triangles-np.roll(triangles,1,axis=1),axis=2).sum()
        area_roundoff=32*np.finfo(float).eps*max(float(np.abs(triangles).max()),float(self.domain.mesh.extents.max()))*perimeter
        np.testing.assert_allclose(contact['triangle_areas_m2'].sum(), entry['row']['area_m2'], rtol=1e-10,atol=area_roundoff)
        assert contact['triangle_areas_m2'].sum() > self.minimum_area
        directions=self.insertion_directions(radius) if self.allowed_directions is not None else None
        if directions is not None:assert D.nonempty(directions)
        # Reconstruct the exported triangles' actual head, independently of
        # radius containment shortcuts used while searching.
        analyzer = D.H.Analyzer(self.domain.mesh, self.geometry_report['normal_depth_m'])
        work_check = self.work_clearance.check_parts(analyzer.heads(contact))
        if not work_check['passed']:
            raise RuntimeError('Final optimized contact head fails work-volume clearance')
        return dict(connectivity=topology, normal_spread=spread,area_roundoff_tolerance_m2=area_roundoff,
                    area_above_minimum=True,
                    hard_feasibility=hard,
                    work_volume_clearance=work_check,
                    certified_common_insertion_directions=directions,
                    force_checks=C.verify_classification(entry['full'], self.targets, entry['mask']))


def run(name, round_number=1, problem=None, curve_points=25,
        efficiency_tolerance=EFFICIENCY_REL_TOL, max_evaluations=512):
    started = time.monotonic()
    problem = SizeProblem(name, round_number, problem)
    def rest_feasible(radius):
        entry=problem.geometry(radius)
        full=I.merge_columns(problem.fixed_full,patch_columns(problem.domain,entry['patch'],problem.floor,problem.scale))
        return C.gravity_check(full,problem.domain,problem.scale)['passed']
    problem.minimum_radius,rest_radius_limit=size_search.feasible_radius_floor(
        problem.minimum_radius,problem.initial_radius,problem.tolerance,rest_feasible)
    out = I.folder(name, OUTPUT_NAME, round_number)
    out.mkdir(parents=True, exist_ok=True)
    I.save(out/'status.json', dict(object=name, complete=False, status='optimizing'))
    initial = problem.score(problem.initial_radius)
    np.testing.assert_array_equal(initial['mask'], problem.initial_mask)
    maximum_radius, limit = problem.maximum_radius()
    maximum_radius,insertion_constraint=problem.insertion_radius_cap(maximum_radius)
    maximum = problem.score(maximum_radius)
    grid = np.unique(np.r_[np.linspace(problem.minimum_radius, maximum_radius, curve_points),
        problem.initial_radius*np.array([.25, .5, .75, .9, .95, 1., 1.05, 1.1, 1.25]), maximum_radius])
    grid = grid[(grid >= problem.minimum_radius) & (grid <= maximum_radius)]
    evaluations = 0

    def evaluate(radius):
        nonlocal evaluations
        row = problem.score(radius)['row']
        evaluations += 1
        if evaluations % 25 == 0:
            print(name, 'round', round_number, 'Step 3.3', evaluations, 'sizes', flush=True)
        return row

    radius, search = maximize_efficiency(evaluate, grid, problem.initial_radius, problem.tolerance,
                                         efficiency_tolerance, max_evaluations)
    adjusted = problem.score(radius)
    states = dict(initial=problem.initial_radius, adjusted=radius, maximum=maximum_radius)
    verification = {key: problem.verify(value) for key, value in states.items()}
    current = problem.contact(radius)
    fixed = problem.fixed+[current]
    I.save_contacts(out/'adjusted_contact.npz', [current])
    I.save_contacts(out/'contacts.npz', fixed)
    np.savez_compressed(out/'sample_coverage.npz', base=problem.base_mask,
        **{key: problem.score(value)['mask'] for key, value in states.items()})
    curves = [entry['row'] for _, entry in sorted(problem.cache.items()) if 'mask' in entry]
    fields = ['radius_m', 'area_m2', 'fixed_area_m2', 'total_area_m2', 'object_area_percent',
              'covered_count', 'covered_percent', 'gain_count', 'efficiency', 'coverage_per_m2']
    with (out/'coverage_curve.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(curves)
    direction = 'shrunk' if radius < problem.initial_radius else 'expanded' if radius > problem.initial_radius else 'unchanged'
    result = dict(object=name, round=round_number, complete=True, candidate_id=current['candidate_id'],
        candidate_index=current['candidate_index'], selected_indices=[p['candidate_index'] for p in fixed],
        center_m=problem.center.tolist(), center_face=problem.seed, fixed_area_m2=problem.fixed_area,
        sample_count=len(problem.targets), sample_seed=problem.samples['seed'],
        random_sample_count=problem.problem.random_sample_count,
        validation_load_count=len(problem.problem.counterexamples),
        baseline_covered_count=problem.base_count, covered_count=int(adjusted['mask'].sum()),
        direction=direction, selection=search, geometry_limit=limit, curve=curves, verification=verification,
        insertion_constraint=insertion_constraint,
        passive_support_constraint=C.U.description(),
        rest_equilibrium_radius_constraint=rest_radius_limit,
        work_volume_constraint=dict(enforced=P.POLICY.ENFORCE_PROCESS_ACCESS, owner_stage=W.STAGE,
            boundary_contact_allowed=False, unresolved_eligible=False,
            scope='Process-access check disabled; contacts still exclude work faces.' if not P.POLICY.ENFORCE_PROCESS_ACCESS else 'Complete installed head at every admissible radius; connector checked in Step 5'),
        area_constraint=problem.area_constraint,
        efficiency_definition='joint coverage / (area of all fixed contacts + area of current contact)',
        efficiency_change_percent=100*(adjusted['row']['efficiency']/initial['row']['efficiency']-1) if initial['row']['efficiency'] else None,
        area_change_percent=100*(adjusted['row']['area_m2']/initial['row']['area_m2']-1),
        **{key: problem.score(value)['row'] for key, value in states.items()},
        provenance=dict(inputs=I.hashes(problem.input_paths), code=code_hashes()),
        artifacts={f: C.sha256(out/f) for f in
                   ['adjusted_contact.npz', 'contacts.npz', 'sample_coverage.npz', 'coverage_curve.csv']},
        elapsed_seconds=time.monotonic()-started)
    I.save(out/'adjustment.json', result)
    I.save(out/'state.json', result)
    I.save(out/'status.json', dict(object=name, complete=True, status='optimized',
                                  state_sha256=C.sha256(out/'state.json')))
    print(name, 'round', round_number, 'Step 3.3', direction, 'coverage', adjusted['row']['covered_percent'],
          'area mm2', 1e6*adjusted['row']['area_m2'], 'efficiency gap', search['relative_gap'], flush=True)
    return result


def read(name, round_number=1):
    return I.check_report(I.folder(name, OUTPUT_NAME, round_number)/'adjustment.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--round', type=int, default=1)
    args = parser.parse_args()
    for name in args.objects or C.OBJECTS:
        run(name, args.round)
