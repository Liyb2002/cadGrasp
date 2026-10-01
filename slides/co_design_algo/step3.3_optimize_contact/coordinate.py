"""Bounded coordinate sizing of every contact, with fixed contact centers."""
from pathlib import Path
import time
import numpy as np
from step3_scheculer.stage_imports import load_stage
from step3_scheculer import contacts as I
from step3_scheculer import paths as PTH
from step2_local_support import insertion_directions as K, withdrawal as D

A = load_stage('optimize', 'adjust')
C = A.C
FAST = load_stage('optimize', 'marginal')


def search_coordinate(problem, budget):
    """Reuse the certified scalar search; once full coverage is found, keep it."""
    if budget < 3:
        raise ValueError('A coordinate needs at least three coverage evaluations')
    initial = problem.score(problem.initial_radius)
    def rest_feasible(radius):
        entry = problem.geometry(radius)
        full = I.merge_columns(problem.fixed_full, A.patch_columns(
            problem.domain, entry['patch'], problem.floor, problem.scale))
        return C.gravity_check(full, problem.domain, problem.scale)['passed']
    lower, rest_limit = A.size_search.feasible_radius_floor(
        problem.minimum_radius, problem.initial_radius, problem.tolerance, rest_feasible)
    upper, geometry_limit = problem.maximum_radius()
    upper, insertion_limit = problem.insertion_radius_cap(upper)
    maximum = problem.score(upper)
    count = len(problem.targets)
    evaluations = lambda: sum('mask' in entry for entry in problem.cache.values())
    full_found = maximum['row']['covered_count'] == count
    if full_found:
        # Monotone coverage at a fixed center: find a small full-coverage radius.
        best = problem.initial_radius if initial['row']['covered_count'] == count else upper
        low = problem.score(lower)
        if low['row']['covered_count'] == count:
            best = lower
            bracket = 0.
        else:
            left, right = lower, best
            while right-left > problem.tolerance and evaluations() < budget:
                middle = (left+right)/2
                if problem.score(middle)['row']['covered_count'] == count:
                    right = middle
                else:
                    left = middle
            best = right
            bracket = right-left
        search = dict(method='shrink_preserving_full_coverage',
                      converged=bool(bracket <= problem.tolerance),
                      relative_gap=None)
    elif upper-lower <= problem.tolerance:
        best = problem.initial_radius
        search = dict(method='radius_resolution', converged=True, relative_gap=0.)
    else:
        best, search = A.maximize_efficiency(
            lambda radius: problem.score(radius)['row'],
            [lower, problem.initial_radius, upper], problem.initial_radius,
            problem.tolerance, max_evaluations=budget)
    assert evaluations() <= budget
    return best, dict(search, evaluated_sizes=evaluations(), max_evaluations=budget,
                     full_coverage_found=full_found, rest_radius_limit=rest_limit,
                     geometry_limit=geometry_limit, insertion_limit=insertion_limit)


def coordinate_order(count):
    return [count-1, *range(count-1)]


def summary(contacts, mask, object_area):
    current = contacts[-1]
    area = I.area(contacts)
    return dict(radius_m=current['radius_m'], area_m2=I.area([current]),
                fixed_area_m2=I.area(contacts[:-1]), total_area_m2=area,
                covered_count=int(mask.sum()), covered_percent=100*float(mask.mean()),
                efficiency=float(mask.mean())*object_area/area)


def run(name, round_number=1, problem=None, sweeps=2, max_evaluations=96):
    if sweeps < 1 or max_evaluations < 3*sweeps*round_number:
        raise ValueError('Sizing budget must allow three evaluations per contact per sweep')
    started = time.monotonic()
    problem = problem or C.Problem(name)
    source = PTH.folder(name, A.M.OUTPUT_NAME, round_number)
    selected = A.M.read(name, round_number)
    before = I.read_contacts(source/'contacts_before_optimization.npz')
    contacts = list(before)
    masks = I.load_npz(source/'sample_coverage.npz')
    mask = masks['selected'].copy()
    object_area = float(problem.domain.mesh.area)
    initial = summary(contacts, mask, object_area)
    catalogue = K.read(name)
    cat = catalogue['direction_catalogue']
    analyzer = D.Analyzer(problem.domain.mesh, catalogue['normal_depth_m'], cat)
    directions = {r['geometry_signature']: r for r in catalogue['candidates']
                  if r.get('geometry_signature')}
    filtering = I.check_report(PTH.folder(name, 'step3_scheculer', round_number)/'candidate_filter.json')
    if filtering.get('previous_direction_record'):
        previous = I.check_report(I.ROOT/filtering['previous_direction_record'])
        directions.update({r['geometry_signature']: r for r in previous['contacts']})

    def record(contact):
        signature = D.signature(contact, catalogue['normal_depth_m'])
        if signature not in directions:
            directions[signature] = analyzer.analyze(contact)
        return directions[signature]

    def make_size_problem(index):
        others = [record(p) for j,p in enumerate(contacts) if j != index]
        allowed = D.common(others, cat['global_allowed_directions'])
        scalar = A.SizeProblem(name, round_number, problem, contacts=contacts,
                               coordinate_index=index, direction_catalogue=cat,
                               allowed_directions=allowed, local_search=True)
        initial_common = D.common([record(contacts[index])], allowed)
        scalar.inherited_direction_radius = scalar.initial_radius
        scalar.inherited_directions = initial_common
        return scalar

    out = PTH.folder(name, A.OUTPUT_NAME, round_number)
    out.mkdir(parents=True, exist_ok=True)
    I.save(out/'status.json', dict(object=name, complete=False, status='optimizing_all_contacts'))
    updates = []
    used = 0
    planned = sweeps*len(contacts)
    for sweep in range(sweeps):
        changed = False
        for index in coordinate_order(len(contacts)):
            quota = (max_evaluations-used)//(planned-len(updates))
            scalar = make_size_problem(index)
            old = scalar.score(scalar.initial_radius)
            np.testing.assert_array_equal(old['mask'], mask)
            radius, search = FAST.search(scalar, quota)
            entry = scalar.score(radius)
            # Every scalar invocation has fresh fixed geometry and coverage hints.
            if mask.all():
                assert entry['mask'].all()
                assert entry['row']['total_area_m2'] <= I.area(contacts)*(1+1e-10)
            elif not entry['mask'].all():
                assert entry['row']['efficiency'] >= old['row']['efficiency']*(1-1e-12)
            changed |= radius != scalar.initial_radius
            prior_record = record(contacts[index])
            contacts[index] = scalar.contact(radius)
            if radius <= scalar.initial_radius:
                # Certified rays of an enclosing, same-center affine head stay
                # clear under nested shrinkage. New rays are discovered once by
                # the full tracker update after this round, before next scoring.
                signature = D.signature(contacts[index], catalogue['normal_depth_m'])
                directions[signature] = dict(certified_directions=prior_record['certified_directions'],
                    geometry_signature=signature, inherited_nested_shrinkage=True)
            mask = entry['mask'].copy()
            used += search['evaluated_sizes']
            updates.append(dict(sweep=sweep+1, coordinate_index=index,
                                candidate_id=contacts[index]['candidate_id'],
                                initial_radius_m=scalar.initial_radius, radius_m=radius,
                                before=old['row'], after=entry['row'], search=search))
            print(name, 'round', round_number, 'coordinate', index, 'sweep', sweep+1,
                  'coverage', entry['row']['covered_percent'], 'sizing evaluations', used, flush=True)
        if not changed:
            break
    # Independent exported-geometry checks are outside the search budget.
    checks = []
    for index in range(len(contacts)):
        scalar = make_size_problem(index)
        checks.append(scalar.verify(scalar.initial_radius))
        np.testing.assert_array_equal(scalar.score(scalar.initial_radius)['mask'], mask)
    adjusted = summary(contacts, mask, object_area)
    verification = dict(hard_feasibility=checks[-1]['hard_feasibility'],
                        work_volume_clearance=dict(passed=all(v['work_volume_clearance']['passed'] for v in checks)),
                        contacts=checks)
    I.save_contacts(out/'contacts.npz', contacts)
    I.save_contacts(out/'adjusted_contact.npz', [contacts[-1]])
    np.savez_compressed(out/'sample_coverage.npz', base=masks['base'],
                        initial=masks['selected'], adjusted=mask)
    inputs = problem.inputs+[source/'selection.json', source/'contacts_before_optimization.npz',
                             source/'sample_coverage.npz', K.path(name)]
    minimum = A.area_limit.MIN_AREA_FRACTION*object_area
    result = dict(object=name, round=round_number, complete=True,
        optimization_mode='coordinate_all_contacts', candidate_id=contacts[-1]['candidate_id'],
        candidate_index=contacts[-1]['candidate_index'], selected_indices=[p['candidate_index'] for p in contacts],
        sample_count=len(mask), random_sample_count=problem.random_sample_count,
        validation_load_count=len(problem.counterexamples), sample_seed=problem.samples['seed'],
        baseline_covered_count=int(masks['base'].sum()), covered_count=int(mask.sum()),
        initial=initial, adjusted=adjusted, fixed_area_m2=adjusted['fixed_area_m2'],
        direction='jointly adjusted', curve=[dict(u['after'], update=i+1) for i,u in enumerate(updates)],
        coordinate_updates=updates,
        selection=dict(method='bounded_coordinate_search', converged=False,
                       relative_gap=None, sweeps_completed=max(u['sweep'] for u in updates),
                       max_sweeps=sweeps, evaluated_sizes=used, max_evaluations=max_evaluations,
                       scope='Local coordinate search; no joint/global optimality certificate. Budget counts coverage evaluations; geometry, gravity and final verification checks are additional.'),
        verification=dict(adjusted=verification),
        area_constraint=dict(minimum_area_m2=minimum, minimum_area_fraction=A.area_limit.MIN_AREA_FRACTION),
        work_volume_constraint=dict(enforced=A.P.POLICY.ENFORCE_PROCESS_ACCESS),
        passive_support_constraint=C.U.description(),
        efficiency_definition='Local actual-area secants propose sizes; exact joint coverage / total area accepts them. Full coverage is preserved.',
        elapsed_seconds=time.monotonic()-started,
        provenance=dict(inputs=I.hashes(inputs), code=dict(A.code_hashes(), **I.hashes([Path(__file__)]))),
        artifacts={f:C.sha256(out/f) for f in ['contacts.npz','adjusted_contact.npz','sample_coverage.npz']})
    I.save(out/'adjustment.json', result)
    I.save(out/'state.json', result)
    I.save(out/'status.json', dict(object=name, complete=True, status='optimized',
                                 state_sha256=C.sha256(out/'state.json')))
    return result
