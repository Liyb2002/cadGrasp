"""Select and size exact contact patches; connected frames and shared paths belong to Step5."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.stage_imports import load_stage
C = load_stage('score', 'contribution')
M = load_stage('select', 'choose')
A = load_stage('optimize', 'adjust')
from step3_scheculer import contacts as I
from step3_scheculer import directions as D
from step3_scheculer import completion as Q

OUTPUT_NAME = 'step3_scheculer'

from step1.cases import pose_name


def iterate(score, choose, optimize, validate=None):
    """Completion is checked AFTER optimization, never on the candidate score."""
    rounds = []
    selected_indices = set()
    while len(rounds) < Q.MAX_CONTACTS:
        number = len(rounds)+1
        contribution = score(number)
        selected = choose(number)
        if selected['winner'] is None:
            return rounds, 'candidates_exhausted'
        index = selected['winner']['index']
        if index in selected_indices:
            raise RuntimeError(f'Step 3.2 selected an already fixed candidate: {index}')
        selected_indices.add(index)
        adjusted = optimize(number)
        if not 0 <= adjusted['covered_count'] <= adjusted['sample_count'] or adjusted['sample_count'] <= 0:
            raise ValueError('Expected a nonempty load set and a valid covered count')
        rounds.append(dict(round=number, candidate_id=selected['winner']['id'],
            candidate_index=selected['winner']['index'],
            before_optimization_covered_percent=selected['winner']['covered_percent'],
            after_optimization_covered_percent=100*adjusted['covered_count']/adjusted['sample_count'],
            current_area_m2=adjusted['adjusted']['area_m2'],
            total_area_m2=adjusted['adjusted']['total_area_m2'],
            size_search_converged=adjusted['selection']['converged'],
            scoring_seconds=contribution['elapsed_seconds'], optimization_seconds=adjusted['elapsed_seconds']))
        if 'insertion' in adjusted:
            rounds[-1]['insertion'] = adjusted['insertion']
            if not adjusted['insertion']['all_contacts_have_certified_direction']:
                return rounds, 'no_common_insertion_direction_after_optimization'
        if adjusted['covered_count'] == adjusted['sample_count']:
            if validate is None:
                return rounds, 'sampled_complete'
            validation = validate(number, adjusted)
            rounds[-1]['continuous_validation'] = validation['status']
            if validation['status'] == 'verified':
                return rounds, 'continuous_contact_model_verified'
            if validation['status'] != 'counterexample':
                return rounds, 'continuous_validation_inconclusive'
    return rounds, 'contact_limit_reached'


def run(name, draw=True, resume=False):
    started = time.monotonic()
    Q.archive_downstream(name)
    problem = C.Problem(name)
    out = C.OUTPUTS/name/pose_name()/OUTPUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    I.save(out/'status.json', dict(object=name, status='running', complete=False))
    previous_state = None
    tracker = D.Tracker(problem, out)
    problem.inputs.append(tracker.catalogue_path)
    wall_timings = []

    def timed(stage, number, operation):
        tick = time.monotonic()
        try:
            return operation()
        finally:
            wall_timings.append(dict(stage=stage, round=number, elapsed_seconds=time.monotonic()-tick))

    def score(number):
        filter_path, eligible, selected = tracker.prepare(number)
        problem.inputs.append(filter_path)
        if resume:
            try:
                result = C.read(name, number)
                assert result['sample_count'] == len(problem.targets)
                assert D.inputs_match(result, problem.inputs)
                return result
            except (OSError, RuntimeError, AssertionError):
                pass
        with D.scoring_mask(problem, eligible, selected):
            return C.run(name, number, previous_state, problem)

    def choose(number):
        if resume:
            try:
                result = M.read(name, number)
                assert result['sample_count'] == len(problem.targets)
                assert D.inputs_match(result, problem.inputs)
                return result
            except (OSError, RuntimeError, AssertionError):
                pass
        return M.run(name, number, previous_state, problem)

    def optimize(number):
        nonlocal previous_state
        result = None
        if resume:
            try:
                result = A.read(name, number)
                assert result['sample_count'] == len(problem.targets)
                assert D.inputs_match(result, problem.inputs)
            except (OSError, RuntimeError, AssertionError):
                result = None
        if result is None:
            result = timed('3.3.size_search', number, lambda: A.run(name, number, problem))
        previous_state = I.folder(name, A.OUTPUT_NAME, number)/'state.json'
        directions, direction_path = timed('3.3.direction_update', number,
            lambda: tracker.update(previous_state.parent/'contacts.npz', previous_state, number))
        problem.inputs.append(direction_path)
        if draw:
            selection_draw = load_stage('select', 'draw')
            adjustment_draw = load_stage('optimize', 'draw')
            timed('3.3.selection_drawing', number, lambda: selection_draw.run(name, number))
            timed('3.3.adjustment_drawing', number, lambda: adjustment_draw.run(name, number))
        return dict(result, insertion=dict(
            mode=D.D.MODE,
            all_contacts_have_certified_direction=directions['all_contacts_have_certified_direction'],
            common_directions=directions['common_directions'],
            common_representative=directions['common_representative'],
            record=str(direction_path.relative_to(C.ROOT))))

    def validate(number, adjusted):
        from step3_scheculer import verification as V
        geometry = I.read_contacts(previous_state.parent/'contacts.npz')
        validation = V.continuous_check(problem, geometry,
            out/'verification'/f'round_{number:03d}_primal_certificate.npz')
        validation.update(object=name, round=number,
                          passive_support_constraint=C.U.description(),
                          geometry_sha256=C.sha256(previous_state.parent/'contacts.npz'),
                          verification_code_sha256=C.sha256(V.__file__))
        I.save(out/'verification'/f'round_{number:03d}_continuous.json', validation)
        if validation['status'] == 'counterexample':
            path = out/'search_loads'/f'after_round_{number:03d}.json'
            problem.add_counterexample(validation['counterexample'], path)
            print(name, 'round', number, 'continuous counterexample added; returning to Step 3.1', flush=True)
        return validation

    try:
        rounds, status = iterate(
            lambda n: timed('3.1', n, lambda: score(n)),
            lambda n: timed('3.2', n, lambda: choose(n)),
            lambda n: timed('3.3', n, lambda: optimize(n)),
            lambda n, a: timed('3.validation', n, lambda: validate(n, a)))
    except Exception as error:
        I.save(out/'status.json', dict(object=name, status='search_error', complete=False,
            error_type=type(error).__name__, error=str(error)))
        raise
    contacts, mask, state_inputs = problem.load_state(previous_state)
    minimum_area = A.area_limit.MIN_AREA_FRACTION*float(problem.domain.mesh.area)
    area_valid = all(I.area([contact]) > minimum_area for contact in contacts)
    assert area_valid
    from step3_scheculer import verification as V
    from step3_scheculer import enclosure as E
    I.save_contacts(out/'final_contacts.npz', contacts)
    directions, direction_path = tracker.finish(out/'final_contacts.npz')
    verification_paths = [out/'verification'/f"round_{entry['round']:03d}_continuous.json"
                          for entry in rounds if 'continuous_validation' in entry]
    work_valid=all(A.read(name,r['round'])['verification']['adjusted']['work_volume_clearance']['passed'] for r in rounds)
    assert work_valid
    result = dict(object=name, complete=True, status=status, rounds=rounds,
        original_floor_reaction_model=C.F.description(),
        passive_support_constraint=C.U.description(),
        passive_support_no_uplift_verified=status=='continuous_contact_model_verified',
        rest_equilibrium_verified=C.gravity_check(problem.supply(contacts),problem.domain,problem.scale)['passed'],
        process_access_enforced=A.P.POLICY.ENFORCE_PROCESS_ACCESS,
        all_contact_heads_clear_of_work_volume=work_valid if A.P.POLICY.ENFORCE_PROCESS_ACCESS else None,
        contact_count=len(contacts), selected_indices=[p['candidate_index'] for p in contacts],
        minimum_contact_area_fraction=A.area_limit.MIN_AREA_FRACTION,
        minimum_contact_area_m2=minimum_area, all_contact_areas_above_minimum=area_valid,
        selected_ids=[p['candidate_id'] for p in contacts], sample_count=len(mask),
        covered_count=int(mask.sum()), covered_percent=100*float(mask.mean()),
        completion_rule='Every actual optimized head retains a common certified 3-D withdrawal direction; each load has one joint equilibrium reaction with sum(head_force_on_workpiece_z) >= 0. Loads pass AFTER optimization, then the continuous-domain certificate passes; counterexamples return to Step 3.1.',
        insertion_direction_record=str(direction_path.relative_to(C.ROOT)),
        insertion_mode=D.D.MODE,
        contact_heads_individually_insertable=directions['all_contacts_have_certified_direction'],
        common_withdrawal_directions=directions['common_directions'],
        common_withdrawal_representative=directions['common_representative'],
        common_head_withdrawal_verified=directions['all_contacts_have_certified_direction'],
        candidate_filter_records=[str(p.relative_to(C.ROOT)) for p in tracker.filter_paths],
        random_sample_count=problem.random_sample_count, validation_load_count=len(problem.counterexamples),
        continuous_domain_status='verified' if status == 'continuous_contact_model_verified' else 'not_verified',
        continuous_coverage_proved=status == 'continuous_contact_model_verified',
        physical_supports_verified=False,
        head_connections_constructed=False, support_structure_design_stage=5,
        final_state=str(previous_state.relative_to(C.ROOT)) if previous_state else None,
        round_limit=Q.MAX_CONTACTS, initial_candidate_count=int(problem.data.valid.sum()),
        elapsed_seconds=time.monotonic()-started,
        wall_timings=wall_timings,
        wall_timing_scope='3.1 includes parallel precomputation; 3.3 includes its named substeps (do not sum parent and children). Resume timings include cache reads, not the historical computation. Other scheduler work remains outside these callbacks.',
        provenance=dict(inputs=I.hashes(problem.inputs+state_inputs),
                        code=dict(A.code_hashes(), **D.code_hashes(), **I.hashes([Path(__file__), Path(Q.__file__), Path(V.__file__), Path(E.__file__)]))),
        artifacts={str(path.relative_to(out)): C.sha256(path) for path in
                   [out/'final_contacts.npz', direction_path]+tracker.filter_paths+verification_paths})
    I.save(out/'schedule.json', result)
    I.save(out/'status.json', dict(object=name, complete=True, status=status,
        continuous_domain_status=result['continuous_domain_status'], physical_supports_verified=False,
        schedule_sha256=C.sha256(out/'schedule.json')))
    print(name, 'SCHEDULER', status, len(contacts), 'contacts', result['covered_percent'], '%', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--no-draw', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse unchanged, hash-validated round outputs; repeat continuous verification.')
    args = parser.parse_args()
    results = []
    for name in args.objects or C.OBJECTS:
        if name not in C.OBJECTS:
            parser.error('objects must be A1-f, B or C5')
        results.append(run(name, draw=not args.no_draw, resume=args.resume))
    sys.exit(0 if all(r['continuous_coverage_proved'] for r in results) else 2)
