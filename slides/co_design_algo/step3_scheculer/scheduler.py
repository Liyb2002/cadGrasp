"""Select and size contact patches with coupled connection/insertion witnesses."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.stage_imports import load_stage
C = load_stage('score', 'contribution')
M = load_stage('select', 'choose')
A = load_stage('optimize', 'adjust')
from step3_scheculer import contacts as I
from step3_scheculer import paths as PTH
from step3_scheculer import directions as D
from step3_scheculer import completion as Q

OUTPUT_NAME = 'step3_scheculer'

from step1.cases import pose_name


def iterate(score, choose, optimize, validate=None, *, initial_rounds=None, stop_after_round=None):
    """Completion is checked AFTER optimization, never on the candidate score."""
    rounds = copy.deepcopy(initial_rounds or [])
    selected_indices = {row['candidate_index'] for row in rounds}
    limit = Q.MAX_CONTACTS if stop_after_round is None else min(stop_after_round, Q.MAX_CONTACTS)
    while len(rounds) < limit:
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
            if not adjusted['insertion'].get('connection', {'passed': True})['passed']:
                return rounds, 'no_connection_witness_after_optimization'
        if adjusted['covered_count'] == adjusted['sample_count']:
            if validate is None:
                return rounds, 'sampled_complete'
            validation = validate(number, adjusted)
            rounds[-1]['continuous_validation'] = validation['status']
            if validation['status'] == 'verified':
                return rounds, 'continuous_contact_model_verified'
            if validation['status'] != 'counterexample':
                return rounds, 'continuous_validation_inconclusive'
    return rounds, 'contact_limit_reached' if len(rounds) == Q.MAX_CONTACTS else 'population_partial'


def run_chain(name, draw=True, resume=False, *, seed=None, top_k=5,
              sizing_sweeps=2, sizing_budget=96, first_score=None,
              prefix=None, stop_after_round=None, particle_id=None, excluded_indices=(), score_source=None, population_seed=None):
    started = time.monotonic()
    import numpy as np
    rng = np.random.default_rng(seed) if seed is not None else None
    problem = C.Problem(name)
    out = PTH.stage_folder(name, OUTPUT_NAME)
    out.mkdir(parents=True, exist_ok=True)
    I.save(out/'status.json', dict(object=name, status='running', complete=False))
    previous_state = None
    tracker = D.Tracker(problem, out)
    problem.inputs.append(tracker.catalogue_path)
    wall_timings = []
    if prefix is not None:
        # Geometry/counterexamples are inherited by reference. No ancestor scoring,
        # sizing or continuous verification is repeated during a particle advance.
        previous_state = C.ROOT/prefix['final_state'] if prefix['final_state'] else None
        tracker.actual = I.read_contacts(previous_state.parent/'contacts.npz') if previous_state else []
        if previous_state:
            tracker.current_path = previous_state.parent/'insertion_directions.json'
            tracker.current = I.check_report(tracker.current_path)
        tracker.filter_paths = [C.ROOT/p for p in prefix['candidate_filter_records']]
        for entry in prefix['rounds']:
            if entry.get('continuous_validation') == 'counterexample':
                ancestor = PTH.owner_stage(name, OUTPUT_NAME, entry['round'])
                validation = json.loads((ancestor/'verification'/f"round_{entry['round']:03d}_continuous.json").read_text())
                problem.add_counterexample(validation['counterexample'],
                    ancestor/'search_loads'/f"after_round_{entry['round']:03d}.json", write=False)
        problem.inputs.extend(C.ROOT/p for p in prefix['provenance']['inputs'])


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
        if score_source is not None:
            from step3_scheculer.smc_search import reuse_score
            return reuse_score(name, number, problem, score_source)
        if number == 1 and first_score is not None and 'source' in first_score:
            from step3_scheculer.random_search import reuse_first_score
            return reuse_first_score(name, problem, first_score['source'])
        with D.scoring_mask(problem, eligible, selected):
            result = C.run(name, number, previous_state, problem)
        if number == 1 and first_score is not None:
            first_score['source'] = PTH.folder(name, C.OUTPUT_NAME, number)
        return result

    def choose(number):
        if rng is not None:
            # Re-draw even on resume: a fresh per-chain RNG reproduces the sequence.
            return M.run(name, number, previous_state, problem, rng=rng, top_k=top_k,
                         excluded_indices=excluded_indices, proposal_seed=seed if particle_id is not None else None)
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
            if rng is not None:
                coordinates = load_stage('optimize', 'coordinate')
                result = timed('3.3.size_search', number, lambda: coordinates.run(
                    name, number, problem, sweeps=sizing_sweeps, max_evaluations=sizing_budget))
            else:
                result = timed('3.3.size_search', number, lambda: A.run(name, number, problem))
        previous_state = PTH.folder(name, A.OUTPUT_NAME, number)/'state.json'
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
            connection=directions['connection'],
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
            lambda n, a: timed('3.validation', n, lambda: validate(n, a)),
            initial_rounds=prefix['rounds'] if prefix else None, stop_after_round=stop_after_round)
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
    verification_paths = [PTH.owner_stage(name, OUTPUT_NAME, entry['round'])/'verification'/f"round_{entry['round']:03d}_continuous.json"
                          for entry in rounds if 'continuous_validation' in entry]
    work_valid=all(A.read(name,r['round'])['verification']['adjusted']['work_volume_clearance']['passed'] for r in rounds)
    assert work_valid
    result = dict(object=name, complete=True, status=status, rounds=rounds,
        search_mode='smc_particle' if particle_id is not None else 'top5' if rng is not None else 'greedy',
        search_config=dict(seed=seed, top_k=top_k, sizing_sweeps=sizing_sweeps, sizing_budget=sizing_budget),
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
        completion_rule='Actual heads have a shared certified withdrawal direction with a finite-thickness connection witness for that same direction; each load has one joint equilibrium reaction with sum(head_force_on_workpiece_z) >= 0. Loads pass AFTER optimization, then the continuous-domain certificate passes; counterexamples return to Step 3.1.',
        insertion_direction_record=str(direction_path.relative_to(C.ROOT)),
        insertion_mode=D.D.MODE,
        contact_heads_individually_insertable=directions['all_contacts_have_certified_direction'],
        common_withdrawal_directions=directions['common_directions'],
        common_withdrawal_representative=directions['common_representative'],
        common_head_withdrawal_verified=directions['all_contacts_have_certified_direction'],
        connection=directions['connection'],
        common_connected_directions=directions['connection']['directions'],
        candidate_filter_records=[str(p.relative_to(C.ROOT)) for p in tracker.filter_paths],
        random_sample_count=problem.random_sample_count, validation_load_count=len(problem.counterexamples),
        random_covered_count=int(mask[:problem.random_sample_count].sum()),
        continuous_domain_status='verified' if status == 'continuous_contact_model_verified' else 'not_verified',
        continuous_coverage_proved=status == 'continuous_contact_model_verified',
        physical_supports_verified=False,
        head_connections_constructed=False, head_connection_witness_verified=directions['connection']['passed'],
        support_structure_design_stage=6,
        final_state=str(previous_state.relative_to(C.ROOT)) if previous_state else None,
        round_limit=Q.MAX_CONTACTS, initial_candidate_count=int(problem.data.valid.sum()),
        elapsed_seconds=time.monotonic()-started,
        wall_timings=wall_timings,
        wall_timing_scope='3.1 includes parallel precomputation; 3.3 includes its named substeps (do not sum parent and children). Resume timings include cache reads, not the historical computation. Other scheduler work remains outside these callbacks.',
        provenance=dict(inputs=I.hashes(problem.inputs+state_inputs),
                        code=dict(A.code_hashes(), **D.code_hashes(), **I.hashes([Path(__file__), Path(Q.__file__), Path(V.__file__), Path(E.__file__), Path(A.__file__).with_name('coordinate.py')]))),
        artifacts={os.path.relpath(path, out): C.sha256(path) for path in
                   [out/'final_contacts.npz', direction_path]+tracker.filter_paths+verification_paths})
    if particle_id is not None:
        result['particle_id'] = particle_id
        result['population_seed'] = population_seed
        result['round_trajectories'] = {str(r['round']): (particle_id if r['round'] == stop_after_round
            else prefix['round_trajectories'][str(r['round'])]) for r in rounds}
        result['provenance']['code'].update(I.hashes([Path(__file__).with_name('smc_search.py')]))
    I.save(out/'schedule.json', result)
    I.save(out/'status.json', dict(object=name, complete=True, status=status,
        continuous_domain_status=result['continuous_domain_status'], physical_supports_verified=False,
        schedule_sha256=C.sha256(out/'schedule.json')))
    print(name, 'SCHEDULER', status, len(contacts), 'contacts', result['covered_percent'], '%', flush=True)
    return result


def run(name, draw=True, resume=False, *, search_mode='top5', trajectories=10,
        seed=0, top_k=5, sizing_sweeps=2, sizing_budget=96):
    from step3_scheculer import random_search
    config = random_search.configuration(search_mode, trajectories, seed, top_k, sizing_sweeps, sizing_budget)
    Q.archive_downstream(name)
    if search_mode == 'greedy':
        return run_chain(name, draw, resume)
    if search_mode == 'smc':
        from step3_scheculer import smc_search
        return smc_search.run(name, config, draw=draw, resume=resume)
    return random_search.run(name, config, draw=draw, resume=resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--no-draw', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse unchanged, hash-validated round outputs; repeat continuous verification.')
    parser.add_argument('--search-mode', choices=['top5', 'greedy', 'smc'], default='top5')
    parser.add_argument('--trajectories', type=int, default=10)
    parser.add_argument('--seed', '--search-seed', type=int, default=0)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--sizing-sweeps', type=int, default=2)
    parser.add_argument('--sizing-budget', type=int, default=96)
    args = parser.parse_args()
    results = []
    for name in args.objects or C.OBJECTS:
        if name not in C.OBJECTS:
            parser.error('objects must be active in objects/cases.json')
        results.append(run(name, draw=not args.no_draw, resume=args.resume,
                           search_mode=args.search_mode, trajectories=args.trajectories,
                           seed=args.seed, top_k=args.top_k, sizing_sweeps=args.sizing_sweeps,
                           sizing_budget=args.sizing_budget))
    sys.exit(0 if all(r['continuous_coverage_proved'] for r in results) else 2)
