"""Audit saved ALL52 PSO/control provenance and masks without force replay."""
from pathlib import Path
import argparse
import json
import numpy as np
from run_pso_campaign import ROOT, BASE, POLICIES, read, save, sha, verify


def audit_policy(system, policy, experiment):
    batch = read(BASE / 'data' / experiment / 'batch.json')
    post = read(BASE / 'data' / (experiment + '_sculpted') / 'batch.json')
    assert batch['complete'] and post['complete']
    assert len(batch['results']) == len(post['results']) == 52
    assert {r['id'] for r in batch['results']} == set(batch['sets'])
    assert {r['id'] for r in post['results']} == set(batch['sets'])
    final = {r['id']: r for r in post['results']}
    masks = 0
    initialization_slots = 0
    events = 0
    pso_moves = 0
    cases = []
    for case in batch['results']:
        label = case['id']
        base = BASE / label
        initial = base / 'step4/step4.1' / experiment
        initialization = read(initial / 'population.json')
        assert initialization['population_size'] == 32
        assert not initialization['warm_start']
        assert initialization['predefined_juxtaposed_pose_ids'] == []
        slots = sorted(initial.glob('slot_*.npz'))
        assert len(slots) == 32
        for slot in slots:
            with np.load(slot) as a:
                active = a['active']
                assert np.array_equal(a['hosts'][active], active)
                assert np.allclose(a['placements'][active], np.eye(4), atol=1e-12, rtol=0)
                assert np.allclose(a['state_offsets'], 0, atol=1e-12, rtol=0)
            initialization_slots += 1
        folder = base / 'step4/step4.2' / experiment
        report = read(folder / 'data/report.json')
        assert report['passed'] == case['passed']
        assert report['load_count_per_pose'] == 32768
        assert report['original_loads_reused']
        assert not report['count_penalty_used']
        assert not report['initialization']['old_answers_used_as_start']
        statistics = report['frontier_search']
        assert statistics['population_size'] == 32
        assert statistics['continuous_proposal_policy'] == policy
        trace = read(folder / 'population32_trace.json')
        assert len(trace) <= 8
        for row in trace:
            assert row['population_size'] == len(row['events']) == 32
            assert row['maximum_trials_per_slot'] == 1 and not row['count_penalty_used']
            events += len(row['events'])
            for event in row['events']:
                assert event['adjustment_passes'] <= 1
                if event['branch'] == 'pso':
                    assert policy == 'pso'
                    detail = event.get('move')
                    # Rejected invalid proposals may have only their error record.
                    if detail:
                        assert detail['coordinate_frame'] == 'common_material'
                        assert detail['coefficients'] == dict(w=.25, c1=.25, c2=.25)
                        assert not detail['translation_noise_added']
                        assert not detail['role_and_host_changed']
                    pso_moves += 1
                if policy == 'pso':
                    assert event['branch'] != 'differential'
        for relative, digest in report['artifacts'].items():
            assert sha(folder / 'data' / relative) == digest, (label, relative)
        for relative, digest in report['provenance']['inputs'].items():
            assert sha(ROOT / relative) == digest, (label, relative)
        with np.load(folder / 'layout.npz') as a:
            active = a['active']
            native = a['native_world_original']
            rotation = np.array([a['native_world'][int(a['hosts'][k]), :3, :3] @
                                 a['placements'][k, :3, :3] for k in active])
            assert np.allclose(rotation, native[active, :3, :3], atol=1e-8, rtol=0)
        if case['passed']:
            core = report['connected_contact_core']
            assert core['connected'] and core['watertight'] and core['component_count'] == 1
            assert core['downward_resultant_passed']
            assert report['force_material_geometry_certified']
            assert not report['connection_material_added_to_core']
            assert report['base_oracle']['passed'] and report['base_oracle']['connected_passed']
            for name in report['poses']:
                with np.load(folder / (name + '_force.npz')) as a:
                    assert a['mask'].shape == (32768,) and a['mask'].all()
                    assert np.isfinite(a['supply_7d']).all()
                masks += 32768
        else:
            assert not final[label]['passed'] and final[label]['status'] == 'STEP4_FAIL'
        if final[label]['passed']:
            for stage in ['step5.2', 'step5.3']:
                saved = base / 'step5' / stage / (experiment + '_sculpted')
                data = read(saved / 'data/report.json')
                assert data['passed']
                assert data['selected_step4_experiment'] == experiment
                for relative, digest in data['artifacts'].items():
                    assert sha(saved / relative) == digest
                with np.load(saved / 'force_certificate.npz') as a:
                    for k in active:
                        assert a['mask_' + str(k)].shape == (32768,)
                        assert a['mask_' + str(k)].all()
                        masks += 32768
        cases.append(dict(id=label, step4_passed=case['passed'], final_passed=final[label]['passed'],
                          rounds=len(trace), initialization_slots=32))
    assert sum(r['pose_count'] for r in batch['results']) == 263
    return dict(passed=True, group_count=52, original_object_poses=263,
                step4_passed=batch['passed'], final_passed=post['passed'],
                initialization_slots=initialization_slots, slot_events=events,
                pso_branch_events=pso_moves, passing_saved_mask_entries=masks,
                cases=cases, original_force_solver_replayed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', default='pso_comparison_all52_20261013_01')
    args = parser.parse_args()
    system = BASE / 'data' / ('system_' + args.system)
    verify(system)
    policies = {p: audit_policy(system, p, e) for p, e in POLICIES.items()}
    images = read(system / 'previous_images.json')
    for row in images:
        assert sha(ROOT / row['archive']) == row['sha256']
    previous_step4 = read(system / 'previous_step4_images.json')
    for row in previous_step4:
        assert sha(ROOT / row['archive']) == row['sha256']
    save(system / 'final_campaign_audit.json', dict(
        passed=True, policies=policies, previous_images_unchanged=len(images),
        previous_step4_images_unchanged=len(previous_step4),
        interrupted_compute=read(system / 'interrupted_compute.json'),
        numerical_sources_unchanged=True, protected_originals_unchanged=True))
    print('ALL52 PSO/CONTROL SAVED-CERTIFICATE AND PROVENANCE AUDIT PASS', flush=True)


if __name__ == '__main__':
    main()
