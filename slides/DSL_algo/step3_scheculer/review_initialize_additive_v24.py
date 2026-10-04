"""Audit additive-only histories, full loads, exits, and batch outcome."""
import json
from pathlib import Path

import numpy as np

from step3_scheculer import contacts as I
from step3_scheculer.initialize_additive_v24 import STAGE, SCHEMA
from step3_scheculer import initialize_cached_v19 as OLD


def main():
    original = I.OUTPUTS/'B'/OLD.STAGE/'review.json'
    baseline = json.loads(original.read_text())
    rows = []
    for case in baseline['failures']:
        name,pose = case['object'],case['pose']
        folder = I.OUTPUTS/name/STAGE/pose/'step3_scheculer'
        d = I.check_report(folder/'schedule.json')
        assert d['schema']==SCHEMA and d['mode']=='additive_fallback'
        assert d['load_count']==32768 and not d['load_subsampling']
        assert d['no_head_deletion'] and d['actual_triangle_force_generators_checked']
        old = I.check_report(I.OUTPUTS/name/OLD.STAGE/pose/'step3_scheculer/schedule.json')
        seeds = {r['trajectory']:r for r in old['trajectory_results']}
        for seed in d['seed_results']:
            original_ids = seeds[seed['seed_trajectory']]['selected_ids']
            assert seed['initial_ids']==original_ids
            assert set(original_ids)<=set(seed['selected_ids'])
            assert seed['heads']<=d['config']['max_heads']
            covered = seed['initial_covered']
            head_count = seed['initial_heads']
            for step in seed['rounds']:
                assert step['covered_before']==covered
                if step['accepted']:
                    added = step['added_ids']
                    assert len(added)==(2 if step['mode']=='demand_guided_pair_addition' else 1)
                    assert step['covered_after']>covered
                    head_count += len(added)
                    assert head_count==step['heads']
                else:
                    assert step['covered_after']==covered
                covered = step['covered_after']
            assert seed['heads']==head_count and seed['covered']==covered
        mask = np.load(folder/'coverage.npz')['mask']
        assert len(mask)==32768 and int(mask.sum())==d['result']['covered']
        assert bool(mask.all())==d['passed']
        if d['passed']:
            assert d['result']['common_direction_ids'] and d['result']['common_path_components']
        else:
            assert d['seeds_run']==30
        contacts = I.read_contacts(folder/f'final_contacts_{pose}.npz')
        assert len(contacts)==d['result']['heads']
        centers = np.array([c['center_m'] for c in contacts])
        assert all(np.linalg.norm(centers[i]-centers[j])>1e-9 for i in range(len(centers)) for j in range(i))
        row = dict(object=name,pose=pose,passed=d['passed'],heads=d['result']['heads'],
            covered=d['result']['covered'],original_covered=case['covered'],seeds_run=d['seeds_run'],
            pair_additions=sum(r['accepted'] and r['mode']=='demand_guided_pair_addition' for r in d['result']['rounds']),
            all_seed_search_seconds=sum(r['seconds'] for r in d['seed_results']))
        rows.append(row)
    passed = sum(r['passed'] for r in rows)
    out = I.OUTPUTS/'B'/STAGE
    report = dict(complete=True,original_passed=baseline['passed'],fallback_cases=len(rows),
        fallback_passed=passed,total_passed=baseline['passed']+passed,total_cases=baseline['total'],
        results=rows,full_support_constructed=False,
        provenance=dict(inputs=I.hashes([original]+[I.OUTPUTS/r['object']/STAGE/r['pose']/'step3_scheculer/schedule.json' for r in rows]),
                        code=I.hashes([Path(__file__)])))
    I.save(out/'review.json',report)
    lines = ['# Additive fallback results','',
        f"Greedy: {baseline['passed']}/{baseline['total']}; fallback rescued: {passed}/{len(rows)}; combined: {baseline['passed']+passed}/{baseline['total']}.",
        '', '| Object | Pose | Passed | Heads | Original coverage | Final coverage | Seeds tried | Search seconds |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['object']} | {r['pose']} | {'yes' if r['passed'] else 'no'} | {r['heads']} | {r['original_covered']} | {r['covered']} | {r['seeds_run']} | {r['all_seed_search_seconds']:.2f} |" for r in rows]
    lines += ['', 'All accepted fallback solutions use all original loads and retain their entire greedy seed. No complete support body was constructed. Failed bounded searches are not infeasibility proofs.']
    (out/'review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
