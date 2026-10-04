"""Audit completed initialization evidence and compare with the old greedy search."""
from pathlib import Path
import json,html
import numpy as np
from step3_scheculer import contacts as I
from step3_scheculer.initialize_gpu_v12 import SCHEMA,AREAS

def main():
    out=I.OUTPUTS/'B'/'independent_poses_gpu_v12';rows=[];reports=[]
    for i in range(1,21):
        pose=f'pose_{i}';path=out/pose/'step3_scheculer/schedule.json';d=I.check_report(path)
        assert d['schema']==SCHEMA and d['independent'] and not d['cross_pose_constraints'] and not d['shared_heads']
        assert not d['complete_fixture_verified'] and not d['force_subsampling'] and not d['forced_upward_exit']
        assert d['max_heads_per_pose']==6 and d['top_k']==10 and d['chain_budget']==30 and d['load_count']==32768
        assert d['backend']['device']=='cuda' and d['passive_support_constraint']['enforced']
        r=d['result'];assert 0<r['heads']<=6
        contacts=I.read_contacts(path.parent/f'final_contacts_{pose}.npz')
        assert len(contacts)==r['heads'] and [c['candidate_id'] for c in contacts]==r['selected_ids']
        for head in d['heads']:assert abs(head['area_fraction']/r['area_fraction']-1)<=1e-4+1e-12
        winner=path.parent/f"trajectory_{r['trajectory']:02d}"
        saved=json.loads((winner/'report.json').read_text());assert saved==r
        with np.load(winner/'coverage.npz') as z:
            assert len(z['mask'])==32768 and int(z['mask'].sum())==r['covered']
            if r['passed']:assert z['mask'].all() and r['exit_check']['clear']
        assert I.sha256(winner/'contacts.npz')==I.sha256(path.parent/f'final_contacts_{pose}.npz')
        for trajectory in d['trajectory_results']:
            assert trajectory['heads']<=6 and trajectory['area_fraction'] in AREAS
            for step in trajectory['rounds']:
                assert len(step['top10'])<=10 and len(step['top10'])==len(step['probabilities'])
                assert abs(sum(step['probabilities'])-1)<1e-12
                assert step['selected_id'] in [c['id'] for c in step['top10']]
        old=json.loads((I.OUTPUTS/'B'/'independent_poses'/pose/'step3_scheculer/schedule.json').read_text())['result']
        families=[dict(area_fraction=a,attempts=sum(t['area_fraction']==a for t in d['trajectory_results']),passed=any(t['area_fraction']==a and t['passed'] for t in d['trajectory_results'])) for a in AREAS]
        rows.append(dict(pose=pose,old_passed=old['passed'],old_covered=old['covered_counts'][0],passed=r['passed'],covered=r['covered'],heads=r['heads'],area_fraction=r['area_fraction'],exit=r['object_exit_direction_world'],chains_run=d['chains_run'],seconds=d['seconds'],families=families,numerically_unresolved_proposals=d['numerically_unresolved_proposal_count']))
        reports.append(path)
    lines=['# B independent initialization comparison','', 'All 20 saved poses, unchanged original 32,768 loads per pose. New initialization: up to six heads, top10 random greedy selection, 30-trajectory budget divided among 0.5%, 1%, 2% area families. CUDA batches LP certificates; original CPU LP checks the selected final solution. Local contact/exit evidence only; full support is not constructed here.','', '| Pose | Old search | Old coverage | New search | Heads | Selected area/head | Trajectories run | Search seconds |','|---|---|---:|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| [{r['pose']}]({r['pose']}/heads.png) | {'PASS' if r['old_passed'] else 'not found'} | {r['old_covered']}/32768 | {'PASS' if r['passed'] else 'not found'} | {r['heads']} | {r['area_fraction']*100:g}% | {r['chains_run']} | {r['seconds']:.1f} |")
    lines+=['', 'Each area family stops after its first successful trajectory. Thirty trajectories is a budget, not a requirement to run all thirty after finding initialization. Search failure is not an infeasibility proof.','']
    (out/'comparison.md').write_text('\n'.join(lines))
    page=['<!doctype html><meta charset="utf-8"><title>Independent initialization</title><style>body{font-family:sans-serif;margin:24px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}img{width:100%}article{border:1px solid #ddd;padding:8px}</style><h1>B: independent initialization</h1><p>Six heads maximum, three area families, top10 sampling, all original loads. Local contacts; no complete support body.</p><div class="grid">']
    for r in rows:page.append(f'<article><h2>{r["pose"]}: {"PASS" if r["passed"] else "not found"}</h2><a href="{r["pose"]}/heads.png"><img src="{r["pose"]}/heads.png"></a><p>{r["heads"]} heads; {r["area_fraction"]*100:g}% area/head; {r["covered"]}/32768 loads</p></article>')
    page.append('</div>');(out/'index.html').write_text('\n'.join(page))
    report=dict(complete=True,pose_count=len(rows),old_passed_count=sum(r['old_passed'] for r in rows),passed_count=sum(r['passed'] for r in rows),poses=rows,
        acceptance='All original loads, shared no-uplift, own work/floor and local continuous exit; complete fixture not constructed',
        provenance=dict(inputs=I.hashes(reports),code=I.hashes([Path(__file__)])),artifacts={n:I.sha256(out/n) for n in ('comparison.md','index.html')})
    I.save(out/'review.json',report);I.check_report(out/'review.json');print('AUDIT',report['old_passed_count'],'->',report['passed_count'],'/',len(rows),flush=True)

if __name__=='__main__':main()
