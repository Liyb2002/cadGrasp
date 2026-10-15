"""Inspect saved masks and search records without rerunning physical acceptance."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    directory = Path(args.run).resolve()
    payload = read(directory/'payload.json')
    out = directory/'B'/payload['id']/'step4/step4.2'/payload['options']['output_name']
    report = read(out/'report.json')
    for relative, expected in report['artifacts'].items():
        if hashlib.sha256((out/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Saved artifact changed: '+relative)
    trace = read(out/'population32_trace.json')
    initial = read(out/'initial_population_summary.json')
    events = [event for row in trace for event in row['events']]
    with np.load(out/'layout.npz') as saved:
        layout = {key:saved[key].copy() for key in saved.files}
    original_centers = np.array([read(ROOT/'objects/B/poses'/pose/'needs.json')
        ['frame']['moment_origin_m'] for pose in report['poses']])
    local_center = np.linalg.inv(layout['native_world_original'][0])@np.r_[original_centers[0],1.]
    centers = np.array([(layout['native_world'][int(layout['hosts'][k])]@layout['placements'][k]@local_center)[:3]
        for k in range(len(report['poses']))])
    with np.load(ROOT/'slides/Co-optimize/data/results/B'/payload['id']/'step4/step4.1/layout.npz') as saved:
        initial_layout = {key:saved[key].copy() for key in saved.files}
    original_exit = np.array([layout['native_world_original'][int(initial_layout['hosts'][k]),:3,:3]@
        initial_layout['directions'][k] for k in range(len(report['poses']))])
    final_exit = np.array([layout['native_world'][int(layout['hosts'][k]),:3,:3]@layout['directions'][k]
        for k in range(len(report['poses']))])
    direction_norm = np.linalg.norm(final_exit-original_exit,axis=1).max()
    rows = []
    for pose in report['poses']:
        with np.load(out/(pose+'_force.npz')) as saved:
            mask, rays = saved['mask'], saved['supply_7d']
        if mask.shape != (32768,):
            raise ValueError('Original demand mask is incomplete')
        if int(mask.sum()) != report['counts'][pose]:
            raise ValueError('Saved force mask differs from the report')
        failed = np.flatnonzero(~mask)
        row = dict(pose=pose, passed=int(mask.sum()), failed=len(failed), supply_rays=len(rays),
            failure_fraction=float(len(failed)/len(mask)))
        if len(failed):
            domain = read(ROOT/'objects/B/poses'/pose/'needs.json')
            samples = read(ROOT/'objects/B/poses'/pose/'samples.json')
            extent = np.ptp(np.asarray(domain['geometry']['vertices_m']), axis=0).max()
            target = np.r_[np.asarray(samples['need_wrench'][int(failed[0])])*np.r_[np.ones(3),np.ones(3)/extent],0.]
            # Farkas witness for this saved cone only. This is a diagnostic;
            # it neither alters masks nor replaces the original acceptance.
            proof = linprog(target, A_ub=-rays, b_ub=np.zeros(len(rays)),
                bounds=[(-1.,1.)]*7, method='highs')
            if proof.success:
                minimum = float(np.min(rays@proof.x))
                separation = float(target@proof.x)
                row['saved_cone_separation'] = dict(sample_index=int(failed[0]),
                    certified=minimum>=-1e-9 and separation < -1e-8,
                    minimum_ray_dot=minimum, demand_dot=separation, dual=proof.x.tolist(),
                    target_scaled_7d=target.tolist(), physical_acceptance_changed=False)
        rows.append(row)
    candidates = initial + [event[key] for event in events for key in ('raw','trial','after') if key in event]
    branches = {branch:sum(event['branch']==branch for event in events)
                for branch in ('discrete','pso','differential','hold')}
    adjustments = sum(event.get('adjustment_passes',0) for event in events)
    if payload['mode']=='juxtapose_only' and (branches['pso'] or branches['differential'] or adjustments):
        raise ValueError('The strict ablation executed a continuous operation')
    diagnostic = dict(complete=True, mode=payload['mode'], set=payload['id'],
        step4_passed=report['passed'], force_passed=report['force_passed'],
        original_world_centers_m=original_centers.tolist(), final_world_centers_m=centers.tolist(),
        maximum_world_center_displacement_m=float(np.linalg.norm(centers-original_centers,axis=1).max()),
        maximum_world_xy_center_radius_m=float(np.linalg.norm(centers[:,:2],axis=1).max()),
        maximum_fixture_installation_offset_m=float(np.linalg.norm(layout['state_offsets'],axis=1).max()),
        maximum_world_exit_direction_change_degrees=float(np.degrees(2*np.arcsin(min(1.,direction_norm/2.)))),
        counts=rows, total_failed=sum(row['failed'] for row in rows), total_demands=32768*len(rows),
        full_force_passed_poses=sum(row['failed']==0 for row in rows),
        initial_failed=min(row['failed_demands'] for row in initial),
        best_observed_failed=min(row['failed_demands'] for row in candidates),
        best_observed_candidate=min(candidates,key=lambda row:row['failed_demands']),
        distinct_candidate_layouts=len({row['id'] for row in candidates}),
        rounds=len(trace), branch_counts=branches, local_adjustment_passes=adjustments,
        accepted=sum(event.get('accepted',False) for event in events),
        raw_oracle_status_counts=dict(Counter(event['raw']['oracle'] for event in events if 'raw' in event)),
        candidate_rejections=[dict(round=row['iteration'],slot=event['slot'],reason=event['error'])
            for row in trace for event in row['events'] if 'error' in event],
        oracle={key:report['base_oracle'].get(key) for key in
                ('status','passed','enclosure_passed','connected_passed','gap_m')},
        actual_contact_certification_run=report.get('force_material_geometry_certified') is not None,
        force_contact_representation=report.get('force_contact_representation',report['acceptance']),
        all_original_demands_reused=True, physical_acceptance_replayed=False,
        saved_artifact_hashes_verified=True, diagnostic_separation_lp_run=True,
        source_run=str(directory.relative_to(ROOT)))
    (directory/'diagnosis.json').write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    progress = [diagnostic['initial_failed']] + [min(row['failed_demands'] for row in r['retained']) for r in trace]
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    axes[0].plot(range(len(progress)),progress,marker='o')
    axes[0].set(xlabel='Round',ylabel='Unmet original demands',title='Best retained force coverage')
    axes[0].grid(alpha=.2)
    axes[1].bar([row['pose'].replace('pose_','') for row in rows], [row['failed'] for row in rows])
    axes[1].set(xlabel='Pose',ylabel='Unmet demands / 32,768',title='Selected layout: failures by pose')
    fig.suptitle(payload['mode']+': '+('PASS' if report['passed'] else 'FAIL'))
    fig.savefig(directory/'failure_analysis.png',dpi=160)
    plt.close(fig)
    print(json.dumps(diagnostic,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
