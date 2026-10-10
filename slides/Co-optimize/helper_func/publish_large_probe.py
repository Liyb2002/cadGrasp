"""Publish an already fully checked own Direction probe without another Boolean."""
import _bootstrap
import argparse
import shutil
import time
from co_common import *
from run_all import output_root
from run_large_pose_sets import groups,run_case as finish_case,record_phase
from whole_pipeline import stage_record,SCHEMA
from whole_search.model import Model
from whole_search.fast_search import FastModel
from whole_search.exact_worker import read_result
from whole_search.search import failed_count
from whole_search.reuse_first import registered,juxtaposed_count
from whole_step4_render import render_search


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case')
    args=parser.parse_args();group=next(g for g in groups() if args.case in [g['id'],g['source_case']])
    base=output_root('B')/group['id'];out=base/'step4/step4.2'
    if (out/'data/report.json').exists():
        I.check_report(out/'data/report.json');print('Existing feasible baseline retained');return
    # Only publish after the original case writer has completed.
    for f in out.glob('data/continuation_*.json'):
        assert json.loads(f.read_text()).get('complete'),'Original continuation is still running'
    checked=[]
    for file in out.glob('direction_probes/*/report.json'):
        proof=json.loads(file.read_text())
        if not proof.get('complete'):continue
        for attempt in proof.get('actual_attempts',[]):
            if attempt['passed']:checked.append((attempt['volume_cm3'],file,attempt))
    if not checked:raise RuntimeError('No fully checked own Direction probe')
    _,file,attempt=min(checked,key=lambda r:r[0]);proof=json.loads(file.read_text())
    assert proof['poses']==group['poses'] and proof['source_layout_sha256']==I.sha256(out/'sampled_layout.npz')
    prefix=file.parent/f'result_{attempt["probe_index"]:02d}'
    actual=read_result(prefix,1)
    assert failed_count(actual)==0 and all(r['passed'] for r in actual['actual_work_surface_checks'])
    for key in ['nominal_sweep_overlap_m3','body_overlap_m3','padded_overlap_outside_contact_cores_m3',
                'work_band_overlap_m3','exported_boundary_max_exclusion_overlap_m3','endpoint_overlap_m3']:
        assert actual['diagnostics'][key]<=1e-10
    archive=base/'step4/_history'/('before_checked_probe_'+time.strftime('%Y%m%d_%H%M%S'))
    shutil.copytree(out,archive)
    source=base/'step3/step3.1/data/report.json';model=FastModel(group['poses'],'B',initialization_report=source)
    initial=I.check_report(base/'step4/step4.1/data/report.json')
    sampled=json.loads((out/'search_report.json').read_text());layout=actual['layout']
    process=json.loads((out/'process.json').read_text());index=len(process)
    np.savez_compressed(out/f'process_states/{index:03d}.npz',placements=layout.placements,
        directions=layout.directions,hosts=layout.hosts,active=np.array(layout.active))
    process.append(dict(index=index,phase='checked_own_direction_probe',layout=f'process_states/{index:03d}.npz',
        serial=1,active_poses=group['poses'],counts={p:32768 for p in group['poses']},failed_load_count=0,
        estimated_volume_cm3=actual['volume_cm3'],volume_epoch=-1,changes=[],
        rotating_reuse_pose_count=sum(registered(layout,k) for k in layout.active),
        juxtaposed_pose_count=juxtaposed_count(layout),geometry_verified=True,elapsed_seconds=0))
    save(out/'process.json',process)
    extra={k:sampled[k] for k in ['initialization','initial_counts','events','sample_evaluations','timing','search_seconds'] if k in sampled}
    extra.update(strategy='whole',policy='own direction probe after separately recorded continuation',
        final_acceptance_run=True,exact_evaluations_during_search=0,status='pass',passed=True,
        rotating_reuse_pose_count=sum(registered(layout,k) for k in layout.active),
        juxtaposed_pose_count=juxtaposed_count(layout),step41_volume_cm3=initial['volume_cm3'],step41_counts=initial['counts'],
        fixture_placement_count_is_not_cost=True,all_pose_juxtapose_disabled=True,separated_fallback_used=False,
        validation_continuation=dict(probe_report=str(file.relative_to(base)),
            additional_sampled_candidates=len(proof['candidates']),additional_actual_candidates=len(proof['actual_attempts']),
            probe_index=attempt['probe_index'],probe=proof['candidates'][attempt['probe_index']],
            publication_recomputed_geometry=False,publication_source_was_already_checked=True))
    report=Model.save(model,actual,out,extra)
    report['provenance']['inputs'].update(I.hashes([file,prefix.with_suffix('.npz'),prefix.with_suffix('.json')]))
    report['provenance']['code'].update(I.hashes([Path(__file__),Path(__file__).with_name('probe_large_directions.py')]))
    render_search(model,out,report);stage_record(out,report,group,'step4.2')
    save(base/'step4/data/report.json',dict(complete=True,schema=SCHEMA,id=group['id'],poses=group['poses'],
        status='pass',passed=True,final_volume_cm3=report['volume_cm3'],
        separately_recorded_search_continuation=True,full_fixture_accepted=False))
    (base/'step4/data/failure.json').unlink(missing_ok=True)
    record_phase(group,'checked_direction_probe',dict(status='pass',passed=True,volume_cm3=report['volume_cm3']))
    print('PUBLISHED_CHECKED_PROBE',group['id'],report['volume_cm3'],flush=True)
    print('FINISHED',finish_case((group,dict(methods=['greedy','beam'],audit=True))),flush=True)


if __name__=='__main__':main()
