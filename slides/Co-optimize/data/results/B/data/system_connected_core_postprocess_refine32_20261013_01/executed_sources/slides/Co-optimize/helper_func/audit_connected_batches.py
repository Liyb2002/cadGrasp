"""Audit saved connected-core results and exact execution versions without replay."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from PIL import Image
from run_dynamic_postprocess import ROOT,COOPT,BASE,sha,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step4-experiment',required=True);p.add_argument('--postprocess-experiment',required=True)
    p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args()
    source=json.loads((BASE/'data'/args.step4_experiment/'batch.json').read_text())
    post=json.loads((BASE/'data'/args.postprocess_experiment/'resolved_batch.json').read_text())
    system=BASE/'data'/('system_'+args.postprocess_experiment)
    evaluation=json.loads((system/'evaluation_results.json').read_text())
    geometry=json.loads((system/'actual_solid_geometry_audit.json').read_text())
    assert geometry['passed']and geometry['groups']==sum(r['step53_passed']for r in evaluation['cases'])
    for row in geometry['cases']:
        assert row['passed']
        for path,digest in row['artifacts'].items():assert sha(ROOT/path)==digest
    complete=source['complete']and post['complete']and evaluation['complete']
    assert complete or args.allow_incomplete
    assert len(source['results'])==len(post['results'])==evaluation['groups']==52
    assert sum(r['pose_count']for r in source['results'])==263
    versions=set();archives=0;frozen_files=0
    for directory in (BASE/'data').glob('system_connected_core_postprocess*'):
        executions=list(directory.glob('execution*.json'))+[directory/'connected_policy_source_transition.json']
        for execution in executions:
            if not execution.exists():continue
            for code in json.loads(execution.read_text()).get('executed_sources',[]):
                assert sha(ROOT/code['archive'])==code['sha256']
                versions.add((code['path'],code['sha256']));archives+=1
        for code in (directory/'source_versions').rglob('*.py'):
            digest=sha(code);assert code.parent.name==digest
            if code.name=='prune_support.py':versions.add(('slides/Co-optimize/step5.2/'+code.name,digest))
            elif code.name=='flat_tapered_base.py':versions.add(('slides/Co-optimize/step5.3/'+code.name,digest))
    for name in ['connected_core_all52_20261012_05','connected_core_all52_20261012_06','connected_core_all52_20261012_08','connected_core_all52_20261013_09','connected_core_all52_20261013_10']:
        for code in json.loads((BASE/'data'/('system_'+name)/'frozen_sources.json').read_text())['files']:
            assert sha(ROOT/code['frozen'])==code['frozen_sha256'];frozen_files+=1
    masks=0;cores=0;finals=0;cases=[];initial_slots=0;slot_events=0
    for row in evaluation['cases']:
        label=row['id'];case=next(r for r in source['results']if r['id']==label)
        folder=BASE/label/'step4/step4.2'/case['experiment'];report=json.loads((folder/'report.json').read_text())
        initial=json.loads((BASE/label/'step4/step4.1'/case['experiment']/'population.json').read_text())
        size=initial['population_size'];assert size==case.get('frontier_width',32)
        assert size in [32,48]
        assert not initial['warm_start']and not initial['predefined_juxtaposed_pose_ids']
        for slot in range(size):
            with np.load(BASE/label/'step4/step4.1'/case['experiment']/f'slot_{slot:02d}.npz')as z:
                np.testing.assert_array_equal(z['hosts'],np.arange(case['pose_count']))
                np.testing.assert_array_equal(z['placements'],np.repeat(np.eye(4)[None],case['pose_count'],axis=0))
        rounds=json.loads((folder/'population32_trace.json').read_text())
        for iteration in rounds:
            assert iteration['population_size']==size and len(iteration['events'])==size
            assert iteration['maximum_trials_per_slot']==1 and not iteration['count_penalty_used']
            slot_events+=size
        initial_slots+=size
        if case['passed']:
            assert report['passed']and report['step4_contact_core_connected']
            core=report['connected_contact_core'];assert core['connected']and core['component_count']==1
            assert not core['connection_material_added']and core['added_material_cm3']<=1e-4
            assert report['force_material_geometry_certified']
            assert report['force_model']['mode']=='connected_massless_support_no_uplift'
            with np.load(folder/'layout.npz')as z:
                for k in z['active']:
                    U=z['native_world'][z['hosts'][k]]@z['placements'][k]
                    assert np.allclose(U[:3,:3],z['native_world_original'][k,:3,:3],atol=1e-10,rtol=0)
            for name,count in case['counts'].items():
                with np.load(folder/(name+'_force.npz'))as z:
                    assert count==32768 and z['mask'].shape==(32768,)and z['mask'].all();masks+=32768
            cores+=1
        assert row['reuse']+row['juxtaposed']==row['object_pose_count']
        if row['step52_passed']:
            prune=json.loads((ROOT/row['step52_source']/'data/report.json').read_text())
            assert prune['passed']and prune['pruning']['connected']and prune['pruning']['component_count']==1
            assert prune['selected_step4_experiment']==case['experiment']
            assert prune['pruning']['added_material_cm3']<=1e-4
            assert row['pruned_bbx_l']<=row['step4_bbx_l']+1e-8
            for name,h in prune['artifacts'].items():assert sha(ROOT/row['step52_source']/name)==h
            for name,h in prune['provenance'].get('code',{}).items():assert (name,h)in versions,(name,h)
            with np.load(ROOT/row['step52_source']/'force_certificate.npz')as z:
                keys=[k for k in z.files if k.startswith('mask_')]
                assert len(keys)==row['object_pose_count']and all(z[k].shape==(32768,)and z[k].all()for k in keys)
        if row['step53_passed']:
            geometry_row=next(r for r in geometry['cases']if r['id']==label)
            assert geometry_row['step52_source']==row['step52_source']and geometry_row['step53_source']==row['step53_source']
            final=json.loads((ROOT/row['step53_source']/'data/report.json').read_text())
            assert final['passed']and final['selected_step4_experiment']==case['experiment']
            assert final['construction']['connected_core_policy']
            assert final['step52_source']==row['step52_source']
            check=final['construction']['serialized_export_verification']
            assert check['passed']and check['history'][-1]['geometry_passed']and check['history'][-1]['force_passed']
            assert len(final['force_counts'])==row['object_pose_count']and all(v==32768 for v in final['force_counts'].values())
            for name,h in final['provenance'].get('code',{}).items():assert (name,h)in versions,(name,h)
            finals+=1
        else:assert row.get('final_bbx_l')is None
        cases.append(dict(id=label,step4=case['passed'],step52=row['step52_passed'],step53=row['step53_passed']))
    archive=json.loads((BASE/'data/system_connected_core_all52_20261012_01/previous_public_images.json').read_text())
    assert all(sha(ROOT/r['archive'])==r['sha256']for r in archive)
    protected=json.loads((BASE/'data/system_dynamic_postprocess_all52_20261012_01/protected_originals.json').read_text())
    changed=[path for path,h in protected.items()if not(ROOT/path).is_file()or sha(ROOT/path)!=h]
    assert not changed
    step4_images=json.loads((BASE/'data'/('system_'+args.step4_experiment)/'step4_results.json').read_text())['pictures']
    for picture in evaluation['pictures']+step4_images:
        path=ROOT/picture['path'];assert sha(path)==picture['sha256']
        with Image.open(path)as image:image.verify()
    with(BASE/('evaluation_'+args.postprocess_experiment+'_poses.csv')).open()as f:roles=list(csv.DictReader(f))
    assert len(roles)==263 and len({(r['set'],r['pose'])for r in roles})==263
    html=list((COOPT/'output/B').rglob('*.html'));assert not html
    result=dict(passed=True,complete=complete,groups=52,object_poses=263,connected_step4_cores=cores,
        final_certified_groups=finals,step4_original_mask_entries=masks,
        protected_numerical_files=len(protected),changed_numerical_files=changed,
        archived_original_pngs=len(archive),verified_current_pngs=len(evaluation['pictures'])+len(step4_images),
        execution_source_archives_verified=archives,immutable_frozen_files_verified=frozen_files,
        registered_initial_slots_verified=initial_slots,fixed_slot_events_verified=slot_events,
        no_html=True,force_solver_run=False,boolean_run=True,cases=cases,
        actual_serialized_solid_groups_verified=geometry['groups'],
        source_version_policy='per-case immutable experiments; explicit source changes retained in campaign version archives')
    filename='saved_artifact_audit.json'if complete else'saved_artifact_audit_partial.json'
    save(system/filename,result)
    save(BASE/'data'/('system_'+args.step4_experiment)/filename,result)
    print('CONNECTED AUDIT',json.dumps({k:v for k,v in result.items()if k!='cases'}))


if __name__=='__main__':main()
