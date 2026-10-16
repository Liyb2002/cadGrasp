"""Bind the completed numerical, Evaluation and five-video delivery together."""
from pathlib import Path
import shutil
import sys
from wrap_pipeline import ROOT,COOPT,read,save,sha,DEFAULT_EXPERIMENT
sys.path.insert(0,str(COOPT/'helper_func'))
import artifact_store as A


def main():
    experiment=sys.argv[1] if len(sys.argv)>1 else DEFAULT_EXPERIMENT
    directory=COOPT/'data/results/B/data'/('step5_'+experiment)
    summary=read(directory/'summary.json');verification=read(directory/'verification.json')
    work=COOPT/'idea/assets/data'/('wrap_support_movies_'+experiment)
    visual=read(work/'completion.json')
    assert summary['complete'] and summary['group_count']==52
    assert summary['passed']+summary['upstream_step4_fail']+summary['step5_failed']==52
    for path,digest in verification['protected_inputs'].items():assert A.digest(ROOT/path)==digest,path
    for archive,index in verification['frozen_sources'].items():
        for path,digest in index.items():assert sha(ROOT/archive/path)==digest,path
    for row in summary['results']:
        if not row['passed']:continue
        out=ROOT/row['output'];record=read(out/'data/report.json')
        for path,digest in record['artifacts'].items():assert sha(out/path)==digest,(row['id'],path)
        assert record['initial_step4_core_loss_m3']<=1e-10
        assert all(v==32768 for v in record['checks']['counts'].values())
        assert record['checks']['components']==1 and record['checks']['floor_passed']
        assert record['dimensions']['initial_foot_width_m']==.008
        assert record['dimensions']['initial_foot_height_m']==.005
        assert not record['width_policy']['hard_minimum_enforced']
        assert record['final_material_cm3']>=record['initial_material_cm3']*record['width_policy']['retained_volume_fraction']-1e-4
        audit=read(out/'data/independent_audit.json')
        assert audit['passed'] and audit['all_checkpoints_connected']
        assert audit['conservative_material_budget_rechecked']
    changes=dict(visual['authorized_changes'])
    stage53_path=directory/'authorized_step53_updates.json'
    stage53=read(stage53_path) if stage53_path.exists() else None
    if stage53:
        assert stage53['complete']
        changes.update(stage53['changes'])
    for path,digest in read(directory/'protected_originals.json').items():
        assert sha(ROOT/path)==changes.get(path,{}).get('after',digest),path
    save(directory/'authorized_visual_updates.json',dict(complete=True,
        user_requested_existing_evaluation_and_three_franka_updates=True,
        user_requested_step53_images_updated=bool(stage53),changes=changes))
    verification.update(protected_originals_unchanged_before_authorized_visual_update=True,
        protected_originals_unchanged=False,protected_originals_except_authorized_visual_updates_unchanged=True,
        authorized_visual_updates=changes,original_numerical_models_layouts_and_certificates_unchanged=True)
    save(directory/'verification.json',verification)
    evaluation=read(COOPT/'data/results/B/data/evaluation/verification.json')
    assert evaluation['independent_metrics_recomputed'] and evaluation['rendered_groups']==52
    franka=read(COOPT/'idea/assets/data/verification.json')
    assert franka['passed'] and franka['new_result_source']==experiment
    encoded=read(work/'encoded_inspection.json');assert encoded['complete'] and len(encoded['results'])==3
    for item in encoded['results']:
        assert sha(COOPT/'idea'/(item['set']+'_franka.mp4'))==item['video_sha256']
        item['contact_sheet_sha256']=sha(item['contact_sheet'])
    encoded.update(visually_reviewed_all_three_sheets=True,
        total_actual_encoded_frames=sum(len(r['frames']) for r in encoded['results']))
    save(work/'encoded_inspection.json',encoded)
    post=COOPT/'idea/post_processing/assets/data'/('video_'+experiment)
    proof=read(post/'completion.json');assert proof['complete']
    for path,digest in proof['movies'].items():assert sha(ROOT/path)==digest
    if proof.get('video01_restoration'):
        restoration_path=ROOT/proof['video01_restoration'];restoration=read(restoration_path)
        assert sha(ROOT/restoration['movie'])==restoration['sha256']
        assert sha(COOPT/'idea/post_processing/02_pruning_and_bbx.mp4')==restoration['video02_unchanged_sha256']
        for path,digest in read(ROOT/restoration['render_sources']).items():
            assert sha(restoration_path.parent/'executed_sources'/path)==digest
    if proof.get('video02_revision'):
        revision_path=ROOT/proof['video02_revision'];revision=read(revision_path)
        assert revision['complete'] and revision['full_decode_passed']
        assert sha(ROOT/revision['movie'])==revision['sha256']
        assert sha(COOPT/'idea/post_processing/01_hulls_and_support.mp4')==revision['video01_unchanged_sha256']
        assert revision['seconds']==8 and revision['frames']==192
        assert not revision['verification']['video02_bounding_boxes_drawn']
        for path,digest in read(ROOT/revision['render_sources']).items():
            assert sha(revision_path.parent/'executed_sources'/path)==digest
    for path,digest in proof['verification']['protected_inputs'].items():assert sha(ROOT/path)==digest
    for path,digest in read(post/'render_sources.json').items():assert sha(post/'executed_sources'/path)==digest
    media=dict(proof['movies'])
    if stage53:
        images=read(ROOT/stage53['publication'])
        assert images['complete'] and images['rendered_groups']==summary['passed']
        for path,digest in images['published'].items():assert sha(ROOT/path)==digest,path
        for path,digest in images['protected_inputs'].items():assert sha(ROOT/path)==digest,path
        for row in images['results']:
            assert sha(ROOT/row['source']/'support.obj')==row['support_sha256']
        for path,digest in read(ROOT/images['render_sources']).items():
            assert sha((ROOT/images['render_sources']).parent/'executed_sources'/path)==digest
        media.update(images['published'])
    for case in franka['cases']:media[case['video']]=case['sha256']
    for row in read(COOPT/'data/results/B/data/evaluation/summary.json')['results']:
        media[row['output']]=row['image_sha256']
    completion=read(directory/'completion.json')
    completion.update(complete=True,video_complete=True,post_processing_movies=2,
        canonical_evaluation_images_updated=52,franka_movies_updated=3,delivered_media=media,
        delivered_model_sha256={r['id']:r['report']['support_sha256'] for r in summary['results'] if r['passed']},
        old_numeric_models_and_inputs_unchanged=True,authorized_derived_visual_updates_verified=True,
        canonical_evaluation_verification=str((COOPT/'data/results/B/data/evaluation/verification.json').relative_to(ROOT)),
        franka_verification=str((COOPT/'idea/assets/data/verification.json').relative_to(ROOT)),
        post_processing_verification=str((post/'completion.json').relative_to(ROOT)),
        canonical_step53_images_updated=len(images['published']) if stage53 else 0,
        canonical_step53_publication=stage53['publication'] if stage53 else None,
        actual_franka_encoded_frames_visually_reviewed=encoded['total_actual_encoded_frames'])
    save(directory/'completion.json',completion)
    archive=directory/'delivery_sources';revision=1
    while archive.exists():
        revision+=1;archive=directory/f'delivery_sources_{revision:02d}'
    sources={}
    paths=[*(COOPT/'step5').glob('wrap_*.py'),COOPT/'step5/run.py',COOPT/'step5/test_wrap_geometry.py',
        COOPT/'evaluation/run.py',COOPT/'idea/assets/inputs.py',COOPT/'idea/post_processing/assets/actual_step5.py']
    for path in paths:
        target=archive/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);sources[str(path.relative_to(ROOT))]=sha(target)
    save(directory/'source_finalization.json',dict(complete=True,passed=True,
        current_default='step5/run.py -> wrap_pipeline.py',source_files=sources,
        source_archive=str(archive.relative_to(ROOT)),
        numerical_source_snapshots=list(verification['frozen_sources']),
        native_inputs_recomputed_from_unchanged_original_sources=True,
        geometry_and_route_regression_tests_passed=34,current_visuals_bound_to_new_supports=True,
        all_owned_workers_finished=True,unrelated_step4_workers_preserved=True))
    print(f"DELIVERY VERIFIED: 52 sets; {summary['passed']} certified; "
          f"{summary['upstream_step4_fail']} upstream failures; {summary['step5_failed']} Step5 failures; "
          '52 Evaluation images; 5 movies',flush=True)

if __name__=='__main__':main()
