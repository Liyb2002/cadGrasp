"""Verify saved masks, layouts, publications and original bytes without replay."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from PIL import Image
from run_dynamic_postprocess import ROOT, BASE, COOPT, sha, save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('experiment')
    args=parser.parse_args();system=BASE/'data'/('system_'+args.experiment)
    summary=json.loads((system/'evaluation_results.json').read_text())
    batch=json.loads((BASE/'data'/args.experiment/'resolved_batch.json').read_text())
    assert summary['complete']and batch['complete']and summary['groups']==52
    cases=summary['cases'];assert len({r['id']for r in cases})==52
    assert sum(r['object_pose_count']for r in cases)==263
    masks=0;floor_points=0;certified=[]
    for row in cases:
        assert row['reuse']+row['juxtaposed']==row['object_pose_count']
        assert len(row['reuse_members'])==row['reuse']and len(row['juxtaposed_members'])==row['juxtaposed']
        assert sum(s['pose_count']for s in row['members'])==row['object_pose_count']
        assert len(row['members'])==row['support_pose_count']
        if row['step51_passed']:floor_points+=row['object_pose_count']*32768
        if row['step52_passed']:
            directory=ROOT/row['step52_source'];report=json.loads((directory/'data/report.json').read_text())
            assert report['passed']and report['original_demand_count_per_pose']==32768
            assert report['selected_step4_experiment']==row['source_experiment']
            assert not report['layout_changed']and not report['step4_rerun']
            for name,h in report['artifacts'].items():assert sha(directory/name)==h
            with np.load(directory/'force_certificate.npz')as proof:
                names=[name for name in proof.files if name.startswith('mask_')]
                assert len(names)==row['object_pose_count']
                for name in names:
                    assert proof[name].shape==(32768,)and proof[name].all();masks+=32768
        if row['step53_passed']:
            directory=ROOT/row['step53_source'];report=json.loads((directory/'data/report.json').read_text())
            assert report['passed']and report['step52_source']==row['step52_source']
            assert report['selected_step4_experiment']==row['source_experiment']and not report['step4_rerun']
            assert len(report['force_counts'])==row['object_pose_count']and all(n==32768 for n in report['force_counts'].values())
            check=report['construction']['serialized_export_verification']
            assert check['passed']and check['history'][-1]['force_passed']and check['history'][-1]['geometry_passed']
            assert abs(row['material_delta_cm3']-(row['final_material_cm3']-row['step4_material_cm3']))<1e-8
            assert abs(row['bbox_delta_l']-(row['final_bbx_l']-row['step4_bbx_l']))<1e-8
            certified.append(row['id'])
        else:
            assert row.get('final_material_cm3')is None and row.get('final_bbx_l')is None
            assert not any((COOPT/'output/B'/row['id']/'step5.3').glob('pose_*.png'))
    csv_path=BASE/('evaluation_'+args.experiment+'_poses.csv')
    with csv_path.open()as stream:poses=list(csv.DictReader(stream))
    assert len(poses)==263 and len({(r['set'],r['pose'])for r in poses})==263
    assert sum(r['final_passed']=='True'for r in poses)==summary['passed_totals']['object_pose_count']
    for picture in summary['pictures']:
        path=ROOT/picture['path'];assert sha(path)==picture['sha256']
        with Image.open(path)as im:im.verify()
    protected=json.loads((system/'protected_originals.json').read_text())
    changed=[p for p,h in protected.items()if not (ROOT/p).is_file()or sha(ROOT/p)!=h]
    archive=json.loads((system/'previous_public_images.json').read_text())
    changed_archive=[r['archive']for r in archive if sha(ROOT/r['archive'])!=r['sha256']]
    code=json.loads((system/'execution.json').read_text())['executed_sources']
    changed_code=[r['path']for r in code if sha(ROOT/r['path'])!=r['sha256']or sha(ROOT/r['archive'])!=r['sha256']]
    old_step4=json.loads((BASE/'data/system_population_soft_all52_20261012_final/step4_results.json').read_text())
    if not any(row.get('connected_core_policy')for row in cases):
        assert all(sha(ROOT/r['path'])==r['sha256']for r in old_step4['pictures'])
    else:
        for row in cases:
            if row['step52_passed']:
                assert row['step4_core_components']==1
                assert row['pruned_bbx_l']<=row['step4_bbx_l']+1e-8
                report=json.loads((ROOT/row['step52_source']/'data/report.json').read_text())
                assert report['pruning']['connected']and report['pruning']['added_material_cm3']<=1e-4
    html=list((COOPT/'output/B').rglob('*.html'))
    result=dict(passed=not changed and not changed_archive and not changed_code and not html,
        groups=52,object_poses=263,step51_passed=summary['step51_passed'],step52_passed=summary['step52_passed'],
        step53_passed=len(certified),step51_original_points=floor_points,certified_original_mask_entries=masks,
        passed_object_poses=summary['passed_totals']['object_pose_count'],
        passed_support_poses=summary['passed_totals']['support_pose_count'],
        verified_pngs=len(summary['pictures']),protected_files=len(protected),changed_inputs=changed,
        archived_pngs=len(archive),changed_archived_pngs=changed_archive,
        original_step4_pngs_unchanged=len(old_step4['pictures']),executed_sources=len(code),changed_code=changed_code,
        html_files=[str(p.relative_to(ROOT))for p in html],force_solver_run=False,boolean_run=False)
    save(system/'saved_artifact_audit.json',result);assert result['passed'],result
    print('POSTPROCESS AUDIT',json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
