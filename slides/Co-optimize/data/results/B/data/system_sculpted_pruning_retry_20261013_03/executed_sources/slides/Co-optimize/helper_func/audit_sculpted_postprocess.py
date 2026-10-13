"""Check compact fillets, exact BBX, nonincreasing material and original certificates."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from PIL import Image
from run_dynamic_postprocess import ROOT,COOPT,BASE,sha,save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True)
    args=p.parse_args();system=BASE/'data'/('system_'+args.experiment)
    old=json.loads((system/'previous_evaluation.json').read_text())
    new=json.loads((system/'evaluation_results.json').read_text())
    geometry=json.loads((system/'actual_solid_geometry_audit.json').read_text())
    assert old['complete']and new['complete']and geometry['complete']and geometry['passed']and geometry['groups']==52
    previous={r['id']:r for r in old['cases']};assert len(new['cases'])==len(previous)==52
    protected=json.loads((system/'protected_originals.json').read_text())
    changed=[path for path,h in protected.items()if not(ROOT/path).is_file()or sha(ROOT/path)!=h]
    assert not changed,changed[:3]
    archive=json.loads((system/'previous_public_images.json').read_text())
    for r in archive:
        assert sha(ROOT/r['archive'])==r['sha256']
        if '/step4.2/'in r['path']:assert sha(ROOT/r['path'])==r['sha256']
    for code in json.loads((system/'execution.json').read_text())['executed_sources']:
        assert sha(ROOT/code['archive'])==code['sha256']
    sys.path[:0]=[str(COOPT/x)for x in ['step5.2','step5.3','helper_func','evaluation']]
    from deterministic_base import solid,volume
    from metrics import static_bounds
    import trimesh
    body=trimesh.load(ROOT/'objects/B/mesh.stl',force='mesh',process=False)
    rows=[];masks=0;radii={}
    for row in new['cases']:
        before=previous[row['id']];assert row['step52_passed']and row['step53_passed']
        assert row['source_experiment']==before['source_experiment']
        for key in ['object_pose_count','support_pose_count','reuse','juxtaposed','step51_source']:
            assert row[key]==before[key],(row['id'],key)
        source=BASE/row['id']/'step4/step4.2'/row['source_experiment']
        with np.load(source/'layout.npz')as z:layout={k:z[k].copy()for k in z.files}
        pr=ROOT/row['step52_source'];fr=ROOT/row['step53_source']
        p_report=json.loads((pr/'data/report.json').read_text());f_report=json.loads((fr/'data/report.json').read_text())
        assert p_report['passed']and f_report['passed']and p_report['pruning_style']=='sculpted'
        assert p_report['pruning']['locked_bbx_unchanged']and not p_report['pruning']['grid_cell_deletion_used']
        assert f_report['construction']['original_core_not_restored']and f_report['construction']['locked_final_bbx_unchanged']
        assert f_report['construction']['connected_core_policy']and f_report['step52_source']==row['step52_source']
        for report in [p_report,f_report]:
            for path,h in report['provenance']['inputs'].items():assert sha(ROOT/path)==h
            for path,h in report['provenance']['code'].items():
                assert sha(system/'executed_sources'/path)==h
        with np.load(fr/'force_certificate.npz')as z:
            for k in layout['active']:assert z['mask_'+str(k)].shape==(32768,)and z['mask_'+str(k)].all()
        with np.load(pr/'force_certificate.npz')as z:
            for k in layout['active']:
                assert z['mask_'+str(k)].shape==(32768,)and z['mask_'+str(k)].all();masks+=32768
        assert len(f_report['force_counts'])==row['object_pose_count']and all(v==32768 for v in f_report['force_counts'].values())
        check=f_report['construction']['serialized_export_verification'];assert check['passed']and check['history'][-1]['geometry_passed']and check['history'][-1]['force_passed']
        delta=dict(id=row['id'],object_pose_count=row['object_pose_count'],support_pose_count=row['support_pose_count'],
            reuse=row['reuse'],juxtaposed=row['juxtaposed'])
        for stage,old_key,new_root,report in [('step52','step52_source',pr,p_report),('step53','step53_source',fr,f_report)]:
            old_root=ROOT/before[old_key];old_report=json.loads((old_root/'data/report.json').read_text())
            old_mesh=trimesh.load(old_root/old_report['support'],force='mesh',process=False)
            mesh=trimesh.load(new_root/report['support'],force='mesh',process=False)
            old_bounds=static_bounds(body.vertices,layout,old_mesh.vertices,old_mesh.vertices,p_report['poses'])
            new_bounds=static_bounds(body.vertices,layout,mesh.vertices,mesh.vertices,p_report['poses'])
            a=old_bounds['with_full_support'];b=new_bounds['with_full_support']
            diff=b['volume_litres']-a['volume_litres']
            assert abs(diff)<=1e-8
            for key in ['min_m','max_m','dimensions_m']:
                np.testing.assert_allclose(a[key],b[key],atol=1e-8,rtol=0)
            assert len(old_bounds['fixture_states'])==len(new_bounds['fixture_states'])
            state_bound_difference=0.
            for old_state,new_state in zip(old_bounds['fixture_states'],new_bounds['fixture_states']):
                assert old_state['poses']==new_state['poses']
                for key in ['min_m','max_m','dimensions_m']:
                    state_bound_difference=max(state_bound_difference,float(np.max(np.abs(
                        np.asarray(old_state['with_full_support'][key])-np.asarray(new_state['with_full_support'][key])))))
            assert volume(solid(mesh)-solid(old_mesh))<=1e-10
            assert mesh.volume<=old_mesh.volume+1e-10
            from sculpted_pruning import sharp_edges
            old_quality=sharp_edges(old_mesh);new_quality=sharp_edges(mesh)
            delta.update({stage+'_old_material_cm3':old_mesh.volume*1e6,stage+'_new_material_cm3':mesh.volume*1e6,stage+'_old_sharp_edge_length_m':old_quality['sharp_edge_length_m'],stage+'_new_sharp_edge_length_m':new_quality['sharp_edge_length_m']})
            delta.update({stage+'_old_bbx_l':a['volume_litres'],stage+'_new_bbx_l':b['volume_litres'],stage+'_delta_l':diff,
                stage+'_max_state_bound_difference_m':state_bound_difference,
                stage+'_old_state_sum_l':old_bounds['state_space_sums_litres']['with_full_support'],
                stage+'_new_state_sum_l':new_bounds['state_space_sums_litres']['with_full_support']})
        radius=p_report['pruning']['rounding_radius_m'];radii[str(radius)]=radii.get(str(radius),0)+1
        delta['final_rounding_radius_m']=f_report['construction']['fillet']['rounding_radius_m']
        delta['compaction_radius_m']=p_report['pruning']['compaction_radius_m']
        delta['step52_surface_flow']=p_report['pruning']['surface_flow']is not None
        delta['step53_surface_flow']=f_report['construction']['fillet']['surface_flow']is not None
        delta['body_edge_fillets']=p_report['pruning']['accepted_edge_fillets']
        delta['foot_edge_fillets']=f_report['construction']['fillet']['accepted_edge_fillets']
        delta['rounding_radius_m']=radius;rows.append(delta)
    for picture in new['pictures']:
        assert sha(ROOT/picture['path'])==picture['sha256']
        with Image.open(ROOT/picture['path'])as im:im.verify()
    assert masks==8617984 and sum(r['object_pose_count']for r in rows)==263
    assert not list((COOPT/'output/B').rglob('*.html'))
    with(BASE/('evaluation_'+args.experiment+'_poses.csv')).open()as f:roles=list(csv.DictReader(f))
    assert len(roles)==263 and len({(r['set'],r['pose'])for r in roles})==263
    result=dict(passed=True,complete=True,groups=52,object_poses=263,support_poses=123,reuse=123,juxtaposed=140,
        original_step52_mask_entries=masks,original_demands_passed_in_both_stages=True,
        protected_files=len(protected),changed_inputs=changed,archived_pngs=len(archive),verified_current_pngs=len(new['pictures']),
        step4_and_step51_rerun=False,all_bbx_and_roles_unchanged=True,rounding_radii=radii,material_nonincreasing_in_every_set=True,
        total_old_pruned_cm3=sum(r['step52_old_material_cm3']for r in rows),total_new_pruned_cm3=sum(r['step52_new_material_cm3']for r in rows),
        total_old_final_cm3=sum(r['step53_old_material_cm3']for r in rows),total_new_final_cm3=sum(r['step53_new_material_cm3']for r in rows),
        unchanged_bbx_metric='static_occupancy_v2_overlapping_objects; overall union BBX',
        per_installation_state_bounds_are_separate_diagnostics=True,
        step52_surface_flow_groups=sum(r['step52_surface_flow']for r in rows),step53_surface_flow_groups=sum(r['step53_surface_flow']for r in rows),
        compacted_groups=sum(r['compaction_radius_m']is not None for r in rows),
        total_body_edge_fillets=sum(r['body_edge_fillets']for r in rows),total_foot_edge_fillets=sum(r['foot_edge_fillets']for r in rows),
        max_step52_bbx_difference_l=max(abs(r['step52_delta_l'])for r in rows),
        max_step53_bbx_difference_l=max(abs(r['step53_delta_l'])for r in rows),
        source_and_archives_unchanged=True,no_html=True,cases=rows)
    save(system/'sculpted_audit.json',result)
    with(BASE/('comparison_'+args.experiment+'.csv')).open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('SCULPTED AUDIT',json.dumps({k:v for k,v in result.items()if k!='cases'}),flush=True)


if __name__=='__main__':main()
