"""Independently audit saved peripheral pruning, contacts, and publication PNGs."""
from pathlib import Path
import argparse,json,shutil,time
from feasible_geometry import C,np,Layout,ForceCertificate,validate,solid,unpack
from periphery import PlaneForceFilter,structural_profile
from rounded_front import connector_quality
from prune_support import material_components
from feasible_pipeline import read,save,digest,relative,ROOT,COOPT


def contact_metrics(checker,mesh):
    contacts=PlaneForceFilter(checker,mesh).contacts
    result={}
    for key,(triangles,sources)in contacts.items():
        area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1).sum()/2
        result[str(key)]=dict(contact_area_cm2=float(area*1e4),
            original_object_faces_in_contact=len(np.unique(sources)),boundary_triangles=len(triangles))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--selection-summary',type=Path,default=COOPT/'data/results/B/data/step5_show_all52_20261014_02_inputs/selection.json')
    parser.add_argument('--step51-experiment',default='pso_all52_20261013_01_floor')
    parser.add_argument('--allow-subset',action='store_true')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args();base=COOPT/'data/results/B'
    directory=base/'data'/('step5_feasible_'+args.experiment)
    summary=read(directory/'summary.json');assert summary['complete']
    if not args.allow_subset:assert summary['group_count']==52
    verification=read(directory/'verification.json')
    assert verification['passed']and verification['source_and_execution_snapshot_hashes_match']
    protected=verification['protected_inputs'].copy()
    for name,expected in verification['executed_code'].items():
        assert digest(directory/'executed_sources'/name)==expected,name
    inputs=C.Inputs('B',args.selection_summary,args.step51_experiment)
    audit_sources=C.freeze(directory/'independent_audit_execution')
    selected={r['id']:r for r in inputs.cases};results=[];masks=0;intermediate_masks=0;began=time.monotonic()
    for case in summary['results']:
        if not case['passed']:
            assert case['status']=='STEP4_FAIL',case
            results.append(case);continue
        source=selected[case['id']];data=inputs.load(source)
        folder=ROOT/case['output'];report=read(folder/'data/report.json')
        worker=read(ROOT/report['checkpoint']/'report.json')
        if worker.get('backbone_projection_evidence'):
            assert digest(ROOT/worker['backbone_projection_evidence'])==worker['backbone_projection_sha256']
        mesh=C.trimesh.load(folder/report['support'],force='mesh',process=False)
        assert digest(folder/report['support'])==worker['support_sha256']
        for name,expected in report['artifacts'].items():assert digest(folder/name)==expected
        with np.load(data['selected']/'layout.npz')as arrays:layout=Layout.from_arrays(arrays)
        checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],'B')
        witness=C.trimesh.load(data['selected']/'oracle_connected_witness.obj',force='mesh',process=False)
        baseline=C.trimesh.load(ROOT/report['optional_operations'][0]['output']/'support.obj',force='mesh',process=False)
        minimum_path=ROOT/worker['minimum_connector_core']
        assert digest(minimum_path)==worker['minimum_connector_core_sha256']
        minimum=C.trimesh.load(minimum_path,force='mesh',process=False)
        # Fresh solver state checks the actual delivered bytes, independently
        # of both plane-cone prefilters and the worker's cached masks.
        proof,checks=validate(mesh,witness,data['body'],layout.placements[list(layout.active)],
            data['rows'],checker,bounds=C.occupied_bounds(data['body'],baseline,data['rows']),mandatory=minimum)
        with np.load(folder/'force_certificate.npz')as saved:
            assert len(saved.files)==source['pose_count']
            for key in saved.files:assert saved[key].shape==(32768,)and saved[key].all()
            masks+=sum(saved[key].size for key in saved.files)
        before=contact_metrics(checker,baseline);after=contact_metrics(checker,mesh)
        reference=read(ROOT/report['thickness_calibration'])
        sampled_passed,profile=connector_quality(mesh,reference)
        native_backbone=(len(material_components(solid(minimum)))==1 and
            worker['minimum_round_connector_radius_m']>reference['guard_radius_m'])
        thickness_passed=(native_backbone if worker.get('thickness_certificate')==
            'connected_native_round_backbone_contained_in_actual_solid'else sampled_passed)
        s52=ROOT/report['step52_source'];intermediate_report=read(s52/'data/report.json')
        for name,expected in intermediate_report['artifacts'].items():assert digest(s52/name)==expected
        intermediate=C.trimesh.load(s52/intermediate_report['support'],force='mesh',process=False)
        intermediate_proof,intermediate_checks=validate(intermediate,witness,data['body'],
            layout.placements[list(layout.active)],data['rows'],checker,
            bounds=C.occupied_bounds(data['body'],baseline,data['rows']),mandatory=minimum)
        if(s52/'force_certificate.npz').exists():
            with np.load(s52/'force_certificate.npz')as saved:
                assert len(saved.files)==source['pose_count']
                for name in saved.files:assert saved[name].shape==(32768,)and saved[name].all()
                intermediate_masks+=sum(saved[name].size for name in saved.files)
        row=dict(id=case['id'],passed=True,thickness_target_passed=thickness_passed,
            object_poses=source['pose_count'],output=case['output'],checks=checks,
            intermediate_checks=intermediate_checks,native_backbone_connected=native_backbone,
            native_backbone_radius_m=worker['minimum_round_connector_radius_m'],
            reference_radius_m=reference['guard_radius_m'],sampled_profile_passed=sampled_passed,
            before_contacts=before,after_contacts=after,
            contact_area_before_cm2=sum(r['contact_area_cm2']for r in before.values()),
            contact_area_after_cm2=sum(r['contact_area_cm2']for r in after.values()),
            baseline_material_cm3=abs(float(baseline.volume))*1e6,
            final_material_cm3=abs(float(mesh.volume))*1e6,
            thickness_profile=profile,full_original_force_recomputed=True,
            added_within_legal_stock_cm3=worker.get('added_within_legal_stock_cm3',0.),
            removed_from_original_cm3=worker.get('removed_from_original_cm3',0.))
        results.append(row)
        save(directory/'actual_solid_audit.json',dict(complete=False,results=results))
        print('PERIPHERY_AUDIT',case['id'],'thickness',thickness_passed,'contact cm2',
            round(row['contact_area_before_cm2'],3),'->',round(row['contact_area_after_cm2'],3),flush=True)
    protected.update(inputs.protected)
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    passing=[r for r in results if r['passed']]
    appearance_passed=all(r['thickness_target_passed']for r in passing)
    audit=dict(complete=True,passed=appearance_passed,physical_checks_passed=True,
        group_count=len(results),physical_passed=len(passing),
        original_step4_failures=len(results)-len(passing),
        thickness_passed=sum(r['thickness_target_passed']for r in passing),
        actual_original_demand_masks=masks,results=results,protected_inputs=protected,
        actual_intermediate_original_demand_masks=intermediate_masks,executed_audit_sources=audit_sources,
        force_solver_rerun=True,previous_numeric_sources_unchanged=True,
        contact_area_before_cm2=sum(r['contact_area_before_cm2']for r in passing),
        contact_area_after_cm2=sum(r['contact_area_after_cm2']for r in passing),
        baseline_material_cm3=sum(r['baseline_material_cm3']for r in passing),
        final_material_cm3=sum(r['final_material_cm3']for r in passing),seconds=time.monotonic()-began)
    save(directory/'actual_solid_audit.json',audit)
    if not appearance_passed:raise SystemExit('Some saved supports still miss the connector reference; do not publish as finished')
    if args.publish:
        removed={};published={}
        for row in passing:
            report=read(ROOT/row['output']/'data/report.json')
            for stage,images in [('step5.3',ROOT/row['output']/'images'),
                ('step5.2',ROOT/report['step52_source']/'images')]:
                if not images.exists():raise FileNotFoundError('Render independent saved-model PNGs before publication')
                target=COOPT/'output/B'/row['id']/stage;target.mkdir(parents=True,exist_ok=True)
                for old in target.glob('*.png'):
                    archived=directory/'previous_publication'/old.relative_to(COOPT/'output')
                    archived.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,archived)
                    removed[relative(old)]=dict(sha256=digest(old),archive=relative(archived));old.unlink()
                for pic in images.glob('*.png'):
                    destination=target/pic.name;shutil.copy2(pic,destination)
                    published[relative(destination)]=digest(destination)
        save(directory/'publication.json',dict(complete=True,previous_pngs_archived=removed,
            published=published,no_html=True,source_audit=relative(directory/'actual_solid_audit.json')))
    print('PERIPHERY_AUDIT_COMPLETE',len(passing),'supports',masks,'demands',flush=True)


if __name__=='__main__':main()
