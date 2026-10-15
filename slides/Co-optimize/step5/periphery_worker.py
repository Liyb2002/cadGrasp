"""Actual contact pruning by inward fronts, round full-width connections."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,time
from feasible_geometry import C,np,Layout,ForceCertificate,validate,solid,unpack,volume
from surface_flow_bounded import flow
from connector_bands import reinforce
from contact_subset_batch import minimal_face_patches
from rounded_front import prune,connector_quality


def saved_mesh(mesh,path):
    from exact_mesh import save_mesh
    return save_mesh(mesh,path)


def executed_sources():
    result={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            p=Path(file).resolve()
            if p.is_file()and p.is_relative_to(C.ROOT)and not p.is_relative_to(C.ROOT/'.venv'):
                result[C.relative(p)]=C.digest(p)
    return result


def contact_areas(checker,mesh):
    from periphery import PlaneForceFilter
    result={}
    for k,(triangles,sources)in PlaneForceFilter(checker,mesh).contacts.items():
        area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1).sum()/2
        result[str(k)]=dict(area_cm2=float(area*1e4),object_faces=len(np.unique(sources)),boundary_triangles=len(triangles))
    return result


def worker(payload):
    began=time.monotonic();args=SimpleNamespace(**payload)
    inputs=C.Inputs(args.object,args.selection_summary,args.step51_experiment);data=inputs.load(args.case)
    witness_path=data['selected']/'oracle_connected_witness.obj'
    witness=C.trimesh.load(witness_path,force='mesh',process=False)
    inputs.protected[C.relative(witness_path)]=C.digest(witness_path)
    with np.load(data['selected']/'layout.npz')as a:layout=Layout.from_arrays(a)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],args.object)
    initial_sources=executed_sources()
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=False)
    checkpoint_source=Path(args.checkpoint);baseline_record=json.loads((checkpoint_source/'report.json').read_text())
    if C.digest(checkpoint_source/'support.obj')!=baseline_record['support_sha256']:raise ValueError('Baseline changed')
    original=C.trimesh.load(checkpoint_source/'support.obj',force='mesh',process=False);core=data['fixture']
    context=dict(witness=witness,body=data['body'],placements=layout.placements[list(layout.active)],
        rows=data['rows'],checker=checker,bounds=C.occupied_bounds(data['body'],original,data['rows']))
    calibration=json.loads(Path(args.calibration).read_text())
    inputs.protected[C.relative(Path(args.calibration))]=C.digest(args.calibration)
    for reference in calibration['references']:
        if C.digest(C.ROOT/reference['source'])!=reference['sha256']:raise ValueError('Thickness reference changed')
        inputs.protected[reference['source']]=reference['sha256']
    phase={'name':'contact_subset','index':0}
    def checkpoint(mesh,proof,history):
        phase['index']+=1;target=folder/'checkpoints'/f'{phase["index"]:04d}'
        saved_mesh(mesh,target/'support.obj')
        np.savez_compressed(target/'force_certificate.npz',**{'mask_'+str(k):v for k,v in proof['masks'].items()})
        C.save(target/'report.json',dict(passed=True,counts=proof['counts'],history=history,
            support_sha256=C.digest(target/'support.obj'),geometry_and_full_demands_checked=True))
        C.save(folder/'progress.json',dict(phase=phase['name'],commits=phase['index'],
            material_cm3=abs(float(mesh.volume))*1e6,latest_checkpoint=C.relative(target),seconds=time.monotonic()-began))
    print('PHASE certified_contact_subset',args.case['id'],flush=True)
    cache=C.COOPT/'data/results'/args.object/'data/contact_subset_cache_20261015'/args.case['id']
    points,ports=minimal_face_patches(checker,original,cache)
    for path in [cache/'report.json',cache/'guide.npz']:inputs.protected[C.relative(path)]=C.digest(path)
    C.save(folder/'contact_subset.json',dict(complete=True,poses=ports,actual_geometry_changed=False))
    before=contact_areas(checker,original)
    def certify_growth(candidate):
        candidate=saved_mesh(candidate,folder/'growth_candidate.obj')
        proof,_=validate(candidate,**context)
        return candidate,proof
    phase['name']='full_width_connections';print('PHASE full_width_connections',args.case['id'],flush=True)
    stock,bands,initial_profile=reinforce(original,witness,data['body'],data['rows'],context['bounds'],
        calibration,certify_growth,checkpoint)
    stock=saved_mesh(stock,folder/'retained_stock.obj')
    C.save(folder/'reinforcement.json',dict(history=bands,profile=initial_profile))
    def certify_deletion(candidate):
        candidate=saved_mesh(candidate,folder/'deletion_candidate.obj')
        proof,_=validate(candidate,**context,stock=stock)
        return candidate,proof
    phase['name']='coherent_inward_front';print('PHASE coherent_inward_front',args.case['id'],flush=True)
    full,front,proof=prune(stock,core,ports,bands,calibration,context,certify_deletion,checkpoint,
        step=getattr(args,'peel_resolution',.0005))
    history=[dict(operation='certified_contact_subset',accepted=True,poses=ports),
        dict(operation='full_width_connector_bands',accepted=True,history=bands),
        dict(operation='outside_in_contact_pruning',accepted=any(h['accepted']for h in front),pruning=front)]
    frames=list({np.round(r['fixture_transform'],11).tobytes():r['fixture_transform']for r in data['rows']}.values())
    objects=np.vstack([C.F.transform_points(data['body'].vertices,r['object_transform'])for r in data['rows']])
    phase['name']='round_actual_free_surfaces'
    for displacement in [.0015,.00075,.00035]:
        print('PHASE round_actual_free_surfaces',displacement,flush=True)
        try:
            candidate,rounding=flow(full,checker,frames,objects,final_support=True,displacement=displacement,
                pin_contacts=False,pin_bounds=False)
            candidate=saved_mesh(unpack(candidate),folder/f'rounding_{displacement:.5f}.obj')
            candidate_proof,_=validate(candidate,**context,stock=full)
            passed,quality=connector_quality(candidate,calibration)
            if not passed:raise ValueError('Actual rounding creates a connector below the reference')
            removed=(abs(float(full.volume))-abs(float(candidate.volume)))*1e6
            full=candidate;proof=candidate_proof
            history.append(dict(operation='actual_free_surface_rounding',accepted=True,rounding=rounding,
                removed_material_cm3=removed,connector_quality=quality))
            checkpoint(full,proof,history);break
        except (ValueError,RuntimeError)as error:
            history.append(dict(operation='actual_free_surface_rounding',accepted=False,error=str(error)))
    full=saved_mesh(full,folder/'support.obj');proof,checks=validate(full,**context,stock=stock)
    passed,quality=connector_quality(full,calibration)
    C.save(folder/'final_thickness.json',quality)
    after=contact_areas(checker,full)
    retained_core=saved_mesh(unpack(solid(core)^solid(full)),folder/'core.obj')
    np.savez_compressed(folder/'force_certificate.npz',**{'mask_'+str(k):v for k,v in proof['masks'].items()})
    inputs.verify();code=executed_sources()
    if any(C.digest(C.ROOT/p)!=h for p,h in initial_sources.items()):raise ValueError('Executed source changed during worker')
    C.save(folder/'report.json',dict(passed=True,complete=True,operation='outside_in',accepted=True,
        history=history,checks=checks,protected=inputs.protected,executed_code=code,
        before_contact_metrics=before,after_contact_metrics=after,
        input_contact_triangles={k:r['boundary_triangles']for k,r in before.items()},
        final_contact_triangles={k:r['boundary_triangles']for k,r in after.items()},
        thickness_reference_radius_m=quality['guard_radius_m'],thickness_sampling_pitch_m=calibration['pitch_m'],
        thickness_target_passed=passed,connector_quality=quality,
        added_to_core_cm3=volume(solid(full)-solid(retained_core))*1e6,
        core_loss_cm3=volume(solid(core)-solid(full))*1e6,
        added_within_legal_stock_cm3=volume(solid(full)-solid(original))*1e6,
        removed_from_original_cm3=volume(solid(original)-solid(full))*1e6,
        support_sha256=C.digest(folder/'support.obj'),core_sha256=C.digest(folder/'core.obj'),
        seconds=time.monotonic()-began))


if __name__=='__main__':
    payload=json.loads(Path(sys.argv[1]).read_text())
    try:worker(payload)
    except Exception as error:
        import traceback
        C.save(Path(payload['output'])/'failure.json',dict(passed=False,error=str(error),traceback=traceback.format_exc()))
        raise
