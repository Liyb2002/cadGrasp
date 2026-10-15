"""Certify selected contact ports and an inward-pruned rounded support."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,time
from feasible_geometry import C,np,Layout,ForceCertificate,validate,solid,unpack,volume
from periphery_worker import saved_mesh,executed_sources,contact_areas
from contact_subset_batch import minimal_face_patches
from reaction_ports import select_ports
from anchor_network import build
from rounded_front import connector_quality
from surface_flow_bounded import flow


def worker(payload):
    began=time.monotonic();args=SimpleNamespace(**payload)
    inputs=C.Inputs(args.object,args.selection_summary,args.step51_experiment);data=inputs.load(args.case)
    witness_path=data['selected']/'oracle_connected_witness.obj'
    witness=C.trimesh.load(witness_path,force='mesh',process=False)
    inputs.protected[C.relative(witness_path)]=C.digest(witness_path)
    with np.load(data['selected']/'layout.npz')as a:layout=Layout.from_arrays(a)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],args.object)
    sources=executed_sources();folder=Path(args.output);folder.mkdir(parents=True,exist_ok=False)
    baseline=Path(args.checkpoint);base_record=json.loads((baseline/'report.json').read_text())
    if C.digest(baseline/'support.obj')!=base_record['support_sha256']:raise ValueError('Certified baseline changed')
    original=C.trimesh.load(baseline/'support.obj',force='mesh',process=False)
    bounds=C.occupied_bounds(data['body'],original,data['rows'])
    context=dict(witness=witness,body=data['body'],placements=layout.placements[list(layout.active)],
        rows=data['rows'],checker=checker,bounds=bounds)
    calibration=json.loads(Path(args.calibration).read_text())
    inputs.protected[C.relative(Path(args.calibration))]=C.digest(args.calibration)
    for ref in calibration['references']:
        if C.digest(C.ROOT/ref['source'])!=ref['sha256']:raise ValueError('Thickness reference changed')
        inputs.protected[ref['source']]=ref['sha256']
    cache=C.COOPT/'data/results'/args.object/'data/contact_subset_cache_20261015'/args.case['id']
    preparation=cache.parent/'preparation.json'
    while preparation.exists()and not(cache/'report.json').exists():
        ready=json.loads(preparation.read_text())
        if ready['complete']:break
        print('WAIT_CONTACT_GUIDE',args.case['id'],flush=True);time.sleep(5)
    _,faces=minimal_face_patches(checker,original,cache)
    for p in [cache/'report.json',cache/'guide.npz']:inputs.protected[C.relative(p)]=C.digest(p)
    ports,port_records=select_ports(checker,original,faces,cache/'reaction_ports')
    path=cache/'reaction_ports/report.json';inputs.protected[C.relative(path)]=C.digest(path)
    before=contact_areas(checker,original)
    C.save(folder/'contact_subset.json',dict(complete=True,faces=faces,ports=ports,poses=port_records))
    state={'index':0,'mesh':None}
    def certify(mesh):
        mesh=unpack(solid(mesh).simplify(0.))
        mesh=saved_mesh(mesh,folder/'candidate.obj')
        proof,_=validate(mesh,**context,stock=state['mesh'])
        return mesh,proof
    def checkpoint(mesh,proof,history):
        state['index']+=1;target=folder/'checkpoints'/f'{state["index"]:04d}'
        saved_mesh(mesh,target/'support.obj')
        np.savez_compressed(target/'force_certificate.npz',**{'mask_'+str(k):v for k,v in proof['masks'].items()})
        C.save(target/'report.json',dict(passed=True,counts=proof['counts'],history=history,
            support_sha256=C.digest(target/'support.obj'),all_original_geometry_checks=True))
        state['mesh']=mesh.copy()
        C.save(folder/'progress.json',dict(phase='certified_round_support',commits=state['index'],
            latest_checkpoint=C.relative(target),material_cm3=abs(float(mesh.volume))*1e6,seconds=time.monotonic()-began))
    def proposal(paths,history,radius):
        C.save(folder/'connection_proposals.json',dict(physical_complete=False,
            paths=[p.tolist()for p in paths],history=history,minimum_radius_m=radius,
            source_sha256=base_record['support_sha256'],set=args.case['id']))
    print('PHASE rounded_required_contact_network',args.case['id'],flush=True)
    full,proof,construction,backbone=build(original,witness,data['body'],data['rows'],bounds,ports,calibration,
        certify,checkpoint,proposal_checkpoint=proposal)
    intermediate=full.copy()
    history=[dict(operation='certified_contact_subset',accepted=True,poses=port_records),
        dict(operation='outside_in_contact_pruning',accepted=True,pruning=construction)]
    minimum=construction['minimum_connector_radius_m']
    saved_mesh(backbone,folder/'minimum_connector_core.obj')
    frames=list({np.round(r['fixture_transform'],11).tobytes():r['fixture_transform']for r in data['rows']}.values())
    objects=np.vstack([C.F.transform_points(data['body'].vertices,r['object_transform'])for r in data['rows']])
    for displacement,pin_contacts in [(.00012,False),(.00012,True),(.00006,True)]:
        print('PHASE actual_curved_surface',displacement,flush=True)
        try:
            candidate,rounding=flow(full,checker,frames,objects,final_support=True,displacement=displacement,
                mandatory=solid(backbone),pin_contacts=pin_contacts,pin_bounds=False,curvature_subdivisions=2)
            candidate=saved_mesh(unpack(candidate),folder/f'curved_{displacement:.5f}.obj')
            certificate,_=validate(candidate,**context,stock=full,mandatory=backbone)
            passed,quality=connector_quality(candidate,calibration)
            if not passed:raise ValueError('Curved support falls below reference connector profile')
            full=candidate;proof=certificate
            history.append(dict(operation='actual_free_surface_rounding',accepted=True,rounding=rounding))
            checkpoint(full,proof,history);break
        except (ValueError,RuntimeError)as error:
            history.append(dict(operation='actual_free_surface_rounding',accepted=False,error=str(error)))
    full=saved_mesh(full,folder/'support.obj')
    proof,checks=validate(full,**context,mandatory=backbone)
    passed,quality=connector_quality(full,calibration)
    after=contact_areas(checker,full)
    saved_mesh(intermediate,folder/'core.obj')
    np.savez_compressed(folder/'force_certificate.npz',**{'mask_'+str(k):v for k,v in proof['masks'].items()})
    C.save(folder/'final_thickness.json',quality)
    inputs.verify()
    if any(C.digest(C.ROOT/p)!=h for p,h in sources.items()):raise ValueError('Worker source changed during execution')
    C.save(folder/'report.json',dict(passed=True,complete=True,operation='rounded_contact_support',accepted=True,
        history=history,checks=checks,protected=inputs.protected,executed_code=executed_sources(),
        before_contact_metrics=before,after_contact_metrics=after,
        input_contact_triangles={k:r['boundary_triangles']for k,r in before.items()},
        final_contact_triangles={k:r['boundary_triangles']for k,r in after.items()},
        thickness_target_passed=passed,connector_quality=quality,
        minimum_round_connector_radius_m=minimum,full_width_corridors_checked=True,
        minimum_connector_core_sha256=C.digest(folder/'minimum_connector_core.obj'),
        minimum_connector_core=C.relative(folder/'minimum_connector_core.obj'),
        unified_core_and_ground_construction=True,intermediate_is_certified_complete_support=True,
        added_to_core_cm3=0. if C.digest(folder/'support.obj')==C.digest(folder/'core.obj')else None,
        core_loss_cm3=None,added_within_legal_stock_cm3=None,removed_from_original_cm3=None,
        set_difference_volume_bounds_cm3=dict(
            added_to_intermediate=[max(0.,abs(float(full.volume))-abs(float(intermediate.volume)))*1e6,abs(float(full.volume))*1e6],
            original_contact_core_loss=[max(0.,abs(float(data['fixture'].volume))-abs(float(full.volume)))*1e6,abs(float(data['fixture'].volume))*1e6],
            added_to_original=[max(0.,abs(float(full.volume))-abs(float(original.volume)))*1e6,abs(float(full.volume))*1e6],
            removed_from_original=[max(0.,abs(float(original.volume))-abs(float(full.volume)))*1e6,abs(float(original.volume))*1e6]),
        comparative_volume_statistics='explicit_conservative_bounds; exact enclosed material volumes and contact areas reported separately',
        support_sha256=C.digest(folder/'support.obj'),core_sha256=C.digest(folder/'core.obj'),
        seconds=time.monotonic()-began))


if __name__=='__main__':
    payload=json.loads(Path(sys.argv[1]).read_text())
    try:worker(payload)
    except Exception as error:
        import traceback
        C.save(Path(payload['output'])/'failure.json',dict(passed=False,error=str(error),traceback=traceback.format_exc()))
        raise
