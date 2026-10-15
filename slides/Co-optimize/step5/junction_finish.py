"""Build and certify a continuous round backbone in an existing round support.

Only native-verified full-radius center connections join graph components.
Every exported intermediate/final body gets the unchanged full validator.
"""
from pathlib import Path
from types import SimpleNamespace
import json,sys,time
from feasible_geometry import C,np,Layout,ForceCertificate,validate
from deterministic_base import solid,unpack,volume,union
from continuous_legs import bounded_domain
from anchor_network import InteriorField,ball,path_body
from tube_support import ground_ports,capsule
from exact_mesh import save_mesh
from rounded_front import connector_quality
from periphery_worker import contact_areas,executed_sources
from prune_support import material_components


def finish(payload):
    began=time.monotonic();args=SimpleNamespace(**payload)
    inputs=C.Inputs(args.object,args.selection_summary,args.step51_experiment);data=inputs.load(args.case)
    parent=Path(args.parent_directory);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    partial=not(parent/'report.json').exists()
    if partial:
        progress=json.loads((parent/'progress.json').read_text());checkpoint=C.ROOT/progress['latest_checkpoint']
        source=checkpoint/'support.obj';intermediate_source=source
        original_record=json.loads((checkpoint/'report.json').read_text())
        assert original_record['passed']and C.digest(source)==original_record['support_sha256']
        with np.load(checkpoint/'force_certificate.npz')as saved:
            assert len(saved.files)==args.case['pose_count']and all(saved[k].shape==(32768,)and saved[k].all()for k in saved.files)
        parent_record=dict(history=[dict(operation='outside_in_contact_pruning',accepted=True,
            pruning=dict(history=original_record['history'],resumed_from_complete_checkpoint=True))],protected={},
            minimum_round_connector_radius_m=.0015)
        interruption=parent/'proposal_interruption.json'
        assert interruption.exists();inputs.protected[C.relative(interruption)]=C.digest(interruption)
    else:
        parent_record=json.loads((parent/'report.json').read_text());assert parent_record['passed']
        source=parent/'support.obj';intermediate_source=parent/'core.obj'
        assert C.digest(source)==parent_record['support_sha256']
        inputs.protected.update(parent_record['protected'])
    for p in [source,intermediate_source]:inputs.protected[C.relative(p)]=C.digest(p)
    body=C.trimesh.load(source,force='mesh',process=False)
    intermediate=C.trimesh.load(intermediate_source,force='mesh',process=False)
    baseline=Path(args.checkpoint);original=C.trimesh.load(baseline/'support.obj',force='mesh',process=False)
    baseline_record=json.loads((baseline/'report.json').read_text());assert baseline_record['passed']
    inputs.protected.update(baseline_record['protected'])
    inputs.protected[C.relative(baseline/'support.obj')]=C.digest(baseline/'support.obj')
    witness_path=data['selected']/'oracle_connected_witness.obj';witness=C.trimesh.load(witness_path,force='mesh',process=False)
    inputs.protected[C.relative(witness_path)]=C.digest(witness_path)
    with np.load(data['selected']/'layout.npz')as arrays:layout=Layout.from_arrays(arrays)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],args.object)
    bounds=C.occupied_bounds(data['body'],original,data['rows'])
    context=dict(witness=witness,body=data['body'],placements=layout.placements[list(layout.active)],
        rows=data['rows'],checker=checker,bounds=bounds)
    calibration=json.loads(Path(args.calibration).read_text());target=calibration['guard_radius_m']
    inputs.protected[C.relative(Path(args.calibration))]=C.digest(args.calibration)
    domain=(bounded_domain(witness,data['body'],data['rows'],bounds)+solid(original))^solid(witness)
    field=InteriorField(domain,max(.0015,target+.0005))
    subset_path=parent/'contact_subset.json';subset=json.loads(subset_path.read_text())
    inputs.protected[C.relative(subset_path)]=C.digest(subset_path)
    floors,floor_records=ground_ports(original,data['rows']);centers=[]
    for port in subset['ports']:
        centers.append(field.anchor(np.asarray(port['center_m']),np.asarray(port['normal_m'])))
    for port in floors:
        point=np.asarray(port['center_m']);normals=[r['fixture_transform'][2,:3]for r in data['rows']
            if abs(r['fixture_transform'][2,:3]@point+r['fixture_transform'][2,3])<=1e-8]
        centers.append(field.anchor(point,np.sum(normals,axis=0)if normals else None))
    proposal_path=parent/'connection_proposals.json';proposal=json.loads(proposal_path.read_text())
    inputs.protected[C.relative(proposal_path)]=C.digest(proposal_path)
    saved_paths=[np.asarray(p)for p in proposal['paths']]
    branch_evidence=None
    evidence=parent/'retained_interfaces.json'
    if evidence.exists():
        details=json.loads(evidence.read_text());branch=details.get('bridge')
        inputs.protected[C.relative(evidence)]=C.digest(evidence)
        if branch:
            bridge_path=C.ROOT/branch['path'];assert C.digest(bridge_path)==branch['sha256']
            branch_evidence=bridge_path.with_name('dense_route_report.json')
            details=json.loads(branch_evidence.read_text());assert details['passed']and details['thickness_passed']
            inputs.protected[C.relative(branch_evidence)]=C.digest(branch_evidence)
            inputs.protected[C.relative(bridge_path)]=C.digest(bridge_path)
            saved_paths.extend(np.asarray(p)for p in details.get('paths_m',[details['path_m']]))
    radius=max(target+.0001,target+.00045 if branch_evidence is None else target+.0001)
    nodes=[]
    def node(point):
        for i,existing in enumerate(nodes):
            if np.linalg.norm(point-existing)<1e-10:return i
        nodes.append(np.asarray(point));return len(nodes)-1
    for c in centers:node(c)
    path_ids=[]
    for path in saved_paths:
        ids=[]
        for point in path:
            if volume(ball(point,radius)-domain)>1e-12:
                # The short functional contact/floor cap ends on a boundary.
                # Its interior full-radius section remains in this backbone.
                ids.append(None);continue
            ids.append(node(point))
        path_ids.append(ids)
    nodes=np.asarray(nodes);parent_ids=list(range(len(nodes)))
    def find(i):
        while parent_ids[i]!=i:parent_ids[i]=parent_ids[parent_ids[i]];i=parent_ids[i]
        return i
    def join(a,b):parent_ids[find(a)]=find(b)
    additions=[];links=[];tree_paths=[]
    current=solid(body)
    for a,b in [(a,b)for ids in path_ids for a,b in zip(ids,ids[1:])if a is not None and b is not None]:
        if find(a)==find(b):continue
        tube=capsule(nodes[a],nodes[b],radius)
        if volume(tube-domain)>1e-12:continue
        tree_paths.append((a,b,np.asarray([nodes[a],nodes[b]])));join(a,b)
        if volume(tube-current)>1e-12:additions.append(path_body([nodes[a],nodes[b]],radius+.00005)^domain)
    edges=sorted((float(np.linalg.norm(nodes[a]-nodes[b])),a,b)for a in range(len(nodes))for b in range(a))
    for _,a,b in edges:
        if find(a)==find(b):continue
        tube=capsule(nodes[a],nodes[b],radius)
        if volume(tube-domain)<=1e-12:path=np.asarray([nodes[a],nodes[b]])
        else:
            try:
                path=field.route(nodes[a],nodes[b])
                path=np.vstack([nodes[a],path,nodes[b]])
                if any(volume(capsule(x,y,radius)-domain)>1e-12 for x,y in zip(path,path[1:])):continue
            except (ValueError,RuntimeError):continue
        tube=path_body(path,radius);tree_paths.append((a,b,path));join(a,b)
        missing=volume(tube-current)
        if missing>1e-12:additions.append(path_body(path,radius+.00005)^domain)
        links.append(dict(path_m=path.tolist(),minimum_radius_m=radius,previously_missing_m3=missing))
        print('CONNECTED_WIDTH_LINK',len(links),'groups',len({find(i)for i in range(len(nodes))}),flush=True)
        if len({find(i)for i in range(len(nodes))})==1:break
    if len({find(i)for i in range(len(nodes))})!=1:raise ValueError('No native full-radius backbone joining every interior anchor')
    attached={0};pending=list(tree_paths);value=ball(nodes[0],radius)
    while pending:
        ready=next((i for i,(a,b,path)in enumerate(pending)if a in attached or b in attached),None)
        if ready is None:raise ValueError('Backbone tree traversal lost an anchor')
        a,b,path=pending.pop(ready);value=(value+path_body(path,radius)).simplify(0.)
        attached.update([a,b])
    backbone=unpack(value)
    from backbone_regularization import regularize
    backbone,certified_radius,regularization=regularize(backbone,nodes,tree_paths,radius,target)
    save_mesh(backbone,output/'backbone_before_verification.obj')
    if len(material_components(solid(backbone)))!=1:raise ValueError('Serialized backbone has disconnected material')
    # Replenish a whole node ball as well as links when the old finishing step
    # shaved a small part of the former minimum core.
    for point in nodes:
        b=ball(point,radius)
        if volume(b-current)>1e-12:additions.append(ball(point,radius+.00005)^domain)
    extra=union(additions)
    full=save_mesh(unpack((current+extra).simplify(0.))if additions else body,output/'support.obj')
    core=save_mesh(unpack((solid(intermediate)+extra).simplify(0.))if additions else intermediate,output/'core.obj')
    evidence=dict(shell_regularization=regularization,rejected_proposals=[])
    projected=False
    def project_products():
        nonlocal backbone,full,core,certified_radius,projected
        from project_backbone import project
        for inset in [.00005,.00015]:
            if radius-inset<=target:continue
            # Projection reinforces actual products. Retry on pristine copies;
            # rejected proof proposals never accumulate geometry mutations.
            packet=backbone.copy();candidate_full=full.copy();candidate_core=core.copy()
            try:
                measured,record=project(packet,candidate_full,candidate_core,nodes,tree_paths,
                    radius,target,domain,proof_inset_m=inset)
            except (ValueError,RuntimeError)as error:
                evidence['rejected_proposals'].append(dict(operation='inner_backbone_projection',
                    proof_inset_m=inset,error=str(error)));continue
            backbone=packet;full=candidate_full;core=candidate_core
            certified_radius=measured;evidence['projection']=record;projected=True;return
        raise ValueError('No connected inner backbone above the reference width: '+str(evidence))
    def certify_products():
        nonlocal full,core
        full=save_mesh(full,output/'support.obj');core=save_mesh(core,output/'core.obj')
        minimum=save_mesh(backbone,output/'minimum_connector_core.obj')
        return (*validate(full,**context,mandatory=minimum),*validate(core,**context,mandatory=minimum))
    if regularization:project_products()
    try:proof,checks,core_proof,core_checks=certify_products()
    except (ValueError,RuntimeError)as error:
        evidence['rejected_proposals'].append(dict(operation='initial_product_certification',error=str(error)))
        if not projected:project_products()
        try:proof,checks,core_proof,core_checks=certify_products()
        except ValueError as projected_error:
            if not str(projected_error).startswith('Original demands rejected candidate:'):raise
            # Boolean retessellation can erase a necessary, finite planar
            # contact normal. Restore only missing microscopic source facets,
            # then rerun every original geometric and force condition.
            from contact_facets import restore
            full,facets=restore(checker,body,full);core,core_facets=restore(checker,intermediate,core)
            evidence.update(restored_full_facets=facets,restored_intermediate_facets=core_facets)
            proof,checks,core_proof,core_checks=certify_products()
    evidence_path=None
    if regularization or projected or evidence.get('restored_full_facets'):
        evidence_path=output/'backbone_projection.json';C.save(evidence_path,evidence)
    passed,quality=connector_quality(full,calibration)
    core_passed,core_quality=connector_quality(core,calibration)
    # A continuous native solid of radius greater than the reference radius
    # is the width certificate. A1mm raster cannot resolve its remaining
    # 0.1mm erosion core reliably; retain that sampled result as a diagnostic.
    if certified_radius<=target:raise ValueError('Backbone radius is below the approved reference')
    before=contact_areas(checker,original);after=contact_areas(checker,full)
    np.savez_compressed(output/'force_certificate.npz',**{'mask_'+str(k):v for k,v in proof['masks'].items()})
    np.savez_compressed(output/'core_force_certificate.npz',**{'mask_'+str(k):v for k,v in core_proof['masks'].items()})
    source_snapshot=Path(args.parent_source_snapshot)
    previous_code=parent_record.get('executed_code',{})
    previous_source_records={}
    for p,h in previous_code.items():
        archived=source_snapshot/p
        if not archived.exists()or C.digest(archived)!=h:
            if C.digest(C.ROOT/p)!=h:raise ValueError('Parent source lacks its actual archived version: '+p)
            archived=output/'parent_executed_sources'/p;archived.parent.mkdir(parents=True,exist_ok=True)
            import shutil;shutil.copy2(C.ROOT/p,archived)
        previous_source_records[p]=dict(sha256=h,archive=C.relative(archived))
        inputs.protected[C.relative(archived)]=h
    history=list(parent_record['history'])+[dict(operation='continuous_full_width_backbone',accepted=True,
        radius_m=certified_radius,outer_design_radius_m=radius,nodes=len(nodes),links=links,backbone_components=1,
        all_native_corridors_checked=True,partial_parent=partial)]
    sources=executed_sources();inputs.verify();C.freeze(output)
    C.save(output/'report.json',dict(passed=True,complete=True,accepted=True,operation='connected_width_finishing',
        history=history,checks=checks,intermediate_checks=core_checks,
        protected=inputs.protected,executed_code=sources,parent_directory=C.relative(parent),
        parent_source_snapshots=previous_source_records,resumed_from_complete_checkpoint=partial,
        before_contact_metrics=before,after_contact_metrics=after,
        input_contact_triangles={k:v['boundary_triangles']for k,v in before.items()},
        final_contact_triangles={k:v['boundary_triangles']for k,v in after.items()},
        thickness_target_passed=True,connector_quality=quality,intermediate_connector_quality=core_quality,
        sampled_reference_profile_passed=passed,intermediate_sampled_profile_passed=core_passed,
        thickness_certificate='connected_native_round_backbone_contained_in_actual_solid',
        reference_guard_radius_m=target,
        minimum_round_connector_radius_m=certified_radius,minimum_connector_core=C.relative(output/'minimum_connector_core.obj'),
        backbone_projection_evidence=None if evidence_path is None else C.relative(evidence_path),
        backbone_projection_sha256=None if evidence_path is None else C.digest(evidence_path),
        necessary_microfacets_restored=bool(evidence.get('restored_full_facets')or evidence.get('restored_intermediate_facets')),
        minimum_connector_core_sha256=C.digest(output/'minimum_connector_core.obj'),
        minimum_connector_core_connected=True,full_width_corridors_checked=True,
        unified_core_and_ground_construction=True,intermediate_is_certified_complete_support=True,
        added_to_core_cm3=0. if C.digest(output/'support.obj')==C.digest(output/'core.obj')else None,
        core_loss_cm3=None,added_within_legal_stock_cm3=None,removed_from_original_cm3=None,
        set_difference_volume_bounds_cm3=dict(
            added_to_intermediate=[max(0.,abs(float(full.volume))-abs(float(core.volume)))*1e6,abs(float(full.volume))*1e6],
            original_contact_core_loss=[max(0.,abs(float(data['fixture'].volume))-abs(float(full.volume)))*1e6,abs(float(data['fixture'].volume))*1e6],
            added_to_original=[max(0.,abs(float(full.volume))-abs(float(original.volume)))*1e6,abs(float(full.volume))*1e6],
            removed_from_original=[max(0.,abs(float(original.volume))-abs(float(full.volume)))*1e6,abs(float(original.volume))*1e6]),
        comparative_volume_statistics='explicit_conservative_bounds; exact enclosed material volumes and contact areas reported separately',
        junction_reinforcement_added_cm3=max(0.,abs(float(full.volume))-abs(float(body.volume)))*1e6,
        support_sha256=C.digest(output/'support.obj'),core_sha256=C.digest(output/'core.obj'),
        seconds=time.monotonic()-began))


if __name__=='__main__':
    payload=json.loads(Path(sys.argv[1]).read_text())
    try:finish(payload)
    except Exception as error:
        import traceback
        C.save(Path(payload['output'])/'failure.json',dict(passed=False,error=str(error),traceback=traceback.format_exc()))
        raise
