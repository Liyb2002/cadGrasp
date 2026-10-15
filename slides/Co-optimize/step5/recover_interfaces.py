"""Reconnect necessary finite interfaces and repeat the original certification.

Recovery changes actual geometry, never the force, floor or space tolerance.
"""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.ndimage import distance_transform_edt
import tube_worker as W
from feasible_geometry import C,Layout,ForceCertificate,validate
from deterministic_base import solid,unpack,union,volume
from anchor_network import ball
from continuous_legs import bounded_domain,raster
from tube_support import ground_ports,path_tube
from tube_routing import TerminalRouteGuide,full_width_middle
from prune_support import material_components


def worker(payload):
    inputs=C.Inputs(payload['object'],payload['selection_summary'],payload['step51_experiment'])
    data=inputs.load(payload['case']);baseline=Path(payload['checkpoint']);output=Path(payload['output'])
    stock=C.trimesh.load(baseline/'support.obj',force='mesh',process=False)
    witness_path=data['selected']/'oracle_connected_witness.obj'
    witness=C.trimesh.load(witness_path,force='mesh',process=False);allowed=solid(witness)
    cache=C.COOPT/'data/results'/payload['object']/'data/contact_subset_cache_20261015'/payload['case']['id']
    ports=json.loads((cache/'reaction_ports/report.json').read_text())['ports'];floors,_=ground_ports(stock,data['rows'])
    retained=solid(stock)^union([ball(p['center_m'],.002)for p in ports]+[ball(p['center_m'],.003)for p in floors])
    plan=output.with_name(output.name+'_plan');plan.mkdir(parents=True,exist_ok=False)
    bounds=C.occupied_bounds(data['body'],stock,data['rows'])
    with np.load(data['selected']/'layout.npz')as a:layout=Layout.from_arrays(a)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],payload['object'])
    context=dict(witness=witness,body=data['body'],placements=layout.placements[list(layout.active)],
        rows=data['rows'],checker=checker,bounds=bounds)
    failed=Path(payload['failed_candidate']);candidate=C.trimesh.load(failed,force='mesh',process=False)
    value=solid(candidate)
    if volume(value-allowed)>1e-10:value=value^allowed
    value=(value+retained).simplify(0.)
    domain=bounded_domain(witness,data['body'],data['rows'],bounds)
    paths=[];bridges=[];route_records=[];inside=None;clearance=None;origin=None;shape=None;pitch=None
    calibration=json.loads(Path(payload['calibration']).read_text());reference=calibration['guard_radius_m']
    for iteration in range(12):
        parts=material_components(value)
        if len(parts)==1:break
        if inside is None:
            bb=unpack(domain).bounds;pitch=max(.0006,float(np.prod(bb[1]-bb[0])/160_000_000)**(1/3))
            origin=bb[0]-pitch;shape=np.ceil((bb[1]-origin)/pitch).astype(int)+2
            print('RECOVERY_DENSE_FIELD',shape,pitch,flush=True)
            inside=raster(domain,origin,shape,pitch)
            clearance=distance_transform_edt(inside,sampling=pitch).astype(np.float32)
        roots=raster(parts[0],origin,shape,pitch)&inside;target=unpack(parts[1]).center_mass;connected=False
        for radius in [reference+.0005,reference+.0002,reference+.0001]:
            guide=object.__new__(TerminalRouteGuide);guide.domain=domain;guide.pitch=pitch
            guide.origin=origin;guide.shape=shape;guide.minimum_radius=radius;guide.clearance=clearance
            guide.valid=inside&(clearance>=radius+.05*pitch);guide.roots=roots&guide.valid
            try:
                path,record=guide.route(target)
                if not full_width_middle(path,radius,domain):raise ValueError('Native full-width middle rejected recovery route')
                bridge=path_tube(path,radius+.0005)^domain;proposal=(value+bridge).simplify(0.)
                if len(material_components(proposal))>=len(parts):raise ValueError('Routed tube did not join a needed interface')
                value=proposal;bridges.append(bridge);paths.append(path.tolist());record['minimum_middle_radius_m']=radius
                route_records.append(record);connected=True;print('RECOVERY_ROUTE',record,flush=True);break
            except (ValueError,RuntimeError)as error:print('RECOVERY_ROUTE_REJECT',radius,str(error),flush=True)
        if not connected:raise ValueError('No reference-width route to the required finite interface')
    if len(material_components(value))!=1:raise ValueError('Interface recovery left disconnected material')
    bridge_path=None
    if bridges:
        trial=W.saved_mesh(unpack(value),plan/'trial.obj');proof,checks=validate(trial,**context)
        bridge_path=plan/'dense_floor_bridge.obj';W.saved_mesh(unpack(union(bridges)),bridge_path)
        C.save(plan/'dense_route_report.json',dict(passed=True,checks=checks,thickness_passed=True,
            thickness_certificate='native_full_radius_middle_of_every_recovery_route',
            path_m=paths[0],paths_m=paths,route_records=route_records,
            minimum_middle_radius_m=min(r['minimum_middle_radius_m']for r in route_records)))
        retained=retained+union(bridges)
    retained=retained.simplify(0.);original_save=W.saved_mesh;events=[]
    def save(mesh,path):
        name=Path(path).name
        if name=='minimum_connector_core.obj'and bridges:mesh=unpack((solid(mesh)+retained).simplify(0.))
        result=original_save(mesh,path)
        if name=='candidate.obj'or name.startswith('curved_'):
            body=solid(result);outside=volume(body-allowed)
            if outside>1e-10:result=original_save(unpack((body^allowed).simplify(0.)),path);body=solid(result)
            if volume(retained-body)>1e-12:result=original_save(unpack((body+retained).simplify(0.)),path)
            events.append(dict(path=C.relative(Path(path)),sha256=C.digest(path),outside_before_m3=outside))
            C.save(output/'retained_interfaces.json',dict(events=events,
                bridge=None if bridge_path is None else dict(path=C.relative(bridge_path),sha256=C.digest(bridge_path)),
                reference=C.relative(baseline/'support.obj'),reference_sha256=C.digest(baseline/'support.obj'),
                physical_tolerances_unchanged=True,force_demands_unchanged=True))
        return result
    W.saved_mesh=save
    W.worker(payload)
    p=output/'report.json';record=json.loads(p.read_text());record['interface_recovery']=True
    if bridges:
        record['main_network_minimum_radius_m']=record['minimum_round_connector_radius_m']
        record['minimum_round_connector_radius_m']=min(r['minimum_middle_radius_m']for r in route_records)
    C.save(p,record)


if __name__=='__main__':
    payload=json.loads(Path(sys.argv[1]).read_text())
    try:worker(payload)
    except Exception as error:
        import traceback
        C.save(Path(payload['output'])/'failure.json',dict(passed=False,error=str(error),traceback=traceback.format_exc()))
        raise
