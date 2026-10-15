"""Round, finite contact ports connected by solid tubes in the exact domain.

Shrinking round retention bands removes material from their exterior. Contact
ports come from full original-demand elimination, and ground ports preserve
the original pressure hull. Every delivered solid is independently certified.
"""
import numpy as np
import manifold3d as md
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint,Point
from feasible_geometry import C,validate
from deterministic_base import solid,unpack,union,volume,SCALE
from continuous_legs import bounded_domain
from tube_routing import TerminalRouteGuide as RouteGuide,full_width_middle as checked_middle
from prune_support import material_components
from rounded_front import round_ports


def capsule(a,b,radius):
    first=md.Manifold.sphere(radius/SCALE,48).translate(tuple(np.asarray(a)/SCALE))
    last=md.Manifold.sphere(radius/SCALE,48).translate(tuple(np.asarray(b)/SCALE))
    return md.Manifold.batch_hull([first,last])


def ground_ports(stock,rows):
    groups={}
    for row in rows:groups.setdefault(np.round(row['fixture_transform'],11).tobytes(),[]).append(row)
    ports=[];records=[]
    for members in groups.values():
        T=members[0]['fixture_transform'];world=stock.vertices@T[:3,:3].T+T[:3,3]
        ids=np.flatnonzero(abs(world[:,2])<=1e-8)
        if not len(ids):continue
        _,unique=np.unique(np.round(world[ids,:2],12),axis=0,return_index=True);ids=ids[unique]
        hull=MultiPoint(world[ids,:2]).convex_hull
        vertices=np.asarray(hull.exterior.coords)[:-1]if hull.geom_type=='Polygon'else np.asarray(hull.coords)
        _,nearest=cKDTree(world[ids,:2]).query(vertices);ids=ids[np.unique(nearest)]
        keep=list(range(len(ids)))
        def covered(selected):
            for row in members:
                points=np.vstack([world[ids[selected],:2],row['object_ground']])
                boundary=MultiPoint(points).convex_hull
                if boundary.is_empty or any(boundary.distance(Point(q[:2]))>1e-8 for q in row['hull']):return False
            return True
        for index in list(keep):
            trial=[i for i in keep if i!=index]
            if covered(trial):keep=trial
        if not covered(keep):raise ValueError('Original floor ports do not cover original pressure demands')
        for i in keep:ports.append(dict(center_m=stock.vertices[ids[i]].tolist(),extent_m=0.))
        records.append(dict(poses=[r['pose']for r in members],initial_hull_vertices=len(ids),retained_vertices=len(keep)))
    return ports,records


def path_tube(path,radius):
    return union([capsule(a,b,radius)for a,b in zip(path,path[1:])])


def full_width_middle(path,radius,domain):
    """Exact full-radius clearance away from functional contact/floor ends."""
    return checked_middle(path,radius,domain)


def closest_pairs(main,guest,count=12):
    a=unpack(main).vertices;b=unpack(guest).vertices
    a=a[::max(1,len(a)//6000)];b=b[::max(1,len(b)//3000)]
    distance,ids=cKDTree(a).query(b);order=np.argsort(distance,kind='stable')
    pairs=[]
    for i in order:
        first,last=a[ids[i]],b[i]
        if any(np.linalg.norm(first-x)+np.linalg.norm(last-y)<.002 for x,y in pairs):continue
        pairs.append((first,last))
        if len(pairs)>=count:break
    return pairs


def build(stock,witness,body,rows,bounds,contact_records,calibration,certify,
          checkpoint=None,minimum_radius=None,proposal_checkpoint=None,resume_paths=None,resume_history=None):
    minimum_radius=minimum_radius or max(.0015,calibration['guard_radius_m']+.0005)
    working_radius=max(.0025,minimum_radius+.001)
    domain=bounded_domain(witness,body,rows,bounds)
    object_ports=[p for record in contact_records for p in record['patches']]
    floor_ports,floor_records=ground_ports(stock,rows)
    ports=object_ports+floor_ports
    port_guide=round_ports(object_ports,.0035)+round_ports(floor_ports,.015)
    paths=[np.asarray(p)for p in(resume_paths or [])];history=list(resume_history or [])
    if any(not full_width_middle(p,minimum_radius,domain)for p in paths):raise ValueError('Saved proposal no longer has its exact full-width corridor')
    guide=port_guide+union([path_tube(p,working_radius)for p in paths])if paths else port_guide
    value=domain^guide
    precertified=None
    for iteration in range(len(material_components(value))+1):
        parts=material_components(value)
        print('ROUND_CONNECTIONS',iteration,'components',len(parts),flush=True)
        if len(parts)==1:break
        main=parts[0]
        try:
            useful,proof=certify(unpack(main))
            precertified=(useful,proof);value=solid(useful)
            history.append(dict(operation='delete_unneeded_disconnected_pieces',accepted=True,
                removed_components=len(parts)-1,all_original_demands_checked=True))
            break
        except (ValueError,RuntimeError):pass
        guests=sorted(parts[1:],key=lambda p:np.linalg.norm(np.subtract(*closest_pairs(main,p,1)[0])))
        success=False
        for guest in guests:
            pairs=closest_pairs(main,guest)
            proposals=[]
            for a,b in pairs:
                path=np.array([a,b])
                if full_width_middle(path,minimum_radius,domain):
                    proposals.append((path,dict(method='straight_full_width_tube')));break
            if not proposals:
                for margin in [.5,.9,1.25]:
                    try:
                        route=RouteGuide(domain,main,pitch=.00075,minimum_radius=minimum_radius,clearance_margin=margin)
                        if not route.roots.any():raise ValueError('No full-width routing root')
                        path,info=route.route(pairs[0][1],limit=350000)
                        if not full_width_middle(path,minimum_radius,domain):raise ValueError('Exact swept tube fails minimum clearance')
                        proposals.append((path,dict(method='routed_full_width_tube',route=info)));break
                    except (ValueError,RuntimeError)as error:
                        print('ROUTE_REJECT',str(error),flush=True)
                        history.append(dict(operation='round_connector',accepted=False,error=str(error),proposal_grid_margin=margin))
            for path,details in proposals:
                proposal=guide+path_tube(path,working_radius)
                candidate=domain^proposal
                if len(material_components(candidate))>=len(parts):continue
                paths.append(path);guide=proposal;value=candidate;success=True
                history.append(dict(operation='round_connector',accepted=True,**details,
                    path_m=path.tolist(),minimum_full_radius_m=minimum_radius,
                    working_radius_m=working_radius,
                    terminal_exception_length_m=2*minimum_radius,
                    exact_mid_tube_inside_legal_domain=True))
                if proposal_checkpoint:proposal_checkpoint(paths,history,minimum_radius)
                break
            if success:break
        if not success:raise ValueError('Cannot connect required ports with the reference minimum radius')
    else:raise ValueError('Required-port graph did not converge')
    full,proof=precertified if precertified is not None else certify(unpack(value))
    if checkpoint:checkpoint(full,proof,history)
    # The same topology is retained while round bands move inward. No local
    # holes or rectangular pockets are cut into a sheet.
    nodes=union([md.Manifold.sphere(working_radius/SCALE,48).translate(tuple(p/SCALE))
        for path in paths for p in [path[0],path[-1]]])if paths else md.Manifold()
    lower=max(.002,minimum_radius+.00025)
    for radius in np.linspace(working_radius,lower,4)[1:]:
        candidate_guide=port_guide+nodes+union([path_tube(path,radius)for path in paths])
        try:
            proposed=domain^candidate_guide
            parts=material_components(proposed)
            candidate,certificate=certify(unpack(parts[0]if len(parts)>1 else proposed))
            removed=(abs(float(full.volume))-abs(float(candidate.volume)))*1e6
            if removed<=1e-7:continue
            full,proof=candidate,certificate
            history.append(dict(operation='outside_in_round_band_shrink',accepted=True,
                radius_m=radius,minimum_radius_m=minimum_radius,removed_material_cm3=removed))
            if checkpoint:checkpoint(full,proof,history)
            print('ROUND_BAND_SHRINK',radius,'removed cm3',round(removed,3),flush=True)
        except (ValueError,RuntimeError)as error:
            history.append(dict(operation='outside_in_round_band_shrink',accepted=False,radius_m=radius,error=str(error)))
    return full,proof,dict(history=history,ground_port_selection=floor_records,
        object_contact_ports=len(object_ports),ground_ports=len(floor_ports),
        minimum_connector_radius_m=minimum_radius,functional_ground_cap_radius_m=.015,actual_round_geometry=True,
        exact_corridor_checks=True,local_pocket_cutting_used=False)


if __name__=='__main__':
    from pathlib import Path
    import argparse,json,time,importlib.util
    from feasible_geometry import Layout,ForceCertificate
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('set')
    parser.add_argument('--name',default='rounded_tube_probe_20261015_01')
    args=parser.parse_args()
    inputs=C.Inputs('B',C.COOPT/'data/results/B/data/step5_show_all52_20261014_02_inputs/selection.json',
        'pso_all52_20261013_01_floor')
    case=next(c for c in inputs.cases if c['id']==args.set);data=inputs.load(case)
    with np.load(data['selected']/'layout.npz')as arrays:layout=Layout.from_arrays(arrays)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],'B')
    source=C.COOPT/'data/results/B'/args.set/'step5/step5.3/step5_show_all52_20261014_02/support_with_base.obj'
    stock=C.trimesh.load(source,force='mesh',process=False)
    witness=C.trimesh.load(data['selected']/'oracle_connected_witness.obj',force='mesh',process=False)
    contact=json.loads((C.COOPT/'data/results/B/data/contact_subset_cache_20261015'/args.set/'report.json').read_text())
    calibration=json.loads((C.COOPT/'data/results/B/data/step5_feasible_outside_in_calibration_20261015_03/thickness_calibration.json').read_text())
    context=dict(witness=witness,body=data['body'],placements=layout.placements[list(layout.active)],rows=data['rows'],
        checker=checker,bounds=C.occupied_bounds(data['body'],stock,data['rows']))
    output=C.COOPT/'data/results/B/data'/args.name/args.set;output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();code=C.freeze(output)
    def certify(mesh):
        from exact_mesh import save_mesh
        mesh=save_mesh(mesh,output/'candidate.obj')
        proof,_=validate(mesh,**context)
        return mesh,proof
    full,proof,record=build(stock,witness,data['body'],data['rows'],context['bounds'],
        contact['records'],calibration,certify)
    from exact_mesh import save_mesh
    save_mesh(full,output/'support.obj')
    np.savez_compressed(output/'force_certificate.npz',**{'mask_'+str(k):v for k,v in proof['masks'].items()})
    spec=importlib.util.spec_from_file_location('tube_renderer',C.COOPT/'step5.3/render.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.reference(C.V.Renderer(),data,output/'overview.png',fixture=full)
    C.save(output/'report.json',dict(complete=True,passed=True,construction=record,counts=proof['counts'],
        material_before_cm3=abs(float(stock.volume))*1e6,material_after_cm3=abs(float(full.volume))*1e6,
        seconds=time.monotonic()-began,executed_sources=code))
