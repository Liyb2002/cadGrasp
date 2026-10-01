"""Stationary floor base and a removable blue module, with an independent dock.

One task at a time. d0 installs blue at rest; vertical d1 docks object+blue.
The finite design family uses an exterior rectangular socket on a ground ring.
"""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from shapely.geometry import MultiPoint, Point
from shapely.ops import nearest_points
from step1.needs import OUTPUTS, sha256
from step1.cases import pose_name
from step2_local_support import withdrawal as W, installation as INIT
from step3_scheculer import contacts as I, connection as C
from step4_connect_support import whole_assembly as A, belt_geometry as B, solids as S, floor_design as FD
from step5_base import rectangular_dock as J

SCHEMA='stationary_base_removable_module_v1'
STAGE='step5_base'
MAX_INSTALLATION_DIRECTIONS=12
OFFSETS=(0.,.0025,.005,.01,.02,.04,.08,.12,.2,.4,.8)
FRICTION=(64.,256.,1024.,4096.,16384.,65536.)


def code_hashes():
    from step5_base import base
    return {**C.code_hashes(), **I.hashes([Path(m.__file__) for m in (J,base,A,B,S,FD,J.Q,J.M)]),
            **I.hashes([Path(__file__)])}


def blue_candidates(domain,contacts,directions):
    cat=directions['direction_catalogue'];initial=cat['installation'];depth=directions['normal_depth_m']
    planes=np.asarray([[0.,0.,1.,0.],initial['floor_plane']])
    checker=C.Checker(domain.mesh,depth,cat,floor_planes=planes,work_ids=domain.work_ids)
    analyzer=W.Analyzer(domain.mesh,depth,cat);scale=float(domain.mesh.extents.max())
    ids=list(directions['connection']['directions']['ids'])
    up=np.asarray(initial['floor_plane'][:3]);vectors=np.asarray(cat['vectors'])
    ids.sort(key=lambda i:-float(vectors[i]@up))
    for index in ids[:MAX_INSTALLATION_DIRECTIONS]:
        u=vectors[index]
        extended=contacts if len(contacts)>1 else contacts*2
        witness=checker.check(extended,W.normalize([index]))
        if not witness['passed']:continue
        frame=checker.parts(extended,witness['witness'])
        heads=[h for c in contacts for h in analyzer.heads(c)]
        row=checker.head(contacts[0]);qplane=witness['witness']['plane_offset_m']
        anchor=row['terminal']+(qplane-row['terminal']@u)*u
        proposals=[]
        for axis in (0,1):
            for sign in (-1,1):
                q=J.port(anchor,domain.mesh,u,planes,(axis,sign),3.2*scale)
                if q is not None:proposals.append(q)
        proposals.sort(key=lambda q:np.linalg.norm(q-anchor))
        for q in proposals:
            peg,sleeve,_,_=J.joint(q)
            radius=min(.0035,checker.radius*.7)
            link=J.H.engine.hull_mesh(np.vstack([anchor+radius*J.CORNERS,q+radius*J.CORNERS]))
            parts=[*heads,*frame,link,peg]
            if min(p.bounds[0,2] for p in parts)<-scale*1e-10:continue
            sweep=analyzer.test(parts,u)
            if not sweep['clear']:continue
            joined,solid=B.union_parts(parts,scale)
            if not solid['one_solid']:continue
            yield dict(parts=parts,joined=joined,solid=solid,port=q,interface=J.description(q),
                sleeves=sleeve,d0_id=index,d0=u,witness=witness,initial_sweep=sweep,
                labels=[f'contact_head_{i}' for i in range(len(heads))]+[f'frame_{i}' for i in range(len(frame))]+['interface_link','rectangular_peg'])


def stationary_base(domain,floor,blue,offset):
    from step5_base.base import extrude
    scale=float(domain.mesh.extents.max());q=blue['port'];height=.025*scale;width=.05*scale
    # The ring follows the load hull, not the whole workpiece silhouette or port.
    cloud=np.vstack([floor['required_hull_xy_m'],np.asarray(floor['original_pivot_m'])[None,:2]])
    inner=MultiPoint(cloud).convex_hull.buffer(float(offset)*scale,join_style=2)
    outer=inner.buffer(width,join_style=2);material=outer.difference(inner)
    ring,triangles=extrude(material,height)
    foot=np.r_[q[:2],height/2]
    point=nearest_points(Point(q[:2]),outer.exterior)[1]
    edge=np.array([point.x,point.y,height/2])
    arm=J.H.engine.hull_mesh(np.vstack([foot+[.004,.004,height/2]*J.CORNERS,edge+[.004,.004,height/2]*J.CORNERS]))
    top=q[2]-J.LENGTH-J.WALL+.001
    if top<=height:return None
    post=J.box(np.r_[q[:2],(top+height/2)/2],[.008,.008,top-height/2])
    parts=[*ring,arm,post,*blue['sleeves']]
    joined,solid=B.union_parts(parts,scale)
    if not solid['one_solid']:return None
    sweep=J.docking_check(domain.mesh,blue['joined'],parts)
    if not sweep['passed']:return None
    # Actual floor triangles from every ring/foot primitive, including its arm.
    ground=np.concatenate([p.triangles[np.max(np.abs(p.triangles[:,:,2]),axis=1)<scale*1e-9] for p in parts])
    if not len(ground):return None
    pads=[t[:,:2] for t in ground]
    footprint=dict(pads_xy_m=np.asarray(pads).tolist(),footprint_area_m2=float(MultiPoint(ground[:,:,:2].reshape(-1,2)).convex_hull.area),
        material_area_m2=float(material.area),outer_xy_m=np.asarray(outer.exterior.coords)[:-1].tolist(),offset_fraction=float(offset),
        inner_xy_m=np.asarray(inner.exterior.coords)[:-1].tolist(),ring_closed=True,
        ring_source='required load hull and original pivot only',width_m=width,height_m=height,
        ring_footprint_area_m2=float(outer.area),demand_hull_area_m2=float(MultiPoint(cloud).convex_hull.area))
    return dict(parts=parts,joined=joined,solid=solid,footprint=footprint,ground=ground,docking=sweep)


def build(name):
    began=perf_counter();domain,contacts,schedule,floor,directions,work,paths=A.read_inputs(name)
    if work is not None:raise ValueError('Stationary modular design currently uses the declared surface-only access policy')
    out=OUTPUTS/name/pose_name()/STAGE;out.mkdir(parents=True,exist_ok=True)
    for f in ('geometry.npz','blue.npz','base.stl','base_mm.stl','contact_module.stl','contact_module_mm.stl','trajectory.json','bearing.npz','audit.json'):
        (out/f).unlink(missing_ok=True)
    I.save(out/'status.json',dict(complete=False,status='searching_stationary_base_and_rectangular_dock'))
    floor=FD.prepare(floor);trials=[];best=None;selected=None;blue_count=0;choices=[]
    if contacts and directions['connection']['passed']:
        for blue in blue_candidates(domain,contacts,directions):
            blue_count+=1
            print(name,pose_name(),'modular blue candidate',blue_count,'initial direction',blue['d0_id'],flush=True)
            for offset in OFFSETS:
                base=stationary_base(domain,floor,blue,offset)
                if base is None:
                    trials.append(dict(d0_id=blue['d0_id'],port_m=blue['port'].tolist(),offset_fraction=offset,status='base_geometry_or_docking_failed'));continue
                choices.append(dict(blue=blue,base=base,bearing=dict(passed=False,continuous_passed=False,sampled_passed=False,
                    status='not_certified' if schedule['continuous_coverage_proved'] else 'not_evaluated_partial_head_coverage'),forces={})
                )
            # Keep the first few geometrically valid ports; this is a finite heuristic.
            if blue_count>=8:break
    choices.sort(key=lambda c:(c['base']['footprint']['footprint_area_m2'],c['base']['footprint']['material_area_m2']))
    if choices:best=choices[0]
    if schedule['continuous_coverage_proved']:
        for entry in choices:
            blue,base=entry['blue'],entry['base'];offset=base['footprint']['offset_fraction']
            for mu in FRICTION:
                mechanics,forces=J.bearing(domain,contacts,floor,base['footprint'],blue['interface'],mu)
                trials.append(dict(d0_id=blue['d0_id'],port_m=blue['port'].tolist(),offset_fraction=offset,
                    footprint_area_m2=base['footprint']['footprint_area_m2'],friction=mu,status=mechanics['status'],bearing=mechanics))
                if mechanics['passed']:
                    print(name,pose_name(),'compact fixed dock bearing verified','offset',offset,'friction',mu,flush=True)
                    entry.update(bearing=mechanics,forces=forces);selected=entry;break
            if selected is not None:break
    chosen=selected or best
    report=dict(schema=SCHEMA,stage=STAGE,object=name,pose=pose_name(),complete=True,
        passed=selected is not None,geometry_constructed=chosen is not None,
        status=('stationary_base_interface_bearing_verified' if selected else
                'step3_load_coverage_incomplete' if best and not schedule['continuous_coverage_proved'] else
                'stationary_base_bearing_unresolved' if best else 'no_stationary_dock_geometry' if contacts else 'no_selected_heads'),
        installation=directions['direction_catalogue']['installation'],base_stationary=True,base_anchored=False,
        docking_direction_independent_of_d0=True,docking_withdrawal_direction=J.UP.tolist(),
        direction_search=dict(candidate_count=min(MAX_INSTALLATION_DIRECTIONS,len(directions['connection']['directions']['ids'])),
            remaining_count=len(directions['connection']['directions']['ids']),blue_geometry_candidates=blue_count),
        required_hull_xy_m=floor['required_hull_xy_m'].tolist(),trials=trials,
        search_scope='At most 12 initial-assembly directions and 8 exterior ports; 11 demand-hull ring offsets, sorted by actual ground footprint area. Independent vertical docking. No global minimality or infeasibility claim.',
        objective='Prefer smaller actual floor footprint, then material area, among certified candidates',
        offset_fractions=list(OFFSETS),geometrically_valid_candidate_count=len(choices),
        task_head_loads_verified=schedule['continuous_coverage_proved'],robot_grasp_assumed=True,
        robot_kinematics_verified=False,strength_verified=False,elapsed_seconds=perf_counter()-began)
    if chosen:
        b=chosen['blue'];g=chosen['base']
        np.savez_compressed(out/'blue.npz',**S.pack_parts(b['parts'],b['labels'],b['joined']))
        np.savez_compressed(out/'geometry.npz',**S.pack_parts(g['parts'],[f'base_{i}' for i in range(len(g['parts']))],g['joined']),floor_triangles_m=g['ground'])
        for mesh,stem in ((b['joined'],'contact_module'),(g['joined'],'base')):
            mesh.export(out/f'{stem}.stl',file_type='stl_ascii');mesh.copy().apply_scale(1000).export(out/f'{stem}_mm.stl',file_type='stl_ascii')
        report.update(interface=b['interface'],d0_direction_id=b['d0_id'],d0_withdrawal_direction=b['d0'].tolist(),
            initial_module_sweep=b['initial_sweep'],blue_solid=b['solid'],base_solid=g['solid'],base=g['footprint'],
            docking=g['docking'],bearing=chosen['bearing'])
        I.save(out/'trajectory.json',g['docking'])
        if chosen['forces']:np.savez_compressed(out/'bearing.npz',**chosen['forces'])
    report['provenance']=dict(inputs=I.hashes(paths+INIT.inputs(name)),code=code_hashes())
    report['artifacts']={p.name:sha256(p) for p in out.iterdir() if p.name in ('blue.npz','geometry.npz','base.stl','base_mm.stl','contact_module.stl','contact_module_mm.stl','bearing.npz','trajectory.json')}
    I.save(out/'base.json',report);I.save(out/'status.json',dict(complete=True,status=report['status'],base_sha256=sha256(out/'base.json')))
    print(name,pose_name(),'modular Step5:',report['status'],flush=True)
    return report


def audit(name):
    domain,contacts,schedule,floor,directions,_,_=A.read_inputs(name)
    floor=FD.prepare(floor);out=OUTPUTS/name/pose_name()/STAGE;report=I.check_report(out/'base.json')
    assert report['schema']==SCHEMA
    checks=dict(initial_pose=report['installation']['initial_pose'],base_stationary=True,separate_bodies=True)
    if report['geometry_constructed']:
        bp=I.load_npz(out/'blue.npz');gp=I.load_npz(out/'geometry.npz')
        blue=S.unpack_parts(bp);base=S.unpack_parts(gp)
        bu,bs=B.union_parts(blue,float(domain.mesh.extents.max()));gu,gs=B.union_parts(base,float(domain.mesh.extents.max()))
        assert bs['one_solid'] and gs['one_solid']
        cat=directions['direction_catalogue'];analyzer=W.Analyzer(domain.mesh,directions['normal_depth_m'],cat)
        d0=np.asarray(report['d0_withdrawal_direction'])
        assert report['d0_direction_id'] in directions['connection']['directions']['ids']
        np.testing.assert_array_equal(d0,cat['vectors'][report['d0_direction_id']])
        assert analyzer.test(blue,d0)['clear']
        assert J.docking_check(domain.mesh,bu,base)['passed']
        assert min(p.bounds[0,2] for p in blue)>=-1e-9
        # Preserve every original fitted head cell, not just its face IDs.
        heads=[h for c in contacts for h in analyzer.heads(c)]
        assert len(blue)>=len(heads)
        for a,b in zip(heads,blue):
            np.testing.assert_array_equal(a.vertices,b.vertices);np.testing.assert_array_equal(a.faces,b.faces)
        checks.update(blue_single_solid=True,base_single_solid=True,initial_module_sweep=True,whole_object_blue_docking_sweep=True,exact_heads_retained=True)
        if report['passed']:
            assert schedule['continuous_coverage_proved']
            checks['three_body_reactions_replayed']=J.replay_bearing(domain,contacts,floor,report['base'],report['interface'],report['bearing'],I.load_npz(out/'bearing.npz'))
    else:assert not report['passed']
    result=dict(complete=True,passed=True,design_passed=report['passed'],checks=checks,
        provenance=dict(inputs=I.hashes([out/'base.json']),code=code_hashes()))
    I.save(out/'audit.json',result);return result
