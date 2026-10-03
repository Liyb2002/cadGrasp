"""Optional local foot pockets AFTER co-design has built the complete solid."""
import copy
import json
from pathlib import Path
import shutil
import time

import manifold3d as md
import numpy as np
from shapely.geometry import Polygon, box
import trimesh

from step3_scheculer import contacts as I
from step4_connect_support.baseline_current import build_coupled_saddle as S,convex_foot as F
from step2_local_support import geometry as G
from step4_connect_support.baseline_current import process_access as ACCESS, acceptance as ACCEPT

WALL=.004
RIB=.006
MAX_DEPTH=.035
MAX_REMOVAL=.30


def union(parts):return md.Manifold.batch_boolean(parts,md.OpType.Add)


def keep_box(points,margin):
    low=np.min(points,axis=0)-margin;high=np.max(points,axis=0)+margin
    return md.Manifold.cube(((high-low)/S.SCALE).tolist()).translate((low/S.SCALE).tolist())


def pockets(xy):
    """Inset each separate foot; a wide central rib divides its two pockets."""
    polygon=Polygon(xy);center=np.asarray(polygon.centroid.coords[0])
    _,_,basis=np.linalg.svd(np.asarray(xy)-center,full_matrices=False)
    local=(np.asarray(xy)-center)@basis.T
    inside=Polygon(local).buffer(-WALL,join_style=2)
    if inside.is_empty:return []
    extent=max(np.ptp(local,axis=0))*2
    result=[]
    for region in (box(-extent,RIB/2,extent,extent),box(-extent,-extent,extent,-RIB/2)):
        pocket=inside.intersection(region)
        if pocket.is_empty or pocket.geom_type!='Polygon':continue
        pocket=pocket.buffer(-.0015).buffer(.0015,quad_segs=4)
        if pocket.is_empty or pocket.area<25e-6:continue
        result.append(np.asarray(pocket.exterior.coords)[:-1]@basis+center)
    return result


def cutter(xy,height,basis,offset):
    points=np.vstack([np.c_[xy,np.full(len(xy),z)] for z in (-.001,height)])@basis+offset
    return S.solid(G.hull_mesh(points))


def ground_hulls(solid,bases,offsets):
    return [F.landing(solid,b,o).convex_hull for b,o in zip(bases,offsets)]


def process(source,output,case,placement,cache):
    began=time.perf_counter();source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    before=I.check_report(source/'report.json');design=json.loads((source/'design.json').read_text())
    before_mesh=trimesh.load(source/'fixture.obj',force='mesh',process=False)
    original=S.solid(before_mesh);full=original
    bases=np.asarray(placement['bases']);offsets=np.asarray(placement['offsets'])
    required=ground_hulls(original,bases,offsets)
    protected=[];patches=[]
    for k,row in enumerate(case.support_seeds):
        for cells in row:
            points=np.vstack(cells)@bases[k]+offsets[k]
            patches.append(points);protected.append(keep_box(points,.004))
    for joint in design['connections']:
        if joint.get('kind')=='carved_connection_envelope':
            protected.append(keep_box(np.asarray(joint['bounds_m']),.001))
        else:
            route=np.asarray(joint['path_m'])
            protected.extend(keep_box(np.array([a,b]),joint['radius_m']+.002) for a,b in zip(route[:-1],route[1:]))
    protected=union(protected)
    accepted=[];rejected=[];cuts=[];initial_volume=float(original.volume())*S.SCALE**3
    for foot in design['consolidated_landings']:
        k=foot['floor_index'];points=patches[foot['head_body']]
        height=float(((points-offsets[k])@bases[k][2]).min())
        depth=min(MAX_DEPTH,height*.65,height-.004)
        if depth<.004:continue
        for xy in pockets(foot['polygon_xy_m']):
            pocket=cutter(xy,depth,bases[k],offsets[k])-protected
            proposed=full-pocket
            record=dict(head_body=foot['head_body'],candidate_id=foot['candidate_id'],
                floor_index=k,floor_pose=foot['floor_pose'],polygon_xy_m=xy.tolist(),depth_m=depth)
            removed=float((full.volume()-proposed.volume())*S.SCALE**3)
            if removed<1e-9:continue
            if len(proposed.decompose())!=1:
                rejected.append(dict(record,reason='would_disconnect_material'));continue
            fraction=1-float(proposed.volume())*S.SCALE**3/initial_volume
            if fraction>MAX_REMOVAL:
                rejected.append(dict(record,reason='moderate_removal_budget'));continue
            actual=ground_hulls(proposed,bases,offsets)
            gaps=[float(p.difference(q.buffer(1e-10)).area) for p,q in zip(required,actual)]
            if max(gaps)>1e-12:
                rejected.append(dict(record,reason='would_reduce_ground_support_hull'));continue
            mesh=S.unpack(proposed)
            if not mesh.is_watertight or not mesh.is_winding_consistent:
                rejected.append(dict(record,reason='invalid_mesh'));continue
            full=proposed;cuts.append(pocket)
            accepted.append(dict(record,removed_volume_cm3=removed*1e6,maximum_ground_hull_loss_m2=max(gaps)))
    mesh=S.unpack(full) if cuts else before_mesh
    sweeps=[]
    for k in range(len(case.poses)):
        with np.load(Path(cache)/f'sweep{k}.npz') as data:
            sweeps.append(trimesh.Trimesh(data['v'],data['f'],process=False))
    passed=True
    try:
        checks,certificate=S.verify(case.tasks,case.groups,case.support_seeds,np.asarray(placement['directions']),bases,offsets,mesh,sweeps,check_equilibrium=False)
    except RuntimeError as error:
        if not hasattr(error,'checks'):raise
        passed=False;checks,certificate=error.checks,error.certificate
    regressed=any(any(was and not ACCEPT.pose_checks(b)[key]
        for key,was in ACCEPT.pose_checks(a).items()) for a,b in zip(before['checks'],checks))
    rollback=bool(cuts and regressed)
    if rollback:
        rejected.extend(dict(p,reason='postprocess_full_check_regression') for p in accepted)
        accepted=[];cuts=[];full=original;mesh=before_mesh
        passed=True
        try:
            checks,certificate=S.verify(case.tasks,case.groups,case.support_seeds,
                np.asarray(placement['directions']),bases,offsets,mesh,sweeps,check_equilibrium=False)
        except RuntimeError as error:
            if not hasattr(error,'checks'):raise
            passed=False;checks,certificate=error.checks,error.certificate
    for check in checks:
        check.update(head_preservation_geometry='constructor_generated_support_roots_and_exact_contact_patch',
            input_head_thickness_m=0.,original_probe_solid_preservation_required=False)
    access=getattr(case,'process_access_guard',None)
    if access is None:
        access=ACCESS.Guard(case,placement)
    access_check=access.verify(mesh)
    if not access_check['passed']:
        raise ACCESS.AccessRejected(dict(access_check,status='hollowed_solid_touches_working_surface'))
    mesh.export(output/'fixture.obj',file_type='obj',digits=17,include_normals=False)
    np.savez_compressed(output/'geometry.npz',vertices_m=mesh.vertices,faces=mesh.faces,rotations=bases,local_offsets_m=offsets)
    certificate_name='geometry_certificate.npz' if passed else 'geometry_diagnostic.npz'
    np.savez_compressed(output/certificate_name,**certificate)
    removed_solid=union(cuts) if cuts else md.Manifold()
    for local in design['local_bodies']:
        body=S.solid(trimesh.load(source/local['file'],force='mesh',process=False))
        body_mesh=S.unpack(body-removed_solid) if cuts else trimesh.load(source/local['file'],force='mesh',process=False)
        body_mesh.export(output/local['file'],file_type='obj',digits=17,include_normals=False)
        local['volume_cm3']=float(body_mesh.volume*1e6)
    if (source/'bridges.obj').exists():shutil.copyfile(source/'bridges.obj',output/'bridges.obj')
    details=dict(stage='after_complete_codesign_construction',global_ground_ring=False,
        wall_m=WALL,central_rib_m=RIB,maximum_depth_m=MAX_DEPTH,maximum_removed_fraction=MAX_REMOVAL,
        pockets=accepted,rejected_pockets=rejected,rolled_back_after_full_checks=rollback,
        before_volume_cm3=float(before_mesh.volume*1e6),after_volume_cm3=float(mesh.volume*1e6),
        removed_fraction=float(1-mesh.volume/before_mesh.volume),strength_verified=False,
        original_ground_hulls_xy_m=[np.asarray(p.exterior.coords)[:-1].tolist() for p in required],
        all_prior_passed_geometry_checks_retained=True,force_torque_authority='step3')
    design['postprocess_hollowing']=details
    I.save(output/'design.json',design)
    artifacts=['fixture.obj','geometry.npz','design.json',certificate_name]+[x['file'] for x in design['local_bodies']]
    if (source/'bridges.obj').exists():artifacts.append('bridges.obj')
    report=copy.deepcopy(before)
    report.update(schema='codesign_separate_feet_with_optional_final_pockets_v1',passed=passed,checks=checks,
        process_access=access_check,
        status='step4_geometry_passed' if passed else 'step4_geometry_failed',
        passed_scope='step4_geometry_only',force_torque_authority='step3',
        step4_force_torque_enforced=False,step4_force_torque_recomputed=False,
        verification_artifact=certificate_name,
        volume_cm3=float(mesh.volume*1e6),dimensions_mm=(mesh.extents*1000).tolist(),
        postprocess_hollowing=details,design='codesign_separate_feet_then_local_pockets',
        solid=dict(component_count=len(mesh.split(only_watertight=False)),watertight=bool(mesh.is_watertight),
            consistently_wound=bool(mesh.is_winding_consistent),volume_m3=float(mesh.volume),one_solid=True),
        body_design=dict(before['body_design'],local_bodies=design['local_bodies'],postprocess_hollowing=details),
        shared_branch_definition='Unshared rigid contact groups, independently seated tasks',
        presentation_description='co-design 局部脚体先构造并连接，再对合适的脚做局部掏空；没有整体大环。',
        artifacts={name:I.sha256(output/name) for name in artifacts})
    report['timings_seconds']['postprocess_hollowing']=time.perf_counter()-began
    report['provenance']['inputs'].update(I.hashes([source/'report.json',source/'fixture.obj',source/'design.json']))
    report['provenance']['code'].update(I.hashes([Path(__file__),Path(F.__file__),Path(ACCEPT.__file__),*ACCESS.sources()]))
    I.save(output/'report.json',report)
    print('FOOT POCKETS',len(accepted),'volume',details['before_volume_cm3'],'->',details['after_volume_cm3'],flush=True)
    return report
