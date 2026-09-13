"""Separate, reviewable contact-refinement proposal; never rewrites Step3.

A direction is supplied explicitly, then compatible areas on the selected source
faces are refitted and every load/shape/path claim is rechecked. This is not an
unconditional fallback in the fixed-contact baseline.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step3_scheculer.batch_cases import ProposalFolder
from step1.cases import selected_pose,pose_name
from step1.needs import OUTPUTS,sha256
from step2_local_support import insertion as D,circles as C,surface as Surf
from step3_scheculer import contacts as I
from step4_floor_contact import equilibrium as Q
from step5_connect_support import whole_assembly as A,belt_assembly as BA,belt_geometry as B
from step5_connect_support import floor_design as FD,solids as S,rigid_path as P,surface_check as U,video


class SweptScene(B.Scene):
    def __init__(self,mesh,direction):
        super().__init__(mesh);self.direction=np.asarray(direction)
    def clear(self,part,ground=False):
        if not super().clear(part,ground):return False
        sweep=D.engine.hull_mesh(np.vstack([part.vertices,part.vertices+4*self.scale*self.direction]))
        return self.volume(sweep)<=1e-11*self.scale**3


def fit_contacts(domain,original,direction):
    _,polygons=C.eligible_polygons(domain)
    polygons={f:p for f,p in polygons.items() if domain.mesh.face_normals[f]@direction >= -1e-10}
    contacts=[];changes=[]
    for c in original:
        selected={f:p for f,p in polygons.items() if f in set(c['source_faces'])}
        if c['center_face'] not in selected:raise ValueError('Direction excludes an original contact center')
        surface=C.SurfaceCircles(domain.mesh,selected)
        patch,record=surface.fit_area(c['center_m'],c['center_face'],.0051*domain.mesh.area)
        if not surface.spread(patch)['wrap_limit_satisfied']:raise ValueError('Refined head exceeds wrap limit')
        triangles=[];faces=[]
        for face,polygon in patch.items():
            pieces=Surf.fan(polygon);triangles.extend(pieces);faces.extend([face]*len(pieces))
        x=dict(c);triangles=np.asarray(triangles)
        x.update(triangles_m=triangles,source_faces=np.asarray(faces),triangle_areas_m2=Surf.areas(triangles),radius_m=record['radius_m'])
        if I.area([x])<=.005*domain.mesh.area:raise ValueError('Refined head falls below area minimum')
        removed=domain.mesh.face_normals[c['source_faces']]@direction < -1e-10
        changes.append(dict(candidate_id=c['candidate_id'],original_area_m2=I.area([c]),actual_area_m2=I.area([x]),
            actual_area_fraction=I.area([x])/domain.mesh.area,
            original_area_on_excluded_faces_fraction=float(c['triangle_areas_m2'][removed].sum()/I.area([c])),
            radius_before_m=c['radius_m'],radius_after_m=x['radius_m'],wrap=surface.spread(patch)))
        contacts.append(x)
    return contacts,changes


def verify_contact_coverage(domain,contacts,floor):
    points,normals,owners=Q.contact_rays(domain,contacts,floor['original_pivot_m'])
    scale=np.r_[np.ones(3),np.ones(3)/domain.mesh.extents.max()]
    matrix=(Q.wrench(points,normals,domain.com)*scale).T;solver=Q.BatchSolver(matrix)
    results={};arrays=dict(matrix=matrix,scale=scale)
    for prefix,key in [('sample','load_wrenches'),('continuous','continuous_outer_load_wrenches')]:
        result=solver.solve(floor[key]*scale,certified=prefix=='continuous')
        results[prefix]=dict(passed=result['passed'],diagnostics=result['diagnostics'],load_count=len(floor[key]))
        if not result['passed']:return results,arrays
        arrays[prefix+'_assignment']=result['assignment'];arrays[prefix+'_coefficients']=result['weights']
        arrays[prefix+'_bases']=np.asarray(solver.bases,int).reshape(-1,6)
    return results,arrays


def construct(domain,contacts,floor,direction,depth):
    scene=SweptScene(domain.mesh,direction);scale=scene.scale
    blocked=np.flatnonzero(domain.mesh.face_normals@direction < -1e-10)
    belt,heads,ribbons,owners=B.belt(domain.mesh,contacts,domain.work_ids,depth,scene,blocked)
    if not belt['passed']:raise RuntimeError('No belt compatible with this complete withdrawal: '+belt['status'])
    skin=heads+ribbons
    if direction[1]<=1e-8:raise ValueError('This proposal constructor currently requires an upward-oblique direction')
    q=domain.mesh.vertices;shadow=COORD.floor(q)-q[:,1,None]*COORD.floor(direction)/direction[1]
    seed=FD.boundary(np.vstack([floor['support_polygon_xz_m'],shadow,shadow+.04*scale*COORD.floor(direction)/direction[1]]))
    base_angle=float(np.rad2deg(np.arctan2(direction[1],direction[0])))
    for expansion in (1.,1.25,1.6):
        for angle in (base_angle,base_angle+180,base_angle+90,base_angle-90):
            value=B.open_ring(domain.mesh,seed,floor['required_hull_xz_m'],floor['original_pivot_m'],angle,expansion,.3,scene)
            if value is None:continue
            ring,base=value
            for links,route in B.short_links(skin,base,scene,limit=4):
                parts=skin+links+ring;joined,solid=B.union_parts(parts,scale)
                if not solid['one_solid']:continue
                ground,tri=A.footprint(parts,floor['original_pivot_m'],floor['required_hull_xz_m'],scale)
                if not ground['passed']:continue
                motion=np.r_[direction,[0,0,0]];vertices=np.vstack([p.vertices for p in parts])
                amount=next(t for t in np.arange(.25,4.01,.05) if P.separated(scene,vertices+t*scale*direction))
                budget=dict(remaining=300,checked=0)
                if not P.segment(B.Scene(domain.mesh),parts,domain.com,motion,0.,amount,budget):continue
                path=dict(passed=True,status='rigid_trajectory_verified',continuous_sweep_verified=True,
                    origin_m=domain.com.tolist(),length_scale_m=scale,motion=motion.tolist(),
                    final_withdrawal_amount=float(amount),segment_amounts=[0.,float(amount)],checked_intervals=budget['checked'],
                    method='exact convex translation sweeps per primitive with float64 Boolean tolerances',
                    geometric_tolerance_m=scale*1e-10,volume_tolerance_m3=1e-11*scale**3)
                labels=[f'contact_head_{i:04d}' for i in range(len(heads))]+[f'belt_face_{face}_{i}' for i,face in enumerate(owners)]
                labels += [f'connector_{i}' for i in range(len(links))]+[f'ground_strip_{i}' for i in range(len(ring))]
                shape=dict(belt=belt,base=base,link=route,ground=ground,solid=solid,
                    withdrawal_direction=direction.tolist(),step5_seed_polygon_xz_m=seed.tolist(),
                    belt_ground_and_connections_share_full_sweep=True,global_shortest_claimed=False)
                return parts,labels,joined,tri,shape,path
    raise RuntimeError('No complete thick support in this direction-dependent shape menu')


def audit(domain,original,floor,direction,out):
    contacts,changes=fit_contacts(domain,original,direction);saved=I.read_contacts(out/'contacts.npz')
    for a,b in zip(contacts,saved):
        for key in ('triangles_m','source_faces','triangle_areas_m2'):np.testing.assert_array_equal(a[key],b[key])
    data=I.load_npz(out/'geometry.npz');parts=S.unpack_parts(data);scene=B.Scene(domain.mesh)
    joined,solid=B.union_parts(parts,scene.scale);assert solid['one_solid']
    np.testing.assert_array_equal(joined.vertices,data['union_vertices_m']);np.testing.assert_array_equal(joined.faces,data['union_faces'])
    np.testing.assert_array_equal(trimesh.load(out/'support.stl',process=False).triangles,joined.triangles)
    np.testing.assert_allclose(trimesh.load(out/'support_mm.stl',process=False).triangles/1000,joined.triangles,atol=1e-12,rtol=1e-12)
    assert all(scene.clear(p,ground=True) for p in parts)
    for c in contacts:
        assert set(c['source_faces']).isdisjoint(domain.work_ids)
        assert U.surface_distances(joined,np.vstack([c['triangles_m'].reshape(-1,3),c['triangles_m'].mean(axis=1)])).max()<=scene.scale*1e-9
    shape=json.loads((out/'shape.json').read_text())
    # Rebuild the route, belt and actual base from the same declared geometry.
    swept=SweptScene(domain.mesh,direction)
    belt,heads,ribbons,owners=B.belt(domain.mesh,contacts,domain.work_ids,shape['belt']['thickness_m'],swept,
        np.flatnonzero(domain.mesh.face_normals@direction < -1e-10))
    assert belt==shape['belt']
    ring,base=B.open_ring(domain.mesh,shape['step5_seed_polygon_xz_m'],floor['required_hull_xz_m'],floor['original_pivot_m'],
        shape['base']['bearing_deg'],shape['base']['expansion'],shape['base']['cut_fraction'],swept)
    assert base==shape['base']
    rebuilt_skin=heads+ribbons
    saved_skin=[p for p,label in zip(parts,data['part_labels']) if str(label).startswith(('contact_head_','belt_face_'))]
    saved_ring=[p for p,label in zip(parts,data['part_labels']) if str(label).startswith('ground_strip_')]
    for aa,bb in ((saved_skin,rebuilt_skin),(saved_ring,ring)):
        assert len(aa)==len(bb)
        for a,b in zip(aa,bb):np.testing.assert_array_equal(a.vertices,b.vertices);np.testing.assert_array_equal(a.faces,b.faces)
    ground,tri=A.footprint(parts,floor['original_pivot_m'],floor['required_hull_xz_m'],scene.scale);assert ground['passed']
    np.testing.assert_array_equal(tri,data['floor_triangles_m'])
    path=json.loads((out/'trajectory.json').read_text());assert P.replay(scene,parts,path)
    coverage=I.load_npz(out/'coverage.npz');p,n,o=Q.contact_rays(domain,contacts,floor['original_pivot_m'])
    matrix=(Q.wrench(p,n,domain.com)*coverage['scale']).T;np.testing.assert_array_equal(matrix,coverage['matrix'])
    errors={}
    for prefix,key in [('sample','load_wrenches'),('continuous','continuous_outer_load_wrenches')]:
        assignment=coverage[prefix+'_assignment'];coef=coverage[prefix+'_coefficients'];bases=coverage[prefix+'_bases']
        assert np.all(assignment>=0) and coef.min()>=-2e-9;maximum=0.;targets=floor[key]*coverage['scale']
        for k,ids in enumerate(bases):
            which=np.flatnonzero(assignment==k)
            if not len(which):continue
            maximum=max(maximum,float(np.abs(coef[which]@matrix[:,ids].T-targets[which]).max()))
            if prefix=='continuous':assert Q.membership(matrix,ids,targets[which],certified=True)[0].all()
        assert maximum<=5e-8;errors[prefix]=maximum
    bearing=json.loads((out/'bearing.json').read_text())
    if bearing['sampled_passed']:
        from step4_floor_contact.audit import replay
        arrays=I.load_npz(out/'bearing.npz');mu=bearing['sufficient_friction_coefficient']
        pp,nn,oo=BA.bearing_rays(domain,contacts,floor['original_pivot_m'],mu)
        for key,value in [('contact_points_m',pp),('contact_normals',nn),('contact_owners',oo)]:np.testing.assert_array_equal(arrays[key],value)
        errors['shared_sample']=replay(arrays,'sample',floor['load_wrenches'],1,mu)
        if bearing['continuous_passed']:errors['shared_continuous']=replay(arrays,'continuous',floor['continuous_outer_load_wrenches'],1,mu)
    result=dict(complete=True,passed=True,geometry_verified=True,trajectory_verified=True,
        contact_coverage_recomputed=True,shared_body_bearing_verified=bearing['continuous_passed'],
        design_passed=bearing['continuous_passed'],contact_changes=changes,reaction_residuals=errors,
        evidence_audit_is_not_design_success=True)
    I.save(out/'audit.json',result);return result


def build(name,direction):
    domain,original,_,points,directions,_,paths=A.read_inputs(name);floor=FD.prepare(points)
    direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
    out=ProposalFolder(OUTPUTS/name/pose_name()/'step5_connect_support');out.mkdir(parents=True,exist_ok=True)
    I.save(out/'status.json',dict(complete=False,status='building_separate_contact_refinement_proposal'))
    contacts,changes=fit_contacts(domain,original,direction)
    coverage,arrays=verify_contact_coverage(domain,contacts,floor)
    if not coverage.get('continuous',{}).get('passed'):raise RuntimeError('Refined contacts do not cover the continuous load domain')
    I.save_contacts(out/'contacts.npz',contacts);np.savez_compressed(out/'coverage.npz',**arrays)
    parts,labels,joined,tri,shape,path=construct(domain,contacts,floor,direction,directions['normal_depth_m'])
    np.savez_compressed(out/'geometry.npz',**S.pack_parts(parts,labels,joined),floor_triangles_m=tri)
    joined.export(out/'support.stl',file_type='stl_ascii')
    millimetres=joined.copy();millimetres.apply_scale(1000);millimetres.export(out/'support_mm.stl',file_type='stl_ascii')
    I.save(out/'shape.json',shape);I.save(out/'trajectory.json',path)
    mechanics,arrays=BA.bearing(domain,contacts,floor,shape['base']);I.save(out/'bearing.json',mechanics)
    if arrays:np.savez_compressed(out/'bearing.npz',**arrays)
    checks=audit(domain,original,floor,direction,out)
    BA.preview(name,domain,contacts,out,'refined_contacts_geometry_and_path_verified_bearing_'+('passed' if mechanics['continuous_passed'] else 'failed'))
    path['bearing_verified']=bool(mechanics['continuous_passed']);path['contact_refinement_proposal']=True
    I.save(out/'trajectory.json',path)
    video.render(domain,I.load_npz(out/'geometry.npz'),path,out)
    report=dict(complete=True,object=name,pose=pose_name(),schema='contact_refinement_proposal_v1',
        proposal_only=True,upstream_step3_unchanged=True,original_contact_interfaces_preserved=False,
        geometry_verified=True,trajectory_verified=True,contact_coverage=coverage,contact_changes=changes,
        bearing=mechanics,passed=bool(mechanics['continuous_passed']),strength_verified=False,
        stl_units={(out/'support.stl').name:'metres',(out/'support_mm.stl').name:'millimetres'},
        scope='Separate reviewable proposal with refitted contacts. A geometric insertion video is not proof of bearing.',
        provenance=dict(inputs=I.hashes(paths),code={**BA.code_hashes(),**I.hashes([Path(__file__),Path(C.__file__),Path(Surf.__file__)])}),
        artifacts={p.name:sha256(p) for p in out.iterdir() if p.suffix in ('.npz','.stl','.mp4','.gif','.png','.json') and p.name not in ('proposal.json','proposal_status.json')})
    I.save(out/'proposal.json',report);I.save(out/'status.json',dict(complete=True,proposal_sha256=sha256(out/'proposal.json')))
    print(name,pose_name(),'proposal: trajectory verified; bearing',mechanics['continuous_passed'],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('name');parser.add_argument('--pose',default='pose_2')
    parser.add_argument('--direction',nargs=3,type=float,required=True);args=parser.parse_args()
    with selected_pose(args.pose):build(args.name,args.direction)
