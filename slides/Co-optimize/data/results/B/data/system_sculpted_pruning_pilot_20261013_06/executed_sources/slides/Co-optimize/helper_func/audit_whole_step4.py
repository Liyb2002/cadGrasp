"""Audit current Step4 exports, full work rays, lineage and preserved inputs."""
import _bootstrap
import argparse,time
from PIL import Image
from trimesh.ray.ray_pyembree import RayMeshIntersector
from co_common import *
from run_all import saved_groups,output_root
from work_access import WorkAccess,tangent_frame


def protected_files(root):
    initialization=json.loads((root/'data/whole_initialization_manifest.json').read_text())
    definitions=[('stage3',json.loads((root/'data/whole_step4_protected_stage3.json').read_text()),ROOT),
        ('inputs',initialization['original_input_hashes'],ROOT),
        ('pose_set_search',initialization['pose_set_search_source_hashes'],ROOT),
        ('old_step4',initialization['protected_downstream_artifacts'],root/'_history/before_whole_step4_20261008'),
        ('executed_sources',json.loads((root/'data/whole_step4_sources/manifest.json').read_text())['sources'],ROOT)]
    result={}
    for label,hashes,directory in definitions:
        for relative,digest in hashes.items():
            path=directory/relative
            if not path.exists() or I.sha256(path)!=digest:raise RuntimeError('Changed protected file: '+str(path))
        result[label]=dict(files=len(hashes),unchanged=True)
    return result


def work_audit(mesh,states,placements,support,repeat_continuous=False):
    # Group coincident work areas so repeated task faces are not ray-tested
    # multiple times. The greatest ORIGINAL angle covers nested smaller cones.
    buckets={}
    for k,(task,T,body) in enumerate(states):
        q=placements[k];angle=task.domain.data['load']['cone_half_deg']
        key=q.tobytes(),float(angle)
        entry=buckets.setdefault(key,dict(q=q,angle=angle,ids=set()))
        entry['ids'].update(map(int,task.domain.work_ids))
    actual=S.solid(support) if repeat_continuous else None
    caster=RayMeshIntersector(support)
    bary=np.array([[i,j,9-i-j] for i in range(1,8) for j in range(1,9-i)],float)/9
    tested=blocked=0;overlaps=[];examples=[]
    for entry in buckets.values():
        q=entry['q'];ids=np.array(sorted(entry['ids']),int)
        access=WorkAccess(mesh,ids,entry['angle'])
        if repeat_continuous:
            local=transform_points(support.vertices,np.linalg.inv(q))
            length=access.required_length(local,.5)
            forbidden=access.solid(length).transform(np.c_[q[:3,:3],q[:3,3]/S.SCALE])
            overlaps.append(material_volume(actual^forbidden))
        triangles=transform_points(mesh.triangles[ids].reshape(-1,3),q).reshape(-1,3,3)
        normals=mesh.face_normals[ids]@q[:3,:3].T
        origins=np.einsum('av,tvc->tac',bary,triangles).reshape(-1,3)
        origins+=1e-6*np.repeat(normals,len(bary),axis=0)
        variants=[]
        for normal in normals:
            frame=tangent_frame(normal);directions=[normal]
            for degrees in [entry['angle']/2,entry['angle']]:
                angle=np.radians(degrees)
                directions.extend(np.cos(angle)*normal+np.sin(angle)*(np.cos(a)*frame[:,0]+np.sin(a)*frame[:,1])
                    for a in np.arange(16)*2*np.pi/16)
            variants.append(directions)
        directions=np.repeat(np.asarray(variants),len(bary),axis=0).reshape(-1,3)
        origins=np.repeat(origins,33,axis=0);tested+=len(origins)
        locations,ray_ids,hit_faces=caster.intersects_location(origins,directions,multiple_hits=False)
        if len(ray_ids):
            weights=trimesh.triangles.points_to_barycentric(support.triangles[hit_faces],locations)
            displacement=locations-origins[ray_ids]
            distance=np.einsum('ij,ij->i',displacement,directions[ray_ids])
            perpendicular=np.linalg.norm(displacement-distance[:,None]*directions[ray_ids],axis=1)
            valid=(weights.min(1)>=-1e-6)&(distance>1e-8)&(perpendicular<1e-7)
            blocked+=int(valid.sum())
            for j in np.flatnonzero(valid)[:8]:
                examples.append(dict(face=int(ids[ray_ids[j]//(33*len(bary))]),
                    origin=origins[ray_ids[j]].tolist(),direction=directions[ray_ids[j]].tolist(),
                    hit=locations[j].tolist(),distance_m=float(distance[j])))
    return dict(rays_tested=tested,blocked_rays=blocked,counterexamples=examples,
        independent_continuous_boolean_repeated=repeat_continuous,
        continuous_work_overlap_m3=max(overlaps) if overlaps else None,
        passed=blocked==0 and max(overlaps,default=0.)<=1e-10)


def audit_case(name,group,repeat_continuous=False):
    base=output_root(name)/group['id'];states=[state(name,p) for p in group['poses']]
    mesh=states[0][2];native=np.array([s[1] for s in states]);rows=[]
    for stage in ['step4.1','step4.2']:
        out=base/'step4'/stage;report=I.check_report(out/'data/report.json')
        render=I.check_report(out/'data/render.json')
        assert report['schema']=='whole_step4_v1' and report['poses']==group['poses']
        assert report['original_loads_reused'] and not report['load_subsampling_for_final_acceptance']
        assert report['initialized_all_poses_together'] and not report['full_fixture_accepted']
        assert not render['text_or_labels'] and not render['geometry_changed']
        support=trimesh.load(out/'support.obj',force='mesh',process=False)
        volume=abs(float(support.volume))*1e6
        np.testing.assert_allclose(volume,report['volume_cm3'],rtol=1e-7,atol=1e-5)
        with np.load(out/'layout.npz') as layout:
            placements=layout['placements'];directions=layout['directions'];hosts=layout['hosts']
            np.testing.assert_array_equal(layout['active'],np.arange(len(states)))
            np.testing.assert_allclose(layout['native_world'],native,atol=1e-12,rtol=0)
        for k,(task,T,body) in enumerate(states):
            q=placements[k];placed=native[hosts[k]]@q
            np.testing.assert_allclose(placed[:3,:3],T[:3,:3],atol=1e-10,rtol=0)
            assert abs(placed[2,3]-T[2,3])<1e-9
            assert (native[hosts[k],:3,:3]@directions[k])[2]>=-1e-12
            np.testing.assert_allclose(np.linalg.norm(directions[k]),1.,atol=1e-12,rtol=0)
            with np.load(out/f'{group["poses"][k]}_force.npz') as force:
                assert force['mask'].shape==(len(task.targets),)==(32768,)
                assert int(force['mask'].sum())==report['counts'][group['poses'][k]]
                allowed=np.setdiff1d(np.arange(len(mesh.faces)),task.domain.work_ids)
                local_d=q[:3,:3].T@directions[k]
                allowed=allowed[mesh.face_normals[allowed]@local_d<=1e-9]
                posed=mesh.copy();posed.apply_transform(q)
                triangles,sources=contact_boundary(posed,support,allowed)
                np.testing.assert_array_equal(sources,force['source_faces'])
                np.testing.assert_allclose(triangles,force['triangles_fixture_m'],atol=1e-12,rtol=0)
                recomputed=supply(task,T@np.linalg.inv(q),triangles,sources)
                np.testing.assert_allclose(recomputed,force['supply_7d'],atol=1e-12,rtol=1e-12)
            classification=report['classifiers'][group['poses'][k]]
            assert classification['equilibrium_tolerance']==2e-9
            assert classification['maximum_primal_replay_residual']<=2e-9
            assert classification['covered_count']==report['counts'][group['poses'][k]]
        diagnostic=report['diagnostics']
        assert max(diagnostic[key] for key in ['nominal_sweep_overlap_m3','body_overlap_m3',
            'padded_overlap_outside_contact_cores_m3','work_band_overlap_m3',
            'exported_boundary_max_exclusion_overlap_m3','endpoint_overlap_m3'])<=1e-10
        assert diagnostic['minimum_endpoint_projection_gap_m']>0
        work=work_audit(mesh,states,placements,support,repeat_continuous)
        assert work['passed'],work
        images=['exit_directions.png','exit_sweeps.png','final_result.png'] if stage=='step4.1' else ['process.png','final_result.png']
        for filename in images:
            with Image.open(out/filename) as image:image.verify()
        if stage=='step4.1':
            for k,item in enumerate(render['states']):
                np.testing.assert_allclose(item['direction_world'],native[k,:3,:3]@directions[k],atol=1e-12,rtol=0)
        else:
            assert report['force_exit_work_passed'] and all(n==32768 for n in report['counts'].values())
            assert report['exact_evaluations_during_search']==0 and not report['separated_fallback_used']
            assert render['fixed_isometric_view'] and render['final_force_exit_work_verified']
            assert render['final_support_source']=='support.obj' and not render['work_forbidden_overlay']
            assert len({item['pose'] for item in render['process_states']})==1
            assert all(item['pose']==group['poses'][0] for item in render['process_states'])
            initial=json.loads((base/'step4/step4.1/report.json').read_text())
            if initial['force_and_path_initialization_gate']:assert report['volume_cm3']<=initial['volume_cm3']+1e-4
            for item in json.loads((out/'process.json').read_text()):
                with np.load(out/item['layout']) as z:np.testing.assert_array_equal(z['active'],np.arange(len(states)))
        rows.append(dict(stage=stage,volume_cm3=volume,original_contact_lineage=True,
            original_load_results_fingerprinted=True,
            original_continuous_boolean_and_float64_roundtrip_verified=True,
            recorded_work_overlap_m3=diagnostic['work_band_overlap_m3'],
            recorded_export_max_exclusion_overlap_m3=diagnostic['exported_boundary_max_exclusion_overlap_m3'],
            passed=True,**{k:v for k,v in work.items() if k!='passed'}))
    fingerprints={stage:I.sha256(base/'step4'/stage/'data/report.json') for stage in ['step4.1','step4.2']}
    return dict(id=group['id'],poses=group['poses'],passed=True,stages=rows,stage_report_hashes=fingerprints)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--completed-only',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--repeat-continuous-booleans',action='store_true',
        help='Repeat expensive continuous work Booleans; default verifies their fingerprinted original checks and independently casts unbounded rays')
    args=parser.parse_args();root=output_root(args.object);began=time.monotonic();preserved=protected_files(root)
    path=root/'data/whole_step4_audit.json';old=json.loads(path.read_text()) if args.resume and path.exists() else {}
    cache={row['id']:row for row in old.get('results',[])};rows=[];pending=[]
    for group in saved_groups(args.object):
        report=root/group['id']/'step4/step4.2/data/report.json'
        if not report.exists() or json.loads(report.read_text()).get('schema')!='whole_step4_v1':
            pending.append(group['id']);continue
        try:
            I.check_report(report)
            I.check_report(root/group['id']/'step4/step4.1/data/report.json')
        except (OSError,RuntimeError):
            if not args.completed_only:raise
            pending.append(group['id']);continue
        fingerprints={stage:I.sha256(root/group['id']/'step4'/stage/'data/report.json') for stage in ['step4.1','step4.2']}
        cached=cache.get(group['id'])
        if cached and cached.get('stage_report_hashes')!=fingerprints:cached=None
        row=cached or audit_case(args.object,group,args.repeat_continuous_booleans);rows.append(row)
        print('STEP4 AUDIT',group['id'],row['passed'],sum(s['rays_tested'] for s in row['stages']),flush=True)
        save(path,dict(complete=False,results=rows,preserved=preserved))
    report=dict(complete=not pending,passed=bool(rows) and all(row['passed'] for row in rows),
        sets=len(rows),pending=pending,pose_instances=sum(len(row['poses']) for row in rows),
        rays_tested=sum(s['rays_tested'] for row in rows for s in row['stages']),
        blocked_rays=sum(s['blocked_rays'] for row in rows for s in row['stages']),
        preserved=preserved,results=rows,seconds=time.monotonic()-began,
        original_force_results_checked_without_repeating_search=True,full_fixture_accepted=False,
        provenance=provenance([root/'data/whole_step4_sources/manifest.json'],[Path(__file__)]))
    save(path,report)
    print('STEP4 AUDIT SUMMARY',len(rows),'sets',report['rays_tested'],'rays',report['blocked_rays'],'blocked',len(pending),'pending',flush=True)
    if pending and not args.completed_only:raise RuntimeError('Some whole Step4 groups remain incomplete')


if __name__=='__main__':main()
