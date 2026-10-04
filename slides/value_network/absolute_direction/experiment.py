"""World-direction contact search followed by actual Step4/Step5 volume feedback."""
import argparse
import hashlib
import shutil
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import trimesh
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'baseline_algo'
sys.path.insert(0,str(BASE))
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
from step3_scheculer.pair_scoring import TaskProblem,J
# The shared trajectory module prepends the original baseline on import.
# Restore the cloned package path before loading modules modified here.
from step2_local_support import insertion
sys.path.insert(0,str(BASE))
from step2_local_support import withdrawal as W
from step3_scheculer.pair_geometry import FreePaths,transform_contact
from step2_local_support import geometry as G,withdrawal as W
from step4_connect_support import zero_thickness_heads as Z,boxed_support as F,space_budget as B
from step4_connect_support import build_coupled_saddle as S,deterministic_space as D,process_access as A
from step4_connect_support import co_design_walkthrough as HIST
from step4_connect_support.envelope_growth import EnvelopeGrow
from step4_connect_support.growing_support import components as material_components
from step0_pose_selection.floor_points import pressure_centers
sys.path.insert(0,str(HERE.parent/'data_producer'))
from search import install_recorded_recovery

OUT=HERE/'output/B'

def save(path,data):I.save(path,data)

def specs():
    cfg=json.loads((HERE.parent/'data/B/pilot20_shared/config.json').read_text())
    new=json.loads((HERE.parent/'train/transfer/plan.json').read_text())
    return cfg['groups']+new['train']+new['test']

def read_problems(name,poses):
    if name in ['pose1+3','pose1+3copied']:
        old=HIST.read_reference()
        problems=[]
        for t in old.tasks:
            p=TaskProblem(t.pose,t.domain,t.floor,t.targets,t.scale)
            p.inputs=[p for p in old.paths if t.pose in Path(p).parts]
            problems.append(p)
        return problems
    return [read_task('B',p,folder=I.OUTPUTS/'B/independent_poses'/p/'step_1_needs') for p in poses]

def root_cells(problem,c):
    offsets,valid=G.vertex_offsets(problem.domain.mesh,1.)
    ids=np.unique(problem.domain.mesh.faces[np.unique(c['source_faces'])])
    if not valid[ids].all():return None
    clearance=float(c['triangles_m'][:,:,2].min())
    if clearance<.002-1e-9:return None
    depth=min(2*S.RELIEF,.5*clearance/np.linalg.norm(offsets[ids],axis=1).max())
    from step4_connect_support.fixture_view import cells_for
    return cells_for(c,problem.domain,depth*offsets)

def candidates(problem,exit_vector):
    mesh=problem.domain.mesh;pose=problem.pose
    source=I.OUTPUTS/'B/independent_poses'/pose/'step2_local_support'/f'candidates_{pose}.npz'
    metadata=json.loads(source.with_suffix('.json').read_text());records={r['id']:r for r in metadata['candidates']}
    current=read_task('B',pose,folder=I.OUTPUTS/'B/independent_poses'/pose/'step_1_needs')
    transform=np.asarray(problem.domain.data['frame']['T_world_mesh'])@np.linalg.inv(current.domain.data['frame']['T_world_mesh'])
    paths=FreePaths(mesh,np.array([[0.,0.,1.,0.]]))
    analyzer=W.Analyzer(mesh,.0004,dict(vectors=[(-exit_vector).tolist()],object_withdrawal_from_static_support=True))
    result=[]
    for c in I.read_contacts(source):
        if records[c['candidate_id']]['reason'] in ['wrap_angle_exceeded','normal_hits_object','fixed_area_unavailable']:continue
        c=transform_contact(c,transform)
        if np.intersect1d(c['source_faces'],problem.domain.work_ids).size:continue
        normals=-mesh.face_normals[c['source_faces']]
        if (normals@exit_vector).min() < -1e-10:continue
        cells=root_cells(problem,c)
        if cells is None:continue
        check=analyzer.test([SimpleNamespace(vertices=v) for v in cells],-exit_vector)
        if not check['clear']:continue
        n=np.average(normals,axis=0,weights=c['triangle_areas_m2']);n/=np.linalg.norm(n)
        vertex=mesh.faces[c['center_face']];weights=trimesh.triangles.points_to_barycentric(mesh.triangles[c['center_face']][None],c['center_m'][None])[0]
        offsets,_=G.vertex_offsets(mesh,.0004);port=paths.ports(c['center_m']+.5*weights@offsets[vertex])
        if not port:continue
        result.append(dict(contact=c,cells=cells,normal=n,alignment=float(np.average(1-normals@exit_vector,weights=c['triangle_areas_m2'])),components=port))
    print('ABS CANDIDATES',pose,len(result),flush=True)
    return result,source

def verify_heads(problem,entries,cache):
    key=','.join(sorted(e['contact']['candidate_id'] for e in entries))
    if key in cache:return cache[key]['passed']
    full=problem.supply([e['contact'] for e in entries]);probe=np.linspace(0,len(problem.targets)-1,96,dtype=int)
    small,info=J.classify(full,problem.targets[probe])
    if not small.all():record=dict(passed=False,counterexample_indices=probe[~small].tolist())
    else:
        mask,info=J.classify(full,problem.targets)
        record=dict(passed=bool(mask.all()),sample_count=len(mask),covered_count=int(mask.sum()),no_uplift_in_same_reaction_solve=True,classifier=info)
    cache[key]=record;return record['passed']

class AbsoluteGrow(EnvelopeGrow):
    def connect(self):
        full,construction=super().connect()
        # Reinsert exact mandatory cells lost by a Boolean union; no tolerance
        # relaxation, clipping, or new material beyond the original cells.
        expected=[S.solid(G.hull_mesh(v@b+o)) for row,b,o in zip(self.case.support_seeds,self.bases,self.offsets) for cells in row for v in cells]
        repairs=[]
        for attempt in range(4):
            missing=[(abs(float((cell-full).volume()))*S.SCALE**3,i) for i,cell in enumerate(expected)]
            bad=[(volume,i) for volume,i in missing if volume>8e-14]
            if not bad:break
            repairs.append(dict(attempt=attempt,missing_cells=len(bad),maximum_missing_volume_m3=max(v for v,_ in bad)))
            for _,i in sorted(bad,reverse=True):full=full+expected[i]
        else:raise RuntimeError('Mandatory contact cells still lost after exact Boolean repair')
        if len(material_components(full))!=1:raise RuntimeError('Contact-cell repair disconnected')
        self.final_solid=full;self.save_stage('shared_tree',full)
        construction['mandatory_cell_union_repairs']=repairs
        return full,construction
    def __init__(self,group,case,placement):
        started=time.monotonic();self.group=group;self.case=case;self.source=group/'step4/data/boxed_support';self.source.mkdir(parents=True,exist_ok=True)
        self.out=group/'step4/data/growing_support';self.out.mkdir(parents=True,exist_ok=True);case.output=self.out
        self.construction_input_paths=[]
        self.bases=np.asarray(placement['bases']);self.offsets=np.asarray(placement['offsets']);self.directions=np.asarray(placement['directions'])
        search=D.Search.__new__(D.Search);self.search=search;search.group=group;search.out=self.source;search.case=case
        search.saved_path=group/'step3/plan.json';search.saved=dict(placement=placement);search.before=I.hashes(case.paths)
        search.bases=self.bases;search.offsets=self.offsets;search.directions=self.directions
        search.root_points=[np.vstack([v for cells in row for v in cells]) for row in case.support_seeds]
        search.patch_points=[np.vstack([c['triangles_m'].reshape(-1,3) for c in row]) for row in case.groups]
        from scipy.spatial import ConvexHull
        search.mandatory=[np.vstack([root,np.c_[xy[ConvexHull(xy).vertices],np.zeros(len(ConvexHull(xy).vertices))]]) for root,xy in zip(search.root_points,case.demands)]
        search.raw=[];search.padded=[];search.sweep_meshes=[];search.sweep_conditioning=[];search.screen_count=0;search.boolean_count=0;search.attempts=[];search.optimization=[];search.placed_geometry={}
        search.precompute()
        mandatory,_,self.forbidden=search.installed_geometry(self.bases,self.offsets)
        initial=D.enlarged(search.envelope(self.bases,self.offsets),.05)
        self.mask=(F.bounded_space(initial,self.bases,self.offsets)-self.forbidden)+F.union(mandatory)
        self.navigation_window=initial;self.reference=S.unpack(self.mask);self.foot_reference=self.reference.copy()
        self.reference_report=dict(placement=placement,space_budget=search.envelope(self.bases,self.offsets),reference_role='navigation domain only; not an accepted constructed support')
        self.guard=A.Guard(case,placement)
        sphere=trimesh.creation.icosphere(subdivisions=2,radius=.0026);self.bead=sphere.vertices
        self.guaranteed_radius=float(np.min(np.abs(np.einsum('ij,ij->i',sphere.face_normals,sphere.triangles[:,0]))))
        self.terminals=[];self.feet=[];self.paths=[];self.original_roots=[];self.original_soles=[];self.thickness=[];self.core_solids=[]
        self.seed_positions=[];self.segments=[];self.grid_ready=False;self.journal=[];self.partial_steps=0;self.occupied_lo=None;self.occupied_hi=None
        self.contact_vertices=np.unique(np.vstack([c['triangles_m'].reshape(-1,3)@b+o for contacts,b,o in zip(case.groups,self.bases,self.offsets) for c in contacts]),axis=0)
        self.buried_contact_rejections=0;self.timings={'setup':time.monotonic()-started}

def placements(case,directions):
    """Keep the same absolute orientation; search XY seating translations only."""
    m=len(case.tasks);bases=np.repeat(np.eye(3)[None],m,axis=0)
    search=D.Search.__new__(D.Search);search.out=case.pair/'step4/data/boxed_support';search.out.mkdir(parents=True,exist_ok=True)
    search.case=case;search.directions=directions;search.raw=[];search.padded=[];search.sweep_meshes=[];search.sweep_conditioning=[];search.precompute()
    roots=case.root_solids
    def compatible(offsets,k):
        for j in range(k):
            for a,b in [(j,k),(k,j)]:
                body=F.transform(roots[a],np.eye(3),offsets[a]-offsets[b])
                if abs(float((body^search.padded[b]).volume()))*S.SCALE**3 > F.VOLUME_TOL_M3:return False
        return True
    def cost(offsets):
        roots_fixture=np.vstack([S.unpack(r).vertices+o for r,o in zip(roots,offsets)])
        demands=np.vstack([np.c_[xy,np.zeros(len(xy))]+o for xy,o in zip(case.demands,offsets)])
        fixture=np.vstack([roots_fixture,demands])
        return B.box([t.domain.mesh.vertices for t in case.tasks]+[fixture-o for o in offsets])['box_volume_cm3']
    for radius in [.02,.04,.08,.12,.18]:
        grid=[0,-radius,-3*radius/4,-radius/2,-radius/4,radius/4,radius/2,3*radius/4,radius]
        beam=[np.zeros((m,3))]
        for k in range(1,m):
            children=[]
            for parent in beam:
                for x in grid:
                    for y in grid:
                        trial=parent.copy();trial[k]=[x,y,0]
                        if compatible(trial,k):children.append((cost(trial),tuple(trial.ravel()),trial))
            if not children:beam=[];break
            beam=[t for _,_,t in sorted(children,key=lambda r:r[:2])[:3]]
        if beam:
            for offsets in beam:
                p=dict(bases=bases.tolist(),offsets=offsets.tolist(),directions=directions.tolist(),object_exit_mode=True,absolute_object_exit=(-directions).tolist())
                try:A.Guard(case,p)
                except A.AccessRejected:continue
                yield p
            return

def joint_layouts(group,problems,pools,exit_vector):
    """Choose translations and contacts together, retaining only shared-free roots."""
    m=len(problems);case=SimpleNamespace(tasks=problems,poses=[p.pose for p in problems],object_exit_mode=True,root_solids=[F.md.Manifold()]*m)
    search=D.Search.__new__(D.Search);search.out=group/'step4/data/boxed_support';search.out.mkdir(parents=True,exist_ok=True)
    search.case=case;search.directions=np.repeat((-exit_vector)[None],m,axis=0)
    search.raw=[];search.padded=[];search.sweep_meshes=[];search.sweep_conditioning=[];search.precompute()
    solids=[[F.union([S.solid(G.hull_mesh(v)) for v in e['cells']]) for e in row] for row in pools]
    compatible_cache={};force_cache=[{} for _ in problems]
    demands=[pressure_centers(p.targets/p.scale,p.domain.com)[0] for p in problems]
    def compatible(a,b,delta):
        key=(a,b,tuple(np.round(delta,8)))
        if key not in compatible_cache:
            compatible_cache[key]={i for i,solid in enumerate(solids[a]) if abs(float((F.transform(solid,np.eye(3),delta)^search.padded[b]).volume()))*S.SCALE**3<=F.VOLUME_TOL_M3}
        return compatible_cache[key]
    def force(k,ids):
        if not ids:return False
        key=tuple(sorted(ids))
        if key not in force_cache[k]:
            p=problems[k];probe=np.linspace(0,len(p.targets)-1,96,dtype=int)
            covered,_=J.classify(p.supply([pools[k][i]['contact'] for i in key]),p.targets[probe])
            force_cache[k][key]=bool(covered.all())
        return force_cache[k][key]
    def cost(offsets):
        fixture=np.vstack([np.c_[xy,np.zeros(len(xy))]+o for xy,o in zip(demands[:len(offsets)],offsets)])
        return B.box([p.domain.mesh.vertices for p in problems]+[fixture-o for o in offsets])['box_volume_cm3']
    for radius in [.02,.04,.08,.12,.18]:
        grid=np.array([-radius,-radius/2,0,radius/2,radius])
        beam=[([np.zeros(3)],[set(range(len(pools[0])))])]
        for k in range(1,m):
            children=[]
            for offsets,allowed in beam:
                for x in grid:
                    for y in grid:
                        current=np.array([x,y,0.]);trial=[set(s) for s in allowed]+[set(range(len(pools[k])))]
                        for j in range(k):
                            trial[j]&=compatible(j,k,offsets[j]-current)
                            trial[k]&=compatible(k,j,current-offsets[j])
                        if not all(force(j,ids) for j,ids in enumerate(trial)):continue
                        os=offsets+[current];children.append((cost(os),tuple(np.array(os).ravel()),os,trial))
            if not children:beam=[];break
            beam=[(os,ids) for _,_,os,ids in sorted(children,key=lambda r:r[:2])[:3]]
        if beam:
            certified=[]
            for os,allowed in beam:
                if all(verify_heads(problems[k],[pools[k][i] for i in sorted(ids)],{}) for k,ids in enumerate(allowed)):certified.append((os,allowed))
            beam=certified
        if beam:
            print('JOINT FEASIBLE',group.name,radius,[cost(os) for os,_ in beam],flush=True)
            return [(np.array(os),[[pools[k][i] for i in sorted(ids)] for k,ids in enumerate(allowed)]) for os,allowed in beam]
    raise RuntimeError('No common-world contact/translation layout in bounded search')

def run(name,poses,exit_vector=np.array([0.,0.,1.])):
    started=time.monotonic();group=OUT/name;group.mkdir(parents=True,exist_ok=True)
    problems=read_problems(name,poses);all_entries=[];source_paths=[];certificates=[]
    pools=[]
    for problem in problems:
        pool,source=candidates(problem,exit_vector)
        source_paths.extend([source,source.with_suffix('.json')]+list(problem.inputs))
        if not pool:raise RuntimeError('No legal world-direction contact: '+problem.pose)
        counts={c:sum(c in e['components'] for e in pool) for c in set(c for e in pool for c in e['components'])}
        component=max(counts,key=counts.get);pools.append([e for e in pool if component in e['components']])
    joint=joint_layouts(group,problems,pools,exit_vector)
    joint_offsets,pools=joint[0]
    for k,pool in enumerate(pools):print('JOINT POOL',problems[k].pose,len(pool),flush=True)
    for problem,entries in zip(problems,pools):
        entries.sort(key=lambda e:(e['alignment'],e['contact']['candidate_id']))
        cache_path=group/'step3'/f'forces_{problem.pose}.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
        # Explore direction-compatible completions. Spatial spread, not head
        # count, orders solutions; mechanics is a terminal hard condition.
        rng=np.random.default_rng(3100+int(problem.pose.split('_')[1]))
        orders=[entries]
        points=np.array([e['contact']['center_m'] for e in entries])
        for trial in range(12):
            anchor=points[rng.integers(len(points))]
            noise=rng.normal(0,.22,len(entries))
            score=np.array([e['alignment'] for e in entries])+noise+2*np.linalg.norm(points-anchor,axis=1)
            orders.append([entries[i] for i in np.argsort(score)])
        complete=[]
        for order in orders:
            for count in sorted(set(list(range(4,len(entries)+1,4))+[len(entries)])):
                trial=order[:count]
                if verify_heads(problem,trial,cache):
                    cloud=np.vstack([e['contact']['triangles_m'].reshape(-1,3) for e in trial])
                    footprint=float(np.prod(np.maximum(np.ptp(cloud[:,:2],axis=0),.001)))
                    angle=float(np.average([e['alignment'] for e in trial],weights=[I.area([e['contact']]) for e in trial]))
                    complete.append((footprint,angle,trial));break
        save(cache_path,cache)
        if not complete:raise RuntimeError('World-direction set does not cover loads: '+problem.pose)
        selected=min(complete,key=lambda row:row[:2])[2]
        all_entries.append(selected)
        key=','.join(sorted(e['contact']['candidate_id'] for e in selected));certificates.append(cache[key])
        print('DIRECTION FORCE PASS',name,problem.pose,'heads',len(selected),flush=True)
    demands=[pressure_centers(p.targets/p.scale,p.domain.com)[0] for p in problems]
    contacts=[[e['contact'] for e in es] for es in all_entries]
    cells=[[e['cells'] for e in es] for es in all_entries]
    case=SimpleNamespace(name='B',poses=poses,pair=group,output=group/'step4/data/growing_support',tasks=problems,groups=contacts,heads=cells,support_seeds=cells,
        root_solids=[F.union([S.solid(G.hull_mesh(v)) for e in es for v in e['cells']]) for es in all_entries],demands=demands,paths=source_paths,
        object_exit_mode=True,schedule=dict(passed=True,covered_counts=[32768]*len(problems),head_model=Z.MODEL))
    plan_path=group/'step3/plan.json'
    plan=dict(complete=True,object='B',poses=poses,head_count_optimized=False,force_alignment=[float(np.average([e['alignment'] for e in es],weights=[I.area([e['contact']]) for e in es])) for es in all_entries],
        selected_ids=[[c['candidate_id'] for c in g] for g in contacts],force_certificates=certificates,absolute_object_exit=exit_vector.tolist(),coordinate_frame='unchanged task-world axes; no conversion into object coordinates',
        object_withdrawal=True,relative_support_direction=(-exit_vector).tolist())
    save(plan_path,plan);case.paths.append(plan_path)
    for k,p in enumerate(problems):
        original=[path for path in p.inputs if Path(path).name=='needs.json'][0]
        target=group/'step4/data/source_inputs'/p.pose/'needs.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(original).read_bytes())
    failures=[];successful=[]
    fixed=dict(bases=np.repeat(np.eye(3)[None],len(poses),axis=0).tolist(),offsets=joint_offsets.tolist(),directions=np.repeat((-exit_vector)[None],len(poses),axis=0).tolist(),object_exit_mode=True,absolute_object_exit=np.repeat(exit_vector[None],len(poses),axis=0).tolist())
    for n,placement in enumerate([fixed]):
        print('ABS PLACEMENT',name,n,placement['offsets'],flush=True)
        try:
            growth=AbsoluteGrow(group,case,placement);report=growth.run(pitch=.008)
            report['construction']['method']='World-direction heads, common fixed orientation, envelope-guided support growth'
            report['previous_material_volume_cm3']=None;report['material_reduction_percent']=None
            report['construction']['reference_used_as']='navigation window only; no preconstructed support'
            report['direction_objective']=plan
            report['provenance']['code'].update(I.hashes([Path(__file__),Path(W.__file__)]))
            save(growth.out/'report.json',report)
            from step5_evaluate.evaluate import evaluate
            evaluation=evaluate(group)
            save(group/'comparison.json',dict(complete=True,passed=True,group=name,step5=evaluation['metrics'],head_count=sum(map(len,contacts)),elapsed_s=time.monotonic()-started))
            archive=group/'step4/data/direction_trials'/str(n)
            if archive.exists():shutil.rmtree(archive)
            shutil.copytree(growth.out,archive)
            successful.append((evaluation['metrics']['object_and_support_poses']['box_volume_cm3'],n,archive))
        except (RuntimeError,ValueError,AssertionError) as error:
            failures.append(dict(placement=n,error=f'{type(error).__name__}: {error}',checks=getattr(error,'checks',None)));save(group/'failures.json',failures)
            print('ABS ATTEMPT FAILED',name,error,flush=True)
    if successful:
        _,chosen,archive=min(successful,key=lambda r:r[:2])
        shutil.copytree(archive,group/'step4/data/growing_support',dirs_exist_ok=True)
        from step5_evaluate.evaluate import evaluate
        evaluation=evaluate(group)
        save(group/'comparison.json',dict(complete=True,passed=True,group=name,step5=evaluation['metrics'],head_count=sum(map(len,contacts)),elapsed_s=time.monotonic()-started,
            construction_trials=[dict(trial=i,volume_cm3=v) for v,i,_ in successful],selected_trial=chosen,head_count_penalty=0))
        return evaluation
    raise RuntimeError(f'No accepted shared support for {name}: {failures}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',default=None);a=p.parse_args()
    if a.groups is None:a.groups=[name for name,_ in specs()]
    OUT.mkdir(parents=True,exist_ok=True);install_recorded_recovery(OUT)
    lookup=dict(specs());rows=[]
    for name in a.groups:
        try:r=run(name,lookup[name]);rows.append(dict(group=name,passed=True,volume_cm3=r['metrics']['object_and_support_poses']['box_volume_cm3']))
        except Exception as e:
            import traceback;traceback.print_exc();rows.append(dict(group=name,passed=False,error=str(e)))
            save(OUT/name/'comparison.json',dict(complete=True,passed=False,error=str(e)))
        save(OUT/'progress.json',dict(complete=False,results=rows))
    save(OUT/'progress.json',dict(complete=True,results=rows))
if __name__=='__main__':main()
