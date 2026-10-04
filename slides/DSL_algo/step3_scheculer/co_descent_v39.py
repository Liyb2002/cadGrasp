"""SRD-inspired joint local search of real heads and a continuous common exit.
Derivative-free stochastic pattern search, not autodiff through greedy Step4.
All edits are atomic; only full-load/full-fixture actual-volume improvements commit.
"""
import argparse,hashlib,json,time,shutil,traceback
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import trimesh
from codes.precompute_objects import head_cache as HC
from codes.precompute_objects.heads.surface import FLOOR_CLEARANCE_M
from step3_scheculer import contacts as I,contact_dsl as D,operation_dsl as F
from step3_scheculer import shared_direction_paths as P,additive_gpu_v24 as GPU
from step3_scheculer.shared_direction_search import bounded_lps,unit
from step3_scheculer import build_shared_fixture_v35 as BUILD
from step3_scheculer.feasible_seating import restore
from step3_scheculer.initialize_current_v16 import current_task
from step3_scheculer.compare_shared_volume_v33 import GROUPS
from step5_current.evaluate import measure

STAGE='co_descent_v39'


def tangent_basis(normal):
    n=unit(normal);a=np.eye(3)[np.argmin(np.abs(n))];u=unit(a-n*(a@n));return u,np.cross(n,u)


def turn(direction,angle,azimuth):
    d=unit(direction);u,v=tangent_basis(d)
    return unit(np.cos(angle)*d+np.sin(angle)*(np.cos(azimuth)*u+np.sin(azimuth)*v))


def paths(direction,count):return tuple(F.ray(-unit(direction),0) for _ in range(count))


def contact_key(row):
    return hashlib.sha256(b''.join(c['triangles_m'].tobytes()+c['source_faces'].tobytes() for c in row)).hexdigest()


def improves(old,new):return bool(np.isfinite(new) and new<old-max(1e-6,abs(old)*1e-7))


class Optimizer:
    def __init__(self,group,config):
        self.started=time.monotonic();self.group=group;self.config=config
        self.base=I.OUTPUTS/'B'/group;prior_stage='shared_fixture_seating_v35' if group in ('pose1+3','pose3+6') else 'shared_fixture_seating_v34'
        self.seed_folder=self.base/'step4'/prior_stage;self.seed_path=self.seed_folder/'report.json';self.seed_report=I.check_report(self.seed_path)
        self.seed_metric=I.check_report(self.base/'step5_evaluate'/prior_stage/'report.json')
        saved=json.loads((self.seed_folder/'state.json').read_text());self.tasks=[current_task('B',p) for p in saved['poses']]
        rows=tuple(tuple(I.read_contacts(self.seed_folder/f'contacts_{t.pose}.npz')) for t in self.tasks)
        self.state=F.State(rows,np.asarray(saved['placement']['bases']),np.asarray(saved['placement']['offsets']),tuple(saved['exit_paths']))
        self.initial_state=self.state;self.direction=unit(self.state.paths[0]['initial_object_exit_world'])
        self.cost=self.initial_cost=self.seed_metric['aggregate']['object_and_support_poses']['box_volume_cm3'];self.witness=self.seed_folder
        self.body=self.seed_report;self.mesh=trimesh.load(self.seed_folder/'shape.obj',force='mesh',process=False)
        self.compilers=[D.Compiler(t,[t],clearance=FLOOR_CLEARANCE_M,device='cpu') for t in self.tasks]
        self.backends=[GPU.GPUClassifier(t.targets,config['device']) for t in self.tasks]
        self.analyzers=[P.PathAnalyzer(t.domain.mesh,F.ROOT_DEPTH) for t in self.tasks]
        self.pools=[];self.force_cache={};self.events=[];self.serial=0
        for i,t in enumerate(self.tasks):
            cache=HC.load(HC.ROOT/'objects/B/poses'/t.pose,200,strict=True)
            self.pools.append([e['contact'] for pool in cache['pools'].values() for e in pool if e['valid']])
        # The initial native contacts already have full original-load proof.
        for i,row in enumerate(rows):self.force_cache[i,contact_key(row)]=dict(passed=True,covered=32768,authority='hash-verified accepted initial Step3 proof')
        self.out=self.base/'step3_scheculer'/STAGE;self.out.mkdir(parents=True,exist_ok=True)
        self.rng=np.random.default_rng(config['seed']+int(hashlib.sha256(group.encode()).hexdigest()[:8],16))

    def compile(self,i,contact,center,radius):
        self.serial+=1;c=self.compilers[i]
        patch=D.Patch(100000+self.serial,tuple(center),float(radius));head=c.compile_patch(patch)
        for _ in range(4):
            if head is None:return None
            area=float(head['triangle_areas_m2'].sum())
            if area<=.02*c.mesh.area*(1+1e-10):break
            patch=replace(patch,radius=patch.radius*np.sqrt(.02*c.mesh.area/area)*.999)
            head=c.compile_patch(patch)
        if head is None or head['triangle_areas_m2'].sum()>.02*c.mesh.area*(1+1e-10):return None
        # Physical identity remains stable when its actual patch moves. Moments
        # are rebuilt from actual triangles by task.supply; no cached old rays.
        return dict(head,candidate_id=contact['candidate_id'],candidate_index=contact['candidate_index'])

    def joint(self,state,angle,azimuth,i,j,sign=1):
        d=turn(state.paths[0]['initial_object_exit_world'],angle,azimuth)
        c=state.groups[i][j];task=self.tasks[i];normal=task.domain.mesh.face_normals[c['center_face']]
        displacement=d-unit(state.paths[0]['initial_object_exit_world']);displacement-=normal*(displacement@normal)
        if np.linalg.norm(displacement)<1e-10:
            u,v=tangent_basis(normal);displacement=np.cos(azimuth)*u+np.sin(azimuth)*v
        local_scale=min(1.,abs(angle)/np.deg2rad(3.7))
        step=min(float(task.domain.mesh.extents.max())*.003,float(c['radius_m'])*.15)*sign*local_scale
        center=c['center_m']+step*unit(displacement)
        radius=float(c['radius_m'])*(1.+.02*sign*local_scale)
        head=self.compile(i,c,center,radius)
        if head is None:return None
        groups=[list(row) for row in state.groups];groups[i][j]=head
        changed=float(np.linalg.norm(head['center_m']-c['center_m']))
        if changed<1e-10 and abs(head['triangle_areas_m2'].sum()-c['triangle_areas_m2'].sum())<1e-14:return None
        offsets=state.offsets.copy()
        if sign>0:
            anchor=offsets[:,:2].mean(axis=0)
            offsets[:,:2]=anchor+(offsets[:,:2]-anchor)*(1.-.015*local_scale)
        return replace(state,groups=tuple(tuple(row) for row in groups),paths=paths(d,len(groups)),offsets=offsets)

    def rewrite(self,kind):
        state=self.state;eligible=[i for i,row in enumerate(state.groups) if len(row)>1]
        i=int(self.rng.choice(eligible));row=list(state.groups[i]);j=int(self.rng.integers(len(row)))
        if kind=='delete':row.pop(j)
        elif kind in ('add','replace'):
            available=[c for c in self.pools[i] if not any(np.linalg.norm(c['center_m']-a['center_m'])<1e-7 for a in row)]
            available.sort(key=lambda c:np.linalg.norm(c['center_m']-row[j]['center_m']))
            if not available:return None
            chosen=available[int(self.rng.integers(min(8,len(available))))]
            chosen={k:v for k,v in chosen.items() if k not in ('wrench_generators','wrench_com_m')}
            if kind=='replace':row[j]=chosen
            else:
                if len(row)>=self.config['max_heads']:return None
                row.append(chosen)
        elif kind=='merge':
            pairs=[(np.linalg.norm(a['center_m']-b['center_m']),x,y) for x,a in enumerate(row) for y,b in enumerate(row) if x<y and a['triangle_areas_m2'].sum()+b['triangle_areas_m2'].sum()<=.02*self.tasks[i].domain.mesh.area]
            if not pairs:return None
            _,x,y=min(pairs);a,b=row[x],row[y]
            center=(a['center_m']+b['center_m'])/2;head=self.compile(i,a,center,np.hypot(a['radius_m'],b['radius_m']))
            if head is None:return None
            row=[c for z,c in enumerate(row) if z not in (x,y)]+[head]
        groups=list(state.groups);groups[i]=tuple(row)
        return replace(state,groups=tuple(groups)),i

    def native_check(self,state):
        records=[]
        for i,(task,row,path,a) in enumerate(zip(self.tasks,state.groups,state.paths,self.analyzers)):
            exitcheck=a.test(a.heads(row),path)
            if not exitcheck['clear']:return False,dict(pose=task.pose,reason='native_full_head_exit',check=exitcheck)
            key=(i,contact_key(row))
            if key not in self.force_cache:
                mask,info=self.backends[i].classify(task.supply(row))
                self.force_cache[key]=dict(passed=bool(mask.all()),covered=int(mask.sum()),authority='all original loads; certified primal/dual GPU batching',info=info)
            check=self.force_cache[key];records.append(dict(pose=task.pose,**check))
            if not check['passed']:return False,dict(pose=task.pose,reason='original_load_or_no_uplift_failure',check=check)
        return True,dict(passed=True,load_subsampling=False,per_pose=records)

    def evaluate(self,state,operation,round_number):
        index=len(self.events);folder=self.out/'proposals'/f'{index:03d}';folder.mkdir(parents=True,exist_ok=True)
        event=dict(index=index,round=round_number,operation=operation,direction=state.paths[0]['initial_object_exit_world'],accepted=False)
        event['changed_centers']=[float(np.linalg.norm(c['center_m']-next((a['center_m'] for a in old if a['candidate_id']==c['candidate_id']),c['center_m']))) for row,old in zip(state.groups,self.state.groups) for c in row]
        event['heads']=[len(row) for row in state.groups];event['starting_volume_cm3']=self.cost
        try:
            passed,check=self.native_check(state)
            event['native_check']=check
            if not passed:return self.record(event)
            F.save_state(folder,self.tasks,state,check)
            proposal=dict(complete=True,passed=True,scope='current proposed native contact programs only; not yet full fixture',load_count_per_pose=32768,force_check=check,common_world_exit=state.paths[0]['initial_object_exit_world'],provenance=dict(inputs=I.hashes([self.seed_path]+[p for t in self.tasks for p in t.inputs]),code=I.hashes([Path(__file__),Path(D.__file__),Path(GPU.__file__)])))
            I.save(folder/'proposal.json',proposal)
            # Check the entire shared fixture at the current free seating first.
            # Only repair translations when the exact constructor rejects it.
            for placement in range(self.config['placement_attempts']):
                seating=state
                if placement:
                    repaired=restore(self.tasks,state,F.roots,attempt=placement-1,iterations=30)
                    if repaired is None:continue
                    seating=repaired[0]
                result=BUILD.attempt(self.group,self.tasks,seating,folder/'proposal.json',folder/f'placement_{placement}',operation,[],False)
                if result is None:continue
                grow,body,mesh=result
                aggregate,_,_=measure([t.domain.mesh.vertices for t in self.tasks],mesh.vertices,seating.bases,seating.offsets)
                volume=aggregate['object_and_support_poses']['box_volume_cm3']
                tolerance=max(1e-6,abs(self.cost)*1e-7)
                accepted=improves(self.cost,volume) or (abs(volume-self.cost)<=tolerance and improves(float(self.body['volume_cm3']),float(body['volume_cm3'])))
                event.update(full_fixture_passed=True,volume_cm3=volume,material_cm3=body['volume_cm3'],accepted=accepted,placement_trial=placement,acceptance_objective='actual occupied XYZ box first; actual material volume breaks numerical ties')
                if event['accepted']:
                    self.state=seating;self.direction=unit(seating.paths[0]['initial_object_exit_world']);self.cost=volume
                    self.witness=folder/f'placement_{placement}';self.body=body;self.mesh=mesh
                break
            else:event['full_fixture_passed']=False
        except (RuntimeError,ValueError,AssertionError) as error:event.update(error=str(error),numerical_or_geometry_rejection=True)
        return self.record(event)

    def record(self,event):
        self.events.append(event);I.save(self.out/'progress.json',dict(complete=False,current_volume_cm3=self.cost,events=self.events))
        print('CO-DESCENT',self.group,event['index'],event['operation'],np.round(event['direction'],5),'accepted',event['accepted'],'volume',event.get('volume_cm3'),'reason',event.get('native_check',{}).get('reason',event.get('error')),flush=True)
        return event['accepted']

    def run(self):
        angle=np.deg2rad(self.config['angle_degrees'])
        with bounded_lps():
            for r in range(self.config['rounds']):
                before=(self.cost,float(self.body['volume_cm3']))
                # Continuous direction/head updates are evaluated jointly, never
                # by a direction-only eligibility query into the cached menu.
                for q in range(self.config['joint_trials']):
                    i=(q+r)%len(self.tasks);j=int(self.rng.integers(len(self.state.groups[i])))
                    azimuth=float(self.rng.uniform(0,2*np.pi));trial=self.joint(self.state,angle,azimuth,i,j,1 if q%2==0 else -1)
                    if trial is not None:self.evaluate(trial,'continuous_direction_and_head',r)
                kinds=('delete','replace') if r%2==0 else ('add','merge')
                for kind in kinds[:self.config['rewrite_trials']]:
                    rewritten=self.rewrite(kind)
                    if rewritten is None:continue
                    state,i=rewritten
                    # Judge the rewrite after a joint continuous local step.
                    # Never reject the raw rewrite before its parameters move.
                    for inner in range(self.config['rewrite_local_steps']):
                        j=int(self.rng.integers(len(state.groups[i])))
                        trial=self.joint(state,angle*.5,float(self.rng.uniform(0,2*np.pi)),i,j,1 if inner%2==0 else -1)
                        if trial is not None and self.evaluate(trial,kind+'_then_joint_step',r):break
                angle=angle*.7 if (self.cost,float(self.body['volume_cm3']))<before else angle*.5
        return self.publish()

    def publish(self):
        out=self.base/'step4'/STAGE;step5=self.base/'step5_evaluate'/STAGE;out.mkdir(parents=True,exist_ok=True);step5.mkdir(parents=True,exist_ok=True)
        for name in ('shape.obj','geometry_certificate.npz','roots.obj','seed_targets.obj','partial_growth.obj','shared_tree.obj'):
            if (self.witness/name).exists():shutil.copy2(self.witness/name,out/name)
        F.save_state(out,self.tasks,self.state,dict(passed=True,all_original_loads_passed=True))
        aggregate,per_pose,_=measure([t.domain.mesh.vertices for t in self.tasks],self.mesh.vertices,self.state.bases,self.state.offsets)
        BUILD.pictures(out,step5,self.tasks,self.state,self.mesh,None,aggregate,self.group)
        report=dict(self.body,schema=STAGE,group=self.group,initial_volume_cm3=self.initial_cost,final_volume_cm3=self.cost,volume_saved_percent=100*(1-self.cost/self.initial_cost),common_object_exit_world=self.direction.tolist(),co_descent_events=self.events,config=self.config,seconds=time.monotonic()-self.started,initial_heads=[len(x) for x in self.initial_state.groups],final_heads=[len(x) for x in self.state.groups],direction_is_continuous=True,cached_direction_menu_used=False,gradient_method='derivative-free coupled local pattern search; no autodiff claim',global_optimality_claim=False,accepted_updates=sum(e['accepted'] for e in self.events),object_poses_changed=False,provenance=dict(inputs=I.hashes([self.seed_path,self.witness/'report.json',out/'state.json']),code=I.hashes([Path(__file__),Path(D.__file__),Path(BUILD.__file__),Path(GPU.__file__)])),artifacts={n:I.sha256(out/n) for n in ['shape.obj','geometry_certificate.npz','state.json','overview.png','construction_steps.png']})
        I.save(out/'report.json',report);I.check_report(out/'report.json')
        metric=dict(complete=True,passed=True,group=self.group,poses=[t.pose for t in self.tasks],aggregate=aggregate,per_pose=per_pose,material_volume_cm3=self.body['volume_cm3'],common_object_exit_world=self.direction.tolist(),initial_volume_cm3=self.initial_cost,volume_saved_percent=report['volume_saved_percent'],provenance=dict(inputs=I.hashes([out/'report.json',out/'shape.obj']),code=I.hashes([Path(__file__)])),artifacts={'overview.png':I.sha256(step5/'overview.png')})
        I.save(step5/'report.json',metric);I.check_report(step5/'report.json')
        summary={k:report[k] for k in ['group','initial_volume_cm3','final_volume_cm3','volume_saved_percent','common_object_exit_world','initial_heads','final_heads','seconds','accepted_updates']}
        summary['passed']=True;print('CO-DESCENT DONE',json.dumps(summary),flush=True);return summary


def run(group,config):return Optimizer(group,config).run()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sets',nargs='+',default=GROUPS);p.add_argument('--jobs',type=int,default=1);p.add_argument('--device',default='cuda');p.add_argument('--rounds',type=int,default=2);p.add_argument('--joint-trials',type=int,default=6);p.add_argument('--rewrite-trials',type=int,default=2);p.add_argument('--rewrite-local-steps',type=int,default=2);p.add_argument('--placement-attempts',type=int,default=2);p.add_argument('--angle-degrees',type=float,default=3.7);p.add_argument('--max-heads',type=int,default=10);p.add_argument('--seed',type=int,default=20261004);a=p.parse_args()
    if min(a.jobs,a.rounds,a.joint_trials,a.rewrite_local_steps,a.placement_attempts,a.max_heads)<1 or a.rewrite_trials<0 or not 0<a.angle_degrees<90:p.error('Invalid budget')
    config={k:getattr(a,k) for k in ('device','rounds','joint_trials','rewrite_trials','rewrite_local_steps','placement_attempts','angle_degrees','max_heads','seed')};began=time.monotonic()
    if a.jobs==1:results=[run(g,config) for g in a.sets]
    else:
        with ProcessPoolExecutor(a.jobs) as pool:
            results=[future.result() for future in as_completed([pool.submit(run,g,config) for g in a.sets])]
    output=I.OUTPUTS/'B'/a.sets[-1]/'step5_evaluate'/STAGE;output.mkdir(parents=True,exist_ok=True)
    I.save(output/'batch.json',dict(complete=True,groups=results,wall_seconds=time.monotonic()-began,config=config))
