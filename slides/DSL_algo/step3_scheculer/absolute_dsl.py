"""Absolute fixture-frame direction search; exact occupied volume commits.

Head count is diagnostic only. Qualified incumbents survive every failed trial.
"""
from pathlib import Path
from dataclasses import replace
import itertools,json,shutil,time
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from step3_scheculer import operation_dsl as F,absolute_metrics as M
from step3_scheculer import operation_navigation_recovery as R
from step3_scheculer.review_operation_dsl import load,render_geometry
from step3_scheculer.feasible_seating import restore
from step3_scheculer.run_dsl import saved_task


def unit(v):
    v=np.asarray(v,float);n=np.linalg.norm(v)
    return v/max(n,1e-15)


def directions(tasks,state):
    force=[];exits=[]
    for t,row,b,p in zip(tasks,state.groups,state.bases,state.paths):
        # Reaction on the object, not the outward object normal. Native row
        # vectors map directly into the fixed fixture, not through T_world_mesh.
        f=sum((-t.domain.mesh.face_normals[c['source_faces']]*c['triangle_areas_m2'][:,None]).sum(axis=0) for c in row)
        force.append(unit(f@b));exits.append(unit(np.asarray(p['initial_object_exit_world'])@b))
    return np.asarray(force),np.asarray(exits)


def spread(vectors):
    return float(np.mean([np.degrees(np.arccos(np.clip(a@b,-1,1))) for a,b in itertools.combinations(vectors,2)])) if len(vectors)>1 else 0.


def diagnostic(tasks,state):
    f,e=directions(tasks,state)
    return dict(force_mean_pair_angle_deg=spread(f),exit_mean_pair_angle_deg=spread(e),reaction_directions_fixture=f.tolist(),object_exit_directions_fixture=e.tolist())


def activate():
    R.activate();original=F.sources
    F.STAGE='dsl_absolute';F.BODY_STAGE='dsl_absolute_support';F.SCHEMA='absolute_direction_dsl_v7'
    F.sources=lambda:list(dict.fromkeys(original()+M.sources()+[Path(__file__),Path(__file__).with_name('run_absolute_dsl.py')]))
    F.objective=lambda tasks,state:(diagnostic(tasks,state)['force_mean_pair_angle_deg'],diagnostic(tasks,state)['exit_mean_pair_angle_deg'])
    F.angle_loss=lambda tasks,state:diagnostic(tasks,state)['exit_mean_pair_angle_deg']


class Optimizer(F.BaseOptimizer):
    def __init__(self,group,tasks,config):
        self.group,self.tasks,self.config=group,tasks,config
        self.out=group/'step3_scheculer'/F.STAGE;self.out.mkdir(parents=True,exist_ok=True)
        self.checker=F.Checker(tasks);self.events=[];self.trial=0
        self.source=group/'step3_scheculer/dsl_operations/final'
        self.seed,_=load(self.source,tasks)
        self.reference=self.source/'shape.obj';self.reference_report=F.I.check_report(self.source/'report.json')
        self.seed_inputs=[self.source/'report.json',self.source/'state.json']+[self.source/f'contacts_{t.pose}.npz' for t in tasks]
        self.incumbent=self.initial_state=self.seed;self.check=self.checker.check(self.seed)
        assert self.check['passed']
        self.initial_volume=M.volume(tasks,self.seed,trimesh.load(self.reference,force='mesh',process=False))
        self.best_volume=self.initial_volume;self.witness=(self.source,self.reference_report,None)
        F.save_state(self.out/'initial',tasks,self.seed,self.check)
        self.event('initialize_qualified',volume_cm3=self.initial_volume,directions=diagnostic(tasks,self.seed))

    def construct(self,*args,**kwargs):
        witness=super().construct(*args,**kwargs)
        if witness:
            folder,report,grow=witness
            # DirectGrow's inherited comparison was a navigation mask. Use
            # actual material instead, and record the actual 2.6 mm bead.
            report['previous_volume_cm3']=float(trimesh.load(self.reference,force='mesh',process=False).volume*1e6)
            report['volume_reduction_percent']=100*(1-report['volume_cm3']/report['previous_volume_cm3'])
            report.setdefault('construction',{})['beam_radius_m']=.0026
            F.I.save(folder/'report.json',report)
        return witness

    def consider(self,state,kind,restore_placement=False,attempt=0):
        try:
            check=self.checker.check(state)
            if not check['passed']:
                self.event('reject_local',proposal=kind,checks=check);return None
            if restore_placement:
                state=F.seat(self.tasks,state,.008)
                if state is None:return None
                seated=restore(self.tasks,state,F.roots,attempt=attempt)
                if seated is None:
                    self.event('reject_seating',proposal=kind);return None
                state,cuts=seated
            witness=self.construct(state,check,kind)
            if witness is None:return None
            metrics=M.evaluate(self.group,self.tasks,state,witness[0]/'shape.obj',witness[0]/'report.json',witness[0]/'step5',render=False)
            value=metrics['metrics']['object_and_support_poses']['box_volume_cm3']
            before=self.best_volume
            accepted=value<before-1e-5
            self.event('measured_trial',proposal=kind,volume_cm3=value,best_before_cm3=before,accepted=accepted,
                directions=diagnostic(self.tasks,state),witness=str((witness[0]/'report.json').relative_to(F.I.ROOT)))
            if accepted:
                self.incumbent,self.witness,self.check=state,witness,check;self.best_volume=value
                F.save_state(self.out/'best',self.tasks,state,check)
            return state,witness,value
        except Exception as error:
            self.event('reject_proposal',proposal=kind,reason=str(error));return None

    def unshared(self,state):
        return replace(state,groups=tuple(tuple(dict(c,candidate_id=t.pose+'_ABS_'+str(j)) for j,c in enumerate(row)) for t,row in zip(self.tasks,state.groups)))

    def align(self):
        # Row-vector rotation around each contact centroid; whole contact and
        # object stance move together, so native force feasibility is invariant.
        start=self.unshared(self.incumbent)
        forces,exits=directions(self.tasks,start);target=unit(np.sum(forces+exits,axis=0))
        for alpha in (.25,.5,1.):
            bases=start.bases.copy();offsets=start.offsets.copy()
            for k,row in enumerate(start.groups):
                v=unit(forces[k]+exits[k]);cross=np.cross(v,target)
                if np.linalg.norm(cross)<1e-10:continue
                angle=np.arccos(np.clip(v@target,-1,1))*alpha
                rot=Rotation.from_rotvec(unit(cross)*angle).as_matrix().T
                center=np.vstack([c['triangles_m'].reshape(-1,3) for c in row]).mean(axis=0)
                installed=center@bases[k]+offsets[k]
                bases[k]=bases[k]@rot;offsets[k]=installed-center@bases[k]
            self.consider(replace(start,bases=bases,offsets=offsets),f'absolute_rotation_{alpha}',True)

    def vertical_heads(self,task,row,path):
        plan=F.ray([0,0,-1],700001)
        analyzer=F.E.PathAnalyzer(task.domain.mesh,F.ROOT_DEPTH)
        # Reuse exact qualified contacts when their actual exit is already clear.
        if analyzer.test(analyzer.heads(row),plan)['clear']:
            return row,plan
        c=F.restrict_compiler(F.D.Compiler(task,[task],device=self.config['device']),np.array([0,0,-1.]))
        seeds=c.seeds(self.config['seeds']);pool=[]
        for seed in seeds:
            for factor in (1.,1.414213562,.70710678):
                head=c.compile_patch(replace(seed,radius=seed.radius*factor))
                if head is not None and analyzer.test(analyzer.heads([head]),plan)['clear']:
                    pool.append(head);break
        if not pool:raise RuntimeError('No actual upward-exit contact candidates: '+task.pose)
        selected=[];sample=np.linspace(0,len(task.targets)-1,48,dtype=int)
        for iteration in range(min(24,len(pool))):
            losses=c.backend.solve([F.D.reduced_rays(task.supply(selected+[h])) for h in pool],task.targets[sample])
            # Direction coherence is a tie guide after force residuals, never a
            # fake normal or an equilibrium constraint relaxation.
            up=np.array([0,0,1.]);guide=[]
            for h in pool:
                v=unit((-task.domain.mesh.face_normals[h['source_faces']]*h['triangle_areas_m2'][:,None]).sum(axis=0))
                guide.append(1-v@up)
            score=losses.mean(axis=1)+losses.max(axis=1)+1e-6*np.asarray(guide)
            j=int(np.argmin(score));selected.append(pool.pop(j))
            mask,_=F.J.classify(task.supply(selected),task.targets)
            print('ABS UP FORCE',task.pose,len(selected),int(mask.sum()),flush=True)
            if mask.all():
                if analyzer.test(analyzer.heads(selected),plan)['clear']:return tuple(selected),plan
            failed=np.flatnonzero(~mask)
            if len(failed):sample=np.unique(np.r_[np.linspace(0,len(mask)-1,24,dtype=int),failed[np.linspace(0,len(failed)-1,min(72,len(failed)),dtype=int)]])
        raise RuntimeError('Upward-exit force search exhausted: '+task.pose)

    def common_up(self):
        rows=[];paths=[]
        for t,row,p in zip(self.tasks,self.seed.groups,self.seed.paths):
            heads,path=self.vertical_heads(t,row,p)
            rows.append(tuple(dict(c,candidate_id=t.pose+'_UP_'+str(j)) for j,c in enumerate(heads)));paths.append(path)
        n=len(rows);bases=np.tile(np.eye(3),(n,1,1))
        # Center actual contact regions in the common support plane. Exact
        # cross-pose sweep/floor seating then resolves incompatible overlap.
        offsets=np.array([-np.r_[np.vstack([c['triangles_m'].reshape(-1,3) for c in row]).mean(axis=0)[:2],0.] for row in rows])
        start=F.State(tuple(rows),bases,offsets,tuple(paths));branches=[]
        for attempt in (0,2,4):
            result=self.consider(start,'common_up_seating_'+str(attempt),True,attempt)
            if result:branches.append(result)
        if not branches:return
        branch=min(branches,key=lambda r:r[2])[0]
        # Compact valid branch even if its first body was larger than the
        # public incumbent. Every retained branch has its own full witness.
        for fraction in (.25,.5,.75):
            center=branch.offsets[:,:2].mean(axis=0)
            offsets=branch.offsets.copy();offsets[:,:2]=(1-fraction)*offsets[:,:2]+fraction*center
            result=self.consider(replace(branch,offsets=offsets),f'common_up_compact_{fraction}',True)
            if result and result[2]<min(r[2] for r in branches):branches.append(result);branch=result[0]

    def compact_existing(self):
        start=self.unshared(self.incumbent)
        # Translation gradient proposals: bring contact centers toward their
        # average, then enforce all floors and exact foreign withdrawal sweeps.
        centers=np.array([np.vstack([c['triangles_m'].reshape(-1,3) for c in row]).mean(axis=0)@b+o for row,b,o in zip(start.groups,start.bases,start.offsets)])
        for fraction in (.15,.35,.6):
            offsets=start.offsets+fraction*(centers.mean(axis=0)-centers)
            self.consider(replace(start,offsets=offsets),'contact_center_descent_'+str(fraction),True)

    def optimize(self):
        for operation in (self.align,self.common_up,self.compact_existing):
            try:operation()
            except Exception as error:self.event('search_branch_failed',proposal=operation.__name__,reason=str(error))
        # Always actually rerun current Step4 for the chosen state. If greedy
        # regrowth is larger, retain the already qualified smaller witness.
        self.consider(self.incumbent,'final_step4_regrowth')
        self.publish()

    def publish(self):
        state=self.incumbent;folder,body,grow=self.witness
        final=self.out/'final';F.save_state(final,self.tasks,state,self.check)
        for name in ('shape.obj','geometry_certificate.npz'):shutil.copy2(folder/name,final/name)
        witness=folder/'report.json';F.I.check_report(witness)
        report=dict(complete=True,passed=True,constructed=True,schema=F.SCHEMA,group=self.group.name,
            poses=[t.pose for t in self.tasks],physical_head_count=state.physical_count,shared_head_count=state.shared_count,
            head_count_optimized=False,objective='actual Step5 XYZ occupied box volume',
            initial_volume_cm3=self.initial_volume,final_volume_cm3=self.best_volume,
            reduction_percent=100*(1-self.best_volume/self.initial_volume),
            initial_directions=diagnostic(self.tasks,self.seed),final_directions=diagnostic(self.tasks,state),
            covered_counts=[r['covered'] for r in self.check['per_pose']],placement=dict(bases=state.bases.tolist(),offsets=state.offsets.tolist()),selected_exit_paths=list(state.paths),
            witness=str(witness.relative_to(F.I.ROOT)),events=self.events,volume_cm3=body['volume_cm3'],
            reused_qualified_incumbent=folder==self.source,step4_candidates_executed=sum(e['operation']=='measured_trial' for e in self.events),
            provenance=dict(inputs=F.I.hashes([witness,final/'state.json']+self.seed_inputs),code=F.I.hashes(F.sources())),
            artifacts={n:F.I.sha256(final/n) for n in ('shape.obj','geometry_certificate.npz')})
        F.I.save(final/'report.json',report)
        F.I.save(self.out/'report.json',dict(report,artifacts={},provenance=dict(inputs=F.I.hashes([final/'report.json']),code=F.I.hashes(F.sources()))))
        output=self.group/'step4/data'/F.BODY_STAGE;output.mkdir(parents=True,exist_ok=True)
        for name in ('shape.obj','geometry_certificate.npz'):shutil.copy2(final/name,output/name)
        # Renderer needs only case/task geometry, not a second acceptance.
        if grow is None:
            from types import SimpleNamespace
            grow=SimpleNamespace(case=SimpleNamespace(tasks=self.tasks,poses=[t.pose for t in self.tasks],groups=state.groups,heads=[[list(c['triangles_m']) for c in row] for row in state.groups],exit_paths=list(state.paths)))
        F.F.draw(output/'overview.png',grow.case,trimesh.load(output/'shape.obj',force='mesh',process=False),state.bases,state.offsets)
        render_geometry(self.group,self.tasks,state,witness)
        published=dict(report,artifacts={n:F.I.sha256(output/n) for n in ('shape.obj','geometry_certificate.npz','overview.png','construction.png')},provenance=dict(inputs=F.I.hashes([final/'report.json']),code=F.I.hashes(F.sources())))
        F.I.save(output/'report.json',published)
        root=self.group/'step4';backup=self.out/'previous_public_step4';backup.mkdir(exist_ok=True)
        for n in ('shape.obj','geometry_certificate.npz','overview.png','construction_steps.png','report.json'):
            if (root/n).exists() and not (backup/n).exists():shutil.copy2(root/n,backup/n)
        for src,dst in (('shape.obj','shape.obj'),('geometry_certificate.npz','geometry_certificate.npz'),('overview.png','overview.png'),('construction.png','construction_steps.png')):shutil.copy2(output/src,root/dst)
        public=dict(report,artifacts={n:F.I.sha256(root/n) for n in ('shape.obj','geometry_certificate.npz','overview.png','construction_steps.png')},provenance=dict(inputs=F.I.hashes([output/'report.json']),code=F.I.hashes(F.sources())))
        F.I.save(root/'report.json',public)
        F.I.save(root/'data/report.json',dict(public,artifacts={'../'+k:v for k,v in public['artifacts'].items()}))
        (root/'README.md').write_text('# Absolute-direction DSL Step4\n\nQualified connected fixture; head count is not an optimization objective. See the Step5 occupied XYZ box comparison.\n\n[Overview](overview.png) · [Construction](construction_steps.png) · [Model](shape.obj)\n')
        M.evaluate(self.group,self.tasks,state,root/'shape.obj',root/'report.json',self.group/'step5_evaluate')
        F.I.save(self.out/'progress.json',dict(complete=True,events=self.events))
