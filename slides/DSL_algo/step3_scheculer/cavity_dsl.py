"""Joint common-cavity proposals: XY, yaw and real contact reassignment.

Old qualified bodies are immutable seeds. Only constructed Step5 gains commit.
"""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import sys,json,itertools
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
from step3_scheculer import absolute_dsl as A,absolute_floor_recovery as R,absolute_local_descent as L
from step3_scheculer.review_operation_dsl import load
from step3_scheculer.run_dsl import saved_task
from step0_pose_selection.floor_points import pressure_centers
F=A.F


def yaw_about(state,k,angle,center):
    b=state.bases.copy();o=state.offsets.copy();installed=center@b[k]+o[k]
    b[k]=b[k]@Rotation.from_euler('z',angle,degrees=True).as_matrix().T
    o[k]=installed-center@b[k]
    return replace(state,bases=b,offsets=o)


def common_floor(state):return np.allclose(state.bases[:,2,:],[0,0,1.],atol=1e-8)


class Optimizer(A.Optimizer):
    def __init__(self,group,tasks):
        self.group,self.tasks,self.config=group,tasks,dict(device='cuda')
        self.out=group/'step3_scheculer'/F.STAGE;self.out.mkdir(parents=True,exist_ok=True)
        self.source=group/'step3_scheculer'/('dsl_absolute_floor' if (group/'step3_scheculer/dsl_absolute_floor/report.json').exists() else 'dsl_absolute_refined')/'final'
        self.seed,_=load(self.source,tasks);self.reference=self.source/'shape.obj';self.reference_report=F.I.check_report(self.source/'report.json')
        self.seed_inputs=[self.source/'report.json',self.source/'state.json']+[self.source/f'contacts_{t.pose}.npz' for t in tasks]
        self.checker=F.Checker(tasks);self.check=self.checker.check(self.seed);assert self.check['passed']
        self.events=[];self.trial=0;self.incumbent=self.initial_state=self.seed
        self.initial_volume=A.M.volume(tasks,self.seed,trimesh.load(self.reference,force='mesh',process=False));self.best_volume=self.initial_volume
        self.witness=(self.source,self.reference_report,None);self.pool={};self.serial=0
        self.demands=[np.c_[pressure_centers(t.targets/t.scale,t.domain.com)[0],np.zeros(len(pressure_centers(t.targets/t.scale,t.domain.com)[0]))] for t in tasks]
        F.save_state(self.out/'initial',tasks,self.seed,self.check)
        self.event('initialize_qualified',volume_cm3=self.initial_volume,directions=A.diagnostic(tasks,self.seed))

    def proxy(self,state):
        cloud=np.vstack([np.vstack([c['triangles_m'].reshape(-1,3) for c in row]+[d])@b+o for row,d,b,o in zip(state.groups,self.demands,state.bases,state.offsets)])
        native=np.vstack([t.domain.mesh.vertices for t in self.tasks]+[(cloud-o)@b.T for b,o in zip(state.bases,state.offsets)])
        volume=np.prod(np.ptp(native,axis=0))*1e6
        outlines=[MultiPoint((t.domain.mesh.vertices@b+o)[:,:2]).convex_hull for t,b,o in zip(self.tasks,state.bases,state.offsets)]
        area=unary_union(outlines).area
        # Coarse overlap guidance; neither this nor the contact/demand box is
        # accepted as a constructed-volume metric.
        return float(volume+.2*area*np.ptp(native,axis=0)[2]*1e6),float(area)

    def geometry(self,state):
        _,owners,_=F.roots(self.tasks,state)
        sweeps=[F.F.transform(F.S.solid(F.E.sweep_mesh(t.domain.mesh,p)),b,o) for t,p,b,o in zip(self.tasks,state.paths,state.bases,state.offsets)]
        return owners,sweeps

    def blocked(self,state):
        owners,sweeps=self.geometry(state);bad={k:set() for k in range(len(self.tasks))}
        for ident,(k,cells,patch) in owners.items():
            body=F.F.union([F.S.solid(F.G.hull_mesh(v)) for v in cells])
            if any(abs(float((body^s).volume()))*F.S.SCALE**3>8e-14 for j,s in enumerate(sweeps) if j!=k):bad[k].add(ident)
        return bad,sweeps

    def candidates(self,k,state,sweeps):
        t=self.tasks[k]
        if k not in self.pool:
            compiler=F.D.Compiler(t,[t],device='cuda')
            if np.allclose(state.paths[k]['initial_object_exit_world'],[0,0,1.]):F.restrict_compiler(compiler,np.array([0,0,-1.]))
            seeds=compiler.seeds(72);heads=[]
            for p in seeds:
                for scale in (.7,1.,1.41421356,1.7320508):
                    c=compiler.compile_patch(replace(p,radius=p.radius*scale))
                    if c is not None:heads.append(c)
            self.pool[k]=(compiler,heads)
        compiler,pool=self.pool[k];analyzer=self.checker.analyzers[k];good=[]
        # Real extrusion and actual continuous foreign sweeps, no artificial
        # normal directions or relaxed force/floor/work constraints.
        for c in pool:
            if not analyzer.test(analyzer.heads([c]),state.paths[k])['clear']:continue
            groups=list(state.groups);groups[k]=(dict(c,candidate_id='CAVITY_CANDIDATE'),)
            try:_,owner,_=F.roots(self.tasks,replace(state,groups=tuple(groups)))
            except ValueError:continue
            body=F.F.union([F.S.solid(F.G.hull_mesh(v)) for v in owner['CAVITY_CANDIDATE'][1]])
            if any(abs(float((body^s).volume()))*F.S.SCALE**3>8e-14 for j,s in enumerate(sweeps) if j!=k):continue
            good.append(c)
        return compiler,good

    def repair(self,state):
        bad,sweeps=self.blocked(state);groups=list(state.groups);edits=[]
        for k,ids in bad.items():
            if not ids:continue
            t=self.tasks[k];row=[c for c in groups[k] if c['candidate_id'] not in ids]
            compiler,pool=self.candidates(k,state,sweeps)
            sample=np.linspace(0,len(t.targets)-1,48,dtype=int)
            for step in range(12):
                mask,_=F.J.classify(t.supply(row),t.targets)
                if mask.all():break
                if not pool:return None
                failed=np.flatnonzero(~mask);sample=np.unique(np.r_[sample,failed[np.linspace(0,len(failed)-1,min(48,len(failed)),dtype=int)]])
                loss=compiler.backend.solve([F.D.reduced_rays(t.supply(row+[c])) for c in pool],t.targets[sample])
                j=int(np.argmin(loss.mean(axis=1)+loss.max(axis=1)));c=pool.pop(j);self.serial+=1
                row.append(dict(c,candidate_id=f'{t.pose}_CAVITY_{self.serial}'))
            else:return None
            if not mask.all():return None
            groups[k]=tuple(row);edits.append(dict(pose=t.pose,removed=len(ids),added=len(row)-(len(state.groups[k])-len(ids))))
        proposal=replace(state,groups=tuple(groups))
        if any(self.blocked(proposal)[0].values()):return None
        if edits:self.event('contact_reassignment',edits=edits,gpu_proposal_only=True)
        return proposal

    def search(self,branch,name):
        branch=self.unshared(branch)
        if not common_floor(branch):return
        for round_number in range(2):
            score,_=self.proxy(branch);proposals=[]
            objects=np.array([t.domain.mesh.vertices.mean(axis=0)@b+o for t,b,o in zip(self.tasks,branch.bases,branch.offsets)])
            contact_centers=[np.vstack([c['triangles_m'].reshape(-1,3) for c in row]).mean(axis=0) for row in branch.groups]
            forces,_=A.directions(self.tasks,branch);xy=forces[:,:2].sum(axis=0)
            if np.linalg.norm(xy)<1e-9:xy=np.array([1.,0.])
            # Joint proposals may cross a fixed-head collision boundary. They
            # are repaired before any qualified branch can be retained.
            for k in range(len(self.tasks)):
                angles=[-90,-45,-20,20,45,90,180]
                if np.linalg.norm(forces[k,:2])>1e-9:
                    theta=np.degrees(np.arctan2(xy[1],xy[0])-np.arctan2(forces[k,1],forces[k,0]));angles += [theta*.5,theta]
                for angle,fraction in itertools.product(angles,(0.,.2,.45)):
                    p=yaw_about(branch,k,angle,contact_centers[k]);o=p.offsets.copy()
                    o[k,:2]+=fraction*(objects.mean(axis=0)-objects[k])[:2]
                    p=replace(p,offsets=o);value,area=self.proxy(p)
                    if value<score*.99:proposals.append((value,k,float(angle),fraction,p))
                for fraction in (.2,.45,.7):
                    o=branch.offsets.copy();o[k,:2]+=fraction*(objects.mean(axis=0)-objects[k])[:2]
                    p=replace(branch,offsets=o);value,_=self.proxy(p)
                    if value<score*.99:proposals.append((value,k,0.,fraction,p))
            proposals.sort(key=lambda p:p[:4]);qualified=[];seen=set()
            for value,k,angle,fraction,p in proposals:
                if k in seen:continue
                seen.add(k)
                try:
                    fixed=self.repair(p)
                    if fixed is None:self.event('reject_contact_reassignment',pose=self.tasks[k].pose,yaw_deg=angle,fraction=fraction);continue
                    result=self.consider(fixed,f'{name}_r{round_number}_yaw{k}_{angle:.2f}_move{fraction}')
                    if result:qualified.append(result)
                except Exception as error:self.event('reject_cavity_candidate',reason=str(error),pose=self.tasks[k].pose)
                if len(seen)>=3:break
            if not qualified:break
            chosen=min(qualified,key=lambda p:p[2]);branch=chosen[0]
            compact,history=L.compact(self.tasks,branch,rounds=3)
            if history:
                self.event('cavity_local_compaction',steps=len(history),history=history)
                result=self.consider(compact,f'{name}_r{round_number}_compact')
                if result and result[2]<chosen[2]:branch=result[0]

    def optimize(self):
        branches=[(self.seed,'incumbent')]
        if not common_floor(self.seed):
            ranked=[]
            for metric in (self.group/'step3_scheculer/dsl_absolute/trials').glob('*common_up*/step5/report.json'):
                r=F.I.check_report(metric);ranked.append((r['metrics']['object_and_support_poses']['box_volume_cm3'],metric.parent.parent))
            if ranked:
                _,folder=min(ranked);F.I.check_report(folder/'report.json');s,_=load(folder,self.tasks);branches.append((s,'alternate_common_cavity'))
        for state,name in branches:
            try:self.search(state,name)
            except Exception as e:self.event('reject_cavity_branch',source=name,reason=str(e))
        self.consider(self.incumbent,'final_step4_regrowth');self.publish()


def activate():
    R.activate();original=F.sources
    F.STAGE='dsl_cavity';F.BODY_STAGE='dsl_cavity_support';F.SCHEMA='joint_cavity_dsl_v10'
    F.sources=lambda:list(dict.fromkeys(original()+[Path(__file__)]))


def main():
    activate();group=F.I.OUTPUTS/'B'/sys.argv[1]
    source=group/'step3_scheculer'/('dsl_absolute_floor' if (group/'step3_scheculer/dsl_absolute_floor/report.json').exists() else 'dsl_absolute_refined')
    r=F.I.check_report(source/'report.json');optimizer=Optimizer(group,[saved_task(group,p) for p in r['poses']]);optimizer.optimize()
    print('CAVITY DONE',group.name,optimizer.initial_volume,optimizer.best_volume,flush=True)
if __name__=='__main__':main()
