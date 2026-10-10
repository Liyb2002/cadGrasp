"""Sample adjustment blocks; calculate small increments from contact boundaries."""
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from co_common import save,D,S,U,provenance,code_sources
from placement_sampling import PlacementSearch
from contact_boundary_model import ContactBoundaryModel
from physics_guided_geometry import tangent_frames,retract
from contact_recovery import preserves_loads
from worst_wrench_descent import farthest_load


class LocalPlacementSearch(PlacementSearch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.boundary=ContactBoundaryModel(self)

    def refresh_targets(self,current):
        if current is None:
            targets=[entry for entry in self.targets if not self.saved_masks[entry[0]].all()]
        else:
            targets=[]
            for k,mask in enumerate(current['masks']):
                if mask.all():continue
                from types import SimpleNamespace
                adapter=SimpleNamespace(states=[self.states[k]],projection=lambda state,j,i:self.projection(current,k,i))
                worst,scan=farthest_load(adapter,dict(masks=[mask]))
                targets.append((k,worst['load_index'],U.target(self.states[k][0].targets[worst['load_index']])))
        return targets

    def increment(self,directions,offsets,kind,indices,x):
        d=directions.copy();o=offsets.copy();frames=tangent_frames(directions)
        z=np.zeros((self.n,2))
        if kind=='coherent-translation':
            for k in indices:o[k]+=.001*(np.eye(3)-np.outer(self.normals[k],self.normals[k]))@np.asarray(x)
            return d,o
        if kind=='coherent-direction':
            delta=np.cross(np.tile(np.asarray(x),(self.n,1)),directions)*np.deg2rad(1.)
            for k in indices:z[k]=frames[k].T@delta[k]
            d=retract(d,frames,z)
            for k in indices:
                dot=d[k]@self.normals[k]
                if dot<0:d[k]-=dot*self.normals[k];d[k]/=np.linalg.norm(d[k])
            return d,o
        x=np.asarray(x).reshape(len(indices),2)
        for j,k in enumerate(indices):
            if kind=='translation':o[k]+=self.frames[k]@(.001*x[j])
            else:z[k]=np.deg2rad(1.)*x[j]
        if kind=='direction':
            d=retract(d,frames,z)
            for k in indices:
                dot=d[k]@self.normals[k]
                if dot<0:d[k]-=dot*self.normals[k];d[k]/=np.linalg.norm(d[k])
        return d,o

    def propose(self,directions,offsets,targets,kind,indices,base):
        n=3 if kind.startswith('coherent') else 2*len(indices);zero=np.zeros(n);h=.25;gradient=np.zeros(n);curvature=np.zeros(n);checks=[]
        for j in range(n):
            step=np.zeros(n);step[j]=h;values=[]
            for sign in [-1,1]:
                d,o=self.increment(directions,offsets,kind,indices,sign*step)
                value=self.boundary.evaluate(d,o,targets)
                values.append(value['loss']+1e-3*value['sum_loss'])
            f=base['loss']+1e-3*base['sum_loss']
            gradient[j]=(values[1]-values[0])/(2*h)
            curvature[j]=abs(values[1]+values[0]-2*f)/h**2
            checks.append(dict(component=j,negative=values[0],positive=values[1],gradient=float(gradient[j])))
        if np.linalg.norm(gradient)<1e-10:return [],dict(kind=kind,indices=indices,gradient=gradient.tolist(),stalled=True)
        H=np.maximum(curvature,np.linalg.norm(gradient)*.1)
        opt=minimize(lambda x:(float(gradient@x+.5*np.sum(H*x*x)),gradient+H*x),zero,jac=True,method='SLSQP',
            constraints=[dict(type='ineq',fun=lambda x:1.-x@x,jac=lambda x:-2*x)],options={'maxiter':30,'ftol':1e-12})
        step=opt.x
        if np.linalg.norm(step)>1.:step/=np.linalg.norm(step)
        candidates=[]
        for fraction in [1.,.5,.25]:
            d,o=self.increment(directions,offsets,kind,indices,fraction*step)
            value=self.boundary.evaluate(d,o,targets)
            # Count both additions and losses; this is a force objective, not area maximization.
            score=(value['loss'],value['sum_loss'],float(np.linalg.norm(o)))
            if (score[0],score[1])<(base['loss']-1e-10,base['sum_loss']):
                candidates.append((score,kind,d,o,value,dict(indices=indices,step=(fraction*step).tolist())))
        return candidates,dict(kind=kind,indices=indices,gradient=gradient.tolist(),curvature=curvature.tolist(),checks=checks)

    def blocks(self,current,count):
        failed=[k for k in range(self.n) if not (current['masks'][k] if current is not None else self.saved_masks[k]).all()]
        blocks=[]
        for k in failed:
            if k:blocks.append(('translation',[k]))
            blocks.append(('direction',[k]))
        # A blocked pose can need another pose to move; sample blockers as well.
        for k in self.rng.permutation(np.arange(1,self.n)):
            if ('translation',[int(k)]) not in blocks:blocks.append(('translation',[int(k)]))
        for k in self.rng.permutation(self.n):
            if ('direction',[int(k)]) not in blocks:blocks.append(('direction',[int(k)]))
        blocks=blocks[:count]
        # Joint moves escape the lock-union plateau.
        blocks.append(('coherent-translation',list(range(1,self.n))))
        blocks.append(('coherent-direction',list(range(self.n))))
        return blocks

    def run(self,iterations=12,candidates=6,finalists=2):
        began=time.monotonic();directions=self.initial_directions.copy();offsets=np.zeros_like(directions)
        current=None;errors=[];events=[];stop='iteration_limit';accepted_steps=0
        try:current=self.exact(directions,offsets)
        except (RuntimeError,ValueError) as error:errors.append(dict(stage='initial',error=str(error)))
        initial_counts=[int(m.sum()) for m in self.saved_masks];exact_initial=current['counts'] if current else None
        for iteration in range(iterations):
            if current is not None and all(m.all() for m in current['masks']):stop='force_exit_feasible';break
            targets=self.refresh_targets(current)
            base=self.boundary.evaluate(directions,offsets,targets);began_proxy=time.monotonic()
            proposed=[];stats=[]
            for kind,indices in self.blocks(current,candidates):
                try:
                    options,info=self.propose(directions,offsets,targets,kind,indices,base)
                    proposed.extend(options);stats.append(info)
                except (RuntimeError,ValueError) as error:stats.append(dict(kind=kind,indices=indices,error=str(error)))
            proposed.sort(key=lambda row:row[0]);row=dict(iteration=iteration+1,targets=[(k,i) for k,i,t in targets],
                proxy_loss=base['loss'],proposal_stats=stats,proposals=len(proposed),proxy_seconds=time.monotonic()-began_proxy,trials=[])
            accepted=False
            for score,kind,d,o,value,step in proposed[:finalists]:
                record=dict(kind=kind,proxy_score=score,step=step,directions=d.tolist(),offsets_m=o.tolist(),
                    **self.boundary.changes(base['geometry'],value['geometry']))
                try:
                    trial=self.exact(d,o);record['counts']=trial['counts']
                    protected=preserves_loads(current if current is not None else dict(masks=self.saved_masks),trial)
                    accepted=protected and (current is None or self.real_score(trial)>self.real_score(current))
                    if protected and not accepted:
                        before,_=farthest_load(self,current);after,_=farthest_load(self,trial)
                        accepted=(0. if after is None else after['loss'])<before['loss']-1e-8
                    if accepted:current=trial;directions=d;offsets=o;accepted_steps+=1
                    record.update(accepted=accepted,protected=protected)
                except (RuntimeError,ValueError) as error:record.update(error=str(error),accepted=False)
                row['trials'].append(record)
                if accepted:break
            row.update(accepted=accepted,counts=current['counts'] if current else None);events.append(row)
            save(self.out/'sampling_trace.json',events)
            print('LOCAL PLACEMENT ROUND',iteration+1,'accepted',accepted,row['counts'],'seconds',row['proxy_seconds'],flush=True)
            if not accepted:stop='local_boundary_search_stalled';break
        passed=current is not None and all(m.all() for m in current['masks'])
        if current is not None:D.export_exact_obj(S.unpack(current['remaining']),self.out/'remaining_support.obj')
        np.savez_compressed(self.out/'layout.npz',directions=directions,offsets_m=offsets,
            T_fixture_to_world=np.asarray(current['transforms']) if current else np.empty((0,4,4)))
        report=dict(complete=True,algorithm='sample pose/tool blocks; compute local contact-boundary gradient and small regularized steps',
            pose_set=self.group['id'],poses=self.group['poses'],initial_counts=initial_counts,exact_initial_counts=exact_initial,
            final_counts=current['counts'] if current else None,force_exit_passed=bool(passed),force_passed=bool(passed),
            geometry_constructed=current is not None,clearance_certified=current is not None,full_fixture_accepted=False,
            baseline_used=False,baseline_passed=False,stop_reason=stop,accepted_sampling_steps=accepted_steps,iterations=events,errors=errors,
            offsets_m=offsets.tolist(),directions=directions.tolist(),maximum_translation_m=float(np.linalg.norm(offsets,axis=1).max()),
            maximum_translation_step_m=.001,maximum_direction_step_degrees=1.,fixed_reference_pose=self.group['poses'][0],
            seconds=time.monotonic()-began,exact_evaluations=self.exact_calls,original_loads_reused=True,
            load_count_per_pose=[len(t.targets) for t,T in self.states],component_count=len(current['remaining'].decompose()) if current else None,
            diagnostics=current['diagnostics'] if current else None,contact_patches=current['contact_patches'] if current else None,
            derivative='central differences of continuously clipped contact polygons, not fixture solids; signed-distance interpolation for initial cross-object overlap is guidance only',
            acceptance='all original loads/no-uplift, continuous nominal exits, 1% clearance and work exclusions; protect every previously passing load',
            deferred='connectivity, installed whole-fixture floor legality/support, strength; illegal group remains a diagnostic',
            provenance=provenance(self.inputs,[Path(__file__),Path(__file__).with_name('placement_sampling.py'),Path(__file__).with_name('contact_boundary_model.py')]+code_sources()))
        save(self.out/'report.json',report);print('LOCAL PLACEMENT FINAL',self.group['id'],report['final_counts'],stop,flush=True)
        return report

    def real_score(self,result):
        fractions=np.array(result['counts'])/np.array([len(t.targets) for t,T in self.states])
        return float(fractions.min()),float(fractions.sum())
