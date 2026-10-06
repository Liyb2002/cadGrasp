"""Shared-trend exploration plus gradients from actual missing-force duals."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'helper_func'))
import _bootstrap
from physics_guided import *
from physics_guided_dual import certificate_candidates,deficit_contact_values
from physics_guided_contact_sweep import NominalContactSweep
from hybrid_directions import fibonacci,project_common,floor_common,common_neighborhood,independent_neighborhood


class HybridSearch(PhysicsSearch):
    def __init__(self,*args,max_proposals=1200,**kwargs):
        super().__init__(*args,**kwargs)
        if self.connectivity:raise ValueError('Connectivity is deferred in this experiment')
        patches=[contact_boundary(self.mesh,S.unpack(part),self.allowed)
                 for part in self.seed.decompose() if material_volume(part)>=1e-12]
        self.tri=np.concatenate([p[0] for p in patches]);self.src=np.concatenate([p[1] for p in patches])
        points=self.tri.reshape(-1,3);sources=np.repeat(self.src,3)
        normals=self.mesh.face_normals[sources]
        _,first=np.unique(np.round(np.c_[points,normals],11),axis=0,return_index=True)
        self.points=points[first];self.point_sources=sources[first];self.ray_normals=normals[first]
        self.rays=[]
        for task,T in self.states:
            world=transform_points(self.points,T);force=-task.domain.mesh.face_normals[self.point_sources]
            self.rays.append(U.heads(np.c_[force,np.cross(world-task.domain.com,force)],task.scale))
        self.distance_model=NominalContactSweep(self.clearance,self.points,self.ray_normals,self.length,float(self.mesh.extents.max()))
        np.savez_compressed(self.out/'potential_contacts.npz',points=self.points,sources=self.point_sources)
        self.max_proposals=max_proposals;self.proposals=0;self.seen=set();self.global_trace=[]
        self.best=None;self.best_common=None;self.selection_trace=[];self.beam=[]
        self.proxy_indices=[{0} for _ in self.states]
        self.additional_code=[Path(__file__),HERE/'helper_func/optimization/physics_guided_dual.py',
            HERE/'helper_func/optimization/physics_guided_contact_sweep.py',HERE/'helper_func/optimization/hybrid_directions.py']
        self.additional_inputs=[self.out/'potential_contacts.npz']
        self.additional_artifacts=['potential_contacts.npz','global_proposals.json','critical_load_selection.json']

    def choose_loads(self,result):
        for k,mask in enumerate(result['masks']):
            indices,info=certificate_candidates(U.target(self.states[k][0].targets),~mask,
                lambda index:self.projection(result,k,int(index)),limit=6)
            self.load_bank.update((k,int(i)) for i in indices)
            self.proxy_indices[k].update(indices)
            self.selection_trace.append(dict(serial=result['serial'],pose=self.group['poses'][k],indices=indices,**info))
        return sorted(self.load_bank)

    def score(self,result):
        counts=np.asarray(result['counts'],float)/np.asarray([len(t.targets) for t,T in self.states])
        return (int(sum(m.all() for m in result['masks'])),float(counts.min()),float(counts.sum()))

    def remember(self,result,common=None):
        # Legacy balanced coverage chooses exploration starts; local acceptance
        # still compares real continuous deficits on a common load bank.
        if self.best is None or self.score(result)>self.score(self.best):
            self.best=result
            if common is not None:self.best_common=common.copy()
        self.choose_loads(result)
        nearby=[r for r in self.beam if np.linalg.norm(r['directions']-result['directions'])<.08]
        if not nearby or self.score(result)>max(self.score(r) for r in nearby):
            nearby_serials={r['serial'] for r in nearby}
            self.beam=[r for r in self.beam if r['serial'] not in nearby_serials]+[result]
        self.beam=sorted(self.beam,key=self.score,reverse=True)[:3]

    def dual_weights(self,result,costs=None):
        loads=self.choose_loads(result)
        projections=[(k,self.projection(result,k,index)) for k,index in loads if not result['masks'][k][index]]
        return deficit_contact_values(self.rays,projections,costs=costs)

    def proxy(self,d):
        keep=np.max(self.ray_normals@d.T,axis=1)<=1e-9
        for k,(task,T) in enumerate(self.states):
            full=np.vstack([self.floors[k],self.rays[k][keep]])
            for index in sorted(self.proxy_indices[k]):
                target=U.target(task.targets[index])
                if C.W.solve(full,target) is not None:continue
                # Only reject from an actual separating certificate. A failed
                # numerical primal solve alone must not discard a candidate.
                projection=cone_projection(full,target)
                norm=np.linalg.norm(projection['dual'])
                if norm>1e-12:
                    normal=-projection['dual']/norm
                    if np.max(full@normal)<=1e-12 and target@normal>1e-8:return False
        return True

    def attempt(self,d,label,common=None,force=False):
        if self.proposals>=self.max_proposals:return None
        d=np.asarray(d,float);d=d/np.linalg.norm(d,axis=1)[:,None]
        if np.min(np.sum(d*self.normals,axis=1))<-1e-12:return None
        key=tuple(np.round(d.ravel(),8))
        if key in self.seen:return None
        self.seen.add(key);self.proposals+=1
        if not force and not self.proxy(d):
            self.global_trace.append(dict(proposal=self.proposals,label=label,status='optimistic cone separator rejected'))
            return None
        try:
            result=self.exact(d);self.remember(result,common)
            self.global_trace.append(dict(proposal=self.proposals,label=label,status='constructed',counts=result['counts'],common=None if common is None else common.tolist()))
            print('HYBRID',self.proposals,label,result['counts'],flush=True)
            if all(m.all() for m in result['masks']):return result
        except (RuntimeError,ValueError) as error:
            self.global_trace.append(dict(proposal=self.proposals,label=label,status='numerically unresolved',error=str(error)))
        return None

    def ordered_common(self,axes,lift=.01):
        if self.best is None:weights=None
        else:weights,_=self.dual_weights(self.best)
        proposals=[]
        for a in axes:
            d=project_common(a,self.normals,lift)
            keep=np.max(self.ray_normals@d.T,axis=1)<=1e-9
            benefit=float(weights@keep) if weights is not None else 0.
            proposals.append((benefit,d,a))
        # Stable order preserves the legacy covering bank when benefits tie.
        return sorted(proposals,key=lambda row:-row[0])

    def refine(self,result,iterations,label):
        d=result['directions'].copy();radius=np.deg2rad(12.)
        for iteration in range(iterations):
            if all(m.all() for m in result['masks']):return result
            loads=self.choose_loads(result)
            normal_costs=acquisition(d,self.ray_normals)[0]
            weights,info=self.dual_weights(result,normal_costs)
            if weights.sum()<1e-15:break
            selected=np.flatnonzero(weights>0)
            if len(selected)>512:selected=selected[np.argsort(weights[selected])[-512:]]
            frozen=weights[selected];frozen/=frozen.sum()
            model=NominalContactSweep(self.clearance,self.points[selected],self.ray_normals[selected],self.length,float(self.mesh.extents.max()))
            frames=tangent_frames(d);zero=np.zeros((len(d),2));values,jac=model.linearize(d,frames)
            normals=self.ray_normals[selected]
            def costs(candidate):
                fields=np.column_stack([model.distances(x) for x in candidate])
                zeros=np.zeros(fields.shape+(2,))
                return acquisition(candidate,normals)[0]+distance_cost(fields,zeros,zero)[0]
            def fun(x):
                z=x.reshape(zero.shape);candidate=retract(d,frames,z)
                c,g=acquisition(candidate,normals)
                mapping=retraction_jacobian(d,frames,z)
                gradient=np.einsum('jnk,nki->jni',g,mapping)
                extra,derivative=distance_cost(values,jac,z)
                return float(frozen@(c+extra)),np.einsum('j,jni->ni',frozen,gradient+derivative).ravel()
            geometry_before=float(frozen@costs(d));value,gradient=fun(zero)
            if np.linalg.norm(gradient)<1e-12:break
            descent=-gradient/np.linalg.norm(gradient);h=1e-5
            finite=(fun(h*descent)[0]-fun(-h*descent)[0])/(2*h)
            error=abs(finite-gradient@descent)/max(abs(finite),abs(gradient@descent),1e-12)
            record=dict(stage=label,iteration=iteration,physics_signal=info,geometry_probe_count=len(selected),
                derivative_relative_error=float(error),trials=[])
            if error>1e-3:
                record['accepted']=False;record['status']='sensitivity unresolved';self.trace.append(record);break
            accepted=False
            for trust in range(3):
                opt=minimize(fun,zero.ravel(),jac=True,method='SLSQP',constraints=[
                    dict(type='ineq',fun=lambda x:np.sum(retract(d,frames,x.reshape(zero.shape))*self.normals,axis=1)),
                    dict(type='ineq',fun=lambda x:radius**2-float(x@x))],options={'maxiter':30,'ftol':1e-9})
                if not np.isfinite(opt.x).all() or np.linalg.norm(opt.x)>radius*(1+1e-5):radius*=.5;continue
                for fraction in [1.,.5,.25]:
                    candidate=retract(d,frames,(fraction*opt.x).reshape(zero.shape))
                    if np.min(np.sum(candidate*self.normals,axis=1))<-1e-12:continue
                    try:
                        trial=self.exact(candidate)
                        if all(m.all() for m in trial['masks']):
                            self.remember(trial);result=trial;d=candidate;accepted=True
                            record['trials'].append(dict(accepted=True,acceptance_reason='full_force_feasible',counts=trial['counts'],fraction=fraction))
                            break
                        expanded=self.choose_loads(trial)
                        before,_=self.actual_loss(result,expanded);after,_=self.actual_loss(trial,expanded)
                        best_loss,_=self.actual_loss(self.best,expanded)
                        geometry_after=float(frozen@costs(candidate))
                        decision=progress_decision(before,after,best_loss,geometry_before,geometry_after)
                        if all(m.all() for m in trial['masks']):
                            decision.update(accepted=True,acceptance_reason='full_force_feasible')
                        record['trials'].append(dict(old_loss=before,new_loss=after,geometry_before=geometry_before,
                            geometry_after=geometry_after,counts=trial['counts'],fraction=fraction,**decision))
                        self.remember(trial)
                        if decision['accepted']:
                            result=trial;d=candidate;accepted=True;radius=min(np.deg2rad(20.),radius*1.4);break
                    except (RuntimeError,ValueError) as error:
                        record['trials'].append(dict(error=str(error),accepted=False))
                if accepted:break
                radius*=.5
            self.distance_model.evaluations+=model.evaluations
            record['accepted']=accepted;self.trace.append(record)
            save(self.out/'optimization_trace.json',self.trace)
            print('DUAL GRADIENT',label,iteration,'accepted',accepted,result['counts'],flush=True)
            if not accepted:break
        return result

    def optimize(self,iterations=12):
        began=time.monotonic();initial_directions=self.warm();initial_directions/=np.linalg.norm(initial_directions,axis=1)[:,None]
        np.savez_compressed(self.out/'initial_directions.npz',directions=initial_directions)
        initial_counts=None;initial_error=None;winner=None
        try:
            native=self.exact(initial_directions);initial_counts=native['counts'];self.remember(native)
            if all(m.all() for m in native['masks']):winner=native
            elif min(native['counts'])/32768>.75:
                refined=self.refine(native,min(iterations,6),'native near-feasible')
                if all(m.all() for m in refined['masks']):winner=refined
        except (RuntimeError,ValueError) as error:initial_error=str(error)
        if winner is None:
            common=floor_common(self.normals)
            if common is not None:winner=self.attempt(project_common(common,self.normals),'common floor cone',common)
        global_limit=max(1,int(self.max_proposals*.65))
        for count in [160,512,2048]:
            if winner is not None or self.proposals>=global_limit:break
            axes=list(fibonacci(count))
            if self.best is not None:
                weights,_=self.dual_weights(self.best)
                important=np.argsort(weights)[-16:]
                axes=[-self.ray_normals[j] for j in important if weights[j]>0]+axes
            ordered=self.ordered_common(axes)
            for rank,(benefit,d,a) in enumerate(ordered):
                winner=self.attempt(d,f'common covering {count}; dual rank {rank}',a,force=rank<3)
                if winner is not None or self.proposals>=global_limit:break
            if winner is not None:break
            if self.best is not None:
                refined=self.refine(self.best,iterations,f'after common {count}')
                if all(m.all() for m in refined['masks']):winner=refined;break
            if self.best_common is not None:
                center=self.best_common.copy()
                local_begin=self.proposals
                for a,lift in common_neighborhood(center):
                    winner=self.attempt(project_common(a,self.normals,lift),'legacy common local recovery',a)
                    if winner is not None or self.proposals>=self.max_proposals or self.proposals-local_begin>=80:break
        if winner is None and self.best is not None and self.proposals<self.max_proposals:
            order=list(np.argsort(self.best['counts']))
            for candidate,k in independent_neighborhood(self.best['directions'],self.normals,order):
                winner=self.attempt(candidate,f'legacy independent recovery {k}')
                if winner is not None or self.proposals>=self.max_proposals:break
        if winner is None and self.best is not None:
            for seed in list(self.beam):
                refined=self.refine(seed,iterations,'final multi-start dual recovery')
                if all(m.all() for m in refined['masks']):winner=refined;break
        result=winner or self.best
        if result is None:raise RuntimeError('No numerically resolved hybrid construction')
        d=result['directions'];np.savez_compressed(self.out/'continuation_directions.npz',directions=d)
        save(self.out/'global_proposals.json',self.global_trace);save(self.out/'critical_load_selection.json',self.selection_trace)
        self.report_extra=dict(algorithm='hybrid shared-trend exploration and actual-cone-dual gradients',
            initial_counts=initial_counts,initial_geometry_error=initial_error,
            proposal_count=self.proposals,proposal_budget=self.max_proposals,
            physics_gradient='missing-ray benefit from the real constructed cone residual; no all-potential-contact pricing LP',
            geometry_gradient='analytic initial compatibility + numerical nominal contact-core trajectory field; positive penetration p=8',
            potential_contact_policy='positive-volume original seed contact boundary; unique existing vertices; no invented attainable patches',
            critical_load_policy='scan all original load vectors with real projection-dual certificates and project finalists; no global maximum certificate',
            connectivity_required=False,acceptance_scope='all original loads and full 1% clearance exits; connectivity recorded separately',
            passed=bool(all(m.all() for m in result['masks']) and max(material_volume(result['construction']['remaining']^self.clearance.sweep(self.length*x,padded=False)) for x in d)<1e-10 and result['partition']<1e-10 and max(result['endpoint_overlap'])<1e-10 and result['construction']['diagnostics']['geometry_resolved']))
        return self.finish(result,initial_counts,began,'aggregate_force_feasible' if winner is not None else 'bounded_hybrid_unresolved',continuation_counts=result['counts'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set',required=True);parser.add_argument('--directions',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--iterations',type=int,default=12)
    parser.add_argument('--max-proposals',type=int,default=1200)
    args=parser.parse_args()
    groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    groups += [dict(g,id='illegal/'+g['id']) for g in json.loads((ROOT/'objects/B/illegal_pose_sets.json').read_text())['sets']]
    group=next(g for g in groups if g['id']==args.set)
    search=HybridSearch('B',group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)))
        raise

if __name__=='__main__':main()
