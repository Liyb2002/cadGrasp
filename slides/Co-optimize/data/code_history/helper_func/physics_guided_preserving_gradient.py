"""Physical-deficit recovery with existing real-force witness retention."""
from physics_guided_hybrid import *
from physics_guided_contact_planes import allocated_contacts

def preserving_refine(self,result,iterations,label):
    d=result['directions'].copy();radius=np.deg2rad(12.);visited=[d.copy()]
    for iteration in range(iterations):
        if all(m.all() for m in result['masks']):return result
        loads=self.choose_loads(result)
        normal_costs=acquisition(d,self.ray_normals)[0]
        weights,info=self.dual_weights(result,normal_costs)
        if weights.sum()<1e-15:break
        selected=np.flatnonzero(weights>0)
        if len(selected)>512:selected=selected[np.argsort(weights[selected])[-512:]]
        frozen=weights[selected];frozen/=frozen.sum()
        points=self.points[selected];normals=self.ray_normals[selected]
        _,old_points,old_normals,old_weights=allocated_contacts(self,result)
        if len(old_weights) and old_weights.sum()>0:
            points=np.vstack([points,old_points]);normals=np.vstack([normals,old_normals])
            frozen=np.r_[.75*frozen,.25*old_weights/old_weights.sum()]
        model=NominalContactSweep(self.clearance,points,normals,self.length,float(self.mesh.extents.max()))
        frames=tangent_frames(d);zero=np.zeros((len(d),2));values,jac=model.linearize(d,frames)
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
        record=dict(stage=label,iteration=iteration,physics_signal=info,geometry_probe_count=len(points),retained_force_probe_count=len(old_points),
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
                if any(np.linalg.norm(candidate-previous)<1e-4 for previous in visited):
                    record['trials'].append(dict(accepted=False,acceptance_reason='repeated_direction_state'))
                    continue
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
                        result=trial;d=candidate;visited.append(d.copy());accepted=True
                        radius=min(np.deg2rad(20.),radius*1.2) if decision['force_improved'] else radius*.75
                        break
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
