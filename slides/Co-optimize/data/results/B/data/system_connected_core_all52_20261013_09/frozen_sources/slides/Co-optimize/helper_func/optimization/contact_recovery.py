"""Select cooperating missing reactions, then descend their sweep obstruction.

Reaction selection and smooth obstruction are guidance only. No force capacity
or contact-only acceptance is introduced. Actual geometry/all loads decide.
"""
import time
import numpy as np
from scipy.optimize import linprog, minimize
from co_common import U
from physics_guided_geometry import acquisition, tangent_frames, retract, retraction_jacobian, distance_cost
from physics_guided_contact_sweep import NominalContactSweep


def preserves_loads(before, after):
    return all(np.all(new[old]) for old,new in zip(before['masks'],after['masks']))


def plateau_allowed(before,after,cost_before,cost_after,angle_degrees,steps):
    return bool(preserves_loads(before,after) and cost_after < cost_before-max(1e-9,abs(cost_before)*1e-4)
                and angle_degrees <= 12.+1e-8 and steps < 6)


def cooperating_reactions(current,potential,target,costs):
    """Current real rays are free; potential contacts jointly complete balance."""
    lengths=np.linalg.norm(potential,axis=1)
    rays=potential/lengths[:,None]
    lp=linprog(np.r_[np.zeros(len(current)),np.asarray(costs)+1e-6],
        A_eq=np.vstack([current,rays]).T,b_eq=target,bounds=(0,None),method='highs',
        options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    if not lp.success:
        raise RuntimeError('Missing-contact completion LP: '+lp.message)
    residual=np.vstack([current,rays]).T@lp.x-target
    if np.max(np.abs(residual))>1e-7*max(1.,np.linalg.norm(target)):
        raise RuntimeError('Unresolved missing-contact balance')
    return lp.x[len(current):],float(np.max(np.abs(residual)))


class RecoveryTarget:
    def __init__(self,search,state,directions,avoided=()):
        self.search=search;self.anchor=directions.copy();self.plateau_steps=0
        bank=search.choose_loads(state);loads=[]
        for k,mask in enumerate(state['masks']):
            if mask.all():continue
            ranked=sorted([(search.projection(state,k,i)['loss'],i) for kk,i in bank if kk==k and not mask[i]],reverse=True)
            loads.extend((k,i) for loss,i in ranked[:2])
        if not loads:raise ValueError('No failed original load to recover')
        # Price all potential contacts initially using normal compatibility.
        # The LP selects cooperating reactions, not isolated ray benefits.
        costs=acquisition(directions,search.ray_normals,width=.005)[0]
        # Full trajectory pricing avoids selecting contacts merely because
        # their initial normal happens to be compatible.
        sweep_values=np.column_stack([search.distance_model.distances(d) for d in directions])
        sweep_cost=distance_cost(sweep_values,np.zeros(sweep_values.shape+(2,)),np.zeros((len(directions),2)),width=.001)[0]
        costs+=sweep_cost
        if len(avoided):costs[np.asarray(avoided,int)]+=1.
        weight=np.zeros(len(search.points));records=[]
        for k,i in loads:
            reactions,residual=cooperating_reactions(state['supplies'][k],search.rays[k],U.target(search.states[k][0].targets[i]),costs)
            total=reactions.sum()
            if total>1e-12:weight+=reactions/total
            records.append(dict(pose=search.group['poses'][k],load_index=i,missing_reaction_count=int(np.count_nonzero(reactions>1e-10)),equilibrium_error=residual))
        self.ids=np.flatnonzero(weight>1e-10)
        if not len(self.ids):raise ValueError('No missing-contact reactions selected')
        self.weights=weight[self.ids];self.weights/=self.weights.sum()
        self.normals=search.ray_normals[self.ids]
        self.model=NominalContactSweep(search.clearance,search.points[self.ids],self.normals,search.length,float(search.mesh.extents.max()))
        self.records=records
        self.locked_poses=(self.normals@directions.T>1e-9).tolist()

    def nonlinear_cost(self,directions):
        normal=acquisition(directions,self.normals,width=.005)[0]
        values=np.column_stack([self.model.distances(d) for d in directions])
        extra=distance_cost(values,np.zeros(values.shape+(2,)),np.zeros((len(directions),2)),width=.001)[0]
        return float(self.weights@(normal+extra))

    def step(self,directions):
        began=time.monotonic();frames=tangent_frames(directions)
        values,jac=self.model.linearize(directions,frames)
        def fun(flat):
            z=flat.reshape(len(directions),2);d=retract(directions,frames,z)
            costs,g=acquisition(d,self.normals,width=.005)
            mapping=retraction_jacobian(directions,frames,z)
            derivative=np.einsum('jnk,nki->jni',g,mapping)
            extra,dextra=distance_cost(values,jac,z,width=.001)
            value=float(self.weights@(costs+extra))+.0001*float(flat@flat)
            gradient=np.einsum('j,jni->ni',self.weights,derivative+dextra).ravel()+.0002*flat
            return value,gradient
        zero=np.zeros(len(directions)*2);value,g=fun(zero)
        axis=g/max(np.linalg.norm(g),1e-12);h=1e-5
        finite=(fun(h*axis)[0]-fun(-h*axis)[0])/(2*h)
        error=abs(finite-g@axis)/max(abs(finite),abs(g@axis),1e-12)
        radius=np.deg2rad(4.)
        # On the tangent chart the hemisphere constraint is exactly linear;
        # normalization is positive and cannot change its sign.
        A=np.zeros((len(directions),len(zero)))
        for k in range(len(directions)):A[k,2*k:2*k+2]=self.search.normals[k]@frames[k]
        b=np.sum(directions*self.search.normals,axis=1)
        constraints=[dict(type='ineq',fun=lambda x:A@x+b,jac=lambda x:A),
                     dict(type='ineq',fun=lambda x:radius**2-x@x,jac=lambda x:-2*x)]
        opt=minimize(fun,zero,jac=True,method='SLSQP',constraints=constraints,options={'maxiter':40,'ftol':1e-10})
        candidates=[]
        if np.isfinite(opt.x).all():candidates.append(opt.x)
        # A projected negative gradient remains available if the nonlinear
        # minimizer stalls at a boundary. This is descent, not random sampling.
        projected=minimize(lambda x:(.5*np.sum((x+g)**2),x+g),zero,jac=True,method='SLSQP',
            constraints=constraints,options={'maxiter':40,'ftol':1e-12})
        if np.isfinite(projected.x).all():candidates.append(projected.x)
        best=zero;best_value=value
        if error<=1e-3:
            for x in candidates:
                if np.linalg.norm(x)>radius:x=x*radius*(1-1e-8)/np.linalg.norm(x)
                for fraction in [1.,.5,.25,.125,.0625]:
                    step=fraction*x
                    if np.min(A@step+b)<-1e-12:continue
                    v=fun(step)[0]
                    if v<best_value-1e-10:best=step;best_value=v
        return retract(directions,frames,best.reshape(len(directions),2)),dict(
            seconds=time.monotonic()-began,loss_before=value,loss_after=best_value,
            derivative_relative_error=error,valid_step=bool(np.linalg.norm(best)>1e-10),
            optimizer_success=bool(opt.success),optimizer_message=str(opt.message),
            target_contacts=len(self.ids),critical_loads=self.records,
            target_contact_indices=self.ids.tolist(),initial_normal_blocking_poses=self.locked_poses)
