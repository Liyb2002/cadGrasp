"""Local force-envelope descent with reactions re-solved as directions change.

A differentiable relaxation guides binary contact switches; it is not physical
acceptance and does not impose artificial contact force capacities.
"""
import time
import numpy as np
from scipy.optimize import minimize
from co_common import U
from physics_guided_geometry import acquisition,tangent_frames,retract,retraction_jacobian,distance_cost
from physics_guided_contact_sweep import NominalContactSweep
from physics_guided_objective import acquisition_equilibrium,aggregate


def prepare_force_objective(search,state,origin,probe_limit=128,physical_state=None):
    # Select physical critical loads, and preserve useful force generators from
    # all load signals rather than a single normalized geometry-weight vector.
    physical=state if physical_state is None else physical_state
    bank=search.choose_loads(physical)
    loads=[]
    for k in range(len(search.states)):
        entries=sorted([(search.projection(physical,k,i)['loss'],i) for kk,i in bank if kk==k],reverse=True)
        if entries:loads.extend((k,i) for loss,i in entries[:(3 if physical_state is not None else 1)])
        elif physical_state is not None:loads.append((k,0))
    weights,_=search.dual_weights(physical,acquisition(origin,search.ray_normals)[0])
    selected=np.flatnonzero(weights>0)
    selected=selected[np.argsort(weights[selected])[-probe_limit:]]
    # Also include contacts actually carrying each selected load's projection.
    for k,i in loads if physical_state is None else []:
        p=search.projection(state,k,i)
        available=np.flatnonzero(state['active'])
        used=np.flatnonzero(p['coefficients'][len(search.floors[k]):]>1e-10)
        selected=np.union1d(selected,available[used])
    if not len(selected):raise ValueError('No force-envelope contact generators')
    frames=tangent_frames(origin)
    model=NominalContactSweep(search.clearance,search.points[selected],search.ray_normals[selected],search.length,float(search.mesh.extents.max()))
    values,jac=model.linearize(origin,frames)
    normals=search.ray_normals[selected]
    targets=[U.target(search.states[k][0].targets[i]) for k,i in loads]
    target_norms=[max(np.linalg.norm(t,ord=1),1e-12) for t in targets]
    calls=0
    def objective(flat):
        nonlocal calls
        calls+=1
        z=flat.reshape(len(origin),2);directions=retract(origin,frames,z)
        costs,g=acquisition(directions,normals)
        mapping=retraction_jacobian(origin,frames,z)
        gradient=np.einsum('jnk,nki->jni',g,mapping)
        extra,derivative=distance_cost(values,jac,z);costs+=extra;gradient+=derivative
        loss=[];derivatives=[]
        for (k,i),target,norm in zip(loads,targets,target_norms):
            physical=acquisition_equilibrium(search.floors[k],search.rays[k][selected],target,costs,penalty=1.)
            loss.append(physical['value']/norm)
            derivatives.append(np.einsum('j,jni->ni',physical['cost_gradient'],gradient)/norm)
        value,derivative=aggregate(loss,derivatives,temperature=.01)
        return value,derivative.ravel()
    return objective,frames,dict(contact_generators=len(selected),critical_loads=loads,lp_calls=lambda:calls*len(loads))


def force_descent(search,state,start,maxiter=12,radius_degrees=12.,physical_state=None):
    began=time.perf_counter();fun,frames,info=prepare_force_objective(search,state,start,physical_state=physical_state)
    preparation=time.perf_counter()-began
    zero=np.zeros(len(start)*2);value,gradient=fun(zero)
    step=1e-5;axis=gradient/max(np.linalg.norm(gradient),1e-12)
    finite=(fun(step*axis)[0]-fun(-step*axis)[0])/(2*step)
    derivative_error=abs(finite-gradient@axis)/max(abs(finite),abs(gradient@axis),1e-12)
    radius=np.deg2rad(radius_degrees)
    opt=minimize(fun,zero,jac=True,method='SLSQP',constraints=[
        dict(type='ineq',fun=lambda x:np.sum(retract(start,frames,x.reshape(len(start),2))*search.normals,axis=1)),
        dict(type='ineq',fun=lambda x:radius**2-float(x@x))],options={'maxiter':maxiter,'ftol':1e-8})
    finite_step=np.isfinite(opt.x).all()
    coordinates=opt.x.copy() if finite_step else zero.copy()
    if np.linalg.norm(coordinates)>radius:
        coordinates*=radius*(1-1e-8)/np.linalg.norm(coordinates)
    endpoint=start.copy();after=value;valid=False
    if finite_step and derivative_error<=1e-3:
        for fraction in [1.,.5,.25,.125]:
            trial=fraction*coordinates
            d=retract(start,frames,trial.reshape(len(start),2))
            if np.min(np.sum(d*search.normals,axis=1))<-1e-12:continue
            v=fun(trial)[0]
            if v<value-1e-10:
                endpoint=d;after=v;valid=True;break
    return endpoint,dict(seconds=time.perf_counter()-began,preparation_seconds=preparation,
                         loss_before=value,loss_after=after,derivative_relative_error=derivative_error,
                         optimizer_success=bool(opt.success),optimizer_message=str(opt.message),step_norm=float(np.linalg.norm(opt.x)),valid_step=bool(valid),lp_calls=info['lp_calls'](),
                         contact_generators=info['contact_generators'],critical_load_count=len(info['critical_loads']))
