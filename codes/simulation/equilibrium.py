"""Inertia-free constant-load equilibrium using MuJoCo collision geometry.

This solves contact reactions; it does not integrate motion or constrain body pose.
A feasible result is a target-pose equilibrium certificate, not a perturbation test.
"""
import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

CONTACT_TOL_M=1e-7  # Compiled meshes use float32; initial mismatch is about 1e-8 m.


def geometric_contact_check(model,data,contact,cache):
    """Independently validate the reported point and normal on both convex solids."""
    solids=[]
    for gid in contact.geom:
        gid=int(gid)
        if model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_PLANE:
            solids.append(None);continue
        if gid not in cache:
            mid=model.geom_dataid[gid];a=model.mesh_vertadr[mid];n=model.mesh_vertnum[mid]
            v=model.mesh_vert[a:a+n].astype(float)@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
            cache[gid]=(v,ConvexHull(v).equations)
        solids.append(cache[gid])
    normal=contact.frame[:3];point=contact.pos
    point_error=max([float(np.max(e[:,:3]@point+e[:,3])) for solid in solids if solid is not None for _,e in [solid]],default=0.)
    if any(s is None for s in solids):
        normal_error=abs(abs(normal[2])-1.)
        gap_error=abs(float(point[2]))
    else:
        a,b=[s[0]@normal for s in solids]
        gap_error=abs(float(b.min()-a.max()))
        normal_error=0.
        point_error=max(point_error,abs(float(point@normal-.5*(a.max()+b.min()))))
    return dict(valid=bool(gap_error<=CONTACT_TOL_M and point_error<=CONTACT_TOL_M and normal_error<1e-7),
                supporting_plane_gap_error_m=gap_error,point_outside_error_m=max(0.,point_error))


def problem(model, case, support_weight=True):
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    bodies=[model.body(name).id for name in ('object','support')]
    point=np.asarray(case['point_body_m'],float)
    force=np.asarray(case['force_body_mg'],float)*model.body_mass[bodies[0]]*9.81
    external=np.zeros((2,6))
    for i,body in enumerate(bodies):
        if i==0 or support_weight:external[i,:3]=model.body_mass[body]*model.opt.gravity
    external[0,:3]+=force
    external[0,3:]+=np.cross(point-data.xipos[bodies[0]],force)
    columns=[];records=[];geometry_cache={};excluded=[]
    for ci,contact in enumerate(data.contact):
        if abs(contact.dist)>CONTACT_TOL_M:continue
        audit=geometric_contact_check(model,data,contact,geometry_cache)
        if not audit['valid']:
            excluded.append(dict(contact_index=ci,geom1=model.geom(contact.geom1).name,geom2=model.geom(contact.geom2).name,**audit))
            continue
        b1,b2=[int(model.geom_bodyid[g]) for g in contact.geom]
        is_floor=0 in (b1,b2)
        normal=contact.frame.reshape(3,3)[0]
        if not is_floor and contact.dim!=1:raise ValueError('Object-support must be normal-only')
        if is_floor:
            frame=contact.frame.reshape(3,3)
            # A fixed finite witness for the declared sufficiently non-sliding floor.
            # Rays remain inside the model's existing cone; no parameter search.
            directions=[normal+contact.friction[0]*(np.cos(p)*frame[1]+np.sin(p)*frame[2]) for p in np.arange(16)*2*np.pi/16]
        else:directions=[normal]
        for direction in directions:
            column=np.zeros((2,6))
            for body,sgn in ((b1,-1),(b2,1)):
                if body==0:continue
                row=bodies.index(body)
                column[row,:3]=sgn*direction
                column[row,3:]=np.cross(contact.pos-data.xipos[body],sgn*direction)
            columns.append(column.ravel())
            records.append(dict(contact_index=ci,body1=b1,body2=b2,point_m=contact.pos.copy(),
                direction_on_body2=direction.copy(),normal=normal.copy(),gap_m=float(contact.dist),
                geom1=model.geom(contact.geom1).name,geom2=model.geom(contact.geom2).name))
    return data,bodies,point,force,external,np.array(columns).T,records,excluded


def solve(model, case, support_weight=True):
    data,bodies,point,force,external,A,records,excluded=problem(model,case,support_weight)
    scale=np.tile([1,1,1,10,10,10],2)
    fit=linprog(np.ones(A.shape[1]),A_eq=scale[:,None]*A,b_eq=-external.ravel()*scale,
                bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    result=dict(id=case['id'],case=case,mode='quasistatic_constant_world_force',
        force_point_world_m=point.tolist(),force_world_N=force.tolist(),
        force_magnitude_N=float(np.linalg.norm(force)),force_magnitude_mg=float(np.linalg.norm(case['force_body_mg'])),
        support_weight_included=support_weight,contact_geometry_tolerance_m=CONTACT_TOL_M,
        contact_point_count=len(data.contact),candidate_ray_count=len(records),
        independently_rejected_collision_contacts=excluded,
        solver_status=int(fit.status),solver_message=fit.message,
        status='equilibrium_feasible' if fit.success else 'equilibrium_unresolved',
        claim='Existence of unilateral contact reactions balancing both free bodies at the target pose; zero velocity and acceleration. No robustness, stiffness or transient-motion claim.',
        videos_are='Equilibrium-state visualization with a persistent world-fixed arrow; display time is not integrated dynamics time.',
        initial_velocity_zero=True,kinetic_energy_J=0.,dynamic_integration_steps=0,
        joints_are_free=bool(model.nv==12 and model.neq==0))
    if not fit.success:return result,None
    # Independently sum forces and moment components, without multiplying A.
    balance=external.copy();witness=[]
    for weight,rec in zip(fit.x,records):
        if weight<=0:continue
        applied=weight*rec['direction_on_body2']
        for body,sgn in ((rec['body1'],-1),(rec['body2'],1)):
            if body==0:continue
            i=bodies.index(body);f=sgn*applied;r=rec['point_m']-data.xipos[body]
            balance[i,:3]+=f
            balance[i,3:]+=np.array([r[1]*f[2]-r[2]*f[1],r[2]*f[0]-r[0]*f[2],r[0]*f[1]-r[1]*f[0]])
        witness.append({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in rec.items()}|dict(normal_ray_weight_N=float(weight),force_on_body2_N=applied.tolist()))
    force_res=float(abs(balance[:,:3]).max());moment_res=float(abs(balance[:,3:]).max())
    assert force_res<1e-7 and moment_res<1e-8 and fit.x.min()>=-1e-10
    # Check that these contact reactions cause zero acceleration in the same free
    # bodies under Newton/Euler, with native soft constraints disabled temporarily.
    old_type=model.geom_contype.copy();old_aff=model.geom_conaffinity.copy()
    try:
        model.geom_contype[:]=0;model.geom_conaffinity[:]=0
        d=mujoco.MjData(model);mujoco.mj_forward(model,d)
        mujoco.mj_applyFT(model,d,force,np.zeros(3),point,bodies[0],d.qfrc_applied)
        if not support_weight:
            mujoco.mj_applyFT(model,d,-model.body_mass[bodies[1]]*model.opt.gravity,np.zeros(3),d.xipos[bodies[1]],bodies[1],d.qfrc_applied)
        for rec in witness:
            f=np.array(rec['force_on_body2_N']);p=np.array(rec['point_m'])
            for body,sgn in ((rec['body1'],-1),(rec['body2'],1)):
                if body: mujoco.mj_applyFT(model,d,sgn*f,np.zeros(3),p,body,d.qfrc_applied)
        mujoco.mj_forward(model,d)
        jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);accelerations=[]
        for body in bodies:
            mujoco.mj_jacBodyCom(model,d,jp,jr,body)
            accelerations.append(dict(body=model.body(body).name,com_acceleration_m_s2=(jp@d.qacc).tolist(),angular_acceleration_rad_s2=(jr@d.qacc).tolist()))
        assert max(np.linalg.norm(x[k]) for x in accelerations for k in ('com_acceleration_m_s2','angular_acceleration_rad_s2'))<1e-5
    finally:model.geom_contype[:]=old_type;model.geom_conaffinity[:]=old_aff
    result.update(max_force_balance_residual_N=force_res,max_moment_balance_residual_Nm=moment_res,
        minimum_normal_ray_weight_N=float(fit.x.min()),active_ray_count=len(witness),
        maximum_active_geometric_gap_m=max(abs(r['gap_m']) for r in witness),
        newton_euler_acceleration_check=accelerations,reaction_witness=witness)
    return result,dict(matrix_SI=A,coefficients_N=fit.x,external_wrenches_SI=external)

