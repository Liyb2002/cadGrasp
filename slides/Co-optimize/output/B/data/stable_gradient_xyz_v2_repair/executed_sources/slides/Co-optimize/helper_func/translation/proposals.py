"""Additional height proposals; horizontal candidates keep their own order."""
import numpy as np
from .layout_update import translation_frame,shift,derivative


def height_proposals(model,layout,owners,*,fine=False):
    rows=[];delta=model.extent*(1/512 if fine else 1/100)
    fractions=[1/1024,1/512,1/256] if fine else [1/128,1/64,1/32,1/16,1/8]
    for k in owners:
        frame=translation_frame(model,layout,k);g=np.zeros(3)
        for axis in range(3):
            negative=min(delta,max(0.,model.workpiece_height(layout,k))) if axis==2 else delta
            minus=shift(model,layout,k,-negative*frame[:,axis])
            plus=shift(model,layout,k,delta*frame[:,axis])
            g[axis]=(model.proxy(plus)['loss']-model.proxy(minus)['loss'])/(delta+negative)
        if model.workpiece_height(layout,k)<=1e-9:g[2]=min(g[2],0.)
        vectors=[(np.eye(3)[2]*sign,'sample') for sign in [-1,1]]
        if abs(g[2])>1e-28:vectors.insert(0,(-g/np.linalg.norm(g),'gradient'))
        for fraction in fractions:
            for vector,kind in vectors:
                trial=shift(model,layout,k,model.extent*fraction*frame@vector)
                if trial.key()==layout.key():continue
                rows.append(('translation-'+kind,trial,dict(pose=model.poses[k],
                    step_m=model.extent*fraction,finite_difference_m=delta,gradient=g.tolist(),
                    all_poses_in_objective=True,supplementary_xyz=True,translation_dimensions=3,
                    world_height_m=model.workpiece_height(trial,k),
                    height_derivative_kind='contact_event_secant_on_ground; resolved_difference_in_air')))
    return rows


def coherent_height_proposals(model,layout,owners,*,fine=False):
    rows=[];delta=model.extent*(1/512 if fine else 1/100)
    fractions=[1/1024,1/512,1/256,1/128] if fine else [1/128,1/64,1/32,1/16]
    g=np.array([derivative(model,layout,owners,axis,delta) for axis in range(3)])
    if min(model.workpiece_height(layout,k) for k in owners)<=1e-9:g[2]=min(g[2],0.)
    def move(world):
        q=layout.copy()
        for k in owners:q=shift(model,q,k,model.native[layout.hosts[k],:3,:3].T@world)
        return q
    if abs(g[2])>1e-28:
        for fraction in fractions:
            rows.append(('translation-gradient-coherent',move(-model.extent*fraction*g/np.linalg.norm(g)),
                dict(poses=[model.poses[k] for k in owners],step_m=model.extent*fraction,
                     finite_difference_m=delta,gradient=g.tolist(),all_poses_in_objective=True,
                     coordinate='shared_world_xyz',supplementary_xyz=True)))
    for fraction in fractions[:3]:
        for sign in [-1,1]:
            q=move(sign*model.extent*fraction*np.eye(3)[2])
            if q.key()!=layout.key():rows.append(('translation-coherent-sample',q,
                dict(step_m=model.extent*fraction,axis=[0.,0.,1.],sign=sign,supplementary_xyz=True)))
    return rows
