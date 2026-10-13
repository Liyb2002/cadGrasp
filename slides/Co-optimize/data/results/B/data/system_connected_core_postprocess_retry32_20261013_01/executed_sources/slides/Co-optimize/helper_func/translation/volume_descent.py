"""Feasible material descent in XYZ, on one shared occupancy measure."""
from itertools import product
import numpy as np
from .layout_update import translation_frame,shift,translation_gradient
from whole_search.reuse_first import registered
from whole_search.search import failed_count
from whole_search.common import tangent_frame


def polish_xyz(search,current,rounds=2):
    model=search.model
    if not getattr(model,'allow_z_translation',False) or failed_count(current):return current
    for iteration in range(rounds):
        layout=current['layout'];owners=[k for k in layout.active if not registered(layout,k)]
        frames={k:translation_frame(model,layout,k) for k in owners}
        radius=model.extent/32;delta=model.extent/512
        # Freeze the integration box before any volume difference or ranking.
        envelope=[shift(model,layout,k,radius*frames[k]@np.array(corner))
                  for k in owners for corner in product([-1.,1.],repeat=3)]
        guidance=model.volume_guidance([layout]+envelope);model.refresh_volume(current)
        base=guidance.estimate(layout);proposals=[]
        # Direction remains a material tool, evaluated in the same frozen box.
        from continuous_support.fast_gradient import turn
        for k in layout.active:
            frame=tangent_frame(layout.directions[k]);angle=np.radians(.125)
            gradient=np.array([(guidance.estimate(turn(model,layout,k,angle*axis))-
                                guidance.estimate(turn(model,layout,k,-angle*axis)))/(2*angle)
                               for axis in frame.T])
            vectors=[(axis*sign,'sample') for axis in np.eye(2) for sign in [-1,1]]
            length=float(np.linalg.norm(gradient))
            if length>1e-12:vectors.insert(0,(-gradient/length,'gradient'))
            for degrees in [.5,2.,5.]:
                for vector,kind in vectors:
                    trial=turn(model,layout,k,np.radians(degrees)*frame@vector)
                    detail=dict(pose=model.poses[k],step_degrees=degrees,gradient=gradient.tolist(),
                        gradient_objective='shared_nominal_material_volume_cm3',all_original_loads_protected=True)
                    proposals.append(('direction-volume-'+kind,trial,detail))
        for k in owners:
            frame=frames[k];g=translation_gradient(model,layout,[k],delta,guidance.estimate)
            vectors=[(axis*sign,'sample') for axis in np.eye(3) for sign in [-1,1]]
            if np.linalg.norm(g)>1e-12:vectors.insert(0,(-g/np.linalg.norm(g),'gradient'))
            for fraction in [1/128,1/64,1/32]:
                for vector,kind in vectors:
                    trial=shift(model,layout,k,model.extent*fraction*frame@vector)
                    detail=dict(pose=model.poses[k],step_m=model.extent*fraction,gradient=g.tolist(),
                        gradient_objective='shared_nominal_material_volume_cm3',finite_difference_m=delta,
                        world_height_m=model.workpiece_height(trial,k),all_original_loads_protected=True)
                    proposals.append(('translation-volume-'+kind,trial,detail))
        ranked=[];seen={layout.key()}
        for kind,trial,detail in proposals:
            if trial.key() in seen:continue
            seen.add(trial.key());estimate=guidance.estimate(trial)
            if estimate<base-1e-9:ranked.append((estimate,kind,trial,detail))
        ranked.sort(key=lambda r:r[0]);event=dict(phase='feasible_xyz_volume_descent',iteration=iteration+1,
            before_volume_cm3=base,volume_epoch=guidance.epoch,sampled_candidates=len(proposals),trials=[])
        best=current
        for estimate,kind,trial,detail in ranked[:6]:
            record=dict(operation=kind,detail=detail,estimated_volume_cm3=estimate,accepted=False)
            try:
                checked=model.evaluate(trial)
                record.update(counts=checked['counts'],volume_cm3=checked['volume_cm3'],serial=checked['serial'])
                if not failed_count(checked) and checked['volume_cm3']<base-1e-9:
                    best=checked;record['accepted']=True
            except (RuntimeError,ValueError,AssertionError) as error:record['error']=str(error)
            event['trials'].append(record)
            if best is not current:break
        event['accepted']=best is not current
        if best is not current:current=best;search.checkpoint(current,event['phase'])
        search.record(event);model.commit(current)
        print('XYZ MATERIAL',iteration+1,'accepted',event['accepted'],'volume',current['volume_cm3'],flush=True)
        if not event['accepted']:break
    return current
