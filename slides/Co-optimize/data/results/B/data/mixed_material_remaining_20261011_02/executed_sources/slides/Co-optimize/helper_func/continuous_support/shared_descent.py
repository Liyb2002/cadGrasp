"""Block-coordinate differences, with every pose retained in the objective."""
import numpy as np
from whole_search.common import tangent_frame,legal_direction
from whole_search.reuse_first import registered

def block_descent(objective,current,kind,difference,step,backtracks=4,*,coupled=True,verification_refinements=2, owners=None):
    """Projected block gradient; backtracking evaluates ONE descent direction.

    All original poses contribute to every difference. There is no hardest
    demand selection, operation-candidate pool or random design sampling.
    Translation coordinates are scaled by object extent.
    """
    model=objective.model;layout=current.layout
    phase='volume' if objective.covered(current) else 'coverage'
    coordinates=[];one_sided=[];differences=[];groups=[]
    for owner in layout.active:
        if owners is not None and owner not in owners:continue
        if kind=='translation' and registered(layout,owner):continue
        group=None
        if kind=='direction' and coupled:
            for existing in groups:
                first=existing[0]
                if np.allclose(layout.placements[owner],layout.placements[first],atol=1e-10,rtol=0) and all(layout.directions[owner]@layout.directions[k]>=np.cos(np.radians(3.)) for k in existing):
                    group=existing;break
        if group is None:groups.append([owner])
        else:group.append(owner)
    for members in groups:
        owner=members[0]
        normal=(np.mean(layout.directions[members],axis=0) if kind=='direction' else model.floor_normal(layout,owner))
        normal=normal/np.linalg.norm(normal)
        frame=tangent_frame(normal)
        boundary=False;owner_difference=difference
        if kind=='direction':
            floors=np.array([model.floor_normal(layout,k) for k in members])
            margins=np.einsum('ij,ij->i',floors,layout.directions[members])
            floor=floors[int(np.argmin(margins))]
            margin=float(normal@floor);inward=floor-margin*normal
            length=float(np.linalg.norm(inward))
            if length>1e-8:
                inward/=length;frame=np.column_stack([inward,np.cross(normal,inward)])
        for axis in range(2):
            tangent=frame[:,axis];side=0;distance=difference
            if kind=='direction':
                slopes=(floors-margins[:,None]*layout.directions[members])@tangent;boundary=margins<1e-5
                if boundary.any():
                    plus=bool(np.all(slopes[boundary]>=-1e-10));minus=bool(np.all(slopes[boundary]<=1e-10))
                    side=0 if plus and minus else (1 if plus else (-1 if minus else 2))
                interior=(~boundary)&(np.abs(slopes)>1e-10)
                if interior.any():distance=min(distance,float(np.min(.45*np.arctan2(margins[interior],np.abs(slopes[interior])))))
            coordinates.append((tuple(members),tangent))
            one_sided.append(side);differences.append(distance)

    def perturb(vector):
        trial=layout.copy()
        angular={}
        for value,(members,tangent) in zip(vector,coordinates):
            for owner in members:
                if kind=='direction':
                    angular[owner]=angular.get(owner,np.zeros(3))+value*tangent
                else:trial.placements[owner,:3,3]+=model.extent*value*tangent
        if kind=='direction':
            for owner,delta in angular.items():
                delta=delta-layout.directions[owner]*(delta@layout.directions[owner])
                angle=float(np.linalg.norm(delta))
                if angle>0:
                    direction=np.cos(angle)*layout.directions[owner]+np.sin(angle)/angle*delta
                    trial.directions[owner]=legal_direction(direction,model.floor_normal(trial,owner))
        return trial

    gradient=np.zeros(len(coordinates));probes=[]
    layouts=[]
    for index in range(len(coordinates)):
        unit=np.zeros(len(coordinates));unit[index]=differences[index]
        side=one_sided[index]
        layouts.append((layout if side in (1,2) else perturb(-unit),layout if side in (-1,2) else perturb(unit)))
    # Every volume difference uses the same integration box and Sobol points.
    if hasattr(objective,'refresh'):
        envelope=[layout]+[q for pair in layouts for q in pair]
        if kind=='translation':
            radius=max(step,difference)
            for index in range(0,len(coordinates),2):
                for a in [-radius,radius]:
                    for b in [-radius,radius]:
                        vector=np.zeros(len(coordinates));vector[index:index+2]=[a,b]
                        envelope.append(perturb(vector))
        model.volume_guidance(envelope)
    center=objective.value(current,phase)
    for index,((members,axis),(left_layout,right_layout)) in enumerate(zip(coordinates,layouts)):
        left=objective.evaluate(left_layout);right=objective.evaluate(right_layout)
        divisor=differences[index]*(1 if one_sided[index] else 2)
        gradient[index]=(objective.value(right,phase)-objective.value(left,phase))/divisor
        if (one_sided[index]==1 and gradient[index]>0) or (one_sided[index]==-1 and gradient[index]<0) or one_sided[index]==2:gradient[index]=0.
        probes.append(dict(pose=model.poses[members[0]],poses=[model.poses[k] for k in members],coordinate=index,
                           difference=differences[index],one_sided=one_sided[index],
                           minus_value=objective.value(left,phase),plus_value=objective.value(right,phase),
                           minus_coverage=left.coverage.tolist(),plus_coverage=right.coverage.tolist(),
                           contact_normal_events=[a['pose'] for a,b in zip(left.details,right.details)
                               if a.get('contact_normal_signature')!=b.get('contact_normal_signature')]))
    norm=float(np.linalg.norm(gradient));trials=[];best=current
    derivative_check=None
    if norm>1e-12:
        block_norm=max(np.linalg.norm(gradient[j:j+2]) for j in range(0,len(gradient),2))
        direction=-gradient/block_norm
        check_step=min(differences)/2
        if any(one_sided):
            positive=objective.evaluate(perturb(check_step*direction))
            measured=(objective.value(positive,phase)-objective.value(current,phase))/check_step
            curvature=0.
        else:
            negative=objective.evaluate(perturb(-check_step*direction))
            positive=objective.evaluate(perturb(check_step*direction))
            center=objective.value(current,phase)
            measured=(objective.value(positive,phase)-objective.value(negative,phase))/(2*check_step)
            curvature=(objective.value(positive,phase)+objective.value(negative,phase)-2*center)/check_step**2
        predicted=float(gradient@direction)
        relative_error=float(abs(measured-predicted)/max(abs(predicted),abs(measured),1e-12))
        normal_event=any(row['contact_normal_events'] for row in probes)
        consistent=bool(measured<0 and relative_error<=.25)
        derivative_check=dict(step=check_step,predicted=predicted,measured=measured,
                              curvature=curvature,descent_confirmed=bool(measured<0),
                              relative_error=relative_error,local_derivative_consistent=consistent,
                              contact_normal_event=normal_event,
                              forward_value=objective.value(positive,phase),center_value=objective.value(current,phase))
        if not consistent and not normal_event:
            if verification_refinements>0:
                refined,record=block_descent(objective,current,kind,difference/2,step,backtracks,
                    coupled=coupled,verification_refinements=verification_refinements-1,owners=owners)
                record.setdefault('gradient_refinement_attempts',[]).append(dict(difference=difference,
                    gradient=gradient.tolist(),derivative_check=derivative_check))
                return refined,record
            return current,dict(operation=kind,accepted=False,phase=phase,gradient=gradient.tolist(),
                gradient_norm=norm,difference=difference,probes=probes,line_search=[],derivative_check=derivative_check,
                coordinate_groups=[[model.poses[k] for k in members] for members in groups],
                update_kind='unresolved_directional_derivative',all_pose_gain_and_loss_included=True,
                random_design_sampling=False)
        maximum_step=step
        if curvature>0 and measured<0:maximum_step=min(step,max(check_step,-measured/curvature))
        for attempt in range(backtracks):
            amount=maximum_step/(2**attempt)
            trial=objective.evaluate(perturb(amount*direction))
            accepted=objective.acceptable(current,trial,phase)
            # Re-evaluate the incumbent after any shared-volume-box rebase.
            accepted=bool(accepted and objective.value(trial,phase)<=objective.value(current,phase)+1e-4*amount*predicted)
            trials.append(dict(step=amount,accepted=accepted,coverage=trial.coverage.tolist(),
                               before_value=objective.value(current,phase),after_value=objective.value(trial,phase),
                               estimated_material_cm3=trial.volume_cm3))
            if accepted:best=trial;break
        if best is not current and not consistent and normal_event:
            lower=0.;upper=amount
            for refinement in range(3):
                midpoint=(lower+upper)/2
                trial=objective.evaluate(perturb(midpoint*direction))
                accepted=bool(objective.acceptable(current,trial,phase))
                trials.append(dict(step=midpoint,accepted=accepted,event_step_refinement=True,
                    before_value=objective.value(current,phase),after_value=objective.value(trial,phase),
                    coverage=trial.coverage.tolist(),estimated_material_cm3=trial.volume_cm3))
                if accepted:upper=midpoint;best=trial
                else:lower=midpoint
    return best,dict(operation=kind,accepted=best is not current,phase=phase,
                     gradient_method='feasible finite differences of all-pose integrated cone distance / material',
                     gradient_norm=norm,gradient=gradient.tolist(),difference=difference,
                     derivative_check=derivative_check,
                     update_kind=('local_gradient' if derivative_check is None or derivative_check['local_derivative_consistent'] else 'contact_event_continuation'),
                     coordinate_groups=[[model.poses[k] for k in members] for members in groups],
                     all_pose_gain_and_loss_included=True,probes=probes,line_search=trials,
                     random_design_sampling=False)


def block_direction_choice(objective,current,owners,*,difference_degrees=.75,step_degrees=2.,backtracks=4):
    updated,record=block_descent(objective,current,'direction',np.radians(difference_degrees),np.radians(step_degrees),backtracks,owners=owners)
    if not record['accepted'] and any(len(group)>1 for group in record.get('coordinate_groups',[])):
        updated,individual=block_descent(objective,current,'direction',np.radians(difference_degrees),np.radians(step_degrees),backtracks,coupled=False,owners=owners)
        individual['shared_direction_attempt']=record
        return updated,individual
    return updated,record
