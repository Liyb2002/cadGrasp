"""Object-only XY proposals with saved fixture floors and native heights.

Single-floor necessary bounds include each object's real floor contacts.
Their success is deliberately not labelled a joint-foot or base proof.
"""
import numpy as np
from scipy.optimize import linprog
import stage5_common as C


def solve(data):
    n=len(data['rows']);size=2*n+1;matrix=[];rhs=[];labels=[]
    for i,row in enumerate(data['rows']):
        inverse=np.linalg.inv(row['fixture_transform'])
        for j,other in enumerate(data['rows']):
            if i==j:continue
            transform=other['fixture_transform']@inverse
            a=transform[2,:2];b=float(transform[2,3])
            projections=row['cloud'][:,:2]@a
            own=row['object_ground']
            indices=np.flatnonzero(projections<float((own@a).min())-1e-10) if len(own) else np.arange(len(projections))
            if not len(indices):continue
            index=int(indices[np.argmin(projections[indices])])
            v=np.zeros(size);v[2*i:2*i+2]=-a
            matrix.append(v);rhs.append((projections[index]+b)*1000)
            labels.append(dict(task=i,other=j,original_demand_index=index,object_floor_included=bool(len(own))))
    for coordinate in range(2*n):
        for sign in [-1.,1.]:
            v=np.zeros(size);v[coordinate]=sign;v[-1]=-1
            matrix.append(v);rhs.append(0.)
    objective=np.zeros(size);objective[-1]=1
    result=linprog(objective,A_ub=np.asarray(matrix),b_ub=np.asarray(rhs),
                   bounds=[(None,None)]*(2*n)+[(0.,None)],method='highs')
    if result.status==2:return dict(passed=False,status='xy_single_floor_bounds_infeasible',mode='xy_only',floor_constraints=labels)
    if not result.success:raise RuntimeError(result.message)
    relative=np.zeros((n,3));relative[:,:2]=result.x[:2*n].reshape(n,2)/1000
    return dict(passed=True,status='xy_single_floor_necessary_bounds_passed',mode='xy_only',
                fixture_lift_m=np.zeros(n).tolist(),object_relative_world_translation_m=relative.tolist(),
                total_object_world_translation_m=relative.tolist(),maximum_displacement_coordinate_mm=float(result.x[-1]),
                actual_base_constructed=False,original_contact_capacity_rechecked=False,
                own_object_floor_contacts_included=True,floor_constraints=labels)
