"""Linear seating proposals, checking every original coupled wrench.

Fixture rotations and exit directions stay fixed.  h_i lifts the assembly;
r_i translates only the object relative to that lifted fixture.  Original
results are read-only.  Actual object translation is r_i+h_i*world_z.
The finite contact/solid construction is a separate acceptance stage.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

import stage5_common as C


def load_demands(data, inputs):
    for row, saved_row in zip(data['rows'], data['record']['per_pose']):
        path = inputs.base/data['case']['id']/'step5/step5.1'/saved_row['arrays']
        with np.load(path, allow_pickle=False) as saved:
            row['old_wrench_world_origin'] = saved['wrench_world_origin'].copy()
            row['horizontal_force_ratio'] = saved['wrench_world_origin'][:, :2]/saved['wrench_world_origin'][:, 2, None]


class Seating:
    def __init__(self, data, width=.005, preserve_material=True, protect_full_wrap=True):
        self.data, self.width = data, width
        self.rows, self.states = data['rows'], np.asarray(data['states'])
        self.n = len(self.rows)
        self.minimum_lift = np.array([max(0., -C.F.transform_points(data['fixture'].vertices, t)[:, 2].min())+.001
                                      if preserve_material else .001 for t in self.states])
        self.banks = []
        self.head_bounds = []
        for i, row in enumerate(self.rows):
            inverse = np.linalg.inv(self.states[i])
            common_body = C.F.transform_points(C.F.transform_points(data['body'].vertices, row['object_transform']), inverse)
            for j, state in enumerate(self.states):
                transform = state @ inverse
                a, c = transform[2, :2], float(transform[2, 2])
                if protect_full_wrap or i == j:
                    head_min = float(C.F.transform_points(common_body, state)[:, 2].min())
                    self.head_bounds.append((i, j, a, c, head_min))
                if i == j:
                    continue
                s = row['horizontal_force_ratio'] @ a
                b = row['cloud'][:, :2] @ a + transform[2, 3]
                margin = width/2*np.abs(a).sum()+width*max(0., -c)
                self.banks.append((i, j, a, s+c, s, b, margin))

    def solve(self, mode='z_xy', relative_limit=None, extra_lift=0., max_iterations=150, fixed_relative=None,
              access_constraints=(), fixed_fixture_lifts=None, fixed_relative_xy=None):
        dimensions = {'z':0, 'z_xy':2, 'z_xyz':3}[mode]
        n, ndim = self.n, dimensions
        relative_start, absolute_start = n, n+n*ndim
        t_index = n+2*n*ndim
        size = t_index+1
        cuts, right, labels, unique = [], [], [], set()
        def add(vector, bound, label):
            cuts.append(vector); right.append(bound*1000); labels.append(label)
        # Millimetres condition positional variables; coefficients are unitless.
        bounds = [(float((v+extra_lift)*1000), None) for v in self.minimum_lift]
        bounds += [(None, None)]*(n*ndim)+[(0., None)]*(n*ndim)+[(0., None)]
        if fixed_fixture_lifts is not None:
            for i,value in enumerate(fixed_fixture_lifts):bounds[i]=(float(value*1000),float(value*1000))
        if fixed_relative_xy is not None:
            for i in range(n):
                for axis in range(min(2,ndim)):
                    value=float(fixed_relative_xy[i,axis]*1000)
                    bounds[relative_start+i*ndim+axis]=(value,value)
        if fixed_relative is not None:
            fixed_relative=np.asarray(fixed_relative)
            for i in range(n):
                for axis in range(ndim):
                    coordinate=float(fixed_relative[i,axis]*1000)
                    bounds[relative_start+i*ndim+axis]=(coordinate,coordinate)
        def relative_index(i, axis):
            return relative_start+i*ndim+axis
        for i in range(n):
            vector = np.zeros(size); vector[i]=1.; vector[t_index]=-1.
            add(vector, 0., dict(kind='maximum_fixture_lift', task_index=i))
            if ndim == 3:
                vector = np.zeros(size); vector[i]=-1.; vector[relative_index(i, 2)]=-1.
                add(vector, 0., dict(kind='object_not_moved_down', task_index=i))
                vector = np.zeros(size); vector[i]=1.; vector[relative_index(i, 2)]=1.; vector[t_index]=-1.
                add(vector, 0., dict(kind='maximum_total_object_lift', task_index=i))
            for axis in range(ndim):
                for sign in [-1., 1.]:
                    vector = np.zeros(size)
                    vector[relative_index(i, axis)]=sign
                    vector[absolute_start+i*ndim+axis]=-1.
                    add(vector, 0., dict(kind='absolute_relative_shift', task_index=i, axis=axis))
                    vector = np.zeros(size); vector[relative_index(i, axis)]=sign; vector[t_index]=-1.
                    add(vector, 0., dict(kind='maximum_relative_shift', task_index=i, axis=axis))
                    if relative_limit is not None:
                        vector = np.zeros(size); vector[relative_index(i, axis)]=sign
                        add(vector, float(relative_limit), dict(kind='relative_shift_limit', task_index=i, axis=axis))
        for i,j,a,c,minimum in self.head_bounds:
            vector = np.zeros(size); vector[j]=-1.
            if ndim:
                vector[relative_index(i, 0)]=-a[0]; vector[relative_index(i, 1)]=-a[1]
            if ndim == 3: vector[relative_index(i, 2)]=-c
            add(vector, minimum-self.width-.001,
                dict(kind='complete_wrap_above_every_floor', task_index=i, other_index=j))
        for constraint in access_constraints:
            vector=np.zeros(size); vector[:n]=-np.asarray(constraint['height_coefficients'])
            if ndim:
                vector[relative_start:absolute_start]=-np.asarray(constraint['relative_coefficients'])[:,:ndim].reshape(-1)
            add(vector, -float(constraint['lower_bound_m']), dict(kind='ground_access_facet', obstacle=constraint['label']))
        objective = np.zeros(size); objective[t_index]=1.
        heights = self.minimum_lift.copy()+extra_lift
        relative = np.zeros((n, 3))
        answer = None
        for iteration in range(max_iterations):
            for i,j,a,k,s,b,margin in self.banks:
                violations=k*heights[i]-heights[j]+s*relative[i, 2]-a @ relative[i, :2]-b+margin
                index=int(np.argmax(violations)); key=(i,j,index)
                if violations[index]>1e-9 and key not in unique:
                    vector=np.zeros(size); vector[i]=k[index]; vector[j]-=1.
                    if ndim:
                        vector[relative_index(i, 0)]=-a[0]; vector[relative_index(i, 1)]=-a[1]
                    if ndim == 3: vector[relative_index(i, 2)]=s[index]
                    add(vector, b[index]-margin, dict(kind='original_demand_ground_compatibility',
                        task_index=i, other_index=j, demand_index=index))
                    unique.add(key)
            answer=linprog(objective, A_ub=np.asarray(cuts), b_ub=np.asarray(right), bounds=bounds,
                           method='highs', options={'primal_feasibility_tolerance':1e-9,
                                                    'dual_feasibility_tolerance':1e-9})
            if answer.status == 2:
                return dict(passed=False, status='linear_seating_constraints_infeasible', mode=mode,
                            iterations=iteration+1, full_demand_cuts=len(unique),
                            bound_labels=labels, constraint_matrix=np.asarray(cuts).tolist(),
                            constraint_rhs_mm=right, variable_bounds_mm=[list(v) for v in bounds],
                            contact_domain_floor_constraint=self.head_bounds!=[], foot_width_m=self.width)
            if not answer.success: raise RuntimeError(f'Seating LP unresolved: {answer.message}')
            heights=answer.x[:n]/1000
            if ndim: relative[:, :ndim]=answer.x[relative_start:absolute_start].reshape(n, ndim)/1000
            worst=max(float(np.max(k*heights[i]-heights[j]+s*relative[i, 2]-a @ relative[i, :2]-b+margin))
                      for i,j,a,k,s,b,margin in self.banks)
            if worst<=1e-9:
                break
        else:
            return dict(passed=False, status='cutting_plane_iteration_limit', mode=mode, iterations=max_iterations)
        # A second LP minimizes total movement while keeping the best maximum.
        optimum=float(answer.x[t_index])
        vector=np.zeros(size); vector[t_index]=1.
        add(vector, (optimum+1e-5)/1000, dict(kind='retain_minimum_maximum_displacement'))
        second=np.zeros(size); second[:n]=1.
        second[absolute_start:t_index]=1.
        if ndim == 3:
            for i in range(n): second[relative_index(i, 2)]+=.1
        refined=linprog(second, A_ub=np.asarray(cuts), b_ub=np.asarray(right), bounds=bounds, method='highs')
        # New minimizers can expose a different demand. Retain the fully
        # checked first solution unless this refinement also covers all loads.
        if refined.success:
            candidate_h=refined.x[:n]/1000; candidate_r=np.zeros((n, 3))
            if ndim: candidate_r[:, :ndim]=refined.x[relative_start:absolute_start].reshape(n, ndim)/1000
            candidate_gap=max(float(np.max(k*candidate_h[i]-candidate_h[j]+s*candidate_r[i, 2]
                -a @ candidate_r[i, :2]-b+margin)) for i,j,a,k,s,b,margin in self.banks)
            if candidate_gap<=1e-9: heights,relative,worst=candidate_h,candidate_r,candidate_gap
        total=relative.copy(); total[:, 2]+=heights
        return dict(passed=True, status='all_original_demands_pass_linear_floor_constraints', mode=mode,
                    fixture_lift_m=heights.tolist(), object_relative_world_translation_m=relative.tolist(),
                    total_object_world_translation_m=total.tolist(),
                    maximum_displacement_coordinate_mm=optimum,
                    minimum_material_floor_lifts_mm=(self.minimum_lift*1000).tolist(),
                    largest_remaining_demand_violation_m=max(0., worst),
                    full_demand_cuts=len(unique), iterations=iteration+1,
                    foot_width_m=self.width, contact_domain_floor_constraint=self.head_bounds!=[],
                    actual_base_constructed=False, original_contact_capacity_rechecked=False)


def shifted(data, seating):
    result=dict(data)
    rows=[]
    for index, original in enumerate(data['rows']):
        row=dict(original)
        height=seating['fixture_lift_m'][index]
        delta=np.asarray(seating['total_object_world_translation_m'][index])
        ft=original['fixture_transform'].copy(); ft[2, 3]+=height
        ot=original['object_transform'].copy(); ot[:3, 3]+=delta
        wrench=original['old_wrench_world_origin'].copy()
        wrench[:, 3:]+=np.cross(delta, wrench[:, :3])
        # Forward projection of every original paired wrench at the new COM.
        points=np.c_[-wrench[:, 4]/wrench[:, 2], wrench[:, 3]/wrench[:, 2], np.zeros(len(wrench))]
        hull=ConvexHull(points[:, :2])
        body_world=C.F.transform_points(data['body'].vertices, ot)
        row.update(fixture_transform=ft, object_transform=ot, cloud=points, hull=points[hull.vertices],
                   hull_original_demand_indices=hull.vertices,
                   object_ground=body_world[np.abs(body_world[:, 2])<=1e-9, :2],
                   wrench_world_origin=wrench, seating_delta_world_m=delta,
                   fixture_lift_m=height)
        rows.append(row)
    result.update(rows=rows, states=[row['fixture_transform'] for row in rows],
                  bounds=C.occupied_bounds(data['body'], data['fixture'], rows), seating=seating)
    return result
