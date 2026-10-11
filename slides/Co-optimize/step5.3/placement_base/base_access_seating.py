"""Choose local separating facets so complete ground feet avoid access cones.

This is a deterministic, finite branch heuristic; its branch infeasibility
is not a proof that all translations or sparse foot arrangements fail.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

import stage5_common as C
from work_access import WorkAccess
import base_seating as B


class AccessSeating:
    def __init__(self, data, tasks, body, *, width=.005, strategy='nearest'):
        self.data,self.tasks,self.body=data,tasks,body
        self.n=len(tasks);self.width=width;self.strategy=strategy
        self.rotations=np.array(data['states'])[:,:3,:3]
        self.inverse_rotations=np.transpose(self.rotations,(0,2,1))
        self.families=[]
        for j,(task,row) in enumerate(zip(tasks,data['rows'])):
            placement=np.linalg.inv(row['fixture_transform'])@row['object_transform']
            mesh=body.copy();mesh.apply_transform(placement)
            work=WorkAccess(mesh,task.domain.work_ids,task.domain.data['load']['cone_half_deg'])
            for face,count in enumerate(work.counts):
                self.families.append((j,'work',face,work.planes[face,:count].copy()))
            direction=np.asarray(row['common_exit_direction']);direction/=np.linalg.norm(direction)
            extent=float(body.extents.max())*4
            points=np.vstack([mesh.vertices,mesh.vertices+extent*direction])
            hull=ConvexHull(points)
            equations=hull.equations[hull.equations[:,:3]@direction<=1e-9].copy()
            _,ids=np.unique(np.round(equations,10),axis=0,return_index=True)
            equations=equations[np.sort(ids)]
            equations[:,3]-=.01*float(body.extents.max())
            self.families.append((j,'exit',0,equations))

    def coefficients(self,i,index,j,equation):
        normal=equation[:3]
        ratio=self.data['rows'][i]['horizontal_force_ratio'][index]
        plane_point=self.data['rows'][i]['cloud'][index]
        common=C.F.transform_points(plane_point,np.linalg.inv(self.data['states'][i]))
        height=np.zeros(self.n);relative=np.zeros((self.n,3))
        height[i]=-normal@self.inverse_rotations[i]@np.r_[ratio,1.]
        relative[i,:2]=normal@self.inverse_rotations[i][:,:2]
        relative[i,2]=-normal@self.inverse_rotations[i]@np.r_[ratio,0.]
        relative[j]-=normal@self.inverse_rotations[j]
        constant=float(normal@common+equation[3])
        return height,relative,constant

    def violations(self, seating, *, max_new=20, mode='z_xy'):
        moved=B.shifted(self.data,seating)
        heights=np.asarray(seating['fixture_lift_m'])
        relative=np.asarray(seating['object_relative_world_translation_m'])
        translated=np.einsum('nij,nj->ni',self.inverse_rotations,relative)
        events=[]; unsafe_count=0
        for i,row in enumerate(moved['rows']):
            points=C.F.transform_points(row['hull'],np.linalg.inv(row['fixture_transform']))
            indices=row['hull_original_demand_indices']
            for j,kind,face,equations in self.families:
                normals=equations[:,:3]
                values=points@normals.T+equations[:,3]-normals@translated[j]
                world_normals=normals@self.rotations[i].T
                margins=self.width/2*np.abs(world_normals[:,:2]).sum(1)+self.width*np.maximum(0.,-world_normals[:,2])+1e-6
                deficits=margins[None]-values
                unsafe=np.flatnonzero(np.min(deficits,axis=1)>1e-8)
                unsafe_count+=len(unsafe)
                for k in unsafe:
                    options=[]
                    for plane,equation in enumerate(equations):
                        hc,rc,constant=self.coefficients(i,int(indices[k]),j,equation)
                        active_r=rc[:,:{'z':0,'z_xy':2,'z_xyz':3}[mode]]
                        gradient=np.sqrt(float(hc@hc)+float((active_r*active_r).sum()))
                        if gradient<1e-12:continue
                        if i==j and hc[i]<=1e-12:continue
                        cost=float(deficits[k,plane])/gradient
                        if self.strategy=='lift' and hc[i]>0:cost*=.65
                        options.append((cost,plane,hc,rc,constant))
                    if not options: continue
                    cost,plane,hc,rc,constant=min(options,key=lambda q:(q[0],q[1]))
                    constraint=dict(height_coefficients=hc.tolist(),relative_coefficients=rc.tolist(),
                        lower_bound_m=float(margins[plane]-constant),
                        label=dict(foot_task=i,demand_index=int(indices[k]),obstacle_task=j,
                                   kind=kind,face=face,plane=plane),local_required_motion_m=cost)
                    events.append(constraint)
        events.sort(key=lambda q:(-q['local_required_motion_m'],tuple(q['label'].values())))
        return events[:max_new],unsafe_count

    def solve(self, seater, *, mode='z_xy', rounds=40, max_displacement_m=.50):
        constraints=[]; seen=set(); history=[]
        seating=seater.solve(mode)
        if not seating['passed']:return seating,history
        for iteration in range(rounds):
            events,count=self.violations(seating,mode=mode)
            history.append(dict(iteration=iteration,maximum_displacement_coordinate_mm=seating['maximum_displacement_coordinate_mm'],
                                unsafe_ground_foot_cell_pairs=count,local_facet_constraints=len(constraints)))
            print('ACCESS SEATING',mode,iteration,'unsafe foot-cell pairs',count,
                  'maximum mm',round(seating['maximum_displacement_coordinate_mm'],3),flush=True)
            if count==0:
                return dict(seating,ground_feet_avoid_every_access_cell=True,
                            local_ground_access_constraints=constraints),history
            for event in events:
                key=tuple(event['label'].values())
                if key not in seen:constraints.append(event);seen.add(key)
            if not events or history[-1]['local_facet_constraints']==len(constraints):
                return dict(seating,passed=False,status='local_ground_access_stalled'),history
            seating=seater.solve(mode,access_constraints=constraints)
            if not seating['passed']:
                return dict(seating,status='local_ground_access_facet_branch_infeasible'),history
            if seating['maximum_displacement_coordinate_mm']>max_displacement_m*1000:
                return dict(seating,passed=False,status='local_ground_access_displacement_limit'),history
        return dict(seating,passed=False,status='local_ground_access_iteration_limit'),history
