"""Gradient guidance for independent thin-shell family merges."""
from collections import OrderedDict
import numpy as np
from scipy.stats import qmc
from scipy.spatial.transform import Rotation
from whole_task.operations.state import transform
from whole_task.operations.pose_change import pose_change
from whole_task.operations.geometry import moved_mesh
from whole_task.common.force_cone import tangent_frame
from whole_task.step3 import independent_geometry as G
from .sampled_guidance import SampledObjective,mesh_contains,swept_contains
from .gradient_repair import block_step


class MembraneObjective(SampledObjective):
    def evaluate(self,layout):
        if getattr(self.model,'oracle_required',False):
            from .ground_oracle import project_families
            layout=project_families(self.model,layout,self.model.oracle_partition)
            if layout is None:return None
        return super().evaluate(layout)

    def __init__(self,model,layout,samples=16,volume_power=14,envelope_layouts=None):
        self.model=model;self.owners=list(range(len(model.ids)));self.samples=samples
        self.extent=float(model.source.extents.max());self.calls=0;self.cache=OrderedDict()
        low,high=model.outer.bounds
        corners=np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
        vertices=np.vstack([transform(corners,M@P) for q in (envelope_layouts or [layout])
            for M,P in zip(q.material,q.poses)])
        self.low=vertices.min(0)-.025;self.high=vertices.max(0)+.025
        self.box_volume=float(np.prod(self.high-self.low)*1e6)
        self.volume_points=self.low+qmc.Sobol(3,scramble=True,seed=42).random_base2(volume_power)*(self.high-self.low)
        self.reference_volume=None;self.material_masks=None;self.outside_points=np.empty((0,3))
        self.world_work=G.mesh(model.world_work)
        self.parent_parts=getattr(model,'guidance_parent_parts',None)

    def parent_contains(self,points,layout):
        result=np.zeros(len(points),bool)
        for owner,old_material,mesh in self.parent_parts:
            old=transform(points,old_material@np.linalg.inv(layout.material[owner]))
            result|=mesh_contains(mesh,old)
        return result

    def data(self,layout):
        rows=[]
        for i,record in enumerate(self.model.records):
            P,M,d=layout.poses[i],layout.material[i],layout.exits[i]
            # Pose rotation about the fixed needle axis preserves the work
            # sector.  No ground or height projection changes this proposal.
            data=self.model.world_data(record,P)
            rows.append(dict(data,id=record['id'],index=i,pose=P,material=M,
                direction=d,length=4*np.linalg.norm(self.high-self.low)))
        return rows,[]

    def admissible(self,points,layout,rows):
        keep=(self.parent_contains(points,layout) if self.parent_parts is not None
              else np.ones(len(points),bool))
        if getattr(self.model,'oracle_required',False):
            normals=layout.material[:,:3,2]
            offsets=np.einsum('ij,ij->i',normals,layout.material[:,:3,3])
            keep&=(points@normals.T-offsets>=-1e-10).all(1)
        for row in rows:
            ids=np.flatnonzero(keep)
            if not len(ids):break
            world=transform(points[ids],np.linalg.inv(row['material']))
            cut=mesh_contains(self.world_work,world)
            remaining=np.flatnonzero(~cut)
            if len(remaining):
                cut[remaining]=swept_contains(row['body_world'],row['direction'],row['length'],world[remaining])
            keep[ids[cut]]=False
        return keep

    def volume(self,layout,rows,plates):
        keep=np.zeros(len(self.volume_points),bool)
        for row in rows:
            points=transform(self.volume_points,np.linalg.inv(row['material']@row['pose']))
            keep|=mesh_contains(self.model.outer,points)
        ids=np.flatnonzero(keep)
        if len(ids):keep[ids]=self.admissible(self.volume_points[ids],layout,rows)
        return float(keep.mean()*self.box_volume)


def repair(model,layout,moving,representatives,rounds=1,*,rebuild=False):
    objective=MembraneObjective(model,layout)
    if rebuild:objective.parent_parts=None
    current=objective.evaluate(layout)
    if current is None:return layout,dict(status='unresolved',blocks=[],evaluations=objective.calls)
    blocks=[]
    for iteration in range(rounds):
        changed=False
        for kind in ('exit_redirection','translation','pose_rotation'):
            base=current['layout'];owner=representatives[-1]
            if current['details']:
                worst=max(current['details'],key=lambda d:d['residual_loss'])
                owner=model.ids.index(worst['task'])
            axes=(np.eye(3) if kind=='translation' else
                np.asarray(tangent_frame(base.exits[owner])) if kind=='exit_redirection' else model.axis[None])
            def perturb(vector):
                q=base.copy()
                if kind=='translation':
                    q.material[moving,:3,3]+=objective.extent*(vector@axes)
                elif kind=='pose_rotation':
                    q=pose_change(q,owner,np.degrees(vector[0]),model.axis,model.head)
                else:
                    tangent=vector@axes
                    q.exits[owner]=Rotation.from_rotvec(np.cross(base.exits[owner],tangent)).apply(base.exits[owner])
                return q
            difference=.004 if kind=='translation' else np.radians(.5)
            step=.035 if kind=='translation' else np.radians(5.)
            current,record=block_step(objective.evaluate,current,perturb,len(axes),difference,step,backtracks=3)
            record.update(operation=kind,iteration=iteration+1);blocks.append(record)
            changed|=record['accepted']
        if not changed:break
    return current['layout'],dict(status='covered' if current['covered'] else 'unresolved',
        diagnostic_only=True,final_loss=current['loss'],evaluations=objective.calls,blocks=blocks,
        material_guidance='rebuilt_thin_layers' if rebuild else 'actual_parent_material')
