"""Axis-neighbor routing and coverage-preserving alternative ground ends."""
import numpy as np
import manifold3d as md
from shapely.geometry import MultiPoint,Point
from continuous_legs import RouteGuide as BaseGuide
from continuous_legs_recovery import Builder as BaseBuilder
from deterministic_base import unpack,volume,transform,SCALE


class RouteGuide(BaseGuide):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Diagonal voxel adjacency can cut through an excluded corner. A raw
        # neighbor then has no valid sampled shortcut, stalling old reduction.
        # Axis neighbors always give the reduction a valid next edge.
        self.directions=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
        self.lengths=np.full(6,self.pitch)


class Builder(BaseBuilder):
    def alternatives(self,target):
        if target['kind']!='ground_terminal':return super().alternatives(target)
        row=next(r for r in self.rows if r['pose']==target['pose'])
        T=row['fixture_transform'];inverse=np.linalg.inv(T)
        point=transform(target['point'][None,:],T)[0]
        candidates=[target['point']]
        for radius in [.002,.005,.010,.018,.030]:
            for angle in np.linspace(0,2*np.pi,16,endpoint=False):
                world=point+np.array([radius*np.cos(angle),radius*np.sin(angle),0.])
                world[2]=0.
                candidates.append(transform(world[None,:],inverse)[0])
        domain=unpack(self.domain)
        world=transform(domain.vertices,T)
        ground_ids=np.flatnonzero(abs(world[:,2])<=1e-8)
        if len(ground_ids):
            distance=np.linalg.norm(world[ground_ids,:2]-point[:2],axis=1)
            candidates.extend(domain.vertices[ground_ids[np.argsort(distance)[:100]]])
        points=np.unique(np.round(candidates,13),axis=0)
        core=unpack(self.body)
        relevant=[r for r in self.rows if np.allclose(T,r['fixture_transform'],atol=1e-11,rtol=0)]
        current=transform(core.vertices,T)
        existing=current[abs(current[:,2])<=1e-8,:2]
        planned=[]
        for other in self.targets:
            if other is target:continue
            q=transform(other['point'][None,:],T)[0]
            if abs(q[2])<=1e-8:planned.append(q[:2])
        planned=np.asarray(planned).reshape(-1,2)
        scores=[]
        for candidate in points:
            xy=transform(candidate[None,:],T)[0,:2]
            valid=True
            for task in relevant:
                cloud=np.vstack([existing,planned,task['object_ground'],xy[None,:]])
                hull=MultiPoint(cloud).convex_hull
                if hull.is_empty or any(hull.distance(Point(p[:2]))>1e-8 for p in task['hull']):valid=False;break
            if not valid:continue
            probe=md.Manifold.sphere(.003/SCALE,16).translate(tuple(candidate/SCALE))
            amount=volume(probe^self.domain)
            if amount<=1e-13:continue
            scores.append((float(np.linalg.norm(candidate-target['point'])),amount,candidate))
        return [p for _,_,p in sorted(scores,key=lambda r:(-r[1],r[0]))[:32]]
