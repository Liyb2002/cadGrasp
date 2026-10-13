"""Bounded solid-leg recovery: more roots and legal extremum alternatives."""
import numpy as np
import manifold3d as md
from scipy.spatial import cKDTree
from continuous_legs import Builder as OriginalBuilder, capsule, no_tunnels, RouteGuide
from deterministic_base import unpack, volume, transform, union, SCALE
from prune_support import material_components


class Builder(OriginalBuilder):
    def alternatives(self,target):
        candidates=[target['point']]
        if target['kind']=='bbx_extremum':
            for row in self.rows:
                world=transform(target['point'][None,:],row['fixture_transform'])[0]
                old=transform(self.old_vertices,row['fixture_transform'])
                for axis in range(3):
                    for side in [0,1]:
                        edge=self.bounds[side,axis]
                        if abs(world[axis]-edge)>1e-10:continue
                        ids=np.flatnonzero(abs(old[:,axis]-edge)<=1e-10)
                        if len(ids):candidates.extend(self.old_vertices[ids[::max(1,len(ids)//60)]])
        points=np.unique(np.round(candidates,13),axis=0)
        roots=unpack(self.body).vertices
        distances=cKDTree(roots).query(points)[0]
        scores=[]
        for i in np.argsort(distances)[:32]:
            probe=md.Manifold.sphere(.003/SCALE,16).translate(tuple(points[i]/SCALE))
            scores.append((volume(probe^self.domain),float(distances[i]),points[i]))
        return [p for _,_,p in sorted(scores,key=lambda r:(-r[0],r[1]))]

    def attach(self,target):
        try:
            return super().attach(target)
        except ValueError as initial:
            initial_error=str(initial)
        current=unpack(self.body)
        tree=cKDTree(current.vertices)
        for point in self.alternatives(target):
            distances,ids=tree.query(point,k=min(len(current.vertices),3000))
            roots=[]
            for index in np.atleast_1d(ids):
                root=current.vertices[index]
                if not roots or min(np.linalg.norm(root-r)for r in roots)>.004:
                    roots.append(root)
                if len(roots)>=32:break
            for radius in [.004,.005,.006,.003,.0025]:
                for root in roots:
                    proposed=capsule(point,root,radius,root_radius=radius*1.3)^self.domain
                    for piece in material_components(proposed):
                        probe=md.Manifold.sphere(.0002/SCALE,12).translate(tuple(point/SCALE))
                        if volume(piece^probe)<=1e-14 or volume(piece^self.body)<=1e-13 or not no_tunnels(piece):continue
                        candidate=self.body+piece
                        if len(material_components(candidate))!=1 or volume(candidate)>volume(self.previous)+1e-10:continue
                        added=volume(candidate)-volume(self.body)
                        self.body=candidate
                        self.history.append(dict(operation='solid_rounded_leg',kind=target['kind'],
                            tip_m=point.tolist(),root_m=root.tolist(),radius_m=radius,root_radius_m=radius*1.3,
                            added_cm3=added*1e6,leg_tunnels=0,recovery=True,
                            original_target_m=target['point'].tolist(),initial_rejection=initial_error))
                        print('RECOVERED SOLID LEG',len(self.history),radius,flush=True)
                        return
            if self.guide is None or self.guide.pitch>.00105:
                self.guide=RouteGuide(self.domain,self.core,pitch=.001)
            try:path,description=self.guide.route(point,limit=600000)
            except ValueError:continue
            for radius in [.003,.0025,.004,.005]:
                proposed=union([capsule(a,b,radius)for a,b in zip(path[:-1],path[1:])])^self.domain
                for piece in material_components(proposed):
                    probe=md.Manifold.sphere(.0002/SCALE,12).translate(tuple(point/SCALE))
                    if volume(piece^probe)<=1e-14 or volume(piece^self.body)<=1e-13 or not no_tunnels(piece):continue
                    candidate=self.body+piece
                    if len(material_components(candidate))!=1 or volume(candidate)>volume(self.previous)+1e-10:continue
                    added=volume(candidate)-volume(self.body)
                    self.body=candidate
                    self.history.append(dict(operation='routed_solid_rounded_leg',kind=target['kind'],
                        tip_m=point.tolist(),path_m=path.tolist(),radius_m=radius,added_cm3=added*1e6,
                        leg_tunnels=0,recovery=True,initial_rejection=initial_error,**description))
                    print('RECOVERED ROUTED LEG',len(self.history),radius,flush=True)
                    return
        raise ValueError('Expanded solid-leg recovery failed: '+str(target)+'; '+initial_error)
