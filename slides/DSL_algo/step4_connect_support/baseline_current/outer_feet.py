"""Sparse exterior soles and tapered co-design bodies for a fixed seating.

The ground polygon is a containment constraint. Its vertices need not each
become a separate branch: a few slightly more exterior pads can contain it.
"""
import itertools
import math
import time

import manifold3d as md
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.spatial import ConvexHull,cKDTree
from shapely.geometry import MultiPoint,Polygon
import trimesh

from step2_local_support import geometry as G
from step4_connect_support.baseline_current import build_coupled_saddle as S,convex_foot as F
from step4_connect_support.baseline_current import process_access as ACCESS


def union(parts):return md.Manifold.batch_boolean(parts,md.OpType.Add)


def component(value,required):
    for part in sorted(value.decompose(),key=lambda p:-p.volume()):
        if all(abs(float((r-part).volume()))*S.SCALE**3<8e-14 for r in required):return part
    return None


def foot_pattern(demands,count,angle,margin):
    """Circumscribed triangle/rectangle; rounded local pads at its corners."""
    theta=angle+np.arange(count)*2*np.pi/count
    normals=np.c_[np.cos(theta),np.sin(theta)]
    limits=np.max(np.asarray(demands)@normals.T,axis=0)+margin
    centers=np.asarray([np.linalg.solve(normals[[i,(i+1)%count]],limits[[i,(i+1)%count]])
        for i in range(count)])
    center=centers.mean(0);polygons=[]
    for point in centers:
        radial=point-center;radial/=np.linalg.norm(radial)
        tangent=np.array([-radial[1],radial[0]])
        # Modest filled landing: 20 x 14 mm, with 2 mm rounded corners.
        rectangle=Polygon([[-.008,-.005],[.008,-.005],[.008,.005],[-.008,.005]]).buffer(.002,quad_segs=3)
        xy=np.asarray(rectangle.exterior.coords)[:-1]@np.array([radial,tangent])+point
        polygons.append(xy)
    footprint=MultiPoint(np.vstack(polygons)).convex_hull
    required=MultiPoint(demands).convex_hull
    if required.difference(footprint.buffer(1e-10)).area>1e-12:raise ValueError('Sparse pads lose original demands')
    return polygons,dict(count=count,angle_rad=float(angle),margin_m=margin,
        required_area_m2=float(required.area),pad_hull_area_m2=float(footprint.area),
        sole_area_m2=float(sum(Polygon(p).area for p in polygons)),centers_xy_m=centers.tolist())


class Designer:
    def __init__(self,case,placement,cache):
        self.case=case;self.placement=placement
        self.bases=np.asarray(placement['bases']);self.offsets=np.asarray(placement['offsets'])
        self.directions=np.asarray(placement['directions'])
        # Reject impossible exact contacts before reusing any old geometry cache.
        self.access=ACCESS.Guard(case,placement)
        with np.load(cache/'forbidden.npz') as z:
            self.forbidden=S.solid(trimesh.Trimesh(z['v'],z['f'],process=False))
        self.sweeps=[]
        for k in range(len(case.poses)):
            with np.load(cache/f'sweep{k}.npz') as z:
                self.sweeps.append(trimesh.Trimesh(z['v'],z['f'],process=False))
        self.patches=[]
        for k,(row,contacts) in enumerate(zip(case.support_seeds,case.groups)):
            for cells,contact in zip(row,contacts):
                world=[p@self.bases[k]+self.offsets[k] for p in cells]
                points=np.vstack(world);points=points[ConvexHull(points).vertices]
                self.patches.append(dict(pose=k,id=contact['candidate_id'],v=points,
                    root=points+.008*self.directions[k]@self.bases[k],
                    original=union([S.solid(G.hull_mesh(p)) for p in world])))

    def carve(self,value):
        for b,o in zip(self.bases,self.offsets):
            value=value.trim_by_plane(b[2].tolist(),float(b[2]@o/S.SCALE))
        # Padded object sweeps exclude work faces from all new material.
        # Re-added roots are checked separately and in the final union.
        return value-self.forbidden

    def menu(self,k):
        trials=[]
        # Search a small declared set, without prescribing a head/terminal fan.
        for count,angle,margin in itertools.product((3,4),np.deg2rad([0,22.5,45,67.5]),(.006,.018)):
            pads,record=foot_pattern(self.case.demands[k],count,angle,margin)
            feet=[];edges=[]
            for j,xy in enumerate(pads):
                vertices=np.vstack([np.c_[xy,np.full(len(xy),z)] for z in (0,.003)])@self.bases[k]+self.offsets[k]
                pad=S.solid(G.hull_mesh(vertices))
                missing=abs(float((pad-self.carve(pad)).volume()))*S.SCALE**3
                if missing>8e-14:break
                feet.append(xy)
                for h,patch in enumerate(self.patches):
                    body=F.loft(patch,xy,self.bases[k],self.offsets[k],self.carve,component)
                    if body is None:continue
                    mesh=S.unpack(body)
                    # Avoid a nominal broad blank shaved into wisps by a sweep.
                    blank=G.hull_mesh(np.vstack([patch['v'],patch['root'],vertices]))
                    retention=float(mesh.volume/blank.volume)
                    if retention<.52:continue
                    length=float(np.linalg.norm(vertices.mean(0)-patch['v'].mean(0)))
                    volume=float(mesh.volume*1e6)
                    cost=25+volume+length*120+10*(1-retention)
                    edges.append(dict(head=h,pad=j,floor=k,body=body,cost=cost,
                        volume_cm3=volume,length_m=length,retained_blank_fraction=retention))
            if len(feet)!=len(pads) or any(not any(e['pad']==j for e in edges) for j in range(count)):continue
            estimate=sum(min(e['cost'] for e in edges if e['pad']==j) for j in range(count))
            trials.append(dict(floor=k,pads=feet,record=record,edges=edges,estimate=estimate))
            print('OUTER FOOT MENU',self.case.poses[k],count,round(float(np.rad2deg(angle))),margin,
                len(edges),round(estimate,2),flush=True)
        return sorted(trials,key=lambda r:r['estimate'])

    def assignments(self,menus):
        results=[];n=len(self.patches)
        for plans in itertools.product(*menus):
            edges=[e for p in plans for e in p['edges']]
            constraints=[];low=[];high=[]
            for h in range(n):
                constraints.append([e['head']==h for e in edges]);low.append(1);high.append(2)
                for k in range(len(plans)):
                    constraints.append([e['head']==h and e['floor']==k for e in edges]);low.append(0);high.append(1)
            for plan in plans:
                for j in range(len(plan['pads'])):
                    constraints.append([e['floor']==plan['floor'] and e['pad']==j for e in edges]);low.append(1);high.append(2)
            result=milp(np.asarray([e['cost'] for e in edges]),integrality=np.ones(len(edges)),
                bounds=Bounds(0,1),constraints=LinearConstraint(np.asarray(constraints,float),low,high),
                options=dict(time_limit=3.))
            if result.x is None:continue
            chosen=[e for e,x in zip(edges,result.x) if x>.5]
            results.append(dict(plans=plans,edges=chosen,cost=float(result.fun)))
        return sorted(results,key=lambda r:r['cost'])

    def join(self,selection):
        bodies=[union([e['body'] for e in selection['edges'] if e['head']==h]) for h in range(len(self.patches))]
        full=union(bodies);bridges=[];records=[]
        bead=trimesh.creation.icosphere(subdivisions=2,radius=.004).vertices
        up=np.sum(self.bases[:,2],axis=0)
        up=up/np.linalg.norm(up) if np.linalg.norm(up)>1e-9 else self.bases[0][0]
        for _ in range(len(bodies)):
            components=full.decompose()
            if len(components)==1:break
            points=[]
            for part in components:
                v=S.unpack(part).vertices
                legal=np.all([((v-o)@b[2])>.005 for b,o in zip(self.bases,self.offsets)],axis=0)
                points.append(v[legal])
            trials=[]
            for i,a in enumerate(points):
                for j,b in enumerate(points[:i]):
                    if not len(a) or not len(b):continue
                    distance,near=cKDTree(b).query(a);selected=[]
                    for q in np.argsort(distance):
                        start,end=a[q],b[near[q]]
                        if any(np.linalg.norm(start-x)+np.linalg.norm(end-y)<.008 for x,y in selected):continue
                        selected.append((start,end));trials.append((float(distance[q]),i,j,start,end))
                        if len(selected)>=12:break
            winner=None
            for lift in (0.,.012,.025):
                for length,i,j,a,b in sorted(trials,key=lambda r:r[0]):
                    if length>.15:continue
                    route=[a,b] if lift==0 else [a,(a+b)/2+lift*up,b]
                    blank=union([S.solid(G.hull_mesh(np.vstack([x+bead,y+bead]))) for x,y in zip(route[:-1],route[1:])])
                    bridge=self.carve(blank)
                    if bridge.volume()<.70*blank.volume():continue
                    if any(abs(float((bridge^components[c]).volume()))*S.SCALE**3<1e-12 for c in (i,j)):continue
                    joined=full+bridge
                    if len(joined.decompose())>=len(components):continue
                    winner=(joined,bridge,dict(path_m=np.asarray(route).tolist(),radius_m=.004,
                        added_volume_cm3=float((joined.volume()-full.volume())*S.SCALE**3*1e6)))
                    break
                if winner is not None:break
            if winner is None:raise RuntimeError('No clean short co-design joint for this sparse foot assignment')
            full,bridge,record=winner;bridges.append(bridge);records.append(record)
        mesh=S.unpack(full)
        if len(full.decompose())!=1 or not mesh.is_watertight or not mesh.is_winding_consistent:
            raise RuntimeError('Sparse local bodies did not form one closed connected solid')
        for k,plan in enumerate(selection['plans']):
            hull=F.landing(full,self.bases[k],self.offsets[k]).convex_hull
            if MultiPoint(self.case.demands[k]).convex_hull.difference(hull.buffer(1e-10)).area>1e-12:
                raise RuntimeError('Actual ground hull lost original demands')
        self.access_check=self.access.verify(mesh)
        if not self.access_check['passed']:
            raise ACCESS.AccessRejected(dict(self.access_check,status='final_solid_touches_working_surface'))
        return mesh,bodies,bridges,records
