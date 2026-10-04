"""Finite, continuously checked object translation paths for fixed supports.

Object motion is +u; virtual head motion used for relative collision is -u.
The floor constrains the REAL moving object, never the virtual moving heads.
Paths are exact rays or two-segment lift/slide paths; rotation is not modeled.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from step2_local_support import withdrawal as W
from step3_scheculer.pair_geometry import horizontal_catalogue

LENGTH = .5


def catalogue(mesh):
    vectors = list(np.asarray(horizontal_catalogue(mesh)['vectors']))
    def add(d):
        d = np.asarray(d, float); d /= np.linalg.norm(d)
        if d[2] > 1e-10 or any(np.linalg.norm(d-v)<1e-7 for v in vectors):return
        vectors.append(d)
    for elevation in (-15,-30,-45,-60,-75):
        e = np.deg2rad(elevation)
        for azimuth in range(0,360,30):
            a = np.deg2rad(azimuth);add([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
    add([0,0,-1])
    for n in mesh.face_normals:
        if n[2] <= 0:add(n.copy())
    plans=[]
    for d in vectors:
        plans.append(dict(kind='ray',object_translation_waypoints_world_m=[np.zeros(3).tolist(),(-LENGTH*d).tolist()]))
    scale=float(mesh.extents.max())
    for height in (.025*scale,.10*scale,.25*scale):
        for azimuth in range(0,360,30):
            a=np.deg2rad(azimuth);end=np.array([LENGTH*np.cos(a),LENGTH*np.sin(a),height])
            plans.append(dict(kind='lift_then_slide',lift_height_m=height,object_translation_waypoints_world_m=[[0.,0.,0.],[0.,0.,height],end.tolist()]))
    for ident,p in enumerate(plans):
        nodes=np.asarray(p['object_translation_waypoints_world_m']);delta=np.diff(nodes,axis=0)
        p.update(id=ident,support_withdrawal_world=(-delta[0]/np.linalg.norm(delta[0])).tolist(),
                 initial_object_exit_world=(delta[0]/np.linalg.norm(delta[0])).tolist(),
                 terminal_object_exit_world=(delta[-1]/np.linalg.norm(delta[-1])).tolist(),
                 path_length_m=float(np.linalg.norm(delta,axis=1).sum()),rotation_allowed=False)
    return dict(options=plans,vectors=[p['support_withdrawal_world'] for p in plans],
        finite_path_set=True,complete_motion_infeasibility_claimed=False,
        frame='saved task-world; moving object, fixed support',rotation_allowed=False,
        horizontal_only=False,minimum_terminal_leg_m=LENGTH)


def local_cost(mesh,plan,margin=.05):
    """Cheap local swept AABB proxy, only for proposal ordering, not acceptance."""
    nodes=np.asarray(plan['object_translation_waypoints_world_m'])
    lo=mesh.bounds[0]-margin;hi=mesh.bounds[1]+margin
    boxes=[]
    for a,b in zip(nodes,nodes[1:]):
        low=np.maximum(lo,mesh.bounds[0]+np.minimum(a,b))
        high=np.minimum(hi,mesh.bounds[1]+np.maximum(a,b))
        boxes.append(float(np.prod(np.maximum(high-low,0))))
    return max(0.,sum(boxes)-float(np.prod(mesh.extents)))*1e6


def diverse(plans,limit):
    """Keep geometrically different CHECKED paths without any cross-pose angle rule."""
    chosen=[]
    for p in plans:
        if all(min(np.dot(p['initial_object_exit_world'],q['initial_object_exit_world']),
                   np.dot(p['terminal_object_exit_world'],q['terminal_object_exit_world'])) < np.cos(np.deg2rad(20))
               or (p['kind']!=q['kind'])
               or abs(p.get('lift_height_m',0.)-q.get('lift_height_m',0.)) > max(.0001,.3*max(p.get('lift_height_m',0.),q.get('lift_height_m',0.)))
               for q in chosen):chosen.append(p)
        if len(chosen)>=limit:break
    return chosen


def ordered_ids(mesh,plans,ids):
    """Interleave horizontal, oblique, vertical and bent alternatives."""
    buckets={name:[] for name in ('vertical','horizontal','oblique','bend')}
    for i in sorted(ids,key=lambda i:(local_cost(mesh,plans[i]),i)):
        p=plans[i];z=p['initial_object_exit_world'][2]
        name='bend' if p['kind']!='ray' else 'vertical' if z>.95 else 'horizontal' if z<.01 else 'oblique'
        buckets[name].append(int(i))
    result=[]
    for index in range(max(map(len,buckets.values()),default=0)):
        result.extend(rows[index] for rows in buckets.values() if index<len(rows))
    return result


class PathAnalyzer:
    def __init__(self,mesh,depth=.0002):
        self.mesh=mesh;self.scale=float(mesh.extents.max())
        # The virtual head translation may point below the floor. Only object
        # floor clearance is physically relevant for this relative-motion test.
        self.relative=W.Analyzer(mesh,depth,dict(vectors=[],installation=dict(floor_plane=[0.,0.,0.,0.])))

    def heads(self,contacts):
        return [h for c in contacts for h in self.relative.heads(c)]

    def test(self,cells,plan):
        nodes=np.asarray(plan['object_translation_waypoints_world_m'],float)
        if nodes.shape[1:]!=(3,) or len(nodes)<2 or not np.isfinite(nodes).all() or np.linalg.norm(nodes[0])>1e-12:
            raise ValueError('Object translation path must start at zero')
        if self.mesh.bounds[0,2]+nodes[:,2].min() < -self.scale*1e-10:
            return dict(clear=False,reason='moving_object_floor_collision')
        checks=[]
        for segment,(a,b) in enumerate(zip(nodes,nodes[1:])):
            delta=b-a;length=float(np.linalg.norm(delta))
            if length<=1e-12:raise ValueError('Zero-length exit segment')
            shifted=[SimpleNamespace(vertices=np.asarray(h.vertices)-a,metadata=dict(getattr(h,'metadata',{}))) for h in cells]
            if segment==len(nodes)-2:
                row=self.relative.test(shifted,-delta/length)
                checks.append(dict(segment=segment,**row))
                if not row['clear']:return dict(clear=False,reason=row['reason'],segments=checks)
            else:
                for ident,h in enumerate(shifted):
                    points=np.vstack([h.vertices,h.vertices-delta])
                    if self.relative.clearance.obstruction(points)<0:continue
                    swept=W.H.engine.hull_mesh(points)
                    try:
                        overlap=abs(float((self.relative.obstacle^W.solid(swept,self.relative.origin,self.scale)).volume()))*self.scale**3
                    except ValueError:
                        return dict(clear=False,reason='unresolved_segment_boolean',segment=segment)
                    if overlap>W.VOLUME_TOL*self.scale**3:
                        return dict(clear=False,reason='finite_segment_collision',segment=segment,head_cell=ident,intersection_volume_m3=overlap)
                checks.append(dict(segment=segment,clear=True,reason='continuous_finite_segment_clear',length_m=length))
        head_points=np.vstack([h.vertices for h in cells]);end=self.mesh.bounds+nodes[-1]
        separated=bool(np.any(end[0]>head_points.max(0)+1e-10) or np.any(end[1]<head_points.min(0)-1e-10))
        return dict(clear=separated,reason='continuous_object_path_clear' if separated else 'terminal_not_separated',
            object_floor_checked=True,terminal_aabb_separated=separated,segments=checks,
            virtual_head_floor_not_a_physical_constraint=True)


def sweep_mesh(mesh,plan):
    """Exact continuous per-segment prism unions, including the starting object."""
    from step4_connect_support import build_coupled_saddle as S, boxed_support as F
    nodes=np.asarray(plan['object_translation_waypoints_world_m'])
    parts=[]
    for a,b in zip(nodes,nodes[1:]):
        placed=mesh.copy();placed.apply_translation(a)
        parts.append(S.solid(S.swept_solid(placed,b-a)))
    return S.unpack(F.union(parts))


def sources():return [Path(__file__),Path(W.__file__)]
