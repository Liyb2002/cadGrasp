"""Floor-feasible SO(3) placements of one certified family against another.

The population seeds use seat alignments or the identity rotation.  For the
last 9+4 merge all 72 such seeds either fail the floor LP or are clipped far
below their contacts.  Here a random rigid rotation of the guest family is
first made floor-feasible by an LP over every pressure hull, then translated
to touch the host.  Each placement is scored on the ACTUAL parent-union
construction of membrane_search.parent_union_candidate and clip_core (about
one second with cached sweeps): its exact volume and largest component, and
a force screen using only the parents' contact rays that remain inside that
component.  Added contacts can only help, so the screen is conservative.
Acceptance is still the unchanged complete audit in endgame.py.
"""
import os
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from whole_task_2.operations.state import Layout,transform
from whole_task_2.operations.geometry import contact_surface,moved_mesh
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.independent_physics import contacts
from whole_task_2.step3.legacy import kernel as K
from whole_task_2.step1.common import sha256
from .sampled_guidance import mesh_contains

FLOOR_MARGIN_M=2e-4


def parent_data(model,group,read_group):
    """Pressure hulls, floors and actual contact rays of one certified fixture."""
    q,support=read_group(group);part=model.subset(group['members']);tasks=part.prepare(q)
    pressure=[];rays=[]
    for t,M in zip(tasks,q.material):
        body=t['body_world'];world=moved_mesh(support,np.linalg.inv(M))
        triangles,sources,_,scale=contacts(body,contact_surface(body,world),t['domain'].work_ids,t['domain'].com)
        points=triangles.reshape(-1,3);normals=np.repeat(body.face_normals[sources],3,axis=0)
        columns=K.U.heads(np.c_[-normals,np.cross(points-t['domain'].com,-normals)],scale)
        # Probe 0.3 mm inside the support, behind each contact vertex.
        rays.append(dict(id=t['id'],points=transform(points+3e-4*normals,M),columns=columns,
            floor=K.U.floor(np.empty((0,6)),scale),outer=t['endpoints']['endpoint_need_wrench']*scale,
            samples=t['samples']['need_wrench']*scale))
        p=G.pressure_points(t['endpoints']['endpoint_need_wrench'],t['domain'].com);hull=ConvexHull(p).vertices
        pressure.append(transform(np.c_[p[hull],np.zeros(len(hull))],M))
    normals=q.material[:,:3,2]
    return dict(group=group,layout=q,mesh=support,pressure=pressure,rays=rays,
        volume_cm3=float(support.volume*1e6),floor_normals=normals,
        floor_offsets=np.einsum('ij,ij->i',normals,q.material[:,:3,3]))


def cached_parent_data(model,group,read_group):
    """parent_data of an immutable certified fixture, cached beside its file."""
    path=Path(group['file']);cache=path.with_name(path.stem+'_placement.pkl')
    binding=[group['sha256'],sha256(Path(__file__))]
    if cache.exists():
        try:
            saved=pickle.loads(cache.read_bytes())
            if saved['binding']==binding:return saved['data']
        except (OSError,EOFError,pickle.UnpicklingError,KeyError):pass
    data=parent_data(model,group,read_group)
    temporary=cache.with_name(cache.name+f'.{os.getpid()}.tmp')
    temporary.write_bytes(pickle.dumps(dict(binding=binding,data=data)));temporary.replace(cache)
    return data


class IncrementalUnion:
    """The parent-union construction, recomputing only what a placement changes.

    Each certified parent already excludes its own work sectors and sweeps, so
    (A u B) - Z_A - Z_B = (A - Z_B) u (B - Z_A). Z_A, Z_B, A and B are unioned
    once; a placement C only moves B and Z_B rigidly before three Booleans,
    the floor clip and the largest-component rule. Screening only: accepted
    candidates are rebuilt by membrane_search.parent_union_candidate.
    """
    def __init__(self,model,host,guest):
        from whole_task_2.operations.floors import planes,clip,connected_material
        self.planes,self.clip,self.connected=planes,clip,connected_material
        na=len(host['layout'].poses);n=len(model.ids)
        def cuts(part,layout):
            return G.union([x for t in part.prepare(layout) for x in (t['sweep'],t['work'])])
        self.Za=cuts(model.subset(list(range(na))),host['layout'])
        self.Zb=cuts(model.subset(list(range(na,n))),guest['layout'])
        self.A=G.solid(host['mesh']);self.B=G.solid(guest['mesh'])

    def __call__(self,model,layout,C):
        T=np.asarray(C,float)[:3].copy();T[:,3]/=G.SCALE
        value=(self.A-self.Zb.transform(T))+(self.B.transform(T)-self.Za)
        value=self.connected(self.clip(value,self.planes(layout.material)))
        return G.mesh(value.simplify(0.))


def merged_layout(host,guest,C):
    a,b=host['layout'],guest['layout']
    q=Layout(np.concatenate([a.poses,b.poses]),np.concatenate([a.material,b.material]),np.concatenate([a.exits,b.exits]))
    q.material[len(a.poses):]=C@b.material
    return q


def floor_translation(host,guest,R,target=None,margin=FLOOR_MARGIN_M):
    """Guest translation keeping every pressure hull above every other floor.

    The necessary ground condition of the oracle, with spatial_search.floor_repair's
    margins: none against an exactly parallel (shared) floor, otherwise 0.2 mm.
    """
    guest_normals=guest['floor_normals']@R.T;rows=[];bound=[]
    def gap(n,m):return 0. if n@m>1.-1e-9 else margin
    for n,o in zip(host['floor_normals'],host['floor_offsets']):
        for p,m in zip(guest['pressure'],guest_normals):
            rows.append(-n);bound.append((p@R.T@n).min()-o-gap(n,m))
    for n,o in zip(guest_normals,guest['floor_offsets']):
        for p,m in zip(host['pressure'],host['floor_normals']):
            rows.append(n);bound.append((p@n).min()-o-gap(n,m))
    rows=np.asarray(rows);bound=np.asarray(bound)
    try:
        if target is None:
            result=linprog(np.zeros(3),A_ub=rows,b_ub=bound,bounds=[(-3,3)]*3,method='highs')
            return result.x if result.success else None
        I=np.eye(3);A=np.vstack([np.c_[rows,np.zeros((len(rows),3))],np.c_[I,-I],np.c_[-I,-I]])
        result=linprog(np.r_[np.zeros(3),np.ones(3)],A_ub=A,b_ub=np.r_[bound,target,-target],
            bounds=[(-3,3)]*3+[(0,None)]*3,method='highs')
    except ValueError:return None  # an unrecognized solver status counts as infeasible
    return result.x[:3] if result.success else None


def score(row,limit=np.inf):
    """Fewer screened force failures, then within the merge gate, then volume."""
    if not row.get('ok'):return (99,1,np.inf)
    return (len(row['proxy'])-sum(row['proxy']),int(row['volume_cm3']>limit),row['volume_cm3'])


class PlacementScreen:
    def __init__(self,model,host,guest,construct,limit=np.inf,proxy_targets=16,exact_fallback=True):
        self.model=model;self.host=host;self.guest=guest;self.construct=construct
        self.na=len(host['layout'].poses);self.limit=limit;self.proxy_targets=proxy_targets
        self.exact_fallback=exact_fallback
        # Shape-relative steps: overlap two shell thicknesses, move 5% of the object.
        self.overlap=2*float(model.policy['mold_outward_displacement_m'])
        self.step=.05*float(model.source.extents.max())

    def actual_columns(self,k,layout,support):
        """Contact rays of the actual constructed support, as the certificate extracts them."""
        data=self.model.world_data(self.model.records[k],layout.poses[k]);body=data['body_world']
        world=moved_mesh(support,np.linalg.inv(layout.material[k]))
        try:return contacts(body,contact_surface(body,world),data['domain'].work_ids,data['domain'].com)[2]
        except RuntimeError:return None

    def feasible(self,columns,ray):
        ids=np.unique(np.linspace(0,len(ray['outer'])-1,self.proxy_targets).astype(int))
        samples=ray['samples'][::max(1,len(ray['samples'])*4//self.proxy_targets)]
        return all(K.J.C.W.solve(columns,K.J.C.U.target(x[None],columns.shape[1])[0]) is not None
                   for x in np.vstack([ray['outer'][ids],samples]))

    def placement_of(self,layout):
        """Every guest installation of a seed carries the same rigid transform."""
        C=layout.material[self.na]@np.linalg.inv(self.guest['layout'].material[0])
        if not np.allclose(layout.material[self.na:],C@self.guest['layout'].material,atol=1e-9,rtol=0):
            raise ValueError('Seed does not move the guest family rigidly')
        return C

    def propose(self,rng,kind,seeds=(),rotations=12,directions=6,centers=6,refine_starts=2,
                iterations=6,offspring=6,deadline=np.inf,enough=12):
        """Rank population seeds and new floor-feasible placements on the actual construction.

        Juxtaposition keeps the guest orientation; rotational reuse samples
        SO(3). The best rows are refined by (1+lambda) search on the same score.
        """
        rows=[]
        for layout in seeds:
            if time.monotonic()>deadline:break
            rows.append(dict(self.evaluate(self.placement_of(layout)),source='population_seed'))
        Rs=([np.eye(3)] if kind=='juxtaposition' else
            Rotation.random(rotations,random_state=int(rng.integers(2**31))).as_matrix())
        good=lambda:sum(1 for r in rows if r.get('ok') and all(r['proxy']) and r['volume_cm3']<=self.limit)
        for R in Rs:
            if time.monotonic()>deadline or good()>=enough:break
            if floor_translation(self.host,self.guest,R) is None:continue
            for target in self.touching(R,rng,directions*(4 if kind=='juxtaposition' else 1),centers):
                if time.monotonic()>deadline:break
                t=floor_translation(self.host,self.guest,R,target)
                if t is None:continue
                C=np.eye(4);C[:3,:3]=R;C[:3,3]=t
                rows.append(dict(self.evaluate(C),source='so3_floor_feasible' if kind!='juxtaposition' else 'floor_feasible_translation'))
        rows.sort(key=lambda r:score(r,self.limit))
        for k,row in enumerate([r for r in rows if r.get('ok')][:refine_starts]):
            if time.monotonic()>deadline:break
            better=self.refine(row,int(rng.integers(2**31)),iterations,offspring,
                degrees=0. if kind=='juxtaposition' else 4.,deadline=deadline)
            if score(better,self.limit)<score(row,self.limit):rows.append(dict(better,source=row.get('source','')+'+refined'))
        rows.sort(key=lambda r:score(r,self.limit))
        return rows

    def evaluate(self,C):
        A,B=self.host,self.guest;q=merged_layout(A,B,C)
        try:support=self.construct(self.model,q,C) if isinstance(self.construct,IncrementalUnion) else self.construct(self.model,q)
        except (ValueError,RuntimeError) as error:return dict(C=C,ok=False,reason=str(error)[:120])
        proxy=[];lost=[];exact=0;k=0
        for family,T in ((A,np.eye(4)),(B,C)):
            for ray in family['rays']:
                kept=mesh_contains(support,transform(ray['points'],T));lost.append(float(1-kept.mean()))
                good=bool(kept.any()) and self.feasible(K.I.merge_columns(ray['floor'],ray['columns'][kept]),ray)
                if not good and self.exact_fallback:
                    # Retained parent rays miss contact polygons clipped by a cut
                    # and new contacts; only a failed task gets the exact rays.
                    columns=self.actual_columns(k,q,support);exact+=1
                    good=columns is not None and self.feasible(columns,ray)
                proxy.append(good);k+=1
        return dict(C=C,ok=True,volume_cm3=float(support.volume*1e6),proxy=proxy,lost=lost,exact_rechecks=exact)

    def touching(self,R,rng,directions=6,centers=6):
        """Translations meeting the host from random sides or inside its material."""
        guest=self.guest['mesh'].vertices@R.T;host=self.host['mesh'].vertices;targets=[]
        for u in rng.normal(size=(directions,3)):
            u/=np.linalg.norm(u)
            targets.append(host[np.argmax(host@u)]-guest[np.argmin(guest@u)]-self.overlap*u)
        for c in host[rng.choice(len(host),centers)]:targets.append(c-guest.mean(0))
        return targets

    def screen(self,seed,rotations=16,deadline=np.inf):
        """Global sampling; deadline is wall-clock time.time() shared by workers."""
        rng=np.random.default_rng(seed);rows=[]
        for R in Rotation.random(rotations,random_state=seed+10000).as_matrix():
            if time.time()>deadline:break
            if floor_translation(self.host,self.guest,R) is None:continue
            for target in self.touching(R,rng):
                if time.time()>deadline:break
                t=floor_translation(self.host,self.guest,R,target)
                if t is None:continue
                C=np.eye(4);C[:3,:3]=R;C[:3,3]=t;rows.append(self.evaluate(C))
        return rows

    def refine(self,start,seed,iterations=24,offspring=8,degrees=4.,deadline=np.inf):
        """(1+lambda) search on the actual construction around one placement.

        Same score as the screen: fewer failed tasks, then the merge gate,
        then volume. Steps are 4 degrees and 5% of the object size.
        """
        rng=np.random.default_rng(seed);best=start;trace=[];metres=self.step
        def key(row):return score(row,self.limit)
        for _ in range(iterations):
            trials=[]
            for _ in range(offspring):
                if time.monotonic()>deadline:break
                R=Rotation.from_rotvec(rng.normal(size=3)*np.radians(degrees)*rng.uniform(.3,1.)).as_matrix()@best['C'][:3,:3]
                t=floor_translation(self.host,self.guest,R,best['C'][:3,3]+rng.normal(size=3)*metres)
                if t is None:continue
                C=np.eye(4);C[:3,:3]=R;C[:3,3]=t;trials.append(self.evaluate(C))
            if trials:
                candidate=min(trials,key=key)
                if key(candidate)<key(best):best=candidate
            trace.append(list(key(best)))
        return dict(best,refinement_trace=trace)
