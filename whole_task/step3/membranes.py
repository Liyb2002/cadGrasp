"""Independent finite-thickness contact membranes, without a floor model.

Each support is rigidly held in its current installation.  Object gravity,
paired processing loads, tool clearance and the complete nonconvex exit are
still checked.  A floor, a common height and connections between independent
fixtures are deliberately absent from this design problem.
"""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import trimesh

from whole_task.step1.common import ROOT
from whole_task.step1.needle_geometry import geometric_mask
from whole_task.operations.geometry import OperationModel, moved_mesh, contact_surface
from whole_task.operations.state import Layout, transform
from whole_task.operations.sweep_cache import sweep_mesh, implementation_key
from . import independent_geometry as G
from .exit_geometry import whole_object_sweep
from .independent_physics import certificate, demands
from .run_independent import make_domain


def positive_components(value):
    return [p for p in value.decompose() if float(p.volume())*G.SCALE**3 > 1e-15]


class MembraneModel(OperationModel):
    def __init__(self, source, outer, records, workspace, physical, policy, name='B'):
        self.source=source; self.outer=outer; self.records=deepcopy(records)
        self.ids=[r['id'] for r in records]; self.workspace=deepcopy(workspace)
        self.head=np.asarray(workspace['reference_head_front_world_m'],float)
        self.axis=np.asarray(workspace['nominal_axis_world'],float)
        self.physical=physical; self.policy=deepcopy(policy); self.name=name
        self.tolerance=policy['boolean_volume_tolerance_m3']
        self.use_floor_constraints=False
        self.pose_cache={}; self.geometry_cache={}; self.path_cache={}
        self.world_sweep_cache={}; self.evaluations=0; self.inputs=[]
        self.world_work=G.work_sector(self.workspace,self.policy)[0]
        self.sweep_cache_implementation=implementation_key()

    def world_data(self,record,pose):
        self.check_deadline()
        key=(record['id'],pose.tobytes())
        if key not in self.pose_cache:
            domain=make_domain(self.source,pose,np.asarray(record['working_face_ids']),
                record['id'],self.workspace,self.physical,self.policy)
            endpoints,samples=demands(domain,self.policy)
            self.pose_cache[key]=dict(body_world=moved_mesh(self.source,pose),domain=domain,
                endpoints=endpoints,samples=samples,stock_world=moved_mesh(self.outer,pose))
            if len(self.pose_cache)>128:self.pose_cache.pop(next(iter(self.pose_cache)))
        return self.pose_cache[key]

    def prepare(self,layout):
        rows=[]
        for i,record in enumerate(self.records):
            P,M,d=layout.poses[i],layout.material[i],layout.exits[i]
            for matrix in (P,M):
                if (not np.isfinite(matrix).all() or
                    not np.allclose(matrix[3],[0,0,0,1],atol=1e-12,rtol=0) or
                    not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-12,rtol=0) or
                    abs(np.linalg.det(matrix[:3,:3])-1)>1e-12):
                    raise ValueError('Expected a proper rigid pose/installation')
            if not np.isfinite(d).all() or abs(np.linalg.norm(d)-1)>1e-12 or d[2]<.05:
                raise ValueError('Invalid upper-hemisphere exit')
            ids=np.asarray(record['working_face_ids'])
            if not geometric_mask(self.source,transform(self.head,np.linalg.inv(P)),
                    P[:3,:3].T@self.axis,self.workspace)[ids].all():
                raise ValueError(record['id']+': original working area outside needle sector')
            if self.source.contains(transform(self.head,np.linalg.inv(P))[None])[0]:
                raise ValueError(record['id']+': working head is inside the object')
            data=self.world_data(record,P)
            key=(record['id'],P.tobytes(),M.tobytes(),d.tobytes())
            if key not in self.geometry_cache:
                # The construction travel is deliberately longer than any
                # local membrane; the final distance is derived from S.
                length=float(np.ceil(4*np.linalg.norm(self.source.extents)/.1)*.1)
                sweep=sweep_mesh(ROOT/'whole_task/data'/self.name/'step3_membrane_sweeps',
                    data['body_world'],d,length,
                    lambda:G.mesh(whole_object_sweep(data['body_world'],d,length)[0]),
                    self.sweep_cache_implementation)
                self.geometry_cache[key]=dict(body=moved_mesh(data['body_world'],M),
                    stock=G.solid(moved_mesh(data['stock_world'],M)),
                    work=G.solid(moved_mesh(G.mesh(self.world_work),M)),
                    sweep=G.solid(moved_mesh(sweep,M)),direction=M[:3,:3]@d,
                    length=length,material=M,pose=P,index=i,id=record['id'],working_face_ids=ids)
                if len(self.geometry_cache)>128:self.geometry_cache.pop(next(iter(self.geometry_cache)))
            rows.append(dict(data,**self.geometry_cache[key]))
        return rows

    def build(self,layout,*,facewise=False):
        began=time.monotonic(); tasks=self.prepare(layout)
        stock=G.union([t['stock'] for t in tasks])
        forbidden=[t[k] for t in tasks for k in ('sweep','work')]
        value=stock-G.union(forbidden)
        pieces=positive_components(value)
        if not pieces:raise ValueError('No membrane material remains')
        # Removing a disconnected fragment is only a construction proposal.
        # Every task is subsequently tested on the saved remaining solid.
        if len(pieces)>1:
            keep=max(pieces,key=lambda p:float(p.volume()))
            for p in pieces:
                if p is not keep:value=value-p
        value=value.simplify(0.)
        support=G.mesh(value)
        # Serialization can expose coincident Boolean slivers.  Recut the
        # same exclusions without changing the physical tolerance.
        actual_forbidden=[p for t in tasks for p in (G.solid(t['body']),t['work'])]
        for _ in range(3):
            native=G.solid(support)
            if max(G.volume(native^f) for f in actual_forbidden)<=self.tolerance:break
            support=G.mesh((native-G.union(actual_forbidden)).simplify(0.))
        candidate=dict(layout=layout.copy(),support=support,tasks=tasks,
            volume_cm3=float(support.volume*1e6),build_seconds=time.monotonic()-began,
            construction=dict(model='union_of_5mm_outer_layers_minus_all_task_cutouts',
                base_plate=False,vertical_backing=False,floor_constraints=False))
        if facewise:self.refine_exits(candidate)
        return candidate

    def exit_paths(self,candidate):
        support=candidate['support']
        for t in candidate['tasks']:
            world=moved_mesh(support,np.linalg.inv(t['material']))
            d=candidate['layout'].exits[t['index']]
            t['length']=max(.01,float((world.bounds[1,2]+.015-t['body_world'].bounds[0,2])/d[2]))
        return [(t['id'],t['body'],t['direction'],t['length']) for t in candidate['tasks']]

    def refine_exits(self,candidate):
        """Recut original face cells at the final travel, retaining no floor."""
        for turn in range(4):
            paths=self.exit_paths(candidate);value=G.solid(candidate['support'])
            rows,bad=self.path_checks(value,paths)
            if all(r['passed'] for r in rows):break
            if turn==3 or not bad:raise ValueError('Unresolved continuous exit '+str(rows))
            for _,_,prism in bad:value=value-prism
            parts=positive_components(value)
            if not parts:raise ValueError('Exit refinement removes all support')
            keep=max(parts,key=lambda p:float(p.volume()))
            for part in parts:
                if part is not keep:value=value-part
            candidate['support']=G.mesh(value.simplify(0.))
        candidate['volume_cm3']=float(candidate['support'].volume*1e6)
        candidate['construction']['exit_refinement_rounds']=turn

    def audit(self,candidate,*,full=True,continuous=True,all_outer=False):
        began=time.monotonic(); self.check_deadline(); self.evaluations+=1
        support=candidate['support']; solid=G.solid(support)
        if (not support.is_watertight or not support.is_winding_consistent or support.volume<=0 or
                self.material_component_count(support,solid)!=1):
            raise ValueError('Each fixture must be one closed connected material body')
        # Derive a finite exit long enough to clear the complete fixture.
        paths=self.exit_paths(candidate)
        checks=self.path_checks(solid,paths)[0] if continuous else []
        if not all(r['passed'] for r in checks):raise ValueError('Continuous source-face exit failed')
        records=[]
        from whole_task.operations.contact_cache import cached_certificate_contacts
        cache=getattr(self,'actual_contact_cache',None)
        if cache is None:self.actual_contact_cache=cache={}
        for t in candidate['tasks']:
            overlap={k:G.volume(solid^(G.solid(t['body']) if k=='body' else t['work'])) for k in ('body','work')}
            if max(overlap.values())>self.tolerance:raise ValueError(t['id']+': body/work interference')
            world=moved_mesh(support,np.linalg.inv(t['material']))
            with cached_certificate_contacts(cache):
                proof,data=certificate(t['body_world'],contact_surface(t['body_world'],world),
                    t['domain'],t['endpoints'],t['samples'],search=True)
            if not proof['passed']:
                if 'unpassed_outer_index' in proof:
                    self.note_repair_targets(t['id'],'endpoint',[proof['unpassed_outer_index']])
                raise ValueError(t['id']+': paired force/moment certificate failed '+str(proof))
            t['contact_data']=data
            records.append(dict(id=t['id'],force=proof,overlap_m3=overlap,
                exit_length_m=t['length'],rigid_fixture_installation=True))
        result=dict(passed=True,full_loads=True,source_face_prisms_checked=continuous,tasks=records,
            continuous_paths=checks,floor_constraints=False,seconds=time.monotonic()-began)
        candidate['audit']=result; return result

    def subset(self,indices):
        model=MembraneModel(self.source,self.outer,[self.records[i] for i in indices],
            self.workspace,self.physical,self.policy,self.name)
        # Pose/contact caches are geometry-bound, never cached force passes.
        model.world_sweep_cache=self.world_sweep_cache
        return model
