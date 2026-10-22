"""Existence of one connected, ground-safe extension for every task in a set.

The enclosing pressure polygon is virtual: only its containing ground
contact points need material.  A positive answer stores an actual connected
free-space witness, with all original work/exit and paired-load checks.
"""
import time
import numpy as np
import trimesh
from scipy.spatial import ConvexHull,QhullError
from whole_task.operations.state import transform
from whole_task.operations.geometry import moved_mesh
from whole_task.operations.floors import planes,clip,connected_material
from whole_task.operations.sweep_cache import sweep_mesh
from whole_task.step1.common import ROOT
from whole_task.step3 import independent_geometry as G
from whole_task.step3.exit_geometry import whole_object_sweep


def pressure_rows(model,layout,tasks=None):
    tasks=model.prepare(layout) if tasks is None else tasks
    rows=[]
    for i,t in enumerate(tasks):
        p=G.pressure_points(t['endpoints']['endpoint_need_wrench'],t['domain'].com)
        hull=p[ConvexHull(p).vertices]
        rows.append(dict(index=i,id=t['id'],hull=hull,material=layout.material[i],
            world_from_support=np.linalg.inv(layout.material[i]),
            ground_material=transform(np.c_[hull,np.zeros(len(hull))],layout.material[i])))
    return rows


def pressure_floor_gap(model,layout):
    normals=layout.material[:,:3,2]
    offsets=np.einsum('ij,ij->i',normals,layout.material[:,:3,3])
    return min(float((r['ground_material']@normals.T-offsets).min()) for r in pressure_rows(model,layout))


def project_families(model,layout,groups):
    from .spatial_search import floor_repair
    return floor_repair(model,layout,groups,groups[0][0])


def clip_core(model,candidate):
    """A proposal may remove unused membrane crossing another installation."""
    value=clip(G.solid(candidate['support']),planes(candidate['layout'].material))
    value=connected_material(value)
    candidate['support']=G.mesh(value.simplify(0.))
    candidate['volume_cm3']=float(candidate['support'].volume*1e6)
    candidate['construction']['clipped_by_every_installation_floor']=True
    return candidate


def foot_regions(model,candidate,paddings=(.15,.3,.5)):
    """Step5's free space for feet, shared by the Step4 oracle and Step5.

    Above every installation floor, minus every work sector and the convex
    envelope of every task's whole exit; plus the membrane. A region must
    contain the membrane and its ground faces must enclose every task's
    pressure hull. Yields (padding, region mesh) in increasing size.
    """
    from whole_task.operations.floors import clip,planes
    rows=pressure_rows(model,candidate['layout']);gap=pressure_floor_gap(model,candidate['layout'])
    if gap<-1e-8:raise ValueError('Required pressure hull crosses another resting plane')
    core=candidate['support'];core_solid=G.solid(core);tasks=candidate['tasks'];q=candidate['layout']
    points=np.vstack([core.vertices,*[r['ground_material'] for r in rows]])
    for padding in paddings:
        low,high=points.min(0)-padding,points.max(0)+padding
        free=clip(G.box(low,high),planes(q.material))
        corners=np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
        for i,t in enumerate(tasks):
            inverse=np.linalg.inv(q.material[i])
            top=(corners@inverse[:3,:3].T+inverse[:3,3])[:,2].max()
            span=max(t['length'],float((top+.015-t['body_world'].bounds[0,2])/q.exits[i,2]))
            vertices=t['body'].vertices
            envelope=trimesh.convex.convex_hull(np.vstack([vertices,vertices+t['direction']*span]))
            free=free-t['work']-G.solid(envelope)
        free=free+core_solid
        for part in free.decompose():
            if float(part.volume())<=0 or G.volume(core_solid-part)>model.tolerance:continue
            region=G.mesh(part.simplify(0.))
            if all(r['passed'] for r in ground_enclosure(region,rows)):yield padding,region,rows


def tangent_installation(model,layout,core,attempts=24,seed=547):
    """Step5's ground model must be satisfiable for an accepted fixture.

    Step5 grounds the membrane itself: each installation rests on the plane
    tangent to the membrane, the only freedom being a seat-preserving turn of
    the whole station about the fixed needle axis. Search such turns with
    Step5's FloorObjective (contact-screened intervals, minimax repair); a
    turn is accepted when Step5's foot region exists for it. The angles are
    recorded so Step5 starts from them.
    """
    from whole_task.step5.ground_contact import FloorObjective,GroundFrameModel,normalize_ground
    began=time.monotonic();objective=FloorObjective(model,layout,core);objective.log=False
    objective.force_ranges();regions=0
    for q_global,heights,meta in objective.proposals(attempts=attempts,seed=seed,log=False):
        ground=GroundFrameModel(model,heights);q=normalize_ground(q_global,heights)
        found=next(foot_regions(ground,dict(layout=q,support=core,tasks=ground.prepare(q))),None)
        regions+=1
        if found is None:continue
        return dict(meta,passed=True,ground_heights_m=heights.tolist(),foot_region_padding_m=found[0],
            foot_region_volume_cm3=float(found[1].volume*1e6),feasible_turns_tried=regions,
            seconds=time.monotonic()-began,
            model='membrane tangent plane per installation, seat-preserving turns about the needle axis, Step5 foot region')
    raise ValueError(f'Oracle: no tangent-ground installation with a foot region in {attempts} trials; '
        f'{regions} angle-feasible turns; best minimum margin {1e3*objective.best_gap:.3f} mm')


def ground_enclosure(mesh,rows):
    output=[]
    for row in rows:
        world=transform(mesh.vertices,row['world_from_support'])
        triangles=world[mesh.faces]
        keep=np.abs(triangles[:,:,2]).max(1)<=1e-8
        ground=triangles[keep]
        if len(ground):
            a=ground[:,1,:2]-ground[:,0,:2];b=ground[:,2,:2]-ground[:,0,:2]
            area=.5*np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
            ground=ground[area>1e-15]
        points=ground[:,:,:2].reshape(-1,2)
        gap=float('inf');polygon=np.empty((0,2))
        if len(points)>=3:
            try:
                hull=ConvexHull(points);polygon=points[hull.vertices]
                gap=max(0.,float((row['hull']@hull.equations[:,:2].T+hull.equations[:,2]).max()))
            except QhullError:pass
        minimum=float(world[:,2].min())
        output.append(dict(id=row['id'],passed=bool(minimum>=-1e-9 and gap<=1e-8),
            minimum_world_z_m=minimum,pressure_hull_gap_m=gap if np.isfinite(gap) else None,
            actual_ground_triangle_count=len(ground),ground_contact_hull_xy_m=polygon.tolist(),
            required_pressure_hull_xy_m=row['hull'].tolist(),convex_hull_is_virtual_not_filled_material=True))
    return output


INHERITED=('paired loads inherited from the fully audited contained membrane; every original face prism, '
    'body and work sector checked on the witness; full witness audit at final disk replay')


def complete_proof(proof,count):
    from .membrane_search import complete_proof as check
    return check(proof,count)


class GroundOracle:
    """inherit=True does not repeat the paired-load audit on the witness.

    The witness contains the fully audited membrane and excludes every body,
    so its contact set contains the membrane's and the load proof carries
    over. Every original face prism and body/work interference is still
    checked on the witness (fast with the exact prism broad phase).
    """
    def __init__(self,model,inherit=True,tangent_attempts=None):
        self.model=model;self.inherit=inherit
        if tangent_attempts is None:
            from .membrane_search import POLICY
            tangent_attempts=POLICY.get('tangent_ground_attempts',0)
        self.tangent_attempts=tangent_attempts

    def witness_exits(self,test):
        """Every original face prism of every task, plus body/work interference."""
        m=self.model;solid=G.solid(test['support'])
        rows,_=m.path_checks(solid,m.exit_paths(test))
        if len(rows)!=len(test['tasks']) or not all(r['passed'] for r in rows):
            raise ValueError('Witness continuous exits unresolved')
        for t,row in zip(test['tasks'],rows):
            overlap=dict(body=G.volume(solid^G.solid(t['body'])),work=G.volume(solid^t['work']))
            if max(overlap.values())>m.tolerance:raise ValueError(t['id']+': witness body/work interference')
            row['overlap_m3']=overlap
        return rows

    def query(self,candidate):
        began=time.monotonic();m=self.model;q=candidate['layout'];tasks=candidate['tasks']
        rows=pressure_rows(m,q,tasks)
        gap=pressure_floor_gap(m,q)
        if gap < -1e-8:raise ValueError('Oracle: pressure hull lies below another installation floor')
        core=candidate['support'];core_solid=G.solid(core)
        for row in rows:
            if transform(core.vertices,row['world_from_support'])[:,2].min()<-1e-9:
                raise ValueError('Oracle: complete membrane crosses an installation ground')
        points=np.vstack([core.vertices,*[r['ground_material'] for r in rows]])
        failures=[];inherit=self.inherit and complete_proof(candidate.get('audit',{}),len(tasks))
        for padding in (.06,.15,.3):
            low,high=points.min(0)-padding,points.max(0)+padding
            free=clip(G.box(low,high),planes(q.material))
            corners=np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
            default_span=float(np.ceil(4*np.linalg.norm(m.source.extents)/.1)*.1)
            for i,t in enumerate(tasks):
                inverse=np.linalg.inv(q.material[i]);d=q.exits[i]
                top=transform(corners,inverse)[:,2].max()
                required=max(.01,float((top+.015-t['body_world'].bounds[0,2])/d[2]))
                if required<=default_span:
                    sweep=t['sweep']
                else:
                    span=float(np.ceil(required/.1)*.1)
                    world=sweep_mesh(ROOT/'whole_task/data'/m.name/'step3_membrane_sweeps',
                        t['body_world'],d,span,
                        lambda t=t,d=d,span=span:G.mesh(whole_object_sweep(t['body_world'],d,span)[0]),
                        m.sweep_cache_implementation)
                    sweep=G.solid(moved_mesh(world,q.material[i]))
                free=free-t['work']-sweep
            # Select a component by retention of the actual core, not by
            # size: a disconnected remote foot cannot certify this fixture.
            choices=[p for p in free.decompose() if float(p.volume())*G.SCALE**3>1e-15]
            choices.sort(key=lambda p:G.volume(p^core_solid),reverse=True)
            for index,part in enumerate(choices):
                missing=G.volume(core_solid-part)
                if missing>m.tolerance:continue
                witness=G.mesh(part.simplify(0.))
                enclosure=ground_enclosure(witness,rows)
                if not all(r['passed'] for r in enclosure):continue
                test=dict(layout=q.copy(),support=witness,tasks=m.prepare(q),
                    construction=dict(model='oracle_connected_free_space_witness'))
                try:
                    m.refine_exits(test)
                    enclosure=ground_enclosure(test['support'],rows)
                    if not all(r['passed'] for r in enclosure):raise ValueError('Witness refinement loses pressure enclosure')
                    missing=G.volume(core_solid-G.solid(test['support']))
                    if missing>m.tolerance:raise ValueError('Witness refinement removes core material')
                    if inherit:
                        exits=self.witness_exits(test)
                        audit=dict(passed=True,mode=INHERITED,continuous_paths=exits,
                            source_face_prisms_checked=True)
                    else:proof=m.audit(test)
                except (ValueError,RuntimeError) as ex:
                    failures.append(dict(padding_m=padding,component=index,reason=str(ex)));continue
                result=dict(passed=True,status='connected_ground_extension_exists',
                    finite_witness_padding_m=padding,pressure_floor_minimum_gap_m=gap,
                    entire_core_retained_m3=missing,ground=enclosure,
                    work_and_original_continuous_exits_clear=True,
                    witness_volume_cm3=float(test['support'].volume*1e6),
                    geometry_tolerance_m3=m.tolerance,seconds=time.monotonic()-began)
                if inherit:result.update(witness_audit_mode=INHERITED,inherited_core_audit=candidate['audit'],
                    witness_exit_and_interference_audit=audit)
                else:result['full_witness_audit']=proof
                if self.tangent_attempts:
                    result['tangent_installation']=tangent_installation(m,q,core,attempts=self.tangent_attempts)
                result['seconds']=time.monotonic()-began
                return test['support'],result
            failures.append(dict(padding_m=padding,status='no_core_component_encloses_every_ground_pressure_hull'))
        raise ValueError('Oracle: no connected ground-foot witness in searched windows '+str(failures))
