"""Reference Model.exact initialization with an algebraic CSG simplification.

Only construction order changes: C subset S-N and N subset P imply
((S-P)+C)-N-(P-C) = (S-P)+C. All original actual-volume guards, repairs,
export checks, paired-load diagnostics and endpoint separation remain.
"""
import time
from subdivide_task.step4.adapter import TaskModel
from whole_search.common import *
from whole_search.common import _PRISM_FACES
from whole_search.classify import classify

def initialize_exact(self, layout):
    key = layout.key()
    if key in self.unresolved_layouts:
        raise RuntimeError('Previously unresolved candidate: '+self.unresolved_layouts[key])
    if key in self.exact_cache:
        self.exact_cache.move_to_end(key)
        return self.exact_cache[key]
    if hasattr(self,'worker_program'):
        return self.isolated_exact(layout)
    began=time.monotonic(); self.exact_calls += 1
    active=layout.active
    for k in active:
        if layout.directions[k] @ self.floor_normal(layout,k)<-1e-12:
            raise ValueError('Direction crosses its world ground plane')
        self.validate_placement(layout,k)
    stage=time.monotonic();timings={}
    wrap_keys=set();wrap_parts=[]
    for k in active:
        wrap_key=(layout.placements[k].tobytes(),tuple(self.allowed[k]))
        if wrap_key not in wrap_keys:
            wrap_keys.add(wrap_key)
            wrap_parts.append(transform_solid(self.wraps[k],layout.placements[k]))
    seed=C.union(wrap_parts)
    if self.registered_seed is not None and all(np.allclose(layout.placements[k],np.eye(4),atol=1e-12,rtol=0) for k in active):
        # Direction-only search restores material from the complete new
        # Step3.1 seed. Re-seated layouts use the original individual wraps,
        # so former work/exit holes can recover when no active pose blocks them.
        seed=self.registered_seed
    work=C.union([transform_solid(self.work_solid(layout,k),layout.placements[k]) for k in active])
    bodies=[];body_keys=set()
    for k in active:
        body_key=layout.placements[k].tobytes()
        if body_key not in body_keys:
            body_keys.add(body_key);bodies.append(transform_solid(self.body,layout.placements[k]))
    nominal_sweeps=[];padded_sweeps=[];objects={};allowed={}
    sweep_keys=set()
    for k in active:
        q=layout.placements[k];d=q[:3,:3].T @ layout.directions[k]
        sweep_key=(q.tobytes(),d.tobytes())
        if sweep_key not in sweep_keys:
            sweep_keys.add(sweep_key)
            nominal_sweeps.append(transform_solid(self.clearance.sweep(self.length*d,padded=False),q))
            padded_sweeps.append(transform_solid(self.clearance.sweep(self.length*d,padded=True),q))
        objects[k]=transform_mesh(self.mesh,q)
        allowed[k]=self.allowed[k][self.mesh.face_normals[self.allowed[k]] @ d <= 1e-9]
    timings['wrap_and_sweeps_s']=time.monotonic()-stage;stage=time.monotonic()
    seed=seed-work
    # Every exact sweep contains its original body; do not union the same
    # coplanar bodies into the cut again. Body overlaps are still checked.
    nominal_cut=C.union(nominal_sweeps)
    padded_cut=C.union(padded_sweeps)
    nominal=seed-nominal_cut
    nominal_mesh=unpack_solid(nominal)
    core_parts=[]
    core_groups={}
    for k in active:
        core_groups.setdefault(layout.placements[k].tobytes(),[]).append(k)
    for members in core_groups.values():
        k=members[0]
        shared_allowed=np.unique(np.concatenate([allowed[j] for j in members]))
        triangles,sources=C.contact_boundary(objects[k],nominal_mesh,shared_allowed)
        for tri,source in zip(triangles,sources):
            prism=C.trimesh.Trimesh(np.vstack([tri,tri+CONTACT_DEPTH_M*objects[k].face_normals[source]]),_PRISM_FACES,process=False)
            if prism.volume<0:
                prism.invert()
            part=C.S.solid(prism)
            # Distribute the intersection over the union only once below.
            # Testing each prism separately repeats the same large Boolean
            # hundreds of times and does not change the resulting core.
            core_parts.append(part)
    core=C.union(core_parts) ^ nominal
    # C is already a subset of S-N and N is contained in P. Thus the
    # reference ((S-P)+C)-N-(P-C) equals (S-P)+C. Avoid redundant, very
    # expensive coplanar recuts; every original exclusion guard stays below.
    print('Step4.1: forming protected contact core', flush=True)
    core = core.as_original()
    remaining=(seed-padded_cut)+core
    print('Step4.1: checking finite material exclusions', flush=True)
    # Only actual exclusions may be recut; never relax the guard.
    volumes=None
    for _ in range(3):
        violations=[remaining ^ v for v in nominal_sweeps+bodies+[work]]
        violations.append((remaining-core)^padded_cut)
        volumes=[C.material_volume(v) for v in violations]
        if max(volumes,default=0.)<=TOL:
            break
        remaining=remaining-C.union([v for v,volume in zip(violations,volumes) if volume>TOL])
        volumes=None
    # Reuse the very intersections just validated. Re-evaluate only if
    # the final repair changed the solid after their computation.
    if volumes is None:
        volumes=[C.material_volume(remaining ^ v) for v in nominal_sweeps+bodies+[work]]
        volumes.append(C.material_volume((remaining-core)^padded_cut))
    ns,nb=len(nominal_sweeps),len(bodies)
    diagnostics=dict(nominal_sweep_overlap_m3=max(volumes[:ns],default=0.),
                     body_overlap_m3=max(volumes[ns:ns+nb],default=0.),
                     padded_overlap_outside_contact_cores_m3=volumes[-1],
                     work_band_overlap_m3=volumes[-2])
    if max(diagnostics.values())>TOL:
        raise RuntimeError('Exact exclusions unresolved: '+str(diagnostics))
    remaining=remaining.as_original()
    for conditioning in range(4):
        final=unpack_solid(remaining);reloaded=C.S.solid(final)
        replay=[C.material_volume(reloaded^part) for part in nominal_sweeps+bodies+[work]]
        replay.append(C.material_volume((reloaded-core)^padded_cut))
        if max(replay,default=0.)<=TOL:break
        if conditioning==3:raise RuntimeError('Exported whole-support exclusions remain unresolved')
        remaining=(reloaded-C.union(nominal_sweeps+bodies+[work]))-(padded_cut-core)
    diagnostics.update(exported_boundary_max_exclusion_overlap_m3=max(replay,default=0.),
                       boolean_boundary_conditioning_rounds=conditioning)
    timings['solid_contacts_and_exclusions_s']=time.monotonic()-stage;stage=time.monotonic()
    if not len(final.faces):
        raise RuntimeError('No remaining material')
    masks={};supplies={};contacts={};classifiers={};footprints=[]
    for k in active:
        tri,src=C.contact_boundary(objects[k],final,allowed[k])
        # Map contacts back to immutable native task coordinates. A lateral
        # layout shift translates its physical task and floor together;
        # this inverse shift preserves COM-relative original wrench data.
        native_map=self.native[k] @ np.linalg.inv(layout.placements[k])
        if self.floor_signature(layout,k):
            full=C.supply(self.tasks[k],native_map,tri,src)
        else:
            points=C.transform_points(tri.reshape(-1,3),native_map)
            normals=np.repeat(-self.tasks[k].domain.mesh.face_normals[src],3,axis=0)
            raw=np.c_[normals,np.cross(points-self.tasks[k].domain.com,normals)]
            full=C.I.merge_columns(self.floor_columns(layout,k),C.U.heads(raw,self.tasks[k].scale))
        mask,info=classify(full,self.tasks[k].targets)
        masks[k]=mask;supplies[k]=full;classifiers[k]=info
        contacts[k]=dict(triangle_count=len(tri),area_m2=float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2),triangles=tri,sources=src)
        xy=C.transform_points(final.vertices,self.native[layout.hosts[k]])[:,:2]
        footprints.append(float(ConvexHull(xy).volume))
    timings['final_contacts_and_loads_s']=time.monotonic()-stage;stage=time.monotonic()
    end_overlaps=[];gaps=[]
    for k in active:
        q=layout.placements[k].copy();q[:3,3]+=self.length*layout.directions[k]
        end_overlaps.append(C.material_volume(transform_solid(self.body,q)^seed))
        d=layout.directions[k]
        gaps.append(float((objects[k].vertices @ d).min()+self.length-(final.vertices @ d).max()))
    if max(end_overlaps,default=0.)>TOL or min(gaps,default=1.)<=0:
        raise RuntimeError('Full exit endpoint not detached')
    diagnostics.update(endpoint_overlap_m3=max(end_overlaps,default=0.),minimum_endpoint_projection_gap_m=min(gaps,default=0.))
    result=dict(layout=layout.copy(),serial=self.exact_calls,remaining=remaining,
                masks=masks,supplies=supplies,contacts=contacts,classifiers=classifiers,
                counts={str(k):int(m.sum()) for k,m in masks.items()},diagnostics=diagnostics,
                volume_cm3=C.material_volume(remaining)*1e6,maximum_projected_footprint_m2=max(footprints),
                seconds=time.monotonic()-began,timings=timings)
    if all(mask.all() for mask in masks.values()):
        result['actual_work_surface_checks']=self.verify_work(result)
        if not all(row['passed'] for row in result['actual_work_surface_checks']):
            raise RuntimeError('Full force coverage but working surface verification unresolved: '+
                               str(result['actual_work_surface_checks']))
    timings['exit_endpoint_and_work_s']=time.monotonic()-stage
    result['seconds']=time.monotonic()-began
    self.exact_cache[key]=result
    self.store_checkpoint(result)
    if len(self.exact_cache)>2:
        self.exact_cache.popitem(last=False)
    print('EXACT',self.exact_calls, 'counts', list(result['counts'].values()),'seconds',round(result['seconds'],2),flush=True)
    return result
