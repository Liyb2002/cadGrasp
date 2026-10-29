"""Merge-family certificates; unchanged families retain their existing proofs."""
from copy import deepcopy
import hashlib
import numpy as np
import shapely
import manifold3d as md
from scipy.optimize import linprog
from scipy.linalg import qr

from subdivide_task.operations.geometry import moved_mesh, contact_surface
from subdivide_task.operations.state import transform
from subdivide_task.common.force_cone import tangent_frame
from subdivide_task.step3 import independent_geometry as G


class RetainedExitError(ValueError):
    def __init__(self,task,bound,tolerance):
        self.task=task
        super().__init__(f'{task}: merge would alter an independent family continuous exit ({bound:g} > {tolerance:g})')


def validate_partition(groups, count):
    if (not groups or any(not g for g in groups)
            or sorted(k for g in groups for k in g) != list(range(count))):
        raise ValueError('Task families must partition the complete task set')


def random_pair(model, layout, groups, rng):
    """Uniform family pair, then uniform representative in each family."""
    from .pose_neighbors import closest_axial_angle
    validate_partition(groups, len(model.ids))
    if len(groups) < 2:
        return None
    a, b = map(int, rng.choice(len(groups), 2, replace=False))
    i, j = int(rng.choice(groups[a])), int(rng.choice(groups[b]))
    axial, _ = closest_axial_angle(layout.poses[i], layout.poses[j], model.axis)
    return dict(owner=i, host=j, axial_deg=axial,
                owner_group=list(groups[a]), host_group=list(groups[b]))


def merged_groups(groups, owner, host):
    a = next(g for g in groups if owner in g)
    b = next(g for g in groups if host in g)
    if a == b:
        raise ValueError('A high-level proposal must join two different families')
    return [list(g) for g in groups if g is not a and g is not b] + [sorted(a+b)]


def freeze_independent_material(model, layout, support, owners):
    """Replace selected-family regions, retaining the actual rest of the parent."""
    outside=[k for k in range(len(model.ids)) if k not in owners]
    if outside:
        context=model.prepare(layout)
        if not hasattr(model,'working_exclusion_world'):
            model.working_exclusion_world=moved_mesh(G.mesh(context[0]['work']),np.linalg.inv(layout.material[0]))
        model.family_before_support=G.solid(support)
        if getattr(model,'simplify_construction_boundaries',False):
            # Proposal geometry only. Retained proofs below still compare
            # the original actual parent with the final actual candidate.
            model.family_before_support=model.family_before_support.simplify(0.)
        model.family_before_layout=layout.copy()
        model.family_edit_ids={model.ids[k] for k in owners}
        feet=[md.Manifold.hull_points(context[k]['patch_points']/G.SCALE) for k in outside]
        model.family_protected_contact_regions=G.union([*feet,*[
            G.box(context[k]['body'].bounds[0]-1e-5,context[k]['body'].bounds[1]+1e-5)
            for k in outside]])
        mode=getattr(model,'family_refit_mode','local')
        if mode in ('free_material','free_material_box'):
            from subdivide_task.operations.ribbed_stock import contact_skin
            skin=contact_skin(model)
            contacts=[G.solid(moved_mesh(skin,layout.material[k]@layout.poses[k])) for k in outside]
            feet=[md.Manifold.hull_points(context[k]['patch_points']/G.SCALE) for k in outside]
            if mode=='free_material_box':
                # Retain the parent's exact material across a simple padded
                # body box. The tighter Minkowski layer can retriangulate
                # sharp contacts when intersected with an old CSG boundary.
                contacts=[G.box(context[k]['body'].bounds[0]-.001,
                                context[k]['body'].bounds[1]+.001) for k in outside]
            model.family_protected_contact_regions=G.union([*contacts,*feet])
            model.family_retained_material=model.family_before_support^model.family_protected_contact_regions
        model.family_previous_regions=[]
        for k in owners:
            task=context[k]
            bounds=np.array([G.mesh(task[key]).bounds for key in ('stock','dense_stock') if key in task])
            model.family_previous_regions.append(G.box(bounds[:,0].min(0)-1e-5,bounds[:,1].max(0)+1e-5))
            model.family_previous_regions.append(md.Manifold.hull_points(task['patch_points']/G.SCALE))


def same_contact_generators(task, before_support, after_support, witness_cache=None, contact_cache=None,
                            _all_contacts=False):
    """Retain the parent proof's reaction bases on actual new contacts.

    Native near-coincident Boolean differences can contain ghost slivers.
    When that guard is inconclusive, compare actual reaction generators;
    A bound witness cache avoids solving its original candidate demands.
    Legacy parents missing witnesses bootstrap them on the OLD fixture once.
    """
    if 'domain' not in task:
        return False, {}
    from subdivide_task.step3.independent_physics import contacts
    rays=[]; arrays=[];bootstrap=False;witness_report={};limited=False
    name=task['id']
    def unresolved(proof):
        if limited:
            passed,full=same_contact_generators(task,before_support,after_support,
                witness_cache,contact_cache,_all_contacts=True)
            full['contact_subset_fallback_used']=True
            if bootstrap:
                full['parent_certificate_replayed_for_witness_bootstrap']=True
                full['parent_load_witness']=witness_report
            return passed,full
        return False,proof
    for index,mesh in enumerate((before_support,after_support)):
        if index==0 and witness_cache is not None and name in witness_cache:
            columns=witness_cache[name]
        else:
            installed=moved_mesh(mesh,np.linalg.inv(task['material']))
            surface=contact_surface(task['body_world'],installed)
            excluded=task['domain'].work_ids
            if index==1 and witness_cache is not None and not _all_contacts:
                # Start with original source faces having the normals used
                # by the certified basis. Their actual contact generators
                # form a safe subset of the full cone. If insufficient,
                # fall back to every allowed source face, never approximate.
                normals={tuple(n) for n in np.round(arrays[0][:,:3],13) if np.linalg.norm(n)>0.}
                selected=np.array([k for k,n in enumerate(np.round(-task['body_world'].face_normals,13))
                    if tuple(n) in normals and k not in excluded],dtype=int)
                if len(selected):
                    excluded=np.setdiff1d(np.arange(len(task['body_world'].faces)),selected)
                    limited=len(excluded)>len(task['domain'].work_ids)
            digest=hashlib.sha256()
            for array in (task['body_world'].vertices,task['body_world'].faces,
                          excluded,task['domain'].com,surface.triangles):
                digest.update(np.ascontiguousarray(array).tobytes())
            key=digest.hexdigest()
            if contact_cache is not None and key in contact_cache:
                columns=contact_cache[key]
            else:
                try:
                    _,_,columns,_=contacts(task['body_world'],surface,excluded,task['domain'].com)
                except RuntimeError:
                    if limited:return unresolved({})
                    raise
                if contact_cache is not None:
                    contact_cache[key]=columns.copy()
                    if len(contact_cache)>96:contact_cache.pop(next(iter(contact_cache)))
            if index==0 and witness_cache is not None:
                from subdivide_task.operations.force_witness import required_generators
                columns,witness_report=required_generators(task,columns)
                witness_cache[name]=columns
                bootstrap=True
        rays.append({tuple(row) for row in np.round(columns,13)})
        arrays.append(columns)
    missing=[row for row in arrays[0] if tuple(np.round(row,13)) not in rays[1]]
    proof=dict(parent_generator_count=len(rays[0]),
        current_generator_count=len(rays[1]),kernel_generator_deduplication_digits=13,
        parent_generators_retained=not missing,missing_identical_generators=len(missing),
        strict_cone_inclusion_generators=0,kernel_rounded_cone_inclusion_generators=0,
        original_loads_resolved=False,parent_certificate_replayed_for_witness_bootstrap=bootstrap,
        parent_load_witness=witness_report,contact_subset_used=limited,
        checked_contact_source_face_count=len(task['body_world'].faces)-len(excluded))
    if witness_cache is not None:
        proof['parent_load_witness_generators_7d']=arrays[0].tolist()
    if not missing:
        return True,proof
    # Every head column has its seventh coordinate equal to Fz. Certify
    # inclusion in the six independent wrench coordinates. A positive,
    # invertible basis absorbs the measured LP residual with nonnegative
    # weights; borderline or rank-deficient cases are rejected, not accepted
    # using the task load classifier's looser feasibility tolerance.
    # U.floor(empty) still contributes the shared auxiliary ray
    # (0,0,0,0,0,0,-1). It is NOT a surface reaction and does not satisfy
    # h=Fz. It was already matched above; separate it before reducing head
    # generators to their six independent wrench coordinates.
    auxiliary=(arrays[1][:,:6]==0.).all(1)&(arrays[1][:,6]==-1.)
    columns=arrays[1][~auxiliary]
    proof['shared_auxiliary_generator_count']=int(auxiliary.sum())
    if not (np.array_equal(columns[:,6],columns[:,2])
            and all(row[6]==row[2] for row in missing)):
        return unresolved(proof)
    matrix=columns[:,:6].T
    _,_,pivots=qr(matrix,mode='economic',pivoting=True)
    basis_ids=pivots[:6];basis=matrix[:,basis_ids]
    full_rank=basis.shape==(6,6) and np.linalg.matrix_rank(basis)==6
    inverse=np.linalg.inv(basis) if full_rank else None
    for target in missing:
        certified=False
        # Boundary rays need not have a positive six-dimensional basis.
        # Retriangulating the same contact polygon commonly replaces an
        # edge vertex by a convex combination on that very edge. Reconstruct
        # it on the LP's active subspace, then demand equality at the kernel's
        # existing 13-digit generator precision. This introduces no new
        # force tolerance and still rejects negative reaction coefficients.
        result=linprog(np.ones(len(columns)),A_eq=matrix,b_eq=target[:6],
            bounds=(0.,None),method='highs',options={'primal_feasibility_tolerance':1e-10})
        if result.success:
            weights=result.x.copy()
            active=np.flatnonzero(weights>0.)
            if len(active):
                _,_,p=qr(matrix[:,active],mode='economic',pivoting=True)
                rank=np.linalg.matrix_rank(matrix[:,active])
                selected=active[p[:rank]]
                active_basis=matrix[:,selected]
                _,_,rows=qr(active_basis.T,mode='economic',pivoting=True)
                rows=rows[:rank]
                correction=np.linalg.solve(active_basis[rows],
                    (target[:6]-matrix@weights)[rows])
                weights[selected]+=correction
                represented=matrix@weights
                if (np.min(weights)>=0. and
                        np.array_equal(np.round(represented,13),np.round(target[:6],13))):
                    proof['kernel_rounded_cone_inclusion_generators']+=1
                    continue
        for interior in ((1e-6,1e-8,1e-10) if full_rank else ()):
            rhs=target[:6]-interior*basis.sum(1)
            result=linprog(np.ones(len(columns)),A_eq=matrix,b_eq=rhs,
                bounds=(0.,None),method='highs',options={'primal_feasibility_tolerance':1e-10})
            if not result.success:continue
            weights=result.x.copy();weights[basis_ids]+=interior
            correction=inverse@(target[:6]-matrix@weights)
            weights[basis_ids]+=correction
            error=128*np.finfo(float).eps*(np.abs(target[:6])+np.abs(matrix)@np.abs(weights)+1.)
            allowance=np.abs(inverse)@error
            if (np.min(weights)>=0. and np.all(weights[basis_ids]>allowance)
                    and np.max(np.abs(matrix@weights-target[:6]))<1e-12):
                certified=True;break
        if not certified:
            proof['unresolved_generator']=target.tolist()
            proof['unresolved_generator_lp_status']=int(result.status)
            if result.success:
                proof['unresolved_generator_lp_residual']=float(np.max(np.abs(matrix@result.x-target[:6])))
            return unresolved(proof)
        proof['strict_cone_inclusion_generators']+=1
    proof['parent_generators_retained']=True
    return True,proof


def place_merged_family(model, before_support, layout, owners, clearance=0.):
    """Translate only the proposed family to preserve all old installation floors.

    New side floors must leave the existing support above them. Conversely,
    the merged family's pressure points must stay above every fixed floor.
    The LP is placement only; the group still needs its full load certificate.
    """
    outside=[k for k in range(len(model.ids)) if k not in owners]
    if not outside:
        return layout
    A,b=[],[]
    cloud=(np.vstack([transform(model.source.vertices,layout.material[k]@layout.poses[k])
        for k in owners]) if clearance>0 else None)
    if clearance>0 and hasattr(model,'working_exclusion_world'):
        cloud=np.vstack([cloud,*[transform(model.working_exclusion_world.vertices,layout.material[k]) for k in owners]])
    for k in owners:
        M=layout.material[k];n=M[:3,2]
        A.append(np.r_[n,np.zeros(3)])
        bound=float(((before_support.vertices-M[:3,3])@n).min())
        old_layout=getattr(model,'family_before_layout',None)
        already_installed=old_layout is not None and any(
            n@F[:3,2]>1.-1e-9 and abs(n@(M[:3,3]-F[:3,3]))<1e-9 for F in old_layout.material)
        if clearance>0 and not already_installed and not any(n@layout.material[j,:3,2]>1.-1e-9 for j in outside):
            # A side floor placed flush against the old fixture leaves the
            # new object inside an unrelated end seat. Reserve the actual
            # depth of the merged family, then move only that family outward.
            bound=min(bound,float((before_support.vertices@n).min()-(cloud@n).max()-clearance))
        b.append(bound)
        data=model.world_data(model.records[k],layout.poses[k])
        pressure=G.pressure_points(data['endpoints']['endpoint_need_wrench'],data['domain'].com)
        points=transform(np.c_[pressure,np.zeros(len(pressure))],M)
        for j in outside:
            F=layout.material[j];normal=F[:3,2]
            A.append(np.r_[-normal,np.zeros(3)])
            b.append(float(((points-F[:3,3])@normal).min()))
    for k in range(3):
        row=np.zeros(6);row[k]=1.;row[k+3]=-1.;A.append(row);b.append(0.)
        row=np.zeros(6);row[k]=-1.;row[k+3]=-1.;A.append(row);b.append(0.)
    result=linprog([0.,0.,0.,1.,1.,1.],A_ub=np.asarray(A),b_ub=np.asarray(b),
        bounds=[(None,None)]*3+[(0.,None)]*3,method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:
        raise ValueError('Merged family has no placement compatible with fixed families')
    q=layout.copy()
    q.material[owners,:3,3]+=result.x[:3]
    return q


def retained_certificates(model, before_layout, before_support, candidate, owners, previous):
    """Transfer independent load proofs through their retained reaction bases.

    An unchanged contact box is a sufficient fast guard. Otherwise retain
    the bases used by the complete parent load proof, allowing unused contact
    material to change. Bound added material in each continuous exit; refine
    ambiguous envelopes with original face prisms on added material only.
    Keep actual installed feet and endpoints. Failure rejects this merge.
    """
    selected = set(owners)
    outside = [k for k in range(len(model.ids)) if k not in selected]
    if not outside:
        return [], [], []
    if not previous.get('passed') or not previous.get('full_loads') or not previous.get('source_face_prisms_checked'):
        raise ValueError('No complete parent certificate for unchanged families')
    rows = {r['id']:r for r in previous['tasks']}
    paths = {r['id']:r for r in previous['continuous_paths']}
    q = candidate['layout']
    for key, array in q.arrays().items():
        if not np.array_equal(array[outside], before_layout.arrays()[key][outside]):
            raise ValueError('A merge changed a task outside its two families')
    old, new = G.solid(before_support), G.solid(candidate['support'])
    # A micrometre-thick protected neighbourhood detects contact-sheet loss,
    # including a removed face with vanishing volume at the object boundary.
    padding, numerical_zero = 1e-6, 1e-18
    inherited, exits, guards = [], [], []
    for k in outside:
        model.check_deadline()
        task = candidate['tasks'][k]
        name = task['id']
        if name not in rows or name not in paths:
            raise ValueError('Missing unchanged-family certificate: '+name)
        low, high = task['body'].bounds
        # Never subtract the two complete, almost coincident fixtures. That
        # Boolean can explode in time and memory across thousands of shared
        # contact sheets. Real tasks use their certified reaction bases;
        # geometry-only fixtures retain the local contact-box fast guard.
        if 'domain' in task:changed_contact=None
        else:
            region=G.box(low-padding,high+padding)
            a,b=old^region,new^region
            changed_contact=G.volume((a-b)+(b-a))
        contact_proof={}
        if changed_contact is None or changed_contact>numerical_zero:
            witnesses=getattr(model,'retention_witnesses',None)
            if witnesses is None:model.retention_witnesses=witnesses={}
            cache=getattr(model,'retention_contact_cache',None)
            if cache is None:model.retention_contact_cache=cache={}
            same, contact_proof=same_contact_generators(task,before_support,candidate['support'],witnesses,cache)
            if not same:
                model.last_retention_failure=dict(task=name,contact_proof=contact_proof)
                summary={key:value for key,value in contact_proof.items()
                    if key not in ('parent_load_witness_generators_7d','parent_load_witness')}
                raise ValueError(name+': merge would alter an independent family contact cone '+str(summary))
        path=deepcopy(paths[name])
        u,v=tangent_frame(task['direction'])
        frame=np.column_stack([u,v,task['direction']])
        projected=task['body'].vertices@frame
        low=projected.min(0)-padding;high=projected.max(0)+padding
        high[2]+=task['length']
        envelope=G.mesh(G.box(low,high));envelope.vertices=envelope.vertices@frame.T
        from subdivide_task.operations.contact_cache import boundary_key
        def local_boundary(mesh):
            triangles=mesh.triangles
            local=triangles@frame
            active=((local.max(1)>=low)&(local.min(1)<=high)).all(1)
            return boundary_key(triangles[active])
        # Identical oriented boundary triangles imply identical material in
        # this complete path region. Both bodies remain excluded below, so
        # the two regions cannot differ by an inside/outside complement.
        # The old positive world-z terminal gap extends its clear path to any
        # greater length without another contact or load computation.
        same_path_material=local_boundary(before_support)==local_boundary(candidate['support'])
        added_path=None
        bound=path.get('continuous_overlap_upper_bound_m3',0.)
        path_guard='identical_local_material'
        if not same_path_material:
            if not hasattr(model,'path_checks'):
                raise RetainedExitError(name,float('inf'),getattr(model,'tolerance',1e-10))
            actual,_=model.path_checks(new,[(name,task['body'],task['direction'],task['length'])])
            bound=actual[0]['continuous_overlap_upper_bound_m3']
            path=deepcopy(actual[0])
            path['geometry_refreshed_without_load_reclassification']=True
            path_guard='actual_new_original_face_prisms'
        if bound>getattr(model,'tolerance',1e-10):
            raise RetainedExitError(name,bound,getattr(model,'tolerance',1e-10))
        body_overlap=G.volume(new^G.solid(task['body']))
        work_overlap=G.volume(new^task['work'])
        if max(body_overlap,work_overlap)>getattr(model,'tolerance',1e-10):
            raise ValueError(name+': merge intrudes into an independent family object/work region')
        installed = moved_mesh(candidate['support'], np.linalg.inv(task['material']))
        if installed.bounds[0,2] < -1e-9:
            raise ValueError(name+': merge crosses an independent installation floor')
        ground = installed.triangles[np.abs(installed.triangles[:,:,2]).max(1)<1e-9,:,:2]
        footprint = shapely.union_all(shapely.polygons(ground))
        pressure_covered=shapely.covers(footprint, shapely.points(task['pressure_world']))
        if not pressure_covered.all():
            if hasattr(model,'note_repair_targets'):
                model.note_repair_targets(name,'pressure',np.flatnonzero(~pressure_covered))
            raise ValueError(name+': merge removes an independent family floor')
        end_world = task['body_world'].vertices + task['length']*q.exits[k]
        gap = float(end_world[:,2].min()-installed.bounds[1,2])
        if gap < .015-1e-12:
            raise ValueError(name+': merge obstructs an independent exit endpoint')
        row = deepcopy(rows[name])
        row['overlap_m3']=dict(body=body_overlap,work=work_overlap)
        row['certificate_reused_from_parent'] = True
        row['floor'].update(minimum_world_z_m=float(installed.bounds[0,2]),
                            actual_contact_area_m2=float(footprint.area))
        row['final_vertical_gap_m'] = gap
        inherited.append(row)
        path['certificate_reused_from_parent'] = True
        path['continuous_overlap_upper_bound_m3']=bound
        exits.append(path)
        guards.append(dict(task=name, unchanged_layout=True,
            changed_contact_neighbourhood_m3=changed_contact,
            added_to_continuous_path_box_m3=added_path, actual_work_overlap_m3=work_overlap,
            inherited_path_overlap_bound_m3=bound,contact_generator_proof=contact_proof,
            continuous_path_guard=path_guard,
            unchanged_oriented_local_boundary=same_path_material,
            numerical_zero_m3=numerical_zero, contact_box_padding_m=padding,
            installed_floor_preserved=True, exit_endpoint_preserved=True))
    return inherited, exits, guards


def combine_audit(local, inherited, exits, guards, parent_sha256, owners, ids):
    result = deepcopy(local)
    by_name = {r['id']:r for r in [*local['tasks'], *inherited]}
    by_path = {r['id']:r for r in [*local['continuous_paths'], *exits]}
    if set(by_name) != set(ids) or set(by_path) != set(ids):
        raise ValueError('Family certificates do not cover every task')
    result.update(tasks=[by_name[n] for n in ids], continuous_paths=[by_path[n] for n in ids],
        acceptance_scope='merged_family_with_unchanged_family_certificates',
        freshly_checked_tasks=[ids[k] for k in owners],
        retained_tasks=[r['id'] for r in inherited], parent_state_sha256=parent_sha256,
        unchanged_family_guards=guards)
    return result
