"""Paired-wrench residual descent, followed by feasible material descent.

The finite differences and backtracking follow Co-optimize's
continuous_support/block_descent.py. Geometry and loads here come from the
current Step3 model, never its legacy fixed-pose dataset. A pair objective
guides repair only; acceptance certifies the merged family, retaining the
proofs of other families only after checking their geometry is unaffected.
"""
from collections import OrderedDict
import numpy as np
import shapely
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog

from subdivide_task.common.force_cone import tangent_frame
from subdivide_task.operations.geometry import moved_mesh, contact_surface
from subdivide_task.operations.construction import overlaps
from subdivide_task.operations.floors import clip, planes
from subdivide_task.operations.pose_change import pose_change
from subdivide_task.step3.independent_physics import contacts
from subdivide_task.step3.legacy import kernel as K
from subdivide_task.step3 import independent_geometry as G
from subdivide_task.operations.state import transform
from .optimize import subset


def global_pressure_feasible(model, layout):
    """Keep every task above every installed floor, including unedited tasks."""
    normals = layout.material[:, :3, 2]
    offsets = np.einsum('ij,ij->i', normals, layout.material[:, :3, 3])
    for record, pose, material in zip(model.records, layout.poses, layout.material):
        data = model.world_data(record, pose)
        points = G.pressure_points(data['endpoints']['endpoint_need_wrench'], data['domain'].com)
        points = transform(np.c_[points, np.zeros(len(points))], material)
        if np.min(points@normals.T-offsets) < -1e-9:
            return False
    return True


def project_pressure(model,layout,owners):
    """Minimal floor-tangent correction after a basic pose perturbation.

    Floor planes remain fixed, including all unedited installations. Only
    participating task positions may move. This is a constraint projection,
    never a force certificate or a change of the family partition.
    """
    q=layout.copy();normals=q.material[:,:3,2]
    offsets=np.einsum('ij,ij->i',normals,q.material[:,:3,3])
    for k in owners:
        data=model.world_data(model.records[k],q.poses[k])
        pressure=G.pressure_points(data['endpoints']['endpoint_need_wrench'],data['domain'].com)
        points=transform(np.c_[pressure,np.zeros(len(pressure))],q.material[k])
        distance=(points@normals.T-offsets).min(0)
        if distance.min()>=-1e-9:continue
        frame=np.asarray(tangent_frame(q.material[k,:3,2]))
        slopes=normals@frame.T
        movable=np.linalg.norm(slopes,axis=1)>1e-10
        if (distance[~movable]<-1e-9).any():return None
        A=[np.r_[-s,np.zeros(2)] for s in slopes[movable]]
        b=(distance[movable]-2e-6).tolist()
        for j in range(2):
            row=np.zeros(4);row[j]=1.;row[2+j]=-1.;A.append(row);b.append(0.)
            row=np.zeros(4);row[j]=-1.;row[2+j]=-1.;A.append(row);b.append(0.)
        result=linprog([0.,0.,1.,1.],A_ub=np.asarray(A),b_ub=np.asarray(b),
            bounds=[(None,None)]*2+[(0.,None)]*2,method='highs')
        if not result.success:return None
        q.material[k,:3,3]+=result.x[:2]@frame
    return q if global_pressure_feasible(model,q) else None


def guidance_contacts(task, support, model, limit=512):
    """Conservative subset of ACTUAL contact polygons, only for gradients.

    Keep spatially distributed source faces, wrench extremes and important
    existing contacts. The full acceptance still uses every allowed face.
    """
    body=task['body_world']
    allowed=np.setdiff1d(np.arange(len(body.faces)),task['domain'].work_ids)
    if not len(allowed):raise RuntimeError('No non-working contact faces remain')
    if len(allowed)>limit:
        ids=allowed[np.unique(np.linspace(0,len(allowed)-1,limit).astype(int))]
        normal=-body.face_normals[allowed]
        wrench=np.c_[normal,np.cross(body.triangles_center[allowed]-task['domain'].com,normal)]
        extreme=allowed[np.r_[wrench.argmin(0),wrench.argmax(0)]]
        profile=getattr(model,'contact_profiles',{}).get(task['id'])
        relevant=allowed[np.argsort(profile[allowed])[-64:]] if profile is not None else np.empty(0,int)
        selected=np.unique(np.r_[ids,extreme,relevant])
        excluded=np.setdiff1d(np.arange(len(body.faces)),selected)
    else:
        excluded=task['domain'].work_ids
    return contacts(body,contact_surface(body,support),excluded,task['domain'].com)


class PairObjective:
    def __init__(self, model, layout, owners, samples=24, obstruction_context=None):
        self.model = model
        self.pair, _ = subset(model, layout, list(owners))
        self.owners = list(owners)
        self.extent = float(model.source.extents.max())
        self.samples = samples
        self.cache = OrderedDict()
        self.calls = 0
        # Unedited tasks still cut the common fixture. Omitting their paths
        # would invent contact material absent from the full candidate.
        selected = {model.ids[k] for k in owners}
        context = model.prepare(layout) if obstruction_context is None else obstruction_context
        local_context=[t for t in context if t['id'] in selected]
        bounds = np.array([G.mesh(t['stock']).bounds for t in local_context])
        points = np.vstack([bounds.reshape(-1,3),*[t['patch_points'] for t in local_context]])
        reference = model.reference_material(layout.material[self.owners])
        normal = reference[:3,2]; offset = normal@reference[:3,3]
        projected = points-(points@normal-offset)[:,None]*normal
        points = np.vstack([points,projected,projected+model.policy['plate_thickness_m']*normal])
        self.pair.guide_envelope_bounds = (points.min(0)-.08,points.max(0)+.08)
        if hasattr(self.pair,'family_before_support'):
            # Unedited distant cells are constant in this objective. Retain
            # every selected stock, actual floor patch and the full trust
            # region, but avoid remeshing the entire Step3 row per derivative.
            region=G.box(*self.pair.guide_envelope_bounds)
            self.pair.family_before_support=self.pair.family_before_support^region
            self.pair.family_previous_regions=[r^region for r in self.pair.family_previous_regions]
            self.pair.family_protected_contact_regions=self.pair.family_protected_contact_regions^region
            if hasattr(self.pair,'family_retained_material'):
                self.pair.family_retained_material=self.pair.family_retained_material^region
        self.foreign = [t[key] for t in context if t['id'] not in selected for key in ('sweep','work')]
        self.foreign_bounds = [G.mesh(s).bounds for s in self.foreign]

    def contextual_support(self, candidate, layout):
        support = candidate['support']
        hits = [s for s,b in zip(self.foreign,self.foreign_bounds) if overlaps(support.bounds,b)]
        solid = clip(G.solid(support),planes(layout.material))
        if hits: solid = solid-G.union(hits)
        result = G.mesh(solid)
        if result.is_empty: raise ValueError('Other tasks remove all pair contact material')
        return result

    def evaluate(self, layout):
        key = tuple(a.tobytes() for a in layout.arrays().values())
        if key in self.cache:
            return self.cache[key]
        self.calls += 1
        result = None
        try:
            if not global_pressure_feasible(self.model, layout):
                raise ValueError('A basic adjustment violates another task installation floor')
            _, local = subset(self.model, layout, self.owners)
            candidate = self.pair.build(local)
            candidate['support'] = self.contextual_support(candidate,layout)
            candidate['volume_cm3'] = float(candidate['support'].volume*1e6)
            losses, maximum, details = [], [], []
            # The original force and moment stay paired at each application
            # point. These fixed stratified demands guide finite differences;
            # they can neither accept nor certify a saved design.
            from continuous_support.projection import project_demands
            for task in candidate['tasks']:
                support = moved_mesh(candidate['support'], np.linalg.inv(task['material']))
                _, contact_ids, rays, scale = guidance_contacts(task,support,self.model)
                targets = []
                for values in (task['samples']['need_wrench'], task['endpoints']['endpoint_need_wrench']):
                    ids = np.unique(np.linspace(0, len(values)-1, self.samples).astype(int))
                    targets.append(values[ids] * scale)
                targets = np.vstack(targets)
                passed, _ = K.J.classify(rays, targets)
                loss, residual = 0., 0.
                if not passed.all():
                    projected = project_demands(rays, K.J.C.U.target(targets, rays.shape[1]))
                    loss = float(np.mean(projected['normalized_losses']))
                    residual = float(np.max(np.abs(projected['residuals'])))
                triangles = support.triangles
                ground = triangles[np.abs(triangles[:, :, 2]).max(1) < 1e-9, :, :2]
                footprint = shapely.union_all(shapely.polygons(ground))
                pressure = task['pressure_world']
                if footprint.is_empty:
                    floor_loss = 1.
                else:
                    distance = shapely.distance(footprint, shapely.points(pressure))
                    floor_loss = float(np.mean((distance / self.extent)**2))
                losses.append(loss + floor_loss)
                maximum.append(max(residual, np.sqrt(floor_loss)))
                details.append(dict(task=task['id'], residual_loss=loss,
                    maximum_residual=residual, floor_loss=floor_loss,
                    sampled_paired_demands=len(targets),
                    sampled_contact_source_faces=int(len(np.unique(contact_ids))),
                    conservative_contact_subset=True,diagnostic_only=True))
            result = dict(layout=layout.copy(), loss=float(np.mean(losses)),
                covered=bool(max(maximum) <= 2e-8), volume_cm3=candidate['volume_cm3'],
                details=details)
        except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError):
            pass
        self.cache[key] = result
        if len(self.cache) > 8:
            self.cache.popitem(last=False)
        return result


def block_step(evaluate, current, perturb, dimension, difference, step, backtracks=4):
    """One measured gradient direction, with feasible one-sided differences."""
    phase = 'volume' if current['covered'] else 'repair'
    reference = max(current['volume_cm3'], 1.)
    value = lambda row: row['volume_cm3']/reference if phase == 'volume' else row['loss']
    center = value(current)
    gradient = np.zeros(dimension)
    probes = []
    for j in range(dimension):
        delta = np.zeros(dimension); delta[j] = difference
        left, right = evaluate(perturb(-delta)), evaluate(perturb(delta))
        if left is not None and right is not None:
            gradient[j] = (value(right)-value(left))/(2*difference)
        elif right is not None:
            gradient[j] = (value(right)-center)/difference
            gradient[j] = min(gradient[j], 0.)
        elif left is not None:
            gradient[j] = (center-value(left))/difference
            gradient[j] = max(gradient[j], 0.)
        probes.append(dict(minus=None if left is None else value(left),
                           plus=None if right is None else value(right)))
    norm = float(np.linalg.norm(gradient))
    record = dict(phase=phase, gradient=gradient.tolist(), difference=difference,
        probes=probes, line_search=[], accepted=False,
        method='projected_finite_difference_gradient_with_backtracking',
        guidance_only=True)
    if norm <= 1e-12:
        return current, record
    direction = -gradient/norm
    slope = float(gradient@direction)
    for attempt in range(backtracks):
        amount = step/2**attempt
        trial = evaluate(perturb(amount*direction))
        accepted = bool(trial is not None and (phase != 'volume' or trial['covered'])
            and value(trial) < center-1e-13
            and value(trial) <= center + 1e-4*amount*slope)
        record['line_search'].append(dict(step=amount, accepted=accepted,
            value=None if trial is None else value(trial)))
        if accepted:
            record['accepted'] = True
            return trial, record
    return current, record


def repair_owners(model,representatives,scope):
    """Include real failed members of the union, even if probes miss them."""
    editable=list(representatives)
    active=set()
    for kind in ('endpoint','sample','pressure'):
        active.update(getattr(model,'repair_'+kind+'_indices',{}))
    for owner in scope:
        if model.ids[owner] in active and owner not in editable:editable.append(owner)
    return editable


def repair(model, layout, owners, rounds=2, log=None, *, obstruction_context=None, guidance_owners=None,
           feasible_layouts=None):
    # A high-level pair can move a cohort. Test the whole affected cohort in
    # the local objective so a third task cannot silently lose its contacts.
    objective_type=PairObjective
    if getattr(model,'guidance_geometry','exact')=='sampled':
        from .sampled_guidance import SampledObjective
        objective_type=SampledObjective
    objective = objective_type(model, layout, guidance_owners or owners,
        obstruction_context=obstruction_context)
    current = objective.evaluate(layout)
    if current is None:
        return None, dict(status='unbuildable_pair', evaluations=objective.calls, blocks=[])
    blocks = []
    for iteration in range(rounds):
        changed = False
        for kind in ('exit_redirection', 'translation', 'pose_rotation'):
            base = current['layout']
            editable = repair_owners(model,owners,guidance_owners or owners)
            if not current['covered']:
                worst = max(current['details'],key=lambda row:row['residual_loss']+row['floor_loss'])
                offender = model.ids.index(worst['task'])
                if offender not in editable: editable.append(offender)
            coordinates = []
            for owner in editable:
                if kind == 'exit_redirection':
                    coordinates.extend((owner, axis) for axis in tangent_frame(base.exits[owner]))
                elif kind == 'translation':
                    coordinates.extend((owner, axis) for axis in tangent_frame(base.material[owner,:3,2]))
                else:
                    coordinates.append((owner, model.axis))

            def perturb(vector):
                q = base.copy()
                angular = {}
                for amount, (owner, axis) in zip(vector, coordinates):
                    if kind == 'translation':
                        q.material[owner, :3, 3] += objective.extent*amount*axis
                    elif kind == 'pose_rotation':
                        q = pose_change(q, owner, np.degrees(amount), axis, model.head)
                    else:
                        angular[owner] = angular.get(owner, np.zeros(3)) + amount*axis
                for owner, vector in angular.items():
                    q.exits[owner] = Rotation.from_rotvec(np.cross(base.exits[owner], vector)).apply(base.exits[owner])
                return q

            difference = .004 if kind == 'translation' else np.radians(.5)
            step = .03 if kind == 'translation' else np.radians(5.)
            current, record = block_step(objective.evaluate, current, perturb,
                len(coordinates), difference, step)
            record.update(operation=kind, iteration=iteration+1,
                adjusted_tasks=[model.ids[k] for k in editable])
            blocks.append(record)
            changed |= record['accepted']
            if feasible_layouts is not None and record['accepted'] and current['covered']:
                feasible_layouts.append(current['layout'].copy())
            if log is not None:
                log('basic_gradient', kind, pair_guidance_loss=current['loss'],
                    pair_guidance_volume_cm3=current['volume_cm3'], gradient_step=record,
                    unchanged_family_interference_loss=current.get('unchanged_family_interference_loss',0.))
        if not changed:
            break
    return current['layout'], dict(status='pair_guidance_covered' if current['covered'] else 'pair_guidance_unresolved',
        guidance_geometry=getattr(model,'guidance_geometry','exact'),
        evaluations=objective.calls, final_loss=current['loss'], final_pair_volume_cm3=current['volume_cm3'],
        volume_material_rule=current.get('volume_material_rule','exact_CSG'),
        final_task_diagnostics=current['details'],
        unchanged_family_interference_loss=current.get('unchanged_family_interference_loss',0.),
        diagnostic_only=True, merged_family_requires_full_acceptance=True, blocks=blocks)
