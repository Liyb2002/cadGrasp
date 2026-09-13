"""Choose a straight withdrawal before constructing a loose, one-piece frame.

Contact interfaces remain exact. Only head interiors feed narrow necks; the
remote frame, base and thick links reserve explicit clearance from the object.
Every primitive is checked over its complete forward translation ray.
"""
from step1.needs import COORD
import numpy as np
from scipy.optimize import linprog
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points

from step2_local_support import insertion as D
from step5_connect_support import belt_geometry as B, routing as T, ground as G
from step5_connect_support import whole_assembly as A, floor_design as FD, rigid_path as P

FRAME_GAP_FRACTION = .02
FRAME_WIDTH_FRACTION = .05
NORMAL_TOLERANCE = 2e-9


def preferred_direction(mesh, work_ids):
    ids = np.asarray(work_ids, int)
    vector = -(mesh.face_normals[ids]*mesh.area_faces[ids, None]).sum(axis=0)
    length = np.linalg.norm(vector)
    return vector/length if length > mesh.area*1e-12 else np.array([1., 0., 0.])


def propose_directions(mesh, contacts, work_ids, limit=24):
    """LP extrema and interior combinations propose directions, not trajectories."""
    preferred = preferred_direction(mesh, work_ids)
    faces = np.unique(np.concatenate([c['source_faces'] for c in contacts])) if contacts else np.array([], int)
    normals = mesh.face_normals[faces]
    matrix = np.unique(np.vstack([normals, [0., 0., 1.]]), axis=0)
    proposals = []
    def add(value):
        value = np.asarray(value, float)
        size = np.linalg.norm(value)
        if size < 1e-7: return
        value = value/size
        if value[2] < -1e-10 or np.min(matrix@value) < -NORMAL_TOLERANCE: return
        if abs(value[2]) < 1e-10:
            value[2] = 0.; value /= np.linalg.norm(value)
        if all(np.linalg.norm(value-p) > 1e-5 for p in proposals): proposals.append(value)
    objectives = np.vstack([preferred, np.eye(3), -np.eye(3),
                            np.random.default_rng(5119).normal(size=(24, 3))])
    diagnostics = []
    for objective in objectives:
        add(objective)
        result = linprog(-objective, A_ub=-matrix, b_ub=np.zeros(len(matrix)),
                         bounds=[(-1., 1.)]*3, method='highs',
                         options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
        diagnostics.append(dict(objective=objective.tolist(), status=int(result.status),
            solution=result.x.tolist() if result.success else None,
            minimum_normal_dot=float(np.min(matrix@result.x)) if result.success else None))
        if result.success: add(result.x)
    extremes = list(proposals)
    if extremes:
        add(np.sum(extremes, axis=0))
        for a in extremes:
            for b in extremes:
                add(a+b)
    # Prefer the working-face back direction, with a small reward for separating
    # from contact planes rather than grazing them. This is a finite heuristic.
    proposals.sort(key=lambda d: -(float(d@preferred)+.25*float(np.min(matrix@d))))
    proposals = proposals[:limit]
    blocked = []
    for c in contacts:
        dots = mesh.face_normals[c['source_faces']]@preferred
        blocked.append(dict(candidate_id=c['candidate_id'], minimum_dot=float(dots.min()),
                            opposing_area_fraction=float(c['triangle_areas_m2'][dots < -NORMAL_TOLERANCE].sum()/c['triangle_areas_m2'].sum())))
    report = dict(preferred_withdrawal_direction=preferred.tolist(),
        preferred_method='negative area-weighted outward normal of actual working faces; heuristic only',
        direction_convention='support withdrawal +d; insertion reverses the same line; workpiece fixed',
        contact_source_faces=faces.tolist(), constraint_normals=matrix.tolist(),
        candidate_directions=[p.tolist() for p in proposals], candidate_count=len(proposals),
        first_order_tolerance=NORMAL_TOLERANCE, floor_requires_nonnegative_withdrawal_z=True,
        preferred_direction_contact_conflicts=blocked, lp_diagnostics=diagnostics,
        global_impossibility_claimed=False, rotation_searched=False,
        scope='Numerical local proposals for fixed contact patches, followed by full head sweeps; not a proof against all rigid paths')
    return proposals, report


def scheduled_directions(record, limit=24):
    from step2_local_support import withdrawal as W
    if record['mode'] != W.MODE: raise ValueError('Rebuild Step2/3 common withdrawal directions')
    catalogue=record['direction_catalogue']; ids=list(record['common_directions']['ids'])
    expected=W.common(record['contacts'],catalogue['global_allowed_directions'])
    assert record['common_directions']==expected
    vectors=np.asarray(catalogue['vectors']); back=catalogue['preferred_withdrawal_direction']
    if back is not None: ids.sort(key=lambda i:-float(vectors[i]@back))
    chosen=ids[:limit]
    return [vectors[i] for i in chosen], dict(candidate_count=len(chosen),candidate_directions=vectors[chosen].tolist(),
        surviving_direction_ids=ids,chosen_direction_ids=chosen,source='Step3 actual optimized common head directions',
        preferred_withdrawal_direction=back, direction_convention=W.MOTION['direction'],
        remaining_count=len(ids),untried_count=max(0,len(ids)-len(chosen)),
        global_impossibility_claimed=False,rotation_searched=False,
        scope='Head rays already certified; frame/base/links and whole trajectory must still pass')


def exit_distance(scene, points, direction):
    """Distance until an AABB separates along a monotonically outward axis."""
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    distances = []
    for axis, value in enumerate(direction):
        if value > 1e-10:
            distances.append((scene.mesh.bounds[1, axis]+.035*scene.scale-low[axis])/value)
        elif value < -1e-10:
            distances.append((high[axis]-scene.mesh.bounds[0, axis]+.035*scene.scale)/-value)
    if not distances: raise ValueError('Withdrawal must be nonzero')
    return max(.01*scene.scale, min(distances))


class SweptScene(B.Scene):
    def __init__(self, mesh, direction, clearance=0.):
        super().__init__(mesh)
        self.direction = np.asarray(direction, float)
        if abs(np.linalg.norm(self.direction)-1) > 1e-8 or self.direction[2] < -1e-10:
            raise ValueError('Expected a unit, non-downward withdrawal direction')
        self.clearance = float(clearance)
        self.check_count = 0
        self.failure_callback = None
        self.context_parts = []
        self.context_labels = []
        self.context_stage = 'heads'

    def capture(self, part, reason, gap):
        if self.failure_callback is None:return
        parts=list(self.context_parts);labels=list(self.context_labels)
        if not any(p is part for p in parts):
            parts.append(part)
            prefix={'loose_frame':'frame_candidate','base':'ground_candidate','thick_links':'connector_candidate'}.get(self.context_stage,'rejected')
            labels.append(prefix)
        self.failure_callback(self.context_stage,parts,labels,dict(
            status=reason,clearance_m=gap,withdrawal_direction=self.direction.tolist(),
            rejected_material_saved=True,accepted_trajectory=False))

    def clear(self, part, ground=False, clearance=None):
        gap = self.clearance if clearance is None else float(clearance)
        if not B.Scene.clear(self, part, ground=ground):
            self.capture(part,'installed_object_or_floor_collision',gap)
            return False
        self.check_count += 1
        # Cube padding is a conservative envelope of a Euclidean clearance ball.
        vertices = part.vertices
        if gap > 0:
            vertices = (vertices[:, None, :]+gap*T.CORNERS).reshape(-1, 3)
        distance = exit_distance(self, vertices, self.direction)
        swept = D.engine.hull_mesh(np.vstack([vertices, vertices+distance*self.direction]))
        passed=self.volume(swept) <= 1e-11*self.scale**3
        if not passed:self.capture(part,'full_sweep_or_reserved_clearance_failed',gap)
        return passed


def collision_witness(scene, part):
    """Find an actual displaced-solid intersection to illustrate a failed sweep.

    Sampling is used only to exhibit a collision, never to accept a trajectory.
    A missing witness does not override the continuous sweep rejection.
    """
    distance=exit_distance(scene,part.vertices,scene.direction)
    for fraction in np.linspace(0.,1.,65)[1:]:
        moved=part.copy();moved.vertices=part.vertices+fraction*distance*scene.direction
        volume=scene.volume(moved)
        if volume>1e-11*scene.scale**3:
            return dict(withdrawal_distance_m=float(fraction*distance),
                intersection_volume_m3=volume,translation_m=(fraction*distance*scene.direction).tolist(),
                actual_displaced_solid_checked=True,sampling_used_for_acceptance=False)
    return None


def make_heads(mesh, contacts, depth):
    analyzer = D.Analyzer(mesh, depth)
    parts, labels = [], []
    for c in contacts:
        for k, part in enumerate(analyzer.heads(c)):
            parts.append(part); labels.append(f'contact_head_{c["candidate_id"]}_{k:04d}')
    return parts, labels


def loose_frame(scene, heads, depth, gap, offset):
    """Extend head interiors to a rear plane, then connect broad joints by a tree."""
    d = scene.direction; scale = scene.scale; radius = .5*FRAME_WIDTH_FRACTION*scale
    plane = float(np.max(scene.mesh.vertices@d)+gap+radius*np.abs(d).sum()+offset)
    neck_gap = min(depth*.01, .0001*scale)
    necks, caps, terminals = [], [], []
    head_labels=getattr(scene,'head_labels',[f'contact_head_{i:04d}_0000' for i in range(len(heads))])
    groups={}
    for index,label in enumerate(head_labels):
        groups.setdefault(label.rsplit('_',1)[0],[]).append(index)
    for contact_id,indices in groups.items():
        shifted_roots=[]
        for index in indices:
            head=heads[index];start=head.vertices.mean(axis=0)
            root=start+.85*(head.vertices-start)
            distance=max(offset,plane-float(start@d))
            shifted=root+distance*d
            neck=D.engine.hull_mesh(np.vstack([root,shifted]))
            scene.context_stage='loose_frame'
            scene.context_parts=heads+necks+caps
            scene.context_labels=head_labels+[f'neck_{k:04d}' for k in range(len(necks))]+[f'frame_{k:04d}' for k in range(len(caps))]
            if not scene.clear(neck,clearance=neck_gap):
                return dict(passed=False,status='head_neck_sweep_or_clearance_failed',failed_cell=index),None
            necks.append(neck);shifted_roots.append(shifted)
        # One broad rear joint per selected contact block, independent of mesh
        # tessellation. Its cap joins every exact interior neck of that block.
        shifted=np.vstack(shifted_roots);terminal=shifted.mean(axis=0)
        terminal[2]=max(terminal[2],radius+gap)
        cap=D.engine.hull_mesh(np.vstack([shifted,terminal+radius*T.CORNERS]))
        scene.context_parts=heads+necks+caps
        scene.context_labels=head_labels+[f'neck_{k:04d}' for k in range(len(necks))]+[f'frame_{k:04d}' for k in range(len(caps))]
        if not scene.clear(cap,clearance=gap):
            return dict(passed=False,status='rear_joint_sweep_or_clearance_failed',failed_contact=contact_id),None
        caps.append(cap);terminals.append(terminal)
    terminals = np.asarray(terminals)
    tree = {0}; remaining = set(range(1, len(terminals))); beams = []; edges = []
    while remaining:
        choices = sorted((float(np.linalg.norm(terminals[a]-terminals[b])), a, b)
                         for a in tree for b in remaining)
        for length, a, b in choices:
            part = T.beam(terminals[a], terminals[b], radius, radius)
            scene.context_parts=heads+necks+caps+beams
            scene.context_labels=getattr(scene,'head_labels',[f'contact_head_{k:04d}' for k in range(len(heads))])+[f'neck_{k:04d}' for k in range(len(necks))]+[f'frame_{k:04d}' for k in range(len(caps)+len(beams))]
            if scene.clear(part, clearance=gap):
                beams.append(part); edges.append(dict(first=a, second=b, length_m=length))
                tree.add(b); remaining.remove(b); break
        else:
            return dict(passed=False, status='rear_frame_disconnected'), None
    parts = heads+necks+caps+beams
    joined, solid = B.union_parts(parts, scale)
    report = dict(passed=solid['one_solid'], status='loose_frame_connected' if solid['one_solid'] else 'loose_frame_union_disconnected',
        construction='head-interior necks to rear joints; finite shortest-edge tree outside the workpiece',
        plane_offset_m=plane, rear_extension_m=offset, gap_m=gap, neck_gap_m=neck_gap,
        width_m=2*radius, thickness_m=depth, terminals_m=terminals.tolist(), edges=edges,
        terminal_contact_ids=list(groups), terminals_grouped_by_contact=True,
        head_cell_count=len(heads), frame_part_count=len(caps)+len(beams), neck_count=len(necks),
        closed_loop_required=False, contact_interfaces_changed=False,
        full_translation_sweep_checked=True, solid=solid)
    if not solid['one_solid']:
        if scene.failure_callback:
            labels=getattr(scene,'head_labels',[f'contact_head_{i}' for i in range(len(heads))])+[f'neck_{i}' for i in range(len(necks))]+[f'frame_{i}' for i in range(len(caps)+len(beams))]
            scene.failure_callback('loose_frame',parts,labels,dict(status=report['status'],withdrawal_direction=scene.direction.tolist()))
        return report, None
    return report, dict(necks=necks, frame=caps+beams, terminals=terminals, joined=joined)


def base_seed(mesh, floor, direction, gap):
    cloud = [floor['support_polygon_xy_m'], COORD.floor(mesh.vertices)]
    if direction[2] > 1e-5:
        q = mesh.vertices
        # Project obstacles backwards to both faces of the thick base. Include
        # the original projection for vertices already below its upper face.
        for height in (0., .04*float(mesh.extents.max())):
            amount = np.maximum(q[:, 2]-height, 0.)/direction[2]
            cloud.append(COORD.floor(q)-amount[:, None]*COORD.floor(direction))
    return FD.boundary(np.vstack(cloud))


def open_u(mesh, floor, direction, gap, expansion, scene):
    """A coarse open fallback around the directional shadow and demand hull."""
    horizontal = np.asarray(COORD.floor(direction))
    if np.linalg.norm(horizontal) < 1e-8: horizontal = np.array([1., 0.])
    # The opening faces the workpiece's relative motion during withdrawal: -d.
    angle = float(np.rad2deg(np.arctan2(-horizontal[1], -horizontal[0])))
    basis = G.frame(angle); scale = scene.scale
    seed = base_seed(mesh, floor, direction, gap)
    local = COORD.lift_floor(seed)@basis.T
    required = COORD.lift_floor(floor['support_polygon_xy_m'])@basis.T
    margin = gap+.025*scale*expansion; width=.05*scale; height=.04*scale
    back = float(local[:,0].min()-margin)
    front = float(required[:,0].max()+margin)
    low, high = float(local[:,1].min()-margin), float(local[:,1].max()+margin)
    bounds = [([back-width,low-width,0.],[back,high+width,height]),
              ([back-width,low-width,0.],[front,low,height]),
              ([back-width,high,0.],[front,high+width,height])]
    parts = [G.box(np.array(a),np.array(b),basis) for a,b in bounds]
    if not all(scene.clear(p,ground=True,clearance=gap) for p in parts):return None
    polygons=[]
    for a,b in bounds:
        xy=np.array([[a[0],a[1],0.],[b[0],a[1],0.],[b[0],b[1],0.],[a[0],b[1],0.]])@basis
        polygons.append(COORD.floor(xy).tolist())
    ground,_=A.footprint(parts,floor['original_pivot_m'],floor['required_hull_xy_m'],scale)
    if not ground['passed']:return None
    return parts,dict(kind='directional_open_u',bearing_deg=angle,expansion=expansion,
        width_m=width,height_m=height,pads_xy_m=polygons,opening='relative object motion -withdrawal',
        seed_polygon_xy_m=seed.tolist(),actual_remaining_footprint_covers_demand=True)


def base_candidates(domain, floor, direction, gap, scene):
    seed = base_seed(domain.mesh, floor, direction, gap)
    horizontal = COORD.floor(direction)
    angle = float(np.rad2deg(np.arctan2(-horizontal[1], -horizontal[0]))) if np.linalg.norm(horizontal)>1e-8 else 0.
    for expansion in (1.,1.25,1.6):
        for turn in (0.,90.,-90.,180.):
            value=B.open_ring(domain.mesh,seed,floor['required_hull_xy_m'],floor['original_pivot_m'],
                              angle+turn,expansion,.3,scene)
            if value is not None:
                parts,base=value
                yield parts,dict(base,kind='directional_open_ring',seed_polygon_xy_m=seed.tolist())
        value=open_u(domain.mesh,floor,direction,gap,expansion,scene)
        if value is not None:yield value


def thick_links(terminals, base, scene, gap):
    radius=.024*scene.scale;end_radius=min(radius,.4*base['height_m'])
    choices=[]
    for first in terminals:
        for polygon in base['pads_xy_m']:
            material=Polygon(polygon);inset=material.buffer(-end_radius*.5)
            if inset.is_empty:inset=material
            xy=np.asarray(nearest_points(Point(COORD.floor(first)),inset)[1].coords[0])
            last=COORD.lift_floor(xy,base['height_m']*.5)
            choices.append((float(np.linalg.norm(last-first)),first,last))
    choices.sort(key=lambda row:row[0])
    for length,first,last in choices:
        part=T.beam(first,last,radius,end_radius)
        if scene.clear(part,clearance=gap):
            yield [part],dict(length_m=length,waypoints_m=[first.tolist(),last.tolist()],
                beam_width_m=2*radius,floor_joint_width_m=2*end_radius,
                method='shortest clear direct link within the current frame/base menu',global_shortest_claimed=False)


def straight_path(scene, parts, origin, direction):
    vertices=np.vstack([p.vertices for p in parts]);distance=exit_distance(scene,vertices,direction)
    amount=distance/scene.scale;motion=np.r_[direction,[0.,0.,0.]]
    budget=dict(remaining=500,checked=0)
    if not P.segment(scene,parts,origin,motion,0.,amount,budget):return None
    if not P.separated(scene,vertices+distance*direction):return None
    return dict(passed=True,status='rigid_trajectory_verified',continuous_sweep_verified=True,
        origin_m=np.asarray(origin).tolist(),length_scale_m=scene.scale,motion=motion.tolist(),
        final_withdrawal_amount=amount,segment_amounts=[0.,amount],checked_intervals=budget['checked'],
        method='exact convex translation sweep per primitive, replayed on the whole assembly',
        path_definition='support moves +s*D*d on withdrawal; insertion reverses s; object fixed',
        rotation_allowed=False,geometric_tolerance_m=scene.scale*1e-10,volume_tolerance_m3=1e-11*scene.scale**3)


def search(domain, contacts, points, depth, budget, progress=None, failure=None, direction_record=None):
    if budget<1:raise ValueError('Direction budget must be positive')
    report=dict(passed=False,geometry_constructed=False,step3_selection_changed=False,
        direction_first=True,contact_interfaces_changed=False,global_impossibility_claimed=False,
        belt_closed_loop_required=False,belt_attempts=[],attempts=[],direction_budget=min(budget,24),
        frame_gap_fraction=FRAME_GAP_FRACTION,frame_width_fraction=FRAME_WIDTH_FRACTION,
        construction_order=['direction','head_sweep','loose_frame','base','thick_links','whole_path'])
    directions,diagnostics=scheduled_directions(direction_record, report['direction_budget']) if direction_record is not None else propose_directions(domain.mesh,contacts,domain.work_ids,report['direction_budget'])
    report['direction_search']=diagnostics
    if not contacts:
        report['status']='no_selected_heads';return report,None
    heads,head_labels=make_heads(domain.mesh,contacts,depth)
    if progress:progress('heads',heads,head_labels)
    if not directions:
        report['status']='no_common_translation_proposed_for_fixed_heads';return report,None
    # Hulls/material are only designed after directions have been proposed.
    floor=FD.prepare(points);scale=float(domain.mesh.extents.max());gap=FRAME_GAP_FRACTION*scale
    for direction in directions:
        print('withdrawal candidate',np.round(direction,5).tolist(),flush=True)
        scene=SweptScene(domain.mesh,direction,gap)
        scene.failure_callback=failure
        scene.head_labels=head_labels
        scene.context_parts=heads;scene.context_labels=head_labels
        attempt=dict(direction=direction.tolist(),head_sweep_passed=False,frame_passed=False,base_candidates=0)
        report['attempts'].append(attempt)
        rejected=[i for i,head in enumerate(heads) if not scene.clear(head,clearance=0.)]
        if rejected:
            attempt.update(status='fixed_head_full_sweep_failed',failed_head_cells=rejected,
                witness_head_cell=rejected[0], witness_head_label=head_labels[rejected[0]],
                collision_witness=collision_witness(scene,heads[rejected[0]]));continue
        attempt['head_sweep_passed']=True
        frame=None
        for offset in (.04*scale,.12*scale,.25*scale):
            belt,frame=loose_frame(scene,heads,depth,gap,offset)
            report['belt_attempts'].append(dict(direction=direction.tolist(),**belt))
            if frame is not None:break
        if frame is None:
            attempt['status']=belt['status'];continue
        skin=heads+frame['necks']+frame['frame']
        labels=head_labels+[f'neck_{i:04d}' for i in range(len(frame['necks']))]+[f'frame_{i:04d}' for i in range(len(frame['frame']))]
        attempt['frame_passed']=True;report.update(belt=belt,withdrawal_direction=direction.tolist(),head_depth_m=depth)
        if progress:progress('belt',skin,labels)
        scene.context_stage='base';scene.context_parts=skin;scene.context_labels=labels
        for ring,base in base_candidates(domain,floor,direction,gap,scene):
            attempt['base_candidates']+=1
            scene.context_stage='thick_links'
            scene.context_parts=skin+ring
            scene.context_labels=labels+[f'ground_strip_{i:04d}' for i in range(len(ring))]
            for links,route in thick_links(frame['terminals'],base,scene,gap):
                parts=skin+links+ring
                names=labels+[f'connector_{i:04d}' for i in range(len(links))]+[f'ground_strip_{i:04d}' for i in range(len(ring))]
                joined,solid=B.union_parts(parts,scale)
                if not solid['one_solid']:
                    if failure:failure('whole_path',parts,names,dict(status='assembly_disconnected',withdrawal_direction=direction.tolist()))
                    continue
                ground,tri=A.footprint(parts,floor['original_pivot_m'],floor['required_hull_xy_m'],scale)
                if not ground['passed']:
                    if failure:failure('whole_path',parts,names,dict(status='actual_footprint_does_not_cover_demand',withdrawal_direction=direction.tolist()))
                    continue
                trajectory=straight_path(B.Scene(domain.mesh),parts,domain.com,direction)
                if trajectory is None:
                    if failure:failure('whole_path',parts,names,dict(status='whole_assembly_path_rejected',withdrawal_direction=direction.tolist()))
                    continue
                module=dict(parts=parts,labels=names,joined=joined,base=base,link=route,solid=solid,
                    ground=ground,floor_triangles=tri,trajectory=trajectory)
                report.update(passed=True,geometry_constructed=True,status='direction_first_frame_trajectory_verified',
                    base=base,link=route,solid=solid,ground=ground,trajectory=trajectory,
                    actual_frame_gap_m=gap,head_depth_m=depth,withdrawal_direction=direction.tolist())
                attempt.update(status='whole_assembly_sweep_verified',trajectory_verified=True)
                if progress:progress('assembly',parts,names)
                return report,module
            scene.context_stage='base';scene.context_parts=skin;scene.context_labels=labels
        attempt['status']='no_base_and_thick_link_in_directional_menu'
    report['status']=('fixed_heads_block_all_proposed_sweeps' if not any(a['head_sweep_passed'] for a in report['attempts'])
                      else 'no_loose_frame_assembly_in_directional_menu')
    return report,None
