"""Read-only illustration model: one contact module/interface, dedicated bases.

Saved poses 2/4 and an explicit 25-degree illustration derived from pose 2.
Source files are read-only. No physical solver runs.
"""
from pathlib import Path
import hashlib
import heapq
import json
import numpy as np
import trimesh
from shapely.geometry import MultiPoint, Point

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
import source_geometry as source
from geometry_utils import tube, resample
G = source.G
POSES = ('pose_2', 'pose_2_tilt_25deg', 'pose_4')
SCALE = .2
VIEW = np.array([1.05, -1.25, .8])


def solid(mesh):
    return G.solid64(mesh, np.zeros(3), SCALE)


def mesh_from_solid(value):
    if value.status() != G.manifold.Error.NoError:
        raise ValueError(value.status())
    data = value.to_mesh64()
    return trimesh.Trimesh(np.asarray(data.vert_properties[:, :3]) * SCALE,
                          np.asarray(data.tri_verts), process=False)


def joined(parts):
    value = G.manifold.Manifold.batch_boolean([solid(m) for m in parts], G.manifold.OpType.Add)
    # Coincident rod/joint surfaces can leave zero-volume Boolean fragments.
    # Discard only numerical dust below 1e-9 mm^3, not disconnected solid rods.
    components = [c for c in value.decompose() if abs(float(c.volume()))*SCALE**3 > 1e-18]
    return mesh_from_solid(G.manifold.Manifold.batch_boolean(components, G.manifold.OpType.Add))


def make_base(obj, channel, port, basis, forked=False):
    """Only the channel mating geometry is fixed; grounded members adapt."""
    x, y, direction = basis.T
    footprint = MultiPoint(np.vstack([obj.vertices[:, :2], port[None, :2]])).convex_hull.buffer(.023)
    xy = resample(footprint, 64)
    parts = tube(np.c_[xy, np.full(len(xy), .005)], .005, True) + [channel]
    mount = port - direction * .0315 - y * .004
    back = mount - direction * .020 + x * .025
    station = footprint.exterior.project(Point(back[:2]))
    for offset in ((-.035, .035) if forked else (0.,)):
        edge = footprint.exterior.interpolate((station + offset) % footprint.exterior.length)
        foot = np.array([edge.x, edge.y, .005])
        elbow = np.array([edge.x, edge.y, max(.018, back[2] - .030 if forked else back[2] - .010)])
        parts += tube(np.array([foot, elbow, back]), .0045)
    parts += tube(np.array([back, mount]), .0045)
    return joined(parts)


def illustrative_work_area(obj, blue, base, old_ids, contact_ids):
    """Reproducible random surface patch for the figure, not a load certificate."""
    seed = 20260920
    adjacency = [[] for _ in obj.faces]
    for a, b in obj.face_adjacency:
        adjacency[a].append(int(b))
        adjacency[b].append(int(a))
    forbidden = set(map(int, contact_ids))
    for _ in range(2):
        forbidden.update(j for i in list(forbidden) for j in adjacency[i])
    allowed = np.ones(len(obj.faces), bool)
    allowed[list(forbidden)] = False
    allowed[old_ids] = False
    view = VIEW / np.linalg.norm(VIEW)
    right = np.cross([0., 0., 1.], view); right /= np.linalg.norm(right)
    screen_up = np.cross(view, right)
    centers = obj.triangles_center
    height = (centers[:, 2] - obj.bounds[0, 2]) / obj.extents[2]
    allowed &= (obj.face_normals @ view > .15) & (height > .48)
    # Keep every vertex of the patch above the ENTIRE fixture, both
    # physically and in the illustration. Excluding contact IDs alone can
    # still leave working faces hidden behind the connecting frame.
    fixture_top = max(blue.bounds[1, 2], base.bounds[1, 2])
    fixture_screen_top = max(float((m.vertices @ screen_up).max()) for m in (blue, base))
    allowed &= obj.triangles[:, :, 2].min(axis=1) > fixture_top + .008
    allowed &= (obj.triangles @ screen_up).min(axis=1) > fixture_screen_top + .006
    # Screen face centers for self-occlusion by the object in the chosen view.
    eligible = np.flatnonzero(allowed)
    blocked = obj.ray.intersects_any(centers[eligible] + view*1e-6,
                                     np.tile(view, (len(eligible), 1)))
    allowed[eligible[blocked]] = False
    candidates = np.flatnonzero(allowed & (height > .55) & (height < .8))
    if not len(candidates):
        raise RuntimeError('No visible seed for the alternative working area')
    start = int(np.random.default_rng(seed).choice(candidates))
    target = float(obj.area_faces[old_ids].sum()) * .30
    queue = [(0., start)]; visited = set(); selected = []; area = 0.
    while queue and area < target:
        distance, face = heapq.heappop(queue)
        if face in visited:
            continue
        visited.add(face); selected.append(face); area += obj.area_faces[face]
        for other in adjacency[face]:
            if allowed[other] and other not in visited:
                heapq.heappush(queue, (distance + np.linalg.norm(centers[face]-centers[other]), other))
    ids = np.array(selected, int)
    assert len(ids) and not np.intersect1d(ids, old_ids).size
    assert not np.intersect1d(ids, contact_ids).size
    return ids, dict(kind='illustrative_random_connected_patch', random_seed=seed,
        seed_face=start, face_ids=ids.tolist(), area_mm2=float(area)*1e6,
        overlaps_original_work_faces=False, overlaps_contact_faces=False,
        vertical_gap_above_blue_and_base_mm=float(obj.triangles[ids, :, 2].min()-fixture_top)*1000,
        projected_gap_above_blue_and_base_mm=float((obj.triangles[ids] @ screen_up).min()-fixture_screen_top)*1000,
        face_center_view_rays_clear=True, tool_swept_volume_verified=False,
        contact_face_buffer_rings=2, task_loads_recomputed=False)


def build():
    meshes, meta = source.geometry()
    direction = meta['basis'][:, 2]
    first = meta['domain']
    interface = meta['channel']
    port = meta['port']
    slider_detail = meta['peg'].copy()
    with np.load(source.SOURCE / 'step3_scheculer/final_contacts.npz') as data:
        points = data['centers_m'].copy()
        contact_ids = data['source_faces'].copy()
        normals = []
        for a, b in zip(data['offsets'][:-1], data['offsets'][1:]):
            face_ids = data['source_faces'][a:b]
            normal = -np.sum(meshes['object'].face_normals[face_ids] * data['triangle_areas_m2'][a:b, None], axis=0)
            normals.append(normal / np.linalg.norm(normal))
    cases = []
    for pose in POSES:
        illustrative = pose == 'pose_2_tilt_25deg'
        source_pose = 'pose_2' if illustrative else pose
        path = source.SOURCE.parent / source_pose / 'step_1_needs/needs.json'
        data = json.loads(path.read_text())
        transform = np.array(data['frame']['T_world_mesh']) @ np.linalg.inv(first['frame']['T_world_mesh'])
        if illustrative:
            # Turn the object AND the unchanged blue module around world X.
            # Restore object floor height; this is a figure pose, not a saved task.
            pivot = meshes['object'].bounds.mean(0)
            transform = trimesh.transformations.rotation_matrix(np.deg2rad(25), [1, 0, 0], point=pivot)
            turned = meshes['object'].copy().apply_transform(transform)
            transform[2, 3] -= turned.bounds[0, 2]
        r, t = transform[:3, :3], transform[:3, 3]
        obj = meshes['object'].copy().apply_transform(transform)
        contact = meshes['contact'].copy().apply_transform(transform)
        channel = interface.copy().apply_transform(transform)
        basis = r @ meta['basis']
        q = r @ port + t
        if not illustrative:
            np.testing.assert_allclose(obj.vertices, data['geometry']['vertices_m'], atol=1e-10)
        np.testing.assert_allclose(contact.vertices, meshes['contact'].vertices @ r.T + t, atol=1e-12)
        np.testing.assert_allclose(channel.vertices, interface.vertices @ r.T + t, atol=1e-12)
        work_ids = np.array(data['geometry']['work_face_ids'], int)
        base = make_base(obj, channel, q, basis, forked=illustrative)
        work_area = dict(kind='saved_task_working_area', face_ids=work_ids.tolist(),
                         area_mm2=float(obj.area_faces[work_ids].sum())*1e6)
        if illustrative:
            work_ids, work_area = illustrative_work_area(obj, contact, base, work_ids, contact_ids)
        cases.append(dict(pose=pose, label='B / pose 2 + 25° tilt' if illustrative else 'B / '+pose.replace('_',' '),
            pose_kind='illustrative_tilt' if illustrative else 'saved_task_pose',
            transform=transform, object=obj, contact=contact,
            base=base,
            base_layout='two_ground_branches' if illustrative else 'single_ground_branch',
            channel=channel, port=q, basis=basis,
            direction=r@direction, work_ids=work_ids, work_area=work_area,
            slider_detail=slider_detail.copy().apply_transform(transform),
            points=points@r.T+t, normals=np.array(normals)@r.T,
            source=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return cases, meta


def screen_initial_installation(case):
    """Move only the blue module against the displayed pose-2 object/base."""
    fixed_obj, fixed_base, moving = [solid(case[k]) for k in ('object', 'base', 'contact')]
    checks = []
    for amount in np.unique(np.r_[np.linspace(0, .04, 81), np.linspace(.05, .20, 16)]):
        displaced = moving.translate((case['direction']*amount/SCALE).tolist())
        checks.append(dict(offset_m=float(amount),
            object_overlap_mm3=abs(float((displaced^fixed_obj).volume()))*SCALE**3*1e9,
            base_overlap_mm3=abs(float((displaced^fixed_base).volume()))*SCALE**3*1e9))
    if max(max(c['object_overlap_mm3'], c['base_overlap_mm3']) for c in checks) > .01:
        raise RuntimeError('An initial module installation sample intersects the object or base')
    return dict(scope='97 static module positions against the displayed pose-2 base; no full robot collision or continuous-path certificate', samples=checks)


def verify(cases, meta):
    records = []
    tolerance = .01
    for c in cases:
        obj, contact, base = [solid(c[k]) for k in ('object', 'contact', 'base')]
        def overlap(delta):
            shift = (delta / SCALE).tolist()
            return dict(object_base_overlap_mm3=abs(float((obj.translate(shift)^base).volume()))*SCALE**3*1e9,
                contact_base_overlap_mm3=abs(float((contact.translate(shift)^base).volume()))*SCALE**3*1e9)
        def maximum(samples):
            return max(max(s['object_base_overlap_mm3'],s['contact_base_overlap_mm3']) for s in samples)
        local = [dict(offset_mm=float(a*1000), **overlap(c['direction']*a)) for a in np.linspace(0,.03,61)]
        extended = [dict(offset_mm=float(a*1000), **overlap(c['direction']*a)) for a in np.linspace(0,.2,101)]
        # A finite direction search after full interface release; this is not robot planning.
        candidates = [np.array([0.,0.,1.])]
        for z in (.8,.5,.2):
            for angle in np.linspace(0,2*np.pi,16,endpoint=False):
                candidates.append(np.array([np.cos(angle)*np.sqrt(1-z*z),np.sin(angle)*np.sqrt(1-z*z),z]))
        departure = None
        for vector in candidates:
            samples = []
            for a in np.linspace(0,.2,101):
                sample = dict(offset_mm=float(a*1000), **overlap(c['direction']*.03+vector*a))
                samples.append(sample)
                if max(sample['object_base_overlap_mm3'],sample['contact_base_overlap_mm3']) > tolerance:
                    break
            if len(samples)==101 and maximum(samples)<tolerance:
                departure = dict(direction=vector.tolist(), samples=samples,
                                 max_sampled_overlap_mm3=maximum(samples))
                break
        # At 30 mm withdrawal, the whole rectangular blue peg is beyond
        # the channel along the docking axis: a conservative separating plane.
        blue_projection=(c['slider_detail'].vertices+c['direction']*.03)@c['direction']
        orange_projection=c['channel'].vertices@c['direction']
        interface_gap=float(blue_projection.min()-orange_projection.max())*1000
        nsolids = len(base.decompose())
        if nsolids != 1:
            raise RuntimeError(f"{c['pose']}: base has {nsolids} disconnected solids")
        records.append(dict(pose=c['pose'], pose_kind=c['pose_kind'], source=c['source'], source_sha256=c['source_sha256'],
            working_area=c['work_area'],
            transform=c['transform'].tolist(), withdrawal_direction=c['direction'].tolist(),
            insertion_direction=(-c['direction']).tolist(), interface_port_m=c['port'].tolist(),
            rigid_contact_reuse=True, rigid_channel_reuse=True, base_layout=c['base_layout'], base_connected_components=nsolids,
            contact_min_z_mm=float(c['contact'].bounds[0,2]*1000),
            interface_axial_gap_at_30mm_mm=interface_gap,
            local_docking=dict(range_mm=[0,30],sample_count=61,max_sampled_overlap_mm3=maximum(local),
                clear=bool(maximum(local)<tolerance),samples=local),
            extended_straight_withdrawal=dict(range_mm=[0,200],sample_count=101,
                max_sampled_overlap_mm3=maximum(extended),clear=bool(maximum(extended)<tolerance),samples=extended),
            post_release_departure=departure))
        print(c['pose'], 'local docking overlap mm3', maximum(local), 'interface gap mm', interface_gap,
              'extended straight overlap mm3', maximum(extended),
              'departure',None if departure is None else departure['direction'],flush=True)
    shared_base = all(c['base'] is cases[0]['base'] for c in cases)
    return dict(status='geometry illustration; shared contact module/interface; '+('one fixed base with three sockets' if shared_base else 'pose-specific bases'),
        geometry_scope='Historical B poses 2/4 and an illustrative 25-degree world-X tilt of pose 2, lowered to object floor height. Tilt has a distinct reproducibly random working patch excluding original work/contact faces; task loads are not recomputed. Object and blue module move together during local 30 mm docking. Subsequent transport changes direction.',
        contact_mesh_sha256=hashlib.sha256(cases[0]['contact'].vertices.tobytes()+cases[0]['contact'].faces.tobytes()).hexdigest(),
        same_interface_in_contact_coordinates=True, same_world_base=shared_base,
        base_mesh_sha256=hashlib.sha256(cases[0]['base'].vertices.tobytes()+cases[0]['base'].faces.tobytes()).hexdigest(),
        overlap_tolerance_mm3=tolerance,
        source_sha256=meta['source_sha256'],
        source_module_relative_to_object_insertion_screening=screen_initial_installation(cases[0]),
        interface=meta['interface'],
        all_local_docking_samples_clear=all(r['local_docking']['clear'] for r in records),
        all_interfaces_fully_released_at_30mm=all(r['interface_axial_gap_at_30mm_mm']>0 for r in records),
        all_post_release_departure_samples_clear=all(r['post_release_departure'] is not None for r in records),
        extended_straight_withdrawal_warning='Continuing along the docking axis beyond release collides with the workpiece; it is not the transport path.',
        physics_tested=False, tool_sweep_tested=False, robot_grasp_tested=False,
        module_retention_during_transport_verified=False, between_pose_rotation_tested=False,
        continuous_collision_certificate=False, records=records)
