"""Search a parallel-jaw grasp; verify with a free object and a driven hand.

Writes only an optional MP4. Candidate transforms and checks go to stdout.
The hand root is prescribed; this does not certify an arm trajectory.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = HERE / 'assets/franka_hand'


def numbers(x):
    return ' '.join(format(float(v), '.17g') for v in np.asarray(x).ravel())


def quaternion(rotation):
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


def build(name, object_xml=None, exact=False, finger_boxes=None, hand_xml=None, render_arm=False,
          floor_hull=False, robot_collisions=False):
    root = ET.parse(hand_xml or ASSETS / 'hand.xml').getroot()
    root.find('compiler').set('meshdir', str(ASSETS / 'assets'))
    root.find('option').attrib.update(timestep='.001', gravity='0 0 -9.81',
        integrator='implicitfast', cone='elliptic', iterations='80', tolerance='1e-9')
    ET.SubElement(root.find('default'), 'geom', friction='.8 .003 .0001', condim='3',
        solref='.006 1', solimp='.95 .99 .001')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='800')
    ET.SubElement(visual, 'headlight', ambient='.4 .4 .4', diffuse='.6 .6 .6')
    world = root.find('worldbody')
    hand = world.find('body')
    hand.attrib.update(quat='1 0 0 0')
    ET.SubElement(hand, 'freejoint', name='hand_free')
    ET.SubElement(world, 'body', name='hand_driver', mocap='true')
    ET.SubElement(root.find('equality'), 'weld', name='hand_drive',
        body1='hand_driver', body2='hand', solref='.004 1', solimp='.99 .999 .001')
    if finger_boxes is not None:
        for label, boxes in finger_boxes.items():
            finger = hand.find(f"body[@name='{label}']")
            for geom in list(finger.findall('geom')):
                finger.remove(geom)
            finger.find('inertial').attrib.update(mass='.10', pos='0 .04 .05',
                diaginertia='.0002 .0002 .0002')
            for center, size in boxes:
                ET.SubElement(finger, 'geom', type='box', pos=numbers(center), size=numbers(size),
                    group='2', rgba='.20 .38 .55 1', contype='1', conaffinity='6')
    for i, geom in enumerate(hand.iter('geom')):
        geom.set('name', f'hand_geom_{i}')
        if geom.get('class') != 'visual':
            geom.attrib.update(contype='1', conaffinity='6')
    for joint in hand.iter('joint'):
        joint.attrib.update(solreflimit='.004 1', solimplimit='.99 .999 .001')
    ET.SubElement(world, 'light', pos='0 -.5 1', directional='true', dir='0 .4 -1')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='1 1 .01',
        rgba='.9 .91 .92 1', contype='4', conaffinity='3')
    scene = ROOT / 'objects' / '_simulation_assets' / name / 'scene.xml'
    if not scene.exists():
        scene = ROOT / 'objects' / name / 'scene.xml'
    source_dir = ROOT / 'objects' / name if object_xml else scene.parent
    source = ET.fromstring(object_xml) if object_xml else ET.parse(scene).getroot()
    for mesh in source.find('asset').findall('mesh'):
        mesh.set('file', str((source_dir / mesh.get('file')).resolve()))
        mesh.set('name', 'object_' + mesh.get('name'))
        root.find('asset').append(mesh)
    obj = source.find('worldbody/body')
    obj.set('name', 'object')
    for geom in obj.findall('geom'):
        if geom.get('mesh'):
            geom.set('mesh', 'object_' + geom.get('mesh'))
        geom.attrib.pop('material', None)
        geom.set('name', 'object_' + geom.get('name'))
        if geom.get('group') == '2':
            geom.set('rgba', '.20 .27 .34 1')
        else:
            geom.attrib.update(contype='2', conaffinity='5', condim='3', friction='.8 .003 .0001')
    world.append(obj)
    if exact and object_xml is None:
        import tetgen
        raw = trimesh.load(ROOT / 'objects' / name / 'mesh.stl', force='mesh')
        nodes, cells, *_ = tetgen.TetGen(raw.vertices, raw.faces).tetrahedralize(switches='pYQ')
        volumes = np.abs(np.linalg.det(nodes[cells[:,1:]]-nodes[cells[:,:1]]))/6
        np.testing.assert_allclose(volumes.sum(), raw.volume, rtol=1e-10)
        faces = cells[:, [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]].reshape(-1,3)
        boundary, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
        assert set(map(tuple, boundary[counts == 1])) == set(map(tuple, np.sort(raw.faces, axis=1)))
        for geom in list(obj.findall('geom')):
            if geom.get('group') == '3':
                obj.remove(geom)
        for i, cell in enumerate(cells):
            ET.SubElement(root.find('asset'), 'mesh', name=f'exact_{i}', vertex=numbers(nodes[cell]))
            ET.SubElement(obj, 'geom', name=f'object_exact_{i}', type='mesh', mesh=f'exact_{i}',
                group='3', contype='2', conaffinity='5', condim='3', friction='.8 .003 .0001')
    if floor_hull:
        # A convex hull has exactly the raw mesh's support function against a
        # plane. Keep concave finger collisions separate from floor collisions.
        raw = trimesh.load(ROOT/'objects'/name/'mesh.stl',force='mesh')
        hull = raw.convex_hull
        for geom in obj.findall('geom'):
            if geom.get('contype') == '2':
                geom.set('conaffinity','1')
        world.find("geom[@name='floor']").set('conaffinity','9')
        ET.SubElement(root.find('asset'),'mesh',name='object_floor_hull_mesh',
                      vertex=numbers(hull.vertices),face=' '.join(map(str,hull.faces.ravel())))
        ET.SubElement(obj,'geom',name='object_floor_hull',type='mesh',mesh='object_floor_hull_mesh',
                      group='3',contype='8',conaffinity='4',condim='1',priority='2',
                      solref='.002 1',solimp='.99 .999 .001',friction='0 0 0')
    actuator = root.find('actuator/general')
    actuator.attrib.update(gainprm='1 0 0', biasprm='0 0 0', ctrlrange='-70 70', forcerange='-70 70')
    if render_arm:
        from kuka_transfer import add_render_arm
        add_render_arm(root,collisions=robot_collisions)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    return model


def object_pose(model, data, transform):
    address = model.jnt_qposadr[model.body_jntadr[model.body('object').id]]
    data.qpos[address:address+3] = transform[:3, 3]
    data.qpos[address+3:address+7] = quaternion(transform[:3, :3])


def hand_pose(data, transform, initialize=False):
    data.mocap_pos[0] = transform[:3, 3]
    data.mocap_quat[0] = quaternion(transform[:3, :3])
    if initialize:
        address = data.model.joint('hand_free').qposadr[0]
        data.qpos[address:address+3] = transform[:3, 3]
        data.qpos[address+3:address+7] = quaternion(transform[:3, :3])


def transform(data, body):
    T = np.eye(4)
    T[:3, :3] = data.xmat[body].reshape(3, 3)
    T[:3, 3] = data.xpos[body]
    return T


def candidates(mesh, initial, limit=300, max_width=.080, depths=(.1029, .1084), min_width=.015):
    sampled, faces = trimesh.sample.sample_surface(mesh, 1800, seed=20260918)
    center_ids = np.arange(len(mesh.faces))
    if len(center_ids) > 1200:
        center_ids = np.random.default_rng(20260918).choice(center_ids,1200,replace=False,
                                                          p=mesh.area_faces/mesh.area)
    points = np.concatenate([mesh.triangles_center[center_ids], sampled])
    face_ids = np.concatenate([center_ids, faces])
    centers = trimesh.transform_points(points, initial)
    normals = mesh.face_normals[face_ids] @ initial[:3, :3].T
    com = trimesh.transform_points(mesh.center_mass[None], initial)[0]
    delta = centers[None] - centers[:, None]
    width = np.linalg.norm(delta, axis=2)
    direction = delta / np.maximum(width[..., None], 1e-12)
    valid = (width > min_width) & (width < max_width)
    valid &= np.einsum('ijk,ik->ij', direction, normals) < -.90
    valid &= np.einsum('ijk,jk->ij', direction, normals) > .90
    valid &= np.triu(np.ones_like(valid), 1)
    a, b = np.nonzero(valid)
    score = np.linalg.norm((centers[a]+centers[b])/2-com, axis=1)
    seen = set()
    count = 0
    for idx in np.argsort(score):
        i, j = a[idx], b[idx]
        middle = (centers[i]+centers[j])/2
        closing = direction[i, j]
        key = tuple(np.round(middle/.004).astype(int)) + tuple(np.round(closing/.12).astype(int))
        if key in seen:
            continue
        seen.add(key)
        base = np.array([0., 0., -1.])
        base -= closing * (base @ closing)
        if np.linalg.norm(base) < .1:
            continue
        base /= np.linalg.norm(base)
        for roll in (0, -15, 15, -30, 30, -45, 45, -60, 60, -75, 75, -90, 90):
            approach = Rotation.from_rotvec(closing*np.radians(roll)).apply(base)
            R = np.column_stack([np.cross(closing, approach), closing, approach])
            for depth in depths:
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = R, middle - R @ [0, 0, depth]
                yield dict(hand=T, width=float(width[i, j]), contacts=centers[[i,j]],
                    opening=min(max_width/2+.005,float(width[i,j])/2+.012),
                    face_ids=[int(face_ids[i]), int(face_ids[j])], roll_deg=roll,
                    com_distance_m=float(score[idx]))
        count += 1
        if count >= limit:
            return


def geometry_check(model, initial, candidate):
    data = mujoco.MjData(model)
    object_pose(model, data, initial)
    fingers = [model.joint(f'finger_joint{i}').qposadr[0] for i in (1, 2)]
    T = candidate['hand'].copy()
    lower,upper=model.jnt_range[model.joint('finger_joint1').id]
    target = float(np.clip((candidate['width']-.003)/2,lower,upper))
    # Approach fully open, then close. Reject any contact except distal pads
    # during closure; reject all hand/object or hand/floor penetration on approach.
    opening_max = min(model.jnt_range[model.joint('finger_joint1').id, 1],
                      candidate.get('opening',np.inf))
    stages = [(s, opening_max, False) for s in np.linspace(.08, 0, 17)]
    stages += [(0., q, True) for q in np.linspace(opening_max, target, 13)]
    for offset, opening, closing in stages:
        P = T.copy(); P[:3, 3] -= offset*T[:3, 2]
        hand_pose(data, P, initialize=True); data.qpos[fingers] = opening
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            ids = [int(contact.geom1), int(contact.geom2)]
            h = [g for g in ids if model.geom(g).name.startswith('hand_geom_')]
            if not h or contact.dist > -1e-5:
                continue
            other = ids[1] if ids[0] == h[0] else ids[0]
            # Pad boxes are intended contact; finger shell and palm are not.
            pad = (model.geom_type[h[0]] == mujoco.mjtGeom.mjGEOM_BOX
                   and model.body(model.geom_bodyid[h[0]]).name in ('left_finger', 'right_finger'))
            if not (closing and pad and model.geom(other).name.startswith('object_')):
                return False
    return True


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3-2*x)


def simulate(model, initial, candidate, capture=False, turn_axis=None, turn_angles=(40,50,60)):
    data = mujoco.MjData(model)
    object_pose(model, data, initial)
    fingers = [model.joint(f'finger_joint{i}').qposadr[0] for i in (1, 2)]
    speeds = [model.joint(f'finger_joint{i}').dofadr[0] for i in (1, 2)]
    opening_max = min(model.jnt_range[model.joint('finger_joint1').id, 1],
                      candidate.get('opening',np.inf))
    data.qpos[fingers] = opening_max
    T = candidate['hand'].copy()
    P = T.copy(); P[:3,3] -= .08*T[:3,2]
    hand_pose(data, P, initialize=True)
    mujoco.mj_forward(model, data)
    body = model.body('object').id
    history, reference, carried_reference = [], None, None
    worst_position, worst_angle = 0., 0.
    carry_position, carry_angle = 0., 0.
    pickup_lift = 0.
    if turn_axis is not None:
        turn_axis = np.asarray(turn_axis, float)
        turn_axis /= np.linalg.norm(turn_axis)
        if len(turn_angles) != 3 or not np.isfinite(turn_angles).all():
            raise ValueError('Three finite turn angles are required')
    pivot = initial[:3,:3] @ model.body_ipos[body] + initial[:3,3] + [.03,0,.08]
    checkpoints = []
    for step in range(22000 if turn_axis is not None else 12000):
        t = step*.001
        P = T.copy()
        P[:3,3] -= .08*(1-smooth(t/1.5))*T[:3,2]
        P[2,3] += .08*smooth((t-3.)/2.)
        P[0,3] += .03*smooth((t-8.)/2.)
        if turn_axis is not None and t >= 12.:
            angle = (turn_angles[0]*smooth((t-12.)/2.)
                     + (turn_angles[1]-turn_angles[0])*smooth((t-15.)/2.)
                     + (turn_angles[2]-turn_angles[1])*smooth((t-18.)/2.))
            rotation = Rotation.from_rotvec(turn_axis*np.radians(angle)).as_matrix()
            P[:3,:3] = rotation @ P[:3,:3]
            P[:3,3] = pivot + rotation @ (P[:3,3]-pivot)
        hand_pose(data, P)
        if t < 1.5:
            data.ctrl[0] = np.clip(3000*(opening_max-np.mean(data.qpos[fingers]))-20*np.mean(data.qvel[speeds]), -70, 70)
        else:
            data.ctrl[0] = -70*smooth((t-1.5)/.7)
        mujoco.mj_step(model, data)
        if step == 2999:
            mujoco.mj_forward(model, data)
            reference = np.linalg.inv(transform(data, model.body('hand').id)) @ transform(data, body)
        if step >= 3000 and step % 20 == 0:
            mujoco.mj_forward(model, data)
            relative = np.linalg.inv(transform(data, model.body('hand').id)) @ transform(data, body)
            worst_position = max(worst_position, float(np.linalg.norm(relative[:3,3]-reference[:3,3])))
            worst_angle = max(worst_angle, float(np.degrees(Rotation.from_matrix(relative[:3,:3]@reference[:3,:3].T).magnitude())))
            if worst_position > .15 or worst_angle > 120:
                return dict(passed=False, slip_m=worst_position, rotation_deg=worst_angle), history
        if step == 7999:
            mujoco.mj_forward(model, data)
            carried_reference = np.linalg.inv(transform(data, model.body('hand').id)) @ transform(data, body)
            pickup_lift = float(data.xpos[body,2]-initial[2,3])
        if step >= 8000 and step % 20 == 0:
            mujoco.mj_forward(model, data)
            relative = np.linalg.inv(transform(data, model.body('hand').id)) @ transform(data, body)
            carry_position = max(carry_position, float(np.linalg.norm(relative[:3,3]-carried_reference[:3,3])))
            carry_angle = max(carry_angle, float(np.degrees(Rotation.from_matrix(relative[:3,:3]@carried_reference[:3,:3].T).magnitude())))
            if turn_axis is not None and step in (14980,17980,20980):
                checkpoints.append(dict(angle_deg=dict(zip((14980,17980,20980),turn_angles))[step],
                    slip_m=float(np.linalg.norm(relative[:3,3]-carried_reference[:3,3])),
                    rotation_deg=float(np.degrees(Rotation.from_matrix(relative[:3,:3]@carried_reference[:3,:3].T).magnitude()))))
        if capture and step % 33 == 0:
            history.append((data.qpos.copy(), data.mocap_pos.copy(), data.mocap_quat.copy()))
    mujoco.mj_forward(model, data)
    touching_fingers = set()
    floor_touch = False
    for contact in data.contact:
        if contact.dist > .0001:
            continue
        ids = [int(contact.geom1), int(contact.geom2)]
        if not any(model.geom(g).name.startswith('object_') for g in ids):
            continue
        for gid in ids:
            bid = model.geom_bodyid[gid]
            if model.body(bid).name in ('left_finger', 'right_finger'):
                touching_fingers.add(model.body(bid).name)
            floor_touch |= model.geom(gid).name == 'floor'
    lift = float(data.xpos[body,2]-initial[2,3])
    result = dict(passed=carry_position < .003 and carry_angle < 3. and pickup_lift > .06
        and not floor_touch and len(touching_fingers) == 2,
        slip_m=worst_position, rotation_deg=worst_angle,
        carried_slip_m=carry_position, carried_rotation_deg=carry_angle,
        bilateral_contact=len(touching_fingers)==2, object_floor_contact=bool(floor_touch),
        finger_positions_m=data.qpos[fingers].tolist(),
        object_vertical_displacement_m=lift,
        pickup_vertical_displacement_m=pickup_lift, turn_checkpoints=checkpoints,
        settled_object_in_hand=carried_reference.tolist())
    return result, history


def seated_targets(model, mesh, candidate, checks, angles):
    """Nominal ground-seated proposals; no dynamic placement or support certificate."""
    relative = np.asarray(checks['settled_object_in_hand'])
    rows = []
    for angle in angles:
        hand = candidate['hand'].copy()
        hand[:3,:3] = Rotation.from_euler('y',angle,degrees=True).as_matrix()@hand[:3,:3]
        obj = hand@relative
        minimum = trimesh.transform_points(mesh.vertices,obj)[:,2].min()
        obj[2,3] -= minimum; hand[2,3] -= minimum
        vertices = trimesh.transform_points(mesh.vertices,obj)
        contacts = vertices[vertices[:,2] < 1e-8]
        com = trimesh.transform_points(mesh.center_mass[None],obj)[0]
        clear = geometry_check(model,obj,dict(hand=hand,
            width=2*np.mean(checks['finger_positions_m'])))
        rows.append(dict(angle_deg=angle,T_world_mesh=obj.tolist(),T_world_hand=hand.tolist(),
            sampled_hand_approach_open_close_clear=bool(clear),ground_vertex_count=len(contacts),
            com_distance_to_first_ground_vertex_m=float(np.linalg.norm(com[:2]-contacts[0,:2])),
            scope='Nominal seated geometry only; not dynamic placement or a baseline support certificate'))
    return rows


def render(model, history, output):
    sys.path.insert(0, str(ROOT / 'slides/baseline_algo'))
    from step4_connect_support.video import mp4_writer
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0., 0., .12]
    camera.distance, camera.azimuth, camera.elevation = .65, 130, -25
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cadgrasp-grasp-') as temp:
        path = Path(temp) / output.name
        with mujoco.Renderer(model, height=800, width=1280) as renderer, mp4_writer(path, 1000/33) as writer:
            for qpos, pos, quat in history:
                data.qpos[:] = qpos; data.mocap_pos[:] = pos; data.mocap_quat[:] = quat
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                writer.append_data(renderer.render())
        shutil.move(str(path), output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    parser.add_argument('--placement', type=int, default=4)
    parser.add_argument('--pairs', type=int, default=1000)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--exact', action='store_true', help='Use boundary-preserving tetrahedra for object collisions')
    parser.add_argument('--tool', choices=('parallel', 'franka'), default='parallel',
                        help='Generic straight jaws (default), or historical Franka Hand experiment')
    parser.add_argument('--turn', action='store_true', help='Also test reorientation about world Y')
    parser.add_argument('--turn-angles', type=float, nargs=3, default=(10,20,30))
    parser.add_argument('--kuka', action='store_true', help='Check sampled full-pose KUKA IK for a passing grasp')
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error('--pairs must be positive')
    if args.video and args.video.suffix.lower() != '.mp4':
        parser.error('--video must be an MP4')
    folder = ROOT / 'objects' / args.object
    mesh = trimesh.load(folder/'mesh.stl', force='mesh')
    saved = json.loads((folder/'poses.json').read_text())
    if saved.get('schema') == 'cadgrasp_sequence_v1':
        if args.placement != 0:
            parser.error('Current data stores one rest pose; use --placement 0 for it, or sequence.py for the ten-target sequence')
        placements = [dict(index=0, T_world_mesh=saved['rest']['T_world_mesh'])]
    else:
        placements = saved['poses']
    initial = np.array(next(p['T_world_mesh'] for p in placements if p['index'] == args.placement))
    initial[2,3] -= trimesh.transform_points(mesh.vertices, initial)[:,2].min()
    model = build(args.object, exact=args.exact,
                  hand_xml=HERE/'assets/parallel_jaw.xml' if args.tool == 'parallel' else None)
    widths = dict(max_width=.15, depths=(.08, .10)) if args.tool == 'parallel' else {}
    axis = [0., 1., 0.] if args.turn else None
    tested = 0
    processed = 0
    for index, candidate in enumerate(candidates(mesh, initial, args.pairs, **widths)):
        processed += 1
        if not geometry_check(model, initial, candidate):
            continue
        tested += 1
        result, history = simulate(model, initial, candidate, capture=args.kuka or bool(args.video),
                                   turn_axis=axis, turn_angles=args.turn_angles)
        print(json.dumps(dict(candidate=index, **{k:v for k,v in result.items()
            if k != 'settled_object_in_hand'})), flush=True)
        if not result['passed']:
            continue
        arm_check = None
        if args.kuka:
            from kuka_transfer import check
            try:
                arm_check = check(history)
            except ValueError as error:
                print(json.dumps(dict(candidate=index, kuka_passed=False, reason=str(error))),flush=True)
                continue
        report = dict(object=args.object, tool=args.tool, placement=args.placement, candidate=index,
            kuka_checks=arm_check,
            proposed_seated_targets=seated_targets(model,mesh,candidate,result,args.turn_angles) if args.turn else [],
            hand_world=candidate['hand'].tolist(), hand_in_object=(np.linalg.inv(initial)@candidate['hand']).tolist(),
            contacts_world_m=candidate['contacts'].tolist(), face_ids=candidate['face_ids'],
            jaw_width_m=candidate['width'], checks=result,
            search=dict(candidates=processed, dynamic_trials=tested, seed=20260918,
                exact_object_collision=args.exact),
            scope='Free-object pickup and carry, optional in-air reorientation; '
                  'KUKA checks only as explicitly reported; seating, full collisions and fixture interference unchecked')
        print(json.dumps(report, indent=2), flush=True)
        if args.video:
            if args.kuka:
                from kuka_transfer import render as render_kuka
                render_kuka(model, history, args.video, args.object, args.tool)
            else:
                render(model, history, args.video)
            print(args.video, flush=True)
        return
    print(json.dumps(dict(object=args.object, placement=args.placement, candidates=processed,
        dynamic_trials=tested, passed=0, exact_object_collision=args.exact)), flush=True)
    raise SystemExit('No passing grasp in sampled menu; this is not an impossibility certificate')


if __name__ == '__main__':
    main()
