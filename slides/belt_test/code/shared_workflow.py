"""Kinematic KUKA concept film: load the blue module and object separately at every dock.

Reuses codes/simulation's robot model, IK, pose helpers and video encoding.
This is a geometry/policy illustration, not a dynamics or grasp certificate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
import trimesh

import shared_base
import fixture_geometry as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'codes/simulation'))
import kuka
from workflow import smooth, mix, set_pose
from video import project, annotate
from step5_connect_support.video import mp4_writer

OUT = shared_base.OUT
ROBOT_BASE = np.array([.69, .32, 0.])
WIDTH, HEIGHT = 1600, 900
FPS = 24
VIDEO_DURATION = 36.  # Presentation playback; the reference trajectory is slower.


def translated(T, delta):
    result = T.copy()
    result[:3, 3] += delta
    return result


def blend_pose(A, B, fraction):
    u = smooth(fraction)
    T = np.eye(4)
    T[:3, :3] = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))([u]).as_matrix()[0]
    T[:3, 3] = mix(A[:3, 3], B[:3, 3], u)
    return T


from separate_loading import Sequence
from separate_scene import ground_balance


def model_for(sequence):
    root = ET.Element('mujoco', model='shared_base_kinematic_concept')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true', inertiafromgeom='false')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(WIDTH), offheight=str(HEIGHT))
    ET.SubElement(visual, 'headlight', ambient='.30 .30 .30', diffuse='.40 .40 .40', specular='.05 .05 .05')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', type='skybox', builtin='gradient', rgb1='1 1 1', rgb2='1 1 1', width='512', height='512')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -.4 1.5', dir='0 .2 -1', directional='true', diffuse='.45 .45 .45', ambient='.08 .08 .08')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='1.2 1.2 .01', rgba='.94 .95 .96 1', contype='0', conaffinity='0')
    def geom(parent, name, mesh, rgba):
        ET.SubElement(asset, 'mesh', name=name+'_mesh', vertex=kuka.numbers(mesh.vertices), face=' '.join(map(str, mesh.faces.ravel())))
        ET.SubElement(parent, 'geom', name=name, type='mesh', mesh=name+'_mesh', rgba=rgba, contype='0', conaffinity='0')
    geom(world, 'shared_base', sequence.base, '.92 .60 .18 1')
    for name, mesh, color in (('object', sequence.obj, '.65 .68 .70 1'), ('contact', sequence.blue, '.145 .44 .74 1')):
        body = ET.SubElement(world, 'body', name=name)
        ET.SubElement(body, 'freejoint', name=name+'_free')
        ET.SubElement(body, 'inertial', pos='0 0 0', mass='.25', diaginertia='.001 .001 .001')
        geom(body, name+'_visual', mesh, color)
        if name == 'object':
            for i, case in enumerate(sequence.cases):
                patch = trimesh.Trimesh(sequence.obj.vertices, sequence.obj.faces[case['work_ids']], process=False)
                patch.remove_unreferenced_vertices()
                patch.vertices += patch.vertex_normals*3e-5
                geom(body, f'work_{i}', patch, '.25 .73 .40 0')
    # Reuse upstream mesh assets and seven-axis chain; retain one grasping arm.
    kuka.add_robots(root)
    world.remove(world.find("body[@name='robot_A_link_0']"))
    world.find("body[@name='robot_B_link_0']").set('pos', kuka.numbers(ROBOT_BASE))
    # Replace the upstream detached finger illustration locally. A fixed palm
    # spans both sliding carriages; each constant-length finger stays attached
    # throughout its stroke. The fingertips end at the existing 135 mm TCP.
    wrist = world.find(".//body[@name='robot_B_link_7']")
    for name in ('robot_B_tool_stem', 'robot_B_finger_-1', 'robot_B_finger_1'):
        wrist.remove(wrist.find(f"geom[@name='{name}']"))
    def tool_geom(parent, name, kind, pos, size, color):
        ET.SubElement(parent, 'geom', name=name, type=kind, pos=pos, size=size,
                      rgba=color, contype='0', conaffinity='0', group='2')
    tool_geom(wrist, 'gripper_flange', 'cylinder', '0 0 .047', '.027 .012', '.30 .33 .36 1')
    tool_geom(wrist, 'gripper_palm', 'box', '0 0 .066', '.082 .022 .014', '.32 .36 .40 1')
    for side in (-1, 1):
        jaw = ET.SubElement(wrist, 'body', name=f'gripper_jaw_{side}', pos=f'{side*.009} 0 0')
        ET.SubElement(jaw, 'joint', name=f'gripper_slide_{side}', type='slide',
                      axis=f'{side} 0 0', limited='true', range='0 .061')
        ET.SubElement(jaw, 'inertial', pos='0 0 .10', mass='.06', diaginertia='.00002 .00002 .00002')
        tool_geom(jaw, f'gripper_carriage_{side}', 'box', '0 0 .077', '.008 .018 .009', '.46 .49 .52 1')
        tool_geom(jaw, f'gripper_finger_{side}', 'box', '0 0 .106', '.004 .010 .029', '.22 .25 .28 1')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


class Display:
    def __init__(self, sequence):
        self.sequence = sequence
        self.model = model_for(sequence)
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(self.camera)
        self.camera.azimuth = 110; self.camera.elevation = -28
        self.addresses = [self.model.jnt_qposadr[self.model.joint(f'robot_B_joint_{i}').id] for i in range(1, 8)]
        self.jaws = [self.model.jnt_qposadr[self.model.joint(f'gripper_slide_{sign}').id] for sign in (-1, 1)]
        self.model.vis.headlight.specular[:] = .1

    def frame(self, renderer, t):
        state = self.sequence.state(t)
        set_pose(self.model, self.data, 'object', state['object'])
        set_pose(self.model, self.data, 'contact', state['blue'])
        self.data.qpos[self.addresses] = self.sequence.joints(t)
        self.data.qpos[self.jaws] = state['gap']-.009
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        for i in range(len(self.sequence.cases)):
            self.model.geom_rgba[self.model.geom(f'work_{i}').id, 3] = float(state['work'] == i)
        self.camera.lookat[:] = [.10, .12, .20]
        self.camera.distance = 1.65 * .70
        renderer.update_scene(self.data, camera=self.camera)
        main = Image.fromarray(renderer.render())
        self.annotate_scene(main, renderer.scene, state)
        return np.asarray(main)

    def annotate_scene(self, image, scene, state):
        ink = ImageDraw.Draw(image)
        if state['task']:
            case = self.sequence.cases[state['work']]
            ids = case['work_ids']
            # Red arrow denotes a task at the exposed patch, not a solved load.
            face = ids[len(ids)//2]
            T = state['object']
            point = T[:3, :3]@self.sequence.obj.triangles_center[face]+T[:3, 3]
            normal = T[:3, :3]@self.sequence.obj.face_normals[face]
            start = point + normal*.045
            annotate(ink, project(scene, start, WIDTH, HEIGHT), project(scene, point, WIDTH, HEIGHT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', help='Make geometry report and storyboard without MP4.')
    parser.add_argument('--fps', type=int, default=FPS)
    parser.add_argument('--duration', type=float, default=VIDEO_DURATION)
    args = parser.parse_args()
    sequence = Sequence()
    report = sequence.check_geometry()
    report['robot'] = sequence.prepare_arm()
    report['parking_equilibrium'] = {
        key: ground_balance(mesh, pose)
        for key, mesh, pose in [('object', sequence.obj, sequence.object_park),
                                ('blue', sequence.blue, sequence.blue_park)]}
    report['connected_components'] = dict(blue=len(sequence.blue.split()), base=len(sequence.base.split()))
    for source in (Path(__file__), Path(__file__).with_name('separate_loading.py'),
                   Path(__file__).with_name('separate_scene.py')):
        sequence.meta['source_sha256'][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    np.savez(OUT/'shared_workflow_trajectory.npz', times=sequence.arm_times,
             joints=sequence.arm_q, time_scale=sequence.time_scale)
    report.update(kind='kinematic_concept', one_stationary_connected_base=True, docking_socket_count=len(sequence.cases),
                  workflow='blue_in_object_in_task_object_out_blue_out',
                  carried_together=False, shared_object_blue_assembly_relation=True,
                  base_extents_mm=(sequence.base.extents*1000).tolist(),
                  robot_base_m=ROBOT_BASE.tolist(),
                  camera_distance_m=1.155,camera_original_distance_m=1.65,camera_distance_reduction_fraction=.30,
                  task_count=3, layout='full_scene', steps_overlay=False, detail_inset=False,
                  gripper='common_palm_two_slide_jaws_fixed_length_fingers',
                  contact_module_installations=3, object_installations=3,
                  object_removals=3, contact_module_removals=3, geometry_sampled_not_continuously_certified=True,
                  robot_collision_verified=False, co_grasp_verified=False, retention_verified=False,
                  dynamics_integrated=False, task_loads_verified=False, lying_stable_pose_index=sequence.lying_index,
                  blue_parking_pose_index=sequence.blue_parking_index,
                  geometry_source=sequence.meta,
                  floor_only_task_poses=[c['ground_balance'] for c in sequence.cases],
                  object_gravity_with_fixed_blue=[c['supported_gravity_balance'] for c in sequence.cases],
                  timeline=[dict(start=r['start'],end=r['end'],phase=r['label'],
                                 moving=r['moving'],dock=r['case']) for r in sequence.segments],
                  duration_s=args.duration,
                  reference_duration_s=sequence.duration*sequence.time_scale,
                  playback_speedup=sequence.duration*sequence.time_scale/args.duration,
                  joint_speed_check_scope='reference trajectory before presentation speedup',
                  simulation_code_reused=['codes/simulation/kuka.py','codes/simulation/workflow.py'],
                  source_sha256=sequence.meta['source_sha256'])
    display = Display(sequence)
    with mujoco.Renderer(display.model, height=HEIGHT, width=WIDTH) as renderer:
        storyboard = Image.new('RGB', (1800, 1685), 'white')
        for i, t in enumerate(sequence.story_times):
            frame = Image.fromarray(display.frame(renderer, t))
            storyboard.paste(frame.resize((600, 337), Image.Resampling.LANCZOS), ((i//5)*600, (i%5)*337))
        storyboard.save(OUT/'shared_workflow_storyboard.png')
        if not args.preview:
            total = round(args.duration*args.fps)
            temporary_video = OUT/'shared_workflow.pending.mp4'
            with mp4_writer(temporary_video, args.fps) as writer:
                for i in range(total):
                    writer.append_data(display.frame(renderer, sequence.duration*i/(total-1)))
                    if i % (args.fps*5) == 0: print(f'Video {i/args.fps:.0f}/{total/args.fps:.0f} s', flush=True)
            temporary_video.replace(OUT/'shared_workflow.mp4')
            report['frames'] = total; report['fps'] = args.fps
    (OUT/'shared_workflow_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(OUT/'shared_workflow_storyboard.png', flush=True)


if __name__ == '__main__':
    main()
