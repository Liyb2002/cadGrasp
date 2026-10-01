"""Search a continuous ten-target sequence with a new grasp at every target.

Regrasps take place at a stable intermediate placement, with the object free.
The optional ideal-grasp demo fixes the object only while the jaws are closed;
it explicitly assumes grasp-force feasibility instead of certifying it.
Search branches restore the complete simulation state; only the chosen branch
is exported. No object state is assigned after initialization in that branch.
"""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
import tempfile
sys.dont_write_bytecode = True

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp

import grasp as G
import sequence as S
import kuka_transfer as K
from sequence_export import WorkRegions, export

IDEAL_HOLD_FORCE_N = 2.


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


def descriptor(candidate, rest):
    inv = np.linalg.inv(rest)
    points = trimesh.transform_points(candidate['contacts'], inv)
    hand = inv @ candidate['hand']
    return dict(contact_points_object_m=points.tolist(),
                center_object_m=points.mean(axis=0).tolist(),
                approach_object=hand[:3, 2].tolist(),
                closing_axis_object=hand[:3, 1].tolist(),
                T_object_hand=hand.tolist(), nominal_width_m=candidate['width'])


def differences(a, b):
    ap, bp = np.array(a['contact_points_object_m']), np.array(b['contact_points_object_m'])
    # Jaws are interchangeable: swapping their labels is not a new location.
    distance = min(np.linalg.norm(ap-bp, axis=1).mean(),
                   np.linalg.norm(ap-bp[::-1], axis=1).mean())
    approach = angle(a['approach_object'], b['approach_object'])
    closing = angle(a['closing_axis_object'], b['closing_axis_object'])
    return float(distance), approach, min(closing, 180-closing)


def diverse_grasp(desc, previous, scale,minimum_pairwise_direction=12.):
    if not previous:
        return True
    for old in previous:
        d, a, c = differences(desc, old)
        if d < max(.003, .035*scale) or max(a, c) < minimum_pairwise_direction:
            return False
    d, a, c = differences(desc, previous[-1])
    return d >= max(.006, .08*scale) and max(a, c) >= 30


def diverse_pose(T, previous, rest):
    if angle(T[2, :3], rest[2, :3]) < 18:
        return False
    if any(angle(T[2, :3], old[2, :3]) < 18 for old in previous):
        return False
    return not previous or angle(T[2, :3], previous[-1][2, :3]) >= 32


class RegraspTrial(S.Trial):
    def __init__(self, *args, grip_force=70., **kwargs):
        super().__init__(*args, **kwargs)
        self.grip_force = grip_force
        self.post_attachment_force = grip_force
        self.events = []
        self.track_frames=[];self.track_q=[]
        self.lock_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_EQUALITY,'ideal_grasp')
        self.constraint_history=[];self.constraint_data_history=[]

    def step(self, h, force=None):
        if force is None:
            attached=self.lock_id>=0 and bool(self.data.eq_active[self.lock_id])
            force=-self.post_attachment_force if attached else -self.grip_force
        super().step(h,force)
        if self.steps % 33 == 0:
            if self.capture and self.lock_id>=0:
                self.constraint_history.append(bool(self.data.eq_active[self.lock_id]))
                self.constraint_data_history.append(self.model.eq_data[self.lock_id].copy())
            for contact in self.data.contact:
                names = [self.model.geom(int(g)).name for g in (contact.geom1, contact.geom2)]
                if 'floor' in names and any(n.startswith('hand_geom_') for n in names):
                    if contact.dist < -.0002:
                        raise ValueError(f'Gripper penetrates floor: depth={-contact.dist:.6g} m, '
                            f'phase={getattr(self,"motion_phase","unknown")}, time={self.steps*.001:.3f} s')

    def snapshot(self):
        return (copy.copy(self.data), self.steps, len(self.history),
                len(self.events), self.offset.copy(), self.opening,
                getattr(self, 'relative_inverse', None),self.model.eq_data.copy(),
                self.post_attachment_force)

    def restore(self, snapshot):
        data, self.steps, n, ne, self.offset, self.opening, relative,eq_data,self.post_attachment_force = snapshot
        self.data = copy.copy(data)
        self.model.eq_data[:]=eq_data
        del self.history[n:]; del self.object_history[n:]; del self.events[ne:]
        del self.constraint_history[n:];del self.constraint_data_history[n:]
        self.offset = self.offset.copy()
        if relative is not None:
            self.relative_inverse = relative.copy()

    def open_force(self):
        return float(np.clip(3000*(self.opening-np.mean(self.data.qpos[self.fingers]))
                             -20*np.mean(self.data.qvel[self.speeds]), -70, 70))

    def hand_move(self, target, duration=1.5, opened=True):
        source = G.transform(self.data, self.hand)
        rotations = Slerp([0, 1], Rotation.from_matrix([source[:3, :3], target[:3, :3]]))
        for j in range(round(duration*1000)):
            f = G.smooth((j+1)/(duration*1000))
            h = np.eye(4); h[:3, :3] = rotations(f).as_matrix()
            h[:3, 3] = (1-f)*source[:3, 3]+f*target[:3, 3]
            self.step(h, self.open_force() if opened else None)

    def move_object(self, target, stable=False):
        self.motion_phase='return to rest' if stable else 'place target'
        source = G.transform(self.data, self.obj)
        self.relative_inverse = np.linalg.inv(source) @ G.transform(self.data, self.hand)
        self.offset[:] = 0
        rotations = Slerp([0, 1], Rotation.from_matrix([source[:3, :3], target[:3, :3]]))
        xy0 = (source[:3, :3]@self.mesh.center_mass+source[:3, 3])[:2]
        xy1 = (target[:3, :3]@self.mesh.center_mass+target[:3, 3])[:2]
        height0 = max(0., float((self.floor_vertices@source[2, :3]).min()+source[2, 3]))
        # During a large turn the fingers can hang below the object's lowest
        # point even when both endpoints are clear. Lift enough for the swept
        # closed tool, instead of rejecting every such reorientation at 35 mm.
        tool_points=[]
        inverse=np.linalg.inv(source)
        corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        for geom in range(self.model.ngeom):
            if (self.model.geom(geom).name.startswith('hand_geom_')
                    and self.model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_BOX):
                world=(corners*self.model.geom_size[geom])@self.data.geom_xmat[geom].reshape(3,3).T+self.data.geom_xpos[geom]
                tool_points.extend(world@inverse[:3,:3].T+inverse[:3,3])
        clearance=.035
        if tool_points:
            tool_points=np.asarray(tool_points)
            for f in np.linspace(0,1,25):
                seated=S.seat(self.mesh,rotations(f).as_matrix(),(1-f)*xy0+f*xy1)
                minimum=float((tool_points@seated[2,:3]).min()+seated[2,3])
                clearance=max(clearance,.005-minimum)
        if clearance>.18:raise ValueError('Turn requires excessive tool clearance')
        for j in range(5500):
            t = j*.001; f = G.smooth((t-.8)/2.2)
            desired = S.seat(self.mesh, rotations(f).as_matrix(), (1-f)*xy0+f*xy1)
            if t < .8:
                height = height0+(clearance-height0)*G.smooth(t/.8)
            elif t < 3.:
                height = clearance
            else:
                preload=0. if self.lock_id>=0 else .00008
                height = clearance*(1-G.smooth((t-3)/1.))-preload*G.smooth((t-3)/1.)
            desired[2, 3] += height
            self.follow(desired)
        h = np.eye(4); h[:3, 3] = self.data.mocap_pos[0]
        h[:3, :3] = Rotation.from_quat(self.data.mocap_quat[0][[1, 2, 3, 0]]).as_matrix()
        for _ in range(700): self.step(h)
        # Finish seating by measured normal force. A small positive gap is
        # not contact, even when it is visually indistinguishable from it.
        if S.contacts(self.model,self.data)[1] < .01:
            stable_steps=0
            for _ in range(3000):
                force=S.contacts(self.model,self.data)[1]
                if force<.01:
                    h[2,3] -= .000001
                    stable_steps=0
                else:
                    stable_steps+=1
                self.step(h)
                if stable_steps>=600:break
        dof=int(self.model.jnt_dofadr[self.model.body_jntadr[self.obj]])
        attached=self.lock_id>=0 and bool(self.data.eq_active[self.lock_id])
        settling_window=None
        if attached and not stable:
            # A weld and floor can produce high-frequency solver velocities
            # with negligible pose variation. For this explicitly kinematic
            # demo, measure actual pose stationarity over a complete dwell.
            # Seventeen samples at 33 ms span 0.528 s; no timing is inferred
            # from controller iterations, which may hold several physics steps.
            last_count=-1
            for extra in range(2001):
                count=len(self.object_history)
                if count>=17 and count!=last_count:
                    last_count=count
                    recent=np.asarray(self.object_history[-17:])
                    delta=recent[:,None,:3,3]-recent[None,:,:3,3]
                    translation=float(np.linalg.norm(delta,axis=-1).max())
                    relative=recent[:,None,:3,:3]@np.swapaxes(recent[None,:,:3,:3],-1,-2)
                    rotation=float(np.degrees(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude()).max())
                    end_time=(self.steps//33)*.033
                    settling_window=dict(method='Ideal-grasp actual-pose dwell',
                        start_time_s=end_time-16*.033,end_time_s=end_time,
                        duration_s=16*.033,samples=17,sample_period_s=.033,
                        translation_range_m=translation,rotation_range_deg=rotation,
                        translation_limit_m=.00005,rotation_limit_deg=.05)
                if settling_window is not None and translation<=.00005 and rotation<=.05:
                    break
                if extra<2000:self.step(h)
            else:
                raise ValueError(f'Ideal-grasp pose has not settled over a full dwell: {settling_window}')
        else:
            for extra in range(0 if stable else 2000):
                velocity=self.data.qvel[dof:dof+6]
                if np.linalg.norm(velocity[:3])<=.001 and np.linalg.norm(velocity[3:])<=.02:break
                self.step(h)
        actual = G.transform(self.data, self.obj)
        gap = float((self.mesh.vertices@actual[2, :3]).min()+actual[2, 3])
        fingers, force = S.contacts(self.model, self.data)
        err = float(np.degrees(Rotation.from_matrix(actual[:3, :3]@target[:3, :3].T).magnitude()))
        attached=self.lock_id>=0 and bool(self.data.eq_active[self.lock_id])
        if (fingers != 2 and not attached) or force < .001 or not -.0003 < gap < .0002 or err > (5 if stable else 12):
            raise ValueError(f'Placement failed: stable={stable}, fingers={fingers}, force={force:.4g}, gap={gap:.4g}, angle={err:.3g}')
        pose = actual.copy(); pose[2, 3] -= gap
        dof = int(self.model.jnt_dofadr[self.model.body_jntadr[self.obj]])
        velocity = self.data.qvel[dof:dof+6]
        # A supported return is followed by a separate unheld stability test.
        # Tiny competing floor/grasp constraint velocities before opening are
        # not a substitute for that test and need not block the release.
        if not stable and not attached and (np.linalg.norm(velocity[:3]) > .001 or np.linalg.norm(velocity[3:]) > .02):
            raise ValueError(f'Target has not settled: stable={stable}, '
                f'linear={np.linalg.norm(velocity[:3]):.5g}, angular={np.linalg.norm(velocity[3:]):.5g}')
        if not stable:
            v = trimesh.transform_points(self.mesh.vertices, pose)
            ids = np.flatnonzero(v[:, 2] < 1e-8)
            com = pose[:3, :3]@self.mesh.center_mass+pose[:3, 3]
            if len(ids) != 1 or np.linalg.norm(com[:2]-v[ids[0], :2]) < .001:
                raise ValueError('Target is not an unsupported single-point pose')
            if not self.regions.feasible(pose):
                raise ValueError('No feasible working region')
        return pose, dict(raw_mesh_floor_gap_m=gap, object_floor_normal_force_N=force,
            carry_clearance_m=clearance,
            bilateral_grip=fingers==2,finger_contacts=fingers,ideal_grasp_active=attached,
            target_position_error_m=abs(gap), target_rotation_error_deg=0.,
            commanded_rotation_error_deg=err, linear_speed_m_s=float(np.linalg.norm(velocity[:3])),
            angular_speed_rad_s=float(np.linalg.norm(velocity[3:])),settling_window=settling_window)

    def release(self, rest):
        self.move_object(rest, stable=True)
        h = G.transform(self.data, self.hand)
        unload_time=0.
        unload_force=0.
        if self.lock_id>=0 and self.data.eq_active[self.lock_id]:
            # Unload the still-closed jaws before releasing the ideal grasp;
            # stored jaw force must not kick a light part as the jaws open.
            unload_force=self.post_attachment_force
            unload_start=self.steps*.001
            for j in range(1000):
                self.step(h,-unload_force*(1-G.smooth((j+1)/1000)))
            unload_time=self.steps*.001-unload_start
        begin = self.steps*.001
        if self.lock_id>=0:self.data.eq_active[self.lock_id]=False
        self.motion_phase='open on rest'
        start_opening=float(np.mean(self.data.qpos[self.fingers]))
        closed_max=float(np.max(self.data.qpos[self.fingers]))
        opening_limit=float(self.model.jnt_range[self.model.joint('finger_joint1').id,1])
        probe=copy.copy(self.data)
        for extra in (.008,.006,.004,.002):
            end_opening=min(opening_limit,closed_max+extra)
            probe.qpos[self.fingers]=end_opening
            mujoco.mj_forward(self.model,probe)
            collides=any(c.dist<-.0002 and
                'floor' in [self.model.geom(int(g)).name for g in (c.geom1,c.geom2)] and
                any(self.model.geom(int(g)).name.startswith('hand_geom_') for g in (c.geom1,c.geom2))
                for c in probe.contact)
            if not collides:break
        else:
            raise ValueError('No collision-free release opening at the intermediate rest')
        for j in range(1500):
            self.opening=start_opening+(end_opening-start_opening)*G.smooth((j+1)/1500)
            self.step(h,self.open_force())
        opening_time=self.steps*.001-begin
        retreat = h.copy(); retreat[:3, 3] -= .06*h[:3, 2]
        self.motion_phase='retreat after release'
        self.hand_move(retreat)
        before = G.transform(self.data, self.obj)
        dwell_start=self.steps*.001
        dwell_index=len(self.object_history)
        for _ in range(800): self.step(retreat, self.open_force())
        after = G.transform(self.data, self.obj)
        drift = float(np.degrees(Rotation.from_matrix(after[:3, :3]@before[:3, :3].T).magnitude()))
        tilt_drift=angle(after[2,:3],before[2,:3])
        dwell_poses=np.asarray([before]+self.object_history[dwell_index:]+[after])
        if len(dwell_poses)<3:
            raise ValueError('Recorded samples are required throughout the unheld dwell')
        gravity=dwell_poses[:,2,:3]
        gravity_range=float(np.degrees(np.arccos(np.clip(gravity@gravity.T,-1.,1.))).max())
        translation_drift=float(np.linalg.norm(after[:3,3]-before[:3,3]))
        object_dof=int(self.model.jnt_dofadr[self.model.body_jntadr[self.obj]])
        linear_speed=float(np.linalg.norm(self.data.qvel[object_dof:object_dof+3]))
        requested_difference=angle(after[2,:3],rest[2,:3])
        fingers, force = S.contacts(self.model, self.data)
        # Frictionless planar support cannot dissipate a residual yaw spin.
        # In the ideal-grasp demo, test resistance to tipping and explicitly
        # track the free object's remaining yaw/translation during pickup.
        stable_drift=tilt_drift if self.lock_id>=0 else drift
        if fingers or force < .001 or stable_drift > .3 or gravity_range > .3:
            raise ValueError('Intermediate rest is not stable after release')
        # A different resting face is allowed: the following pickup is mapped
        # through this actual pose and reruns approach/closure geometry checks.
        # Never replace the simulated free object's pose with the requested one.
        self.events.append(dict(event='released_on_stable_rest',start_time_s=begin,
            end_time_s=self.steps*.001,unheld_rotation_drift_deg=drift,
            unheld_tilt_drift_deg=tilt_drift,
            unheld_gravity_direction_range_deg=gravity_range,
            unheld_dwell_start_time_s=dwell_start,unheld_dwell_end_time_s=self.steps*.001,
            unheld_dwell_samples=len(dwell_poses),unheld_translation_drift_m=translation_drift,
            unheld_linear_speed_m_s=linear_speed,
            unheld_translation_reference='mesh-origin free joint',
            requested_rest_transform=rest.tolist(),actual_rest_transform=after.tolist(),
            requested_rest_gravity_difference_deg=requested_difference,
            stability_metric='gravity_direction' if self.lock_id>=0 else 'full_orientation',
            residual_yaw_tracked=self.lock_id>=0,
            open_hand_retreat_m=.06,
            closed_jaw_force_unload_s=unload_time,opening_ramp_s=opening_time,
            closed_jaw_force_unload_from_N=unload_force,
            release_jaw_position_m=end_opening,
            release_extra_clearance_per_jaw_m=max(0.,end_opening-closed_max),
            finger_contacts=fingers,object_floor_normal_force_N=force))

    def pickup(self, candidate, reference_rest, first=False):
        self.motion_phase='empty hand transit'
        actual = G.transform(self.data, self.obj)
        h = actual @ np.linalg.inv(reference_rest) @ candidate['hand']
        opening = min(float(self.model.jnt_range[self.model.joint('finger_joint1').id, 1]), candidate['opening'])
        transformed = dict(candidate, hand=h, opening=opening)
        if not G.geometry_check(self.model, actual, transformed):
            raise ValueError('Regrasp approach collision')
        pre = h.copy(); pre[:3, 3] -= .08*h[:3, 2]
        if not first:
            if getattr(self,'_demo_direct_joint_transit',False):
                # Joint interpolation stays within reach; the full scene
                # validation rejects any collision along the resulting sweep.
                self.opening=opening
                self.hand_move(pre, 2.)
            else:
                # Retract above the entire part before changing the wrist attitude.
                current = G.transform(self.data, self.hand)
                safe_z = max(.32, current[2, 3], pre[2, 3]+.10)
                up = current.copy(); up[2, 3] = safe_z
                self.hand_move(up)
                # Set the incoming grasp width while safely above the part,
                # before descending; the previous grasp may have been narrower.
                self.opening=opening
                across = pre.copy(); across[2, 3] = safe_z
                self.hand_move(across, 2.)
                self.hand_move(pre)
            # Correct small unheld drift accumulated during the overhead move.
            actual=G.transform(self.data,self.obj)
            h=actual@np.linalg.inv(reference_rest)@candidate['hand']
        self.opening = opening
        object_hand=np.linalg.inv(reference_rest)@candidate['hand']
        if not first and self.lock_id>=0:
            # An unheld part can drift during the overhead transit. Align at
            # the live pregrasp first, so insertion does not begin with jaws
            # centred on the object's old location.
            self.motion_phase='align updated pregrasp'
            source=G.transform(self.data,self.hand)
            for j in range(600):
                goal=G.transform(self.data,self.obj)@object_hand
                goal[:3,3]-=.08*goal[:3,2]
                f=G.smooth((j+1)/600)
                h=np.eye(4);h[:3,3]=(1-f)*source[:3,3]+f*goal[:3,3]
                h[:3,:3]=Slerp([0,1],Rotation.from_matrix([source[:3,:3],goal[:3,:3]]))(f).as_matrix()
                self.step(h,self.open_force())
        begin = self.steps*.001
        self.motion_phase='grasp approach'
        if self.lock_id>=0:
            source=G.transform(self.data,self.hand)
            for j in range(1500):
                goal=G.transform(self.data,self.obj)@object_hand
                f=G.smooth((j+1)/1500)
                h=np.eye(4);h[:3,3]=(1-f)*source[:3,3]+f*goal[:3,3]
                h[:3,:3]=Slerp([0,1],Rotation.from_matrix([source[:3,:3],goal[:3,:3]]))(f).as_matrix()
                self.step(h,self.open_force())
        else:self.hand_move(h, 1.5)
        closure_frozen=self.lock_id<0
        bilateral_steps=0
        closing_start=self.steps*.001
        self.motion_phase='jaw closure'
        for j in range(1200):
            if not closure_frozen:
                if S.contacts(self.model,self.data)[0]>0:
                    # Once a jaw touches, hold the actual wrist pose rather
                    # than chasing object motion induced by closing pressure.
                    h=G.transform(self.data,self.hand)
                    closure_frozen=True
                else:
                    h=G.transform(self.data,self.obj)@object_hand
            self.step(h, -self.grip_force*G.smooth((j+1)/700))
            bilateral_steps=bilateral_steps+1 if S.contacts(self.model,self.data)[0]==2 else 0
            # The ideal demo assumes rigid capture once both jaws have made
            # sustained contact. Waiting at full load before enabling that
            # assumption needlessly tests the excluded grasp-slip dynamics.
            if self.lock_id>=0 and bilateral_steps>=33:break
        closure_time=self.steps*.001
        closure_contacts=S.contacts(self.model,self.data)[0]
        if self.lock_id>=0:
            if S.contacts(self.model,self.data)[0]!=2:
                raise ValueError('Cannot activate ideal grasp without bilateral finger contact')
            relative=np.linalg.inv(G.transform(self.data,self.hand))@G.transform(self.data,self.obj)
            self.model.eq_data[self.lock_id,3:6]=relative[:3,3]
            self.model.eq_data[self.lock_id,6:10]=G.quaternion(relative[:3,:3])
            self.data.eq_active[self.lock_id]=True
            self.post_attachment_force=min(self.grip_force,IDEAL_HOLD_FORCE_N)
        lift = h.copy(); lift[2, 3] += .035
        self.motion_phase='grasp lift'
        self.hand_move(lift, 1.2, opened=False)
        for _ in range(500): self.step(lift)
        obj = G.transform(self.data, self.obj)
        fingers, _ = S.contacts(self.model, self.data)
        gap = float((self.mesh.vertices@obj[2, :3]).min()+obj[2, 3])
        attached=self.lock_id>=0 and bool(self.data.eq_active[self.lock_id])
        if (fingers != 2 and not attached) or gap < .015:
            raise ValueError(f'New grasp did not pick up the object: fingers={fingers}, clearance={gap:.5g}')
        self.relative_inverse = np.linalg.inv(obj) @ G.transform(self.data, self.hand)
        self.events.append(dict(event='new_grasp',start_time_s=begin,end_time_s=self.steps*.001,
                               bilateral_grip=fingers==2,bilateral_closure_verified=closure_contacts==2,
                               contact_sample_time_s=closure_time,lift_clearance_m=gap,
                               closing_duration_s=closure_time-closing_start,
                               sustained_bilateral_controller_steps=bilateral_steps,
                               post_attachment_closing_force_N=self.post_attachment_force,
                               ideal_grasp_assumption=self.lock_id>=0))


def target_menu(mesh, rest, previous, relative_inverse, model, regions, jaw_position):
    menu = []
    for tilt in (25, 45, 65, 85):
        for azimuth in np.arange(0, 360, 30):
            axis = np.array([np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth)), 0.])
            # Avoid aligning every sampled orientation with the same regular
            # tessellation edges (rounded parts often use two-degree facets).
            R = Rotation.from_rotvec(axis*np.radians(tilt+1.3)).as_matrix()@rest[:3, :3]
            for yaw in (0, -35, 35):
                T = S.seat(mesh, Rotation.from_euler('z',yaw,degrees=True).as_matrix()@R)
                if not diverse_pose(T, previous, rest): continue
                vertices=trimesh.transform_points(mesh.vertices,T)
                ids=np.flatnonzero(vertices[:,2]<1e-8)
                if len(ids)!=1: continue
                com=T[:3,:3]@mesh.center_mass+T[:3,3]
                if np.linalg.norm(com[:2]-vertices[ids[0],:2])<.001: continue
                hand = T@relative_inverse
                candidate = dict(hand=hand,width=.08,opening=.08)
                # Ground clearance at the actual closed grasp, not an invented
                # approach to the held target. Full rollout checks follow.
                d = mujoco.MjData(model); G.object_pose(model,d,T)
                G.hand_pose(d,hand,initialize=True)
                d.qpos[[model.joint(f'finger_joint{i}').qposadr[0] for i in (1,2)]]=jaw_position
                mujoco.mj_forward(model,d)
                if any(c.dist<-.00001 and 'floor' in [model.geom(int(g)).name for g in (c.geom1,c.geom2)]
                       and any(model.geom(int(g)).name.startswith('hand_geom_') for g in (c.geom1,c.geom2))
                       for c in d.contact): continue
                if not regions.feasible(T): continue
                separation = min([angle(T[2,:3],p[2,:3]) for p in previous] or [45.])
                menu.append((separation-.12*abs(yaw)-.45*abs(tilt-45),tilt,T))
    # Give the short rollout budget options at several inclinations rather
    # than spending every attempt on the same high-separation tilt range.
    ranked=sorted(menu,key=lambda row:row[0],reverse=True)
    buckets={tilt:[T for _,value,T in ranked if value==tilt] for tilt in (45,25,65,85)}
    balanced=[]
    for i in range(max((len(rows) for rows in buckets.values()),default=0)):
        balanced.extend(rows[i] for rows in buckets.values() if i<len(rows))
    return balanced


def save_checkpoint(path,trial,poses,checks,grasps,first_candidate):
    specification=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(trial.model,specification))
    mujoco.mj_getState(trial.model,trial.data,state,specification)
    q,pos,quat=zip(*trial.history)
    candidate={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in first_candidate.items()}
    np.savez_compressed(path,state=state,qpos=q,mocap_pos=pos,mocap_quat=quat,
        object_history=trial.object_history,steps=trial.steps,opening=trial.opening,
        relative_inverse=trial.relative_inverse,offset=trial.offset,arm_q=trial.arm_q,
        post_attachment_force_N=trial.post_attachment_force,
        track_frames=np.asarray(trial.track_frames,dtype=int),track_q=np.asarray(trial.track_q),
        eq_data=trial.model.eq_data,ideal_grasp_active=trial.constraint_history,
        ideal_grasp_eq_data=trial.constraint_data_history,
        poses=poses,checks=json.dumps(checks),grasps=json.dumps(grasps),
        events=json.dumps(trial.events),first_candidate=json.dumps(candidate))
    shutil.copy2(path,path.with_name(f'{path.stem}-{len(poses):02}.npz'))


def append_track(trial,frames,qs,start):
    for frame,q in zip(frames+start,qs):
        if trial.track_frames and frame<=trial.track_frames[-1]:continue
        trial.track_frames.append(int(frame));trial.track_q.append(q.copy())


def find(name, pairs=120, candidate_budget=80, resume=True, anchor_index=0, anchor_yaw=0., grasp_method='sampled', com_weight=2.5, soft_pads=False, ideal_grasp=False,pairwise_grasp_direction_deg=12.,tool='standard',min_grasp_width=.015,pose_count=10):
    from regrasp_contacts import measured_grasps
    measured_contact_min=.006
    measured_direction_min=29.5
    if isinstance(pose_count, bool) or not isinstance(pose_count, int) or pose_count < 1:
        raise ValueError('pose_count must be a positive integer')
    folder = G.ROOT/'objects'/name
    saved = json.loads((folder/'poses.json').read_text())
    mesh = trimesh.load(folder/'mesh.stl',force='mesh')
    rest = S.seat(mesh,np.asarray(saved['rest']['T_world_mesh'])[:3,:3])
    anchor=rest.copy()
    if anchor_index:
        stable,probabilities=mesh.compute_stable_poses(sigma=0.,n_samples=1,threshold=.005)
        if not 1<=anchor_index<=len(stable):
            raise ValueError(f'{name}: anchor index must be 0..{len(stable)}')
        anchor=S.seat(mesh,stable[anchor_index-1,:3,:3])
        alignment=rest[:3,:3]@anchor[:3,:3].T
        heading=np.arctan2(alignment[1,0]-alignment[0,1],alignment[0,0]+alignment[1,1])
        anchor=S.seat(mesh,Rotation.from_euler('z',heading).as_matrix()@anchor[:3,:3])
    if anchor_yaw:
        anchor=S.seat(mesh,Rotation.from_euler('z',anchor_yaw,degrees=True).as_matrix()@anchor[:3,:3])
    if soft_pads and ideal_grasp:raise ValueError('Choose either ideal-grasp demo or compliant-contact simulation')
    if tool not in ('standard','compact'):raise ValueError('Unknown gripper family')
    tool_size='_compact' if tool=='compact' else '_180' if name=='D4' else ''
    tool_name='parallel_jaw'+tool_size+('_ideal' if ideal_grasp else '_pads' if soft_pads else '')+'.xml'
    hand_xml = G.HERE/'assets'/tool_name
    model = G.build(name,exact=name=='B',hand_xml=hand_xml,floor_hull=True)
    regions = WorkRegions(mesh)
    arm_model=K.Arm('A');arm_model.base[:]=[-.45,-.35,0.]
    depth = (.10,.08,.112,.095,.118) if mesh.extents.max()>.10 else (.112,.118,.10)
    generator=G.candidates
    if grasp_method=='rays':
        from antipodal_rays import candidates
        generator=candidates
    max_width=.095 if tool=='compact' else .17 if name=='D4' else .15
    if not 0<min_grasp_width<max_width:raise ValueError('Minimum grasp width is outside the tool range')
    catalog = list(generator(mesh,anchor,pairs,max_width=max_width,depths=depth,min_width=min_grasp_width))
    for candidate in catalog:
        candidate['opening']=min(float(model.jnt_range[model.joint('finger_joint1').id,1]),
                                 float(candidate['width'])/2+.01)
    descriptions = [descriptor(c,anchor) for c in catalog]
    scale = float(mesh.extents.max())
    trial=None; poses=[]; checks=[]; grasps=[]; first_candidate=None
    checkpoint_path=Path(tempfile.gettempdir())/f'cadgrasp-regrasp-{name}{"-compact" if tool=="compact" else ""}{"-ideal" if ideal_grasp else "-pads" if soft_pads else ""}-checkpoint.npz'
    if pose_count != 10:
        checkpoint_path = checkpoint_path.with_name(f'{checkpoint_path.stem}-n{pose_count}.npz')
    if resume and checkpoint_path.exists():
        with np.load(checkpoint_path) as z:
            first_candidate=json.loads(str(z['first_candidate']))
            for key in ('hand','contacts'):first_candidate[key]=np.array(first_candidate[key])
            trial=RegraspTrial(model,mesh,rest,first_candidate,regions=regions,
                               grip_force=10.5 if name=='D4' else 70.)
            mujoco.mj_setState(model,trial.data,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION)
            if 'eq_data' in z:model.eq_data[:]=z['eq_data']
            mujoco.mj_forward(model,trial.data)
            trial.history=list(zip(z['qpos'].copy(),z['mocap_pos'].copy(),z['mocap_quat'].copy()))
            trial.object_history=list(z['object_history'].copy());trial.steps=int(z['steps'])
            trial.opening=float(z['opening']);trial.relative_inverse=z['relative_inverse'].copy()
            trial.offset=z['offset'].copy();trial.arm_q=z['arm_q'].copy()
            if 'track_frames' in z:
                trial.track_frames=list(z['track_frames']);trial.track_q=list(z['track_q'].copy())
            if 'ideal_grasp_active' in z:
                trial.constraint_history=list(z['ideal_grasp_active'])
                trial.constraint_data_history=list(z['ideal_grasp_eq_data'].copy())
            trial.events=json.loads(str(z['events']));poses=list(z['poses'].copy())
            checks=json.loads(str(z['checks']));grasps=json.loads(str(z['grasps']))
            pickups=[event for event in trial.events if event['event']=='new_grasp']
            legacy_force=(pickups[-1].get('post_attachment_closing_force_N',trial.grip_force)
                          if pickups else trial.grip_force)
            trial.post_attachment_force=(float(z['post_attachment_force_N'])
                if 'post_attachment_force_N' in z else float(legacy_force))
        print(json.dumps(dict(object=name,resumed=len(poses),checkpoint=str(checkpoint_path))),flush=True)
    if grasps and any('measured' not in grasp for grasp in grasps):
        for grasp,actual in zip(grasps,measured_grasps(model,trial.history,trial.events)):
            grasp['measured']=actual
    if len(poses) > pose_count:
        raise ValueError('Checkpoint has more targets than requested')
    for target_index in range(len(poses),pose_count):
        if trial is not None:
            return_start=len(trial.history)-1
            previous_q=trial.arm_q[-1]
            trial.release(anchor)
            return_frames,return_q,_=K.check_scene(model,trial.history[return_start:],name,
                stride=3,initial_q=previous_q,hand_xml=hand_xml)
            trial.empty_arm_q=return_q[-1].copy()
            released_frame=len(trial.history)-1
            checkpoint=trial.snapshot()
        elif anchor_index or anchor_yaw:
            raise ValueError('An alternate regrasp anchor requires an existing accepted-target checkpoint')
        eligible=[i for i,d in enumerate(descriptions) if diverse_grasp(d,grasps,scale,pairwise_grasp_direction_deg)]
        def score(i):
            if not grasps:
                return -catalog[i]['com_distance_m']-abs(catalog[i]['roll_deg'])*.0001
            diffs=[differences(descriptions[i],g) for g in grasps]
            return (min(d/scale+max(a,c)/180 for d,a,c in diffs)
                    -catalog[i]['com_distance_m']/scale*com_weight-abs(catalog[i]['roll_deg'])/180.)
        eligible.sort(key=score,reverse=True)
        tried=0; success=False
        for ci in eligible:
            candidate=catalog[ci]
            if not G.geometry_check(model,anchor,candidate): continue
            tried+=1
            if tried>candidate_budget: break
            if trial is None or target_index==0:
                trial=RegraspTrial(model,mesh,rest,candidate,regions=regions,
                                   grip_force=10.5 if name=='D4' else 70.)
            else:
                trial.restore(checkpoint)
            try:
                trial.pickup(candidate,anchor,first=target_index==0)
                actual_grasp=measured_grasps(model,trial.history,[trial.events[-1]])[0]
                if grasps:
                    distance,approach,closing=differences(actual_grasp,grasps[-1]['measured'])
                    if distance<measured_contact_min or max(approach,closing)<measured_direction_min:
                        raise ValueError('Measured pickup contact centroids or direction repeat the preceding grasp: '
                            f'location={distance:.6g} m (min {measured_contact_min:.6g}), '
                            f'approach={approach:.4g} deg, closing={closing:.4g} deg '
                            f'(direction min {measured_direction_min:.4g})')
                picked=trial.snapshot()
                # The release and approach prefix is shared by every target
                # tried with this grasp. Reject an impossible prefix once.
                start_frame=0 if not checks else released_frame
                initial_q=None if not checks else return_q[-1]
                pickup_frames,pickup_q,_=K.check_scene(model,trial.history[start_frame:],name,
                    stride=3,initial_q=initial_q,hand_xml=hand_xml,
                    robot_track=getattr(trial,'pickup_robot_track',None))
                picked_frame=len(trial.history)-1
                menu=target_menu(mesh,anchor,poses,trial.relative_inverse,model,regions,
                                 trial.data.qpos[trial.fingers])
                print(json.dumps(dict(object=name,target=target_index+1,candidate=ci,menu=len(menu))),flush=True)
                for target in menu[:18]:
                    trial.restore(picked)
                    try:
                        K.solve(arm_model,target@trial.relative_inverse,pickup_q[-1])
                        pose,row=trial.move_object(target)
                        if not diverse_pose(pose,poses,rest): raise ValueError('Reached pose lacks diversity')
                        # Reject unreachable or colliding branches before retaining them.
                        frames,qs,scene=K.check_scene(model,trial.history[picked_frame:],name,
                            stride=3,initial_q=pickup_q[-1],hand_xml=hand_xml)
                        # Verify that the new target can return to the regrasp rest.
                        finish=trial.snapshot()
                        if target_index<pose_count-1:
                            return_frame=len(trial.history)-1
                            trial.release(anchor)
                            K.check_scene(model,trial.history[return_frame:],name,
                                stride=3,initial_q=qs[-1],hand_xml=hand_xml)
                        trial.restore(finish)
                    except ValueError as error:
                        print(json.dumps(dict(object=name,target=target_index+1,candidate=ci,
                                              rejected=str(error))),flush=True)
                        continue
                    trial.arm_q=qs
                    if checks:append_track(trial,return_frames,return_q,return_start)
                    append_track(trial,pickup_frames,pickup_q,start_frame)
                    append_track(trial,frames,qs,picked_frame)
                    row.update(pose_id=f'pose_{target_index+1}',
                        start_time_s=0. if not checks else checks[-1]['end_time_s'],
                        end_time_s=trial.steps*.001,grasp_id=f'grasp_{target_index+1}',
                        commanded_transform=target.tolist(),
                        regrasp_rest_transform=anchor.tolist(),anchor_index=anchor_index,anchor_yaw_deg=anchor_yaw)
                    grasps.append(dict(descriptions[ci],grasp_id=f'grasp_{target_index+1}',
                                       measured=actual_grasp,
                                       candidate_index=ci,candidate_method=grasp_method,
                                       closing_force_scalar_N=trial.grip_force,
                                       post_attachment_closing_force_N=trial.post_attachment_force))
                    poses.append(pose);checks.append(row)
                    if first_candidate is None: first_candidate=candidate
                    save_checkpoint(checkpoint_path,trial,poses,checks,grasps,first_candidate)
                    print(json.dumps(dict(object=name,accepted=target_index+1,candidate=ci,
                                          time_s=trial.steps*.001)),flush=True)
                    success=True;break
                if success: break
            except ValueError as error:
                print(json.dumps(dict(object=name,target=target_index+1,candidate=ci,
                                      rejected=str(error))),flush=True)
        if not success:
            raise RuntimeError(f'{name}: found {len(poses)}/{pose_count} diverse regrasp targets; exhausted {tried} grasps')
    track=None
    if trial.track_frames and trial.track_frames[0]==0:
        track=(np.asarray(trial.track_frames),np.asarray(trial.track_q))
    trial.arm_frames,trial.arm_q,scene=K.check_scene(model,trial.history,name,hand_xml=hand_xml,
                                                 robot_track=track)
    arm=dict(scene_checks=scene,scope='Sampled KUKA full-pose IK, joint limits/speeds and scene collisions')
    rule=dict(method='diverse regrasp search',minimum_pairwise_gravity_direction_deg=18.,
        minimum_adjacent_gravity_direction_deg=32.,minimum_adjacent_contact_change_m=max(.006,.08*scale),
        minimum_adjacent_grasp_direction_deg=30.,minimum_pairwise_contact_change_m=max(.003,.035*scale),
        minimum_pairwise_grasp_direction_deg=pairwise_grasp_direction_deg,candidate_pair_budget=pairs,
        minimum_candidate_grasp_width_m=min_grasp_width,
        minimum_measured_adjacent_contact_change_m=measured_contact_min,
        minimum_measured_adjacent_grasp_direction_deg=measured_direction_min,
        description='Different contact locations and approach/closing axes for every target; '
                    f'released stable intermediate rests connect the {pose_count} unstable task targets.')
    metadata=dict(grasps=grasps,regrasp_events=trial.events,
        transfer_mode='Return to stable rest, open and retreat, approach a different grasp, lift with swept-tool floor clearance, reorient and seat',
        trajectory_revision='diverse_regrasp_v1',
        grasp_contact_dynamics_simulated=not ideal_grasp,
        gripper_contact_model=('Ideal rigid closed-grasp constraint, enabled only after bilateral closure; '
            'released on supported intermediate rest. Grasp wrench feasibility is assumed.' if ideal_grasp else
            'Generic flat compliant pads: condim 4, sliding friction 0.8, '
            'torsional effective length 0.003 m; assumed rather than hardware-calibrated'
            if soft_pads else 'Straight rigid fingers with condim 3 point contacts'))
    if ideal_grasp:
        metadata['verification_scope']='Kinematic manipulation demo with explicitly assumed rigid closed grasps; '
        metadata['verification_scope']+='free unheld intermediate rests and sampled KUKA IK, limits, speeds, collisions are checked. '
        metadata['verification_scope']+='Grasp-force capacity and hardware robustness are not established.'
    from regrasp_contacts import measured_grasps
    measured=measured_grasps(model,trial.history,trial.events)
    for grasp,actual in zip(grasps,measured):grasp['measured']=actual
    for a,b in zip(measured,measured[1:]):
        distance,approach,closing=differences(a,b)
        if distance<rule['minimum_measured_adjacent_contact_change_m'] or max(approach,closing)<rule['minimum_measured_adjacent_grasp_direction_deg']:
            raise ValueError('Measured consecutive grasps do not satisfy location and direction diversity')
    export(name,mesh,rest,first_candidate,trial,poses,rule,checks,arm,
           generator='codes/setup/regrasp_sequence.py',metadata=metadata,hand_xml=hand_xml)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object');parser.add_argument('--pairs',type=int,default=120)
    parser.add_argument('--candidate-budget',type=int,default=80)
    parser.add_argument('--pose-count',type=int,default=10)
    parser.add_argument('--fresh',action='store_true',help='Ignore the temporary search checkpoint')
    parser.add_argument('--anchor-index',type=int,default=0,
                        help='0 keeps original rest; 1..N chooses a mesh-computed stable intermediate rest')
    parser.add_argument('--anchor-yaw',type=float,default=0.)
    parser.add_argument('--grasp-method',choices=('sampled','rays'),default='sampled')
    parser.add_argument('--soft-pads',action='store_true',help='Generic flat compliant pads; uses a separate search checkpoint')
    parser.add_argument('--ideal-grasp',action='store_true',help='Explicit demo assumption of rigid closed grasps; separate checkpoint')
    parser.add_argument('--com-weight',type=float,default=2.5)
    parser.add_argument('--pairwise-grasp-direction',type=float,default=12.)
    parser.add_argument('--tool',choices=('standard','compact'),default='standard')
    parser.add_argument('--min-grasp-width',type=float,default=.015)
    args=parser.parse_args()
    find(args.object,args.pairs,args.candidate_budget,resume=not args.fresh,
         anchor_index=args.anchor_index,anchor_yaw=args.anchor_yaw,grasp_method=args.grasp_method,
         com_weight=args.com_weight,soft_pads=args.soft_pads,ideal_grasp=args.ideal_grasp,
         pairwise_grasp_direction_deg=args.pairwise_grasp_direction,tool=args.tool,
         min_grasp_width=args.min_grasp_width,pose_count=args.pose_count)
