"""Record independent geometry and release probes in existing sequence manifests.

No videos, images or sidecar reports are produced. Probe branches never alter the
saved continuous trajectory. Run after generating or replacing an object's data.
"""
import sys
sys.dont_write_bytecode=True
import argparse
import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import grasp as G
from sequence_export import digest,write
from verify_sequences import verify


def audit(name):
    folder=G.ROOT/'objects'/name
    verify(folder)
    manifest=folder/'poses.json'
    record=json.loads(manifest.read_text())
    tool=G.ROOT/record['grasp']['model']
    model=G.build(name,hand_xml=tool,floor_hull=True)
    data=mujoco.MjData(model)
    with np.load(folder/'trajectory.npz') as z:
        worst=0.;contact_count=0
        for q,pos,quat in zip(z['qpos'],z['mocap_pos'],z['mocap_quat']):
            data.qpos[:]=q;data.mocap_pos[:]=pos;data.mocap_quat[:]=quat
            mujoco.mj_forward(model,data)
            for contact in data.contact:
                names=[model.geom(int(g)).name for g in (contact.geom1,contact.geom2)]
                if 'floor' in names and any(n.startswith('hand_geom_') for n in names):
                    contact_count+=1;worst=min(worst,float(contact.dist))
        if worst<-.0002:raise ValueError(f'{name}: gripper penetrates floor by {-worst} m')
        for g in range(model.ngeom):
            if model.geom(g).name.startswith('hand_geom_'):
                model.geom_contype[g]=0;model.geom_conaffinity[g]=0
        releases=[]
        for i,t in enumerate(z['target_times_s']):
            frame=max(0,int(np.searchsorted(z['time_s'],t,side='right'))-1)
            data=mujoco.MjData(model)
            data.qpos[:]=z['qpos'][frame]
            data.mocap_pos[:]=z['mocap_pos'][frame];data.mocap_quat[:]=z['mocap_quat'][frame]
            mujoco.mj_forward(model,data)
            initial=G.transform(data,model.body('object').id)
            for _ in range(1000):mujoco.mj_step(model,data)
            final=G.transform(data,model.body('object').id)
            angle=np.rad2deg(Rotation.from_matrix(final[:3,:3]@initial[:3,:3].T).magnitude())
            releases.append(dict(pose_id=f'pose_{i+1}',source_frame=frame,
                unheld_rotation_after_1s_deg=float(angle),
                translation_after_1s_m=float(np.linalg.norm(final[:3,3]-initial[:3,3]))))
        result=dict(generator='codes/setup/audit_sequences.py',generator_sha256=digest(__file__),
            trajectory_sha256=digest(folder/'trajectory.npz'),tool_model_sha256=digest(tool),
            gripper_floor_collision=dict(sample_frames=len(z['qpos']),contact_count=contact_count,
                maximum_penetration_m=-worst,collision_tolerance_m=.0002),
            unheld_release_probe=dict(scope='Independent 1 s free evolution from the last stored frame at or before each target time, zero initial velocities, gripper collisions disabled. Uses saved convex decomposition and exact raw-mesh support height. These branches do not alter the saved trajectory.',results=releases))
    record.setdefault('verification',{})['independent_audit']=result
    write(manifest,record)
    for pose in record['poses']:
        task=folder/'tasks'/pose['pose_id']
        path=task/'setup.npz'
        with np.load(path) as z:fields=dict(z)
        fields['poses_sha256']=digest(manifest)
        np.savez_compressed(path,**fields)
        report=json.loads((task/'setup.json').read_text())
        report['source_snapshot_sha256']=digest(path)
        write(task/'setup.json',report)
    verify(folder)
    print(json.dumps(dict(object=name,maximum_gripper_floor_penetration_m=-worst,
        minimum_unheld_rotation_after_1s_deg=min(r['unheld_rotation_after_1s_deg'] for r in releases))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects',nargs='*')
    args=parser.parse_args()
    names=args.objects or [p.name for p in sorted((G.ROOT/'objects').iterdir()) if (p/'trajectory.npz').exists()]
    for name in names:audit(name)
