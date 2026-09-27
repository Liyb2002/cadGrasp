"""Check saved sequence/task/trajectory consistency without regenerating data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from sequence_export import WORK_AREA_FRACTION

ROOT=Path(__file__).resolve().parents[2]


def verify_segments(folder,data):
    index=json.loads((folder/data['trajectory_segments']).read_text())
    digest=hashlib.sha256((folder/'trajectory.npz').read_bytes()).hexdigest()
    assert index['full_trajectory_sha256']==digest
    assert len(index['segments'])==10
    frames=[];robot_frames=[];robot_values=[]
    with np.load(folder/'trajectory.npz') as full:
        for i,row in enumerate(index['segments']):
            with np.load(folder/'trajectories'/row['file']) as segment:
                ids=segment['full_frame_indices']
                for key in ('qpos','mocap_pos','mocap_quat','T_world_object'):
                    np.testing.assert_array_equal(segment[key],full[key][ids])
                for key in ('ideal_grasp_active','ideal_grasp_eq_data'):
                    if key in full:np.testing.assert_array_equal(segment[key],full[key][ids])
                np.testing.assert_array_equal(segment['sequence_time_s'],full['time_s'][ids])
                frames.extend(ids if i==0 else ids[1:])
                for frame,q in zip(segment['robot_frame_indices']+ids[0],segment['robot_q']):
                    if robot_frames and frame==robot_frames[-1]:continue
                    robot_frames.append(frame);robot_values.append(q)
                assert str(segment['full_trajectory_sha256'])==digest
                assert data['transitions'][i]['trajectory_file']=='trajectories/'+row['file']
        np.testing.assert_array_equal(frames,np.arange(len(full['qpos'])))
        np.testing.assert_array_equal(robot_frames,full['robot_frame_indices'])
        np.testing.assert_array_equal(robot_values,full['robot_q'])
    if 'video' in data:
        assert hashlib.sha256((folder/data['video']).read_bytes()).hexdigest()==data['video_sha256']
        assert data['video_trajectory_sha256']==digest


def verify(folder):
    manifest=folder/'poses.json'
    data=json.loads(manifest.read_text())
    assert data.get('schema')=='cadgrasp_sequence_v1',f'{folder.name}: sequence missing'
    expected=[f'pose_{i}' for i in range(1,11)]
    assert data['order']==['rest']+expected
    assert [p['pose_id'] for p in data['poses']]==expected
    assert json.loads((folder/'tasks.json').read_text())['poses']==expected
    assert sorted(p.name for p in (folder/'tasks').iterdir() if p.name!='.DS_Store')==sorted(expected)
    raw=trimesh.load(folder/'mesh.stl',force='mesh')
    pose_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    mesh_hash=hashlib.sha256((folder/'mesh.stl').read_bytes()).hexdigest()
    assert data['mesh_sha256']==mesh_hash
    targets=np.array([p['T_world_mesh'] for p in data['poses']])
    pair_angles=Rotation.from_matrix(np.array([a[:3,:3]@b[:3,:3].T
        for i,a in enumerate(targets) for b in targets[:i]])).magnitude()
    assert np.degrees(pair_angles).min()>.5,f'{folder.name}: duplicate target orientations'
    if data.get('trajectory_revision')=='diverse_regrasp_v1':
        from regrasp_sequence import angle,differences
        rule=data['pose_selection'];grasps=data['grasps']
        assert len(grasps)==10
        gravity=targets[:,2,:3]
        for i in range(10):
            assert data['transitions'][i]['grasp_id']==grasps[i]['grasp_id']
            for j in range(i):
                assert angle(gravity[i],gravity[j])>=rule['minimum_pairwise_gravity_direction_deg']-1e-8
                distance,approach,closing=differences(grasps[i],grasps[j])
                assert distance>=rule['minimum_pairwise_contact_change_m']-1e-8
                assert max(approach,closing)>=rule['minimum_pairwise_grasp_direction_deg']-1e-8
            if i:
                assert angle(gravity[i],gravity[i-1])>=rule['minimum_adjacent_gravity_direction_deg']-1e-8
                distance,approach,closing=differences(grasps[i],grasps[i-1])
                assert distance>=rule['minimum_adjacent_contact_change_m']-1e-8
                assert max(approach,closing)>=rule['minimum_adjacent_grasp_direction_deg']-1e-8
                if 'measured' in grasps[i] and 'measured' in grasps[i-1]:
                    distance,approach,closing=differences(grasps[i]['measured'],grasps[i-1]['measured'])
                    assert distance>=rule.get('minimum_measured_adjacent_contact_change_m',rule['minimum_adjacent_contact_change_m'])-1e-8
                    assert max(approach,closing)>=rule.get('minimum_measured_adjacent_grasp_direction_deg',rule['minimum_adjacent_grasp_direction_deg'])-1e-8
        releases=[e for e in data['regrasp_events'] if e['event']=='released_on_stable_rest']
        pickups=[e for e in data['regrasp_events'] if e['event']=='new_grasp']
        assert len(releases)==9 and len(pickups)==10
        for i,release in enumerate(releases):
            assert release['finger_contacts']==0 and release['object_floor_normal_force_N']>.001
            if release.get('stability_metric')=='gravity_direction':
                assert data.get('grasp_contact_dynamics_simulated') is False
                assert release['unheld_tilt_drift_deg']<=.3 and release['residual_yaw_tracked']
            else:assert release['unheld_rotation_drift_deg']<=.3
            if 'unheld_gravity_direction_range_deg' in release:
                assert release['unheld_gravity_direction_range_deg']<=.3
            assert data['transitions'][i]['end_time_s']<=release['start_time_s']
            assert release['end_time_s']<=pickups[i+1]['start_time_s']
    for i,(pose,transition) in enumerate(zip(data['poses'],data['transitions'])):
        T=np.array(pose['T_world_mesh']);R=T[:3,:3]
        np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-10)
        assert np.linalg.det(R)>0.999999
        points=trimesh.transform_points(raw.vertices,T)
        assert abs(points[:,2].min())<1e-9
        assert len(np.flatnonzero(points[:,2]<1e-8))==1
        assert transition['source']==('rest' if i==0 else expected[i-1])
        assert transition['destination']==expected[i]
        assert transition['bilateral_grip'] or (
            data.get('grasp_contact_dynamics_simulated') is False and transition.get('ideal_grasp_active'))
        assert transition['object_floor_normal_force_N']>.001
        assert -.001<transition['raw_mesh_floor_gap_m']<.0003
        assert transition['target_position_error_m']<.003 and transition['target_rotation_error_deg']<3
        if i:assert transition['start_time_s']==data['transitions'][i-1]['end_time_s']
        with np.load(folder/'tasks'/expected[i]/'setup.npz') as z:
            assert str(z['poses_sha256'])==pose_hash and str(z['mesh_sha256'])==mesh_hash
            np.testing.assert_allclose(z['T_world_mesh'],T,atol=1e-12)
            np.testing.assert_allclose(z['com_m'],R@raw.center_mass+T[:3,3],atol=1e-12)
            np.testing.assert_allclose(z['floor_contact_m'],points[np.argmin(points[:,2])],atol=1e-9)
            assert np.linalg.norm(z['com_m'][:2]-z['floor_contact_m'][:2])>.001
            mask=z['work_faces'].copy()
        report=json.loads((folder/'tasks'/expected[i]/'setup.json').read_text())
        assert report['source_snapshot_sha256']==hashlib.sha256(
            (folder/'tasks'/expected[i]/'setup.npz').read_bytes()).hexdigest()
        assert WORK_AREA_FRACTION[0]<=report['checks']['work_area_fraction']<=WORK_AREA_FRACTION[1]
        assert report['checks']['work_components']==1
        mesh=raw
        for _ in range(report['uniform_subdivision_rounds']):mesh=mesh.subdivide()
        assert len(mask)==len(mesh.faces) and mask.dtype==bool
        fraction=mesh.area_faces[mask].sum()/mesh.area
        np.testing.assert_allclose(fraction,report['checks']['work_area_fraction'],atol=1e-12)
        world=trimesh.transform_points(mesh.vertices,T)
        assert world[mesh.faces[mask],2].min()>.0015
        assert (mesh.face_normals[mask]@T[2,:3]).min()>.35
        ids=np.flatnonzero(mask)
        pairs=mesh.face_adjacency[mask[mesh.face_adjacency].all(axis=1)]
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        remap=np.full(len(mask),-1);remap[ids]=np.arange(len(ids))
        edges=remap[pairs]
        graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(ids),len(ids)))
        assert connected_components(graph,directed=False,return_labels=False)==1
    with np.load(folder/'trajectory.npz') as z:
        assert len(z['target_transforms'])==10
        np.testing.assert_allclose(z['time_s'],.033*(np.arange(len(z['time_s']))+1),atol=1e-9)
        np.testing.assert_allclose(z['target_transforms'],[p['T_world_mesh'] for p in data['poses']],atol=1e-12)
        np.testing.assert_allclose(z['T_world_object'][:,:3,3],z['qpos'][:,9:12],atol=1e-10)
        rotations=Rotation.from_quat(z['qpos'][:,[13,14,15,12]]).as_matrix()
        np.testing.assert_allclose(z['T_world_object'][:,:3,:3],rotations,atol=1e-10)
        assert len(z['robot_q'])==len(z['robot_frame_indices'])
        assert z['robot_frame_indices'][-1]==len(z['qpos'])-1
        assert np.isfinite(z['qpos']).all() and np.isfinite(z['robot_q']).all()
        if data.get('grasp_contact_dynamics_simulated') is False:
            assert len(z['ideal_grasp_active'])==len(z['qpos'])
            for event in data['regrasp_events']:
                if event['event']=='released_on_stable_rest':
                    ids=(z['time_s']>=event['start_time_s']+.033)&(z['time_s']<=event['end_time_s'])
                    assert ids.any() and not z['ideal_grasp_active'][ids].any()
                elif event['event']=='new_grasp':
                    assert event['bilateral_closure_verified']
        for event in data.get('regrasp_events',[]):
            if 'unheld_gravity_direction_range_deg' not in event:continue
            ids=np.flatnonzero((z['time_s']>=event['unheld_dwell_start_time_s']-1e-9)
                              &(z['time_s']<=event['unheld_dwell_end_time_s']+1e-9))
            assert len(ids)>=2
            gravity=z['T_world_object'][ids,2,:3]
            observed=float(np.degrees(np.arccos(np.clip(gravity@gravity.T,-1.,1.))).max())
            assert observed<=.3 and observed<=event['unheld_gravity_direction_range_deg']+1e-6
        for transition in data['transitions']:
            window=transition.get('settling_window')
            if window is None:continue
            assert data.get('grasp_contact_dynamics_simulated') is False
            ids=np.flatnonzero((z['time_s']>=window['start_time_s']-1e-9)
                              &(z['time_s']<=window['end_time_s']+1e-9))
            assert len(ids)==window['samples']==17
            duration=float(z['time_s'][ids[-1]]-z['time_s'][ids[0]])
            assert duration>=.5 and abs(duration-window['duration_s'])<1e-9
            assert z['ideal_grasp_active'][ids].all()
            held=z['T_world_object'][ids]
            translation=float(np.linalg.norm(held[:,None,:3,3]-held[None,:,:3,3],axis=-1).max())
            relative=held[:,None,:3,:3]@np.swapaxes(held[None,:,:3,:3],-1,-2)
            rotation=float(np.degrees(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude()).max())
            assert translation<=.00005 and rotation<=.05
            assert abs(translation-window['translation_range_m'])<1e-10
            assert abs(rotation-window['rotation_range_deg'])<1e-8
        endpoint_ids=np.argmin(abs(z['time_s'][:,None]-z['target_times_s']),axis=0)
        assert np.max(abs(z['time_s'][endpoint_ids]-z['target_times_s']))<=.033001
        actual=z['T_world_object'][endpoint_ids]
        measured_angles=Rotation.from_matrix(np.array([a[:3,:3]@b[:3,:3].T
            for i,a in enumerate(actual) for b in actual[:i]])).magnitude()
        assert np.degrees(measured_angles).min()>.5,f'{folder.name}: actual holds not distinct'
    assert data['kuka_checks']['scene_checks']['maximum_penetration_m']<=.0002
    if 'trajectory_segments' in data:verify_segments(folder,data)
    return 10


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    count=0;skipped=[]
    for folder in sorted((ROOT/'objects').iterdir()):
        if not folder.is_dir() or not (folder/'mesh.stl').exists():continue
        data=json.loads((folder/'poses.json').read_text())
        if args.partial and data.get('schema')!='cadgrasp_sequence_v1':
            skipped.append(folder.name);continue
        count+=verify(folder)
    print(json.dumps(dict(verified_objects=count//10,verified_targets=count,pending=skipped)))
