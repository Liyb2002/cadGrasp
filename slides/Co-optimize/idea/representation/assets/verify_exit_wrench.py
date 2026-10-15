"""Audit the actual saved exit/wrench demonstration and independent endpoint LPs."""
from pathlib import Path
import json
import subprocess

import numpy as np
from scipy.optimize import linprog

import exit_wrench_demo as demo


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    folder = demo.DATA
    evidence = json.loads((folder/'evidence.json').read_text())
    rows = json.loads((folder/'states.json').read_text())
    experiment = demo.Experiment(evidence['pose_set'])
    native = np.asarray(evidence['fixed_original_native_world'])
    wrenches = np.array([row['applied_external_wrench_world'] for row in evidence['selection']])
    require(not all(rows[0]['supported']), 'The initial state must fail at least one displayed load')
    require(all(rows[-1]['supported']), 'The final state must support every displayed load')
    previous = None
    minimum_floor_height = float('inf')
    for row in rows:
        source = folder/row['geometry']
        require(demo.sha(source) == row['geometry_sha256'], 'Saved geometry hash differs')
        with np.load(source) as z:
            np.testing.assert_array_equal(z['native_world'], native)
            np.testing.assert_array_equal(z['applied_wrenches_world'], wrenches)
            np.testing.assert_array_equal(z['supported'], row['supported'])
            directions = z['directions'].copy()
            np.testing.assert_allclose(directions, row['directions_material'], atol=1e-14)
            np.testing.assert_allclose(np.linalg.norm(directions,axis=1),1.,atol=1e-12)
            world = np.einsum('kij,kj->ki', native[:,:3,:3], directions)
            np.testing.assert_allclose(world,row['directions_world'],atol=1e-14)
            require(world[:,2].min() >= -1e-12, 'An exit crosses its installation floor')
            if previous is not None:
                changed = np.flatnonzero(np.linalg.norm(directions-previous,axis=1)>1e-12)
                require(len(changed)<=1, 'Multiple exits change at the same time')
                require(not len(changed) or int(changed[0])==row['active_pose_index'],
                        'The highlighted task differs from the changing exit')
            previous = directions
            if len(z['vertices']):
                heights = z['vertices'] @ native[:,2,:3].T + native[:,2,3]
                minimum_floor_height = min(minimum_floor_height,float(heights.min()))
                require(heights.min() >= -1e-8, 'Saved material is below an installation floor')
                require(row['retained_component_count']==1 and row['watertight'],
                        'Nonempty displayed support must be one closed material component')
    np.testing.assert_allclose(previous,experiment.target,atol=1e-12)
    np.testing.assert_allclose(native,experiment.native,atol=1e-14)
    inputs_checked = 0
    for path, expected in evidence['input_sha256'].items():
        require(demo.sha(demo.ROOT/path)==expected,'An original task/result input changed: '+path)
        inputs_checked += 1
    original = json.loads((demo.HERE/'assets/data/manifest.json').read_text())
    for path, expected in original['protected_inputs'].items():
        require(demo.sha(demo.ROOT/path)==expected,'An existing representation source changed: '+path)
    for name, expected in original['outputs'].items():
        require(demo.sha(demo.HERE/name)==expected,'An existing representation figure changed: '+name)

    # Solve each displayed endpoint wrench afresh on the actual serialized
    # boundary. No contact samples or force status flags enter these LPs.
    endpoints = []
    certificate = {}
    for name, row in [('initial',rows[0]),('final',rows[-1])]:
        with np.load(folder/row['geometry']) as z:
            mesh = demo.trimesh.Trimesh(z['vertices'],z['faces'],process=False)
            directions = z['directions'].copy()
        supplies, _ = experiment.supplies(mesh,directions)
        for k, task in enumerate(experiment.tasks):
            selected = evidence['selection'][k]['original_sample_index']
            np.testing.assert_allclose(-task.targets[selected]/task.scale,wrenches[k],atol=1e-14)
            rays = supplies[k]
            target = demo.C.U.target(task.targets[selected])
            result = linprog(np.ones(len(rays)),A_eq=rays.T,b_eq=target,
                             bounds=(0,None),method='highs-ds',
                             options=dict(presolve=False,primal_feasibility_tolerance=1e-9,
                                          dual_feasibility_tolerance=1e-9))
            require(bool(result.success)==bool(row['supported'][k]),
                    'Independent LP differs from the displayed endpoint status')
            record = dict(endpoint=name,pose=experiment.poses[k],
                          original_sample_index=selected,supported=bool(result.success),
                          solver_status=int(result.status),reaction_ray_count=len(rays))
            if result.success:
                reaction = np.asarray(result.x,np.longdouble)
                error = float(np.max(np.abs(reaction @ np.asarray(rays,np.longdouble)-target)))
                require(error <= 2e-9 and reaction.min()>=0.,'Original seven-equation equilibrium fails')
                record['maximum_original_equilibrium_residual'] = error
                record['minimum_nonnegative_reaction_coefficient'] = float(reaction.min())
                certificate[f'{name}_{k}_rays'] = rays
                certificate[f'{name}_{k}_target'] = target
                certificate[f'{name}_{k}_coefficients'] = result.x
            endpoints.append(record)
    np.savez_compressed(folder/'displayed_wrench_certificate.npz',**certificate)

    video = demo.HERE/'exit_wrench_coupling.mp4'
    probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,pix_fmt,nb_frames,r_frame_rate:format=duration',
        '-of','json',str(video)],capture_output=True,text=True,check=True)
    metadata = json.loads(probe.stdout)
    stream = metadata['streams'][0]
    recorded_video = json.loads((folder/'video.json').read_text())
    require(demo.sha(video)==recorded_video['sha256'],'Video hash differs')
    require([stream['width'],stream['height']]==[1920,1080],'Unexpected video dimensions')
    require(stream['pix_fmt']=='yuv420p' and stream['codec_name']=='h264','Unexpected video encoding')
    require(int(stream['nb_frames'])==recorded_video['frame_count'],'Frame count differs')
    subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True)
    result = dict(complete=True,passed=True,states_checked=len(rows),
        fixed_original_poses_and_positions=True,fixed_original_applied_wrenches=True,
        only_one_exit_changes_per_state=True,all_world_exits_above_floor=True,
        all_nonempty_supports_watertight_and_one_material_component=True,
        minimum_material_height_over_all_installation_floors_m=minimum_floor_height,
        original_inputs_unchanged=inputs_checked,
        preexisting_representation_sources_unchanged=len(original['protected_inputs']),
        preexisting_representation_figures_unchanged=list(original['outputs']),
        initial_supported_wrench_count=sum(rows[0]['supported']),
        final_supported_wrench_count=sum(rows[-1]['supported']),
        independent_endpoint_equilibrium_LPs=endpoints,
        scope=evidence['physical_scope'],video=metadata,
        complete_video_decode_passed=True,video_sha256=demo.sha(video),
        generating_script_sha256=demo.sha(Path(demo.__file__)),
        verification_script_sha256=demo.sha(__file__))
    demo.save_json(folder/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
