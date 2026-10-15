"""Audit full R6 coverage, immutable projections and the one-world-pose movie."""
from pathlib import Path
import json
import subprocess

import numpy as np
from scipy.optimize import linprog

import r6_satisfaction_demo as demo
from whole_search.model import Model,Layout
from material_force import certify_material_contacts


def require(value,message):
    if not value:raise AssertionError(message)


def main():
    e=demo.Experiment()
    rows=json.loads((demo.DATA/'states.json').read_text())
    evidence=json.loads((demo.DATA/'evidence.json').read_text())
    video=json.loads((demo.DATA/'video.json').read_text())
    for field in ['inputs_sha256','sources_sha256']:
        for path,digest in evidence[field].items():
            require(demo.old.sha(demo.old.ROOT/path)==digest,'Changed provenance source: '+path)
    with np.load(demo.DATA/'demand_projection.npz') as z:
        physical=np.array([t.targets/t.scale for t in e.tasks])
        np.testing.assert_array_equal(z['physical_demands_world_com'],physical)
        np.testing.assert_allclose(z['projected_demands'],
            (physical*e.scale-z['mean'])@z['projection'].T,atol=1e-13)
        np.testing.assert_allclose(z['projection']@z['projection'].T,np.eye(2),atol=1e-12)
        require(z['projected_demands'].shape==(4,32768,2),'Incomplete demand projection')
    minimal_floor_height=float('inf')
    for row in rows:
        require(demo.old.sha(demo.DATA/row['cache'])==evidence['cache_sha256'][row['cache']],
                'Changed full-demand force mask')
        with np.load(demo.DATA/row['cache']) as z:
            masks=z['masks'];directions=z['directions'];placements=z['placements']
            require(masks.shape==(4,32768) and masks.dtype==bool,'Each pose needs its complete original mask')
            np.testing.assert_array_equal(masks.sum(1),row['counts'])
            np.testing.assert_array_equal(z['native_world'],e.native)
            np.testing.assert_allclose(np.linalg.norm(directions,axis=1),1.,atol=1e-12)
            world_exits=np.einsum('kij,kj->ki',e.native[:,:3,:3],directions)
            require(world_exits[:,2].min()>=-1e-12,'A world exit crosses its floor')
            for k in range(4):
                actual=e.native[k]@placements[k]
                np.testing.assert_allclose(actual[:3,:3],e.native[k,:3,:3],atol=1e-12)
                if k!=demo.ACTIVE:np.testing.assert_array_equal(placements[k],np.eye(4))
            if row['kind']=='exit':
                np.testing.assert_array_equal(placements,np.tile(np.eye(4),(4,1,1)))
                np.testing.assert_allclose(directions[:3],e.original.target[:3],atol=1e-12)
            else:
                np.testing.assert_allclose(directions,e.original.target,atol=1e-12)
                actual=e.native[demo.ACTIVE]@placements[demo.ACTIVE]
                np.testing.assert_allclose(actual[:3,3]-e.native[demo.ACTIVE,:3,3],
                                           row['world_translation_m'],atol=1e-12)
                require(abs(actual[2,3]-e.native[demo.ACTIVE,2,3])<1e-12,
                        'The horizontal translation changed height')
        with np.load(demo.HERE/row['geometry']) as z:
            mesh=demo.trimesh.Trimesh(z['vertices'],z['faces'],process=False)
            require(mesh.is_watertight,'An actual displayed boundary is open')
            floor=mesh.vertices@e.native[:,2,:3].T+e.native[:,2,3]
            minimal_floor_height=min(minimal_floor_height,float(floor.min()))
            require(floor.min()>=-1e-8,'Displayed material is below a final installation floor')

    model=Model(e.poses,initialization_report=e.original.base/'step3/step3.1/data/report.json',
                allow_z_translation=True)
    sampled=[];independent=[];certificates={}
    choices=[r for r in rows if (r['kind']=='exit' and r['source_index'] in [72,84,96])
             or (r['kind']=='translation' and r.get('translation_step') in [6,12])]
    for choice_index,row in enumerate(choices):
        with np.load(demo.DATA/row['cache']) as z:
            layout=Layout(z['placements'],z['directions'],np.arange(4),tuple(range(4)))
            expected=z['masks'].copy()
        with np.load(demo.HERE/row['geometry']) as z:
            mesh=demo.trimesh.Trimesh(z['vertices'],z['faces'],process=False)
        proof=certify_material_contacts(model,layout,mesh)
        for k,task in enumerate(model.tasks):
            np.testing.assert_array_equal(proof['masks'][k],expected[k])
            rays=proof['supplies'][k]
            magnitude=np.linalg.norm((task.targets/task.scale)[:,3:],axis=1)
            for supported in [True,False]:
                candidates=np.flatnonzero(expected[k]==supported)
                if not len(candidates):continue
                index=int(candidates[np.argmax(magnitude[candidates])])
                target=demo.old.C.U.target(task.targets[index])
                result=linprog(np.ones(len(rays)),A_eq=rays.T,b_eq=target,bounds=(0,None),
                    method='highs-ds',options=dict(presolve=False,
                    primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
                require(bool(result.success)==supported,'Independent equilibrium differs from the point color')
                record=dict(configuration=row['cache'],pose=e.poses[k],sample_index=index,
                            supported=supported,solver_status=int(result.status))
                if supported:
                    error=float(np.max(np.abs(np.asarray(result.x,np.longdouble)@
                        np.asarray(rays,np.longdouble)-target)))
                    require(error<=2e-9 and result.x.min()>=0.,'Original seven-equation balance fails')
                    record['original_equilibrium_residual']=error
                    prefix=f'configuration{choice_index}_pose{k}'
                    certificates[prefix+'_rays']=rays
                    certificates[prefix+'_target']=target
                    certificates[prefix+'_coefficients']=result.x
                independent.append(record)
        sampled.append(dict(configuration=row['cache'],counts=expected.sum(1).tolist(),
                            original_masks_rechecked=4*32768))
        print('RECHECKED',row['cache'],expected.sum(1).tolist(),flush=True)
    np.savez_compressed(demo.DATA/'independent_equilibrium_witnesses.npz',**certificates)
    original=json.loads((demo.HERE/'assets/data/manifest.json').read_text())
    for path,digest in original['protected_inputs'].items():
        require(demo.old.sha(demo.old.ROOT/path)==digest,'An original representation source changed')
    for path,digest in original['outputs'].items():
        require(demo.old.sha(demo.HERE/path)==digest,'An original representation figure changed')
    movie=demo.HERE/video['file']
    require(demo.old.sha(movie)==video['sha256'],'Changed video bytes')
    require(demo.old.sha(Path(demo.__file__))==video['source_sha256'],'Changed rendering source')
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=codec_name,width,height,pix_fmt,nb_frames:format=duration','-of','json',str(movie)],
        capture_output=True,text=True,check=True)
    media=json.loads(result.stdout);stream=media['streams'][0]
    require([stream['width'],stream['height']]==[1920,1080],'Wrong video dimensions')
    require(int(stream['nb_frames'])==video['frame_count'],'Wrong video frame count')
    subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(movie),'-f','null','-'],check=True)
    record=dict(complete=True,passed=True,configurations_checked=len(rows),
        unique_configurations=len(set(r['cache'] for r in rows)),
        fixed_demand_sets_and_fixed_projection_verified=True,
        colors_and_counts_use_full_32768_original_demands_per_pose=True,
        only_pose27_exit_or_world_horizontal_translation_changes=True,
        all_original_rotations_and_other_task_positions_preserved=True,
        all_displayed_boundaries_watertight=True,minimum_floor_height_m=minimal_floor_height,
        representative_configurations_full_rechecks=sampled,
        independent_point_color_equilibrium_LPs=independent,
        original_representation_sources_unchanged=len(original['protected_inputs']),
        original_representation_figures_unchanged=list(original['outputs']),
        complete_video_decode_passed=True,video_metadata=media,video_sha256=demo.old.sha(movie),
        verification_source_sha256=demo.old.sha(__file__))
    demo.old.save_json(demo.DATA/'verification.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='independent_point_color_equilibrium_LPs'},indent=2))


if __name__=='__main__':main()
