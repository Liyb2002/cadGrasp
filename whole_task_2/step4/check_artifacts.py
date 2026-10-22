"""Check actual checkpoint files, immutable upstream inputs and one-image output."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from whole_task_2.step1.common import ROOT,DEFAULT_OUTPUT,sha256,write_json
from whole_task_2.step1.needle_geometry import geometric_mask
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.operations import Layout
from whole_task_2.operations.construction import CompactConstructionModel
from whole_task_2.operations.state import transform
from whole_task_2.operations.installations import standing_pairs


def check(folder,output_root=DEFAULT_OUTPUT,name='B'):
    folder=Path(folder).resolve();output_root=Path(output_root).resolve()
    report=json.loads((folder/'report.json').read_text())
    validation=json.loads((folder/'validation.json').read_text())
    current=json.loads((folder/'current.json').read_text())
    model=CompactConstructionModel(output_root,name)
    assert report['status']==validation['status']=='passed'
    assert report['checkpoint_manifest_sha256']==sha256(folder/'current.json')
    assert report['validation_sha256']==sha256(folder/'validation.json')
    assert report['support_obj_sha256']==sha256(folder/'support.obj')
    assert validation['fresh_disk_replay'] and validation['original_sample_count']==32768*len(model.ids)
    for path,digest in report['source_sha256'].items():assert sha256(path)==digest,path
    for path,digest in model.report['input_sha256'].items():assert sha256(path)==digest,path
    for path,digest in model.report['artifact_sha256'].items():assert sha256(model.folder/path)==digest,path
    rows=[];previous=float('inf')
    for index in range(current['accepted']+1):
        file=folder/f'state_{index:03}.npz'
        meta=json.loads(file.with_suffix('.json').read_text())
        assert sha256(file)==meta['mesh_sha256']
        with np.load(file)as z:
            q=Layout.read(z)
            mesh=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
        volume=float(mesh.volume*1e6)
        assert abs(volume-meta['volume_cm3'])<1e-7
        assert volume<previous-1e-5
        assert mesh.is_watertight and mesh.is_winding_consistent and volume>0
        solid=G.solid(mesh)
        assert sum(p.volume()*G.SCALE**3>1e-15 for p in solid.decompose())==1
        if index==0:
            np.testing.assert_array_equal(mesh.vertices,model.initial_support.vertices)
            np.testing.assert_array_equal(mesh.faces,model.initial_support.faces)
        else:
            proof=meta['audit']
            assert proof['passed'] and proof['full_loads'] and proof['source_face_prisms_checked']
            assert len(proof['tasks'])==len(model.ids)==len(proof['continuous_paths'])
            assert {t['id']for t in proof['tasks']}==set(model.ids)
            assert {t['id']for t in proof['continuous_paths']}==set(model.ids)
            assert all(p['passed']for p in proof['continuous_paths'])
            for task in proof['tasks']:
                f=task['force']
                assert f['passed'] and not f.get('preflight_only',False)
                assert f['original_sample_count']==f['passed_original_sample_count']==32768
                assert f['continuous_outer_target_count']==f['passed_continuous_outer_target_count']>0
                assert f['classification_tolerance']==2e-9
        for k,record in enumerate(model.records):
            pose,M=q.poses[k],q.material[k]
            for T in(pose,M):
                np.testing.assert_allclose(T[3],[0,0,0,1],atol=1e-12,rtol=0)
                np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-12,rtol=0)
                assert abs(np.linalg.det(T[:3,:3])-1)<1e-12
            assert abs(np.linalg.norm(q.exits[k])-1)<1e-12 and q.exits[k,2]>=.05
            object_head=(model.head-pose[:3,3])@pose[:3,:3]
            object_axis=pose[:3,:3].T@model.axis
            assert geometric_mask(model.source,object_head,object_axis,model.workspace)[record['working_face_ids']].all()
            assert not model.source.contains(object_head[None])[0]
            assert transform(model.source.vertices,pose)[:,2].min()>=-1e-12
            world=transform(mesh.vertices,np.linalg.inv(M))
            assert -1e-9<=world[:,2].min()<=1e-9
        rows.append(dict(index=index,operation=meta['operation'],volume_cm3=volume,sha256=sha256(file)))
        previous=volume
    assert abs(previous-report['final_volume_cm3'])<1e-7
    if report.get('design_requirements',{}).get('final_standing_installation_required'):
        assert report['design_requirements_sha256']==sha256(folder/'requirements.json')
        assert standing_pairs(q), 'Final result must actually install the complete fixture on its side'
        assert report['final_uses_90_degree_side_installation']
        assert report['side_installation_task_pairs']==[[model.ids[i],model.ids[j]]for i,j in standing_pairs(q)]
        for pair in report['shared_object_seat_task_pairs']:
            a,b=[model.ids.index(name)for name in pair['tasks']]
            np.testing.assert_allclose(q.material[a]@q.poses[a],q.material[b]@q.poses[b],atol=1e-8,rtol=0)
    old=json.loads((ROOT/'whole_task_2/data'/name/'step4_spatial_best/report.json').read_text())
    assert previous<old['final_volume_cm3']-1e-5
    out=output_root/name/'step4'
    assert [p.name for p in out.iterdir()]==['overview.png']
    assert sha256(out/'overview.png')==report['overview_sha256']
    with Image.open(out/'overview.png')as img:assert img.format=='PNG' and img.size==(3500,3090)
    videos=ROOT/'whole_task_2/operations'
    checked=[]
    for filename in('translation.mp4','exit_redirection.mp4','pose_rotation.mp4','juxtaposition.mp4','rotational_reuse.mp4'):
        path=videos/filename
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True,capture_output=True)
        checked.append(dict(file=filename,sha256=sha256(path),full_decode_passed=True))
    final=dict(status='passed',all_checkpoint_meshes_and_metadata_bound=True,
        actual_volumes_strictly_decrease=True,all_checkpoint_original_regions_and_head_clearances=True,
        all_checkpoint_proper_installations_and_actual_floors=True,
        task_count=len(model.ids),checkpoints=rows,upstream_input_count=len(model.report['input_sha256']),
        final_disk_replay=validation['fresh_disk_replay'],original_sample_count=validation['original_sample_count'],
        previous_best_volume_cm3=old['final_volume_cm3'],final_volume_cm3=previous,
        additional_reduction_fraction=1-previous/old['final_volume_cm3'],
        output_files=['overview.png'],videos=checked)
    write_json(folder/'final_artifact_check.json',final)
    print('STEP4 ARTIFACT CHECK PASS',len(rows),'states;',round(previous,3),'cm3;',len(checked),'videos',flush=True)
    return final


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--object',default='B')
    a=p.parse_args();check(a.workdir,a.output_root,a.object)


if __name__=='__main__':main()
