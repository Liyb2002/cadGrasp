"""Replay diagnostic witnesses in raw task frames and decode all exit videos."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.exit_support_geometry import read_case
from step4_connect_support import build_coupled_saddle as S


def review(name):
    base=I.OUTPUTS/name
    groups=sorted(p for p in base.glob('pose*+*') if (p/'step4/data/report.json').exists())
    results=[];inputs=[]
    for group in groups:
        output=group/'step4'
        physical=I.check_report(output/'data/report.json')
        case=read_case(output)
        published=I.check_report(output/'data/exit_replay.json')
        support_record=I.check_report(output/'data/exit_replay/support.json')
        support=trimesh.load(output/'exit_support.obj',force='mesh',process=False)
        assert published['poses']==physical['poses']
        assert published['all_poses_included'] and published['diagnostic_only']
        assert len(S.solid(support).decompose())==1 and support.is_watertight and support.is_winding_consistent
        assert support_record['support']['diagnostic_candidate']==(not physical['constructed'])
        if physical['constructed']:
            saved=trimesh.load(output/'shape.obj',force='mesh',process=False)
            np.testing.assert_array_equal(saved.faces,support.faces)
            np.testing.assert_allclose(saved.vertices,support.vertices,atol=1e-14,rtol=0)
        inputs.extend([output/'data/report.json',output/'data/exit_replay.json',output/'data/exit_replay/support.json'])
        first=np.asarray(case.tasks[0].domain.data['frame']['T_world_mesh'])
        for k,row in enumerate(published['per_pose']):
            collision_path=output/f'data/exit_replay/collision_{row["pose"]}.json'
            evidence=I.check_report(collision_path);inputs.append(collision_path)
            frame=np.asarray(case.tasks[k].domain.data['frame']['T_world_mesh'])@np.linalg.inv(first)
            transformed=support.vertices@frame[:3,:3].T+frame[:3,3]
            expected=(support.vertices-case.offsets[k])@case.bases[k].T
            np.testing.assert_allclose(transformed,expected,atol=1e-12,rtol=0)
            assert transformed[:,2].min()>-1e-9
            # Independent motion replay: raw task mesh and raw task-to-task
            # support transform, without using the presentation's common exit.
            body=S.solid(trimesh.Trimesh(transformed,support.faces,process=False))
            moved=case.tasks[k].domain.mesh.copy()
            moved.vertices-=case.directions[k]*row['witness_mm']/1000
            actual=max(0.,float((body^S.solid(moved)).volume())*S.SCALE**3)
            expected_volume=evidence['overlap_volumes_m3'][row['witness_index']]
            np.testing.assert_allclose(actual,expected_volume,atol=1e-12,rtol=1e-6)
            np.testing.assert_allclose(frame[:3,:3]@case.exits[k],-case.directions[k],atol=1e-12,rtol=0)
            path=np.asarray(row['path_world_m'])
            distances=np.asarray(row['distances_mm'])/1000
            np.testing.assert_allclose(path-path[0],-distances[:,None]*case.directions[k],atol=1e-12,rtol=0)
            with Image.open(output/row['png']) as picture:
                assert list(picture.size)==row['video_size']
                picture.verify()
            reader=imageio.get_reader(output/row['video'])
            metadata=reader.get_meta_data()
            count=reader.count_frames()
            assert count==row['frame_count'] and metadata['fps']==row['fps']
            assert list(metadata['size'])==row['video_size'] and metadata['codec']=='h264'
            reader.close()
            decoded=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(output/row['video']),'-f','null','-'],capture_output=True,text=True)
            assert decoded.returncode==0 and not decoded.stderr,decoded.stderr
            results.append(dict(group=group.name,pose=row['pose'],frame_count=count,
                raw_transform_witness_volume_m3=actual,first_sampled_collision_mm=row['first_sampled_collision_mm'],
                original_result_preserved=True,video_decode_passed=True))
        for filename in ['exit_paths.html','exit_support.html']:
            path=output/filename;text=path.read_text()
            for href in re.findall(r'(?:href|src)="([^"]+)"',text):
                if href.startswith(('data:','http:','https:','#')):continue
                assert (path.parent/href.split('#')[0]).is_file(),(path,href)
            if filename.endswith('support.html'):
                scripts=re.findall(r'<script>([\s\S]*?)</script>',text)
                with tempfile.NamedTemporaryFile(suffix='.js',mode='w') as js:
                    js.write(scripts[-1]);js.flush()
                    subprocess.run(['node','--check',js.name],check=True,capture_output=True)
        print('EXIT REVIEW',group.name,'raw geometry, all videos and links pass',flush=True)
    assert len(groups)==8 and len(results)==28
    gallery=base/'exit_paths.html'
    for href in re.findall(r'(?:href|src)="([^"]+)"',gallery.read_text()):
        assert (gallery.parent/href.split('#')[0]).is_file(),href
    I.save(base/'exit_review.json',dict(complete=True,passed=True,group_count=len(groups),pose_view_count=len(results),
        connected_support_count=len(groups),all_video_frames_decoded=True,results=results,
        original_physical_reports_unchanged=True,browser_interaction_tested=False,
        scope='Diagnostic geometry and media verification, not support acceptance or a robot execution certificate',
        provenance=dict(inputs=I.hashes(inputs),code=I.hashes([Path(__file__)])),
        artifacts={'exit_paths.html':I.sha256(gallery)}))
    print('EXIT REVIEW COMPLETE',len(groups),len(results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B')
    review(parser.parse_args().object)
