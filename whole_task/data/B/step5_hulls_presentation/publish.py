"""Publish the approved reference layout without rerunning support search."""
import json
import shutil
from pathlib import Path

import numpy as np

from whole_task.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from whole_task.step5.membrane_feet import validate


def main():
    data=Path(__file__).resolve().parent
    folder=data.parent/'step5_height_seed'
    before=json.loads((data/'before.json').read_text())
    for path,digest in before['step5_numeric_files'].items():assert sha256(path)==digest
    for path,digest in before['source'].items():assert sha256(path)==digest
    immutable=json.loads((data.parent/'step5_height_normalized/step4_immutable.json').read_text())
    assert all(sha256(path)==digest for path,digest in immutable.items())
    step4=json.loads((data.parent/'step4_compact_best/report.json').read_text())
    assert all(sha256(path)==digest for path,digest in step4['source_sha256'].items())
    report=json.loads((folder/'report.json').read_text())
    previous_report_sha=sha256(folder/'report.json')
    revision=[]
    for path,digest in report['input_sha256'].items():
        actual=sha256(path)
        if actual!=digest:
            p=Path(path)
            assert p.parent==ROOT/'whole_task/step5' and p.name in ('membrane_movie.py','membrane_feet.py')
            archive=folder/'source_snapshot_before_hulls_presentation'/p.name
            assert sha256(archive)==digest
            revision.append(dict(file=path,before_sha256=digest,after_sha256=actual,
                                 previous_source=str(archive)))
            report['input_sha256'][path]=actual
    for name in ('hulls_presentation.py','draw_required_hulls.py'):
        path=ROOT/'whole_task/step5'/name
        report['input_sha256'][str(path)]=sha256(path)
    info=json.loads((data/'hulls_presentation.json').read_text())
    assert info['source_sha256']==sha256(ROOT/'whole_task/step5/hulls_presentation.py')
    assert info['frames']==384 and info['duration_s']==16.
    task_count=0;sample_count=0;endpoint_count=0
    for g in info['groups']:
        assert len(g['right_task_ids'])==4 and len(set(g['right_task_ids']))==4
        assert set(g['right_task_ids'])<=set(g['all_task_ids'])
        assert sha256(g['support_file'])==g['support_sha256']
        task_count+=len(g['all_task_ids'])
        for face in g['ground_faces']:
            sample_count+=face['drawn_original_sample_count']
            endpoint_count+=sum(t['endpoint_count'] for t in face['task_demands'])
    assert (task_count,sample_count,endpoint_count)==(13,425984,35437)
    with np.load(data/'video_motion.npz') as z:
        motion=z['set_angle_camera_direction']
        assert motion.shape==(384,5)
        for number in (1,2):
            rows=motion[motion[:,0]==number]
            assert len(rows)==192
            np.testing.assert_allclose(rows[[0,-1],1],[0,2*np.pi],atol=1e-14,rtol=0.)
            np.testing.assert_allclose(rows[0,2:],rows[-1,2:],atol=1e-14,rtol=0.)
        np.testing.assert_array_equal(z['fixed_head_world_m'],report['height_reference']['head_world_m'])
    for name,key in [('result.mp4','sha256'),('video_motion.npz','motion_sha256'),('hulls_and_support.png','poster_sha256')]:
        assert sha256(data/name)==info[key]
        shutil.copy2(data/name,folder/name)
    shutil.copy2(data/'encoded_review.jpg',folder/'video_review.jpg')
    info.update(file=str((folder/'result.mp4').resolve()),motion_file=str((folder/'video_motion.npz').resolve()),
                poster_file=str((folder/'hulls_and_support.png').resolve()),
                right_selected_task_count=sum(len(g['right_task_ids']) for g in info['groups']),
                original_hull_task_count=task_count)
    write_json(folder/'hulls_presentation.json',info)
    report['video']=info
    report['hulls_figure']=dict(file=info['poster_file'],sha256=info['poster_sha256'],
                              right_tasks_per_set=4,all_original_pressure_samples_drawn=True)
    reference=ROOT/'slides/Co-optimize/seperate_base_support_proof/videos/01_hulls_and_support.mp4'
    report['rendering_revision']=dict(reference=str(reference),reference_sha256=sha256(reference),
        previous_report_sha256=previous_report_sha,changed_sources=revision,
        numeric_meshes_and_task_poses_changed=False,previous_unfolded_hull_image_removed=True,
        all_hulls_and_support_orbit_together=True,no_axes=True,no_text=True,
        four_fixed_task_views_per_set=True)
    write_json(folder/'report.json',report)
    _,proof=validate(folder)
    assert proof['original_sample_count']==425984 and proof['continuous_outer_target_count']==35437
    report['validation_sha256']=sha256(folder/'validation.json')
    write_json(folder/'report.json',report)
    out=DEFAULT_OUTPUT/'B/step5'
    for name in ('overview.png','result.mp4','hulls_and_support.png'):shutil.copy2(folder/name,out/name)
    for path in (out/'required_hulls.png',folder/'required_hulls.png',folder/'required_hulls.json'):
        path.unlink(missing_ok=True)
    assert {p.name for p in out.iterdir()}=={'overview.png','result.mp4','hulls_and_support.png'}
    final=dict(status='passed',geometry_and_task_poses_unchanged=True,
        original_sample_count=425984,continuous_outer_target_count=35437,
        all_original_pressure_samples_drawn=sample_count,
        ground_face_counts=[len(g['ground_faces']) for g in info['groups']],
        right_task_ids=[g['right_task_ids'] for g in info['groups']],
        full_orbit_degrees_per_set=360,video_frames=384,video_duration_s=16.,
        complete_video_decode_passed=True,encoded_review_viewed=True,
        obsolete_required_hulls_image_removed=True,
        step4_frozen_files_unchanged=len(immutable),step4_sources_unchanged=len(step4['source_sha256']),
        visible_output_sha256={p.name:sha256(p) for p in out.iterdir()},
        report_sha256=sha256(folder/'report.json'))
    write_json(folder/'final_artifact_check.json',final)
    comparison=json.loads((folder/'comparison.json').read_text())
    comparison['source_report_sha256'][str(folder/'report.json')]=sha256(folder/'report.json')
    write_json(folder/'comparison.json',comparison)
    write_json(ROOT/'whole_task/data/B/step5_active.json',dict(workdir=str(folder),
        report_sha256=sha256(folder/'report.json'),group_count=report['final_group_count'],volume_cm3=report['final_volume_cm3']))
    print(json.dumps(final,indent=2),flush=True)


if __name__=='__main__':main()
