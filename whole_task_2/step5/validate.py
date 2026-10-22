"""Independent replay of final exported Step5 geometry and every load target."""
import argparse
import json
from pathlib import Path
import numpy as np
from whole_task_2.step1.common import ROOT,write_json,sha256
from whole_task_2.step3.legacy import kernel as K
from .source_model import restore
from .geometry import Validator,solid
from whole_task_2.evaluation.metrics import static_bounds


def validate(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text())
    if report.get('method') in ('ground_membrane_sparse_feet','direct_contact_membrane_feet'):
        from .membrane_feet import validate as validate_membranes
        result=validate_membranes(folder)[1]
        report['validation_sha256']=sha256(folder/'validation.json');write_json(folder/'report.json',report)
        active=ROOT/'whole_task_2/data'/report['object']/'step5_active.json'
        if active.exists():
            current=json.loads(active.read_text())
            if Path(current['workdir']).resolve()==folder.resolve():
                current['report_sha256']=sha256(folder/'report.json');write_json(active,current)
        return result
    for path,digest in report['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Changed Step5 input: '+path)
    for path,digest in report['artifacts'].items():
        if sha256(folder/path)!=digest:raise ValueError('Changed output: '+path)
    model,layout=restore(report,folder/'layout.npz')
    with np.load(folder/'layout.npz') as z:arrays={k:z[k].copy() for k in z.files}
    model.check_station(layout)
    mesh=K.trimesh.load(folder/'support.obj',force='mesh',process=False)
    stock_path=folder/'stock.obj' if (folder/'stock.obj').exists() else folder/'construction/stock.obj'
    stock=K.trimesh.load(stock_path,force='mesh',process=False)
    witness=K.trimesh.load(Path(report['step4'])/'step4.2/legal_component.obj',force='mesh',process=False)
    # Fresh exclusions and fresh contact extraction, without construction caches.
    checker=Validator(model,layout,witness,mesh)
    _,proof,checks=checker.check(mesh,solid(stock),np.asarray(report['final_checks']['bounds_world_m']))
    with np.load(folder/'force_certificate.npz') as z:
        for k in layout.active:
            np.testing.assert_array_equal(z[f'sample_mask_{k}'],proof['masks'][k])
            np.testing.assert_array_equal(z[f'endpoint_mask_{k}'],proof['endpoint_masks'][k])
    metrics=json.loads((folder/'evaluation.json').read_text())
    picture=Path(report.get('public_output',folder))/'evaluation.png'
    if sha256(picture)!=metrics['image_sha256']:raise ValueError('Changed evaluation image')
    for path,digest in report.get('public_artifacts',{}).items():
        if sha256(path)!=digest:raise ValueError('Changed published result: '+path)
    actual=static_bounds(model.mesh.vertices,arrays,full_vertices=mesh.vertices,pose_names=model.poses,
        execution=[model.execution(layout,k) for k in range(len(model.poses))])
    for key in ['object_static','with_full_support','support_pose_count','object_pose_counts_per_support_pose']:
        if actual[key]!=metrics[key]:raise ValueError('Evaluation replay differs: '+key)
    video_validation=None
    if report.get('video'):
        video=report['video']
        if sha256(video['video'])!=video['sha256']:raise ValueError('Changed encoded video')
        if video.get('intermediates_retained',True):
            if sha256(video['poster'])!=video['poster_sha256']:raise ValueError('Changed poster')
            for source,digest in video['inputs'].items():
                if sha256(source)!=digest:raise ValueError('Changed video input: '+source)
            from .video.verify import verify
            video_validation=verify(folder/'video')
        else:
            video_validation=dict(status='passed',encoded_video_sha256_verified=True,
                native_motion_replayed=False,
                generation_motion_validation=video['motion_validation'])
    result=dict(video_validation=video_validation,status='passed',checks=checks,evaluation_recomputed=True,report_sha256=sha256(folder/'report.json'))
    write_json(folder/'validation.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    print(json.dumps(validate(p.parse_args().folder),indent=2))
