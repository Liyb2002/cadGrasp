"""Expose only the movie and evaluation; retain minimal computational state."""
from pathlib import Path
import json
import shutil
import tempfile
from subdivide_task.step1.common import ROOT,sha256,write_json

PUBLIC_FILES=('video.mp4','evaluation.png','evaluation.json')
DATA_FILES=('support.obj','layout.npz','execution.json','force_certificate.npz','evaluation.json')


def publish(work,output):
    work=Path(work).resolve();output=Path(output).resolve()
    if output.exists():
        unexpected=[p.name for p in output.iterdir() if p.name not in PUBLIC_FILES or not p.is_file()]
        if unexpected:raise ValueError('Output contains unrelated files: '+', '.join(unexpected))
    report=json.loads((work/'report.json').read_text())
    data=ROOT/'subdivide_task/data'/report['object']/'step5'
    data.parent.mkdir(parents=True,exist_ok=True)
    video=report.get('video')
    with tempfile.TemporaryDirectory(prefix='cadgrasp_step5_publish_') as temp:
        stage=Path(temp)
        for name in ('evaluation.png','evaluation.json'):shutil.copy2(work/name,stage/name)
        if video:shutil.copy2(video['video'],stage/'video.mp4')
        with tempfile.TemporaryDirectory(prefix='.step5_',dir=data.parent) as state_temp:
            state=Path(state_temp)
            for name in DATA_FILES:shutil.copy2(work/name,state/name)
            stock=work/'stock.obj' if (work/'stock.obj').exists() else work/'construction/stock.obj'
            shutil.copy2(stock,state/'stock.obj')
            motion=work/'video_motion.zip'
            if motion.exists():shutil.copy2(motion,state/motion.name)
            # The cleanup changes output drivers, never the accepted geometry.
            # Preserve their original generation hashes as provenance instead
            # of claiming that newly edited driver files generated this run.
            retired={}
            for path,digest in list(report['input_sha256'].items()):
                source=Path(path)
                if source.exists() and sha256(source)==digest:continue
                if source in [ROOT/'subdivide_task/step5/run.py',ROOT/'subdivide_task/step5/movie.py']:
                    retired[path]=report['input_sha256'].pop(path)
                else:raise ValueError('Changed numerical input during publication: '+path)
            if retired:report['original_output_driver_sha256']=retired
            report['schema']='whole_task_step5_compact_v2'
            report['public_output']=str(output)
            report['artifacts']={name:sha256(state/name) for name in (*DATA_FILES,'stock.obj')}
            if (state/'video_motion.zip').exists():report['artifacts']['video_motion.zip']=sha256(state/'video_motion.zip')
            report['public_artifacts']={str(output/name):sha256(stage/name) for name in PUBLIC_FILES if (stage/name).exists()}
            if video:
                video=dict(video)
                video['generation_input_sha256']=video.pop('inputs',video.get('generation_input_sha256',{}))
                video['video']=str(output/'video.mp4')
                for key in ('poster','poster_sha256','motion_plan','encoded_contact_sheet'):video.pop(key,None)
                video['intermediates_retained']=False
                report['video']=video
            write_json(state/'report.json',report)
            if data.exists():shutil.rmtree(data)
            shutil.copytree(state,data)
        output.mkdir(parents=True,exist_ok=True)
        for name in PUBLIC_FILES:
            source=stage/name;destination=output/name
            if source.exists():shutil.copy2(source,destination)
            elif destination.exists():destination.unlink()
    return report,data
