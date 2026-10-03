"""Decode every compact exit frame and check offline presentation links."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I


def review(name):
    base=I.OUTPUTS/name;results=[];paths=[]
    htmls=[base/'compact_layout.html']
    for group in ['pose3+6','pose5+7']:
        output=base/group/'step4/compact'
        physical=I.check_report(output/'data/report.json')
        replay=I.check_report(output/'data/compact_replay.json')
        visual=I.check_report(output/'data/visualization.json')
        assert replay['fixture_acceptance_passed']==physical['passed']
        assert visual['poses']==physical['poses']
        for k,row in enumerate(visual['per_pose']):
            np.testing.assert_allclose(row['object_direction_world'],-np.asarray(physical['placement']['directions'][k]),atol=1e-14)
            assert row['support_stationary']
            assert row['withdrawal_passed']==physical['construction']['checks'][k]['withdrawal']['clear']
            with Image.open(output/row['image']) as picture:
                assert picture.size==(800,910);picture.verify()
            reader=imageio.get_reader(output/row['video'])
            metadata=reader.get_meta_data();count=reader.count_frames();reader.close()
            assert count==row['frames'] and metadata['fps']==row['fps']
            assert metadata['size']==(1400,830) and metadata['codec']=='h264'
            decoded=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(output/row['video']),'-f','null','-'],capture_output=True,text=True)
            assert decoded.returncode==0 and not decoded.stderr,decoded.stderr
            results.append(dict(group=group,pose=row['pose'],frames=count,decoded=True))
        paths.extend([output/'data/report.json',output/'data/compact_replay.json',output/'data/visualization.json'])
        htmls.extend([output/'index.html',output/'shape.html'])
    for path in htmls:
        content=path.read_text()
        for href in re.findall(r'(?:href|src)="([^"]+)"',content):
            if href.startswith(('data:','http:','https:','#')):continue
            assert (path.parent/href.split('#')[0]).is_file(),(path,href)
        scripts=re.findall(r'<script>([\s\S]*?)</script>',content)
        for script in scripts:
            with tempfile.NamedTemporaryFile(suffix='.js',mode='w') as js:
                js.write(script);js.flush()
                subprocess.run(['node','--check',js.name],check=True,capture_output=True)
    I.save(base/'compact_review.json',dict(complete=True,passed=True,results=results,
        all_video_frames_decoded=True,browser_interaction_tested=False,
        provenance=dict(inputs=I.hashes(paths+htmls),code=I.hashes([Path(__file__)]))))
    print('COMPACT MEDIA REVIEW PASSED',len(results),'pose videos')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('object',nargs='?',default='B')
    review(p.parse_args().object)
