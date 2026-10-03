"""Check every pose image, placement transform and local static-view link."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current.run_reseated import groups


def review(name):
    base=I.OUTPUTS/name;rows=[];paths=[];htmls=[base/'step4_results.html']
    for group in groups(name):
        output=group/'step4/reseated'
        physical=I.check_report(output/'data/report.json')
        replay=I.check_report(output/'data/independent_review.json')
        visual=I.check_report(output/'data/visualization.json')
        assert visual['poses']==physical['poses']
        assert visual['videos_generated'] is False and not list(output.rglob('*.mp4'))
        assert replay['fixture_acceptance_passed']==physical['passed']
        for k,row in enumerate(visual['per_pose']):
            b=np.asarray(physical['placement']['bases'][k]);o=np.asarray(physical['placement']['offsets'][k])
            np.testing.assert_allclose(row['fixture_rotation_world'],b,atol=1e-14)
            np.testing.assert_allclose(row['fixture_translation_world'],-b@o,atol=1e-14)
            np.testing.assert_allclose(row['object_exit_direction_world'],-np.asarray(physical['placement']['directions'][k]),atol=1e-14)
            with Image.open(output/row['image']) as picture:
                assert list(picture.size)==visual['image_size'];picture.verify()
            rows.append(dict(group=group.name,pose=row['pose'],image=row['image'],checked=True))
        with Image.open(output/'placements.png') as picture:
            assert list(picture.size)==visual['combined_size'];picture.verify()
        assert I.sha256(output/'placements.png')==I.sha256(group/'step4/placements.png')
        paths.extend([output/'data/report.json',output/'data/visualization.json',output/'data/independent_review.json'])
        htmls.append(output/'index.html')
        if physical['constructed']:htmls.append(output/'shape.html')
    assert len(groups(name))==8 and len(rows)==28
    for path in htmls:
        text=path.read_text()
        assert '<video' not in text
        for href in re.findall(r'(?:href|src)="([^"]+)"',text):
            if href.startswith(('data:','http:','https:','#')):continue
            assert (path.parent/href.split('#')[0]).is_file(),(path,href)
        for script in re.findall(r'<script>([\s\S]*?)</script>',text):
            with tempfile.NamedTemporaryFile(suffix='.js',mode='w') as js:
                js.write(script);js.flush()
                subprocess.run(['node','--check',js.name],check=True,capture_output=True)
    result=dict(complete=True,passed=True,group_count=len(groups(name)),pose_image_count=len(rows),
        results=rows,no_new_videos=True,browser_interaction_tested=False,
        provenance=dict(inputs=I.hashes(paths+htmls),code=I.hashes([Path(__file__)])))
    I.save(base/'reseated_review.json',result)
    print('RESEATED PRESENTATION REVIEW PASSED',len(rows),'pose pictures')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('object',nargs='?',default='B')
    review(p.parse_args().object)
