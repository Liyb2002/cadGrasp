"""Check every pose image, placement transform and local static-view link."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.run_reseated import groups
from step4_connect_support import convex_foot as F, build_coupled_saddle as S


def review(name):
    base=I.OUTPUTS/name;rows=[];paths=[];htmls=[base/'step4_results.html'];structure=[]
    for group in groups(name):
        output=group/'step4/convex_landing'
        physical=I.check_report(output/'data/report.json')
        replay=I.check_report(output/'data/independent_review.json')
        visual=I.check_report(output/'data/visualization.json')
        assert visual['poses']==physical['poses']
        assert visual['text_in_images'] is False and visual['arrows_in_images'] is False
        assert visual['object_opacity']==.22 and visual['support_color']=='#bfc2c0'
        colors=list(visual['head_colors'].values())
        assert len(colors)==len(set(colors))==physical['physical_head_count']
        assert {p.name for p in (group/'step4').iterdir()}=={'convex_landing','placements.png'}
        if physical['constructed']:
            assert replay['open_ground_frames'] and replay['one_slender_stem_per_head']
            work=output/physical['body_directory']
            design=json.loads((work/'design.json').read_text())
            assert len(design['head_stems'])==physical['physical_head_count']
            assert all(s['stem_count']==1 for s in design['head_stems'])
            structure.append(dict(group=group.name,head_stems=len(design['head_stems']),
                ground_frames=len(design['ground_frames']),ground_frame_checks=replay['ground_frame_checks']))
            paths.append(work/'design.json')
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
        if physical['constructed']:
            htmls.append(output/'shape.html')
            viewer=(output/'shape.html').read_text()
            assert 'let ghost=true' in viewer
            assert 'object.material.opacity=.22' in viewer
            assert 'body=makeMesh(data,0xbfc2c0)' in viewer
    assert len(groups(name))==8 and len(rows)==28
    for name in ('compact_layout.html','compact_review.json','exit_paths.html','exit_review.json','reseated_batch.json'):
        assert not (base/name).exists()
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
        results=rows,head_floor_structure=structure,no_new_videos=True,browser_interaction_tested=False,
        provenance=dict(inputs=I.hashes(paths+htmls),code=I.hashes([Path(__file__),Path(F.__file__)])))
    I.save(base/'convex_landing_review.json',result)
    print('OPEN FRAME PRESENTATION REVIEW PASSED',len(rows),'pose pictures')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('object',nargs='?',default='B')
    review(p.parse_args().object)
