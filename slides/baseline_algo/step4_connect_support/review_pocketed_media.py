"""Check image-only Step4 outputs, including sparse exterior reference feet."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.run_reseated import groups
from step4_connect_support import convex_foot as F, build_coupled_saddle as S


def review(name):
    base=I.OUTPUTS/name;rows=[];paths=[];structure=[]
    for group in groups(name):
        output=group/'step4'
        physical=I.check_report(output/'data/report.json')
        replay=I.check_report(output/'data/independent_review.json')
        visual=I.check_report(output/'data/visualization.json')
        assert visual['poses']==physical['poses']
        assert visual['html_generated'] is False
        assert not any(p.suffix.lower() in ('.html','.htm') for p in output.rglob('*'))
        assert visual['text_in_images'] is False and visual['arrows_in_images'] is False
        sparse=physical['construction_model']=='sparse_exterior_codesign_feet'
        assert visual['object_opacity']==(.88 if sparse else .22)
        assert visual['support_color']==('#cbd2ce' if sparse else '#bfc2c0')
        colors=list(visual['head_colors'].values())
        assert len(colors)==len(set(colors))==physical['physical_head_count']
        expected={'data','overview.png','support.png','shape.obj'}
        expected.update(f'{pose}.png' for pose in physical['poses'])
        assert {p.name for p in output.iterdir() if p.name!='.DS_Store'}==expected
        if physical['constructed']:
            work=output/physical['body_directory']
            design=json.loads((work/'design.json').read_text())
            if sparse:
                assert replay['sparse_exterior_feet_verified'] and replay['reference_pictures_verified']
                assert design['global_ground_ring'] is False
                structure.append(dict(group=group.name,separate_feet=sum(r['separate_pad_count'] for r in replay['ground_checks']),
                    pockets=design['optional_final_hollowing']['admissible_pocket_count'],removed_fraction=0))
            else:
                assert replay['codesign_construction_before_hollowing']
                assert replay['global_ground_ring'] is False and replay['hollowing_is_only_subtraction']
                structure.append(dict(group=group.name,separate_feet=replay['separate_head_floor_feet'],
                    pockets=replay['pocket_count'],removed_fraction=replay['removed_fraction']))
            paths.append(work/'design.json')
        assert visual['videos_generated'] is False and not list(output.rglob('*.mp4'))
        assert replay['fixture_acceptance_passed']==physical['passed']
        for k,row in enumerate(visual['per_pose']):
            b=np.asarray(physical['placement']['bases'][k]);o=np.asarray(physical['placement']['offsets'][k])
            np.testing.assert_allclose(row['fixture_rotation_world'],b,atol=1e-14)
            np.testing.assert_allclose(row['fixture_translation_world'],-b@o,atol=1e-14)
            if 'object_exit_direction_world' in row:
                np.testing.assert_allclose(row['object_exit_direction_world'],-np.asarray(physical['placement']['directions'][k]),atol=1e-14)
            with Image.open(output/row['image']) as picture:
                assert list(picture.size)==visual['image_size'];picture.verify()
            camera=row['camera']
            if not sparse:
                best=camera['selected'];previous=camera['previous_fixed_view']
                assert best['score']>=previous['score']-1e-12
                assert best['direction_world'][2]>0
                np.testing.assert_allclose(np.asarray(camera['basis_world'])[2],best['direction_world'],atol=1e-14)
            else:
                basis=np.asarray(camera['basis_world'])
                np.testing.assert_allclose(basis@basis.T,np.eye(3),atol=1e-14)
                assert basis[2,2]>0 and camera['span_m']>0
            rows.append(dict(group=group.name,pose=row['pose'],image=row['image'],checked=True))
        assert visual['overview_panels']==[r['image'] for r in visual['per_pose']]+['support.png']
        with Image.open(output/'support.png') as picture:
            assert list(picture.size)==visual['image_size'];picture.verify()
        with Image.open(output/'overview.png') as picture:
            assert list(picture.size)==visual['combined_size'];picture.verify()
        paths.extend([output/'data/report.json',output/'data/visualization.json',output/'data/independent_review.json'])
    assert len(groups(name))==8 and len(rows)==28
    for obsolete in ('compact_layout.html','compact_review.json','exit_paths.html','exit_review.json','reseated_batch.json'):
        assert not (base/obsolete).exists()
    assert not list(base.glob('*.html')) and not list(base.glob('*.htm'))
    assert not list(base.glob('*.json'))
    result=dict(complete=True,passed=True,group_count=len(groups(name)),pose_image_count=len(rows),
        support_image_count=len(groups(name)),overview_image_count=len(groups(name)),
        results=rows,head_floor_structure=structure,no_new_videos=True,html_generated=False,
        provenance=dict(inputs=I.hashes(paths),code=I.hashes([Path(__file__),Path(F.__file__)])))
    I.save(groups(name)[0]/'step4/data/batch_visual_review.json',result)
    print('STEP4 PRESENTATION REVIEW PASSED',len(rows),'pose pictures and',len(groups(name)),'support pictures')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('object',nargs='?',default='B')
    review(p.parse_args().object)
