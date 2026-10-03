"""Check sparse actual ground contacts, original physics and reference pictures."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current import reseating as R,build_coupled_saddle as S
from step4_connect_support.baseline_current.review_reseated import review as physical_review


def review(group_name):
    out=I.OUTPUTS/'B'/group_name/'step4'
    physical=physical_review(out)
    report=I.check_report(out/'data/report.json');visual=I.check_report(out/'data/visualization.json')
    case=R.load_case(out);work=out/report['body_directory']
    design=json.loads((work/'design.json').read_text())
    body=trimesh.load(out/'shape.obj',force='mesh',process=False)
    assert report['construction_model']=='sparse_exterior_codesign_feet'
    assert design['global_ground_ring'] is False
    assert all(r['count'] in (3,4) for r in design['floor_patterns'])
    pairs=[(e['head_body'],e['floor_index']) for e in design['consolidated_landings']]
    assert len(pairs)==len(set(pairs))
    assert {e['head_body'] for e in design['consolidated_landings']}==set(range(report['physical_head_count']))
    checks=[]
    for k,(b,o) in enumerate(zip(report['placement']['bases'],report['placement']['offsets'])):
        world=(body.vertices-np.asarray(o))@np.asarray(b).T
        ground=world[np.abs(world[:,2])<1e-9,:2]
        hull=ConvexHull(ground);h=hull.equations
        debt=np.max(case.demands[k]@h[:,:2].T+h[:,2])
        assert debt<=1e-9,(case.poses[k],debt)
        for j,pad in enumerate(design['feet_xy_m'][k]):
            assert any(e['floor_index']==k and e['pad_index']==j for e in design['consolidated_landings'])
        checks.append(dict(pose=case.poses[k],original_floor_demand_count=len(case.demands[k]),
            maximum_demand_halfspace_residual_m=float(debt),separate_pad_count=len(design['feet_xy_m'][k])))
    expected={'data','overview.png','support.png','shape.obj'}|{f'{p}.png' for p in report['poses']}
    assert {p.name for p in out.iterdir() if p.name!='.DS_Store'}==expected
    assert visual['text_in_images'] is False and visual['arrows_in_images'] is False
    assert visual['html_generated'] is False and visual['videos_generated'] is False
    assert visual['object_opacity']==.88 and visual['object_smooth_normals']
    assert visual['support_color']=='#cbd2ce'
    assert len(set(visual['head_colors'].values()))==report['physical_head_count']
    assert visual['overview_panels']==[f'{p}.png' for p in report['poses']]+['support.png']
    for filename in visual['overview_panels']:
        with Image.open(out/filename) as im:assert list(im.size)==visual['image_size'];im.verify()
    with Image.open(out/'overview.png') as im:assert list(im.size)==visual['combined_size'];im.verify()
    assert not any(p.suffix.lower() in ('.html','.htm','.mp4') for p in out.rglob('*'))
    assert not list((I.OUTPUTS/'B').glob('*.json'))
    physical.update(sparse_exterior_feet_verified=True,ground_checks=checks,reference_pictures_verified=True,
        image_count=len(visual['overview_panels'])+1,no_html=True,no_object_level_json=True)
    physical['provenance']['inputs'].update(I.hashes([work/'design.json',out/'data/visualization.json']))
    physical['provenance']['code'].update(I.hashes([Path(__file__)]))
    I.save(out/'data/independent_review.json',physical)
    print('REFERENCE DESIGN REVIEW',group_name,'fixture accepted',physical['fixture_acceptance_passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('group');review(p.parse_args().group)
