"""Replay real material, convex support coverage and unchanged load certificates."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Polygon
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support import convex_foot as F,build_coupled_saddle as S
from step4_connect_support.review_reseated import review as review_physics


def review(output):
    output=Path(output).resolve();result=review_physics(output)
    report=I.check_report(output/'data/report.json')
    if not report['constructed']:return result
    inputs=I.check_report(output/'data/design_inputs.json')
    work=output/report['body_directory'];design=json.loads((work/'design.json').read_text())
    body=S.solid(trimesh.load(output/'shape.obj',force='mesh',process=False))
    stems=design['head_stems']
    ids=[s['candidate_id'] for s in stems]
    assert len(ids)==len(set(ids))==report['physical_head_count']
    assert all(s['stem_count']==1 for s in stems)
    assert all(c.get('kind')!='carved_connection_envelope' for c in design['connections'])
    checks=[]
    for pose,b,o,xy in zip(report['poses'],report['placement']['bases'],report['placement']['offsets'],inputs['ground_hulls_xy_m']):
        actual=F.landing(body,np.asarray(b),np.asarray(o));required=Polygon(xy)
        missing=required.difference(actual.convex_hull.buffer(1e-10)).area
        assert missing<1e-12,(pose,missing)
        fraction=actual.intersection(required).area/required.area
        # A renamed filled panel cannot satisfy the open-frame design rule.
        assert fraction<.5,(pose,fraction)
        checks.append(dict(pose=pose,missing_required_hull_area_m2=missing,
            actual_ground_material_fraction_inside_required_hull=fraction,
            coverage_is_by_convex_hull_not_filled_material=True))
    result.update(open_ground_frames=True,one_slender_stem_per_head=True,
        filled_hull_required=False,ground_frame_checks=checks)
    result['provenance']['inputs'].update(I.hashes([work/'design.json',output/'data/design_inputs.json']))
    result['provenance']['code'].update(I.hashes([Path(__file__),Path(F.__file__)]))
    I.save(output/'data/independent_review.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('outputs',type=Path,nargs='+')
    for out in p.parse_args().outputs:
        r=review(out);print(out.parent.parent.name,'open-frame review passed',r['fixture_acceptance_passed'],flush=True)
