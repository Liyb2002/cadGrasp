"""Check complete consolidated soles in exported bodies, plus original physics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Polygon
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current import convex_foot as F, build_coupled_saddle as S
from step4_connect_support.baseline_current.review_reseated import review as review_physics


def review(output):
    output=Path(output).resolve()
    result=review_physics(output)
    report=I.check_report(output/'data/report.json')
    if not report['constructed']:return result
    work=output/report['body_directory'];design=json.loads((work/'design.json').read_text())
    body=S.solid(trimesh.load(output/'shape.obj',force='mesh',process=False))
    bases=np.asarray(report['placement']['bases']);offsets=np.asarray(report['placement']['offsets'])
    soles=design['consolidated_landings'];keys=[(s['head_body'],s['floor_index']) for s in soles]
    assert len(keys)==len(set(keys))
    assert design['max_connections_per_head_floor']==1
    grounds={k:F.landing(body,b,o) for k,(b,o) in enumerate(zip(bases,offsets))}
    checks=[]
    for sole in soles:
        polygon=Polygon(sole['polygon_xy_m']);k=sole['floor_index']
        assert sole['connection_count']==1
        assert polygon.convex_hull.symmetric_difference(polygon).area<1e-12
        missing=polygon.difference(grounds[k].buffer(1e-10)).area
        assert missing<1e-12,(sole['candidate_id'],sole['floor_pose'],missing)
        assigned=[a for a in design['attachments'] if a['head_body']==sole['head_body'] and a['floor_pose']==sole['floor_pose']]
        for item in assigned:assert Polygon(item['polygon_xy_m']).difference(polygon.buffer(1e-10)).area<1e-12
        checks.append(dict(candidate_id=sole['candidate_id'],floor_pose=sole['floor_pose'],
            assigned_terminals=len(assigned),convex=True,missing_exported_floor_area_m2=missing))
    result.update(single_consolidated_loft_per_head_floor=True,consolidated_landing_checks=checks)
    result['provenance']['inputs'].update(I.hashes([work/'design.json']))
    result['provenance']['code'].update(I.hashes([Path(__file__),Path(F.__file__)]))
    I.save(output/'data/independent_review.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('outputs',type=Path,nargs='+')
    for output in p.parse_args().outputs:
        result=review(output);print(output.parent.parent.name,'convex-landing review passed',result['fixture_acceptance_passed'],flush=True)
