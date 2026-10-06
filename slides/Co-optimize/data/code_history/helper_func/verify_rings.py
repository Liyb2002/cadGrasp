"""Check saved minimum-hull evidence and coverage flags; no model replay."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
from scipy.spatial import ConvexHull

groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']+[dict(g,id='illegal/'+g['id']) for g in json.loads((ROOT/'objects/B/illegal_pose_sets.json').read_text())['sets']]
passed=0;failed_rows=[];count=0
for group in groups:
    out=HERE/'output/B'/group['id']/'step3/step3.3'
    report=I.check_report(out/'data/report.json');render=I.check_report(out/'data/render.json')
    assert render['boundary_kind']=='minimum_convex_hull' and not render['text_or_labels']
    for row in report['state_results']:
        p=np.load(ROOT/'objects/B/poses'/row['pose']/'floor_contact.npz')['floor_demands_xy_m']
        saved=np.load(out/'data'/f"{row['pose']}.npz")
        np.testing.assert_array_equal(saved['saved_demands_world_xy_m'],p)
        hull=ConvexHull(p);polygon=p[hull.vertices]
        np.testing.assert_array_equal(saved['demand_hull_world_xy_m'],polygon)
        np.testing.assert_array_equal(row['demand_hull_world_xy_m'],polygon)
        assert row['boundary_sample_indices']==hull.vertices.tolist()
        assert row['boundary_kind']=='minimum_convex_hull' and row['outer_expansion_m']==0
        assert abs(row['minimum_hull_area_cm2']-hull.volume*1e4)<1e-10
        actual=np.asarray(row['actual_ground_hull_world_xy_m'])
        covered=False
        if len(actual)>=3:
            equations=ConvexHull(actual).equations
            margins=(p@equations[:,:2].T+equations[:,2]).max(1)
            covered=bool(np.all(margins<=1e-9))
            if not covered:failed_rows.append(dict(group=group['id'],pose=row['pose'],uncovered_points=int((margins>1e-9).sum()),maximum_miss_mm=float(margins.max()*1000)))
        assert covered==row['all_saved_demands_covered']
        count+=1
    assert report['passed']==(all(r['all_saved_demands_covered'] for r in report['state_results']) and all(r['working_surface_clear'] for r in report['state_geometry']))
    passed+=report['passed']
save(HERE/'output/B/data/step33_convex_hull_review.json',dict(complete=True,sets=len(groups),pose_perimeters=count,passed_sets=passed,unresolved_sets=len(groups)-passed,minimum_convex_hulls_verified=True,original_points_unchanged=True,outer_expansion_m=0,failed_ground_coverage=failed_rows,export_geometry_replayed=False,step4_rerun=False))
print('MINIMUM HULL EVIDENCE:',len(groups),'sets;',count,'pose perimeters;',passed,'PASS;',len(groups)-passed,'UNRESOLVED')
print('Uncovered:',failed_rows)
