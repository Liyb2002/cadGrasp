"""Publish the already checked jump; never repeat its mesh validation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import json
import shutil
from co_common import HERE, ROOT, I, save, trimesh, np
from whole_search.fast_search import FastModel
from whole_search.model import Model
from whole_search.classify import classify
from whole_search.search import failed_count
from whole_step4_render import render_search


def main():
    base=HERE/'output/B/pose1+3+7+10+14+18+23+27'
    out=base/'step4/step4.2/reuse_jump_probe_v1'
    original=json.loads((out/'probe.json').read_text());assert original['complete'] and original['passed']
    poses=json.loads((base/'step4/step4.1/data/report.json').read_text())['poses']
    model=FastModel(poses,'B',initialization_report=base/'step3/step3.1/data/report.json')
    from whole_search.model import Layout
    prefix=out/'data/exact_states/feasible_1'
    arrays=np.load(prefix.with_suffix('.npz'));metadata=json.loads(prefix.with_suffix('.json').read_text())
    layout=Layout(arrays['placements'],arrays['directions'],arrays['hosts'],tuple(map(int,arrays['active'])))
    contacts={};masks={};supplies={};classifiers={}
    for k in layout.active:
        with np.load(out/f'{poses[k]}_force.npz') as stored:
            tri=stored['triangles_fixture_m'];sources=stored['source_faces']
            np.testing.assert_array_equal(stored['mask'],arrays[f'mask_{k}'])
            np.testing.assert_array_equal(stored['supply_7d'],arrays[f'supply_{k}'])
            contacts[k]=dict(triangles=tri,sources=sources,triangle_count=len(tri),
                area_m2=float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2))
        supplies[k]=arrays[f'supply_{k}'];masks[k],classifiers[k]=classify(supplies[k],model.tasks[k].targets)
        np.testing.assert_array_equal(masks[k],arrays[f'mask_{k}'])
    latest=json.loads((out/'data/exact_states/latest.json').read_text())
    mesh=trimesh.Trimesh(arrays['vertices'],arrays['faces'],process=False)
    assert abs(abs(mesh.volume)*1e6-metadata['volume_cm3'])<1e-4
    actual=dict(layout=layout,serial=1,remaining=None,remaining_mesh=mesh,contacts=contacts,
        masks=masks,supplies=supplies,classifiers=classifiers,seconds=original['seconds'],
        maximum_projected_footprint_m2=latest['maximum_projected_footprint_m2'],**metadata)
    assert failed_count(actual)==0 and all(row['passed'] for row in actual['actual_work_surface_checks'])
    save(out/'publication_failure.json',original)
    original.pop('error',None);original.update(publication_complete=True,publication_recomputed_geometry=False)
    save(out/'probe.json',original)
    manifest=json.loads((out/'source_manifest.json').read_text())
    snapshot=out/'executed_sources'/Path(__file__).relative_to(ROOT)
    snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,snapshot)
    report=Model.save(model,actual,out,dict(strategy='restored_juxtapose_probe',passed=True,
        old_answers_used_as_start=False,publication_recomputed_geometry=False,
        numerical_search_scope='one cold-start operation, not a whole-ten-set benchmark',
        options=dict(screen_budget=96,full_budget=6)))
    report['provenance']['code'].update({manifest['snapshots'][p]:h for p,h in manifest['code'].items()})
    report['provenance']['code'].update(I.hashes([snapshot]))
    report['provenance']['inputs'].update(I.hashes([out/'data/exact_states/feasible_1.json',
        out/'data/exact_states/feasible_1.npz']))
    save(out/'report.json',report)
    data=report.copy();data['artifacts']={'../'+k:v for k,v in report['artifacts'].items()}
    save(out/'data/report.json',data);render_search(model,out,report)
    for name in ['process.png','final_result.png']:report['artifacts'][name]=I.sha256(out/name)
    save(out/'report.json',report)
    data=report.copy();data['artifacts']={'../'+k:v for k,v in report['artifacts'].items()}
    save(out/'data/report.json',data);I.check_report(out/'data/report.json')
    print('PUBLISHED',report['volume_cm3'],report['counts'],flush=True)


if __name__=='__main__':main()
