"""Publish an already checked compact solution after the metadata save fix.

The original complete search, real finalist attempts and geometry remain
unchanged. This continuation independently replays all original force loads.
"""
import _bootstrap
import argparse,contextlib,time
from co_common import *
from compact_volume import CompactSearch,Node,load_incumbent
from whole_pipeline import load_layout
from whole_search.fast_search import FastModel
from whole_search.classify import classify
from whole_step4_render import render_search
from run_all import output_root,saved_groups


def checked_partial(model,out,attempts,baseline):
    accepted=[r for r in attempts if r['accepted']]
    if not accepted:
        return baseline,None,[]
    assert len(accepted)==1 and accepted[0]['force_exit_work_passed']
    assert all(v==32768 for v in accepted[0]['counts'].values())
    layout=load_layout(out/'layout.npz');support=trimesh.load(out/'support.obj',force='mesh',process=False)
    tree=json.loads((out/'tree.json').read_text());index=accepted[0]['tree_node']
    assert layout.key()==load_layout(out/tree['nodes'][index]['layout']).key()
    metadata=None;checkpoint=None
    for path in (out/'exact_states').glob('feasible_*.npz'):
        if layout.key()!=load_layout(path).key():continue
        with np.load(path) as z:
            np.testing.assert_allclose(z['vertices'],support.vertices,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(z['faces'],support.faces)
        metadata=json.loads(path.with_suffix('.json').read_text());checkpoint=path;break
    if metadata is None:raise RuntimeError('No matching actual feasible checkpoint')
    assert all(v==32768 for v in metadata['counts'].values())
    assert abs(metadata['volume_cm3']-accepted[0]['actual_volume_cm3'])<=1e-4
    assert abs(abs(support.volume)*1e6-metadata['volume_cm3'])<=1e-4
    assert all(r['passed'] for r in metadata['actual_work_surface_checks'])
    contacts,masks,supplies,classifiers={},{},{},{}
    evidence=[checkpoint,checkpoint.with_suffix('.json'),out/'layout.npz',out/'support.obj']
    with np.load(checkpoint) as checked:
        for k in layout.active:
            path=out/f'{model.poses[k]}_force.npz';evidence.append(path)
            with np.load(path) as z:
                triangles=z['triangles_fixture_m'].copy();sources=z['source_faces'].copy()
                np.testing.assert_array_equal(checked[f'mask_{k}'],z['mask'])
                np.testing.assert_array_equal(checked[f'supply_{k}'],z['supply_7d'])
                posed=model.mesh.copy();posed.apply_transform(layout.placements[k])
                actual_triangles,actual_sources=contact_boundary(posed,support,model.allowed[k])
                np.testing.assert_array_equal(actual_sources,sources)
                np.testing.assert_allclose(actual_triangles,triangles,atol=1e-12,rtol=0)
                T=model.native[k]@np.linalg.inv(layout.placements[k])
                actual_supply=supply(model.tasks[k],T,triangles,sources)
                np.testing.assert_allclose(actual_supply,z['supply_7d'],atol=1e-12,rtol=1e-12)
                masks[k],classifiers[k]=classify(actual_supply,model.tasks[k].targets)
                assert masks[k].all() and len(masks[k])==32768
                supplies[k]=actual_supply
                contacts[k]=dict(triangles=triangles,sources=sources,area_m2=float(np.linalg.norm(
                    np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1).sum()/2))
    actual=dict(layout=layout,serial=0,remaining=None,remaining_mesh=support,contacts=contacts,
        masks=masks,supplies=supplies,classifiers=classifiers,seconds=0.,
        counts=metadata['counts'],diagnostics=metadata['diagnostics'],volume_cm3=metadata['volume_cm3'],
        maximum_projected_footprint_m2=float(np.prod((support.bounds[1]-support.bounds[0])[:2])),
        actual_work_surface_checks=metadata['actual_work_surface_checks'],
        component_volumes_cm3=metadata['component_volumes_cm3'],lossless_worker_geometry_transport=True)
    detail=tree['nodes'][index]
    result=dict(actual,volume_cm3=detail['estimated_volume_cm3'])
    return actual,Node(index,detail['parent'],result,detail['operation'],detail['detail'],detail['generation']),evidence


def publish(name,group,mode):
    source=output_root(name)/group['id']/'step4/step4.2';out=source/'compact'/mode
    if (out/'data/report.json').exists():return I.check_report(out/'data/report.json')
    failure=json.loads((out/'data/failure.json').read_text())
    if "multiple values for keyword argument 'relocation_sweep_carved'" not in failure['error']:
        raise RuntimeError('This is only a metadata save continuation')
    manifest=json.loads((out/'data/source_manifest.json').read_text());options=manifest['options']
    old_sources=[out/'data/executed_sources'/Path(p).name for p in manifest['code']]
    for path,(relative,digest) in zip(old_sources,manifest['code'].items()):assert I.sha256(path)==digest
    attempts=json.loads((out/'actual_validation.json').read_text())
    original_hashes=I.hashes([out/'tree.json',out/'trace.json',out/'process.json',out/'selected_path.json',out/'actual_validation.json'])
    began=time.monotonic()
    with (out/'data/publication.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        model=FastModel(group['poses'],name,initialization_report=source.parent.parent/'step3/step3.1/data/report.json')
        baseline,baseline_report=load_incumbent(model,source)
        actual,node,evidence=checked_partial(model,out,attempts,baseline)
        search=CompactSearch(model,out,mode=mode,rounds=options['rounds'],width=options['width'],
            repair_width=options['repair_width'],screen_budget=options['screen_budget'],
            full_budget=options['full_budget'],seed=options['seed'])
        trace=json.loads((out/'trace.json').read_text())
        search.full_checks=[r['full_checks'] for r in trace if 'full_checks' in r][-1]
        search.search_seconds=None
        search.export_path=lambda selected,result:None  # Preserve original ancestor path.
        search.actual_finalists=lambda nodes,budget:[] if node is None else [node]
        model.exact_cache[actual['layout'].key()]=actual
        evidence_hashes=I.hashes(evidence)
        report=search.validate_and_save([] if node is None else [node],baseline,baseline_report,source,options['real_budget'])
        save(out/'actual_validation.json',attempts);report['final_validation_attempts']=attempts
        render_search(model,out,report)
        for relative,digest in original_hashes.items():assert I.sha256(ROOT/relative)==digest
        for relative,digest in evidence_hashes.items():assert I.sha256(ROOT/relative)==digest
        report.update(pose_set=group['id'],stage='step4.2',metadata_only_publication_recovery=True,
            exact_geometry_reused_unchanged=True,original_search_seconds_unknown=True,
            prior_search_and_validation_attempt_seconds=failure['seconds'],
            seconds=failure['seconds']+time.monotonic()-began,
            original_executed_compact_sources=I.hashes(old_sources),publication_checkpoint_evidence=evidence_hashes)
        report['provenance']['inputs'].update(evidence_hashes)
        report['provenance']['code'].update(I.hashes(old_sources+[Path(__file__)]))
        report['artifacts']={p.name:I.sha256(p) for p in out.iterdir() if p.is_file() and p.name not in ['report.json','README.md']}
        report['artifacts']['data/render.json']=I.sha256(out/'data/render.json')
        save(out/'report.json',report);data=report.copy()
        data['artifacts']={'../'+k:v for k,v in report['artifacts'].items()}
        save(out/'data/report.json',data);I.check_report(out/'data/report.json')
        (out/'README.md').write_text('# Whole 实体体积优化\n\n'
            f"`{mode}`：{report['baseline_volume_cm3']:.3f} → {report['volume_cm3']:.3f}cm³，减少 {100*report['material_saved_fraction']:.2f}%。\n\n"
            '[过程](process.png) · [最终各pose](final_result.png) · [原始搜索树](tree.json) · [实际候选](actual_validation.json)\n\n'
            '完整搜索和实际候选未重跑；修复保存字段冲突后从相同的真实检查点发布。'
            '独立重建接触并重放全部原载荷；原搜索、真实几何、所有原候选失败与执行源码均保留。'
            '过程图固定等轴测、无文字，末格为真实支撑。接地、连通和强度仍待后续。\n')
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',default='B',nargs='?')
    parser.add_argument('--sets',nargs='+',required=True);parser.add_argument('--mode',choices=['greedy','beam'],required=True)
    args=parser.parse_args()
    for group in saved_groups(args.object):
        if group['id'] in args.sets:
            report=publish(args.object,group,args.mode)
            print('COMPACT PUBLISH',group['id'],args.mode,report['volume_cm3'],report['material_saved_fraction'],flush=True)


if __name__=='__main__':main()
