"""Collect completed large-set audits and snapshot sources without changing results."""
import _bootstrap
import shutil
from co_common import *
from run_all import output_root
from run_large_pose_sets import groups, EXPERIMENT
from audit_whole_step4 import protected_files


def main():
    root=output_root('B'); directory=root/'data'/EXPERIMENT
    rows=[]; sources={}; result_hashes={}
    load_replays=compact_rays=baseline_rays=blocked=0
    for group in groups():
        base=root/group['id']; path=base/'data'/EXPERIMENT/'audit.json'
        audit=json.loads(path.read_text())
        assert audit['complete'] and audit['passed']
        methods={a['method']:a['result'] for a in audit['audits']}
        assert set(methods)=={'baseline','greedy','beam'}
        baseline=methods['baseline']
        for stage,digest in baseline['stage_report_hashes'].items():
            assert I.sha256(base/'step4'/stage/'data/report.json')==digest
        assert baseline['passed']
        baseline_rays+=sum(r['rays_tested'] for r in baseline['stages'])
        blocked+=sum(r['blocked_rays'] for r in baseline['stages'])
        for mode in ['baseline','greedy','beam']:
            out=base/'step4/step4.2'
            if mode!='baseline':out=out/'compact'/mode
            report=I.check_report(out/'data/report.json')
            assert report['force_exit_work_passed'] and report['poses']==group['poses']
            assert report['original_loads_reused'] and not report['load_subsampling_for_final_acceptance']
            assert report['counts']=={pose:32768 for pose in group['poses']}
            result_hashes[str((out/'data/report.json').relative_to(ROOT))]=I.sha256(out/'data/report.json')
            for relative,digest in report['provenance']['code'].items():
                assert relative not in sources or sources[relative]==digest
                sources[relative]=digest
            if mode!='baseline':
                checked=methods[mode]
                assert checked['passed'] and checked['blocked_rays']==0
                assert checked['report_sha256']==I.sha256(out/'data/report.json')
                assert [r['pose'] for r in checked['original_load_replays']]==group['poses']
                assert all(r['loads']==32768 and r['classifier']['primal_batch_accepted']==32768
                           and r['classifier']['dual_batch_rejected']==0 for r in checked['original_load_replays'])
                load_replays+=sum(r['loads'] for r in checked['original_load_replays'])
                compact_rays+=checked['rays_tested'];blocked+=checked['blocked_rays']
                manifest=json.loads((out/'data/source_manifest.json').read_text())
                assert manifest['baseline_report_sha256']==baseline['stage_report_hashes']['step4.2']
                for relative,digest in manifest['code'].items():
                    assert relative not in sources or sources[relative]==digest
                    sources[relative]=digest
        rows.append(dict(id=group['id'],pose_count=len(group['poses']),passed=True,
                         audit_sha256=I.sha256(path),audits=methods))
    assert len(rows)==10 and load_replays==sum(r['pose_count'] for r in rows)*2*32768
    assert blocked==0
    save(directory/'audit_summary.json',dict(complete=True,passed=True,sets=len(rows),
        pose_instances=sum(r['pose_count'] for r in rows),optimized_results=len(rows)*2,
        independently_replayed_original_loads=load_replays,
        independently_tested_compact_work_rays=compact_rays,
        independently_tested_baseline_work_rays=baseline_rays,blocked_rays=blocked,
        continuous_geometry_checks_verified_by_fingerprints_not_repeated=True,
        original_loads_per_pose=32768,result_report_hashes=result_hashes,rows=rows))
    preservation=json.loads((directory/'protection.json').read_text())
    for relative,digest in preservation['files'].items():
        assert I.sha256(ROOT/relative)==digest,relative
    save(directory/'preservation_check.json',dict(passed=True,unchanged_files=len(preservation['files']),
        original_physical_source_and_inputs=protected_files(root)))
    helpers=['run_large_pose_sets','summarize_large_pose_sets','continue_large_pose_sets',
             'continue_large_blockers','probe_large_directions','publish_large_probe',
             'probe_large_warm_start','finalize_large_pose_sets','audit_compact_volume','audit_whole_step4']
    extra=[HERE/'helper_func'/f'{name}.py' for name in helpers]
    extra+=list((HERE/'helper_func/whole_search').glob('*.py'))
    sources.update(I.hashes(extra))
    aliases={}; frozen=directory/'executed_sources'
    for relative,digest in sources.items():
        source=ROOT/relative;assert I.sha256(source)==digest,relative
        target=frozen/relative
        if target.exists() and I.sha256(target)!=digest:
            # Retain earlier publication helpers without changing any
            # numerical report or overwriting its source snapshot.
            target=frozen/'_revisions'/digest/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert I.sha256(target)==digest
        else:shutil.copy2(source,target)
        aliases[relative]=str(target.relative_to(ROOT))
    save(directory/'source_snapshot.json',dict(complete=True,sources=sources,snapshots=aliases,
        referenced_reports_unchanged=True,numerical_artifacts_changed=False,
        snapshot_only_no_provenance_rewrite=True,result_report_hashes=result_hashes))
    print('LARGE AUDIT',dict(sets=len(rows),original_load_replays=load_replays,
          work_rays=compact_rays+baseline_rays,blocked_rays=blocked,
          preserved_files=len(preservation['files']),source_snapshots=len(sources)),flush=True)


if __name__=='__main__':main()
