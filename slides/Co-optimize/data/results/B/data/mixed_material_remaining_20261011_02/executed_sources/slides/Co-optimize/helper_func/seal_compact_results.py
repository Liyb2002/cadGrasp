"""Freeze actual executed sources and disambiguate publication-only counts."""
import _bootstrap
import argparse,shutil
from co_common import *
from run_all import saved_groups,output_root


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--version',default='v1');args=parser.parse_args();root=output_root(args.object)
    frozen=root/'data/whole_compact_sources'/args.version;paths=[];sources={}
    for group in saved_groups(args.object):
        for mode in ['greedy','beam','structural-greedy']:
            out=root/group['id']/'step4/step4.2/compact'/mode
            if not (out/'data/report.json').exists():continue
            report=I.check_report(out/'data/report.json');paths.append((out,report))
            sources.update(report['provenance']['code'])
    sources.update(I.hashes([Path(__file__),HERE/'step4.2/compact.py',
        HERE/'helper_func/publish_compact_summary.py',HERE/'helper_func/audit_compact_volume.py',
        HERE/'tests/test_compact_volume.py']))
    aliases={}
    for relative,digest in sources.items():
        source=ROOT/relative;assert I.sha256(source)==digest
        target=frozen/relative
        if target.exists():assert I.sha256(target)==digest
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        aliases[relative]=str(target.relative_to(ROOT))
    for out,report in paths:
        archive=out/'data/metadata_history/before_source_seal'
        if not archive.exists():
            archive.mkdir(parents=True)
            shutil.copy2(out/'report.json',archive/'report.json')
            shutil.copy2(out/'data/report.json',archive/'data_report.json')
        physical=I.hashes([out/'support.obj',out/'layout.npz',out/'process.json']+list(out.glob('*_force.npz')))
        if report.get('metadata_only_publication_recovery'):
            report['geometry_evaluations_in_publication']=0
            report['original_search_actual_candidate_evaluations']=len(report['final_validation_attempts'])
            report['exact_evaluations']=len(report['final_validation_attempts'])
            report['timing']=None
            report['original_search_timing_unavailable']=True
        report['provenance']['code']={aliases.get(p,p):digest for p,digest in report['provenance']['code'].items()}
        report['provenance']['code'].update({aliases[str(Path(__file__).relative_to(ROOT))]:I.sha256(Path(__file__))})
        report['source_freeze_manifest']=str((frozen/'manifest.json').relative_to(ROOT))
        current=dict(report);current['artifacts']={k[3:] if k.startswith('../') else k:v for k,v in report['artifacts'].items()}
        save(out/'report.json',current);data=current.copy();data['artifacts']={'../'+k:v for k,v in current['artifacts'].items()}
        save(out/'data/report.json',data);I.check_report(out/'data/report.json')
        for relative,digest in physical.items():assert I.sha256(ROOT/relative)==digest
    save(frozen/'manifest.json',dict(complete=True,sources=sources,frozen_code_aliases=aliases,
        frozen_code_sha256={aliases[k]:v for k,v in sources.items()},
        material_result_count=len(paths),numerical_artifacts_changed=False,
        publication_only_counts_are_separately_recorded=True))
    print('COMPACT SEALED',len(paths),'results',len(sources),'actual source files',flush=True)


if __name__=='__main__':main()
