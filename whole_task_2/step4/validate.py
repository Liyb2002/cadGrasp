"""Replay the optimized support stored outside the one-image output directory."""
import argparse
import json
from pathlib import Path
from whole_task_2.step1.common import ROOT,DEFAULT_OUTPUT,sha256,write_json
from .validate_optimized import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',default='B')
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--workdir',type=Path)
    args=parser.parse_args()
    active=ROOT/'whole_task_2/data'/args.object/'step4_active.json'
    folder=args.workdir or (Path(json.loads(active.read_text())['workdir'])if active.exists()
                           else ROOT/'whole_task_2/data'/args.object/'step4_spatial_operations')
    report=json.loads((folder/'report.json').read_text())
    if report.get('method')=='independent_membrane_family_merges':
        from .membrane_search import validate as validate_membranes
        source=ROOT/'whole_task_2/data'/args.object/'step3_membranes'
        model,result=validate_membranes(source,report['groups'],folder,
            max_merge_ratio=report.get('merge_volume_ratio_limit'))
        report['validation_sha256']=sha256(folder/'validation.json');write_json(folder/'report.json',report)
        if active.exists():
            current=json.loads(active.read_text())
            if Path(current['workdir']).resolve()==folder.resolve():
                current['report_sha256']=sha256(folder/'report.json');write_json(active,current)
        return model,result
    validate(folder,args.output_root,args.object)


if __name__=='__main__':
    main()
