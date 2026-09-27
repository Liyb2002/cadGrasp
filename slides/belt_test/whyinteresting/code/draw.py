"""Three-task overview: one blue module and one base with three sockets."""
from pathlib import Path
import argparse
import json
import sys

HERE=Path(__file__).resolve().parent
COMMON=HERE.parents[1]/'code'
sys.path.insert(0,str(COMMON))
from fixture_geometry import verify
from shared_base import build, figure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-forces','--no-force-labels',dest='no_forces',action='store_true')
    parser.add_argument('--both',action='store_true',help='Regenerate both overview variants.')
    args=parser.parse_args()
    cases,_,meta=build()
    report=verify(cases,meta)
    (HERE/'geometry_report.json').write_text(json.dumps(report,indent=2)+'\n')
    for no_forces in ([False,True] if args.both else [args.no_forces]):
        suffix='_no_labels' if no_forces else ''
        figure.draw(cases,HERE.parent/f'overview{suffix}.png',forces=not no_forces)


if __name__=='__main__':
    main()
