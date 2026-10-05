"""Draw the connected support, its verified motion and current failure evidence."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step5_connect_support import connect as C
from step5_connect_support.belt_assembly import draw as draw_assembly
from step1.cases import pose_name


def run(name, static_only=False):
    out = C.OUTPUTS/name/pose_name()/C.STAGE
    for filename in ('bearing_failure.json', 'bearing_failure.npz', 'bearing_failure.png'):
        (out/filename).unlink(missing_ok=True)
    for filename in ('connectivity_failure.json', 'connectivity_failure.png',
                     'connectivity_union_mm.stl', 'connectivity_extra_shells_mm.stl'):
        (out/filename).unlink(missing_ok=True)
    draw_assembly(name, static_only)
    from step5_connect_support.bearing_failure import run as draw_bearing_failure
    draw_bearing_failure(name)
    from step5_connect_support.connectivity_failure import run as draw_connectivity_failure
    draw_connectivity_failure(name)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('objects',nargs='*');parser.add_argument('--static-only',action='store_true')
    args=parser.parse_args()
    for name in args.objects or C.OBJECTS:run(name,args.static_only)
