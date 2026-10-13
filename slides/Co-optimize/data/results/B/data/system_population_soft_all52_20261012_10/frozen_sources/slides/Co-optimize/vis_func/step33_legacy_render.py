"""Existing rendering function separated from the numerical solver."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
def render(mesh,shell,parts,rows,out,points,name="B",group=None):
    from step33_render import draw_group
    root=HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name
    if group is None:group=dict(id=str(out.parent.parent.relative_to(root)),poses=[row['pose'] for row in rows])
    draw_group(name,group,out.parent,[S.unpack(part) for part in parts],rows)

