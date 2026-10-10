"""Step3.2 displays the full registered work-exclusion union; no new material."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap

def run(name,group,base=None):
    from step32_render import render
    return render(name,group,base)
