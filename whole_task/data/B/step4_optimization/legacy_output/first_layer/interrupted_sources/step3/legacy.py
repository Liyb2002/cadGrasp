"""Reuse the Co-optimize numerical kernels without its fixed-pose readers."""
import sys
from whole_task.step1.common import ROOT

COOPT = ROOT / "slides/Co-optimize"
sys.path.insert(0, str(COOPT / "helper_func"))
import co_common as kernel
from work_access import unpack_solid, _contains


def loaded_sources():
    """Record the actual repository Python dependencies, including legacy LPs."""
    result = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            from pathlib import Path
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix == ".py" and ".venv" not in path.parts:
                result.add(path)
    return sorted(result)
