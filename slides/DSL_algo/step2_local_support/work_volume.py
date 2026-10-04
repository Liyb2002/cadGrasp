"""Algorithm adapter for the shared dataset contact-head kernel."""
import importlib.util
from pathlib import Path
import sys
_ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(_ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
_spec=importlib.util.spec_from_file_location(__name__,_ROOT/'codes/precompute_objects/heads/work_volume.py')
_module=importlib.util.module_from_spec(_spec)
sys.modules[__name__]=_module
_spec.loader.exec_module(_module)

if __name__ == "__main__" and hasattr(_module, "main"):
    _module.main()
