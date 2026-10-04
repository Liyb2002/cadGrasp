"""Algorithm adapter for dataset head geometry and cached native candidates."""
import importlib.util
from pathlib import Path
import sys
_ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(_ROOT))
_spec=importlib.util.spec_from_file_location(__name__,_ROOT/'codes/precompute_objects/head_geometry.py')
_module=importlib.util.module_from_spec(_spec)
sys.modules[__name__]=_module
_spec.loader.exec_module(_module)
