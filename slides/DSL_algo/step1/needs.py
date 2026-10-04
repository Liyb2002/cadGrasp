"""Algorithm adapter: immutable loads come from codes/precompute_objects."""
import importlib.util
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[3]
_BASELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_BASELINE))
_NAME = 'step1.needs' if __name__ == '__main__' else __name__
_spec = importlib.util.spec_from_file_location(_NAME, _ROOT/'codes/precompute_objects/loads.py')
_module = importlib.util.module_from_spec(_spec)
sys.modules[_NAME] = _module
_spec.loader.exec_module(_module)
_module.BASELINE = _BASELINE
_module.OUTPUTS = _BASELINE/'output'
if __name__ == '__main__':
    _module.main()
else:
    sys.modules[__name__] = _module
