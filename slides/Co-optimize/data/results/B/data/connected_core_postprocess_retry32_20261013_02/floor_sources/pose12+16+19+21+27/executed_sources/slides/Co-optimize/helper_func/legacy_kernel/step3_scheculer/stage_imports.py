"""Load the numbered Step 3 subdirectories without treating dots as packages."""
from importlib import import_module
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
from pathlib import Path
import sys


DIRECTORIES = {
    'score': 'step3.1_score_candidate',
    'select': 'step3.2_select_contact',
    'optimize': 'step3.3_optimize_contact',
}


def load_stage(stage, module):
    directory = Path(__file__).resolve().parent.parent / DIRECTORIES[stage]
    package_name = f'_cadgrasp_{stage}'
    if package_name not in sys.modules:
        spec = ModuleSpec(package_name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(directory)]
        sys.modules[package_name] = module_from_spec(spec)
    return import_module(f'{package_name}.{module}')
