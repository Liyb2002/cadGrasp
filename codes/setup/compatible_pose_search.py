"""Compatibility adapter; dataset precomputation lives in codes/precompute_objects."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from codes.precompute_objects.search import *
if __name__ == '__main__':
    import runpy
    runpy.run_module('codes.precompute_objects.search', run_name='__main__')
