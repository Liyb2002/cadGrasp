"""Set up direct script entry points for the organized Co-optimize tree."""
import os
import sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/co_optimize_matplotlib')
COOPT_ROOT = Path(__file__).resolve().parents[1]
for folder in ('helper_func', 'helper_func/optimization', 'vis_func', 'step3.1', 'step3.2', 'step3.3', 'step4.1', 'step4.2'):
    location = str(COOPT_ROOT / folder)
    if location not in sys.path:
        sys.path.insert(0, location)
