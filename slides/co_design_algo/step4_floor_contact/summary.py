"""Compatibility entry point for the current Step 4/5 twelve-case summary.

The old ground-only summary is superseded; the default pipeline now runs Step 5
and retains individual successes even when a complete case is not certified.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step5_connect_support.summary import build

if __name__ == '__main__':
    build()
