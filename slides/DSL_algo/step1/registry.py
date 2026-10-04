"""Algorithm adapter for the shared object dataset registry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from codes.precompute_objects.registry import *
