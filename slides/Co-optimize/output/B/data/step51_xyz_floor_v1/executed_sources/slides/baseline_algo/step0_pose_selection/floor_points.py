"""Compatibility adapter for dataset precomputation geometry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from codes.precompute_objects.floor_points import *
