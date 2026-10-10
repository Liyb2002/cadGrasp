"""Whole-set continuation: optimize actual material while retaining feasibility."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
from run_compact_volume import main

if __name__=='__main__':main()
