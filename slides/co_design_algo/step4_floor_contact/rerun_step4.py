"""Batch entry point; outputs stay in each object/pose/stage directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sys
from step3_scheculer.batch_cases import main

if __name__ == '__main__':
    sys.exit(main(first=4, last=4))
