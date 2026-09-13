"""Rebuild Step2–5 for every object/pose with common 3-D withdrawal filtering."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sys
from step3_scheculer.batch_cases import main
if __name__ == '__main__':
    sys.exit(main(first=2,last=5,budget=24))
