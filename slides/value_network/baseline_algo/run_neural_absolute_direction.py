"""Trained absolute-volume Step3 followed by fresh Step4 and unchanged Step5."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train/absolute_volume'))
from batch import main

if __name__ == '__main__':
    main()
