"""Step0-only entry: original ground demands, then pairwise floor penetration checks."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step0_pose_selection.run_floor_points import main

if __name__ == '__main__':
    main()
