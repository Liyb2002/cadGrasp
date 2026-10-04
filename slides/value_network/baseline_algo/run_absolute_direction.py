"""Current world-direction/Step5 experiment entry; old NN weights stay archived."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'absolute_direction'))
from pipeline import main
if __name__=='__main__':main()
