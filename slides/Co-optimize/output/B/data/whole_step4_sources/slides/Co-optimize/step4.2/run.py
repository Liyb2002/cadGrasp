import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_pipeline import main_step42
if __name__=='__main__':main_step42()
