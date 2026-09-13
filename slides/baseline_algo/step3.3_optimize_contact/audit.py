"""Size optimization is audited together with its preceding and following rounds."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer.audit import run
from step1.needs import OBJECTS

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or OBJECTS:
        run(name)
