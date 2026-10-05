"""CLI entry point for current whole-assembly floor-demand figures."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import OBJECTS
from step4_floor_contact.whole_assembly import draw as run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    for name in parser.parse_args().objects or OBJECTS:
        run(name)
