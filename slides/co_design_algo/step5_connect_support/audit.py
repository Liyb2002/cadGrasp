"""Audit the current connected support, shared forces and whole insertion sweep."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import OBJECTS
from step5_connect_support.belt_assembly import audit as run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    for name in parser.parse_args().objects or OBJECTS:
        run(name)
