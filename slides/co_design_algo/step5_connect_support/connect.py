"""Build one connected rigid support with the current direction-first search."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import OUTPUTS, OBJECTS
from step5_connect_support.belt_assembly import build, SCHEMA, STAGE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--edge-budget', type=int, default=2000,
                        help='Direction candidate budget, capped at 24 by the current search')
    args = parser.parse_args()
    for name in args.objects or OBJECTS:
        build(name, edge_budget=args.edge_budget)
