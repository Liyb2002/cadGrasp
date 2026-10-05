"""Index saved stage reports in their existing case directories."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.batch_cases import CASES, summarize
from step1.registry import active_objects


def build(objects=None):
    objects = active_objects() if objects is None else objects
    return summarize(6, [case for case in CASES if case[0] in objects])


if __name__ == '__main__':
    build()
