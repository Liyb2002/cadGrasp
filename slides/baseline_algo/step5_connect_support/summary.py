"""Index saved stage reports in their existing case directories."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.batch_cases import CASES, summarize


def build(objects=('A1-f', 'B', 'C5')):
    return summarize(5, [case for case in CASES if case[0] in objects])


if __name__ == '__main__':
    build()
