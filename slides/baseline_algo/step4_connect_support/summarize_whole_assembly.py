"""Index current per-case evidence without a separate summary directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.batch_cases import summarize


def main():
    summarize(5)


if __name__ == '__main__':
    main()
