"""Six-second pose2 translation AFTER the existing Juxtapose operation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'combined/code'))
from render import main


if __name__ == '__main__':
    main(only='translation')
