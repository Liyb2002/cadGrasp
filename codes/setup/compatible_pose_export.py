"""Compatibility adapter for the shared dataset publisher and verifier."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from codes.precompute_objects.run import publish_plan as publish

def verify(folder):
    from codes.precompute_objects.verify import verify as audit
    return audit(Path(folder).name)
