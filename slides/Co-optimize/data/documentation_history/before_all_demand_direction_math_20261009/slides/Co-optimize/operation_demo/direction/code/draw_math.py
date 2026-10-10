"""Restore the original direction math sheet without added explanations."""
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "vis" / "data" / "history" / "direction_math_before_geometry_chain_20261008.png"
OUT = HERE / "vis" / "direction_math.png"
shutil.copyfile(SOURCE, OUT)
assert OUT.read_bytes() == SOURCE.read_bytes()
print(OUT)
