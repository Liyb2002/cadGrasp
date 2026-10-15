"""Use the repository's bit-exact float64 OBJ writer for new Step5 solids."""
from pathlib import Path
from feasible_geometry import C
from co_common import D


def save_mesh(mesh,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    D.export_exact_obj(mesh,path)
    return C.trimesh.load(path,force='mesh',process=False)
