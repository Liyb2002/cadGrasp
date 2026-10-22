"""Share foot proposals across installations using the same material plane."""
import numpy as np
from whole_task.operations.state import transform


def shared_plane_rows(rows):
    """Change construction coordinates only; keep physical demands identical."""
    references=[];result=[]
    for row in rows:
        T=row['fixture_transform'];plane=np.r_[T[2,:3],-T[2,3]]
        reference=next((R for p,R in references if np.allclose(p,plane,atol=1e-8,rtol=0)),None)
        if reference is None:
            reference=T;references.append((plane,T))
        material=transform(row['hull'],np.linalg.inv(T))
        hull=transform(material,reference)
        if np.abs(hull[:,2]).max()>1e-8:raise ValueError('Construction demands do not share a ground plane')
        hull[:,2]=0.
        result.append(dict(row,fixture_transform=reference,hull=hull))
    return result
