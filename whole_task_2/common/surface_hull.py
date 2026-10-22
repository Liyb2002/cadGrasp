"""Extreme application points sufficient for a common world-direction cone."""
import numpy as np
from scipy.spatial import ConvexHull


def hull_vertex_ids(points):
    points=np.asarray(points,dtype=float);center=points.mean(0);delta=points-center
    _,singular,axes=np.linalg.svd(delta,full_matrices=False)
    rank=int(np.count_nonzero(singular>max(singular[0]*1e-12,1e-14)))
    if rank==0:
        ids=np.array([0]);residual=float(np.linalg.norm(delta,axis=1).max())
    elif rank==1:
        x=delta@axes[0];ids=np.unique([x.argmin(),x.argmax()]);residual=float(np.linalg.norm(delta-x[:,None]*axes[0],axis=1).max())
    else:
        projected=delta@axes[:rank].T
        hull=ConvexHull(projected);ids=np.asarray(hull.vertices)
        planar=np.linalg.norm(delta-projected@axes[:rank],axis=1).max()
        residual=float(max(0.,planar,np.max(projected@hull.equations[:,:rank].T+hull.equations[:,rank])))
    if residual>1e-12:raise ValueError('Application-point hull containment unresolved')
    return ids,dict(source_point_count=len(points),hull_vertex_count=len(ids),dimension=rank,maximum_residual_m=residual)
