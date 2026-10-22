"""Small group translations that leave all installation floor planes fixed."""
import numpy as np
from whole_task.operations.state import transform


def alternatives(model,layout,owners):
    own=set(owners)
    normals=layout.material[:,:3,2]
    _,s,vh=np.linalg.svd(normals[list(owners)],full_matrices=True)
    rank=np.count_nonzero(s>1e-8)
    tangent=vh[rank:]
    if not len(tangent):return
    if len(tangent)==1:
        directions=np.array([tangent[0],-tangent[0]])
    else:
        phi=np.linspace(0.,2*np.pi,24,endpoint=False)
        directions=np.cos(phi)[:,None]*tangent[0]+np.sin(phi)[:,None]*tangent[1]
    directions=directions[(directions@normals.T).min(1)>-1e-10]
    if not len(directions):return
    centers=np.array([transform(model.source.center_mass,M@P)for M,P in zip(layout.material,layout.poses)])
    center=centers[list(owners)].mean(0)
    others=centers[[k for k in range(len(centers))if k not in own]]
    if not len(others):return
    scores=np.linalg.norm(center+.1*directions[:,None]-others,axis=2).min(1)
    chosen=[]
    for index in np.argsort(-scores):
        direction=directions[index]
        if all(direction@old<.9 for old in chosen):chosen.append(direction)
        if len(chosen)==2:break
    for distance in(.05,.1,.2):
        for direction in chosen:
            delta=distance*direction
            q=layout.copy()
            q.material[list(owners),:3,3]+=delta
            yield q,delta
