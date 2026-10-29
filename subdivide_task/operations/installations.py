"""Relationships between installations of one complete material body."""
import numpy as np


def floor_angles(layout):
    normals=layout.material[:,:3,2]
    return np.degrees(np.arccos(np.clip(normals@normals.T,-1.,1.)))


def standing_pairs(layout,tolerance_deg=1e-5):
    angles=floor_angles(layout)
    return [(int(i),int(j))for i,j in np.argwhere(np.triu(np.abs(angles-90.)<tolerance_deg,1))]


def representative_views(layout,object_center):
    """Choose a flat/side pair that most clearly exposes a shared seat."""
    normals=layout.material[:,:3,2]
    main=int(np.argmax((normals@normals.T>1.-1e-9).sum(1)))
    flat=np.flatnonzero(normals@normals[main]>1.-1e-9)
    angles=floor_angles(layout)
    side=np.flatnonzero(np.abs(angles[main]-90.)<1e-5)
    if not len(side):
        side=np.flatnonzero(angles[main]>1e-5)
    if not len(side):return main,main
    transforms=layout.material@layout.poses
    centers=np.einsum('nij,j->ni',transforms[:,:3,:3],object_center)+transforms[:,:3,3]
    return min(((int(i),int(j))for i in flat for j in side),
               key=lambda ij:np.linalg.norm(centers[ij[0]]-centers[ij[1]])+
                   .05*np.linalg.norm(transforms[ij[0],:3,:3]-transforms[ij[1],:3,:3]))
