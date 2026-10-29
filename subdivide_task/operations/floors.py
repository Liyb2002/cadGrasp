"""Actual installation floor halfspaces in the common material frame."""
import numpy as np
from subdivide_task.step3 import independent_geometry as G


def planes(material):
    rows = []
    for matrix in material:
        n = matrix[:3, 2]
        offset = float(n@matrix[:3, 3])
        if not any(np.linalg.norm(n-a)<1e-10 and abs(offset-b)<1e-10 for a,b in rows):
            rows.append((n, offset))
    return rows


def clip(value, floor_planes):
    for normal, offset in floor_planes:
        value = value.trim_by_plane(normal, offset/G.SCALE)
    G.volume(value)
    return value


def connected_material(value):
    """Keep the largest material component and its enclosed cavities.

    Local cutouts can detach unused scraps. Keeping a component is only a
    proposal: the full audit must still prove all loads and all actual feet.
    No global material-z direction is interpreted as gravity here.
    """
    positive, cavities = [], []
    for part in value.decompose():
        signed = float(part.volume())*G.SCALE**3
        if signed>1e-15:
            positive.append((signed, part))
        elif signed < -1e-15:
            mesh = G.mesh(part)
            mesh.invert()
            cavities.append(G.solid(mesh))
    if not positive:
        raise ValueError('No connected material remains')
    result = max(positive, key=lambda x:x[0])[1]
    if cavities:
        result = result-G.union(cavities)
    G.volume(result)
    return result


def condition(value, forbidden, envelope, tolerance, max_rounds=3):
    for turn in range(max_rounds+1):
        actual = G.mesh(value)
        native = G.solid(actual)
        residual = [G.volume(native^f) for f in forbidden]
        extra = G.volume(native-envelope)
        if max([extra,*residual])<=tolerance:
            return actual, dict(rounds=turn,forbidden_overlap_m3=residual,added_outside_parent_m3=extra)
        if turn==max_rounds:
            break
        value = native^envelope
        for f in forbidden:
            value = value-f
        if turn:
            # Repeating an exactly coincident Boolean can retain the same
            # tiny intrusion. Remove its local components with extra design
            # clearance; the original body and volume tolerance stay fixed.
            # A full load/exit audit must accept any contact area lost here.
            from .clearance import local_clearance
            cuts=[]
            for f,volume in zip(forbidden,residual):
                if volume>tolerance:
                    cuts.append(local_clearance(native,f,turn*1e-6))
            if cuts:value=value-G.union(cuts)
        value = connected_material(value)
    raise RuntimeError('Unresolved installed-solid exclusions: '+str(residual))
