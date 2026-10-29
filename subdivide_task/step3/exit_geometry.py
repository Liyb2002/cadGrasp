"""Continuous translation of the original nonconvex object, in one fixed frame."""
import numpy as np
import manifold3d as md
from . import independent_geometry as G

PRISM_FACES=np.array([[2,1,0],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]],dtype=np.uint64)


def grounded_material(value):
    """Keep the grounded outer boundary and its cavities, discard orphan voids."""
    positive=[];cavities=[]
    for part in value.decompose():
        signed=float(part.volume())*G.SCALE**3
        if signed>1e-15:
            if G.mesh(part).bounds[0,2]<=1e-9:positive.append(part)
        elif signed<-1e-15:
            cavity=G.mesh(part);cavity.invert();cavities.append(G.solid(cavity))
    if len(positive)!=1:raise RuntimeError('Expected one grounded material boundary')
    # Decomposition gives oriented boundaries. Rebuild the set explicitly so
    # a cavity belonging to a discarded floating body cannot survive alone.
    result=positive[0]
    if cavities:result=result-G.union(cavities)
    G.volume(result)
    return result


def condition_exit(value,forbidden,parent,tolerance,max_rounds=3):
    """Recut unchanged forbidden sets; never enlarge the accepted material."""
    for turn in range(max_rounds+1):
        actual=G.mesh(value);native=G.solid(actual)
        overlaps=[G.volume(native^f)for f in forbidden];extra=G.volume(native-parent)
        if max([extra,*overlaps])<=tolerance:
            return actual,dict(rounds=turn,forbidden_overlap_m3=overlaps,added_outside_parent_m3=extra)
        if turn==max_rounds:break
        value=native^parent
        for f in forbidden:value=value-f
        value=grounded_material(value)
    raise RuntimeError('Unresolved exact-exit export: '+str(dict(overlaps=overlaps,extra=extra)))


def leading_prisms(body,direction,length):
    direction=np.asarray(direction,float);delta=length*direction/G.SCALE
    if not np.isfinite(delta).all() or abs(np.linalg.norm(direction)-1)>1e-12 or length<=0:
        raise ValueError('Positive length and a unit direction are required')
    vertices=np.asarray(body.vertices)/G.SCALE;triangles=vertices[body.faces]
    extended=triangles.astype(np.longdouble)
    signed=np.cross(extended[:,1]-extended[:,0],extended[:,2]-extended[:,0])@delta.astype(np.longdouble)
    for face in np.flatnonzero(signed>0):
        points=np.vstack([triangles[face],triangles[face]+delta])
        prism=md.Manifold(md.Mesh64(np.ascontiguousarray(points),PRISM_FACES))
        if prism.status()!=md.Error.NoError or prism.is_empty() or prism.volume()<=0:
            raise RuntimeError('A positive leading face prism is numerically unresolved: '+str(face))
        yield int(face),prism


def whole_object_sweep(body,direction,length):
    direction=np.asarray(direction,float);parts=[G.solid(body)]
    leading=0
    for face,prism in leading_prisms(body,direction,length):
        parts.append(prism);leading+=1
    # This is S union all forward boundary-face prisms, exactly S+[0,L*d]
    # in real arithmetic. No convex hull across the bunny or sampled poses.
    while len(parts)>1:
        parts=[G.union(parts[i:i+8]) for i in range(0,len(parts),8)]
    value=parts[0];G.volume(value)
    return value,dict(method='original nonconvex solid plus every positive leading face prism',
        whole_object=True,continuous_parameter_range=[0.,1.],leading_face_count=leading,
        source_face_count=len(body.faces),displacement_m=(length*direction).tolist(),
        convex_envelope=False,transverse_padding_m=0.,all_positive_leading_prisms_retained=True,
        arithmetic='Mesh64 in the same fixed scale as the support')


def continuous_checks(support,paths,tolerance):
    """Check every continuous face prism, avoiding a serialized sweep union."""
    rows=[];bad=[]
    for id,body,direction,length in paths:
        initial=G.solid(body);body_overlap=G.volume(support^initial)
        values=[];task_bad=[]
        threshold=tolerance/(len(body.faces)+1)
        for face,prism in leading_prisms(body,direction,length):
            overlap=G.volume(support^prism);values.append(overlap)
            if overlap>threshold:task_bad.append((id,face,prism))
        total=body_overlap+sum(values)
        if total>tolerance:
            bad.extend(task_bad)
            if body_overlap>threshold:bad.append((id,-1,initial))
        rows.append(dict(id=id,continuous_overlap_upper_bound_m3=float(total),
            initial_object_overlap_m3=float(body_overlap),leading_face_count=len(values),
            largest_face_prism_overlap_m3=float(max(values,default=0.)),passed=total<=tolerance))
    return rows,bad


def refine_face_prisms(value,paths,parent,forbidden,tolerance,max_rounds=3):
    """Resolve union-Boolean artifacts using the unchanged individual cells."""
    for turn in range(max_rounds+1):
        actual=G.mesh(value);native=G.solid(actual)
        rows,bad=continuous_checks(native,paths,tolerance)
        if all(r['passed']for r in rows):
            residual=[G.volume(native^f)for f in forbidden]
            extra=G.volume(native-parent)
            if max([extra,*residual])<=tolerance:
                return actual,dict(facewise_refinement_rounds=turn,continuous_checks=rows,
                    forbidden_overlap_m3=residual,added_outside_parent_m3=extra)
        if turn==max_rounds or not bad:break
        value=native
        for _,_,prism in bad:value=value-prism
        value=grounded_material(value)
    raise RuntimeError('Continuous source-face-prism certificate unresolved: '+str(rows))
