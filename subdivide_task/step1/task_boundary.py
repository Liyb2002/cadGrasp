"""Closed boundaries on the actual triangulated task surface, without projection."""
import numpy as np


def candidate_region(arrays,pose_index):
    """All processable surface for one pose; exclusive assignment is a later step."""
    region=np.asarray(arrays['reach'][:,pose_index],bool)&arrays['target']
    if np.any(region&~arrays['surface_inside_box'][:,pose_index]):
        raise ValueError('A displayed reachable face lies outside the 3D work box')
    return region


def boundary_loops(mesh,mask):
    """Closed 3D boundary walks, longest first, including lobes touching at a vertex.

    At a pinched boundary, an Euler walk visits every incident boundary edge
    exactly once and returns to its start without dropping either lobe.
    """
    edges={}
    for f in np.flatnonzero(mask):
        vertices=mesh.faces[f]
        for a,b in zip(vertices,np.roll(vertices,-1)):
            edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append(int(f))
    boundary=[(a,b,faces[0]) for (a,b),faces in sorted(edges.items()) if len(faces)==1]
    neighbours={}
    for i,(a,b,_) in enumerate(boundary):
        neighbours.setdefault(a,[]).append((i,b));neighbours.setdefault(b,[]).append((i,a))
    if any(len(v)%2 for v in neighbours.values()):
        raise ValueError('Task boundary has an open end; cannot draw a closed contour')
    available=set(range(len(boundary)));out=[]
    while available:
        first=min(available);start=boundary[first][0]
        stack=[start];edge_stack=[];reverse_vertices=[];reverse_edges=[]
        while stack:
            options=[(i,b) for i,b in neighbours[stack[-1]] if i in available]
            if options:
                edge,following=min(options);available.remove(edge)
                stack.append(following);edge_stack.append(edge)
            else:
                reverse_vertices.append(stack.pop())
                if edge_stack:reverse_edges.append(edge_stack.pop())
        ids=reverse_vertices[::-1];walk=reverse_edges[::-1]
        faces=[boundary[edge][2] for edge in walk]
        if ids[0]!=ids[-1] or any(set(ids[k:k+2])!=set(boundary[edge][:2]) for k,edge in enumerate(walk)):
            raise ValueError('Task boundary walk does not close on its source edges')
        length=float(np.linalg.norm(np.diff(mesh.vertices[ids],axis=0),axis=1).sum())
        out.append(dict(vertex_ids=np.asarray(ids,int),face_ids=np.asarray(faces,int),length_m=length))
    return sorted(out,key=lambda loop:-loop['length_m'])


def sample_loop(mesh,loop,face_axes,intervals):
    """Visit every 3D boundary edge and corner, including a final return to the start.

    The axis at a vertex blends its two incident faces' certified axes. Every
    resulting contact and axis must still be checked before a video is published.
    """
    ids=loop['vertex_ids'];faces=loop['face_ids'];n=len(faces)
    if intervals<n:raise ValueError('Need at least one interval per boundary edge')
    vertices=mesh.vertices[ids];lengths=np.linalg.norm(np.diff(vertices,axis=0),axis=1)
    extra=(intervals-n)*lengths/lengths.sum();counts=1+np.floor(extra).astype(int)
    counts[np.argsort(-(extra-np.floor(extra)))[:intervals-int(counts.sum())]]+=1
    edge_axes=np.asarray(face_axes)[faces]
    vertex_axes=edge_axes+np.roll(edge_axes,1,axis=0)
    vertex_axes/=np.linalg.norm(vertex_axes,axis=1)[:,None]
    points=[];axes=[];owners=[];edge_ids=[];parameters=[]
    for i,count in enumerate(counts):
        t=np.arange(count)/count
        points.extend((1-t[:,None])*vertices[i]+t[:,None]*vertices[i+1])
        directions=(1-t[:,None])*vertex_axes[i]+t[:,None]*vertex_axes[(i+1)%n]
        axes.extend(directions/np.linalg.norm(directions,axis=1)[:,None])
        owners.extend([faces[i]]*count);edge_ids.extend([i]*count);parameters.extend(t)
    points.append(points[0]);axes.append(axes[0]);owners.append(faces[0]);edge_ids.append(0);parameters.append(0.)
    return dict(points=np.asarray(points),axes=np.asarray(axes),faces=np.asarray(owners,int),
                edge_indices=np.asarray(edge_ids,int),edge_parameters=np.asarray(parameters))
