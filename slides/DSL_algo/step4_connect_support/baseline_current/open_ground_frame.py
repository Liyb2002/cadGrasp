"""Open ground frames and one slender stem from each fixed physical head."""
import manifold3d as md
import numpy as np
from shapely.geometry import Polygon, MultiPoint
import trimesh

from step4_connect_support.baseline_current import build_coupled_saddle as S
from step4_connect_support.baseline_current import convex_foot as F
from step2_local_support import geometry as G

WIDTH=.003
HEIGHT=.003
RADIUS=.003


def union(parts):
    return md.Manifold.batch_boolean(parts,md.OpType.Add)


def prism(xy,low,high,basis,offset):
    xy=np.asarray(xy)
    points=np.vstack([np.c_[xy,np.full(len(xy),z)] for z in (low,high)])@basis+offset
    return S.solid(G.hull_mesh(points))


def frame(xy,basis,offset,carve):
    """Keep a 3 mm boundary strip, not the interior of the support polygon."""
    polygon=Polygon(xy).convex_hull
    inner=polygon.buffer(-WIDTH,join_style=2)
    if inner.is_empty:raise RuntimeError('Ground hull is too small for an open frame')
    outer=prism(np.asarray(polygon.exterior.coords)[:-1],0,HEIGHT,basis,offset)
    hollow=prism(np.asarray(inner.exterior.coords)[:-1],-.001,HEIGHT+.001,basis,offset)
    ring=carve(outer-hollow)
    # Clipping can leave tiny islands inside the required contact hull. They
    # serve no hull constraint and need no artificial connecting material.
    removed_volume=0.
    parts=sorted(ring.decompose(),key=lambda x:x.volume())
    for part in list(parts):
        remaining=[p for p in parts if p is not part]
        if not remaining:continue
        candidate=union(remaining)
        ground=F.landing(candidate,basis,offset)
        if polygon.difference(ground.convex_hull.buffer(1e-10)).area<=1e-12:
            removed_volume+=abs(float(part.volume()))*S.SCALE**3
            parts=remaining
    ring=union(parts)
    ground=F.landing(ring,basis,offset)
    missing=polygon.difference(ground.convex_hull.buffer(1e-10)).area
    if missing>1e-12:raise RuntimeError(f'Actual frame convex hull loses required ground area: {missing}')
    # Object exclusions can break a ring into arcs; final material connectivity
    # is checked after stems and thin inter-body joints have been constructed.
    mesh=S.unpack(ring);world=S.local_to_world(mesh.vertices,basis,offset)
    ids=mesh.faces[np.all(np.abs(world[mesh.faces,2])<1e-9,axis=1)]
    points=world[ids,:2].mean(axis=1)
    anchors=np.c_[points,np.full(len(points),HEIGHT/2)]@basis+offset
    return ring,anchors,dict(outer_xy_m=np.asarray(polygon.exterior.coords)[:-1].tolist(),
        width_m=WIDTH,height_m=HEIGHT,required_hull_area_m2=float(polygon.area),
        actual_material_area_m2=float(ground.area),actual_contact_hull_area_m2=float(ground.convex_hull.area),
        missing_required_hull_area_m2=float(missing),interior_is_not_material=True)


def attach(patch,frames,anchors,bases,offsets,carve,connected_piece):
    """One stem with a short head collar; no head-to-entire-polygon hull."""
    direction=patch['root'].mean(0)-patch['v'].mean(0)
    direction/=np.linalg.norm(direction)
    bead=trimesh.creation.icosphere(subdivisions=1,radius=RADIUS).vertices
    choices=[]
    center=patch['v'].mean(0)
    for k,points in enumerate(anchors):
        for j in np.argsort(np.linalg.norm(points-center,axis=1)):
            choices.append((float(np.linalg.norm(points[j]-center)),k,int(j)))
    # Adjacent triangle centroids are redundant. Keep diverse landing options.
    selected=[]
    for distance,k,j in sorted(choices):
        point=anchors[k][j]
        if any(f==k and np.linalg.norm(point-p)<.005 for _,f,p in selected):continue
        selected.append((distance,k,point))
        if len(selected)>=80:break
    for depth in (.008,.014,.022,.035):
        start=center+depth*direction
        cap=carve(S.solid(G.hull_mesh(np.vstack([patch['v'],start+bead]))))+patch['original']
        cap=connected_piece(cap,[patch['original']])
        if cap is None:continue
        for distance,k,end in selected:
            # A direct rod or one elbow still forms a single serial connection.
            for lift in (0.,.012,.025):
                route=[start,end] if lift==0 else [start,start+lift*direction,end]
                stem=carve(union([S.solid(G.hull_mesh(np.vstack([a+bead,b+bead])))
                    for a,b in zip(route[:-1],route[1:])]))
                proposed=connected_piece(cap+stem,[patch['original']])
                if proposed is None:continue
                overlap=abs(float((proposed^frames[k]).volume()))*S.SCALE**3
                if overlap<1e-12:continue
                return proposed,dict(floor_index=k,path_m=np.asarray(route).tolist(),
                    stem_radius_m=RADIUS,head_collar_depth_m=depth,
                    stem_count=1,frame_intersection_volume_m3=overlap)
    raise RuntimeError(f'No slender head-to-frame route found for {patch["id"]}')
