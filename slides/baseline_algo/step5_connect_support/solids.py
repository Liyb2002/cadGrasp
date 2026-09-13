"""One separately insertable solid from each contact head to its own ground pad."""
from step1.needs import COORD
import numpy as np
import trimesh
from shapely.geometry import Polygon
from step2_local_support import insertion as D
from step5_connect_support import ground as G,ring as R


def joined_heads(mesh,contacts,depth):
    analyzer=D.Analyzer(mesh,depth)
    heads=[];owners=[]
    for contact in contacts:
        new=analyzer.heads(contact);heads.extend(new)
        owners.extend(dict(candidate_id=contact['candidate_id'],source_face=int(f)) for f in np.unique(contact['source_faces']))
    return heads,owners


def ring_plan(mesh,contact,heads,ring,angle):
    basis=G.frame(angle);scale=float(mesh.extents.max())
    hit=R.ray_exit(ring,COORD.floor(contact['center_m']),-COORD.floor(basis[0]))
    if hit is None:raise ValueError('Withdrawal ray does not reach this expanded boundary')
    boundary,distance=hit
    # A narrow post may straddle the strip's inner edge; it only needs a positive
    # area joint to its assigned ground arc. Keep its center on the exact ray hit.
    local=np.concatenate([h.vertices for h in heads])@basis.T
    back=float(COORD.lift_floor(boundary)@basis[0]);y=float(COORD.lift_floor(boundary)@basis[1])
    thickness=.022*scale
    if back+thickness/2>local[:,0].min():
        raise ValueError('Boundary is not yet behind the complete contact head')
    if local[:,2].min()<=0:raise ValueError('Contact head reaches the floor')
    return dict(candidate_id=contact['candidate_id'],bearing_deg=float(angle),basis=basis,
        direction=basis[0],back_m=back,post_y_m=y,member_width_m=thickness,
        anchor_xz_m=boundary,ray_distance_m=float(distance),expansion=ring['expansion'],
        ground_height_m=ring['height_m'])


def connector(mesh,contact,heads,plan):
    """Withdrawal-aligned ribs feed a local crossbar and a narrow vertical post.

    The crossbar is only as tall/wide as the contact, not a wall down to ground.
    Every source head is retained in its rib. Final sweeps check all material.
    """
    basis=plan['basis'];back=plan['back_m'];y=plan['post_y_m'];t=plan['member_width_m']
    local=np.concatenate([h.vertices for h in heads])@basis.T
    zlo=float(local[:,2].min());zhi=float(local[:,2].max());middle=(zlo+zhi)/2
    bottom=min(.5*plan['ground_height_m'],.5*zlo)
    low=[back-t/2,min(y-t/2,float(local[:,1].min())),max(bottom,zlo-t/2)]
    high=[back+t/2,max(y+t/2,float(local[:,1].max())),zhi+t/2]
    parts=[G.box([back-t/2,y-t/2,bottom],[back+t/2,y+t/2,middle+t/2],basis),
           G.box(low,high,basis)]
    labels=['ground_post','contact_crossbar']
    for index,head in enumerate(heads):
        points=head.vertices@basis.T;rear=points.copy();rear[:,0]=back
        parts.append(D.engine.hull_mesh(np.vstack([points,rear])@basis))
        labels.append(f'contact_rib_{index:03d}')
    return parts,labels


def with_ground(connector_parts,connector_labels,plan,polygons):
    height=plan['ground_height_m'];parts=[];labels=[]
    for index,xy in enumerate(polygons):
        points=COORD.lift_floor(xy)
        parts.append(D.engine.hull_mesh(np.vstack([points,points+[0.,height,0.]])))
        labels.append(f'ground_strip_{index:03d}')
    corners=COORD.lift_floor(np.concatenate(polygons))
    plan={**plan,'ground_polygons_xz_m':polygons,'ground_corners_m':corners,
          'ground_area_m2':float(sum(Polygon(p).area for p in polygons))}
    return parts+connector_parts,labels+connector_labels,plan


def union_parts(parts,scale):
    origin=np.concatenate([p.vertices for p in parts]).mean(axis=0)
    copies=[G.solid64(part,origin,scale) for part in parts]
    union=G.manifold.Manifold.batch_boolean(copies,G.manifold.OpType.Add)
    if union.status()!=G.manifold.Error.NoError:raise ValueError(f'Manifold union failed: {union.status()}')
    data=union.to_mesh64()
    joined=trimesh.Trimesh(np.asarray(data.vert_properties[:,:3])*scale+origin,np.asarray(data.tri_verts),process=False)
    # Boundary components are not material components: a cavity has its own
    # closed, negatively oriented shell but does not disconnect the solid.
    shells=list(joined.split(only_watertight=False))
    positive=[p for p in shells if p.volume>0]
    negative=[p for p in shells if p.volume<0]
    zero=[p for p in shells if p.volume==0]
    contained=not negative; outside_volume=0.
    if len(positive)==1 and negative:
        outer=G.solid64(positive[0],origin,scale)
        cavities=[]
        for shell in negative:
            flipped=trimesh.Trimesh(shell.vertices,shell.faces[:,::-1],process=False)
            cavities.append(G.solid64(flipped,origin,scale))
        voids=G.manifold.Manifold.batch_boolean(cavities,G.manifold.OpType.Add)
        outside=voids-outer
        if outside.status()!=G.manifold.Error.NoError:
            raise ValueError('Cavity containment Boolean is unresolved')
        outside_volume=abs(float(outside.volume()))*scale**3
        contained=outside_volume<=1e-14*scale**3
    count=len(positive)+len(zero)
    if len(positive)==1 and negative and not contained:count+=len(negative)
    record=dict(component_count=count,boundary_component_count=len(shells),
        positive_boundary_shells=len(positive),cavity_count=len(negative),zero_boundary_shells=len(zero),
        cavity_containment_verified=contained,cavity_outside_volume_m3=outside_volume,
        cavity_containment_tolerance_m3=1e-14*scale**3,
        connectivity_rule='One positive outer shell; all negative cavity shells contained; no separate positive or zero-volume shell',
        watertight=bool(joined.is_watertight),
        consistently_wound=bool(joined.is_winding_consistent),volume_m3=float(joined.volume))
    record['one_solid']=bool(count==1 and contained and record['watertight'] and record['consistently_wound'] and record['volume_m3']>0)
    return joined,record


def pack_parts(parts,labels,joined):
    return dict(part_vertices_m=np.concatenate([p.vertices for p in parts]),
        part_faces=np.concatenate([p.faces for p in parts]),
        vertex_offsets=np.cumsum([0]+[len(p.vertices) for p in parts]),
        face_offsets=np.cumsum([0]+[len(p.faces) for p in parts]),part_labels=np.asarray(labels,str),
        union_vertices_m=np.asarray(joined.vertices),union_faces=np.asarray(joined.faces))


def unpack_parts(arrays):
    parts=[]
    for i in range(len(arrays['part_labels'])):
        va,vb=arrays['vertex_offsets'][i:i+2];fa,fb=arrays['face_offsets'][i:i+2]
        parts.append(trimesh.Trimesh(arrays['part_vertices_m'][va:vb],arrays['part_faces'][fa:fb],process=False))
    return parts
