"""Source/direction visibility bounds, independent of any support or trajectory.

A ray is pt + offset*n + t*(n + slope_x*e1 + slope_y*e2). An occluder
above that source plane projects to the source at xy-height*slope. For a convex
slope cell, the convex hull of vertex projections is its complete possible shadow.
The intersection of vertex shadows is an always-shadowed region when the
occluder's facing sign is fixed. Eroding the latter and expanding the former
leave an explicit numerical/visibility transition band; it is never called clear.
"""
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint, GeometryCollection
from shapely.ops import unary_union
from scipy.spatial import ConvexHull

POSITION_MARGIN = 1e-7   # lengths normalized by object maximum extent
MAX_DEPTH = 2          # complete cap -> 32 sectors -> four subtriangles each
RAY_BACKWARD_TOLERANCE_M = 1e-6  # trimesh.ray.ray_triangle_id: distance > -1e-6


def clipped_front(triangle, minimum_height=0.):
    """Clip to height >= 0 above the Step 1 offset ray-origin plane."""
    result=[]
    for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
        if a[2]>=minimum_height:result.append(a)
        if (a[2]<minimum_height<b[2]) or (b[2]<minimum_height<a[2]):
            result.append(a+(b-a)*((minimum_height-a[2])/(b[2]-a[2])))
    return np.asarray(result) if len(result)>=3 else np.empty((0,3))


def front_occluders(mesh_triangles, origin, basis, scale, offset, source, radius):
    local=(mesh_triangles-origin)@basis.T/scale
    local[:,:,2]-=offset/scale
    height=np.abs(local[:,:,2]).max(axis=1)
    lo=local[:,:,:2].min(axis=1)-height[:,None]*radius
    hi=local[:,:,:2].max(axis=1)+height[:,None]*radius
    bounds=np.asarray(source.bounds).reshape(2,2)
    possible=(local[:,:,2].max(axis=1)>=-RAY_BACKWARD_TOLERANCE_M/scale)&np.all(hi>=bounds[0]-POSITION_MARGIN,axis=1)&np.all(lo<=bounds[1]+POSITION_MARGIN,axis=1)
    polygons=[]
    for triangle in local[possible]:
        polygon=clipped_front(triangle,-RAY_BACKWARD_TOLERANCE_M/scale)
        if len(polygon)>=3:
            normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
            norm=np.linalg.norm(normal)
            if norm>0:polygons.append((polygon,normal/norm,clipped_front(triangle)))
    return polygons


def shadows(occluders, slopes, source):
    possible=[];certain=[]
    directions=np.c_[slopes,np.ones(len(slopes))]
    source_buffer=source.buffer(POSITION_MARGIN,join_style='mitre')
    for occluder in occluders:
        vertices,normal=occluder[:2]
        projected=vertices[:,None,:2]-vertices[:,None,2:]*slopes[None]
        envelope=MultiPoint(projected.reshape(-1,2)).convex_hull
        if not envelope.intersects(source_buffer):continue
        possible.append(envelope)
        dots=directions@normal
        if not (dots.min()>1e-10 or dots.max()<-1e-10):continue
        if len(occluder)>2:
            vertices=occluder[2]
            if len(vertices)<3:continue
            projected=vertices[:,None,:2]-vertices[:,None,2:]*slopes[None]
        common=None
        for index in range(len(slopes)):
            shadow=MultiPoint(projected[:,index]).convex_hull
            common=shadow if common is None else common.intersection(shadow)
            if common.is_empty or common.area==0:break
        if common is not None and common.area>0:certain.append(common)
    a=unary_union(possible) if possible else GeometryCollection()
    b=unary_union(certain) if certain else GeometryCollection()
    return a,b


def triangle_parts(shape):
    """Preserve even line/point sources: their cone sweeps can have volume."""
    if shape.is_empty:return []
    if shape.geom_type=='Point':return [np.repeat(np.asarray(shape.coords),3,axis=0)]
    if shape.geom_type in ['LineString','LinearRing']:
        xy=np.asarray(shape.coords)
        return [np.array([a,b,b]) for a,b in zip(xy[:-1],xy[1:])]
    if shape.geom_type!='Polygon':
        return [tri for part in shape.geoms for tri in triangle_parts(part)]
    triangulated=shapely.constrained_delaunay_triangles(shape)
    pieces=[np.asarray(t.exterior.coords)[:3] for t in triangulated.geoms if t.area>0]
    measured=sum(abs(float(np.cross(t[1]-t[0],t[2]-t[0])))/2 for t in pieces)
    if not np.isclose(measured,shape.area,rtol=1e-8,atol=1e-18):
        raise RuntimeError('Visibility triangulation changed source area')
    return pieces


def split_slopes(slopes):
    if len(slopes)>3:
        return [np.vstack([np.zeros(2),a,b]) for a,b in zip(slopes,np.roll(slopes,-1,axis=0))]
    a,b,c=slopes;ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
    return [np.array([a,ab,ca]),np.array([ab,b,bc]),np.array([ca,bc,c]),np.array([ab,bc,ca])]


def build_cells(mesh, triangles, normals, e1, e2, scale, offset, slopes, max_depth=MAX_DEPTH):
    cells=[];records=[]
    for face,(tri,n,x,y) in enumerate(zip(triangles,normals,e1,e2)):
        basis=np.array([x,y,n]);origin=tri[0]
        local=(tri-origin)@basis[:2].T/scale;source=Polygon(local)
        occluders=front_occluders(mesh.triangles,origin,basis,scale,offset,source,np.linalg.norm(slopes,axis=1).max())
        counts={'visible_cells':0,'transition_cells':0,'fully_occluded_cells':0,'visited_cells':0,'occluded_parameter_measure':0.,'transition_cover_added_parameter_measure':0.}
        def emit(shape,directions,clear,depth):
            for part in triangle_parts(shape):
                cells.append((origin+part@basis[:2]*scale,face,directions,clear,depth))
                counts['visible_cells' if clear else 'transition_cells']+=1
        def visit(source,directions,depth):
            if source.is_empty:return
            if source.area==0:
                emit(source,directions,False,depth);return
            counts['visited_cells']+=1
            possible,certain=shadows(occluders,directions,source)
            visible=source.difference(possible.buffer(POSITION_MARGIN,join_style='mitre')) if not possible.is_empty else source
            upper=source.difference(certain.buffer(-POSITION_MARGIN,join_style='mitre')) if not certain.is_empty else source
            counts['occluded_parameter_measure']+=max(0.,source.area-upper.area)*Polygon(directions).area
            emit(visible,directions,True,depth)
            transition=upper.difference(visible)
            if upper.is_empty or upper.area==0:counts['fully_occluded_cells']+=1
            if transition.is_empty:return
            # A convex cover prevents polygon/line fragments from multiplying.
            # It enlarges only the unresolved upper bound, never the clear set.
            cover=transition.convex_hull
            counts['transition_cover_added_parameter_measure']+=max(0.,cover.area-transition.area)*Polygon(directions).area
            transition=cover
            if depth>=max_depth:
                emit(transition,directions,False,depth)
            else:
                for child in split_slopes(directions):visit(transition,child,depth+1)
        visit(source,slopes,0)
        records.append(dict(work_face_index=face,possible_occluder_faces=len(occluders),**counts))
        if (face+1)%100==0:print('  work visibility',face+1,'/',len(triangles),'families',len(cells),flush=True)
    return cells,dict(method='Adaptive source shadows over convex direction cells',
        max_direction_depth=max_depth,normalized_position_margin=POSITION_MARGIN,
        ray_backward_tolerance_m=RAY_BACKWARD_TOLERANCE_M,
        possible_shadow='Convex hull of projections for all direction-cell vertices',
        always_shadow='Intersection of projections with constant occluder-facing sign; eroded for numerical margin',
        transition_policy='Retained as an explicit upper bound; an occluded witness is never a collision certificate',
        transition_source_cover='Convex cover of each unresolved source region; includes all line and point remnants',
        work_faces=records,cell_count=len(cells),
        certified_visible_cells=sum(r['visible_cells'] for r in records),
        transition_cells=sum(r['transition_cells'] for r in records),
        fully_occluded_cells_removed=sum(r['fully_occluded_cells'] for r in records),
        occluded_parameter_measure=sum(r['occluded_parameter_measure'] for r in records),
        transition_cover_added_parameter_measure=sum(r['transition_cover_added_parameter_measure'] for r in records),
        parameter_measure='Source area divided by D^2 times direction-slope area; not physical volume')
