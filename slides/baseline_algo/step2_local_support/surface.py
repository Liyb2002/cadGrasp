"""Surface sampling and true-area polygon utilities for circular candidates."""
import numpy as np
from trimesh.triangles import closest_point

CENTER_COUNT=200
FLOOR_CLEARANCE_M=.0015
SEED=20260907

def fan(poly):
    if len(poly) < 3:
        return np.empty((0, 3, 3))
    # A center fan avoids the very thin, nearly collinear triangles produced
    # by a vertex fan on a finely approximated circle.
    return np.stack([np.repeat(poly.mean(axis=0)[None], len(poly), axis=0),
                     poly, np.roll(poly, -1, axis=0)], axis=1)


def areas(tri):
    return np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2


def area(poly):
    return float(areas(fan(poly)).sum())


def clip(poly, direction, bound, upper=True):
    values = poly@direction-bound
    if not upper:
        values = -values
    inside = values <= 0
    if inside.all():
        return poly.copy()
    if not inside.any():
        return np.empty((0, 3))
    result = []
    for j in range(len(poly)):
        i = j-1
        if inside[i] != inside[j]:
            result.append(poly[i]+values[i]/(values[i]-values[j])*(poly[j]-poly[i]))
        if inside[j]:
            result.append(poly[j])
    return np.array(result)


def eligible_polygons(domain):
    """Exclude work faces and clip the actual surface at the floor-band plane."""
    mesh = domain.mesh
    work = np.zeros(len(mesh.faces), bool)
    work[domain.work_ids] = True
    polygons = {}
    for i in np.flatnonzero(~work):
        # Preserve the original clipping start vertex, then restore outward winding.
        poly = clip(mesh.triangles[i][[0, 2, 1]], np.array([0., 1., 0.]), FLOOR_CLEARANCE_M, upper=False)
        if len(poly): poly = poly[np.r_[0, np.arange(len(poly)-1, 0, -1)]]
        if area(poly) > mesh.area*1e-16:
            polygons[int(i)] = poly
    allowed = np.zeros(len(mesh.faces), bool)
    allowed[list(polygons)] = True
    return allowed, polygons


def surface_groups(polygons,mesh=None):
    """Connected surface charts with a common dominant normal direction.

    Opposite sides and disconnected layers cannot share a chart. The six normal
    bins only delimit charts: allocation uses actual area, never projected area.
    """
    ids=sorted(polygons);parent={f:f for f in ids};groups={}
    for f in ids:
        p=polygons[f]
        normal=np.cross(p-p[0],np.roll(p,-1,axis=0)-p[0]).sum(axis=0)
        axis=int(np.array([0,2,1])[np.abs(normal)[[0,2,1]].argmax()])
        groups[f]=2*axis+int(normal[axis]<0)
    def root(f):
        while parent[f]!=f:
            parent[f]=parent[parent[f]];f=parent[f]
        return f
    def join(a,b):
        if a not in parent or b not in parent or groups[a]!=groups[b]:return
        a,b=root(a),root(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    if mesh is not None:
        for (a,b),edge in zip(mesh.face_adjacency,mesh.face_adjacency_edges):
            if mesh.vertices[edge,1].max()>FLOOR_CLEARANCE_M:
                join(int(a),int(b))
    else:
        # Geometry-only entry point for analytic layered-sheet checks.
        edges={}
        for f,p in polygons.items():
            for a,b in zip(p,np.roll(p,-1,axis=0)):
                key=tuple(sorted((tuple(a),tuple(b))))
                for other in edges.get(key,[]):join(f,other)
                edges.setdefault(key,[]).append(f)
    components={}
    for f in ids:components.setdefault(root(f),[]).append(f)
    return list(components.values())


def area_quotas(weights,count):
    """Systematic area allocation: every chart receives floor or ceil(N*A/Aall)."""
    weights=np.asarray(weights,float)
    if count<1 or not len(weights) or np.any(weights<=0):raise ValueError('Positive count and chart areas required')
    cumulative=np.cumsum(weights)/weights.sum()*count
    ends=np.floor(cumulative+.5+1e-12).astype(int);ends[-1]=count
    return np.diff(np.r_[0,ends])


def triangle_cut_fraction(ordered,bound):
    """Exact fraction of a triangle on the lower side of an axis-aligned plane."""
    a,b,c=ordered.T
    result=np.zeros(len(a));result[bound>=c]=1.
    rising=(bound>a)&(bound<b)
    result[rising]=(bound-a[rising])**2/((b[rising]-a[rising])*(c[rising]-a[rising]))
    falling=(bound>=b)&(bound<c)
    result[falling]=1-(c[falling]-bound)**2/((c[falling]-a[falling])*(c[falling]-b[falling]))
    return result


def split_triangles(triangles,sources,axis,bound):
    values=triangles[:,:,axis];left=[];right=[];lf=[];rf=[]
    lower=values.max(axis=1)<=bound;upper=values.min(axis=1)>=bound
    left.extend(triangles[lower]);lf.extend(sources[lower])
    right.extend(triangles[upper&~lower]);rf.extend(sources[upper&~lower])
    direction=np.eye(3)[axis]
    for t,source in zip(triangles[~(lower|upper)],sources[~(lower|upper)]):
        for keep,target,faces in ((True,left,lf),(False,right,rf)):
            p=clip(t,direction,bound,upper=keep)
            for j in range(1,len(p)-1):
                piece=p[[0,j,j+1]]
                if areas(piece[None])[0]>0:target.append(piece);faces.append(source)
    return (np.asarray(left).reshape(-1,3,3),np.asarray(lf,int)),(np.asarray(right).reshape(-1,3,3),np.asarray(rf,int))


def equal_area_centers(triangles,sources,count):
    """Split a chart by actual area, then choose a point on each leaf surface."""
    triangles=np.asarray(triangles)
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]).sum(axis=0)
    tangent_axes=np.array([a for a in (0,2,1) if a!=int(np.array([0,2,1])[np.abs(normal)[[0,2,1]].argmax()])])
    points=[];faces=[];cell_areas=[]
    def divide(tri,source,n):
        weights=areas(tri);total=weights.sum()
        if n==1:
            centroid=np.average(tri.mean(axis=1),weights=weights,axis=0)
            scale=np.ptp(tri.reshape(-1,3),axis=0).max()
            candidates=centroid+scale*closest_point((tri-centroid)/scale,np.zeros((len(tri),3)))
            selected=int(np.argmin(np.linalg.norm(candidates-centroid,axis=1)))
            # Stay strictly within the chosen triangle for later contact fitting.
            point=(1-1e-5)*candidates[selected]+1e-5*tri[selected].mean(axis=0)
            points.append(point);faces.append(int(source[selected]));cell_areas.append(float(total))
            return
        extent=np.ptp(tri.reshape(-1,3),axis=0);axis=int(tangent_axes[extent[tangent_axes].argmax()])
        ordered=np.sort(tri[:,:,axis],axis=1)
        low=float(ordered[:,0].min());high=float(ordered[:,2].max());left_n=n//2
        target=total*left_n/n
        for _ in range(50):
            bound=(low+high)/2
            value=float(weights@triangle_cut_fraction(ordered,bound))
            if abs(value-target)<=total*1e-11:break
            if value<target:low=bound
            else:high=bound
        left,right=split_triangles(tri,source,axis,bound)
        divide(*left,left_n);divide(*right,n-left_n)
    divide(np.asarray(triangles),np.asarray(sources,int),count)
    return points,faces,cell_areas


def surface_centers(polygons,count=CENTER_COUNT,mesh=None,with_report=False):
    """Allocate by true chart area, then put one point in each equal-area cell."""
    # Sampling uses fixed corner order independently of the outward winding.
    polygons={f:p[np.r_[0,np.arange(len(p)-1,0,-1)]] for f,p in polygons.items()}
    groups=surface_groups(polygons,mesh)
    weights=np.array([sum(area(polygons[f]) for f in group) for group in groups])
    quotas=area_quotas(weights,count)
    points=[];faces=[];records=[];point_groups=[];all_cells=[]
    for index,(group,weight,n) in enumerate(zip(groups,weights,quotas)):
        cell_areas=[]
        if n:
            triangles=[];sources=[]
            for f in group:
                p=polygons[f]
                for j in range(1,len(p)-1):
                    piece=p[[0,j,j+1]]
                    if areas(piece[None])[0]>0:triangles.append(piece);sources.append(f)
            p,f,cell_areas=equal_area_centers(triangles,sources,int(n))
            points.extend(p);faces.extend(f);point_groups.extend([index]*n);all_cells.extend(cell_areas)
        records.append(dict(index=index,source_faces=group,area_m2=float(weight),
                            area_quota=float(count*weight/weights.sum()),point_count=int(n),
                            max_cell_relative_area_error=float(max((abs(a/(weight/n)-1) for a in cell_areas),default=0.))))
    report=dict(method='true_area_chart_quotas_and_equal_area_cells',count=count,
                eligible_area_m2=float(weights.sum()),chart_count=len(groups),
                unsampled_chart_area_fraction=float(weights[quotas==0].sum()/weights.sum()),
                max_chart_quota_error=float(np.abs(quotas-count*weights/weights.sum()).max()),
                point_charts=point_groups,cell_areas_m2=all_cells,charts=records,
                allocation_rule='Each connected normal chart receives floor or ceil(N*A_chart/A_eligible). Each allocated chart is split into equal-area cells; one on-surface point per cell. No visibility or cross-layer distance weighting.')
    result=(np.asarray(points),np.asarray(faces,int))
    return (*result,report) if with_report else result
