"""Surface paths: one complete patch outline, then three broad interior strokes.

All paths use the accepted mesh, with selected-face witnesses. Connectors follow
shared triangle edges; they never bridge concavities through the object.
"""
import heapq
import numpy as np
import trimesh
from shapely.geometry import LineString,Polygon
from shapely.ops import unary_union


def resample(points,faces,spacing=.001):
    """Subdivide surface segments, retaining their selected triangle witness."""
    points=np.asarray(points);out=[points[0]];ids=[faces[0]]
    for i in range(1,len(points)):
        n=max(1,int(np.ceil(np.linalg.norm(points[i]-points[i-1])/spacing)))
        for u in np.linspace(0,1,n+1)[1:]:
            out.append((1-u)*points[i-1]+u*points[i]);ids.append(faces[i])
    return np.asarray(out),np.asarray(ids,dtype=int)


class SurfacePath:
    def __init__(self,mesh,ids,axis):
        self.mesh=mesh;self.ids=np.asarray(ids,dtype=int);self.allowed=set(self.ids.tolist())
        self.axis=np.asarray(axis);self.axis/=np.linalg.norm(self.axis)
        a=np.array([0.,1.,0.])
        if abs(a@self.axis)>.95:a=np.array([1.,0.,0.])
        a-=self.axis*(a@self.axis);a/=np.linalg.norm(a)
        self.basis=np.column_stack([a,np.cross(self.axis,a)])
        uv=mesh.vertices@self.basis
        center=uv[np.unique(mesh.faces[self.ids])].mean(0)
        _,_,v=np.linalg.svd(uv[np.unique(mesh.faces[self.ids])]-center,full_matrices=False)
        self.basis=self.basis@v.T;self.uv=mesh.vertices@self.basis
        self.graph={int(f):[] for f in self.ids}
        owned={}
        for f in self.ids:
            tri=mesh.faces[f]
            for x,y in zip(tri,np.roll(tri,-1)):
                key=tuple(sorted((int(x),int(y))));owned.setdefault(key,[]).append((int(f),int(x),int(y)))
        boundary=[]
        for edge,owners in owned.items():
            if len(owners)==1:boundary.append(owners[0])
            elif len(owners)==2:
                x,y=owners[0][0],owners[1][0];mid=mesh.vertices[list(edge)].mean(0)
                cost=np.linalg.norm(mesh.triangles_center[x]-mid)+np.linalg.norm(mesh.triangles_center[y]-mid)
                self.graph[x].append((y,cost,mid));self.graph[y].append((x,cost,mid))
            else:raise ValueError('Non-manifold selected work patch')
        outgoing={x:(y,f) for f,x,y in boundary}
        if len(outgoing)!=len(boundary):raise ValueError('Branching patch boundary')
        loops=[]
        while outgoing:
            start=next(iter(outgoing));current=start;vertices=[];faces=[]
            while True:
                vertices.append(current);nxt,f=outgoing.pop(current);faces.append(f);current=nxt
                if current==start:break
            vertices.append(start);q=self.uv[vertices]
            area=abs(np.sum(q[:-1,0]*q[1:,1]-q[1:,0]*q[:-1,1]))/2
            loops.append((area,vertices,faces))
        _,vertices,edge_faces=max(loops,key=lambda x:x[0])
        # A 2.5% triangle-local inset avoids ambiguous ray/vertex ties. This
        # moves the trace by tenths of a millimetre, not a region-sized crop.
        points=[];witness=[];epsilon=.025
        for i,face in enumerate(edge_faces):
            centroid=mesh.triangles_center[face]
            begin=(1-epsilon)*mesh.vertices[vertices[i]]+epsilon*centroid
            end=(1-epsilon)*mesh.vertices[vertices[i+1]]+epsilon*centroid
            if points:
                connector,cf=self.connect(points[-1],witness[-1],begin,face,corner=vertices[i])
                points.extend(connector[1:]);witness.extend(cf[1:])
            else:points.append(begin);witness.append(face)
            points.append(end);witness.append(face)
        connector,cf=self.connect(points[-1],witness[-1],points[0],witness[0],corner=vertices[0])
        points.extend(connector[1:]);witness.extend(cf[1:])
        self.outline,self.outline_faces=resample(points,witness)
        self.outer_vertices=np.asarray(vertices,dtype=int)
        self.boundary_loop_count=len(loops)
        polygon=unary_union([Polygon(self.uv[tri]) for tri in mesh.faces[self.ids]])
        bounds=polygon.bounds;interior=[self.outline[-1]];witness=[int(self.outline_faces[-1])]
        stroke_lengths=[]
        for i,fraction in enumerate((.28,.5,.72)):
            y=bounds[1]+fraction*(bounds[3]-bounds[1])
            line=LineString([(bounds[0]-.01,y),(bounds[2]+.01,y)])
            intersection=polygon.intersection(line)
            segments=[intersection] if intersection.geom_type=='LineString' else [g for g in getattr(intersection,'geoms',[]) if g.geom_type=='LineString']
            segment=max(segments,key=lambda g:g.length)
            # Stay a fraction of a millimetre inside the perimeter at each end.
            margin=min(.0004,segment.length*.02)
            points,faces=self.line_stroke(segment,margin,reverse=bool(i%2))
            connector,connector_faces=self.connect(interior[-1],witness[-1],points[0],int(faces[0]))
            interior.extend(connector[1:]);witness.extend(connector_faces[1:])
            interior.extend(points[1:]);witness.extend(faces[1:])
            stroke_lengths.append(float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()))
        self.interior,self.interior_faces=resample(interior,witness)
        self.outline_length=float(np.linalg.norm(np.diff(self.outline,axis=0),axis=1).sum())
        self.interior_length=float(np.linalg.norm(np.diff(self.interior,axis=0),axis=1).sum())
        self.stroke_lengths=stroke_lengths
        self.outline_s=self.arc(self.outline);self.interior_s=self.arc(self.interior)
        self.validate()

    @staticmethod
    def arc(points):
        s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        return s/s[-1]

    def line_stroke(self,segment,margin,reverse=False):
        begin=np.array(segment.interpolate(margin).coords[0])
        end=np.array(segment.interpolate(segment.length-margin).coords[0])
        if reverse:begin,end=end,begin
        line=LineString([begin,end]);direction=end-begin;direction/=np.linalg.norm(direction)
        pieces=[]
        for face in self.ids:
            tri_uv=self.uv[self.mesh.faces[face]]
            intersection=Polygon(tri_uv).intersection(line)
            if intersection.geom_type!='LineString' or intersection.length<1e-10:continue
            q=np.array(intersection.coords)[[0,-1]];s=(q-begin)@direction
            if s[0]>s[1]:q=q[::-1];s=s[::-1]
            A=(tri_uv[1:]-tri_uv[0]).T
            bary=np.linalg.solve(A,(q-tri_uv[0]).T).T
            tri=self.mesh.triangles[face]
            xyz=tri[0]+bary@(tri[1:]-tri[0])
            pieces.append((s[0],s[1],xyz,int(face)))
        pieces.sort(key=lambda x:x[0]);points=[];faces=[]
        for _,_,xyz,face in pieces:
            if points:
                if np.linalg.norm(points[-1]-xyz[0])>2e-5:
                    connector,cf=self.connect(points[-1],faces[-1],xyz[0],face)
                    points.extend(connector[1:]);faces.extend(cf[1:])
                else:
                    points.append(xyz[0]);faces.append(face)
            else:points.append(xyz[0]);faces.append(face)
            points.append(xyz[1]);faces.append(face)
        return np.asarray(points),np.asarray(faces)

    def connect(self,start,start_face,end,end_face,corner=None):
        if start_face==end_face:return [start,end],[start_face,end_face]
        queue=[(0.,start_face)];distance={start_face:0.};previous={}
        while queue:
            dist,f=heapq.heappop(queue)
            if dist!=distance[f]:continue
            if f==end_face:break
            for other,cost,mid in self.graph[f]:
                if corner is not None and corner not in self.mesh.faces[other]:continue
                d=dist+cost
                if d<distance.get(other,np.inf):
                    distance[other]=d;previous[other]=(f,mid);heapq.heappush(queue,(d,other))
        if end_face not in previous:raise ValueError('Disconnected working surface')
        steps=[];f=end_face
        while f!=start_face:
            prev,mid=previous[f];steps.append((prev,f,mid));f=prev
        points=[start];faces=[start_face]
        # Direct paths through shared midpoints remain inside each convex triangle.
        for prev,f,mid in reversed(steps):
            if corner is not None:mid=.975*self.mesh.vertices[corner]+.025*mid
            points.append(mid);faces.append(prev)
        points.append(end);faces.append(end_face)
        return points,faces

    def validate(self):
        for points,faces in [(self.outline,self.outline_faces),(self.interior,self.interior_faces)]:
            close=trimesh.triangles.closest_point(self.mesh.triangles[faces],points)
            if np.linalg.norm(close-points,axis=1).max()>2e-5:raise ValueError(f'Path leaves selected triangle: point {np.argmax(np.linalg.norm(close-points,axis=1))} / {len(points)}, face {faces[np.argmax(np.linalg.norm(close-points,axis=1))]}, distance {np.linalg.norm(close-points,axis=1).max()}')
            origins=points-self.axis*.18
            hit,ray,_=self.mesh.ray.intersects_location(origins,np.repeat(self.axis[None],len(points),axis=0),multiple_hits=True)
            depth=(hit-origins[ray])@self.axis;first=np.full(len(points),np.inf)
            np.minimum.at(first,ray,np.where(depth>1e-7,depth,np.inf))
            if np.max(np.abs(first-.18))>2e-5:raise ValueError(f'Working path is not first-hit visible: {np.flatnonzero(abs(first-.18)>2e-5)[:20]} depths {first[abs(first-.18)>2e-5][:20]}')

    @staticmethod
    def interpolate(points,faces,s,u):
        j=min(int(np.searchsorted(s,u,side='right')),len(s)-1);i=max(j-1,0)
        v=(u-s[i])/max(s[j]-s[i],1e-15)
        return points[i]*(1-v)+points[j]*v,int(faces[j])

    def at(self,u):
        if u<=.57:
            return (*self.interpolate(self.outline,self.outline_faces,self.outline_s,float(np.clip((u-.10)/.47,0,1))),1 if u>=.10 else 0)
        return (*self.interpolate(self.interior,self.interior_faces,self.interior_s,float(np.clip((u-.57)/.33,0,1))),2 if u<=.90 else 3)

    def info(self):
        return dict(boundary_loop_count=self.boundary_loop_count,outline_closed=True,
            complete_outer_boundary=True,triangle_local_boundary_inset_fraction=.025,outline_length_m=self.outline_length,
            interior_length_m=self.interior_length,interior_strokes=3,stroke_lengths_m=self.stroke_lengths,
            path_first_hit_verified=True,path_selected_triangles_verified=True)
