"""Thin connector proposals following the certified free-space boundary."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import trimesh


class SurfaceRoutes:
    def __init__(self,witness):
        self.mesh=witness
        self.centers=witness.triangles_center
        edges=witness.edges_unique
        self.points=np.vstack([self.centers,witness.vertices[edges].mean(1)])
        faces=np.repeat(np.arange(len(witness.faces)),3)
        edge_ids=witness.edges_unique_inverse.reshape(-1,3).ravel()+len(witness.faces)
        distance=np.linalg.norm(self.points[faces]-self.points[edge_ids],axis=1)
        self.base_i=np.r_[faces,edge_ids];self.base_j=np.r_[edge_ids,faces]
        self.base_w=np.r_[distance,distance]

    def paths(self,meshes):
        """Shortest boundary routes between actual disconnected components.

Face centers connect through shared edge midpoints. Every surface segment
lies in one real boundary triangle, including at concave corners. Virtual
component nodes connect actual material to nearby boundary triangles.
"""
        count=len(self.points);ii=[self.base_i];jj=[self.base_j];ww=[self.base_w]
        anchors={}
        for k,mesh in enumerate(meshes):
            ids=np.unique(np.linspace(0,len(mesh.vertices)-1,min(96,len(mesh.vertices))).astype(int))
            actual=mesh.vertices[ids]
            near,distance,faces=trimesh.proximity.closest_point(self.mesh,actual)
            # One closest anchor per face avoids duplicate sparse edges
            # (whose costs would otherwise be added by COO conversion).
            for f in np.unique(faces):
                candidates=np.flatnonzero(faces==f)
                cost=distance[candidates]+np.linalg.norm(near[candidates]-self.centers[f],axis=1)
                j=candidates[int(np.argmin(cost))]
                anchors[k,int(f)]=[actual[j],near[j],self.centers[f]]
                ii.append(np.array([count+k,f]));jj.append(np.array([f,count+k]));ww.append(np.full(2,float(cost.min())))
        graph=coo_matrix((np.concatenate(ww),(np.concatenate(ii),np.concatenate(jj))),
            shape=(count+len(meshes),count+len(meshes))).tocsr()
        costs,previous=dijkstra(graph,directed=False,indices=np.arange(count,count+len(meshes)),return_predecessors=True)
        pairs=sorted((costs[a,count+b],a,b) for a in range(len(meshes)) for b in range(a))
        for length,a,b in pairs:
            if not np.isfinite(length):continue
            node=count+b;chain=[node]
            while node!=count+a:
                node=int(previous[a,node])
                if node<0:raise ValueError('Incomplete certified-boundary route')
                chain.append(node)
            chain.reverse();path=[]
            for u,v in zip(chain,chain[1:]):
                if u>=count:path.extend(anchors[u-count,v])
                elif v>=count:path.extend(reversed(anchors[v-count,u]))
                else:path.extend([self.points[u],self.points[v]])
            yield float(length),a,b,np.asarray(path)
