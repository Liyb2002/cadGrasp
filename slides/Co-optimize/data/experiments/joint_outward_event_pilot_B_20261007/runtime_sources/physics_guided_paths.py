"""Material-path acquisition guidance on the immutable seed surface.

A graph is an optimization surrogate, never a connectivity certificate.
Edges follow seed-adjacent inset face centers and are kept only when sampled
points lie in the seed. True connectivity is accepted on the exact solid.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import _bootstrap
from co_common import np,trimesh,S,material_volume
from physics_guided_geometry import SweepDistanceModel,distance_cost
from scipy.sparse import coo_matrix,csr_matrix
from scipy.sparse.csgraph import dijkstra,connected_components
from scipy.spatial import cKDTree


class MaterialPathModel:
    def __init__(self,seed_mesh,clearance,contact_points,contact_normals,length,extent):
        # Node positions are inside the original material, not new rods or
        # arbitrary straight bridges through empty space.
        self.mesh=seed_mesh
        face_points=seed_mesh.triangles_center-.000025*seed_mesh.face_normals
        adjacent=seed_mesh.face_adjacency
        edge_points=seed_mesh.vertices[seed_mesh.face_adjacency_edges].mean(axis=1)
        bisectors=seed_mesh.face_normals[adjacent].sum(axis=1)
        norms=np.linalg.norm(bisectors,axis=1)
        bisectors=np.divide(bisectors,norms[:,None],out=np.zeros_like(bisectors),where=norms[:,None]>1e-12)
        edge_points-=.000025*bisectors
        self.points=np.vstack([face_points,edge_points])
        inside=trimesh.proximity.signed_distance(seed_mesh,self.points)>1e-9
        middle=len(face_points)+np.arange(len(adjacent))
        edges=np.vstack([np.c_[adjacent[:,0],middle],np.c_[middle,adjacent[:,1]]])
        edges=edges[inside[edges].all(axis=1)]
        for fraction in [.25,.5,.75]:
            samples=(1-fraction)*self.points[edges[:,0]]+fraction*self.points[edges[:,1]]
            edges=edges[trimesh.proximity.signed_distance(seed_mesh,samples)>1e-9]
        self.edges=edges
        self.lengths=np.linalg.norm(self.points[edges[:,0]]-self.points[edges[:,1]],axis=1)/extent
        adjacency=coo_matrix((np.ones(2*len(edges)),
            (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(self.points),)*2).tocsr()
        _,self.labels=connected_components(adjacency,directed=False)
        # Locate each hypothetical bearing ray in existing material.
        probes=contact_points+.0005*contact_normals
        valid_ids=np.flatnonzero(inside)
        if not len(valid_ids):raise RuntimeError('no interior material graph nodes')
        nearest=cKDTree(self.points[valid_ids]).query(probes)[1]
        self.contact_nodes=valid_ids[nearest]
        self.sweep= SweepDistanceModel(clearance,self.points,np.zeros_like(self.points),length,extent,depth=0.)
        self.root=None;self.path_matrix=None;self.fixed_cost=None

    def linearize(self,directions,frames,anchor_mesh):
        # Anchor inside the current bearing component, fixed for this local
        # solve. Never connect to an initially disconnected seed component.
        scores=trimesh.proximity.signed_distance(anchor_mesh,self.points)
        self.root=int(np.argmax(scores))
        if scores[self.root]<=0:raise RuntimeError('no material graph anchor in component')
        values,jac=self.sweep.linearize(directions,frames)
        node_cost,_=distance_cost(values,jac,np.zeros((len(directions),2)))
        weights=self.lengths*(.5*(node_cost[self.edges[:,0]]+node_cost[self.edges[:,1]])+1e-4)
        graph=coo_matrix((np.r_[weights,weights],
            (np.r_[self.edges[:,0],self.edges[:,1]],np.r_[self.edges[:,1],self.edges[:,0]])),shape=(len(self.points),)*2).tocsr()
        distances,predecessors=dijkstra(graph,directed=False,indices=self.root,return_predecessors=True)
        edge_lookup={tuple(sorted(edge)):i for i,edge in enumerate(self.edges)}
        rows,columns,data=[],[],[]
        fixed=np.zeros(len(self.contact_nodes))
        for contact,node in enumerate(self.contact_nodes):
            if not np.isfinite(distances[node]):
                fixed[contact]=2.;continue
            current=int(node)
            while current!=self.root:
                parent=int(predecessors[current])
                if parent<0:raise RuntimeError('unresolved material path')
                edge=edge_lookup[tuple(sorted((current,parent)))]
                weight=self.lengths[edge]*.5
                rows.extend([contact,contact]);columns.extend([current,parent]);data.extend([weight,weight])
                current=parent
        self.path_matrix=csr_matrix((data,(rows,columns)),shape=(len(self.contact_nodes),len(self.points)))
        self.fixed_cost=fixed
        return values,jac

    def cost(self,values,jac,coordinates):
        node_cost,node_gradient=distance_cost(values,jac,coordinates)
        cost=self.path_matrix@node_cost+self.fixed_cost
        gradient=np.stack([self.path_matrix@node_gradient[:,i,:] for i in range(coordinates.shape[0])],axis=1)
        return cost,gradient
