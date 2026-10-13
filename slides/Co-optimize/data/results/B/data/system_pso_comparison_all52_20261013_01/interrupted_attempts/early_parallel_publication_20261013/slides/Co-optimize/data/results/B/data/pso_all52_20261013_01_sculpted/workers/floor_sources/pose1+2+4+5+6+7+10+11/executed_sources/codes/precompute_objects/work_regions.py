"""Connected visible work patches, independent of slide rendering."""
import hashlib
import heapq
import json
from pathlib import Path
import zlib
import numpy as np
import trimesh
ROOT = Path(__file__).resolve().parents[2]
WORK_AREA_FRACTION = (.06, .10)
QUANTUM = .005
COARSE = 2000
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def refine(mesh, rounds=6):
    """Split a COARSE mesh uniformly until no face is bigger than the quantum.

    PIPELINE s6's standing trap -- "subdivide before patching" -- with its own
    correction beside it: what the danger scales with is FACE SIZE AGAINST THE
    PATCH, not face count against the part.  A4 is 92 faces over a 150 mm frame
    and its largest face is 13.9 % of the surface, so a patch grown there is one
    or two triangles and its boundary can only run along edges; A1-f is 25k faces
    whose largest is 4.5 % and needs nothing.

    UNIFORM rounds, and only on a mesh coarse enough to afford them.  Splitting
    just the faces over the quantum leaves T-JUNCTIONS -- the neighbours keep
    their own edges, the shared edge is no longer shared, and `face_adjacency`
    quietly loses it.  That is not a cosmetic loss here: the region is grown by
    Dijkstra ON that graph, and A1-f's patches stalled at 4-7 % of the surface,
    under the band, because the disc could not cross the seam.  Uniform 4-splits
    keep every adjacency; the cost is why `COARSE` gates them.

    Subdivision moves no surface -- every new vertex is a convex combination of
    old ones -- so the tip's own geometry is untouched, and the asserts say so.
    """
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    got = 0
    if len(f) <= COARSE and (mesh.area_faces/mesh.area).max() > QUANTUM:
        for got in range(1, rounds + 1):
            if len(f)*4 > 5000:
                got -= 1
                break
            v, f = trimesh.remesh.subdivide(v, np.asarray(f))
            f = np.asarray(f)
            m = trimesh.Trimesh(v, f, process=False)
            if (m.area_faces / m.area).max() <= QUANTUM:
                break
    out = trimesh.Trimesh(v, f, process=False)
    assert abs(out.area - mesh.area) <= 1e-9 * mesh.area, "splitting moved the surface"
    assert abs(out.volume - mesh.volume) <= 1e-9 * abs(mesh.volume), \
        "splitting moved the solid"
    assert len(out.face_adjacency) == len(mesh.face_adjacency) * 4 ** got, \
        "splitting lost a face adjacency, which is the graph the region grows on"
    return out, got

def grow(mesh, seed, target, cap, blocked, adj, w):
    """region.json's `growth_rule`: a geodesic disc, in area, off the floor.

    Dijkstra on the face-adjacency graph weighted by the distance between face
    centres, which is a disc in surface distance and connected by construction.
    The floor band is not entered.  A face that would carry the patch past `cap`
    -- the top of the band -- is never added, though the disc still grows past
    it, so the stopping rule can never turn into a hole in the middle.
    """
    area = mesh.area_faces
    seen = np.zeros(len(area), bool)
    take = np.zeros(len(area), bool)
    got = 0.0
    heap = [(0.0, int(seed))]
    while heap and got < target:
        d, f = heapq.heappop(heap)
        if seen[f] or blocked[f]:
            continue
        seen[f] = True
        if got + area[f] <= cap:
            take[f] = True
            got += area[f]
        for g, e in zip(adj[f], w[f]):
            if not seen[g] and not blocked[g]:
                heapq.heappush(heap, (d + e, int(g)))
    return take, got

def components(mesh, take):
    """How many connected patches the drawn region is, over face adjacency."""
    idx = np.flatnonzero(take)
    if not len(idx):
        return 0
    pos = -np.ones(len(mesh.faces), int)
    pos[idx] = np.arange(len(idx))
    pair = mesh.face_adjacency
    keep = take[pair[:, 0]] & take[pair[:, 1]]
    g = trimesh.graph.connected_components(pos[pair[keep]], nodes=np.arange(len(idx)))
    return len(g)

class WorkRegions:
    def __init__(self,raw):
        self.mesh,self.rounds=refine(raw)
        # Large source triangles may individually exceed the new work band,
        # even when the historical refinement's face-count budget is reached.
        while self.mesh.area_faces.max()/self.mesh.area>WORK_AREA_FRACTION[0]:
            if len(self.mesh.faces)*4 > 5000:
                raise ValueError('Work mesh exceeds 5000 faces; use lowpoly.py to split large faces locally')
            self.mesh=self.mesh.subdivide()
            self.rounds+=1
        m=self.mesh
        self.visible=~m.ray.intersects_any(m.triangles_center+1e-5*m.face_normals,m.face_normals)
        self.adj=[[] for _ in m.faces];self.weights=[[] for _ in m.faces]
        pairs=m.face_adjacency
        distance=np.linalg.norm(m.triangles_center[pairs[:,0]]-m.triangles_center[pairs[:,1]],axis=1)
        for (i,j),d in zip(pairs,distance):
            self.adj[i].append(j);self.adj[j].append(i)
            self.weights[i].append(d);self.weights[j].append(d)

    def feasible(self,T):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        m=self.mesh
        world_z=m.vertices@T[2,:3]+T[2,3]
        eligible=self.visible & ((m.face_normals@T[2,:3])>.35)
        eligible &= world_z[m.faces].min(axis=1)>.0015
        pairs=m.face_adjacency
        pairs=pairs[eligible[pairs].all(axis=1)]
        graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(m.faces),len(m.faces)))
        _,labels=connected_components(graph,directed=False)
        sums=np.bincount(labels,weights=m.area_faces*eligible)
        return float(sums.max())>=WORK_AREA_FRACTION[0]*m.area

    def choose(self,T,key):
        m=self.mesh
        world=trimesh.transform_points(m.vertices,T)
        eligible=self.visible & ((m.face_normals@T[:3,:3].T)[:,2]>.35)
        eligible &= world[m.faces,2].min(axis=1)>.0015
        ids=np.flatnonzero(eligible)
        if not len(ids):raise ValueError(f'{key}: no work area candidates')
        rng=np.random.default_rng(20260918^zlib.crc32(key.encode()))
        seeds=rng.choice(ids,min(80,len(ids)),replace=False,p=m.area_faces[ids]/m.area_faces[ids].sum())
        for seed in seeds:
            target=rng.uniform(*WORK_AREA_FRACTION)*m.area
            mask,area=grow(m,int(seed),target,WORK_AREA_FRACTION[1]*m.area,~eligible,self.adj,self.weights)
            fraction=float(area/m.area)
            if WORK_AREA_FRACTION[0]<=fraction<=WORK_AREA_FRACTION[1] and components(m,mask)==1:
                return mask,dict(work_area_fraction=fraction,work_seed=int(seed),
                    work_components=1,work_normal_rays_clear=True,
                    minimum_work_vertex_z_m=float(world[m.faces[mask],2].min()),
                    minimum_work_outward_normal_z=float((m.face_normals@T[:3,:3].T)[mask,2].min()))
        raise ValueError(f'{key}: no connected 6-10% eligible working patch')

