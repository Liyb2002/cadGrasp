"""Task regions for the six tabletop poses.

Keep raw certified reach separate from task membership. Fill small pockets whose
whole boundary belongs to one task, and discard detached components below 1%
of target area except for each task's largest component. Larger disconnected
parts remain valid. The same cleanup applies to per-pose regions and the
non-overlapping split. Other gaps remain unassigned; no coverage completion or
corridor borrowing is required. Uncertified enclosed faces are recorded apart.
"""
from collections import deque
import heapq

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .partition import BIG,Potts

SIDES=('+X','+Y','+Z','-X','-Y','-Z')
VIEWS=np.vstack([np.eye(3),-np.eye(3)])


class Surface:
    """Face adjacency with shared-edge lengths."""
    def __init__(self,mesh):
        F=len(mesh.faces);pairs=mesh.face_adjacency;self.centers=mesh.triangles_center
        w=np.linalg.norm(np.diff(mesh.vertices[mesh.face_adjacency_edges],axis=1)[:,0],axis=1)
        self.nb=[[] for _ in range(F)];self.length={}
        for (a,b),l in zip(pairs,w):self.nb[a].append(b);self.nb[b].append(a);self.length[a,b]=self.length[b,a]=l
        self.graph=csr_matrix((np.ones(2*len(pairs)),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(F,F))

    def pieces(self,mask):
        sub=np.flatnonzero(mask)
        if len(sub)==0:return []
        n,label=connected_components(self.graph[sub][:,sub],directed=False)
        return [sub[label==k] for k in range(n)]

    def keeps_connected(self,label,f):
        """Would the area of f stay connected around f without it?"""
        k=label[f];start=[g for g in self.nb[f] if label[g]==k]
        if len(start)<=1:return True
        seen={start[0]};queue=deque([start[0]]);need=set(start[1:])
        while queue and need:
            u=queue.popleft()
            for v in self.nb[u]:
                if v!=f and label[v]==k and v not in seen:seen.add(v);need.discard(v);queue.append(v)
        return not need


def extra_direction(ids,area,covered,grid,grid_reach):
    """The candidate axis covering the most target area not yet covered (None if none adds any)."""
    gain=(grid_reach&~covered[:,None])[ids].T@area[ids];k=int(gain.argmax())
    return None if gain[k]<=0 else k


def connected_tasks(mesh,label):
    """Split final direction labels into tasks connected through shared mesh edges.

    Returns per-face task IDs (-1 outside) and (parent direction, faces) pairs.
    Within each direction, larger tasks come first; face IDs break area ties.
    """
    surface=Surface(mesh);task=np.full(len(label),-1);pieces=[]
    for j in np.unique(label[label>=0]):
        for faces in sorted(surface.pieces(label==j),key=lambda f:(-mesh.area_faces[f].sum(),int(f[0]))):
            task[faces]=len(pieces);pieces.append((int(j),faces))
    return task,pieces


def grow(surface,ids,label,reach,cost):
    """Unlabelled faces join an adjacent area that reaches them, the cheapest offer first."""
    target=np.zeros(len(label),bool);target[ids]=True;heap=[]
    for f in ids:
        if label[f]<0:
            for g in surface.nb[f]:
                k=label[g]
                if k>=0 and reach[f,k]:heapq.heappush(heap,(cost[f,k],int(f),int(k)))
    while heap:
        _,f,k=heapq.heappop(heap)
        if label[f]>=0:continue
        label[f]=k
        for g in surface.nb[f]:
            if target[g] and label[g]<0 and reach[g,k]:heapq.heappush(heap,(cost[g,k],int(g),k))
    return label


def connect(surface,ids,label,reach,cost):
    """Enclosed faces: borrow a corridor from a side that paints them; donors stay connected."""
    target=np.zeros(len(label),bool);target[ids]=True;nb=surface.nb;stuck=[]
    while True:
        left=[f for f in ids if label[f]<0 and f not in stuck]
        if not left:return label,stuck
        pocket=next(p for p in surface.pieces(label<0) if left[0] in set(p.tolist()));pocket=pocket[target[pocket]]
        done=False
        for k in sorted(np.flatnonzero(reach[pocket].any(0)),key=lambda k:cost[pocket,k][reach[pocket,k]].mean()):
            blocked=set()
            for _ in range(40):
                prev={};queue=deque(np.flatnonzero(label==k).tolist());seen=set(queue);goal=None
                while queue:
                    u=queue.popleft()
                    if label[u]<0 and reach[u,k]:goal=u;break
                    for v in nb[u]:
                        if v not in seen and v not in blocked and target[v] and reach[v,k]:seen.add(v);prev[v]=u;queue.append(v)
                if goal is None:break
                path=[];u=goal
                while label[u]!=k:path.append(u);u=prev[u]
                moved=[];failed=None
                for v in reversed(path):
                    if label[v]>=0 and not surface.keeps_connected(label,v):failed=v;break
                    moved.append((v,label[v]));label[v]=k
                if failed is None:done=True;label=grow(surface,ids,label,reach,cost);break
                for v,old in moved:label[v]=old
                blocked.add(failed)
            if done:break
        if not done:
            # No corridor keeps everyone connected: each face of the pocket goes to its most
            # head-on side that paints it, as separate pieces.
            c=np.where(reach[pocket],cost[pocket],np.inf);paintable=np.isfinite(c).any(1)
            label[pocket[paintable]]=c[paintable].argmin(1);stuck+=pocket[~paintable].tolist()
            label=grow(surface,ids,label,reach,cost)


def smooth(surface,ids,label,reach,cost,kappa_length):
    """Boundary faces move when boundary length plus incidence drops and both areas stay connected."""
    while True:
        moved=0
        for f in ids:
            j=label[f];shared={}
            for g in surface.nb[f]:
                k=label[g]
                if k>=0:shared[k]=shared.get(k,0.)+surface.length[f,g]
            best=None
            for k in shared:
                if k==j or not reach[f,k]:continue
                gain=shared[k]-shared.get(j,0.)+kappa_length[f]*(cost[f,j]-cost[f,k])
                if gain>1e-12 and (best is None or gain>best[0]):best=(gain,k)
            if best and surface.keeps_connected(label,f):label[f]=best[1];moved+=1
        if not moved:return label


def corridor(surface,label,reach,k,piece,target,area):
    """Join a stray piece of side k to k's main area along faces k paints, never cutting the lender.

    Only a corridor no larger than the piece itself counts: a groove's walls, not a long strip."""
    main=max(surface.pieces(label==k),key=len);inside=set(piece.tolist());blocked=set()
    for _ in range(40):
        prev={};queue=deque(main.tolist());seen=set(queue);goal=None
        while queue:
            u=queue.popleft()
            if u in inside:goal=u;break
            for v in surface.nb[u]:
                if v not in seen and v not in blocked and target[v] and reach[v,k]:seen.add(v);prev[v]=u;queue.append(v)
        if goal is None:return False
        path=[];u=prev.get(goal)
        while u is not None and label[u]!=k:path.append(u);u=prev.get(u)
        if area[path].sum()>area[piece].sum():return False
        moved=[];failed=None
        for v in path:
            if not surface.keeps_connected(label,v):failed=v;break
            moved.append((v,label[v]));label[v]=k
        if failed is None:return True
        for v,old in moved:label[v]=old
        blocked.add(failed)
    return False


def areas(mesh,ids,tilt,axes,beta,kappa,small_piece=.01,strict_reach=False):
    """A non-overlapping split of the target faces ids among the given directions.

    tilt: (faces, directions) certified needle tilt in radians (inf: that direction cannot paint the face).
    Returns the area (direction index) of each face (-1 outside or uncovered) and a summary."""
    F=len(mesh.faces);area=mesh.area_faces;surface=Surface(mesh)
    target=np.zeros(F,bool);target[ids]=True;reach=np.isfinite(tilt)&target[:,None]
    cost=np.arccos(np.clip(-mesh.face_normals@axes.T,-1,1))/(np.pi/2)          # incidence, 0 head-on
    covered=ids[reach[ids].any(1)];model=Potts(mesh,covered,beta,kappa)
    D=np.where(reach[covered],np.round(kappa*model.perimeter[:,None]*cost[covered]/1e-5),BIG).astype(np.int64)
    cut,_=model.solve(D,D.argmin(1));label=np.full(F,-1);label[covered]=cut
    perimeter=np.zeros(F);perimeter[covered]=model.perimeter
    label=smooth(surface,covered,label,reach,cost,kappa*perimeter)
    if strict_reach:
        if not np.all(reach[covered,label[covered]]):raise ValueError('Task assignment left certified reach')
        return label,dict(pieces_absorbed=0,pieces_joined_by_corridor=0,small_pieces_dropped=0,
                          enclosed_face_ids=[],small_piece_fraction=small_piece,strict_reach=True,
                          pieces=[len(surface.pieces(label==k)) for k in range(len(axes))])
    small=small_piece*area[ids].sum()
    filled=fill_enclosed(surface,ids,label,area,small)
    dropped=drop_small(surface,ids,label,area,small)
    filled+=fill_enclosed(surface,ids,label,area,small)
    return label,dict(pieces_absorbed=0,pieces_joined_by_corridor=0,small_pieces_dropped=dropped,
                      enclosed_face_ids=sorted(set(filled)),
                      small_piece_fraction=small_piece,pieces=[len(surface.pieces(label==k)) for k in range(len(axes))])


def fill_enclosed(surface,ids,label,area,small):
    """Fill a small surface pocket only when its entire boundary belongs to one task.

    A pocket may contain unreachable faces or faces assigned to other tasks.
    A boundary touching a different task or the edge of the target is not enclosed.
    This changes task membership only; it never changes reach certification.
    """
    target=np.zeros(len(label),bool);target[ids]=True;filled=[]
    while True:
        changed=False
        for k in np.unique(label[ids]):
            if k<0:continue
            for piece in surface.pieces(target&(label!=k)):
                if area[piece].sum()>=small:continue
                inside=set(piece.tolist())
                boundary={g for f in piece for g in surface.nb[f] if g not in inside}
                if boundary and all(target[g] and label[g]==k for g in boundary):
                    label[piece]=k;filled+=piece.tolist();changed=True
        if not changed:return filled


def task_regions(mesh,ids,certified,small_fraction=.01):
    """Fill enclosed pockets and discard small detached islands; keep raw reach unchanged."""
    surface=Surface(mesh);regions=np.zeros_like(certified,dtype=bool)
    small=small_fraction*mesh.area_faces[ids].sum()
    for j in range(certified.shape[1]):
        label=np.full(len(mesh.faces),-1);label[ids]=np.where(certified[ids,j],j,-1)
        fill_enclosed(surface,ids,label,mesh.area_faces,small)
        drop_small(surface,ids,label,mesh.area_faces,small)
        fill_enclosed(surface,ids,label,mesh.area_faces,small)
        regions[:,j]=label==j
    return regions,regions&~certified


def drop_small(surface,ids,label,area,small):
    """Discard detached pieces below `small`, keeping each task's largest component.

    Removed faces stay unassigned. Enclosed pockets are handled separately.
    """
    dropped=0
    for k in np.unique(label[ids]):
        if k<0:continue
        for piece in sorted(surface.pieces(label==k),key=lambda p:-area[p].sum())[1:]:
            if area[piece].sum()>=small:continue
            label[piece]=-1;dropped+=1
    return dropped


def nearest_area(surface,ids,label):
    """Target faces no direction paints go to the nearest area, growing out from the painted ones,
    so nothing is left without a task (labels are set in place)."""
    target=np.zeros(len(label),bool);target[ids]=True;left=target&(label<0)
    queue=deque(f for f in np.flatnonzero(target&(label>=0)) if any(left[g] for g in surface.nb[f]))
    while queue:
        u=queue.popleft()
        for v in surface.nb[u]:
            if left[v]:label[v]=label[u];left[v]=False;queue.append(v)
    # A user-drawn target can have a separate, entirely unreachable island.
    # No target-adjacency path leads to it; use the closest assigned face in 3D.
    if left.any():
        seeds=np.flatnonzero(target&(label>=0));missing=np.flatnonzero(left)
        if len(seeds):
            _,near=cKDTree(surface.centers[seeds]).query(surface.centers[missing])
            label[missing]=label[seeds[near]]
