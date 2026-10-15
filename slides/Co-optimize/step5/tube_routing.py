"""Full-width routes with explicit functional terminal caps."""
import numpy as np
import manifold3d as md
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric
from continuous_legs import RouteGuide,capsule
from deterministic_base import volume,SCALE


class TerminalRouteGuide(RouteGuide):
    def __init__(self,*args,minimum_radius=.0015,clearance_margin=.5,**kwargs):
        super().__init__(*args,**kwargs)
        self.domain=args[0]
        self.minimum_radius=minimum_radius
        self.valid &= self.clearance>=minimum_radius+clearance_margin*self.pitch
        self.roots &= self.valid
    def route(self,target,limit=500000):
        target=np.asarray(target)
        center=np.rint((target-self.origin)/self.pitch).astype(int)
        reach=8
        offsets=np.array([(x,y,z)for x in range(-reach,reach+1)
            for y in range(-reach,reach+1)for z in range(-reach,reach+1)])
        ids=center+offsets
        ids=ids[np.all((ids>=0)&(ids<self.shape),axis=1)]
        ids=ids[self.valid[tuple(ids.T)]]
        if not len(ids):raise ValueError('No full-width seed beyond the functional terminal')
        labels,_=ndimage.label(self.valid,structure=np.ones((3,3,3),bool))
        root_labels=np.unique(labels[self.roots]);root_labels=root_labels[root_labels>0]
        ids=ids[np.isin(labels[tuple(ids.T)],root_labels)]
        if not len(ids):raise ValueError('No terminal seed connects to the main body in this proposal grid')
        xyz=self.origin+ids*self.pitch
        distance=np.linalg.norm(xyz-target,axis=1)
        seed=None;chosen=None
        for index in np.argsort(distance,kind='stable')[:96]:
            proposed=xyz[index]
            sphere=md.Manifold.sphere(self.minimum_radius/SCALE,24).translate(tuple(proposed/SCALE))
            if volume(sphere-self.domain)>1e-10:continue
            if not full_width_middle(np.array([target,proposed,proposed]),self.minimum_radius,self.domain):continue
            seed=proposed;chosen=index;break
        if seed is None:raise ValueError('No exact full-width terminal seed in the routed component')
        label=labels[tuple(ids[chosen])]
        self.roots &= labels==label
        roots=np.argwhere(self.roots)
        self.root_tree=cKDTree(self.origin+roots[::max(1,len(roots)//8000)]*self.pitch)
        print('FULL_WIDTH_ROUTE_SEED',float(distance[chosen]),flush=True)
        seed_index=np.rint((seed-self.origin)/self.pitch).astype(int)
        roots=np.argwhere(self.roots);ends=roots[::max(1,len(roots)//12000)]
        costs=np.where(self.valid,1.+(.0025/np.maximum(self.clearance,.0008))**2,np.inf)
        solver=MCP_Geometric(costs,fully_connected=True)
        cumulative,_=solver.find_costs([tuple(seed_index)],ends=[tuple(p)for p in ends],find_all_ends=False)
        values=cumulative[tuple(ends.T)];best=int(np.argmin(values))
        if not np.isfinite(values[best]):raise ValueError('No connected full-width proposal route')
        cells=np.asarray(solver.traceback(tuple(ends[best])))
        if np.linalg.norm(cells[0]-seed_index)>np.linalg.norm(cells[-1]-seed_index):cells=cells[::-1]
        xyz=self.origin+cells*self.pitch
        selected=[xyz[0]];k=0
        while k<len(xyz)-1:
            found=False
            for j in range(min(len(xyz)-1,k+32),k,-1):
                tube=capsule(xyz[k],xyz[j],self.minimum_radius)
                if volume(tube-self.domain)>1e-10:continue
                selected.append(xyz[j]);k=j;found=True;break
            if not found:raise ValueError('Grid path has no exact swept-width shortcut')
        path=np.asarray(selected)
        return np.vstack([target,path]),dict(functional_terminal_seed_distance_m=float(distance[chosen]),
            guide_pitch_m=self.pitch,compiled_grid_search=True,grid_path_vertices=len(cells),
            path_length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()))


def full_width_middle(path,radius,domain):
    for index,(a,b)in enumerate(zip(path,path[1:])):
        a=np.asarray(a);b=np.asarray(b);length=np.linalg.norm(b-a)
        if length<1e-9:continue
        direction=(b-a)/length
        # Actual object/floor ports terminate a solid tube on a functional
        # boundary. Its short end cap is clipped there, not certified as a
        # free-space full ball. Every remaining midsection is checked exactly.
        start=2*radius if index==0 else 0.
        stop=length-(2*radius if index==len(path)-2 else 0.)
        if stop<=start:continue
        tube=capsule(a+direction*start,a+direction*stop,radius)
        if volume(tube-domain)>1e-10:return False
    return True
