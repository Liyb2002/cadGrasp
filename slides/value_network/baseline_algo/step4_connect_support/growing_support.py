"""Grow a sparse support from frozen contact roots inside the compact space.

The accepted compact solid is a navigation/clipping domain, never initial
material. Emit only terminal lofts and a shared short connecting tree.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

import manifold3d as md
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import MultiPoint
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step2_local_support import geometry as G
from step3_scheculer import contacts as I
from step4_connect_support import deterministic_space as D, boxed_support as F
from step4_connect_support import build_coupled_saddle as S, process_access as A
from step4_connect_support.run_boxed_batch import protected_hashes


def components(solid):
    return [p for p in solid.decompose() if p.volume()*S.SCALE**3 > 1e-16]


def prune_ground(xy, demands):
    """Delete redundant hull corners while retaining every original demand."""
    points = xy[ConvexHull(xy).vertices].copy()
    required = MultiPoint(demands).convex_hull
    while len(points) > 3:
        candidates = []
        for index in range(len(points)):
            trial = np.delete(points, index, axis=0)
            hull = MultiPoint(trial).convex_hull
            if hull.geom_type == 'Polygon' and required.difference(hull.buffer(1e-10)).area <= 1e-12:
                candidates.append((hull.area, index, trial))
        if not candidates:
            break
        points = min(candidates, key=lambda r:r[:2])[2]
    return points


class Grow:
    def __init__(self, group):
        self.group = group
        self.source = group/'step4/data/boxed_support'
        self.reference_report = I.check_report(self.source/'report.json')
        assert self.reference_report['passed']
        self.reference = trimesh.load(self.source/'shape.obj', force='mesh', process=False)
        self.mask = S.solid(self.reference)
        self.out = group/'step4/data/growing_support'
        self.out.mkdir(parents=True, exist_ok=True)
        self.search = D.Search(group)
        self.case = self.search.case
        self.case.output = self.out
        p = self.reference_report['placement']
        self.bases, self.offsets, self.directions = map(np.asarray, (p['bases'], p['offsets'], p['directions']))
        self.guard = A.Guard(self.case, p)
        self.bead = trimesh.creation.icosphere(subdivisions=1, radius=.004).vertices
        self.terminals, self.feet, self.paths = [], [], []

    def clip(self, solid):
        return solid ^ self.mask

    def graph(self, pitch):
        axes = [np.arange(lo+pitch/2, hi, pitch) for lo, hi in self.reference.bounds.T]
        grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
        flat = grid.reshape(-1, 3)
        occupied = np.empty(len(flat), bool)
        for start in range(0, len(flat), 50000):
            occupied[start:start+50000] = self.reference.contains(flat[start:start+50000])
        inside = occupied.reshape(grid.shape[:3])
        labels = np.full(inside.shape, -1, np.int64)
        labels[inside] = np.arange(inside.sum())
        clearance = distance_transform_edt(inside)
        start_rows, end_rows, weights = [], [], []
        for axis in range(3):
            lower, upper = [slice(None)]*3, [slice(None)]*3
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            a, b = labels[tuple(lower)], labels[tuple(upper)]
            selected = (a >= 0)&(b >= 0)
            depth = np.minimum(clearance[tuple(lower)][selected], clearance[tuple(upper)][selected])
            start_rows.append(a[selected]);end_rows.append(b[selected])
            weights.append(pitch*(1+.5/np.maximum(depth, 1)))
        a, b, w = map(np.concatenate, (start_rows, end_rows, weights))
        node_points = flat[occupied]
        legal = np.ones(len(a), bool)
        # Sampling only proposes routes; exact clipped-solid connectivity and
        # complete continuous sweep checks remain mandatory final acceptance.
        for fraction in (.25, .5, .75):
            indices = np.flatnonzero(legal)
            probes = node_points[a[indices]]*(1-fraction)+node_points[b[indices]]*fraction
            for start in range(0, len(indices), 50000):
                rows = indices[start:start+50000]
                legal[rows] = self.reference.contains(probes[start:start+50000])
        a,b,w = a[legal],b[legal],w[legal]
        graph = coo_matrix((np.r_[w,w], (np.r_[a,b],np.r_[b,a])), shape=(inside.sum(),)*2).tocsr()
        _, regions = connected_components(graph, directed=False)
        largest = int(np.argmax(np.bincount(regions)))
        selected = np.flatnonzero(regions == largest)
        self.nodes = flat[occupied][selected]
        self.graph_data = graph[selected][:, selected]
        self.tree = cKDTree(self.nodes)
        print('GROW GRID',self.group.name,'pitch_mm',pitch*1000,'nodes',len(self.nodes),flush=True)

    def attach(self, original, points, name):
        centroid = np.asarray(points).mean(axis=0)
        _, nearest = self.tree.query(centroid, k=min(48,len(self.nodes)))
        for node in np.atleast_1d(nearest):
            anchor = S.solid(G.hull_mesh(self.nodes[node]+self.bead))
            blank = S.solid(G.hull_mesh(np.vstack([points, self.nodes[node]+self.bead])))
            grown = self.clip(blank)+original
            if len(components(grown)) != 1:
                continue
            if abs(float((original-grown).volume()))*S.SCALE**3 > 8e-14:
                continue
            if abs(float((grown^anchor).volume()))*S.SCALE**3 < 1e-12:
                continue
            self.terminals.append(dict(name=name,node=int(node),solid=grown))
            return
        raise RuntimeError('No local growth anchor for '+name)

    def roots(self):
        for k, (row, contacts, b, o) in enumerate(zip(self.case.support_seeds,self.case.groups,self.bases,self.offsets)):
            for index, (cells, contact) in enumerate(zip(row,contacts)):
                points = [v@b+o for v in cells]
                root = F.union([S.solid(G.hull_mesh(v)) for v in points])
                self.attach(root,np.vstack(points),self.case.poses[k]+':'+contact['candidate_id'])

    def ground(self):
        for k, (b,o,demands) in enumerate(zip(self.bases,self.offsets,self.case.demands)):
            world = (self.reference.vertices-o)@b.T
            ground = world[np.abs(world[:,2]) < 1e-9, :2]
            selected = prune_ground(ground, demands)
            self.feet.append(dict(pose=self.case.poses[k],centers_xy_m=selected.tolist()))
            for index, xy in enumerate(selected):
                # Grow a local four-millimetre sole at a feasible hull corner.
                angles = np.arange(16)*2*np.pi/16
                disk = xy+.007*np.c_[np.cos(angles),np.sin(angles)]
                vertices = np.vstack([np.c_[disk,np.full(16,z)] for z in (-1e-6,.004)])@b+o
                foot = self.clip(S.solid(G.hull_mesh(vertices)))
                if foot.is_empty():
                    raise RuntimeError('Local ground sole is empty')
                if len(components(foot)) != 1:
                    selected_part = None
                    for part in sorted(components(foot), key=lambda p:-p.volume()):
                        part = self.clip(part)  # Preserve all true cavities.
                        v = (S.unpack(part).vertices-o)@b.T
                        floor_xy = v[np.abs(v[:,2]) < 1e-9,:2]
                        if len(floor_xy) and MultiPoint(floor_xy).convex_hull.distance(MultiPoint([xy])) < 1e-9:
                            if len(components(part)) == 1:
                                selected_part = part
                                break
                    if selected_part is None:
                        raise RuntimeError('No connected sole at the required ground corner')
                    foot = selected_part
                points = S.unpack(foot).vertices
                self.attach(foot,points,f'{self.case.poses[k]}:foot{index}')

    def connect(self):
        """Incrementally grow the nearest unconnected terminal into one tree."""
        targets = {t['node'] for t in self.terminals}
        network = {self.terminals[0]['node']}
        edge_set = set()
        while targets-network:
            distance, predecessor, _ = dijkstra(self.graph_data,directed=False,
                indices=np.array(sorted(network)),min_only=True,return_predecessors=True)
            target = min(targets-network,key=lambda q:(distance[q],q))
            if not np.isfinite(distance[target]):
                raise RuntimeError('Terminals have no grid route')
            route = [target]
            while route[-1] not in network:
                next_node = int(predecessor[route[-1]])
                if next_node < 0:
                    raise RuntimeError('Invalid growth predecessor')
                edge_set.add(tuple(sorted((route[-1],next_node))))
                route.append(next_node)
            network.update(route)
            self.paths.append(self.nodes[route].tolist())
        # Merge straight collinear steps to reduce Boolean work. A branch
        # node stays an endpoint, so shared trunks are emitted only once.
        adjacency = {q:[] for q in network}
        for a,b in sorted(edge_set):adjacency[a].append(b);adjacency[b].append(a)
        remaining=set(edge_set);segments=[]
        while remaining:
            first=min(remaining);route=list(first);remaining.remove(first)
            for reverse in (False,True):
                if reverse:route.reverse()
                while len(adjacency[route[-1]]) == 2 and route[-1] not in targets:
                    candidates=[q for q in adjacency[route[-1]] if tuple(sorted((q,route[-1]))) in remaining]
                    if not candidates:break
                    q=candidates[0]
                    a=self.nodes[route[-1]]-self.nodes[route[-2]];b=self.nodes[q]-self.nodes[route[-1]]
                    if np.linalg.norm(np.cross(a,b)) > 1e-12:break
                    remaining.remove(tuple(sorted((q,route[-1]))));route.append(q)
            segments.append((route[0],route[-1]))
        beams=[self.clip(S.solid(G.hull_mesh(np.vstack([self.nodes[a]+self.bead,self.nodes[b]+self.bead]))))
               for a,b in segments]
        full=F.union([t['solid'] for t in self.terminals]+beams)
        if len(components(full)) != 1:
            retained, detail = F.maximal_component(full, md.Manifold(), [t['solid'] for t in self.terminals])
            if retained is None:
                raise RuntimeError('Clipped growth tree is disconnected')
            full=retained
        return full,dict(grid_edge_count=len(edge_set),grown_segment_count=len(segments),terminal_count=len(self.terminals))

    def verify(self, mesh):
        if not self.search.sweep_meshes:
            self.search.precompute()
        sweeps=[trimesh.Trimesh(m.vertices@b+o,m.faces,process=False)
                for m,b,o in zip(self.search.sweep_meshes,self.bases,self.offsets)]
        with contextlib.redirect_stdout(io.StringIO()):
            checks,cert=S.verify(self.case.tasks,self.case.groups,self.case.support_seeds,
                self.directions,self.bases,self.offsets,mesh,sweeps,check_equilibrium=False)
        coverage=[]
        for check,demands in zip(checks,self.case.demands):
            actual=MultiPoint(check['actual_ground_hull_xy_m']).convex_hull
            missing=MultiPoint(demands).convex_hull.difference(actual.buffer(1e-10)).area
            coverage.append(dict(pose=check['pose'],uncovered_area_m2=float(missing),passed=missing <= 1e-12))
        work=self.guard.verify(mesh)
        boxes=F.box_checks(mesh,self.reference_report['space_budget'],self.bases,self.offsets,self.case.poses)
        assert mesh.is_watertight and mesh.is_winding_consistent
        assert all(c['passed'] for c in coverage) and work['passed'] and all(c['all_inside'] for c in boxes)
        return dict(box=self.reference_report['space_budget'],checks=checks,
            ground_coverage=coverage,working_surface=work,box_checks=boxes),cert

    def run(self,pitch=.008):
        started=time.monotonic()
        self.graph(pitch);self.roots();self.ground();solid,construction=self.connect()
        mesh=S.unpack(solid);result,cert=self.verify(mesh)
        D.export_exact_obj(mesh,self.out/'shape.obj')
        np.savez_compressed(self.out/'geometry_certificate.npz',**cert)
        # Replay the serialized material, not just the in-memory proposal.
        exported=trimesh.load(self.out/'shape.obj',force='mesh',process=False)
        replay,_=self.verify(exported)
        actual=D.measure(self.case,mesh,self.bases,self.offsets)
        report=dict(complete=True,passed=True,constructed=True,export_roundtrip_bit_exact=True,
            object=self.case.name,poses=self.case.poses,placement=self.reference_report['placement'],
            status='growing_support_geometry_passed',passed_scope='step4_geometry_only',
            step3_passed=self.case.schedule['passed'],step3_verdict_modified=False,
            objective='Preserve compact workstation box; reduce material through local growth and shared short paths',
            minimum_material_claim=False,sweep_conditioning=self.search.sweep_conditioning,
            construction=dict(method='roots and sparse ground soles grow local lofts; shared shortest-path tree',
                initial_block_material=False,reference_used_as='navigation and clipping domain only',
                grid_pitch_m=pitch,beam_radius_m=.004,feet=self.feet,paths_m=self.paths,**construction),
            result=result,exported_model_replay=replay,space_budget=actual,
            volume_cm3=float(mesh.volume*1e6),previous_material_volume_cm3=float(self.reference.volume*1e6),
            material_reduction_percent=100*(1-mesh.volume/self.reference.volume),
            previous_space_budget=self.reference_report['space_budget'],seconds=time.monotonic()-started,
            deterministic=True,random_sampling=False,global_optimality_claim=False,
            provenance=dict(inputs=I.hashes([self.source/'report.json',self.source/'shape.obj']+self.case.paths),
                code=I.hashes([Path(__file__),Path(D.__file__),Path(F.__file__),Path(S.__file__),
                    Path(G.__file__),Path(S.swept_solid.__code__.co_filename)]+A.sources())),
            artifacts={p:I.sha256(self.out/p) for p in ('shape.obj','geometry_certificate.npz')})
        I.save(self.out/'report.json',report)
        print('GROW PASS',self.group.name,'material_reduction',round(report['material_reduction_percent'],1),'seconds',round(report['seconds'],1),flush=True)
        return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups',nargs='+',default=['pose3+6'])
    p.add_argument('--pitch-mm',type=float,default=8)
    args=p.parse_args();root=I.OUTPUTS/'B';protected=protected_hashes(root)
    for name in args.groups:
        if name=='pose1+3':raise ValueError('Excluded reference')
        Grow(root/name).run(args.pitch_mm/1000)
    assert protected==protected_hashes(root)
