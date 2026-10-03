"""Apply current compact-space and sparse growth to a COPY of the historical pair.

Historical six physical patches remain six patches; five shared IDs do not
establish valid strict registration. Geometry acceptance is separate.
"""
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint,Polygon
from scipy.sparse.csgraph import connected_components,dijkstra
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support import co_design_walkthrough as C, boxed_support as F
from step4_connect_support import deterministic_space as D, growing_support as L
from step4_connect_support import build_coupled_saddle as S, space_budget as B
from step4_connect_support import replay_compact as R, render_compact_images as V
from step0_pose_selection.floor_points import pressure_centers
from step2_local_support import geometry as G

ROOT=I.OUTPUTS/'B'
ORIGINAL=ROOT/'pose1+3'
COPY=ROOT/'pose1+3copied'

def tree_hash(path):
    return {str(p.relative_to(path)):I.sha256(p) for p in sorted(path.rglob('*')) if p.is_file()}

def historical_case(group,out):
    if group != COPY:raise ValueError('Adapter only supports explicit historical copy')
    old=C.read_reference()
    xy=[pressure_centers(t.targets/t.scale,t.domain.com)[0] for t in old.tasks]
    case=SimpleNamespace(name='B',poses=C.POSES,pair=group,output=out,tasks=old.tasks,
        groups=old.groups,heads=old.heads,support_seeds=old.heads,schedule=old.schedule,
        paths=old.paths,demands=xy)
    case.root_solids=[F.union([S.solid(G.hull_mesh(v)) for cells in row for v in cells]) for row in old.heads]
    space=B.analyze([t.domain.mesh.vertices for t in old.tasks],
                    [np.c_[v,np.zeros(len(v))] for v in xy],C.POSES)
    return case,space

class StagedGrow(L.Grow):
    def __init__(self,group):
        super().__init__(group);self.original_roots=[];self.original_soles=[]
        # D.Search adds the mutable public report, which must not become a
        # self-referential physical input of this historical adapter.
        self.case.paths=[p for p in self.case.paths if p != group/'step4/data/report.json']
        # A 6 mm sphere tessellation has >5 mm inscribed diameter.
        sphere=trimesh.creation.icosphere(subdivisions=2,radius=.003)
        self.bead=sphere.vertices
        self.guaranteed_radius=float(np.min(np.abs(np.einsum('ij,ij->i',sphere.face_normals,sphere.triangles[:,0]))))
        self.thickness=[];self.core_solids=[]
    def graph(self,pitch):
        super().graph(pitch)
        distance=trimesh.proximity.ProximityQuery(self.reference).on_surface(self.nodes)[1]
        # Lipschitz bound: each centerline point is <=pitch/2 from an endpoint.
        legal=distance >= .003+pitch/2+1e-7
        graph=self.graph_data[legal][:,legal]
        _,regions=connected_components(graph,directed=False)
        largest=int(np.argmax(np.bincount(regions)))
        selected=np.flatnonzero(regions==largest)
        self.nodes=self.nodes[legal][selected];self.graph_data=graph[selected][:,selected]
        self.tree=cKDTree(self.nodes)
        print('THICK GRID',len(self.nodes),'guaranteed_diameter_mm',self.guaranteed_radius*2000,flush=True)
    def clip(self,solid):
        # Never clip a body into a thin residual neck.
        missing=abs(float((solid-self.mask).volume()))*S.SCALE**3
        if missing > 8e-14:raise RuntimeError('Unclipped body exceeds legal domain: '+str(missing))
        return solid
    def attach(self,original,points,name):
        (self.original_soles if ':foot' in name else self.original_roots).append(original)
        centroid=np.asarray(points).mean(axis=0)
        contact_start=None;foot_column=None
        if ':foot' in name:
            pose=name.split(':',1)[0];k=self.case.poses.index(pose)
            normal=self.bases[k][2]
            # Ground cylinder continues vertically into the full ball, so
            # no tapered or clipped foot-to-stem neck is needed.
            contact_start=centroid+.0005*normal
            lower=np.asarray(points)
            upper=np.asarray(points)+.001*normal
            foot_column=S.solid(G.hull_mesh(np.vstack([lower,upper])))
            if abs(float((foot_column-self.mask).volume()))*S.SCALE**3 > 8e-14:
                raise RuntimeError('Full six-mm foot column outside legal domain')
        if ':foot' not in name:
            pose,ident=name.split(':',1);k=self.case.poses.index(pose)
            contact=next(c for c in self.case.groups[k] if c['candidate_id']==ident)
            centroid=np.asarray(contact['center_m'])@self.bases[k]+self.offsets[k]
            normal=self.case.tasks[k].domain.mesh.face_normals[int(contact['center_face'])]@self.bases[k]
            contact_start=centroid+.0031*normal
        _,nearest=self.tree.query(centroid,k=min(2048,len(self.nodes)))
        for node in np.atleast_1d(nearest):
            anchor=self.nodes[node]
            vector=anchor-centroid;length=float(np.linalg.norm(vector))
            if length < 1e-8:continue
            # Original contact taper is a short local transition only. Beyond
            # it, a full circumscribing >=5mm sphere and rod are mandatory.
            start=contact_start if contact_start is not None else centroid+vector/length*min(.003,length)
            ball=S.solid(G.hull_mesh(start+self.bead))
            if abs(float((ball-self.mask).volume()))*S.SCALE**3 > 8e-14:continue
            stem=S.solid(G.hull_mesh(np.vstack([start+self.bead,anchor+self.bead])))
            if abs(float((stem-self.mask).volume()))*S.SCALE**3 > 8e-14:continue
            transition=foot_column if foot_column is not None else S.solid(G.hull_mesh(np.vstack([points,start+self.bead]))) ^ self.mask
            grown=original+transition+stem
            if len(L.components(grown)) != 1:continue
            if abs(float((original-grown).volume()))*S.SCALE**3 > 8e-14:continue
            self.terminals.append(dict(name=name,node=int(node),solid=grown))
            self.core_solids.append((name,stem))
            self.thickness.append(dict(name=name,start_m=start.tolist(),anchor_m=anchor.tolist(),
                guaranteed_core_diameter_mm=self.guaranteed_radius*2000,
                original_contact_transition_length_mm=(np.linalg.norm(start-centroid))*1000,
                transition_may_taper=foot_column is None))
            return
        raise RuntimeError('No deterministic local anchor for '+name)
    def save_stage(self,name,solid):
        D.export_exact_obj(S.unpack(solid),self.out/(name+'.obj'))
    def roots(self):
        super().roots();self.save_stage('roots',F.union(self.original_roots))
    def ground(self):
        for k,(b,o,demands) in enumerate(zip(self.bases,self.offsets,self.case.demands)):
            world=(self.reference.vertices-o)@b.T
            xy=world[np.abs(world[:,2]) < 1e-9,:2]
            fitted=trimesh.Trimesh(world,self.reference.faces,process=False)
            section=fitted.section(plane_origin=[0,0,.006],plane_normal=[0,0,1])
            if section is None:raise RuntimeError('No five-mm sole upper section')
            polygons=[Polygon(v[:,:2]) for v in section.discrete if len(v)>=4]
            upper=max(polygons,key=lambda p:p.area)
            inset=MultiPoint(xy).convex_hull.intersection(upper).buffer(-.00301,join_style=2)
            if inset.geom_type != 'Polygon':raise RuntimeError('No thick sole floor region')
            selected=L.prune_ground(np.asarray(inset.exterior.coords)[:-1],demands)
            required=MultiPoint(demands).convex_hull
            # Actual sole material, rather than centers, supplies ground hull.
            self.feet.append(dict(pose=self.case.poses[k],centers_xy_m=selected.tolist()))
            actual_ground=[]
            for index,center in enumerate(selected):
                angle=np.arange(32)*2*np.pi/32
                disk=center+.003*np.c_[np.cos(angle),np.sin(angle)]
                vertices=np.vstack([np.c_[disk,np.full(32,z)] for z in (0.,.005)])@b+o
                actual_ground.append(disk)
                sole=S.solid(G.hull_mesh(vertices))
                self.clip(sole)
                self.attach(sole,vertices,f'{self.case.poses[k]}:foot{index}')
            if required.difference(MultiPoint(np.vstack(actual_ground)).convex_hull.buffer(1e-10)).area > 1e-12:
                raise RuntimeError('Actual thick sole hull does not cover demands')
        self.save_stage('soles',F.union(self.original_roots+self.original_soles))
        self.save_stage('local_bodies',F.union([t['solid'] for t in self.terminals]))
    def connect(self):
        targets={t['node'] for t in self.terminals};network={self.terminals[0]['node']}
        beams=[];direct_count=0;shortcut_count=0
        def rod(a,b):
            return S.solid(G.hull_mesh(np.vstack([self.nodes[a]+self.bead,self.nodes[b]+self.bead])))
        def legal(a,b):
            body=rod(a,b)
            return body if abs(float((body-self.mask).volume()))*S.SCALE**3 <= 8e-14 else None
        while targets-network:
            pairs=sorted((float(np.linalg.norm(self.nodes[a]-self.nodes[b])),a,b)
                for a in targets-network for b in network)
            chosen=None
            for _,a,b in pairs:
                body=legal(a,b)
                if body is not None:chosen=(a,b,body);break
            if chosen is not None:
                a,b,body=chosen;beams.append(body);network.add(a)
                self.paths.append(self.nodes[[a,b]].tolist());direct_count+=1;continue
            distance,pred,_=dijkstra(self.graph_data,directed=False,indices=np.asarray(sorted(network)),min_only=True,return_predecessors=True)
            target=min(targets-network,key=lambda q:(distance[q],q))
            if not np.isfinite(distance[target]):raise RuntimeError('No thick route')
            route=[target]
            while route[-1] not in network:route.append(int(pred[route[-1]]))
            kept=[route[0]];index=0
            while index < len(route)-1:
                for end in range(len(route)-1,index,-1):
                    body=legal(route[index],route[end])
                    if body is not None:break
                else:raise RuntimeError('No unclipped shortcut edge')
                beams.append(body);kept.append(route[end]);shortcut_count+=end-index-1;index=end
            network.update(kept);self.paths.append(self.nodes[kept].tolist())
        full=F.union([t['solid'] for t in self.terminals]+beams)
        if len(L.components(full)) != 1:raise RuntimeError('Thick network disconnected')
        self.save_stage('shared_tree',full)
        # All beam polyhedra are preserved in their entirety, verified against
        # the serialized final body after construction.
        self.beams=beams
        return full,dict(grown_segment_count=len(beams),terminal_count=len(self.terminals),
            direct_connection_count=direct_count,removed_grid_bends=shortcut_count,
            routing='whole-solid direct connection first; farthest-visible shortcuts on fallback grid paths')

def main():
    before=tree_hash(ORIGINAL)
    if not COPY.exists():shutil.copytree(ORIGINAL,COPY)
    I.save(COPY/'step4/data/copied_reference_manifest.json',dict(source=str(ORIGINAL.relative_to(I.ROOT)),original_hashes=before,copied_without_numerical_change=True))
    old=C.read_reference()
    placement=dict(bases=old.bases.tolist(),offsets=old.offsets.tolist(),directions=old.directions.tolist())
    # Only the COPY receives the adapter's placement schema.
    I.save(COPY/'step4/data/report.json',dict(complete=True,placement=placement,passed=None,
        historical_source=True,strict_shared_head_registration_passed=False))
    F.read_case=historical_case
    # Keep compact placements, expand only the spatial budget enough for full rods.
    search=D.Search(COPY)
    search.case.paths=[p for p in search.case.paths if p != COPY/'step4/data/report.json']
    search.before=I.hashes(search.case.paths)
    existing=json.loads((search.out/'report.json').read_text())
    search.precompute()
    bases=old.bases.copy();offsets=old.offsets.copy()
    existing['placement']=dict(bases=bases.tolist(),offsets=offsets.tolist(),directions=old.directions.tolist())
    report=None
    for margin_mm in (8,16,24):
        budget=D.enlarged(search.envelope(bases,offsets),margin_mm/1000)
        candidate=search.construct(bases,offsets,budget)
        if candidate is None:continue
        mesh,cert,check=candidate
        D.export_exact_obj(mesh,search.out/'shape.obj')
        np.savez_compressed(search.out/'geometry_certificate.npz',**cert)
        boxed=dict(existing,provenance=dict(existing['provenance'],inputs=search.before),result=check,space_budget=D.measure(search.case,mesh,bases,offsets),volume_cm3=float(mesh.volume*1e6),thickening_domain_margin_mm=margin_mm)
        boxed['artifacts']={name:I.sha256(search.out/name) for name in existing['artifacts']}
        I.save(search.out/'report.json',boxed)
        try:
            growth=StagedGrow(COPY);report=growth.run(pitch=.004)
            break
        except RuntimeError as error:
            print('THICK RETRY',margin_mm,str(error),flush=True)
    if report is None:raise RuntimeError('No complete thick growth found')
    final_mesh=trimesh.load(growth.out/'shape.obj',force='mesh',process=False)
    final_solid=S.solid(final_mesh)
    cores=growth.core_solids+[(f'beam_{j}',core) for j,core in enumerate(growth.beams)]+[(f'sole_{j}',core) for j,core in enumerate(growth.original_soles)]
    core_checks=[dict(name=name,missing_volume_m3=abs(float((core-final_solid).volume()))*S.SCALE**3) for name,core in cores]
    assert all(c['missing_volume_m3'] <= 8e-14 for c in core_checks)
    report['construction']['beam_radius_m']=.003
    report['minimum_branch_thickness']=dict(minimum_required_diameter_mm=5.,guaranteed_inscribed_beam_diameter_mm=growth.guaranteed_radius*2000,
        beam_outer_diameter_mm=6.,sole_thickness_mm=5.,unclipped_body_validation=True,
        centerline_validation='Exact mesh surface distances at grid endpoints; Lipschitz bound over each 4mm edge',
        scope='load-bearing connecting rods and 5mm ground soles; original contact margins and first 3.1mm contact transitions excluded',
        local_branches=growth.thickness,exported_unclipped_core_checks=core_checks,all_complete_cores_preserved=True)
    report['historical_reference']=dict(source_head_id_count=5,physical_contact_patch_count=6,
        strict_shared_head_registration_passed=False,original_source_rejection_preserved=True,
        archived_pose_revision=True,force_torque_not_resolved_by_step4=True)
    I.save(growth.out/'report.json',report)
    object_mesh=growth.case.tasks[0].domain.mesh.copy()
    object_mesh.vertices=object_mesh.vertices@growth.bases[0]+growth.offsets[0]
    D.export_exact_obj(object_mesh,growth.out/'object_pose1_fixture.obj')
    I.save(growth.out/'stages.json',dict(placement=report['placement'],poses=C.POSES,reference_pose='pose_1',
        object_mesh='object_pose1_fixture.obj',camera_view_fixture=((-growth.directions[0]+np.array([0.,0.,.65]))@growth.bases[0]).tolist(),
        coordinate_frame='fixture',stages=[dict(key=k,label=label,mesh=k+'.obj') for k,label in
            [('roots','Contact roots'),('soles','Sparse ground soles'),('local_bodies','Local tapered bodies'),('shared_tree','Shared connecting tree')]]+
            [dict(key='final',label='Verified support',mesh='shape.obj')]))
    R.replay(COPY,'growing_support')
    # Remove inherited construction sheet; parent generates the NEW actual stages.
    for p in (COPY/'step4').glob('*.png'):p.unlink()
    V.render(COPY,'growing_support')
    from step4_connect_support.draw_growth_steps import draw
    draw(growth.out)
    shutil.copyfile(growth.out/'shape.obj',COPY/'step4/shape.obj')
    I.save(COPY/'step4/data/report.json',dict(report,artifacts={'growing_support/'+p:digest for p,digest in report['artifacts'].items()}))
    assert before==tree_hash(ORIGINAL),'Original historical pair modified'
    I.save(COPY/'step4/data/original_preservation.json',dict(passed=True,file_count=len(before),original_unchanged=True))

if __name__=='__main__':main()
