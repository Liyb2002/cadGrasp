"""Grow a shared fixture directly from actual contact-root starts toward foot targets.

No preselected grid skeleton or terminal-to-grid loft is emitted. Grid routing
is lazy fallback only, and every emitted rod is tested as a complete solid.
"""
from pathlib import Path
import numpy as np
import time
from step3_scheculer import contacts as I
from step4_connect_support import process_access as A
import trimesh
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import cKDTree
from step4_connect_support import run_thick_batch as B, growing_support as L
from step4_connect_support import boxed_support as F,build_coupled_saddle as S
from step4_connect_support import deterministic_space as D
from step2_local_support import geometry as G

class DirectGrow(B.BatchGrow):
    def __init__(self,group):
        super().__init__(group)
        if group.name=='pose1+3copied':
            sphere=trimesh.creation.icosphere(subdivisions=2,radius=.003)
            self.bead=sphere.vertices
            self.guaranteed_radius=float(np.min(np.abs(np.einsum('ij,ij->i',sphere.face_normals,sphere.triangles[:,0]))))
        self.seed_positions=[];self.segments=[];self.grid_ready=False;self.journal=[];self.partial_steps=0
        # Placement and ground targets are frozen inputs, not a box to solve.
        self.foot_reference=self.reference.copy()
        self.search.directions=self.directions
        self.search.precompute()
        mandatory,_,self.forbidden=self.search.installed_geometry(self.bases,self.offsets)
        initial=D.enlarged(self.search.envelope(self.bases,self.offsets),
            .05)
        # Finite navigation window only; no block is constructed or certified.
        self.mask=(F.bounded_space(initial,self.bases,self.offsets)-self.forbidden)+F.union(mandatory)
        self.navigation_window=initial
        self.reference=S.unpack(self.mask)
        self.occupied_lo=None;self.occupied_hi=None
    def graph(self,pitch):
        L.Grow.graph(self,pitch)
        query=trimesh.proximity.ProximityQuery(self.reference)
        distance=np.empty(len(self.nodes))
        for first in range(0,len(self.nodes),2048):
            distance[first:first+2048]=query.on_surface(self.nodes[first:first+2048])[1]
        legal=distance >= .003+pitch/2+1e-7
        graph=self.graph_data[legal][:,legal]
        _,regions=connected_components(graph,directed=False)
        largest=int(np.argmax(np.bincount(regions)))
        selected=np.flatnonzero(regions==largest)
        self.nodes=self.nodes[legal][selected];self.graph_data=graph[selected][:,selected]
        self.tree=cKDTree(self.nodes)
        self.grid_ready=True
        print('THICK GRID',len(self.nodes),'chunked clearance',flush=True)
    def sphere(self,point):return S.solid(G.hull_mesh(point+self.bead))
    def rod(self,a,b):return S.solid(G.hull_mesh(np.vstack([a+self.bead,b+self.bead])))
    def legal(self,body):return abs(float((body-self.mask).volume()))*S.SCALE**3 <= 8e-14
    def attach(self,original,points,name):
        foot=':foot' in name
        (self.original_soles if foot else self.original_roots).append(original)
        pose=name.split(':',1)[0];k=self.case.poses.index(pose)
        if foot:
            center=np.asarray(points).mean(axis=0)
            normal=self.bases[k][2]
            start=center+.0005*normal
            column=S.solid(G.hull_mesh(np.vstack([points,np.asarray(points)+.001*normal])))
            if not self.legal(column):raise RuntimeError('Foot target complete column outside domain: '+name)
            choices=[('full_column',start,column,.0005)]
        else:
            ident=name.split(':',1)[1];c=next(c for c in self.case.groups[k] if c['candidate_id']==ident)
            center=np.asarray(c['center_m'])@self.bases[k]+self.offsets[k]
            face=self.case.tasks[k].domain.mesh.face_normals[int(c['center_face'])]@self.bases[k]
            average=np.sum(self.case.tasks[k].domain.mesh.face_normals[c['source_faces']]*c['triangle_areas_m2'][:,None],axis=0)@self.bases[k]
            average/=np.linalg.norm(average)
            choices=[(label,center+depth*normal,None,depth) for label,normal in [('center_face',face),('patch_area_average',average)]
                for depth in (.0031,.0036,.0041,.0046,.0051)]
        for mode,start,column,depth in choices:
            ball=self.sphere(start)
            if not self.legal(ball):continue
            transition=column if foot else (S.solid(G.hull_mesh(np.vstack([points,start+self.bead]))) ^ self.mask)
            seed=original+transition+ball
            if len(L.components(seed)) != 1:continue
            if abs(float((original-seed).volume()))*S.SCALE**3 > 8e-14:continue
            index=len(self.seed_positions);self.seed_positions.append(np.asarray(start))
            self.terminals.append(dict(name=name,node=index,solid=seed,kind='foot_target' if foot else 'head_start'))
            self.core_solids.append((name+':start_ball',ball))
            if foot:self.core_solids.append((name+':column',column))
            self.thickness.append(dict(name=name,start_m=start.tolist(),guaranteed_core_diameter_mm=self.guaranteed_radius*2000,
                original_contact_transition_length_mm=depth*1000,transition_normal=mode,transition_may_taper=not foot,
                preset_grid_anchor=False))
            return
        raise RuntimeError('No legal original contact growth start: '+name)
    def ground(self):
        navigation=self.reference
        self.reference=self.foot_reference
        try:super().ground()
        finally:self.reference=navigation
        # Legacy superclass stages never enter construction; remove its former
        # panel name so no initial terminal-to-grid body is implied.
        (self.out/'local_bodies.obj').unlink(missing_ok=True)
        self.save_stage('seed_targets',F.union([t['solid'] for t in self.terminals]))
    def initialize_occupancy(self):
        if not hasattr(self,'case'):return
        points=[task.domain.mesh.vertices for task in self.case.tasks]
        for terminal in self.terminals:
            vertices=S.unpack(terminal['solid']).vertices
            points.extend((vertices-o)@b.T for b,o in zip(self.bases,self.offsets))
        joined=np.vstack(points);self.occupied_lo=joined.min(axis=0);self.occupied_hi=joined.max(axis=0)

    def volume_key(self,points):
        if getattr(self,'occupied_lo',None) is None:return 0.
        vertices=np.vstack([p+self.bead for p in points])
        world=np.vstack([(vertices-o)@b.T for b,o in zip(self.bases,self.offsets)])
        lo=np.minimum(self.occupied_lo,world.min(axis=0));hi=np.maximum(self.occupied_hi,world.max(axis=0))
        return float(np.prod(hi-lo)-np.prod(self.occupied_hi-self.occupied_lo))

    def include_path(self,points):
        if getattr(self,'occupied_lo',None) is None:return
        vertices=np.vstack([p+self.bead for p in points])
        world=np.vstack([(vertices-o)@b.T for b,o in zip(self.bases,self.offsets)])
        self.occupied_lo=np.minimum(self.occupied_lo,world.min(axis=0))
        self.occupied_hi=np.maximum(self.occupied_hi,world.max(axis=0))

    def verify(self,mesh):
        # The box is measured from the emitted material, never a size gate.
        previous=self.reference_report
        self.reference_report=dict(previous,space_budget=D.measure(self.case,mesh,self.bases,self.offsets))
        try:return super().verify(mesh)
        finally:self.reference_report=previous

    def candidates(self,targets,network):
        candidates=[]
        for target in sorted(targets):
            start=self.seed_positions[target]
            ends=[self.seed_positions[j] for j in sorted(network)]
            for a,b in self.segments:
                vector=b-a;fraction=np.clip(np.dot(start-a,vector)/np.dot(vector,vector),0,1)
                ends.append(a+fraction*vector)
            seen=set()
            for end in ends:
                key=tuple(end)
                if key in seen:continue
                seen.add(key);candidates.append((self.volume_key([start,end]),float(np.linalg.norm(start-end)),target,key,np.asarray(end)))
        return sorted(candidates,key=lambda c:c[:4])
    def fallback(self,target,network):
        if not self.grid_ready:
            # This creates only a navigation graph. No grid material or
            # terminal-to-grid loft is added before an actual route is chosen.
            self.graph(self.pitch)
        def visible(point):
            _,near=self.tree.query(point,k=min(2048,len(self.nodes)))
            for node in np.atleast_1d(near):
                body=self.rod(point,self.nodes[node])
                if self.legal(body):return int(node)
            return None
        source=self.seed_positions[target];begin=visible(source)
        ends=[]
        for node in sorted(network):
            endpoint=visible(self.seed_positions[node])
            if endpoint is not None:ends.append((endpoint,node))
        if begin is None or not ends:raise RuntimeError('No legal head-grown grid attachment')
        dist,pred=dijkstra(self.graph_data,directed=False,indices=begin,return_predecessors=True)
        end,owned=min(ends,key=lambda row:(dist[row[0]],row))
        if not np.isfinite(dist[end]):raise RuntimeError('No fallback route from actual head')
        chain=[end]
        while chain[-1]!=begin:
            node=int(pred[chain[-1]])
            if node<0:raise RuntimeError('Invalid lazy grid route')
            chain.append(node)
        points=[source]+list(self.nodes[chain[::-1]])+[self.seed_positions[owned]]
        route=[points[0]];index=0
        while index<len(points)-1:
            for last in range(len(points)-1,index,-1):
                body=self.rod(points[index],points[last])
                if self.legal(body):break
            else:raise RuntimeError('No full-solid shortcut')
            route.append(points[last]);index=last
        return route
    def connect(self):
        self.initialize_occupancy()
        targets=set(range(len(self.terminals)));network={0};beams=[];direct=0;fallback=0;projected=0
        midpoint_saved=False;goal=max(2,len(targets)//2)
        while targets-network:
            chosen=None
            for _,_,target,_,end in self.candidates(targets-network,network):
                body=self.rod(self.seed_positions[target],end)
                if self.legal(body):chosen=(target,end,body);break
            if chosen is not None:
                target,end,body=chosen;start=self.seed_positions[target]
                beams.append(body);self.segments.append((start.copy(),end.copy()));network.add(target)
                self.include_path([start,end]);self.paths.append([start.tolist(),end.tolist()]);direct+=1
                self.journal.append(dict(step=len(self.journal)+1,target=self.terminals[target]['name'],target_kind=self.terminals[target]['kind'],
                    start_network_point_m=end.tolist(),path_m=[end.tolist(),start.tolist()],core_segments=1))
                if not any(np.array_equal(end,self.seed_positions[j]) for j in network):projected+=1
            else:
                routes=[]
                for target in sorted(targets-network):
                    try:
                        route=self.fallback(target,network)
                    except RuntimeError:continue
                    length=sum(np.linalg.norm(b-a) for a,b in zip(route,route[1:]))
                    routes.append((self.volume_key(route),float(length),target,route))
                if not routes:raise RuntimeError('No head-grown route in finite navigation window')
                _,_,target,route=min(routes,key=lambda row:row[:3])
                self.include_path(route)
                for a,b in zip(route,route[1:]):
                    body=self.rod(a,b)
                    if not self.legal(body):raise RuntimeError('Fallback emitted illegal rod')
                    beams.append(body);self.segments.append((np.asarray(a),np.asarray(b)))
                network.add(target);self.paths.append([p.tolist() for p in route]);fallback+=1
                self.journal.append(dict(step=len(self.journal)+1,target=self.terminals[target]['name'],target_kind=self.terminals[target]['kind'],
                    start_network_point_m=route[-1].tolist(),path_m=[p.tolist() for p in route[::-1]],core_segments=len(route)-1))
            if not midpoint_saved and len(network)>=goal:
                actual=F.union([self.terminals[j]['solid'] for j in sorted(network)]+beams)
                self.save_stage('partial_growth',actual);midpoint_saved=True;self.partial_steps=len(self.journal)
        full=F.union([t['solid'] for t in self.terminals]+beams)
        if len(L.components(full)) != 1:raise RuntimeError('Actual head-grown fixture disconnected')
        self.beams=beams;self.save_stage('shared_tree',full)
        if not midpoint_saved:self.save_stage('partial_growth',full)
        return full,dict(terminal_count=len(self.terminals),grown_segment_count=len(beams),
            direct_connection_count=direct,fallback_route_count=fallback,projection_shared_joint_count=projected,
            lazy_grid_constructed=self.grid_ready,preset_skeleton=False,preset_grid_anchor=False,
            initial_local_loft_to_grid=False,growth_journal=self.journal,partial_stage_step_count=self.partial_steps,
            initial_head_name=self.terminals[0]['name'],routing='actual head starts grow toward other heads and ground targets; share existing rod projections')

    def run(self,pitch=.008):
        started=time.monotonic()
        self.pitch=pitch
        self.roots();self.ground();solid,construction=self.connect()
        mesh=S.unpack(solid);result,cert=self.verify(mesh)
        D.export_exact_obj(mesh,self.out/'shape.obj')
        np.savez_compressed(self.out/'geometry_certificate.npz',**cert)
        # Replay the serialized material, not just the in-memory proposal.
        recheck_export=getattr(self,'recheck_export',True)
        replay=None
        if recheck_export:
            exported=trimesh.load(self.out/'shape.obj',force='mesh',process=False)
            replay,_=self.verify(exported)
        actual=D.measure(self.case,mesh,self.bases,self.offsets)
        report=dict(complete=True,passed=True,constructed=True,export_roundtrip_bit_exact=recheck_export,exact_export_format=True,
            object=self.case.name,poses=self.case.poses,placement=self.reference_report['placement'],
            status='growing_support_geometry_passed',passed_scope='step4_geometry_only',
            step3_passed=self.case.schedule['passed'],step3_verdict_modified=False,
            objective='Grow deterministically with minimum candidate occupied-box volume increment, then rod length; measure final actual occupied box',
            minimum_material_claim=False,sweep_conditioning=self.search.sweep_conditioning,
            construction=dict(method='actual contact starts grow full rods into a shared network, reaching original heads and foot targets',
                initial_block_material=False,reference_used_as='frozen placement and ground targets only',navigation_window=self.navigation_window,box_is_postconstruction_measurement=True,candidate_order=['occupied_box_volume_increment','rod_length','target_index','endpoint_coordinates'],
                grid_pitch_m=pitch,beam_radius_m=.003 if self.group.name=='pose1+3copied' else .0026,feet=self.feet,paths_m=self.paths,**construction),
            result=result,space_budget=actual,
            validation_policy=dict(construction_validation=True,export_recheck=recheck_export,independent_replay=False),
            volume_cm3=float(mesh.volume*1e6),previous_material_volume_cm3=float(self.reference.volume*1e6),
            material_reduction_percent=100*(1-mesh.volume/self.reference.volume),
            previous_space_budget=self.reference_report['space_budget'],seconds=time.monotonic()-started,
            deterministic=True,random_sampling=False,global_optimality_claim=False,
            provenance=dict(inputs=I.hashes(getattr(self,'construction_input_paths',[self.source/'report.json',self.source/'shape.obj'])+self.case.paths),
                code=I.hashes([Path(__file__),Path(L.__file__),Path(B.__file__),Path(B.ADAPTER.__file__),Path(D.__file__),Path(F.__file__),Path(S.__file__),
                    Path(G.__file__),Path(S.swept_solid.__code__.co_filename)]+A.sources())),
            artifacts={p:I.sha256(self.out/p) for p in ('shape.obj','geometry_certificate.npz')})
        if replay is not None:report['exported_model_replay']=replay
        I.save(self.out/'report.json',report)
        print('GROW PASS',self.group.name,'material_reduction',round(report['material_reduction_percent'],1),'seconds',round(report['seconds'],1),flush=True)
        return report

