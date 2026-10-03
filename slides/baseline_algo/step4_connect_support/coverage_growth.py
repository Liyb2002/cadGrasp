"""Grow ground contacts on demand until their material hull covers requirements.

Ground requirements are constraints, not mandatory preselected foot centers.
"""
from pathlib import Path
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import nearest_points
from step4_connect_support.direct_head_growth import DirectGrow
from step4_connect_support import boxed_support as F,build_coupled_saddle as S
from step4_connect_support import deterministic_space as D,growing_support as L
from step2_local_support import geometry as G
from step3_scheculer import contacts as I

class CoverageGrow(DirectGrow):
    recheck_export=False
    def __init__(self,group):
        started=time.monotonic()
        super().__init__(group)
        self.contact_vertices=np.unique(np.vstack([c['triangles_m'].reshape(-1,3)@b+o
            for contacts,b,o in zip(self.case.groups,self.bases,self.offsets) for c in contacts]),axis=0)
        self.buried_contact_rejections=0
        self.timings={'setup':time.monotonic()-started}

    def legal(self,body):
        if not super().legal(body):return False
        if not hasattr(self,'contact_vertices'):return True
        # Volume tolerances alone can admit micron-scale burial of a contact.
        # Primitives are convex; reject original vertices strictly inside them.
        mesh=S.unpack(body);lo,hi=mesh.bounds
        points=self.contact_vertices[np.all((self.contact_vertices>=lo)&(self.contact_vertices<=hi),axis=1)]
        if not len(points):return True
        normals=mesh.face_normals;offsets=np.einsum('ij,ij->i',normals,mesh.triangles[:,0])
        buried=np.any(np.max(points@normals.T-offsets,axis=1)<-1e-10)
        if buried:self.buried_contact_rejections+=1
        return not buried

    def roots(self):
        started=time.monotonic();super().roots()
        self.timings['contact_starts']=time.monotonic()-started

    def ground(self):
        self.feet=[dict(pose=p,centers_xy_m=[],creation_steps=[]) for p in self.case.poses]
        self.ground_candidates={};self.generated_ground=[];self.foot_records=[]
        self.required_hulls=[MultiPoint(row).convex_hull for row in self.case.demands]
        self.required_vertices=[np.asarray(hull.exterior.coords)[:-1] if hull.geom_type=='Polygon' else np.asarray(row)
            for hull,row in zip(self.required_hulls,self.case.demands)]
        self.save_stage('seed_targets',F.union([t['solid'] for t in self.terminals]))
        (self.out/'local_bodies.obj').unlink(missing_ok=True)

    def center_region(self,k):
        """Inset floor halfspaces for a complete 3 mm radius, 6 mm high sole."""
        b,o=self.bases[k],self.offsets[k];window=self.navigation_window
        lo=np.asarray(window['min_m'])[:2]+.00301;hi=np.asarray(window['max_m'])[:2]-.00301
        vertices=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
        for bj,oj in zip(self.bases,self.offsets):
            normal=bj[2];a=b[:2]@normal
            rhs=float((oj-o)@normal+.00301*np.linalg.norm(a)-min(0.,.006*(b[2]@normal)))
            if np.linalg.norm(a)<1e-12:
                if rhs>1e-9:raise RuntimeError('No ground plane compatible with all installed floors')
                continue
            clipped=[]
            for first,last in zip(vertices,np.roll(vertices,-1,axis=0)):
                f=float(first@a-rhs);v=float(last@a-rhs)
                if f>=-1e-12:clipped.append(first)
                if (f>=0)!=(v>=0):clipped.append(first+(last-first)*f/(f-v))
            vertices=np.asarray(clipped).reshape(-1,2)
            if len(vertices)<3:raise RuntimeError('No full-thickness ground center region')
        return Polygon(vertices)

    def compact_floor_regions(self,k,region):
        b,o=self.bases[k],self.offsets[k]
        for margin in (0.,.012,.048):
            vertices=np.asarray(region.exterior.coords)[:-1]
            for bj,oj in zip(self.bases,self.offsets):
                for axis,normal in enumerate(bj):
                    a=b[:2]@normal;nz=float(b[2]@normal)
                    lower=self.occupied_lo[axis]-margin+normal@(oj-o)+.00301*np.linalg.norm(a)-min(0.,.006*nz)
                    upper=self.occupied_hi[axis]+margin+normal@(oj-o)-.00301*np.linalg.norm(a)-max(0.,.006*nz)
                    for direction,rhs in ((a,lower),(-a,-upper)):
                        if not len(vertices):break
                        clipped=[]
                        for first,last in zip(vertices,np.roll(vertices,-1,axis=0)):
                            f=float(first@direction-rhs);v=float(last@direction-rhs)
                            if f>=-1e-12:clipped.append(first)
                            if (f>=0)!=(v>=0):clipped.append(first+(last-first)*f/(f-v))
                        vertices=np.asarray(clipped).reshape(-1,2)
            if len(vertices)>=3:
                polygon=Polygon(vertices)
                if polygon.is_valid and polygon.area>1e-12:yield polygon

    def foot_pool(self,k):
        if k in self.ground_candidates:return self.ground_candidates[k]
        region=self.center_region(k);demands=self.required_vertices[k]
        required=self.required_hulls[k]
        # Eight extreme directions give a small deterministic proposal set.
        directions=np.c_[np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)]
        anchors=list(np.unique(demands[np.argmax(demands@directions.T,axis=0)],axis=0))
        anchors.append(demands.mean(axis=0))
        polygon=np.asarray(region.exterior.coords)[:-1]
        positions=list(polygon)
        for compact in self.compact_floor_regions(k,region):
            boundary=np.asarray(compact.exterior.coords)[:-1]
            positions.extend(boundary)
            positions.extend(np.asarray(nearest_points(compact,Point(q))[0].coords[0]) for q in anchors)
            for first,last in zip(boundary,np.roll(boundary,-1,axis=0)):
                positions.append((first+last)/2)
        for q in anchors:
            q=np.asarray(nearest_points(region,Point(q))[0].coords[0])
            positions.append(q)
            for radius in (.006,.024,.048):
                for angle in np.arange(8)*2*np.pi/8:
                    trial=q+radius*np.array([np.cos(angle),np.sin(angle)])
                    positions.append(np.asarray(nearest_points(region,Point(trial))[0].coords[0]))
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            positions.extend(a+t*(b-a) for t in (.25,.5,.75))
        unique={tuple(np.round(p,12)):p for p in positions}
        rows=[];b,o=self.bases[k],self.offsets[k]
        disk=.003*np.c_[np.cos(np.arange(32)*2*np.pi/32),np.sin(np.arange(32)*2*np.pi/32)]
        for key,xy in sorted(unique.items()):
            circle=xy+disk
            vertices=np.vstack([np.c_[circle,np.zeros(32)],np.c_[circle,np.full(32,.006)]])@b+o
            # Cheap floor rejection precedes any Boolean operation.
            if any(np.min(((vertices-oj)@bj.T)[:,2]) < -1e-10 for bj,oj in zip(self.bases,self.offsets)):continue
            start=np.r_[xy,.003]@b+o
            world=np.vstack([(np.vstack([vertices,start+self.bead])-oj)@bj.T for bj,oj in zip(self.bases,self.offsets)])
            rows.append(dict(xy=xy,footprint=circle,vertices=vertices,start=start,lo=world.min(axis=0),hi=world.max(axis=0)))
        if not rows:raise RuntimeError('No legal ground growth candidates: '+self.case.poses[k])
        print("GROUND CANDIDATES",self.case.poses[k],len(rows),flush=True)
        self.ground_candidates[k]=rows
        return rows

    def legal_foot(self,row):
        # Only candidates about to be used incur exact solid checks.
        if 'legal' not in row:
            row['sole']=S.solid(G.hull_mesh(row['vertices']))
            row['ball']=self.sphere(row['start'])
            row['legal']=self.legal(row['sole']) and self.legal(row['ball'])
        return row['legal']

    def floor_points(self,solid,k):
        mesh=S.unpack(solid);world=(mesh.vertices-self.offsets[k])@self.bases[k].T
        return world[np.abs(world[:,2])<1e-9,:2]

    def coverage(self,points,k):
        required=self.required_hulls[k]
        if not len(points):return False,float('inf'),required.area
        points=np.unique(points,axis=0)
        if len(points)<3 or np.linalg.matrix_rank(points-points[0],tol=1e-12)<2:
            hull=MultiPoint(points).convex_hull
            distance=sum(Point(p).distance(hull) for p in self.required_vertices[k])
            return distance<=1e-10,float(distance),required.area
        hull=ConvexHull(points)
        distances=self.required_vertices[k]@hull.equations[:,:2].T+hull.equations[:,2]
        violations=np.maximum(0.,distances.max(axis=1))
        done=bool(np.max(violations)<=1e-10)
        return done,float(violations.sum()),0. if done else float('nan')

    def branch_ends(self,point):
        ends=np.asarray(self.seed_positions)
        if self.segments:
            segments=np.asarray(self.segments);a=segments[:,0];delta=segments[:,1]-a
            denom=np.einsum('ij,ij->i',delta,delta)
            good=denom>1e-20
            fraction=np.clip(np.einsum('ij,ij->i',point-a[good],delta[good])/denom[good],0.,1.)
            ends=np.vstack([ends,a[good]+fraction[:,None]*delta[good]])
        return np.unique(ends,axis=0)

    def connection_scores(self,row,ends):
        transformed=np.array([(ends-o)@b.T for b,o in zip(self.bases,self.offsets)])
        bead_lo=np.array([(self.bead@b.T).min(axis=0) for b in self.bases])
        bead_hi=np.array([(self.bead@b.T).max(axis=0) for b in self.bases])
        lo=np.minimum(np.minimum(self.occupied_lo,row['lo']),(transformed+bead_lo[:,None,:]).min(axis=0))
        hi=np.maximum(np.maximum(self.occupied_hi,row['hi']),(transformed+bead_hi[:,None,:]).max(axis=0))
        volumes=np.prod(hi-lo,axis=1)-np.prod(self.occupied_hi-self.occupied_lo)
        return volumes,np.linalg.norm(ends-row['start'],axis=1)

    def body_increment(self,vertices):
        world=np.vstack([(vertices-o)@b.T for b,o in zip(self.bases,self.offsets)])
        return float(np.prod(np.maximum(self.occupied_hi,world.max(axis=0))-np.minimum(self.occupied_lo,world.min(axis=0)))-np.prod(self.occupied_hi-self.occupied_lo))

    def graph(self,pitch):
        # Sample a navigation graph only. Exact clearance is lazy on used edges.
        L.Grow.graph(self,pitch)
        self.grid_ready=True;self.active_grid_pitch=pitch
        self.visible_cache={};self.edge_checks={};self.blocked_edges=0
        self.original_edge_weights=self.graph_data.data.copy();self.weight_box=None

    def weight_navigation(self):
        key=(tuple(self.occupied_lo),tuple(self.occupied_hi))
        if key==self.weight_box:return
        lo=np.broadcast_to(self.occupied_lo,(len(self.nodes),3)).copy()
        hi=np.broadcast_to(self.occupied_hi,(len(self.nodes),3)).copy()
        for b,o in zip(self.bases,self.offsets):
            points=(self.nodes-o)@b.T;bead=self.bead@b.T
            lo=np.minimum(lo,points+bead.min(axis=0));hi=np.maximum(hi,points+bead.max(axis=0))
        current=float(np.prod(self.occupied_hi-self.occupied_lo))
        penalty=1.+100.*np.maximum(0.,np.prod(hi-lo,axis=1)/current-1.)
        rows=np.repeat(np.arange(len(self.nodes)),np.diff(self.graph_data.indptr))
        active=np.isfinite(self.graph_data.data)
        weights=self.original_edge_weights*.5*(penalty[rows]+penalty[self.graph_data.indices])
        self.graph_data.data[active]=weights[active];self.weight_box=key

    def route_on_grid(self,target,network):
        self.weight_navigation()
        def visible(point):
            key=tuple(point)
            if key not in self.visible_cache:
                _,near=self.tree.query(point,k=min(256,len(self.nodes)))
                self.visible_cache[key]=next((int(node) for node in np.atleast_1d(near)
                    if self.legal(self.rod(point,self.nodes[node]))),None)
            return self.visible_cache[key]
        source=self.seed_positions[target];begin=visible(source)
        ends=[(node,j) for j in sorted(network) if (node:=visible(self.seed_positions[j])) is not None]
        if begin is None or not ends:raise RuntimeError('No legal lazy navigation attachment')
        for attempt in range(64):
            distances,pred=dijkstra(self.graph_data,directed=False,indices=begin,return_predecessors=True)
            end,owned=min(ends,key=lambda row:(distances[row[0]],row))
            if not np.isfinite(distances[end]):raise RuntimeError('Lazy navigation exhausted')
            chain=[end]
            while chain[-1]!=begin:
                node=int(pred[chain[-1]])
                if node<0:raise RuntimeError('Invalid lazy navigation predecessor')
                chain.append(node)
            chain=chain[::-1];invalid=[]
            for a,b in zip(chain,chain[1:]):
                key=tuple(sorted((a,b)))
                if key not in self.edge_checks:self.edge_checks[key]=self.legal(self.rod(self.nodes[a],self.nodes[b]))
                if not self.edge_checks[key]:invalid.append((a,b))
            if not invalid:break
            # Reject failed edges and search again; unchanged exact solid policy.
            for a,b in invalid:
                for first,last in ((a,b),(b,a)):
                    lo,hi=self.graph_data.indptr[first:first+2]
                    rows=np.flatnonzero(self.graph_data.indices[lo:hi]==last)+lo
                    self.graph_data.data[rows]=np.inf
                self.blocked_edges+=1
        else:raise RuntimeError('Lazy navigation retry limit exceeded')
        points=[source]+list(self.nodes[chain])+[self.seed_positions[owned]]
        route=[points[0]];index=0
        while index<len(points)-1:
            for last in range(len(points)-1,index,-1):
                if self.legal(self.rod(points[index],points[last])):break
            else:raise RuntimeError('No legal full-rod route shortcut')
            route.append(points[last]);index=last
        return route

    def fallback(self,target,network):
        if not self.grid_ready:self.graph(self.pitch)
        try:return self.route_on_grid(target,network)
        except RuntimeError:
            if self.active_grid_pitch<=.004:raise
            self.graph(.004)
            return self.route_on_grid(target,network)

    def connect_heads(self):
        self.initialize_occupancy()
        targets=set(range(len(self.terminals)));network={0};beams=[];direct=0;fallback=0;projected=0
        while targets-network:
            chosen=None
            for _,_,target,_,end in self.candidates(targets-network,network):
                body=self.rod(self.seed_positions[target],end)
                if self.legal(body):chosen=(target,end,body);break
            if chosen is not None:
                target,end,body=chosen;start=self.seed_positions[target]
                route=[start,end];parts=[body];direct+=1
                if not any(np.array_equal(end,self.seed_positions[j]) for j in network):projected+=1
            else:
                # Keep the first feasible detour instead of optimizing all routes.
                order=sorted(targets-network,key=lambda j:(min(self.volume_key([self.seed_positions[j],self.seed_positions[n]]) for n in network),j))
                for target in order:
                    try:route=self.fallback(target,network);break
                    except RuntimeError:continue
                else:raise RuntimeError('No head-grown route in finite navigation window')
                parts=[self.rod(a,b) for a,b in zip(route,route[1:])];fallback+=1
            beams.extend(parts);network.add(target);self.include_path(route)
            self.segments.extend((np.asarray(a),np.asarray(b)) for a,b in zip(route,route[1:]))
            self.paths.append([p.tolist() for p in route])
            self.journal.append(dict(step=len(self.journal)+1,target=self.terminals[target]['name'],target_kind=self.terminals[target]['kind'],
                start_network_point_m=route[-1].tolist(),path_m=[p.tolist() for p in route[::-1]],core_segments=len(parts)))
        full=F.union([t['solid'] for t in self.terminals]+beams)
        if len(L.components(full))!=1:raise RuntimeError('Head-grown fixture disconnected')
        self.beams=beams
        return full,dict(terminal_count=len(self.terminals),grown_segment_count=len(beams),
            direct_connection_count=direct,fallback_route_count=fallback,projection_shared_joint_count=projected,
            lazy_grid_constructed=self.grid_ready,preset_skeleton=False,preset_grid_anchor=False,
            initial_local_loft_to_grid=False,initial_head_name=self.terminals[0]['name'])

    def volume_bin(self,increment):
        extent=self.occupied_hi-self.occupied_lo
        # A 1 mm change on six box faces sets the ranking resolution only.
        quantum=.002*(extent[0]*extent[1]+extent[0]*extent[2]+extent[1]*extent[2])
        return int(np.floor(max(0.,increment)/quantum+.5))

    def material_floor_points(self,vertices,k):
        world=(vertices-self.offsets[k])@self.bases[k].T
        return world[np.abs(world[:,2])<1e-9,:2]

    def reduce_floor(self,points):
        points=np.unique(points,axis=0)
        if len(points)>=3 and np.linalg.matrix_rank(points-points[0],tol=1e-12)==2:
            return points[ConvexHull(points).vertices]
        return points

    def grow_ground(self,full):
        head_full=full;head_beams=list(self.beams)
        head_floor=[self.floor_points(full,k) for k in range(len(self.case.poses))]
        floors=[self.reduce_floor(p) for p in head_floor]
        direct=0;routed=0
        def assemble(records):
            # One flat union preserves primitive cores at coincident junctions.
            return F.union([t['solid'] for t in self.terminals]+head_beams+
                [solid for r in records for solid in [r['row']['sole'],r['row']['ball']]+r['parts']])
        for k in range(len(self.case.poses)):
            pool=self.foot_pool(k);used=set();count=0
            while True:
                floor=floors[k];done,loss,_=self.coverage(floor,k)
                if done:break
                proposals=[]
                for index,row in enumerate(pool):
                    if index in used or row.get('legal') is False:continue
                    _,next_loss,_=self.coverage(np.vstack([floor,row['footprint']]),k)
                    gain=loss-next_loss if np.isfinite(loss) else 1.
                    if gain>1e-9:proposals.append((gain,index,row))
                if not proposals:raise RuntimeError('Candidate ground coverage exhausted: '+self.case.poses[k])
                # Favor large coverage gains; volume only ranks these useful proposals.
                threshold=max(p[0] for p in proposals)*.75
                candidates=[]
                for gain,index,row in proposals:
                    if gain<threshold:continue
                    ends=self.branch_ends(row['start']);volumes,lengths=self.connection_scores(row,ends)
                    order=sorted(range(len(ends)),key=lambda j:(self.volume_bin(float(volumes[j])),float(lengths[j]),tuple(ends[j])))
                    # Four nearby attachment choices suffice for the quick pass.
                    for j in order[:4]:
                        candidates.append((self.volume_bin(float(volumes[j])),-gain,float(lengths[j]),index,tuple(ends[j]),row,ends[j]))
                selected=None
                for _,_,_,index,_,row,end in sorted(candidates,key=lambda v:v[:5]):
                    if not self.legal_foot(row):continue
                    rod=self.rod(row['start'],end)
                    if self.legal(rod):selected=(index,row,[row['start'],end],[rod]);break
                if selected is None:
                    short=sorted(proposals,key=lambda p:(self.volume_bin(self.body_increment(p[2]['vertices'])),-p[0],p[1]))
                    for _,index,row in short:
                        if not self.legal_foot(row):continue
                        target=len(self.seed_positions);self.seed_positions.append(row['start'])
                        try:route=self.fallback(target,set(range(target)))
                        except RuntimeError:continue
                        finally:self.seed_positions.pop()
                        parts=[self.rod(a,b) for a,b in zip(route,route[1:])]
                        selected=(index,row,route,parts);routed+=1;break
                    if selected is None:raise RuntimeError('No legal branch reaches useful ground: '+self.case.poses[k])
                else:direct+=1
                index,row,route,parts=selected;used.add(index);count+=1
                if count>32:raise RuntimeError('Ground coverage progress limit exceeded')
                name=f'{self.case.poses[k]}:grown_foot{count}'
                vertices=np.vstack([row['vertices'],row['start']+self.bead]+[S.unpack(part).vertices for part in parts])
                floor_parts=[self.material_floor_points(vertices,j) for j in range(len(floors))]
                # Conservative branch dependencies allow cheap leaf pruning.
                dependencies=[]
                for record in self.foot_records:
                    for a,b in zip(record['route'],record['route'][1:]):
                        delta=b-a;fraction=np.clip((route[-1]-a)@delta/(delta@delta),0.,1.) if delta@delta>1e-20 else 0.
                        if np.linalg.norm(route[-1]-(a+fraction*delta))<=.0061:
                            dependencies.append(record['name']);break
                self.foot_records.append(dict(pose_index=k,name=name,row=row,parts=parts,route=np.asarray(route),
                    floor_points=floor_parts,dependencies=dependencies,step=len(self.journal)+1))
                self.original_soles.append(row['sole']);self.core_solids.extend([(name+':sole',row['sole']),(name+':start_ball',row['ball'])])
                self.seed_positions.append(row['start']);self.generated_ground.append((k,row['xy']))
                for a,b in zip(route,route[1:]):self.segments.append((np.asarray(a),np.asarray(b)))
                self.include_path(route)
                world=np.vstack([(row['vertices']-o)@b.T for b,o in zip(self.bases,self.offsets)])
                self.occupied_lo=np.minimum(self.occupied_lo,world.min(axis=0));self.occupied_hi=np.maximum(self.occupied_hi,world.max(axis=0))
                self.paths.append([p.tolist() for p in route])
                self.journal.append(dict(step=len(self.journal)+1,target=name,target_kind='generated_ground_contact',start_network_point_m=route[-1].tolist(),path_m=[p.tolist() for p in route[::-1]],core_segments=len(parts)))
                print('GROUND GROW',self.case.poses[k],count,'xy',row['xy'].tolist(),flush=True)
                self.feet[k]['centers_xy_m'].append(row['xy'].tolist());self.feet[k]['creation_steps'].append(len(self.journal))
                floors=[self.reduce_floor(np.vstack([points,added])) for points,added in zip(floors,floor_parts)]
            if k==max(0,len(self.case.poses)//2-1):
                self.save_stage('partial_growth',assemble(self.foot_records));self.partial_steps=len(self.journal)
        active=list(self.foot_records);removed=[]
        # Test coverage using part vertices, avoiding repeated whole-body Booleans.
        for record in reversed(self.foot_records):
            if any(record['name'] in r['dependencies'] for r in active if r is not record):continue
            trial_records=[r for r in active if r is not record]
            if not all(self.coverage(np.vstack([head_floor[j]]+[r['floor_points'][j] for r in trial_records]),j)[0] for j in range(len(floors))):continue
            active=trial_records;removed.append(record['name'])
        full=assemble(active)
        if len(L.components(full))!=1:
            # A conservative numerical recovery keeps all accepted branches.
            active=list(self.foot_records);removed=[];full=assemble(active)
        active_names={r['name'] for r in active}
        self.core_solids=[(name,body) for name,body in self.core_solids if ':grown_foot' not in name or name.rsplit(':',1)[0] in active_names]
        self.original_soles=[r['row']['sole'] for r in active]
        self.beams=head_beams+[body for r in active for body in r['parts']]
        self.generated_ground=[(r['pose_index'],r['row']['xy']) for r in active]
        self.feet=[dict(pose=pose,centers_xy_m=[r['row']['xy'].tolist() for r in active if r['pose_index']==k],
            creation_steps=[r['step'] for r in active if r['pose_index']==k]) for k,pose in enumerate(self.case.poses)]
        self.removed_ground_branches=removed
        if len(L.components(full))!=1:raise RuntimeError('Coverage-grown support disconnected')
        self.save_stage('shared_tree',full)
        return full,direct,routed

    def connect(self):
        started=time.monotonic();full,info=self.connect_heads()
        self.timings=getattr(self,'timings',{})
        self.timings['head_network']=time.monotonic()-started
        started=time.monotonic();full,direct,routed=self.grow_ground(full)
        self.timings['ground_growth']=time.monotonic()-started
        info.update(grown_segment_count=len(self.beams),direct_connection_count=info['direct_connection_count']+direct,
            fallback_route_count=info['fallback_route_count']+routed,growth_journal=self.journal,partial_stage_step_count=self.partial_steps,
            fixed_ground_targets=False,ground_volume_ranking_resolution_mm=1.,ground_requirement='actual material ground hull covers original demand hull in every pose',
            generated_ground_contact_count=len(self.generated_ground),redundant_ground_branches_removed=self.removed_ground_branches,ground_candidate_source='demand neighborhoods, current-envelope ground sections and all-pose floor halfspaces; no previous support footprint',
            routing='greedy shared head growth; coarse lazy routing; useful coverage contacts with on-demand solid checks',
            ground_proposal_count=sum(map(len,self.ground_candidates.values())),
            ground_exact_solid_checks=sum('legal' in r for rows in self.ground_candidates.values() for r in rows),
            ground_minimum_relative_gain=.75,ground_attachment_choices=4,
            active_grid_pitch_m=getattr(self,'active_grid_pitch',None),
            navigation_edge_checks=len(getattr(self,'edge_checks',{})),
            rejected_navigation_edges=getattr(self,'blocked_edges',0),
            buried_contact_rejections=getattr(self,'buried_contact_rejections',0))
        self.final_solid=full
        return full,info

    def verify(self,mesh):
        started=time.monotonic();result=super().verify(mesh)
        self.timings.setdefault('validation_passes',[]).append(time.monotonic()-started)
        return result

    def run(self,pitch=.008):
        report=super().run(pitch)
        report['timing_seconds']=dict(self.timings,run_with_internal_validation=report['seconds'],
            construction=sum(self.timings.get(key,0.) for key in ('contact_starts','head_network','ground_growth')))
        report['timing_seconds']['setup_and_run']=self.timings.get('setup',0.)+report['seconds']
        report['previous_material_volume_cm3']=float(self.foot_reference.volume*1e6)
        report['material_reduction_percent']=100*(1-report['volume_cm3']/report['previous_material_volume_cm3'])
        report['comparison_reference']='boxed_support/shape.obj; not the previously displayed growing model'
        report['construction']['ground_candidate_order']=['volume_increment_at_1mm_resolution','coverage_gain_descending','branch_length','fixed_geometric_ties']
        report['construction'].update(method='Fast head growth with adaptive ground coverage; no prescribed foot targets',reference_used_as='saved placement only')
        report['objective']='Fast deterministic coverage growth with compact-volume preference; no global optimization'
        report['provenance']['code'].update(I.hashes([Path(__file__)]))
        I.save(self.out/'report.json',report)
        return report
