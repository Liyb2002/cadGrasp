"""Grow ground contacts on demand until their material hull covers requirements.

Ground requirements are constraints, not mandatory preselected foot centers.
"""
from pathlib import Path
import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import nearest_points
from step4_connect_support.direct_head_growth import DirectGrow
from step4_connect_support import boxed_support as F,build_coupled_saddle as S
from step4_connect_support import deterministic_space as D,growing_support as L
from step2_local_support import geometry as G
from step3_scheculer import contacts as I

class CoverageGrow(DirectGrow):
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
        for margin in (0.,.003,.006,.012,.024,.048):
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
        anchors=list(np.asarray(required.exterior.coords)[:-1]) if required.geom_type=='Polygon' else list(demands)
        anchors.append(demands.mean(axis=0))
        polygon=np.asarray(region.exterior.coords)[:-1]
        positions=list(polygon)
        for compact in self.compact_floor_regions(k,region):
            boundary=np.asarray(compact.exterior.coords)[:-1]
            positions.extend(boundary)
            positions.extend(np.asarray(nearest_points(compact,Point(q))[0].coords[0]) for q in anchors)
            for first,last in zip(boundary,np.roll(boundary,-1,axis=0)):
                positions.extend(first+t*(last-first) for t in (.25,.5,.75))
        for q in anchors:
            q=np.asarray(nearest_points(region,Point(q))[0].coords[0])
            positions.append(q)
            for radius in (.003,.006,.012,.024,.048):
                for angle in np.arange(16)*2*np.pi/16:
                    trial=q+radius*np.array([np.cos(angle),np.sin(angle)])
                    positions.append(np.asarray(nearest_points(region,Point(trial))[0].coords[0]))
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            positions.extend(a+t*(b-a) for t in np.linspace(0,1,max(2,int(np.linalg.norm(b-a)/.012)+1)))
        unique={tuple(np.round(p,12)):p for p in positions}
        rows=[];b,o=self.bases[k],self.offsets[k]
        disk=.003*np.c_[np.cos(np.arange(32)*2*np.pi/32),np.sin(np.arange(32)*2*np.pi/32)]
        for key,xy in sorted(unique.items()):
            circle=xy+disk
            vertices=np.vstack([np.c_[circle,np.zeros(32)],np.c_[circle,np.full(32,.006)]])@b+o
            # Cheap floor rejection precedes any Boolean operation.
            if any(np.min(((vertices-oj)@bj.T)[:,2]) < -1e-10 for bj,oj in zip(self.bases,self.offsets)):continue
            sole=S.solid(G.hull_mesh(vertices));start=np.r_[xy,.003]@b+o
            ball=self.sphere(start)
            if not self.legal(sole) or not self.legal(ball):continue
            world=np.vstack([(np.vstack([vertices,start+self.bead])-oj)@bj.T for bj,oj in zip(self.bases,self.offsets)])
            rows.append(dict(xy=xy,footprint=circle,vertices=vertices,start=start,sole=sole,ball=ball,lo=world.min(axis=0),hi=world.max(axis=0)))
        if not rows:raise RuntimeError('No legal ground growth candidates: '+self.case.poses[k])
        print("GROUND CANDIDATES",self.case.poses[k],len(rows),flush=True)
        self.ground_candidates[k]=rows
        return rows

    def floor_points(self,solid,k):
        mesh=S.unpack(solid);world=(mesh.vertices-self.offsets[k])@self.bases[k].T
        return world[np.abs(world[:,2])<1e-9,:2]

    def coverage(self,points,k):
        required=self.required_hulls[k]
        if not len(points):return False,float('inf'),required.area
        hull=MultiPoint(points).convex_hull
        missing=float(required.difference(hull.buffer(1e-10)).area)
        distance=sum(Point(p).distance(hull) for p in self.required_vertices[k])
        return missing<=1e-12,float(distance+missing/.1),missing

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

    def fallback(self,target,network):
        if not self.grid_ready:self.graph(self.pitch)
        # Penalize excursions beyond the current occupied box during routing.
        # Final route selection still uses its exact aggregate box increment.
        lo=np.broadcast_to(self.occupied_lo,(len(self.nodes),3)).copy()
        hi=np.broadcast_to(self.occupied_hi,(len(self.nodes),3)).copy()
        for b,o in zip(self.bases,self.offsets):
            points=(self.nodes-o)@b.T
            bead=self.bead@b.T
            lo=np.minimum(lo,points+bead.min(axis=0));hi=np.maximum(hi,points+bead.max(axis=0))
        current=float(np.prod(self.occupied_hi-self.occupied_lo))
        penalty=1.+100.*np.maximum(0.,np.prod(hi-lo,axis=1)/current-1.)
        original=self.graph_data
        weighted=original.copy()
        rows=np.repeat(np.arange(len(self.nodes)),np.diff(weighted.indptr))
        weighted.data*=.5*(penalty[rows]+penalty[weighted.indices])
        self.graph_data=weighted
        try:return super().fallback(target,network)
        finally:self.graph_data=original

    def volume_bin(self,increment):
        extent=self.occupied_hi-self.occupied_lo
        # A 1 mm change on six box faces sets the ranking resolution only.
        quantum=.002*(extent[0]*extent[1]+extent[0]*extent[2]+extent[1]*extent[2])
        return int(np.floor(max(0.,increment)/quantum+.5))

    def grow_ground(self,full):
        head_full=full;head_beams=list(self.beams)
        beams=list(self.beams);direct=0;routed=0
        for k in range(len(self.case.poses)):
            pool=self.foot_pool(k);used=set();count=0
            while True:
                floor=self.floor_points(full,k);done,loss,_=self.coverage(floor,k)
                if done:break
                proposals=[]
                for index,row in enumerate(pool):
                    if index in used:continue
                    trial=np.vstack([floor,row['footprint']]);_,next_loss,_=self.coverage(trial,k)
                    gain=loss-next_loss if np.isfinite(loss) else 1.
                    if gain<=1e-9:continue
                    proposals.append((gain,index,row))
                if not proposals:raise RuntimeError('Candidate ground coverage exhausted: '+self.case.poses[k])
                # Demand meaningful progress, then minimize occupied volume.
                threshold=max(p[0] for p in proposals)*.25
                candidates=[]
                for gain,index,row in proposals:
                    if gain<threshold:continue
                    start=row['start']
                    ends=self.branch_ends(start);volumes,lengths=self.connection_scores(row,ends)
                    candidates.extend((self.volume_bin(float(volume)),-gain,float(length),index,tuple(end),row,end)
                        for volume,length,end in zip(volumes,lengths,ends))
                selected=None
                for _,_,_,index,_,row,end in sorted(candidates,key=lambda v:v[:5]):
                    rod=self.rod(row['start'],end)
                    if self.legal(rod):selected=(index,row,[row['start'],end],[rod]);break
                if selected is None:
                    short=sorted(proposals,key=lambda p:(self.body_increment(p[2]['vertices']),-p[0],p[1]))[:12]
                    routes=[]
                    for _,index,row in short:
                        target=len(self.seed_positions);self.seed_positions.append(row['start'])
                        try:route=self.fallback(target,set(range(target)))
                        except RuntimeError:continue
                        finally:self.seed_positions.pop()
                        vertices=np.vstack([row['vertices']]+[p+self.bead for p in route])
                        routes.append((self.volume_bin(self.body_increment(vertices)),sum(np.linalg.norm(b-a) for a,b in zip(route,route[1:])),index,row,route))
                    if not routes:raise RuntimeError('No legal branch reaches useful ground: '+self.case.poses[k])
                    _,_,index,row,route=min(routes,key=lambda v:v[:3]);parts=[self.rod(a,b) for a,b in zip(route,route[1:])]
                    selected=(index,row,route,parts);routed+=1
                else:direct+=1
                index,row,route,parts=selected;used.add(index);count+=1
                if count>32:raise RuntimeError('Ground coverage progress limit exceeded')
                name=f'{self.case.poses[k]}:grown_foot{count}'
                self.foot_records.append(dict(pose_index=k,name=name,row=row,parts=parts,step=len(self.journal)+1))
                self.original_soles.append(row['sole']);self.core_solids.extend([(name+':sole',row['sole']),(name+':start_ball',row['ball'])])
                self.seed_positions.append(row['start']);self.generated_ground.append((k,row['xy']))
                for a,b in zip(route,route[1:]):self.segments.append((np.asarray(a),np.asarray(b)))
                beams.extend(parts);self.include_path(route)
                world=np.vstack([(row['vertices']-o)@b.T for b,o in zip(self.bases,self.offsets)])
                self.occupied_lo=np.minimum(self.occupied_lo,world.min(axis=0));self.occupied_hi=np.maximum(self.occupied_hi,world.max(axis=0))
                self.paths.append([p.tolist() for p in route])
                self.journal.append(dict(step=len(self.journal)+1,target=name,target_kind='generated_ground_contact',start_network_point_m=route[-1].tolist(),path_m=[p.tolist() for p in route[::-1]],core_segments=len(parts)))
                print('GROUND GROW',self.case.poses[k],count,'xy',row['xy'].tolist(),flush=True)
                self.feet[k]['centers_xy_m'].append(row['xy'].tolist());self.feet[k]['creation_steps'].append(len(self.journal))
                full=F.union([t['solid'] for t in self.terminals]+beams+self.original_soles+[c for name,c in self.core_solids if name.endswith(':start_ball')])
            if k==max(0,len(self.case.poses)//2-1):
                self.save_stage('partial_growth',full);self.partial_steps=len(self.journal)
        active=list(self.foot_records)
        def assemble(records):
            return F.union([head_full]+[solid for r in records for solid in [r['row']['sole'],r['row']['ball']]+r['parts']])
        # Remove redundant generated branches by reconstruction, never carving.
        removed=[]
        for record in reversed(self.foot_records):
            trial_records=[r for r in active if r is not record]
            trial=assemble(trial_records)
            if len(L.components(trial))!=1:continue
            if not all(self.coverage(self.floor_points(trial,j),j)[0] for j in range(len(self.case.poses))):continue
            active=trial_records;removed.append(record['name'])
        full=assemble(active)
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
        full,info=super().connect()
        full,direct,routed=self.grow_ground(full)
        info.update(grown_segment_count=len(self.beams),direct_connection_count=info['direct_connection_count']+direct,
            fallback_route_count=info['fallback_route_count']+routed,growth_journal=self.journal,partial_stage_step_count=self.partial_steps,
            fixed_ground_targets=False,ground_volume_ranking_resolution_mm=1.,ground_requirement='actual material ground hull covers original demand hull in every pose',
            generated_ground_contact_count=len(self.generated_ground),redundant_ground_branches_removed=self.removed_ground_branches,ground_candidate_source='demand neighborhoods, current-envelope ground sections and all-pose floor halfspaces; no previous support footprint',
            routing='grow shared heads, then generate only ground contacts that improve unmet coverage')
        return full,info

    def run(self,pitch=.004):
        report=super().run(pitch)
        report['previous_material_volume_cm3']=float(self.foot_reference.volume*1e6)
        report['material_reduction_percent']=100*(1-report['volume_cm3']/report['previous_material_volume_cm3'])
        report['comparison_reference']='boxed_support/shape.obj; not the previously displayed growing model'
        report['construction']['ground_candidate_order']=['volume_increment_at_1mm_resolution','coverage_gain_descending','branch_length','fixed_geometric_ties']
        report['construction'].update(method='Head-grown network with adaptive ground coverage; no prescribed foot targets',reference_used_as='saved placement only')
        report['provenance']['code'].update(I.hashes([Path(__file__)]))
        I.save(self.out/'report.json',report)
        return report
