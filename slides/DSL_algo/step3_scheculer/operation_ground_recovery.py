"""Broaden demand-neighborhood proposals when baseline greedy growth stalls."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points
from step3_scheculer import operation_dsl as F,operation_growth_recovery as R
from step3_scheculer.run_dsl import saved_task


class GroundGrow(R.RecoveryGrow):
    def foot_pool(self,k):
        rows=super().foot_pool(k)
        if getattr(self,'dense_floor_done',set()).__contains__(k):return rows
        if not hasattr(self,'dense_floor_done'):self.dense_floor_done=set()
        self.dense_floor_done.add(k)
        region=self.center_region(k);b,o=self.bases[k],self.offsets[k]
        directions=np.c_[np.cos(np.arange(32)*np.pi/16),np.sin(np.arange(32)*np.pi/16)]
        disk=.003*np.c_[np.cos(np.arange(32)*np.pi/16),np.sin(np.arange(32)*np.pi/16)]
        positions={tuple(np.round(row['xy'],12)) for row in rows}
        candidates=[]
        for anchor in self.required_vertices[k]:
            for radius in (.006,.012,.018,.03,.06):
                for direction in directions:candidates.append(np.asarray(nearest_points(region,Point(anchor+radius*direction))[0].coords[0]))
        # Actual free-space sections provide obstacle-boundary candidates that
        # radial demand samples can miss. These are proposals, not certificates.
        from shapely.geometry import Polygon,GeometryCollection
        native_mask=F.F.transform(self.mask,b.T,-o@b.T)
        free=None
        for height in (1e-8,.0015,.003,.0045,.006):
            section=GeometryCollection()
            for contour in native_mask.slice(height/F.S.SCALE).to_polygons():
                polygon=Polygon(np.asarray(contour)*F.S.SCALE)
                if not polygon.is_valid:polygon=polygon.buffer(0)
                section=section.symmetric_difference(polygon)
            free=section if free is None else free.intersection(section)
        centers=free.buffer(-.00301,resolution=16).intersection(region)
        components=list(centers.geoms) if hasattr(centers,'geoms') else [centers]
        for component in components:
            if component.is_empty or component.geom_type!='Polygon':continue
            candidates.append(np.asarray(component.representative_point().coords[0]))
            candidates.extend(np.asarray(nearest_points(component,Point(anchor))[0].coords[0]) for anchor in self.required_vertices[k])
            for ring in [component.exterior]+list(component.interiors):
                boundary=np.asarray(ring.coords)[:-1]
                candidates.extend(boundary[::max(1,len(boundary)//128)])
                candidates.extend(np.unique(boundary[np.argmax(boundary@directions.T,axis=0)],axis=0))
        for xy in candidates:
            key=tuple(np.round(xy,12))
            if key in positions:continue
            positions.add(key);circle=xy+disk
            vertices=np.vstack([np.c_[circle,np.zeros(32)],np.c_[circle,np.full(32,.006)]])@b+o
            if any(((vertices-oj)@bj.T)[:,2].min() < -1e-10 for bj,oj in zip(self.bases,self.offsets)):continue
            start=np.r_[xy,.003]@b+o
            native=np.vstack([(np.vstack([vertices,start+self.bead])-oj)@bj.T for bj,oj in zip(self.bases,self.offsets)])
            rows.append(dict(xy=xy,footprint=circle,vertices=vertices,start=start,lo=native.min(axis=0),hi=native.max(axis=0)))
        print('DENSE GROUND CANDIDATES',self.group.name,self.case.poses[k],len(rows),flush=True)
        return rows

    def ground_actions(self,state):
        proposals=super().ground_actions(state)
        if proposals or self.direct_only:return proposals
        # Consider each exposed hull vertex and a broader route shortlist.
        for k,points in enumerate(state['floors']):
            old=self.distances(points,self.required_vertices[k]);old_loss=float(old.sum())
            if np.max(old)<=1e-10:continue
            candidates=[]
            for index,row in enumerate(self.ground_candidates[k]):
                if (k,index) in state['feet_used'] or row.get('legal') is False:continue
                trial=self.distances(np.vstack([points,row['footprint']]),self.required_vertices[k]);gain=old_loss-float(trial.sum())
                if gain<=1e-12:continue
                nearest=self.ports(state,row['start'])[0][0]
                candidates.append((-gain,float(np.linalg.norm(row['start']-nearest)),index))
            for _,_,index in sorted(candidates)[:32]:
                row=self.ground_candidates[k][index]
                if not self.legal_foot(row):continue
                for end,owner in self.compact_ports(state,row['start'],row)[:12]:
                    action=self.action(state,'ground',[row['start'],end],owner,owner,k,index,row)
                    if action is not None:return [action]
                try:route,owner=self.routed(state,row['start'])
                except RuntimeError:continue
                action=self.action(state,'ground',route,owner,owner,k,index,row)
                if action is not None:return [action]
        print('STALLED GROUND',self.group.name,'components',len(set(state['labels'])),
            'max_missing_mm',[float(self.distances(p,self.required_vertices[k]).max()*1000) for k,p in enumerate(state['floors'])],flush=True)
        return []


def activate():
    R.activate();original_sources=F.sources
    F.Grow=GroundGrow
    F.sources=lambda:list(dict.fromkeys(original_sources()+[Path(__file__)]))


def main():
    activate();group=F.I.OUTPUTS/'B'/sys.argv[1]
    poses=['pose_'+v for v in group.name.removeprefix('pose').removesuffix('copied').split('+')]
    optimizer=F.Optimizer(group,[saved_task(group,p) for p in poses],dict(device='cuda',steps=3,samples=24,init_seeds=96))
    optimizer.optimize()
    print('DENSE RECOVERY DONE',group.name,optimizer.incumbent.physical_count,flush=True)

if __name__=='__main__':main()
