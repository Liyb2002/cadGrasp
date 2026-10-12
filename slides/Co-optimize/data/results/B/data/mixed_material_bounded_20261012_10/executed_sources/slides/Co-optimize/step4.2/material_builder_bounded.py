"""Plan compact solids inside all installed-state boxes and the legal witness."""
import numpy as np
from material_builder_closed import Builder as ClosedBuilder,backend


def state_boxes(fixture,rows):
    groups={}
    for row in rows:
        T=np.asarray(row['fixture_transform']);key=np.round(T,11).tobytes()
        if key not in groups:groups[key]=[T,[backend.transform(fixture.vertices,T)]]
        groups[key][1].append(np.asarray(row['hull']))
        if 'object_vertices' in row:groups[key][1].append(np.asarray(row['object_vertices']))
    return [(T,np.vstack(clouds).min(0),np.vstack(clouds).max(0))for T,clouds in groups.values()]


def complete_box_cost(mesh,rows):
    groups={}
    for row in rows:
        T=np.asarray(row['fixture_transform']);key=np.round(T,11).tobytes()
        if key not in groups:groups[key]=[backend.transform(mesh.vertices,T)]
        if 'object_vertices' in row:groups[key].append(np.asarray(row['object_vertices']))
    clouds=[np.vstack(v)for v in groups.values()]
    xy=np.max(np.array([np.ptp(p[:,:2],axis=0)for p in clouds]),axis=0)
    z=max(float(p[:,2].max())for p in clouds)-min(float(p[:,2].min())for p in clouds)
    return float(xy.prod()*z*1000),abs(float(mesh.volume))*1e6


class Builder:
    def __init__(self,fixture,witness,rows,width=backend.WIDTH):
        self.fixture=fixture;self.witness=witness;self.rows=rows;self.width=width

    def run(self):
        allowed=backend.solid(self.witness);original=backend.solid(self.fixture)
        legal=(original if backend.volume(original-allowed)<=backend.VOLUME_TOL else original^allowed)
        boxes=state_boxes(self.fixture,self.rows);plans=[];trials=[];last_error=None
        for margin in [.005,.02,.05,.1,None]:
            domain=allowed
            if margin is not None:
                for T,low,high in boxes:
                    for axis in range(3):
                        domain=domain.trim_by_plane(tuple(T[axis,:3]),float((low[axis]-margin-T[axis,3])/backend.SCALE))
                        domain=domain.trim_by_plane(tuple(-T[axis,:3]),float((T[axis,3]-high[axis]-margin)/backend.SCALE))
            row=dict(world_box_margin_m=margin,passed=False)
            try:
                if backend.volume(domain)<=backend.VOLUME_TOL:raise ValueError('Empty construction window')
                window=backend.unpack(domain)
                if not all(r['passed']for r in backend.enclosure(window,self.rows)):
                    raise ValueError('No pressure enclosure in the construction window')
                mesh,record=ClosedBuilder(self.fixture,window,self.rows,width=self.width).run()
                body=backend.solid(mesh);outside=backend.volume(body-allowed);retained=backend.volume(legal-body)
                accepted=record['passed'] and record['watertight'] and outside<=1e-10 and retained<=1e-10
                row.update(passed=accepted,cost=complete_box_cost(mesh,self.rows),
                    outside_raw_witness_cm3=outside*1e6,retained_legal_nominal_loss_cm3=retained*1e6)
                if accepted:
                    record.update(outside_saved_legal_component_cm3=outside*1e6,
                        retained_legal_step4_material_loss_cm3=retained*1e6,
                        construction_world_box_margin_m=margin)
                    plans.append((row['cost'],mesh,record))
            except (ValueError,RuntimeError)as error:row['error']=str(error);last_error=error
            trials.append(row)
        if not plans:raise ValueError('No certified solid in bounded or full legal space: '+str(last_error))
        cost,mesh,record=min(plans,key=lambda x:x[0]);record['material_plan_trials']=trials
        record['material_plan_cost']=cost
        return mesh,record
