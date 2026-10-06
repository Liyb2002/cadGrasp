"""Find essential fragments by deletion, then guide path recovery toward them."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_topology_common import *

class ValueGuidedSearch(TopologySearch):
    def initialize_value(self):
        import step42_contact_consistency as context
        context._current_directions=self.warm
        force_module.contact_boundary=motion_consistent_boundary;connected_module.contact_boundary=motion_consistent_boundary
        r=Search.candidate(self,self.warm,label='value audit baseline',proxy=False)
        if r is None:return False
        parts=[p for p in r['remaining'].decompose() if material_volume(p)>1e-12];parts.sort(key=lambda p:material_volume(p))
        chunks=[]
        for p in parts:
            tri,src=motion_consistent_boundary(self.mesh,S.unpack(p),self.allowed);chunks.append(dict(part=p,tri=tri,src=src))
        active=list(range(len(chunks)));rows=[]
        for k in list(active):
            rest=[j for j in active if j!=k];tri=np.concatenate([chunks[j]['tri'] for j in rest]) if rest else np.empty((0,3,3));src=np.concatenate([chunks[j]['src'] for j in rest]) if rest else np.empty(0,int)
            masks,infos,supplies=self.classify(tri,src);counts=[int(m.sum()) for m in masks];dispensable=all(m.all() for m in masks)
            if dispensable:active.remove(k)
            row=dict(component=k,volume_cm3=material_volume(chunks[k]['part'])*1e6,dispensable=dispensable,all_original_counts_without_component=counts);rows.append(row);print('VALUE AUDIT',self.group['id'],row,flush=True)
        self.critical_sources=set(np.concatenate([chunks[k]['src'] for k in active]).tolist()) if active else set()
        self.critical_points=cKDTree(np.vstack([chunks[k]['tri'].reshape(-1,3) for k in active])) if active else None
        self.seen.clear()
        save(self.out/'data/component_values.json',dict(complete=True,policy='Sequentially remove a fragment only if all saved demands still pass; current remaining fragments are essential to this contact configuration',remaining_essential_components=active,checks=rows));return True

    def candidate(self,directions,common=None,label='proposal',proxy=True):
        import step42_contact_consistency as context
        context._current_directions=np.asarray(directions)
        force_module.contact_boundary=motion_consistent_boundary;connected_module.contact_boundary=motion_consistent_boundary
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max(30.,10.*len(self.states)))
        try:
            r=Search.candidate(self,directions,common,label,proxy)
            if r is None:return None
            parts=[p for p in r['remaining'].decompose() if material_volume(p)>1e-12];parts.sort(key=lambda p:-material_volume(p));entries=[];winner=None;important=[]
            for k,p in enumerate(parts):
                tri,src=motion_consistent_boundary(self.mesh,S.unpack(p),self.allowed)
                if len(src) and set(src.tolist()).intersection(self.critical_sources) and (self.critical_points is None or float(self.critical_points.query(tri.reshape(-1,3))[0].min())<.001):important.append(S.unpack(p).vertices)
                if k>=3 or not len(tri):continue
                masks,infos,supplies=self.classify(tri,src);e=dict(part=p,triangles=tri,sources=src,masks=masks,infos=infos,supplies=supplies,counts=[int(m.sum()) for m in masks]);entries.append(e)
                if all(m.all() for m in masks):winner=e;break
            if not entries:return None
            best=max(entries,key=lambda e:(sum(m.all() for m in e['masks']),min(e['counts']),sum(e['counts'])))
            dist=np.zeros((len(important),len(important)))
            for i in range(len(important)):
                tree=cKDTree(important[i])
                for j in range(i+1,len(important)):dist[i,j]=dist[j,i]=max(float(tree.query(important[j])[0].min()),1e-12)
            gap=float(minimum_spanning_tree(dist).sum()) if len(important)>1 else 0.
            score=(-len(important),-gap,min(best['counts']),sum(best['counts']),sum(m.all() for m in best['masks']))
            row=dict(proposal=self.evaluations,label=label,component_count=len(parts),essential_contact_components=len(important),gap_m=gap,counts=best['counts']);self.connection_trace.append(row)
            if getattr(self,'value_best',None) is None or score>self.value_best['score']:
                self.value_best=dict(score=score,directions=r['directions'].copy(),counts=best['counts'],common=common);save(self.out/'data/value_checkpoint.json',dict(**row,directions=r['directions'].tolist()));print('VALUE-GUIDED RECOVERY',self.group['id'],best['counts'],'essential parts',len(important),'gap',gap,'proposal',self.evaluations,flush=True)
            if winner is None:return None
            original=r['remaining'];r.update(remaining=winner['part'],removed=self.seed-winner['part'],triangles=winner['triangles'],sources=winner['sources'],masks=winner['masks'],infos=winner['infos'],supplies=winner['supplies'],counts=winner['counts'],pre_prune_component_count=len(parts),pre_prune_volume_cm3=material_volume(original)*1e6,pruned_volume_cm3=(material_volume(original)-material_volume(winner['part']))*1e6,pruning_evidence=row)
            r['overlap']=max(material_volume(r['remaining']^s) for s in r['sweeps']);r['partition']=abs(material_volume(self.seed)-material_volume(r['remaining'])-material_volume(r['removed']))
            return r if r['overlap']<1e-10 and r['partition']<1e-10 and len(r['remaining'].decompose())==1 else None
        except (EvaluationDeadline,RuntimeError,ValueError,np.linalg.LinAlgError):return None
        finally:signal.setitimer(signal.ITIMER_REAL,0.)

    def connected_search(self):
        if not self.initialize_value():return None
        r=self.candidate(self.warm,label='essential component warm',proxy=False)
        if r is not None:return r
        for iteration in range(6):
            previous=self.value_best['score'] if getattr(self,'value_best',None) is not None else None
            for k in [0,1,5,4,2,3]:
                base=self.value_best['directions'].copy() if getattr(self,'value_best',None) is not None else self.warm.copy();n=self.normals[k];t=base[k]-(base[k]@n)*n;t/=np.linalg.norm(t);side=np.cross(n,t)
                for angle in [2,-2,5,-5,10,-10,20,-20,40,-40,80,-80]:
                    for lift in [.0001,.003,.01,.05,.2]:
                        ds=base.copy();a=np.deg2rad(angle);ds[k]=np.cos(a)*t+np.sin(a)*side+lift*n;ds[k]/=np.linalg.norm(ds[k]);r=self.candidate(ds,label=f'essential gap pose {k}')
                        if r is not None:return r
            if getattr(self,'value_best',None) is None or self.value_best['score']==previous:break
        for count in [160,512,2048]:
            for a in fibonacci(count):
                r=self.candidate(project_common(a,self.normals),a,label='essential joint tendency')
                if r is not None:return r
        return None
if __name__=='__main__':
    g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose1+6+11+13+14+17');s=ValueGuidedSearch('B',g);began=time.monotonic();r=s.connected_search()
    if r is None:print('VALUE GUIDED UNRESOLVED',flush=True)
    else:
        row=finish_connected(s,r,began);p=s.base/'step4/step4.2/data/report.json';report=json.loads(p.read_text());report['provenance']['code'].update(I.hashes([HERE/f for f in ['step4.2/step42_connected_safe.py','step4.2/step42_connected_local.py','step4.2/step42_connected_independent.py','step4.2/step42_contact_consistency.py','step4.2/step42_topology.py','step4.2/step42_topology_common.py','step4.2/step42_value_guided.py']]));report['artifacts']['component_values.json']=I.sha256(p.parent/'component_values.json');save(p,report);I.check_report(p);print(row,flush=True)
