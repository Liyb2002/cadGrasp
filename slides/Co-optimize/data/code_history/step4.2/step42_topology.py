"""Use separation between material components to guide path recovery."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_contact_consistency import *
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import minimum_spanning_tree
class TopologySearch(ConsistentSearch):
    def candidate(self,directions,common=None,label='proposal',proxy=True):
        global_placeholder=None
        import step42_contact_consistency as context
        context._current_directions=np.asarray(directions)
        force_module.contact_boundary=motion_consistent_boundary;connected_module.contact_boundary=motion_consistent_boundary
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max(30.,10.*len(self.states)))
        try:
            r=Search.candidate(self,directions,common,label,proxy)
            if r is None:return None
            parts=[p for p in r['remaining'].decompose() if material_volume(p)>1e-12];parts.sort(key=lambda p:-material_volume(p));entries=[];winner=None
            for part in parts:
                tri,src=motion_consistent_boundary(self.mesh,S.unpack(part),self.allowed)
                if not len(tri):continue
                masks,infos,supplies=self.classify(tri,src);e=dict(part=part,triangles=tri,sources=src,masks=masks,infos=infos,supplies=supplies,counts=[int(m.sum()) for m in masks]);entries.append(e)
                if all(m.all() for m in masks):winner=e;break
            if not entries:return None
            best=max(entries,key=lambda e:(sum(m.all() for m in e['masks']),min(e['counts']),sum(e['counts'])))
            # Surface-contact components only: untouched detached rings without
            # object contact do not attract the path search.
            vertices=[S.unpack(e['part']).vertices for e in entries];dist=np.zeros((len(vertices),len(vertices)))
            for i in range(len(vertices)):
                tree=cKDTree(vertices[i])
                for j in range(i+1,len(vertices)):
                    gap=float(tree.query(vertices[j])[0].min());dist[i,j]=dist[j,i]=max(gap,1e-12)
            gap=float(minimum_spanning_tree(dist).sum()) if len(vertices)>1 else 0.
            score=(int(sum(m.all() for m in best['masks'])),-gap,min(best['counts']),sum(best['counts']))
            row=dict(proposal=self.evaluations,label=label,component_count=len(parts),contact_components=len(entries),gap_m=gap,counts=best['counts']);self.connection_trace.append(row)
            if getattr(self,'topology_best',None) is None or score>self.topology_best['score']:
                self.topology_best=dict(score=score,directions=r['directions'].copy(),counts=best['counts']);save(self.out/'data/topology_checkpoint.json',dict(**row,directions=r['directions'].tolist()));print('TOPOLOGY RECOVERY',self.group['id'],best['counts'],'gap',gap,'proposal',self.evaluations,flush=True)
            if winner is None:return None
            original=r['remaining'];r.update(remaining=winner['part'],removed=self.seed-winner['part'],triangles=winner['triangles'],sources=winner['sources'],masks=winner['masks'],infos=winner['infos'],supplies=winner['supplies'],counts=winner['counts'],pre_prune_component_count=len(parts),pre_prune_volume_cm3=material_volume(original)*1e6,pruned_volume_cm3=(material_volume(original)-material_volume(winner['part']))*1e6,pruning_evidence=row)
            r['overlap']=max(material_volume(r['remaining']^s) for s in r['sweeps']);r['partition']=abs(material_volume(self.seed)-material_volume(r['remaining'])-material_volume(r['removed']))
            return r if r['overlap']<1e-10 and r['partition']<1e-10 and len(r['remaining'].decompose())==1 else None
        except (EvaluationDeadline,RuntimeError,ValueError,np.linalg.LinAlgError):return None
        finally:signal.setitimer(signal.ITIMER_REAL,0.)
    def connected_search(self):
        r=self.candidate(self.warm,label='topology warm',proxy=False)
        if r is not None:return r
        for iteration in range(8):
            before=self.topology_best['score'] if getattr(self,'topology_best',None) is not None else None
            for k in [0,1,5,4,2,3]:
                base=self.topology_best['directions'].copy() if getattr(self,'topology_best',None) is not None else self.warm.copy();n=self.normals[k];t=base[k]-(base[k]@n)*n;t/=np.linalg.norm(t);side=np.cross(n,t)
                for angle in [2,-2,5,-5,10,-10,20,-20,40,-40,80,-80]:
                    for lift in [.0001,.003,.01,.05,.2]:
                        ds=base.copy();a=np.deg2rad(angle);ds[k]=np.cos(a)*t+np.sin(a)*side+lift*n;ds[k]/=np.linalg.norm(ds[k]);r=self.candidate(ds,label=f'gap-directed pose {k}')
                        if r is not None:return r
            if getattr(self,'topology_best',None) is None or self.topology_best['score']==before:break
        return None
if __name__=='__main__':
    g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose1+6+11+13+14+17');s=TopologySearch('B',g);began=time.monotonic();r=s.connected_search()
    if r is not None:
        row=finish_connected(s,r,began);p=s.base/'step4/step4.2/data/report.json';report=json.loads(p.read_text());report['provenance']['code'].update(I.hashes([HERE/'step4.2/step42_connected_safe.py',HERE/'step4.2/step42_connected_local.py',HERE/'step4.2/step42_connected_independent.py',HERE/'step4.2/step42_contact_consistency.py',HERE/'step4.2/step42_topology.py']));save(p,report);I.check_report(p);print(row,flush=True)
    else:print('TOPOLOGY UNRESOLVED',flush=True)
