"""Continuous two-segment withdrawal search, without adding support material."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_contact_consistency import *
import step42_contact_consistency as consistency

class BentSearch(ConsistentSearch):
    def paths_candidate(self,lengths,second,label):
        self.evaluations+=1;signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max(30.,10.*len(self.states)))
        try:
            ds=self.warm.copy();consistency._current_directions=ds
            paths=[np.array([np.zeros(3),l*d,l*d+self.length*b]) for l,d,b in zip(lengths,ds,second)]
            if any(np.min(np.diff(p,axis=0)@n)<-1e-12 for p,n in zip(paths,self.normals)):return None
            # Endpoint separation along the final segment direction.
            if any(((self.mesh.vertices+p[-1])@b).min()<=(self.seed_mesh.vertices@b).max()+1e-9 for p,b in zip(paths,second)):return None
            sweeps=[];padded_sweeps=[]
            for path in paths:
                pieces=[];padded_pieces=[]
                for a,b in zip(path[:-1],path[1:]):
                    m=self.mesh.copy();m.apply_translation(a);pieces.append(S.solid(S.swept_solid(m,b-a,fan_in=8)))
                    padded_pieces.append(self.clearance.sweep(b-a,8).translate((a/S.SCALE).tolist()))
                sweeps.append(union(pieces));padded_sweeps.append(union(padded_pieces))
            construction=self.clearance.construct(self.seed,sweeps,padded_sweeps,self.allowed[np.max(self.mesh.face_normals[self.allowed]@ds.T,axis=1)<=1e-9],boundary=motion_consistent_boundary)
            if not construction['diagnostics']['geometry_resolved']:return None
            cut=construction['padded_cut'];remaining=construction['remaining'];parts=remaining.decompose();parts.sort(key=lambda p:-material_volume(p));evidence=[];winner=None
            for part in parts[:3]:
                if material_volume(part)<1e-12:continue
                tri,src=motion_consistent_boundary(self.mesh,S.unpack(part),self.allowed);masks,infos,supplies=self.classify(tri,src);counts=[int(m.sum()) for m in masks]
                e=dict(part=part,triangles=tri,sources=src,masks=masks,infos=infos,supplies=supplies,counts=counts,volume_cm3=material_volume(part)*1e6);evidence.append(e)
                if all(m.all() for m in masks):winner=e;break
            if not evidence:return None
            best=max(evidence,key=lambda e:(sum(m.all() for m in e['masks']),min(e['counts']),sum(e['counts'])));score=(int(sum(m.all() for m in best['masks'])),min(best['counts']),sum(best['counts']))
            row=dict(proposal=self.evaluations,label=label,counts=best['counts'],first_lengths_m=list(lengths),component_count=len(parts));self.connection_trace.append(row)
            if getattr(self,'bent_best',None) is None or score>self.bent_best['score']:
                self.bent_best=dict(score=score,lengths=np.array(lengths),second=np.array(second),counts=best['counts']);save(self.out/'data/bent_checkpoint.json',dict(score=score,lengths_m=list(map(float,lengths)),second_directions=np.asarray(second).tolist(),counts=best['counts']));print('BENT RECOVERY',self.group['id'],best['counts'],'proposal',self.evaluations,flush=True)
            if winner is None:return None
            remaining=winner['part'];removed=self.seed-remaining;overlap=max(material_volume(remaining^s) for s in sweeps);partition=abs(material_volume(self.seed)-material_volume(remaining)-material_volume(removed))
            if overlap>=1e-10 or partition>=1e-10 or len(remaining.decompose())!=1:return None
            return dict(**{k:v for k,v in winner.items() if k not in ['part','volume_cm3']},clearance_diagnostics=construction['diagnostics'],remaining=remaining,removed=removed,directions=ds,paths=paths,fan=8,sweeps=sweeps,cut=cut,overlap=overlap,partition=partition,pre_prune_component_count=len(parts),pre_prune_volume_cm3=material_volume(self.seed-cut)*1e6,pruned_volume_cm3=(material_volume(self.seed-cut)-material_volume(remaining))*1e6,pruning_evidence=row)
        except (EvaluationDeadline,RuntimeError,ValueError,np.linalg.LinAlgError):return None
        finally:signal.setitimer(signal.ITIMER_REAL,0.)
    def search_bent(self):
        for length in [.001,.003,.005,.01,.02,.04,.06,.08,.10,.15,.20]:
            r=self.paths_candidate(np.full(len(self.states),length),self.normals,'turn to native +z')
            if r is not None:return r
        for round_index in range(4):
            previous=self.bent_best['score'] if getattr(self,'bent_best',None) is not None else None
            for k in range(len(self.states)):
                if getattr(self,'bent_best',None) is None:break
                base=self.bent_best
                for length in [.001,.003,.005,.01,.02,.04,.06,.08,.10,.15,.20]:
                    lengths=base['lengths'].copy();lengths[k]=length;r=self.paths_candidate(lengths,base['second'],f'independent turn distance pose {k}')
                    if r is not None:return r
            if getattr(self,'bent_best',None) is None or self.bent_best['score']==previous:break
        return None

def bent_case(name,group):
    search=BentSearch(name,group);began=time.monotonic();r=search.search_bent()
    if r is None:return dict(id=group['id'],passed=False)
    row=finish_connected(search,r,began);p=search.base/'step4/step4.2/data/report.json';report=json.loads(p.read_text());report['path_kind']='Independent two-segment continuous translations';report['motion_consistent_contacts_required']=True
    for state_row,path,(task,T) in zip(report['state_results'],r['paths'],search.states):
        state_row['path_fixture_m']=path.tolist();state_row['segment_world_displacements_m']=(np.diff(path,axis=0)@T[:3,:3].T).tolist();state_row['endpoint_completely_separated']=True
    report['provenance']['code'].update(I.hashes([HERE/'step4.2/step42_connected_safe.py',HERE/'step4.2/step42_connected_local.py',HERE/'step4.2/step42_connected_independent.py',HERE/'step4.2/step42_contact_consistency.py',HERE/'step4.2/step42_bent.py']));save(p,report);I.check_report(p);return row
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--set',required=True);args=p.parse_args();g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']==args.set);print(bent_case('B',g),flush=True)
