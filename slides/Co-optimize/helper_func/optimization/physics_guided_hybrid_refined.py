"""Hybrid refinement: same physics, iterative cone projections and balanced local scales."""
from physics_guided_hybrid import *
from physics_guided_cone_iterative import cone_projection


def interleaved_common_neighborhood(center):
    # Visit every angular scale before spending the budget on more azimuths.
    families=list(common_neighborhood(center));block=len(families)//5
    for start in range(0,block,4):
        for scale in range(5):
            for axis,lift in families[scale*block+start:scale*block+start+4]:
                yield axis,lift


class RefinedHybridSearch(HybridSearch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.additional_code += [Path(__file__),HERE/'helper_func/optimization/physics_guided_cone_iterative.py']

    def projection(self,result,k,index):
        key=(result['serial'],k,index)
        if key not in self.projection_cache:
            target=U.target(self.states[k][0].targets[index])
            value=cone_projection(result['supplies'][k],target)
            if value['kkt_max_violation']>1e-7*max(1.,np.linalg.norm(target)):
                raise RuntimeError('iterative cone projection KKT unresolved')
            self.projection_cache[key]=value
        return self.projection_cache[key]

    def choose_loads(self,result):
        bank=super().choose_loads(result)
        # Screening is a necessary condition only. Keep its working set small;
        # final construction still checks every original load and geometry.
        for k,row in enumerate(self.selection_trace[-len(self.states):]):
            current=row['indices'];previous=sorted(self.proxy_indices[k])
            self.proxy_indices[k]=set(current+previous[:max(0,8-len(current))])
        return bank

    def proxy(self,d):
        keep=np.max(self.ray_normals@d.T,axis=1)<=1e-9
        for k,(task,T) in enumerate(self.states):
            full=np.vstack([self.floors[k],self.rays[k][keep]])
            for index in sorted(self.proxy_indices[k]):
                target=U.target(task.targets[index])
                if C.W.solve(full,target) is not None:continue
                projection=cone_projection(full,target);norm=np.linalg.norm(projection['dual'])
                if norm>1e-12:
                    normal=-projection['dual']/norm
                    if np.max(full@normal)<=1e-12 and target@normal>1e-8:return False
        return True

    def optimize(self,iterations=12):
        began=time.monotonic();initial_directions=self.warm();initial_directions/=np.linalg.norm(initial_directions,axis=1)[:,None]
        np.savez_compressed(self.out/'initial_directions.npz',directions=initial_directions)
        initial_counts=None;initial_error=None;winner=None
        try:
            native=self.exact(initial_directions);initial_counts=native['counts'];self.remember(native)
            if all(m.all() for m in native['masks']):winner=native
            elif min(native['counts'])/32768>.75:
                refined=self.refine(native,min(iterations,6),'native near-feasible')
                if all(m.all() for m in refined['masks']):winner=refined
        except (RuntimeError,ValueError) as error:initial_error=str(error)
        if winner is None:
            common=floor_common(self.normals)
            if common is not None:winner=self.attempt(project_common(common,self.normals),'common floor cone',common)
        global_limit=max(1,int(self.max_proposals*.65))
        for count in [160,512,2048]:
            if winner is not None or self.proposals>=global_limit:break
            axes=list(fibonacci(count))
            if self.best is not None:
                weights,_=self.dual_weights(self.best)
                important=np.argsort(weights)[-16:]
                axes=[-self.ray_normals[j] for j in important if weights[j]>0]+axes
            ordered=self.ordered_common(axes)
            for rank,(benefit,d,a) in enumerate(ordered):
                winner=self.attempt(d,f'common covering {count}; dual rank {rank}',a,force=rank<3)
                if winner is not None or self.proposals>=global_limit:break
            if winner is not None:break
            if self.best is not None:
                refined=self.refine(self.best,iterations,f'after common {count}')
                if all(m.all() for m in refined['masks']):winner=refined;break
            if self.best_common is not None:
                center=self.best_common.copy()
                local_begin=self.proposals
                for a,lift in interleaved_common_neighborhood(center):
                    winner=self.attempt(project_common(a,self.normals,lift),'legacy common local recovery',a)
                    if winner is not None or self.proposals>=self.max_proposals or self.proposals-local_begin>=80:break
        if winner is None and self.best is not None and self.proposals<self.max_proposals:
            order=list(np.argsort(self.best['counts']))
            for candidate,k in independent_neighborhood(self.best['directions'],self.normals,order):
                winner=self.attempt(candidate,f'legacy independent recovery {k}')
                if winner is not None or self.proposals>=self.max_proposals:break
        if winner is None and self.best is not None:
            for seed in list(self.beam):
                refined=self.refine(seed,iterations,'final multi-start dual recovery')
                if all(m.all() for m in refined['masks']):winner=refined;break
        result=winner or self.best
        if result is None:raise RuntimeError('No numerically resolved hybrid construction')
        d=result['directions'];np.savez_compressed(self.out/'continuation_directions.npz',directions=d)
        save(self.out/'global_proposals.json',self.global_trace);save(self.out/'critical_load_selection.json',self.selection_trace)
        self.report_extra=dict(algorithm='hybrid shared-trend exploration and actual-cone-dual gradients; iterative projection and interleaved local scales',
            initial_counts=initial_counts,initial_geometry_error=initial_error,
            proposal_count=self.proposals,proposal_budget=self.max_proposals,
            physics_gradient='missing-ray benefit from the real constructed cone residual; no all-potential-contact pricing LP',
            geometry_gradient='analytic initial compatibility + numerical nominal contact-core trajectory field; positive penetration p=8',
            potential_contact_policy='positive-volume original seed contact boundary; unique existing vertices; no invented attainable patches',
            critical_load_policy='scan all original load vectors with real projection-dual certificates and project finalists; no global maximum certificate',
            connectivity_required=False,acceptance_scope='all original loads and full 1% clearance exits; connectivity recorded separately',
            passed=bool(all(m.all() for m in result['masks']) and max(material_volume(result['construction']['remaining']^self.clearance.sweep(self.length*x,padded=False)) for x in d)<1e-10 and result['partition']<1e-10 and max(result['endpoint_overlap'])<1e-10 and result['construction']['diagnostics']['geometry_resolved']))
        return self.finish(result,initial_counts,began,'aggregate_force_feasible' if winner is not None else 'bounded_hybrid_unresolved',continuation_counts=result['counts'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set',required=True);parser.add_argument('--directions',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--iterations',type=int,default=12)
    parser.add_argument('--max-proposals',type=int,default=1200)
    args=parser.parse_args()
    groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    groups += [dict(g,id='illegal/'+g['id']) for g in json.loads((ROOT/'objects/B/illegal_pose_sets.json').read_text())['sets']]
    group=next(g for g in groups if g['id']==args.set)
    search=RefinedHybridSearch('B',group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)))
        raise

if __name__=='__main__':main()
