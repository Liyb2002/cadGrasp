"""Legacy balanced sampling stream, short parallel gradients, bounded final recovery."""
from physics_guided_parallel import *

def all_load_search(self,iterations=12):
    began=time.monotonic();initial=self.warm();initial/=np.linalg.norm(initial,axis=1)[:,None]
    np.savez_compressed(self.out/'initial_directions.npz',directions=initial)
    initial_counts=None;initial_error=None;winner=None
    try:
        native=self.exact(initial);initial_counts=native['counts'];self.remember(native)
        if all(m.all() for m in native['masks']):winner=native
    except (RuntimeError,ValueError) as error:initial_error=str(error)
    if winner is None:
        common=floor_common(self.normals)
        if common is not None:winner=self.attempt(project_common(common,self.normals),'common floor cone',common)
    stop=threading.Event();executor=ThreadPoolExecutor(max_workers=1)
    worker=None;future=None;branch_count=0;used_seeds=[];local=None;independent=None;local_serial=None
    def receive():
        nonlocal future,winner
        if future is None or not future.done():return
        try:
            result,trace=future.result();self.trace.extend(trace);self.remember(result)
            self.global_trace.append(dict(status='gradient branch returned',counts=result['counts'],branch=branch_count))
            if all(m.all() for m in result['masks']):winner=result
        except (RuntimeError,ValueError) as error:
            self.global_trace.append(dict(status='gradient branch numerically unresolved',error=str(error)))
        future=None
    def launch():
        nonlocal worker,future,branch_count
        if future is not None or branch_count>=8 or self.best is None:return
        for seed in self.beam:
            if any(np.linalg.norm(seed['directions']-d)<.08 for d in used_seeds):continue
            if worker is None:
                worker=GradientBranch(self.name,self.group,out=self.out/'gradient_branch',directions=self.start_directions,
                    max_proposals=self.max_proposals,stop_event=stop)
            used_seeds.append(seed['directions'].copy());branch_count+=1
            future=executor.submit(worker.run_branch,seed,min(2,iterations))
            self.global_trace.append(dict(status='gradient branch launched',seed_counts=seed['counts'],branch=branch_count))
            break
    try:
        for count in [160,512,2048]:
            if winner is not None or self.proposals>=self.max_proposals:break
            axes=list(fibonacci(count));ordered=self.ordered_common(axes)
            for rank,(benefit,d,a) in enumerate(ordered):
                receive()
                if winner is not None or self.proposals>=self.max_proposals:break
                winner=self.attempt(d,f'concurrent common covering {count}; dual rank {rank}',a,force=rank<3)
                if winner is not None:break
                launch()
                # Spread local trials across angular scales and poses. A
                # changed incumbent resets the local families, never the global stream.
                if self.best is not None and rank%8==7:
                    if local_serial!=self.best['serial']:
                        local_serial=self.best['serial']
                        local=iter(interleaved_common_neighborhood(self.best_common)) if self.best_common is not None else None
                        independent=iter(independent_scales(self.best['directions'].copy(),self.normals,np.argsort(self.best['counts'])))
                    for stream in ['common','independent']:
                        for _ in range(4):
                            receive()
                            if winner is not None or self.proposals>=self.max_proposals:break
                            try:
                                if stream=='common':
                                    if local is None:break
                                    axis,lift=next(local);candidate=project_common(axis,self.normals,lift)
                                    winner=self.attempt(candidate,'interleaved common recovery',axis)
                                else:
                                    candidate,k=next(independent)
                                    winner=self.attempt(candidate,f'interleaved independent recovery {k}')
                            except StopIteration:break
                        if winner is not None:break
                save(self.out/'global_proposals.json',self.global_trace)
            if winner is not None:break
        # Sampling keeps running while gradients work; wait only for the
        # bounded final branch after the sampling budget is exhausted.
        if winner is None and future is not None:
            result,trace=future.result();future=None;self.trace.extend(trace);self.remember(result)
            if all(m.all() for m in result['masks']):winner=result
    finally:
        stop.set();executor.shutdown(wait=True)
    # Keep the original bounded multi-start recovery after the parallel branch
    # stream; it repaired a case missed by the short-branch schedule on B.
    if winner is None:
        for seed in list(self.beam):
            refined=self.refine(seed,iterations,'final multi-start real-cone recovery')
            if all(m.all() for m in refined['masks']):
                winner=refined;break
    result=winner or self.best
    if result is None:raise RuntimeError('No resolved construction in concurrent search')
    np.savez_compressed(self.out/'continuation_directions.npz',directions=result['directions'])
    save(self.out/'global_proposals.json',self.global_trace);save(self.out/'critical_load_selection.json',self.selection_trace)
    self.report_extra=dict(algorithm='concurrent sampling and short real-cone-dual gradient branches',initial_counts=initial_counts,
        initial_geometry_error=initial_error,proposal_count=self.proposals,proposal_budget=self.max_proposals,
        gradient_exact_evaluations=(worker.exact_calls-1000000) if worker is not None else 0,
        total_exact_evaluations=self.exact_calls+((worker.exact_calls-1000000) if worker is not None else 0),
        gradient_branches=branch_count,gradient_steps_per_branch=min(2,iterations),gradient_branch_limit=8,
        critical_load_policy='all original load vectors scanned with real cone projection duals; exact projection of finalists',
        physics_gradient='real constructed contact cone residual; iterative nonnegative cone projection',
        connectivity_required=False,acceptance_scope='all original loads and full 1% clearance exits; connectivity separately recorded',
        passed=bool(all(m.all() for m in result['masks']) and result['overlap']<1e-10 and result['partition']<1e-10 and max(result['endpoint_overlap'])<1e-10))
    return self.finish(result,initial_counts,began,'aggregate_force_feasible' if winner is not None else 'bounded_concurrent_unresolved',continuation_counts=result['counts'])
