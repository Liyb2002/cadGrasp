"""Six-head greedy, then monotone single/pair addition from its failed seeds."""
import argparse
import contextlib
import itertools
import json
import multiprocessing
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from codes.precompute_objects.heads.surface import FLOOR_CLEARANCE_M
from codes.precompute_objects.registry import active_objects, task_poses
from step3_scheculer import contacts as I, passive_support as U
from step3_scheculer import additive_gpu_v24 as GPU, initialize_cached_v19 as OLD
from step3_scheculer.joint_prepared import scoring_sources
from step3_scheculer.pair_scoring import C, J
from step3_scheculer.run_sequential import numerical_recovery

SCHEMA = 'residual_witness_fallback_v27'
STAGE = 'independent_poses_residual_v27'


def candidate_key(e):
    return e['contact']['candidate_id']


def extendable(entries, proposed, catalogue, scale):
    combined = entries + proposed
    centers = [e['contact']['center_m'] for e in combined]
    if any(np.linalg.norm(a-b) < scale*1e-6 for i,a in enumerate(centers) for b in centers[:i]):
        return False
    directions, ports = OLD.compatibility(combined, catalogue)
    return bool(directions and ports)


def witness_proposals(selected, eligible, columns, floor, targets, mask, catalogue,
                      scale, load_probes=32, pair_budget=128, max_added=6):
    """Sparse failed-load witnesses propose bundles of any size within budget.

    Solve each compatible ray/port group's optimistic cone. A witness's rays
    are assigned to a distinct-center compatible set of heads. Acceptance
    still checks all original loads. Infeasible probes rule out that group
    for full coverage, but do not assert continuous geometric infeasibility.
    """
    directions, ports = OLD.compatibility(selected, catalogue)
    groups = {}
    for di in sorted(directions):
        for port in sorted(ports):
            group = [e for e in eligible if di in e['directions'][0] and port in e['path_components']]
            if group:
                groups.setdefault(tuple(sorted(candidate_key(e) for e in group)), group)
    failed = np.flatnonzero(~mask)
    probes = failed[np.unique(np.linspace(0,len(failed)-1,min(load_probes,len(failed))).astype(int))]
    proposals = {}
    evidence = dict(probe_indices=probes.tolist(), group_count=len(groups), witness_solves=0,
                    unresolved=0, infeasible_group_probes=0, oversized_witnesses=0)
    existing = I.merge_columns(floor, *[columns[candidate_key(e)] for e in selected])
    ray_key = lambda row: tuple(np.round(row,13))
    old_rays = {ray_key(row) for row in existing}
    for group in groups.values():
        owners = {}
        by_id = {candidate_key(e):e for e in group}
        for e in group:
            for row in columns[candidate_key(e)]:
                owners.setdefault(ray_key(row),set()).add(candidate_key(e))
        full = I.merge_columns(existing,*[columns[candidate_key(e)] for e in group])
        for index in probes:
            try:
                witness = C.W.solve(full,U.target(targets[index],7))
                evidence['witness_solves'] += 1
            except (RuntimeError,np.linalg.LinAlgError):
                evidence['unresolved'] += 1
                continue
            if witness is None:
                evidence['infeasible_group_probes'] += 1
                continue
            needed = [owners[ray_key(row)] for row in full[witness['indices']] if ray_key(row) not in old_rays]
            chosen = []
            while needed:
                candidates = set().union(*needed)
                ranked = sorted(candidates,key=lambda k:(-sum(k in o for o in needed),k))
                key = next((k for k in ranked if extendable(selected,chosen+[by_id[k]],catalogue,scale)),None)
                if key is None:
                    break
                chosen.append(by_id[key])
                needed = [o for o in needed if key not in o]
            if not needed and chosen:
                if len(chosen)<=max_added:
                    proposals.setdefault(tuple(sorted(candidate_key(e) for e in chosen)),chosen)
                else:
                    evidence['oversized_witnesses'] += 1
            if len(proposals)>=pair_budget:
                return list(proposals.values()),evidence
    return sorted(proposals.values(),key=lambda p:(len(p),tuple(candidate_key(e) for e in p))),evidence


class AdditiveSearch(OLD.Search):
    def __init__(self,name,pose,config):
        super().__init__(name,pose,device=config['device'])
        self.out = I.OUTPUTS/name/STAGE/pose/'step3_scheculer'
        self.out.mkdir(parents=True,exist_ok=True)
        self.backend = GPU.GPUClassifier(self.task.targets,config['device'])
        self.force_cache = {}
        self.unresolved = {}
        self.config = config
        self.pool = [e for pool in self.pools.values() for e in pool if e['valid']]
        self.by_id = {candidate_key(e):e for e in self.pool}

    def grow(self,seed):
        selected = [self.by_id[k] for k in seed['selected_ids']]
        initial_ids = list(seed['selected_ids'])
        assert extendable([],selected,self.catalogue,self.scale)
        source = I.OUTPUTS/self.name/OLD.STAGE/self.pose/'step3_scheculer'/f"trajectory_{seed['trajectory']:02d}"/'coverage.npz'
        if source.is_file():
            with np.load(source) as data:
                saved_mask = data['mask'].copy()
            assert saved_mask.dtype == np.dtype(bool)
            assert len(saved_mask) == len(self.task.targets)
            assert int(saved_mask.sum()) == seed['covered']
            key = tuple(sorted(seed['selected_ids']))
            self.force_cache[key] = (saved_mask,dict(reused_greedy_coverage=True))
            self.inputs.append(source)
        mask,_ = self.classify(selected)
        initial_covered = int(mask.sum())
        rounds = []
        upper_bound = None
        started = time.monotonic()
        status = 'head_budget_exhausted'
        while len(selected) < self.config['max_heads'] and not mask.all():
            before = int(mask.sum())
            eligible = [e for e in self.pool if extendable(selected,[e],self.catalogue,self.scale)]
            # Relax shared exits and duplicate-center constraints to get a
            # superset cone. One infeasible original demand excludes every
            # possible additive completion of this seed in the finite pool.
            optimistic = I.merge_columns(self.floor,*[self.columns[candidate_key(e)] for e in selected+eligible]) if hasattr(self,'columns') else None
            failed = np.flatnonzero(~mask)
            probes = failed[np.unique(np.linspace(0,len(failed)-1,min(32,len(failed))).astype(int))]
            impossible = None
            if hasattr(self,'columns'):
                for index in probes:
                    try:
                        target = U.target(self.task.targets[index],7)
                        separation = linprog(-target,A_ub=optimistic,
                            b_ub=np.zeros(len(optimistic)),bounds=[(-1,1)]*7,
                            method='highs',options=dict(time_limit=.5,
                                primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
                        if separation.success and np.linalg.norm(separation.x)>1e-12:
                            normal = separation.x/np.linalg.norm(separation.x)
                            if np.max(optimistic@normal)<=1e-12 and target@normal>1e-8+1e-12:
                                impossible = int(index)
                                separator = normal.tolist()
                                break
                    except (RuntimeError,np.linalg.LinAlgError):
                        continue
            if impossible is not None:
                upper_bound = dict(failed_original_load=impossible,separator=separator,eligible_heads=len(eligible),
                    relaxed_common_exits=True,relaxed_duplicate_centers=True)
                status = 'finite_candidate_superset_cannot_complete_seed'
                break
            best = None
            unresolved_before = len(self.unresolved)
            for e in eligible:
                try:
                    proposed_mask,_ = self.classify(selected+[e],mask)
                except RuntimeError:
                    continue
                score = int(proposed_mask.sum())
                if score > before and (best is None or score > best[0]):
                    best = (score,[e],proposed_mask)
            mode = 'single_addition'
            guidance = None
            if best is None and len(selected)<self.config['max_heads']:
                pairs,guidance = witness_proposals(selected,eligible,self.columns,self.floor,
                    self.task.targets,mask,self.catalogue,self.scale,
                    self.config['load_probes'],self.config['pair_budget'],self.config['max_heads']-len(selected))
                guidance['pair_proposals'] = len(pairs)
                mode = 'residual_witness_bundle_addition'
                for pair in pairs:
                    try:
                        proposed_mask,_ = self.classify(selected+pair,mask)
                    except RuntimeError:
                        continue
                    score = int(proposed_mask.sum())
                    if score > before and (best is None or score > best[0]):
                        best = (score,pair,proposed_mask)
            if best is None:
                status = 'no_improving_tested_addition'
                rounds.append(dict(mode=mode,covered_before=before,covered_after=before,
                    eligible_single_heads=len(eligible),guidance=guidance,
                    unresolved_proposals=len(self.unresolved)-unresolved_before,accepted=False))
                break
            score,added,new_mask = best
            assert np.all(new_mask[mask]) and score > before
            selected += added
            mask = new_mask
            assert set(initial_ids) <= {candidate_key(e) for e in selected}
            assert extendable([],selected,self.catalogue,self.scale)
            rounds.append(dict(mode=mode,added_ids=[candidate_key(e) for e in added],
                covered_before=before,covered_after=score,heads=len(selected),
                eligible_single_heads=len(eligible),guidance=guidance,accepted=True))
            I.save(self.out/'progress.json',dict(complete=False,seed=seed['trajectory'],
                heads=len(selected),covered=score))
            print(self.name,self.pose,'seed',seed['trajectory'],mode,'heads',len(selected),'covered',score,flush=True)
        ds,cs = OLD.compatibility(selected,self.catalogue)
        passed = bool(mask.all() and ds and cs)
        if passed:
            status = 'all_original_loads_and_cached_exits_passed'
        result = dict(seed_trajectory=seed['trajectory'],initial_ids=initial_ids,
            initial_heads=len(initial_ids),initial_covered=initial_covered,
            selected_ids=[candidate_key(e) for e in selected],heads=len(selected),
            covered=int(mask.sum()),passed=passed,status=status,rounds=rounds,upper_bound=upper_bound,
            common_direction_ids=sorted(ds),common_path_components=sorted(cs),
            object_exit_world=(-np.asarray(self.catalogue['vectors'][min(ds)])).tolist() if ds else None,
            seconds=time.monotonic()-started)
        return selected,mask,result

    def run_fallback(self,greedy,greedy_path):
        records = []
        best = None
        # Try all failed greedy seeds if necessary, prioritizing coverage.
        seeds = sorted(greedy['trajectory_results'],key=lambda r:(-r['covered'],r['trajectory']))
        for seed in seeds:
            selected,mask,result = self.grow(seed)
            records.append(result)
            I.save(self.out/f"seed_{seed['trajectory']:02d}.json",result)
            if best is None or result['covered'] > best[2]['covered']:
                best = selected,mask,result
            if result['passed']:
                best = selected,mask,result
                break
        selected,mask,result = best
        # Every accepted result has a fresh independent original CPU LP check
        # built from actual triangles, never just cached moments or probes.
        contacts = []
        for e in selected:
            c = e['contact']
            assert e['valid'] and e['local_clearance']['valid']
            assert not np.intersect1d(c['source_faces'],self.task.domain.work_ids).size
            assert c['triangles_m'][:,:,2].min() >= FLOOR_CLEARANCE_M-self.scale*1e-10
            points = c['triangles_m'].reshape(-1,3)
            normals = np.repeat(-self.task.domain.mesh.face_normals[c['source_faces']],3,axis=0)
            np.testing.assert_allclose(np.c_[normals,np.cross(points-self.task.domain.com,normals)],
                                       c['wrench_generators'],atol=1e-12,rtol=0)
            contacts.append({k:v for k,v in c.items() if k not in ('wrench_generators','wrench_com_m')})
        final_mask,info = J.classify(self.task.supply(contacts),self.task.targets)
        if result['passed']:
            assert final_mask.all()
        final = self.out/f'final_contacts_{self.pose}.npz'
        I.save_contacts(final,[e['contact'] for e in selected])
        np.savez_compressed(self.out/'coverage.npz',mask=final_mask)
        result['covered'] = int(final_mask.sum())
        result['passed'] = bool(final_mask.all())
        self.inputs += [greedy_path]
        if self.unresolved:
            I.save(self.out/'unresolved_proposals.json',dict(proposals=[dict(ids=list(k),error=v) for k,v in self.unresolved.items()]))
        report = dict(complete=True,schema=SCHEMA,object=self.name,poses=[self.pose],
            passed=result['passed'],mode='additive_fallback',config=self.config,
            greedy_max_heads=6,greedy_chain_budget=30,seed_budget=len(seeds),seeds_run=len(records),
            load_count=len(final_mask),load_subsampling=False,full_support_constructed=False,
            no_head_deletion=True,step2_recomputed=False,passive_support_constraint=U.description(),
            result=result,seed_results=records,original_cpu_final_check=info,
            actual_triangle_force_generators_checked=True,
            numerical_unresolved_proposals=len(self.unresolved),
            backend=dict(self.backend.info,calls=self.backend.calls,lp_count=self.backend.lp_count),
            provenance=dict(inputs=I.hashes(self.inputs),code=I.hashes([
                Path(__file__),Path(GPU.__file__),Path(OLD.__file__)]+scoring_sources())),
            artifacts={p.name:I.sha256(p) for p in (final,self.out/'coverage.npz')},
            failure_meaning='Bounded additive search failed; not proof of infeasibility.')
        I.save(self.out/'schedule.json',report)
        I.check_report(self.out/'schedule.json')
        return report


def solve(args):
    name,pose,config = args
    out = I.OUTPUTS/name/STAGE/pose/'step3_scheculer'
    out.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (out/'pipeline.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            greedy_path = I.OUTPUTS/name/OLD.STAGE/pose/'step3_scheculer/schedule.json'
            if not greedy_path.exists():
                original = OLD.solve((name,pose,30,config['device']))
                if 'error' in original:
                    raise RuntimeError(original['error'])
            greedy = I.check_report(greedy_path)
            assert greedy['chain_budget'] == 30 and greedy['max_heads'] == 6
            if greedy['passed']:
                report = dict(complete=True,schema=SCHEMA,object=name,poses=[pose],passed=True,
                    mode='greedy_reused',config=config,result=greedy['result'],load_count=greedy['load_count'],
                    full_support_constructed=False,fallback_not_run=True,
                    provenance=dict(inputs=I.hashes([greedy_path]),code=I.hashes([Path(__file__)])))
                I.save(out/'schedule.json',report)
            else:
                with numerical_recovery(out/'numerical_retries',{}):
                    search = AdditiveSearch(name,pose,config)
                    report = search.run_fallback(greedy,greedy_path)
            I.check_report(out/'schedule.json')
            return dict(object=name,pose=pose,passed=report['passed'],mode=report['mode'],
                heads=report['result']['heads'],covered=report['result']['covered'],
                seeds_run=report.get('seeds_run',0),seconds=time.monotonic()-start)
        except Exception as error:
            traceback.print_exc()
            return dict(object=name,pose=pose,passed=False,error=repr(error),seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects',nargs='+',default=list(active_objects()))
    parser.add_argument('--poses',nargs='+')
    parser.add_argument('--failed-only',action='store_true')
    parser.add_argument('--jobs',type=int,default=6)
    parser.add_argument('--max-heads',type=int,default=12)
    parser.add_argument('--load-probes',type=int,default=32)
    parser.add_argument('--pair-budget',type=int,default=128)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    if args.max_heads < 7 or args.load_probes < 1 or args.pair_budget < 1:
        parser.error('Fallback budgets must allow growth and positive pair/probe counts')
    config = {k:getattr(args,k) for k in ('max_heads','load_probes','pair_budget','device')}
    cases,rows = [],[]
    for name in args.objects:
        for pose in args.poses or task_poses(name):
            if pose not in task_poses(name):
                raise ValueError('Unknown current pose')
            original = I.OUTPUTS/name/OLD.STAGE/pose/'step3_scheculer/schedule.json'
            if args.failed_only and (not original.is_file() or I.check_report(original)['passed']):
                continue
            path = I.OUTPUTS/name/STAGE/pose/'step3_scheculer/schedule.json'
            if path.is_file():
                d = I.check_report(path)
                if d['config'] == config:
                    rows.append(dict(object=name,pose=pose,passed=d['passed'],mode=d['mode'],
                        heads=d['result']['heads'],covered=d['result']['covered'],resumed=True))
                    continue
                raise ValueError('Existing result has different config; preserve accepted outputs')
            cases.append((name,pose,config))
    ledger = I.OUTPUTS/'B'/STAGE/'batch.json'
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(solve,c) for c in cases]):
            row = future.result()
            rows.append(row)
            print('POSE COMPLETE',row,flush=True)
            I.save(ledger,dict(complete=False,config=config,results=rows))
    I.save(ledger,dict(complete=True,config=config,results=rows))
    print('TOTAL',sum(r['passed'] for r in rows),'/',len(rows),flush=True)
    if any('error' in r for r in rows):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
