"""One sequential chain: perturb one pose, jointly refine all poses."""
import time
import numpy as np
from physics_guided_geometry import tangent_frames


def sample_one(directions, normals, pose, rng, max_angle=30.):
    """Random tangent cardinal direction and 5..max_angle degree step."""
    if not 5. < max_angle <= 90.:
        raise ValueError('max_angle must be greater than 5 and at most 90')
    candidate = directions.copy()
    frame = tangent_frames(directions)[pose]
    for _ in range(100):
        axis = int(rng.integers(4))
        angle = float(rng.uniform(5., max_angle))
        tangent = frame[:, axis // 2] * (1 if axis % 2 == 0 else -1)
        vector = directions[pose] * np.cos(np.deg2rad(angle)) + tangent * np.sin(np.deg2rad(angle))
        if vector @ normals[pose] >= 0:
            candidate[pose] = vector / np.linalg.norm(vector)
            return candidate, axis, angle
    raise ValueError('No legal single-pose sample in 100 draws')


def single_pose_search(self, iterations=2):
    from physics_guided import save
    began = time.monotonic()
    rng = np.random.default_rng(self.chain_seed)
    initial = self.normals.copy()  # every pose's native upward direction
    np.savez_compressed(self.out/'initial_directions.npz', directions=initial)
    current = self.exact(initial)
    initial_counts = current['counts']
    self.remember(current)
    events = [dict(stage='initial', directions=initial.tolist(), counts=initial_counts)]
    remaining = []
    winner = current if all(m.all() for m in current['masks']) else None
    for step in range(self.max_proposals):
        if winner is not None:
            break
        if not remaining:
            remaining = list(rng.permutation(len(initial)))
        pose = int(remaining.pop())
        self.proposals += 1
        event = dict(step=step+1, stage='sample', pose=self.group['poses'][pose],
                     before=current['directions'].tolist())
        start = len(self.trace)
        try:
            candidate, axis, angle = sample_one(current['directions'], self.normals, pose, rng, self.sample_angle)
            event.update(axis=axis, angle_degrees=angle, directions=candidate.tolist())
            sampled = self.exact(candidate)
            event['counts'] = sampled['counts']
            self.remember(sampled)
            current = sampled  # a valid sample remains the chain state if refinement fails
            current = self.refine(sampled, iterations, f'single pose chain step {step+1}')

        except (RuntimeError, ValueError) as error:
            event['error'] = str(error)
        event['descent_iterations'] = len(self.trace)-start
        event['accepted_descent_iterations'] = sum(bool(t.get('accepted')) for t in self.trace[start:])
        events.append(event)
        events.append(dict(step=step+1, stage='after_descent', directions=current['directions'].tolist(), counts=current['counts']))
        if all(m.all() for m in current['masks']):
            winner = current
        save(self.out/'chain_trajectory.json', events)
        print('CHAIN', step+1, 'pose', self.group['poses'][pose], current['counts'], flush=True)
    result = winner or self.best
    np.savez_compressed(self.out/'continuation_directions.npz', directions=current['directions'])
    save(self.out/'chain_trajectory.json', events)
    save(self.out/'global_proposals.json', events)
    save(self.out/'critical_load_selection.json', self.selection_trace)
    self.report_extra = dict(algorithm='single-pose random sampling followed by joint all-pose descent',
        proposal_count=self.proposals, proposal_budget=self.max_proposals, random_seed=self.chain_seed,
        sampling_angle_degrees=[5., self.sample_angle], descent_steps_per_sample=iterations,
        connectivity_required=False, component_pruning_deferred=True,
        acceptance_scope='all original loads and full 1% clearance exits; connectivity separately recorded',
        passed=bool(all(m.all() for m in result['masks']) and result['overlap']<1e-10 and
                    result['partition']<1e-10 and max(result['endpoint_overlap'])<1e-10))
    return self.finish(result, initial_counts, began,
        'aggregate_force_feasible' if winner is not None else 'bounded_single_pose_chain_unresolved',
        returned_best=winner is None, continuation_counts=current['counts'])
