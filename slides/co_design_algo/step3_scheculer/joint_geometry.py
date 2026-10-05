"""Task-local pools with an unrestricted, append-only active-task assignment.

A selected physical head retains its owner surface and solid. For each task,
activate it when its addition preserves that task's local geometry, withdrawal
and basic paths. Inactive material and fixture placements remain Step5 work.
"""
from step3_scheculer.sequential_geometry import SequentialGeometry, active_entries, local_group


class JointGeometry(SequentialGeometry):
    def __init__(self, problems, count=200):
        super().__init__(problems, count)
        self.initial = [entry for pool in self.pools for entry in pool]

    def propose(self, entries, candidate):
        if not candidate['valid']:
            return None, dict(reason=candidate['reason'], per_pose=[])
        if self.same_center(candidate, entries):
            return None, dict(reason='center_already_selected', per_pose=[])
        tasks, checks = [], []
        for task, geometry in enumerate(self.local):
            view = self.view(candidate, task)
            if not view['valid']:
                check = dict(passed=False, reason=view['reason'])
            else:
                active = [self.view(e, task) for e in active_entries(entries, task)]
                check = local_group(geometry, active+[view])
            checks.append(check)
            if check['passed']:
                tasks.append(task)
        if not tasks:
            return None, dict(reason='no_compatible_task', per_pose=checks)
        return dict(candidate, active_tasks=tuple(tasks)), dict(reason='eligible', per_pose=checks)

    def task_check(self, entries, task):
        active = active_entries(entries, task)
        if not active:
            # Some tasks can balance their loads at the original object foot.
            # An empty prefix also must not prevent other tasks from progressing.
            return dict(passed=True, reason='no_active_heads', common_direction_ids=[],
                        common_path_components=[])
        return local_group(self.local[task], [self.view(e, task) for e in active])

    def group_check(self, entries):
        checks = [self.task_check(entries, k) for k in range(len(self.problems))]
        return dict(passed=all(c['passed'] for c in checks), per_pose=checks,
                    inactive_material_checked=False, complete_fixture_constructed=False)

    def contacts_by_pose(self, entries):
        return [[self.view(e, k)['contact'] for e in active_entries(entries, k)]
                for k in range(len(self.problems))]

    def expand(self, entry, factor):
        raise RuntimeError('Joint search only adds fixed-area heads; resizing is disabled')
