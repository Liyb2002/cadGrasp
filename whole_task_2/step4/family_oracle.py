"""Select usable contact material by full loads, not by retaining every probe."""
import tempfile
from pathlib import Path
import time
import numpy as np

from .adapter import ComputedOracle
from whole_task_2.step3.legacy import kernel as C
from whole_search.common import transform_solid, transform_mesh, unpack_solid
from base_oracle import cover
from material_force import certify_material_contacts
from prune_support import material_components


class FamilyFloorOracle(ComputedOracle):
    def __init__(self, model):
        super().__init__(model)
        self.loads = model.current_endpoint_targets
        if any(float(load[:, 2].min()) <= 0 for load in self.loads):
            raise ValueError('Continuous ground enclosure requires positive reaction Fz')

    def query(self, layout, *, connected=False):
        result = super().query(layout, connected=connected)
        result['contact_anchor_policy'] = 'actual_force_certified_subset_of_legal_contact_material'
        return result

    def components(self, layout, rows, low, high):
        self.component_queries += 1; began = time.monotonic(); m = self.model
        self.core_clip_proposal = None
        stock_mesh = C.trimesh.creation.box(high-low)
        stock_mesh.apply_translation((high+low)/2)
        stock = C.S.solid(stock_mesh)
        for row in rows:
            t = row['T']
            stock = stock.trim_by_plane(tuple(t[2, :3]), float(-t[2, 3]/C.S.SCALE))
        bounds = np.array([[x, y, z] for x in [low[0], high[0]]
                           for y in [low[1], high[1]] for z in [low[2], high[2]]])
        obstacles = []
        for task in self.pose_constraints(layout).poses:
            k, q = task.index, task.Q_fixture_object
            obstacles.append(transform_solid(task.work_access.solid(), q))
            direction = q[:3, :3].T @ layout.directions[k]
            local = (bounds-q[:3, 3]) @ q[:3, :3]
            length = max(m.length, float((local @ direction).max()-(m.mesh.vertices @ direction).min())+.01)
            obstacles.append(transform_solid(m.clearance.sweep(length*direction, padded=False), q))
        free = stock-C.union(obstacles)
        from whole_step4_render import NominalBuilder
        nominal = NominalBuilder(m).solid(layout)
        trials = []; winner = None; winner_rows = []
        for index, part in enumerate(material_components(free)):
            mesh = unpack_solid(part); per_pose = []
            for row in rows:
                ground = mesh.vertices @ row['T'][:3, :3].T+row['T'][:3, 3]
                feet = ground[np.abs(ground[:, 2]) <= 1e-8, :2]
                passed, gap, _ = cover(row['hull'], np.vstack([feet, row['own']]))
                per_pose.append(dict(pose=row['pose'], passed=passed, gap_m=gap))
            record = dict(component=index, all_ground_enclosures_passed=all(r['passed'] for r in per_pose))
            if not record['all_ground_enclosures_passed']:
                trials.append(record); continue
            core = unpack_solid(nominal ^ part)
            if not len(core.faces):
                record['error'] = 'No nominal contact material in this component'; trials.append(record); continue
            from .family_gate import condition
            try:
                with tempfile.TemporaryDirectory(prefix='cadgrasp-family-component-') as directory:
                    core, correction = condition(m, layout, core, Path(directory)/'core.obj', False, mesh)
                proof = certify_material_contacts(m, layout, core)
                sample_counts = {m.poses[k]: int(mask.sum()) for k, mask in proof['masks'].items()}
                record['sample_counts'] = sample_counts
                if not all(mask.all() for mask in proof['masks'].values()):
                    trials.append(record); continue
                endpoints, failed_endpoints = {}, {}
                for k in layout.active:
                    passed, _ = C.J.classify(proof['supplies'][k], self.loads[k]*m.tasks[k].scale)
                    endpoints[m.poses[k]] = dict(passed=int(passed.sum()), total=len(passed))
                    if not passed.all():
                        failed_endpoints[k] = np.flatnonzero(~passed)
                record['continuous_endpoint_counts'] = endpoints
                if not all(row['passed'] == row['total'] for row in endpoints.values()):
                    m.family_failed_endpoints[layout.key()] = failed_endpoints
                    trials.append(record); continue
                m.family_legal_cores[layout.key()] = dict(mesh=core, proof=proof,
                    endpoints=endpoints, conditioning=correction,
                    discarded_nominal_material_cm3=C.material_volume(nominal-C.S.solid(core))*1e6)
                winner = mesh; winner_rows = per_pose; trials.append(record)
                break
            except (ValueError, RuntimeError, AssertionError) as error:
                record['error'] = f'{type(error).__name__}: {error}'
                trials.append(record)
        if winner is not None:
            self.witness = (layout.key(), winner)
        return dict(passed=winner is not None, connected_passed=winner is not None,
            status='force_certified_common_component' if winner is not None else 'no_force_certified_component_in_window',
            component_per_pose=winner_rows, component_seconds=time.monotonic()-began,
            component_queries=self.component_queries, component_trials=trials,
            force_contacts_replayed=True, normal_pressure_enclosure_only=False,
            every_original_sample_and_continuous_target_checked_on_retained_contact_material=True,
            all_potential_contact_probes_required=False)
