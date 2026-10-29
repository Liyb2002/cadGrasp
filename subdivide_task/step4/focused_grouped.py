"""Actual connected-contact counterexamples guide poses; full guards accept them."""
from pathlib import Path
import tempfile
import time
import numpy as np

from subdivide_task.step3.legacy import kernel as C
from subdivide_task.step1.common import write_json
from .compact_grouped import CompactGroupedSearch, execution_box, actual_failure_rank
from .grouped_search import memberships
from .grouped_placement import fit_groups
from .family_search import FamilySearch
from .family_gate import condition, FamilyEvaluator
from whole_search.common import transform_solid, transform_mesh, unpack_solid
from whole_step4_render import NominalBuilder
from prune_support import material_components
from base_oracle import cover
from fixture_state import installed_state
from translation.layout_update import shift


def spin_with_world_exit(model, layout, owner, angle):
    model.activate(layout)
    before = installed_state(model.native, layout, int(layout.hosts[owner]))
    world_exit = before[:3,:3] @ layout.directions[owner]
    trial = model.spin_guest(layout, owner, angle)
    after = installed_state(model.native, trial, int(trial.hosts[owner]))
    trial.directions[owner] = after[:3,:3].T @ world_exit
    return trial


class ConnectedContactProbe:
    """Reduced-demand guidance on exact legal contact polygons, never a PASS gate.

    Full pressure hulls, object/exit/work exclusions and connected stock are
    unchanged. Only the demand batch used for ranking is reduced. A separate
    unmodified full certificate must accept every proposed final fixture.
    """
    def __init__(self, model):
        self.model = model; self.endpoints = {}; self.queries = 0
        self.samples = {k: np.minimum(len(task.targets)-1, ((np.arange(32)+.5)*len(task.targets)/32).astype(int))
                        for k, task in enumerate(model.tasks)}

    def learn(self, layout, item):
        m = self.model
        for k, ids in m.family_failed_endpoints.get(layout.key(), {}).items():
            self.endpoints[k] = np.unique(np.r_[self.endpoints.get(k, []), ids]).astype(int)
        for k, mask in (item.result or {}).get('masks', {}).items():
            if not mask.all():
                self.samples[k] = np.arange(len(m.tasks[k].targets))
        for row in item.oracle.get('component_trials', []):
            for k in layout.active:
                if row.get('sample_counts', {}).get(m.poses[k], 32768) < 32768:
                    self.samples[k] = np.arange(len(m.tasks[k].targets))

    def core(self, layout):
        m = self.model; m.activate(layout); rows = m.base_oracle.rows(layout)
        points = [m.mesh.vertices @ layout.placements[k, :3, :3].T+layout.placements[k, :3, 3] for k in layout.active]
        for row in rows:
            t = np.linalg.inv(row['T']); p = np.c_[row['hull'], np.zeros(len(row['hull']))]
            points.append(p @ t[:3, :3].T+t[:3, 3])
        joined = np.vstack(points); low = joined.min(0)-m.base_oracle.padding; high = joined.max(0)+m.base_oracle.padding
        box = C.trimesh.creation.box(high-low); box.apply_translation((high+low)/2)
        stock = C.S.solid(box)
        for row in rows:
            t = row['T']; stock = stock.trim_by_plane(tuple(t[2, :3]), float(-t[2, 3]/C.S.SCALE))
        bounds = np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])
        obstacles = []
        for task in m.base_oracle.pose_constraints(layout).poses:
            k, q = task.index, task.Q_fixture_object
            obstacles.append(transform_solid(task.work_access.solid(), q))
            direction = q[:3,:3].T @ layout.directions[k]
            local = (bounds-q[:3,3]) @ q[:3,:3]
            length = max(m.length, float((local @ direction).max()-(m.mesh.vertices @ direction).min())+.01)
            obstacles.append(transform_solid(m.clearance.sweep(length*direction, padded=False), q))
        free = stock-C.union(obstacles); nominal = NominalBuilder(m).solid(layout)
        for part in material_components(free):
            mesh = unpack_solid(part)
            if not all(cover(row['hull'], np.vstack([(mesh.vertices @ row['T'][:3,:3].T+row['T'][:3,3])[np.abs((mesh.vertices @ row['T'][:3,:3].T+row['T'][:3,3])[:,2])<=1e-8,:2], row['own']]))[0] for row in rows):
                continue
            core = unpack_solid(nominal ^ part)
            with tempfile.TemporaryDirectory(prefix='cadgrasp-focused-core-') as temporary:
                core, _ = condition(m, layout, core, Path(temporary)/'core.obj', False, mesh)
            return core
        raise ValueError('No connected legal material enclosing every full pressure hull')

    def score(self, layout):
        began = time.monotonic(); m = self.model; self.queries += 1
        core = self.core(layout); sample_bad = end_bad = 0; failures = {}
        for k in sorted(set(self.endpoints) | set(self.samples)):
            q = layout.placements[k]; direction = q[:3,:3].T @ layout.directions[k]
            allowed = m.allowed[k][m.mesh.face_normals[m.allowed[k]] @ direction <= 1e-9]
            triangles, sources = C.contact_boundary(transform_mesh(m.mesh,q),core,allowed)
            mapping = m.native[k] @ np.linalg.inv(q)
            points = C.transform_points(triangles.reshape(-1,3),mapping)
            normals = np.repeat(-m.tasks[k].domain.mesh.face_normals[sources],3,axis=0)
            raw = np.c_[normals,np.cross(points-m.tasks[k].domain.com,normals)]
            rays = C.I.merge_columns(m.floor_columns(layout,k),C.U.heads(raw,m.tasks[k].scale))
            bad_s = bad_e = 0
            if k in self.samples:
                mask,_ = C.J.classify(rays,m.tasks[k].targets[self.samples[k]]); bad_s = int((~mask).sum())
            if k in self.endpoints:
                mask,_ = C.J.classify(rays,m.base_oracle.loads[k][self.endpoints[k]]*m.tasks[k].scale); bad_e = int((~mask).sum())
            failures[k] = (bad_s,bad_e); sample_bad += bad_s; end_bad += bad_e
        occupied,_ = execution_box(m,layout)
        return dict(rank=(sample_bad,end_bad,occupied),failures=failures,seconds=time.monotonic()-began,
                    guidance_only=True,full_certificate_required=True)


class FocusedEvaluator(FamilyEvaluator):
    force_geometry = False

    def space_candidate(self, item):
        if self.force_geometry:
            return item.result is not None
        return super().space_candidate(item)


class FocusedGroupedSearch(CompactGroupedSearch):
    def __init__(self, model, folder, group_count=5):
        super().__init__(model, folder, group_count)
        self.evaluator = FocusedEvaluator(self.context)

    def certify(self,q,label,_existing=None):
        item = _existing if _existing is not None else FamilySearch.certify(self,q,label)
        if self.accepted is not None or item is None:
            return item
        probe = ConnectedContactProbe(self.model); probe.learn(q,item)
        if not probe.endpoints and not probe.samples:
            return item
        m = self.model; current = q.copy(); best_item = item
        for sweep in range(8):
            try: best = probe.score(current)
            except (ValueError,RuntimeError,AssertionError): return best_item
            owners = [k for k, failure in best['failures'].items() if any(failure)]
            proposals = []
            for owner in owners:
                peers = [k for k in memberships(current)[int(current.hosts[owner])] if k != owner]
                # Explore the full axial family without silently tilting the
                # independent withdrawal direction along with the object.
                for member in list(reversed(peers))+[owner]:
                    for angle in [180.,90.,-90.,60.,-60.,30.,-30.]:
                        proposals.append((spin_with_world_exit(m,current,member,current.angles_deg[member]+angle),
                            dict(owner=m.poses[member],delta_angle_deg=angle,world_exit_preserved=True)))
                # Heights of co-served objects can expose a contact strip while
                # preserving the failing task's pose and useful floor reaction.
                for member in list(reversed(peers))+[owner]:
                    for height in [.001,.003,.005,.01,.02,.03,0.]:
                        m.activate(current); t = installed_state(m.native,current,int(current.hosts[member]))
                        delta = height-m.workpiece_height(current,member)
                        proposals.append((shift(m,current,member,t[:3,:3].T@[0.,0.,delta]),
                                          dict(owner=m.poses[member],height_m=height)))
                for member in list(reversed(peers))+[owner]:
                    for angle in [1.,-1.,3.,-3.,8.,-8.,20.,-20.]:
                        proposals.append((spin_with_world_exit(m,current,member,current.angles_deg[member]+angle),
                                          dict(owner=m.poses[member],delta_angle_deg=angle)))
            for radius in [.005,.015,.03,.05]:
                proposals.append((self.spread(current,radius),dict(group_clearance_m=radius)))
            improved = False
            for trial,detail in proposals:
                try:
                    trial,_ = fit_groups(m,trial,preserve=True)
                    self.group_invariant(trial)
                    value = probe.score(trial)
                except (ValueError,RuntimeError,AssertionError):
                    continue
                self.save(trial,'focused-connected-contact-probe',sweep=sweep,rank=list(value['rank']),
                          probe_seconds=value['seconds'],guidance_only=True,**detail)
                if value['rank'] < best['rank']:
                    current,best = trial,value; improved = True
                if value['rank'][:2] == (0,0):
                    # Do not reject a promising exact-polygon probe merely
                    # because point quadrature underestimates its contacts.
                    # The real full 425984 + continuous tests remain mandatory.
                    self.evaluator.force_geometry = True
                    try:
                        candidate = FamilySearch.certify(self,trial,'focused-proposal-full-original-certificate')
                    finally:
                        self.evaluator.force_geometry = False
                    if self.accepted is not None:
                        return candidate
                    if candidate is not None:
                        probe.learn(trial,candidate); best_item=candidate
                        current=trial; improved=True
                    break
            if not improved:
                break
        return best_item
