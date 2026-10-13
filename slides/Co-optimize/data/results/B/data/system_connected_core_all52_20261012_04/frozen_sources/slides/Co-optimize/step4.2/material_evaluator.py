"""Complete-layout costs and a headless adapter to the existing solid builder.

The original contact/force kernels and normal-ray Oracle remain authoritative.
Contact roles are diagnostic; no protected contact is deleted. The first
integration plans feet/connections with the existing exact Builder backend.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path
import importlib.util
import numpy as np
from scipy.optimize import linprog

from whole_search.reuse_first import registered
from whole_search.search import failed_count
from fixture_state import installed_state
from system_floor import transform_points, system_floor_demands


def membership(layout):
    reuse = sum(registered(layout, k) for k in layout.active)
    return reuse, len(layout.active)-reuse


def mixed_target_applicable(layout):
    """The requested 2..4 / 2..4 combination applies only to 4..8 tasks."""
    return 4 <= len(layout.active) <= 8


def target_gap(layout, minimum=2, maximum=4):
    if not mixed_target_applicable(layout):
        return 0
    reuse, juxtaposed = membership(layout)
    return sum(max(0, minimum-n, n-maximum) for n in (reuse, juxtaposed))


def mandatory_contact_points(model,layout):
    if not hasattr(model,'contact_delta'):return np.empty((0,3))
    flags=model.contact_delta.state(layout).available
    parts=[transform_points(model.points[flags[k]],layout.placements[k]) for k in layout.active]
    return np.vstack(parts) if parts else np.empty((0,3))


def lower_box(model, layout):
    """Necessary WORLD box including mandatory material in EVERY state."""
    rows = model.base_oracle.rows(layout)
    clouds = []
    for row in rows:
        clouds.append(transform_points(model.mesh.vertices, row['U']))
        clouds.append(np.c_[row['hull'], np.zeros(len(row['hull']))])
    ports=mandatory_contact_points(model,layout)
    if len(ports):
        for host in sorted(set(int(layout.hosts[k]) for k in layout.active)):
            clouds.append(transform_points(ports,installed_state(model.native,layout,host)))
    points = np.vstack(clouds)
    low, high = points.min(0), points.max(0)
    return dict(min_m=low.tolist(), max_m=high.tolist(),
                volume_litres=float(np.prod(high-low)*1000))


def align_world_xy(model, layout, fixture_vertices=None):
    """Eliminate free horizontal installation translations for a fixed shape.

    Work, exits, common material, z=0 and COM reaction rays are invariant to
    this whole-state XY motion. Aligning state box centres attains the smallest
    horizontal union extents for those fixed state shapes; z stays a variable.
    """
    trial = layout.copy()
    if fixture_vertices is None:
        ports=mandatory_contact_points(model,layout)
        if len(ports):fixture_vertices=ports
    for host in sorted(set(int(layout.hosts[k]) for k in layout.active)):
        T = installed_state(model.native, layout, host)
        clouds = [transform_points(model.mesh.vertices, T@layout.placements[k])
                  for k in layout.active if int(layout.hosts[k]) == host]
        if fixture_vertices is not None:
            clouds.append(transform_points(fixture_vertices, T))
        points = np.vstack(clouds)
        trial.state_offsets[host, :2] -= (points[:, :2].min(0)+points[:, :2].max(0))/2
    return trial


def floor_height_proposal(model, layout, max_cuts=12, fixed_hosts=()):
    """Minimum added height satisfying an enlarged necessary floor screen.

    All ORIGINAL pressure demands are used as a cutting-plane separation
    oracle. Ground reactions of the object are omitted here, so failure is
    merely a failed PROPOSAL, never an infeasibility certificate. The original
    BaseOracle (including object ground) judges every proposed layout.
    """
    hosts = sorted(set(int(layout.hosts[k]) for k in layout.active))
    index = {h: j for j, h in enumerate(hosts)}
    bounds = []
    for h in hosts:
        lowest = min(model.workpiece_height(layout, k) for k in layout.active
                     if int(layout.hosts[k]) == h)
        lower = -lowest
        bounds.append((lower, lower if h in fixed_hosts else None))
    fields = []
    rows = model.base_oracle.rows(layout)
    for row in rows:
        k = row['index']; T = row['T']; loads = model.base_oracle.loads[k]
        com = transform_points(model.mesh.center_mass[None], row['U'])[0]
        pressure = system_floor_demands(loads, com)['floor_demands_world_m']
        common = transform_points(pressure, np.linalg.inv(T))
        slopes_world = -np.c_[loads[:, :2]/loads[:, 2, None], np.ones(len(loads))]
        slopes = slopes_world@T[:3, :3]
        for other in hosts:
            if other == int(layout.hosts[k]):
                continue
            B = installed_state(model.native, layout, other)
            fields.append((index[int(layout.hosts[k])], index[other],
                           common@B[2, :3]+B[2, 3], slopes@B[2, :3]))
    constraints, rhs, seen = [], [], set()
    delta = np.array([b[0] for b in bounds])
    for _ in range(max_cuts):
        added = False
        for owner, other, base, slope in fields:
            values = base+slope*delta[owner]+delta[other]
            p = int(values.argmin())
            if values[p] >= -1e-9:
                continue
            key = owner, other, p, float(base[p]), float(slope[p])
            if key in seen:
                continue
            seen.add(key); added = True
            row = np.zeros(len(hosts)); row[owner] -= slope[p]; row[other] -= 1
            constraints.append(row); rhs.append(float(base[p]))
        if not added:
            if fields and min(float((b+a*delta[o]+delta[j]).min())
                              for o, j, b, a in fields) < -1e-8:
                return None
            trial = layout.copy()
            trial.state_offsets[hosts, 2] += delta
            for k in trial.active:
                model.validate_placement(trial, k)
            return trial
        solution = linprog(np.ones(len(hosts)), A_ub=np.asarray(constraints),
                           b_ub=np.asarray(rhs), bounds=bounds, method='highs',
                           options=dict(primal_feasibility_tolerance=1e-9,
                                        dual_feasibility_tolerance=1e-9))
        if not solution.success:
            return None
        delta = solution.x
    return None


@dataclass(eq=False)
class MaterialSample:
    layout: object
    loss: float
    oracle: dict
    lower_litres: float
    operation: str = 'parent'
    parent: str = ''
    detail: dict = field(default_factory=dict)
    result: object = None
    witness: object = None
    nominal: object = None
    solid: object = None
    construction: object = None
    evaluation: object = None

    @property
    def completed(self):
        core=self.result.get('connected_contact_core')if self.result is not None else None
        return (self.result is not None and failed_count(self.result) == 0
                and (core is None or (core.get('connected')and core.get('downward_resultant_passed')
                    and self.result.get('force_material_geometry_certified')))
                and bool(self.oracle.get('connected_passed'))
                and bool(self.oracle.get('passed'))
                and bool(self.construction and self.construction.get('passed')))

    @property
    def metrics(self):
        return (target_gap(self.layout), self.loss,
                int(not self.oracle.get('enclosure_passed')),
                float(self.oracle.get('gap_m', 0)), self.lower_litres)

    @property
    def cost(self):
        if not self.completed:
            return (float('inf'), float('inf'))
        return (self.evaluation['with_full_support']['volume_litres'],
                self.evaluation['material_full_cm3'])


class MaterialEvaluator:
    def __init__(self, search):
        self.search = search; self.model = search.model
        self.connection_attempts = set(); self.build_attempts = set()
        self.builders = None
        self.stats = dict(previews=0, full_demands=0, connections=0, constructions=0,
                          material_contact_certifications=0)

    def space_candidate(self,item):
        if item.result is None:return False
        if getattr(self.model,'require_connected_core',False):
            core=item.result.get('connected_contact_core',{})
            return (failed_count(item.result)==0 and core.get('connected',False)
                and core.get('downward_resultant_passed',False)
                and item.result.get('force_material_geometry_certified',False))
        failure=failed_count(item.result)
        return not failure or (failure<=512 and hasattr(self.model,'mesh') and
            len(item.layout.active)==len(self.model.poses) and all(
                len(item.result['masks'][k])==32768 for k in item.layout.active))

    def preview(self, layout, operation='parent', parent='', detail=None, result=None):
        m = self.model
        for k in layout.active:
            m.validate_placement(layout, k)
        self.stats['previews'] += 1
        oracle = m.base_oracle.query(layout)
        loss = float(m.proxy(layout)['loss'])
        if loss < 1e-28:
            loss = 0.
        if not np.isfinite(loss):
            raise ValueError('Nonfinite whole-demand loss')
        return MaterialSample(layout, loss, oracle, lower_box(m, layout)['volume_litres'],
                              operation, parent, detail or {}, result)

    def classify(self, item):
        if item.result is None:
            self.stats['full_demands'] += 1
            item.result = self.model.evaluate(item.layout)
        if (getattr(self.model,'require_connected_core',False)and
                not item.result.get('material_contact_certification_run')and failed_count(item.result)<=512):
            from connected_core import prepare
            from material_force import certify_material_contacts
            core,record=prepare(self.model,item.layout)
            previous=failed_count(item.result)
            proof=certify_material_contacts(self.model,item.layout,core)
            item.result=dict(item.result,**proof,grid_failed_demands=previous,
                material_contact_certification_run=True,
                force_material_geometry_certified=all(v.all()for v in proof['masks'].values()))
            self.stats['material_contact_certifications']+=1
        return item

    def connect(self, item):
        key = item.layout.key()
        if key in self.connection_attempts or not self.space_candidate(item):
            return item
        if not item.oracle.get('enclosure_passed'):
            return item
        self.connection_attempts.add(key); self.stats['connections'] += 1
        item.oracle = self.model.base_oracle.query(item.layout, connected=True)
        item.result['base_oracle'] = item.oracle
        witness = getattr(self.model.base_oracle, 'witness', (None, None))
        if item.oracle.get('connected_passed') and witness[0] == key:
            item.witness = witness[1]
        return item

    def construct(self, item):
        key = item.layout.key()
        if key in self.build_attempts or item.witness is None:
            return item
        self.build_attempts.add(key); self.stats['constructions'] += 1
        if self.builders is None:
            from whole_step4_render import NominalBuilder
            import material_builder_bounded as backend
            self.builders = NominalBuilder(self.model), backend
        nominal_builder, backend = self.builders
        from whole_search.common import unpack_solid
        if getattr(self.model,'require_connected_core',False):
            from connected_core import prepare
            item.nominal,core_record=prepare(self.model,item.layout)
            if not self.space_candidate(item):return item
        else:
            item.nominal = unpack_solid(nominal_builder.solid(item.layout))
        rows = [dict(pose=r['pose'], fixture_transform=r['T'], object_ground=r['own'],
                     hull=np.c_[r['hull'], np.zeros(len(r['hull']))],
                     object_vertices=transform_points(self.model.mesh.vertices,r['U']))
                for r in self.model.base_oracle.rows(item.layout)]
        try:
            item.solid, item.construction = backend.Builder(item.nominal, item.witness, rows).run()
            if not item.construction['passed'] or not item.construction['watertight']:
                item.construction['passed'] = False
                return item
            if failed_count(item.result):
                # Point quadrature is a sufficient screen. A near miss can
                # still have enough real contact polygon vertices. Certify
                # every original demand on the actual solid before acceptance.
                from material_force import certify_material_contacts
                previous=failed_count(item.result)
                proof=certify_material_contacts(self.model,item.layout,item.solid)
                self.stats['material_contact_certifications']+=1
                item.result=dict(item.result,**proof,grid_failed_demands=previous,
                    material_contact_certification_run=True,
                    force_material_geometry_certified=all(v.all()for v in proof['masks'].values()))
                item.detail=dict(item.detail,grid_failed_demands=previous,
                    material_contact_certification=True)
            import sys
            folder = str(Path(__file__).resolve().parents[1]/'evaluation')
            if folder not in sys.path:
                sys.path.insert(0, folder)
            from metrics import static_bounds
            item.evaluation = static_bounds(self.model.mesh.vertices,
                item.layout.arrays(self.model.native), item.nominal.vertices,
                item.solid.vertices, self.model.poses)
            item.evaluation['material_full_cm3'] = abs(float(item.solid.volume))*1e6
            # Exact horizontal translation symmetry: Q, directions, rotations,
            # all installed floor planes and relative reaction rays stay fixed.
            # Optimise actual complete state boxes, rather than object boxes.
            aligned=align_world_xy(self.model,item.layout,item.solid.vertices)
            if aligned.key()!=item.layout.key():
                assert np.array_equal(aligned.placements,item.layout.placements)
                assert np.array_equal(aligned.directions,item.layout.directions)
                assert np.array_equal(aligned.hosts,item.layout.hosts)
                hosts=set(int(aligned.hosts[k])for k in aligned.active)
                assert all(np.array_equal(installed_state(self.model.native,aligned,h)[2],
                    installed_state(self.model.native,item.layout,h)[2])for h in hosts)
                fast=self.model.base_oracle.query(aligned)
                if fast.get('enclosure_passed'):
                    accepted=dict(fast,passed=True,connected_passed=True,
                        status='common_connected_enclosure_witness',
                        component_per_pose=item.oracle.get('component_per_pose',[]),
                        connection_witness_reused_under_world_xy_translation=True)
                    adjusted=replace(item,layout=aligned,oracle=accepted,
                        result=dict(item.result,layout=aligned,base_oracle=accepted),
                        lower_litres=lower_box(self.model,aligned)['volume_litres'],
                        detail=dict(item.detail,installation_xy_optimized=True))
                    adjusted.evaluation=static_bounds(self.model.mesh.vertices,
                        aligned.arrays(self.model.native),item.nominal.vertices,
                        item.solid.vertices,self.model.poses)
                    adjusted.evaluation['material_full_cm3']=item.evaluation['material_full_cm3']
                    self.model.base_oracle.cache[aligned.key(),True]=accepted
                    self.connection_attempts.add(aligned.key());self.build_attempts.add(aligned.key())
                    return adjusted
        except (ValueError, RuntimeError) as error:
            item.construction = dict(passed=False, error=str(error))
        return item
