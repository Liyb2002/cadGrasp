"""Complete-layout costs and a headless adapter to the existing solid builder.

The original contact/force kernels and normal-ray Oracle remain authoritative.
Contact roles are diagnostic; no protected contact is deleted. The first
integration plans feet/connections with the existing exact Builder backend.
"""
from dataclasses import dataclass, field
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


def target_gap(layout, minimum=2, maximum=4):
    reuse, juxtaposed = membership(layout)
    return sum(max(0, minimum-n, n-maximum) for n in (reuse, juxtaposed))


def lower_box(model, layout):
    """Necessary WORLD box: actual objects and required ground-pressure points."""
    rows = model.base_oracle.rows(layout)
    clouds = []
    for row in rows:
        clouds.append(transform_points(model.mesh.vertices, row['U']))
        clouds.append(np.c_[row['hull'], np.zeros(len(row['hull']))])
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
        return (self.result is not None and failed_count(self.result) == 0
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
        self.stats = dict(previews=0, full_demands=0, connections=0, constructions=0)

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
        return item

    def connect(self, item):
        key = item.layout.key()
        if key in self.connection_attempts or item.result is None or failed_count(item.result):
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
            path = Path(__file__).resolve().parents[1]/'step5.2/deterministic_base.py'
            spec = importlib.util.spec_from_file_location('coopt_material_builder', path)
            backend = importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
            self.builders = NominalBuilder(self.model), backend
        nominal_builder, backend = self.builders
        from whole_search.common import unpack_solid
        item.nominal = unpack_solid(nominal_builder.solid(item.layout))
        rows = [dict(pose=r['pose'], fixture_transform=r['T'], object_ground=r['own'],
                     hull=np.c_[r['hull'], np.zeros(len(r['hull']))])
                for r in self.model.base_oracle.rows(item.layout)]
        try:
            item.solid, item.construction = backend.Builder(item.nominal, item.witness, rows).run()
            if not item.construction['passed'] or not item.construction['watertight']:
                item.construction['passed'] = False
                return item
            import sys
            folder = str(Path(__file__).resolve().parents[1]/'evaluation')
            if folder not in sys.path:
                sys.path.insert(0, folder)
            from metrics import static_bounds
            item.evaluation = static_bounds(self.model.mesh.vertices,
                item.layout.arrays(self.model.native), item.nominal.vertices,
                item.solid.vertices, self.model.poses)
            item.evaluation['material_full_cm3'] = abs(float(item.solid.volume))*1e6
        except (ValueError, RuntimeError) as error:
            item.construction = dict(passed=False, error=str(error))
        return item
