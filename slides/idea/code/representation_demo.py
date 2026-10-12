"""Small algebraic examples for the proposed shared-material representation.

This enumerates tiny fixed-layout problems, not CAD shapes or saved B demands.
Port/legality/adjacency data are assumed inputs. Production needs certified
contact interfaces, clipped volumes and geometric connection witnesses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.optimize import nnls
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class Port:
    pose: str
    position: tuple[float, float, float]
    center_of_mass: tuple[float, float, float]
    normal: tuple[float, float, float] = (0., 0., 1.)

    def wrench(self):
        normal = np.asarray(self.normal, float)
        if not np.isclose(np.linalg.norm(normal), 1.):
            raise ValueError('A port needs a unit admissible reaction direction')
        return np.r_[normal, np.cross(np.asarray(self.position)-self.center_of_mass, normal)]


@dataclass(frozen=True)
class Cell:
    name: str
    volume_cm3: float
    legal_for: frozenset[str]
    ports: tuple[Port, ...] = ()
    neighbors: frozenset[str] = frozenset()
    ground_xy: dict[str, tuple] = field(default_factory=dict)


@dataclass
class Problem:
    cells: tuple[Cell, ...]
    demands: dict[str, np.ndarray]
    ground_demands: dict[str, np.ndarray] = field(default_factory=dict)
    mandatory_cells: frozenset[str] = frozenset()
    require_connected: bool = False
    geometry_poses: frozenset[str] | None = None


def demand_residual(rays, target):
    target = np.asarray(target, float)
    rays = np.asarray(rays, float).reshape(-1, len(target))
    if not len(rays):
        return float(np.linalg.norm(target))
    # Positive column scaling does not change an unbounded reaction cone.
    lengths = np.linalg.norm(rays, axis=1)
    rays = rays[lengths > 0]/lengths[lengths > 0, None]
    if not len(rays):
        return float(np.linalg.norm(target))
    return float(nnls(rays.T, target)[1])


def connected(selected):
    if not selected:
        return False
    by_name = {c.name: c for c in selected}
    seen = {selected[0].name}
    pending = list(seen)
    while pending:
        for name in by_name[pending.pop()].neighbors:
            if name in by_name and name not in seen:
                # The toy graph assumes geometrically certified edges.
                seen.add(name)
                pending.append(name)
    return len(seen) == len(selected)


def ground_contains(points, targets):
    points = np.unique(np.asarray(points, float).reshape(-1, 2), axis=0)
    if len(points) < 3:
        return False
    if np.linalg.matrix_rank(points-points.mean(0)) < 2:
        return False
    equations = ConvexHull(points).equations
    return bool(np.all(np.asarray(targets)@equations[:, :2].T+equations[:, 2] <= 1e-10))


def evaluate(problem, selected):
    poses = set(problem.geometry_poses if problem.geometry_poses is not None else problem.demands)
    legal = all(poses <= set(c.legal_for) for c in selected)
    names = {c.name for c in selected}
    mandatory = problem.mandatory_cells <= names
    residuals = {}
    for pose, demands in problem.demands.items():
        rays = [p.wrench() for c in selected for p in c.ports if p.pose == pose]
        residuals[pose] = max(demand_residual(rays, b) for b in np.atleast_2d(demands))
    ground = {}
    for pose, demands in problem.ground_demands.items():
        points = [p for c in selected for p in c.ground_xy.get(pose, ())]
        ground[pose] = ground_contains(points, demands)
    connectivity = connected(selected) if problem.require_connected else True
    return dict(selected=sorted(names), material_cm3=sum(c.volume_cm3 for c in selected),
                geometric_legal=legal, mandatory_retained=mandatory,
                maximum_reaction_residual=residuals, ground_enclosure=ground,
                connected=connectivity,
                passed=legal and mandatory and connectivity and all(ground.values())
                       and all(r < 1e-10 for r in residuals.values()))


def solve(problem):
    if len(problem.cells) > 12:
        raise ValueError('This exact enumeration is only for tiny examples (at most 12 cells)')
    names = [c.name for c in problem.cells]
    if len(set(names)) != len(names):
        raise ValueError('Shared material cell IDs must be unique')
    if any(c.volume_cm3 <= 0 for c in problem.cells):
        raise ValueError('Cells need positive material volume')
    known = set(names)
    if not problem.mandatory_cells <= known:
        raise ValueError('Unknown mandatory material')
    if problem.geometry_poses is not None and not set(problem.demands) <= problem.geometry_poses:
        raise ValueError('Every force task must be included in the geometric scope')
    for cell in problem.cells:
        if not cell.neighbors <= known:
            raise ValueError('Unknown adjacency endpoint')
        for neighbor in cell.neighbors:
            other = problem.cells[names.index(neighbor)]
            if cell.name not in other.neighbors:
                raise ValueError('Undirected material adjacency must be symmetric')
    best = None
    checked = 0
    for count in range(len(problem.cells)+1):
        for selected in combinations(problem.cells, count):
            row = evaluate(problem, selected)
            checked += 1
            if row['passed'] and (best is None or (row['material_cm3'], row['selected']) <
                                  (best['material_cm3'], best['selected'])):
                best = row
    if best is None:
        return dict(passed=False, checked_subsets=checked)
    return dict(best, checked_subsets=checked)


def examples():
    legal = frozenset({'A', 'B'})
    a = Port('A', (0., 0., 0.), (1., 0., 2.))
    b = Port('B', (0., 0., 0.), (-1., 0., 2.))
    demands = {'A': a.wrench()[None], 'B': b.wrench()[None]}
    shared = Cell('shared', 1.5, legal, (a, b))
    private_a = Cell('A_only', 1., legal, (Port('A', (0., 0., 1.), (1., 0., 2.)),))
    private_b = Cell('B_only', 1., legal, (Port('B', (0., 0., .5), (-1., 0., 2.)),))

    first = solve(Problem((shared, private_a, private_b), demands))
    separate = {pose: solve(Problem((shared, private_a, private_b), {pose: target}, geometry_poses=legal))
                for pose, target in demands.items()}
    assert first['selected'] == ['shared'] and first['material_cm3'] == 1.5
    assert all(r['material_cm3'] == 1. for r in separate.values())

    forbidden = Cell('tempting_but_forbidden_for_B', .25, frozenset({'A'}), (a, b))
    second = solve(Problem((shared, private_a, private_b, forbidden), demands))
    assert second['selected'] == ['shared']
    forbidden_alone = evaluate(Problem((forbidden,), demands), (forbidden,))
    assert not forbidden_alone['passed'] and not forbidden_alone['geometric_legal']
    assert all(r < 1e-10 for r in forbidden_alone['maximum_reaction_residual'].values())
    private_under_global_geometry = solve(Problem((shared, private_a, private_b, forbidden),
        {'A': demands['A']}, geometry_poses=legal))
    assert private_under_global_geometry['selected'] == ['A_only']

    weak = Cell('overlap_without_the_needed_moments', .1, legal,
        (Port('A', (1., 0., 0.), (1., 0., 2.)),
         Port('B', (1., 0., 0.), (-1., 0., 2.))))
    third = solve(Problem((shared, private_a, private_b, weak), demands))
    weak_alone = evaluate(Problem((weak,), demands), (weak,))
    assert third['selected'] == ['shared'] and not weak_alone['passed']

    epsilon = 1e-6
    full_residual = demand_residual([a.wrench()], demands['A'][0])
    scaled_residual = demand_residual([epsilon*a.wrench()], demands['A'][0])
    zero_residual = demand_residual([np.zeros(6)], demands['A'][0])
    assert full_residual < 1e-10 and scaled_residual < 1e-10 and zero_residual > .1

    rectangle = ((-2., -1.), (2., -1.), (2., 1.), (-2., 1.))
    required = np.array([[-1., -.5], [1., -.5], [1., .5], [-1., .5]])
    head = Cell('head', 1., legal, (a, b), frozenset({'link'}))
    link = Cell('link', .2, legal, (), frozenset({'head', 'foot'}))
    foot = Cell('foot', .4, legal, (), frozenset({'link'}), {'A': rectangle, 'B': rectangle})
    linked_problem = Problem((head, link, foot), demands,
        {'A': required, 'B': required}, frozenset({'head'}), True)
    fifth = solve(linked_problem)
    disconnected = evaluate(linked_problem, (head, foot))
    assert fifth['selected'] == ['foot', 'head', 'link']
    assert np.isclose(fifth['material_cm3'], 1.6)
    assert all(disconnected['ground_enclosure'].values()) and not disconnected['connected']
    assert not disconnected['passed']

    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([2., 3., 4.])
    world = Port('A', tuple(rotation@a.position+translation),
                 tuple(rotation@a.center_of_mass+translation), tuple(rotation@a.normal))
    w = world.wrench()
    pulled_back = np.r_[rotation.T@w[:3], rotation.T@w[3:]]
    assert np.allclose(pulled_back, a.wrench(), atol=1e-12)

    return dict(complete=True, scope='six synthetic algebraic examples; no CAD geometry acceptance',
        actual_B_demands_run=False, production_search_changed=False, production_contact_selection_changed=False,
        actual_solid_volume_measured=False,
        shared_material=dict(joint=first, separate=separate,
            separate_material_sum_cm3=sum(r['material_cm3'] for r in separate.values()),
            shared_port_wrenches={p.pose:p.wrench().tolist() for p in (a,b)}),
        geometric_conflict=dict(solution=second, rejected_cell=forbidden_alone,
            private_A_with_all_task_geometry=private_under_global_geometry),
        mechanical_conflict=dict(solution=third, insufficient_cell=weak_alone),
        positive_density_trap=dict(epsilon=epsilon, full_residual=full_residual,
            positive_scaled_residual=scaled_residual, zero_column_residual=zero_residual,
            invalid_relaxed_cost_cm3=epsilon*shared.volume_cm3,
            binary_activation_cost_cm3=shared.volume_cm3),
        ground_and_connectivity=dict(solution=fifth, disconnected_candidate=disconnected),
        common_frame=dict(common_wrench=a.wrench().tolist(), world_wrench=world.wrench().tolist(),
            pulled_back_wrench=pulled_back.tolist(), torque_reference='pose center of mass', passed=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=Path(__file__).resolve().parents[1]/'data/representation_demo.json')
    args = parser.parse_args()
    report = examples()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print('SIX REPRESENTATION EXAMPLES PASSED')
    print('Material: separate 2.0 cm3; shared 1.5 cm3')
    print('Geometric conflict, missing moment, density trap, connectivity and frame invariance checked')
    print(args.output)


if __name__ == '__main__':
    main()
