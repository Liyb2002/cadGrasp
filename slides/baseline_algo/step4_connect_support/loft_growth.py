"""Unified connected greedy selection of whole lofts, priced by their union."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse.csgraph import dijkstra

from step4_connect_support import material_graph as M, material_network as N
from step4_connect_support.loft_candidates import volume


def enable_whole_body_growth(graph, context):
    graph['body_context'] = context
    graph['compiled_lofts'] = {}


def selection_solid(graph, selected):
    if 'body_context' not in graph:
        return M.union(graph['solids'][i] for i in np.flatnonzero(selected))
    from step4_connect_support.loft_candidates import carve, component_with
    from step4_connect_support import build_coupled_saddle as S
    context = graph['body_context']; parts = []; groups = {}
    for i in np.flatnonzero(selected):
        record = graph['records'][i]
        if record['kind'] == 'head_floor_loft':
            groups.setdefault((record['head'],record['floor']),[]).append(int(i))
        else: parts.append(graph['solids'][i])
    for (head,floor), nodes in groups.items():
        if len(nodes) == 1:
            parts.append(graph['solids'][nodes[0]]);continue
        key = tuple(nodes)
        if key not in graph['compiled_lofts']:
            v = graph['vertices'][head]; record = context['records'][head]
            pose = list(dict.fromkeys(row['head_pose'] for row in context['records'])).index(record['head_pose'])
            root = v+.008*context['directions'][pose]@context['bases'][pose]
            b,o = context['bases'][floor],context['offsets'][floor]
            points = [v,root]
            for node in nodes:
                xy = np.asarray(graph['records'][node]['polygon_xy_m'])
                pad = np.c_[xy,np.zeros(len(xy))]@b+o
                points += [pad,pad+.003*b[2]]
            hull = S.solid(S.G.hull_mesh(np.vstack(points)))
            sources = [graph['solids'][i] for i in nodes]
            # Analytically each source is contained in the carved hull. Retain
            # the exact sources too, avoiding loss at coplanar Boolean seams.
            merged = component_with(carve(hull,context)+M.union(sources),sources)
            if merged is None: raise RuntimeError('Merged whole loft lost one of its source branches')
            graph['compiled_lofts'][key] = merged
        parts.append(graph['compiled_lofts'][key])
    return M.union(parts)


def marginal_volume(addition, current, fixed_heads):
    """Shared and mandatory material is charged only once, including in paths."""
    return max(0., volume((addition-current)-fixed_heads))


def greedy(graph, groups, root, seed=None):
    count = len(graph['cost']); selected = np.zeros(count, bool) if seed is None else seed.copy()
    selected[root] = True
    if not N.connected_selection(selected, graph['edges'], count): raise RuntimeError('Disconnected loft growth seed')
    full = selection_solid(graph, selected)
    fixed_heads = M.union(graph['solids'][h] for h in graph['heads'])
    incidence = np.zeros((count, len(groups)), bool)
    for j, group in enumerate(groups): incidence[group.members, j] = True
    weights = N.group_weights(groups); history = []
    head_neighbors = {h:np.unique(graph['edges'][np.any(graph['edges'] == h, axis=1)]) for h in graph['heads']}
    for iteration in range(len(groups)+1):
        hit = incidence[selected].any(0)
        if hit.all(): return selected, history
        # Individual volumes propose paths. Every proposed path is re-priced
        # by its exact Boolean union before an action is chosen.
        proxy = np.where(selected, 0., graph['cost'])
        distance, pred, _ = dijkstra(N.adjacency(count, graph['edges'], proxy), directed=True,
            indices=np.flatnonzero(selected), min_only=True, return_predecessors=True)
        targets = set()
        for group, done in zip(groups, hit):
            if done: continue
            ids = group.members[np.isfinite(distance[group.members])]
            if not len(ids): raise RuntimeError('A required loft group is unreachable')
            targets.update(ids[np.argsort(distance[ids])[:3]].tolist())
        proposals = {}; path_records = []
        for target in sorted(targets):
            path = []; current = target
            while not selected[current]:
                path.append(current); current = int(pred[current])
                if current < 0 or len(path) > count: raise RuntimeError('Invalid loft path')
            proposed = selected.copy(); proposed[path] = True
            # A loft can attach another immutable head as a side effect.
            for head, neighbors in head_neighbors.items():
                if proposed[neighbors].any(): proposed[head] = True
            added = tuple(np.flatnonzero(proposed & ~selected))
            if added in proposals: continue
            new_groups = incidence[proposed].any(0) & ~hit
            gain = float(weights[new_groups].sum())
            proposals[added] = (proposed, new_groups, gain)
            path_records.append(added)
        def evaluate(added):
            proposed, new_groups, gain = proposals[added]
            proposed_full = selection_solid(graph,proposed)
            cost = marginal_volume(proposed_full, full, fixed_heads)
            return (cost/gain, cost, added), proposed, new_groups, proposed_full
        with ThreadPoolExecutor(max_workers=4) as pool:
            choices = list(pool.map(evaluate, path_records))
        key, selected, new_groups, full = min(choices, key=lambda row:row[0])
        history.append(dict(iteration=iteration, added_nodes=list(map(int,key[2])),
            added_union_volume_cm3=key[1], satisfied_groups=np.flatnonzero(new_groups).tolist(),
            exact_paths_compared=len(choices)))
    raise RuntimeError('Loft greedy did not finish')


def solve_cover(graph, demands, groups=None, seed=None, max_rounds=24, roots=None):
    if groups is None:
        groups = [N.Group(np.array([h]), 'head') for h in graph['heads']]
        for pose in (0,1):
            for a in np.arange(16)*2*np.pi/16:
                N.add_direction(groups, graph['floors'], demands, pose, [np.cos(a),np.sin(a)], .002)
    else: groups = list(groups)
    history = []
    for iteration in range(max_rounds):
        solutions = []
        for root in (roots or graph['heads']) if seed is None else graph['heads'][:1]:
            selected, steps = greedy(graph, groups, root, seed)
            selected = N.prune(selected, graph['cost'], graph['edges'], groups, graph['heads'], seed)
            full = selection_solid(graph, selected)
            checks = [N.containment(N.floor_points(selected, graph['floors'], k), demands[k]) for k in (0,1)]
            solutions.append((volume(full), selected, root, steps, checks))
            print('Loft greedy root',root,'volume',volume(full),'coverage',[c['passed'] for c in checks],flush=True)
        solutions.sort(key=lambda r:r[0])
        complete = [r for r in solutions if all(c['passed'] for c in r[4])]
        history.append(dict(round=iteration, groups=len(groups), root_volumes_cm3=[r[0] for r in solutions]))
        if complete: return complete, groups, history
        changed = False
        for k, check in enumerate(solutions[0][4]):
            for normal in check['normals']:
                changed |= N.add_direction(groups, graph['floors'], demands, k, normal, .002)
        if not changed: raise RuntimeError('The finite loft library cannot resolve floor coverage')
    raise RuntimeError('Loft ground-separation round limit')
