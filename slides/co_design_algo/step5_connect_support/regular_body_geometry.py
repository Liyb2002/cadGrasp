"""Replace a selected material network by a few planar, carved local hulls.

The unified solver has already selected the feet and connections. This stage
only adds legal material around that network; it never assigns a head to a
preselected pose or invents a separate ground frame.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from step5_connect_support import material_graph as M
from step5_connect_support import build_coupled_saddle as S
from step5_connect_support.material_network import connected_selection


def reconstruct(graph, selected, context, max_fill_ratio=8.):
    if not np.asarray(selected)[graph['heads']].all() or not connected_selection(selected, graph['edges'], len(selected)):
        raise RuntimeError('Cannot regularize a disconnected or incomplete material seed')
    ids = np.flatnonzero(selected)
    mapping = np.full(len(selected), -1, int); mapping[ids] = np.arange(len(ids))
    edge = graph['edges'][selected[graph['edges']].all(1)]
    centers = np.array([v.mean(0) for v in graph['vertices']])
    lengths = np.linalg.norm(centers[edge[:, 0]]-centers[edge[:, 1]], axis=1)
    a, b = mapping[edge[:, 0]], mapping[edge[:, 1]]
    network = coo_matrix((np.r_[lengths, lengths], (np.r_[a, b], np.r_[b, a])), shape=(len(ids), len(ids))).tocsr()
    distance, _, sources = dijkstra(network, directed=False, indices=mapping[graph['heads']],
        min_only=True, return_predecessors=True)
    if not np.isfinite(distance).all():
        raise RuntimeError('Cannot regularize a disconnected material seed')
    owner = np.full(len(selected), -1, int); owner[ids] = ids[sources]
    parts = []; records = []; splits = []
    costs = graph['cost'].copy()
    for head in graph['heads']:
        costs[head] = context['heads'][head].volume()*S.SCALE**3*1e6
    pending = [(head, ids[owner[ids] == head]) for head in graph['heads']]
    while pending:
        head, cells = pending.pop(0)
        points = np.vstack([graph['vertices'][i] if i in graph['heads'] else np.round(graph['vertices'][i], 12) for i in cells])
        hull = S.solid(S.G.hull_mesh(points))
        body = M.trim_floors(hull, context['bases'], context['offsets'])-context['forbidden']
        for item in cells:
            if item in graph['heads']: body = body+context['heads'][item]
        volume = float(body.volume()*S.SCALE**3*1e6)
        seed_cost = float(costs[cells].sum())
        # Avoid filling the empty space between remote branches into a slab.
        # Split an overly inflated hull into two connected geodesic regions.
        if len(cells) > 1 and volume > max_fill_ratio*seed_cost:
            sub = network[mapping[cells]][:, mapping[cells]]
            first = int(np.argmax(dijkstra(sub, directed=False, indices=0)))
            second = int(np.argmax(dijkstra(sub, directed=False, indices=first)))
            _, _, nearest = dijkstra(sub, directed=False, indices=[first, second], min_only=True, return_predecessors=True)
            left, right = cells[nearest == first], cells[nearest == second]
            if not len(left) or not len(right): raise RuntimeError('Degenerate regular-body split')
            splits.append(dict(head_node=int(head), old_volume_cm3=volume,
                seed_cost_cm3=seed_cost, groups=[left.tolist(), right.tolist()]))
            pending.extend([(head, left), (head, right)])
            continue
        parts.append(body)
        records.append(dict(head_node=int(head), selected_material_nodes=cells.tolist(),
            hull_volume_cm3=float(hull.volume()*S.SCALE**3*1e6),
            carved_volume_cm3=volume, seed_cost_cm3=seed_cost, fill_ratio=volume/seed_cost))
    full = M.union(parts)
    seed = M.material_solid(graph, selected)
    missing = float((seed-full).volume()*S.SCALE**3)
    if missing > 8e-14:
        raise RuntimeError(f'Local hull reconstruction lost {missing} m3 of the selected network')
    full, regularization = M.regularize_export(full, context['heads'])
    mesh = S.unpack(full)
    return full, dict(method='geodesic_head_regions_then_carved_convex_hulls',
        original_network_preserved=True, missing_seed_volume_m3=missing,
        head_floor_assignments_changed=False, local_hulls=records,
        maximum_local_fill_ratio=max_fill_ratio, excessive_fill_splits=splits,
        seed_volume_cm3=float(seed.volume()*S.SCALE**3*1e6),
        added_volume_cm3=float((full.volume()-seed.volume())*S.SCALE**3*1e6),
        output_volume_cm3=float(mesh.volume*1e6), export_regularization=regularization,
        cosmetic_mesh_smoothing=False, strength_verified=False)
