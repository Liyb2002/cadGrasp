"""Expand and partition a convex ground ring, then verify separate insertions."""
from step1.needs import COORD
from itertools import combinations
from heapq import heappush,heappop
from collections import Counter
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.ops import unary_union
from step2_local_support import insertion as D
from step5_connect_support import ground as G,solids as S,motion as M,ring as R,routing as T


def candidate_angles(allowed,step_deg=15.):
    values=list(allowed['isolated_angles_deg'])
    for lo,hi in allowed['intervals_deg']:
        values.extend(np.linspace(lo,hi,max(2,int(np.ceil((hi-lo)/step_deg))+1)))
        values.append((lo+hi)/2)
    return sorted(set(float(v)%360 for v in values if D.contains(allowed,v)))


def ground_check(plans,required,pivot,scale):
    points=np.vstack([COORD.floor(p['ground_corners_m']) for p in plans]+[COORD.floor(pivot)[None]])
    hull=ConvexHull(points)
    violation=np.max(np.asarray(required)@hull.equations[:,:2].T+hull.equations[:,2],axis=1)
    return dict(passed=bool(np.all(violation<=scale*1e-9)),
        covered_hull_vertices=int(np.sum(violation<=scale*1e-9)),hull_vertex_count=len(required),
        maximum_outside_distance_m=max(0.,float(violation.max())),
        supplied_hull_xy_m=points[hull.vertices].tolist(),tolerance_m=scale*1e-9)


def pads_separate(plans,tolerance):
    """Ground arcs may meet at endpoints/faces, but their interiors cannot overlap."""
    polygons=[unary_union(R.polygons_of(p)) for p in plans]
    return all(a.intersection(b).area<=tolerance for a,b in combinations(polygons,2))


def installation_order(count,edges):
    edges=set(map(tuple,edges));remaining=set(range(count));order=[]
    while remaining:
        free=sorted(i for i in remaining if not any(b==i and a in remaining for a,b in edges))
        if not free:return None
        chosen=free[0];order.append(chosen);remaining.remove(chosen)
    return order


def check_assembly(mesh,modules,work_volume=None):
    scale=float(mesh.extents.max());tolerance=1e-11*scale**3
    all_vertices=np.vstack([mesh.vertices]+[m['joined'].vertices for m in modules])
    sweeps=[];lengths=[];own=[]
    for module in modules:
        a=module['plan']['direction']
        # Start each moving solid outside both the object and every installed support.
        length=float((module['joined'].vertices@a).max()-(all_vertices@a).min()+.1*scale)
        replay=M.sweep_check(mesh,module['parts'],a,length)
        own.append(replay);lengths.append(replay['length_m'])
        sweeps.append(G.swept_pieces(module['parts'],a,replay['length_m']))
    pairs=[];edges=[];overlap=False
    for i,j in combinations(range(len(modules)),2):
        volume=G.intersection_volume(modules[j]['joined'],modules[i]['joined'],scale)
        overlap |= volume>tolerance
        ij=max(G.intersection_volumes(modules[j]['joined'],sweeps[i],scale),default=0.)
        ji=max(G.intersection_volumes(modules[i]['joined'],sweeps[j],scale),default=0.)
        if ij>tolerance:edges.append((i,j))
        if ji>tolerance:edges.append((j,i))
        pairs.append(dict(first=i,second=j,final_intersection_m3=volume,
            first_sweep_vs_second_m3=ij,second_sweep_vs_first_m3=ji))
    order=installation_order(len(modules),edges)
    work_checks=[work_volume.check_parts(m['parts'],m.get('labels')) for m in modules] if work_volume is not None else []
    return dict(passed=bool(not overlap and order is not None and all(r['passed'] for r in own) and
                           all(r['passed'] for r in work_checks)),
        final_work_volume_checks=work_checks,
        object_sweeps=own,lengths_m=lengths,final_supports_disjoint=not overlap,
        precedence_edges=[list(e) for e in edges],installation_order=order,
        pair_checks=pairs,volume_tolerance_m3=tolerance,
        order_rule='Edge i->j means support i must be inserted before j because its sweep meets final support j.',
        scope='Object held in its target pose during sequential installation; no transient load-bearing claim.')


def head_precheck(mesh,contacts,heads):
    scale=float(mesh.extents.max());tolerance=1e-11*scale**3
    joined=[S.union_parts(h,scale)[0] for h in heads]
    collisions=[]
    for i,j in combinations(range(len(contacts)),2):
        volume=G.intersection_volume(joined[j],joined[i],scale)
        if volume>tolerance:
            collisions.append(dict(first=contacts[i]['candidate_id'],second=contacts[j]['candidate_id'],
                                   head_intersection_m3=volume))
    return dict(passed=not collisions,collisions=collisions,volume_tolerance_m3=tolerance,
                scope='Constructed contact backings at the recorded depth; this does not test all possible backing shapes'),joined


def cut_weights(count,diameter):
    # Deterministic shifts of the cuts, still requiring every anchor in its own cell.
    yield np.zeros(count)
    for i in range(count-1):
        for fraction in (-.08,.08,-.18,.18):
            weights=np.zeros(count);weights[i]=fraction*diameter**2
            yield weights


def ground_order_possible(groups,plans,scale):
    """Cheap necessary precedence test using only the floor strips."""
    polygons=[unary_union([Polygon(p) for p in group]) for group in groups]
    length=4*max(np.ptp(np.concatenate([p for group in groups for p in group]),axis=0))
    swept=[]
    for group,plan in zip(groups,plans):
        shift=-length*COORD.floor(plan['direction'])
        swept.append(unary_union([Polygon(np.vstack([p,p+shift])).convex_hull for p in group]))
    edges=[]
    for i,j in combinations(range(len(groups)),2):
        if swept[i].intersection(polygons[j]).area>scale**2*1e-10:edges.append((i,j))
        if swept[j].intersection(polygons[i]).area>scale**2*1e-10:edges.append((j,i))
    return installation_order(len(groups),edges) is not None


def backing_heads(mesh, contacts, depth, work_volume):
    """Reduce backing depth without changing any contact point or face normal."""
    surface_checks = [dict(candidate_id=c['candidate_id'], **work_volume.check_surface(
        c['triangles_m'], c['source_faces'])) for c in contacts] if work_volume is not None else []
    surface = dict(passed=all(r['passed'] for r in surface_checks), contacts=surface_checks)
    trials = []
    factors = [1., .75, .5, .25, .125] if surface['passed'] else [1.]
    for factor in factors:
        analyzer = D.Analyzer(mesh, depth*factor)
        heads = [analyzer.heads(c) for c in contacts]
        precheck, joined = head_precheck(mesh, contacts, heads)
        checks = [dict(candidate_id=c['candidate_id'], **work_volume.check_parts(h))
                  for c,h in zip(contacts,heads)] if work_volume is not None else []
        work = dict(passed=all(r['passed'] for r in checks), contacts=checks)
        trials.append(dict(depth_factor=factor, normal_depth_m=depth*factor,
                           head_precheck=precheck, work_volume_head_precheck=work))
        if surface['passed'] and precheck['passed'] and work['passed']:
            break
    search = dict(passed=bool(surface['passed'] and precheck['passed'] and work['passed']),
        source_normal_depth_m=depth, selected_depth_factor=factor,
        selected_normal_depth_m=depth*factor, trials=trials,
        contact_triangles_changed=False, source_direction_set_retained=True,
        scope='Uniform subsets of the original joined offset skin; finite depth menu, no structural strength claim')
    return heads,joined,precheck,work,surface,search


def ranked_proposals(options, limit=512):
    """Enumerate best Cartesian combinations lazily instead of allocating n^k."""
    groups = [sorted(group,key=lambda p:p['plan']['connection_length_m']) for group in options]
    initial = tuple(0 for _ in groups)
    def cost(key):return sum(groups[i][j]['plan']['connection_length_m'] for i,j in enumerate(key))
    queue = [(cost(initial),initial)]; seen = {initial}; count = 0
    while queue and count < limit:
        _,key = heappop(queue); count += 1
        yield tuple(groups[i][j] for i,j in enumerate(key))
        for i in range(len(groups)):
            nxt = list(key); nxt[i] += 1; nxt = tuple(nxt)
            if nxt[i] < len(groups[i]) and nxt not in seen:
                seen.add(nxt); heappush(queue,(cost(nxt),nxt))


def connection_options(mesh, contact, heads, ring, candidates, work, routers, rejections):
    options = []
    for angle in candidates:
        key = (contact['candidate_id'], angle)
        if key not in routers:routers[key] = T.Router(mesh,work,angle)
        router = routers[key]
        goals = T.anchors(ring,contact,angle)
        found = []
        for anchor in goals:
            route = router.connect(heads,anchor,ring['height_m'],allow_roadmap=False)
            if route is not None:found.append((anchor,route))
            if len(found) >= 2:break
        # A roadmap is a fallback for shapes the short bent templates cannot find.
        if not found:
            for anchor in goals[:2]:
                route = router.connect(heads,anchor,ring['height_m'],allow_roadmap=True)
                if route is not None:
                    found.append((anchor,route));break
        if not found:rejections['No routed connector at this sampled angle'] += 1
        for anchor,route in found:
            plan = dict(candidate_id=contact['candidate_id'],bearing_deg=float(angle),
                direction=router.direction,anchor_xy_m=np.asarray(anchor),
                ground_height_m=ring['height_m'],expansion=ring['expansion'],
                connection_length_m=route['record']['length_m'],routing=route['record'],
                construction='Retained contact backing with finite-width routed bars to an independently chosen ring anchor')
            parts = list(heads)+route['parts']
            labels = [f'contact_head_{i:03d}' for i in range(len(heads))]
            labels += [f'routed_bar_{i:03d}' for i in range(len(route['parts']))]
            options.append(dict(plan=plan,parts=parts,labels=labels))
    return options


def search(mesh,contacts,directions,required,pivot,depth,center=None,expansions=None,assembly_budget=128,work_volume=None):
    scale=float(mesh.extents.max())
    heads,joined_heads,precheck,work_precheck,surface_precheck,backing=backing_heads(mesh,contacts,depth,work_volume)
    angles=[candidate_angles(r['certified_directions']) for r in directions]
    settings=dict(angular_step_deg=15.,ground_width_fraction=.025,ground_height_fraction=.018,
                  connector_width_fraction=.022,assembly_budget_per_expansion=assembly_budget,
                  proposal_budget_per_expansion=512,route_edge_budget=1600,route_lattice_spacing_fraction=.075,
                  anchor_count_per_angle=6,roadmap_anchor_count_per_angle=2,
                  anchor_restricted_to_withdrawal_ray=False,
                  partition='Power cells of independently chosen ring anchors, with finite cut offsets')
    base=R.make(required,1.,scale,center)
    diagnostic=dict(head_precheck=precheck,work_volume_head_precheck=work_precheck,
                    work_volume_surface_precheck=surface_precheck,backing_search=backing)
    if not backing['passed']:
        # Keep an honest head-only diagnostic, never label these as ground supports.
        modules=[]
        for contact,parts,joined,candidates in zip(contacts,heads,joined_heads,angles):
            angle=candidates[len(candidates)//2];basis=G.frame(angle)
            _,solid=S.union_parts(parts,scale)
            plan=dict(candidate_id=contact['candidate_id'],direction=basis[0],bearing_deg=angle,
                      ground_polygons_xy_m=[],ground_corners_m=np.empty((0,3)),ground_area_m2=0.,
                      construction='head_only',anchor_xy_m=COORD.floor(contact['center_m']),backing_depth_factor=backing['selected_depth_factor'])
            modules.append(dict(plan=plan,parts=parts,labels=[f'contact_head_{i:03d}' for i in range(len(parts))],joined=joined,solid=solid))
        status='contact_surfaces_block_work_volume' if not surface_precheck['passed'] else 'no_feasible_contact_backing'
        error='Actual contact interfaces meet the reserved volume' if not surface_precheck['passed'] else 'No clear backing in the tested thickness menu'
        if not surface_precheck['passed'] and not any(r['classification']=='circular_cone_intersection' for r in surface_precheck['contacts'] if not r['passed']):
            status='work_volume_visibility_unresolved';error='Only unresolved visibility-bound intersections at the contact interfaces'
        return dict(status=status,passed=False,installation=None,ground=None,
                    geometry_error=error,
                    ring=R.record(base),ring_coverage=None,**diagnostic,attempts=[],expansion_trials=[],
                    search_settings=settings,failure_is_global_impossibility_proof=False),modules
    expansions=list(expansions) if expansions is not None else list(np.round(np.arange(1.,2.501,.05),8))+[2.6,2.7,2.8,2.9,3.]
    attempts=[];trials=[];best=None;routers={}
    for expansion in expansions:
        ring=R.make(required,expansion,scale,center);options=[];rejections=Counter()
        for contact,head,candidates in zip(contacts,heads,angles):
            group=connection_options(mesh,contact,head,ring,candidates,work_volume,routers,rejections)
            for option in group:option['plan']['backing_depth_factor']=backing['selected_depth_factor']
            options.append(group)
            print('   ',contact['candidate_id'],'routed options',len(group),flush=True)
        trial=dict(expansion=float(expansion),option_counts=list(map(len,options)),rejections=dict(rejections),
                   full_assembly_attempts=0,partition_rejections=0,ground_order_rejections=0)
        trials.append(trial)
        print('  ring scale',expansion,'connection options',trial['option_counts'],flush=True)
        if any(not group for group in options):continue
        proposals=ranked_proposals(options)
        diameter=float(np.ptp(ring['outer_xy_m'],axis=0).max())
        for proposal in proposals:
            if trial['full_assembly_attempts']>=assembly_budget:break
            plans=[p['plan'] for p in proposal]
            for weights in cut_weights(len(contacts),diameter):
                if trial['full_assembly_attempts']>=assembly_budget:break
                groups=R.partition(ring,[p['anchor_xy_m'] for p in plans],weights)
                if groups is None:trial['partition_rejections']+=1;continue
                if not ground_order_possible(groups,plans,scale):trial['ground_order_rejections']+=1;continue
                modules=[];failure=None;installation=None
                trial['full_assembly_attempts']+=1
                for prepared,polygons in zip(proposal,groups):
                    try:
                        parts,labels,plan=S.with_ground(prepared['parts'],prepared['labels'],prepared['plan'],polygons)
                        if work_volume is not None and not work_volume.check_parts(parts,labels)['passed']:
                            raise ValueError('Full support enters reserved work volume')
                        sweep=M.sweep_check(mesh,parts,plan['direction'])
                        if not sweep['passed']:raise ValueError('Full support sweep meets object or floor')
                        joined,solid=S.union_parts(parts,scale)
                        if not solid['one_solid']:raise ValueError('Ground arc and contact are not one closed solid')
                        modules.append(dict(plan=plan,parts=parts,labels=labels,joined=joined,solid=solid))
                    except (ValueError,RuntimeError) as error:failure=str(error);break
                floor=coverage=None
                if failure is None:
                    actual=[m['plan'] for m in modules]
                    coverage=R.check(ring,actual,scale);floor=ground_check(actual,required,pivot,scale)
                    installation=check_assembly(mesh,modules,work_volume)
                attempt=dict(expansion=float(expansion),bearings_deg=[p['bearing_deg'] for p in plans],
                    cut_weights_m2=weights.tolist(),ground=floor,ring_coverage=coverage,
                    geometry_error=failure,installation=installation)
                attempts.append(attempt)
                rank=(len(modules),int(coverage is not None and coverage['passed']))
                if best is None or rank>best[0]:best=(rank,attempt,modules,R.record(ring))
                if failure is None and floor['passed'] and coverage['passed'] and installation['passed']:
                    return dict(status='separate_supports_and_insertions_verified',passed=True,**attempt,
                        ring=R.record(ring),**diagnostic,attempts=attempts,expansion_trials=trials,
                        search_settings=settings,smallest_feasible_tested_expansion=float(expansion),
                        global_minimum_proved=False),modules
    if best is None:best=((0,0),dict(ground=None,ring_coverage=None,installation=None,
        geometry_error='No complete ring partition with viable connections found',bearings_deg=[]),[],R.record(base))
    return dict(status='no_verified_separate_support_layout',passed=False,**best[1],ring=best[3],
        **diagnostic,attempts=attempts,expansion_trials=trials,search_settings=settings,
        failure_is_global_impossibility_proof=False),best[2]
