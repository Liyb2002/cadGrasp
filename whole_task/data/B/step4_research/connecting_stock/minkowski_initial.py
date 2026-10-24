"""Choose connecting material without changing contact skins or floor layers.

The grid is a source of connectors, rather than a required final topology.
All object bounding neighbourhoods contain the complete original 5 mm skin;
all installation floor layers keep their original thickness. A spanning set
of the remaining material joins these protected components. This is only
candidate construction: the normal complete acceptance is still mandatory.
"""
import numpy as np
from whole_task.step3 import independent_geometry as G
from .geometry import moved_mesh
from .floors import planes


def positive_parts(value):
    return [p for p in value.decompose() if p.volume()*G.SCALE**3>1e-15]


def connecting_stock(model,candidate):
    before=candidate['support'];q=candidate['layout'];solid=G.solid(before)
    report=dict(method='spanning_connectors_with_complete_contact_and_floor_layers',
        applied=False,volume_before_cm3=float(before.volume*1e6),
        original_contact_skin_halfwidth_m=.005,
        original_floor_layer_thickness_m=model.policy['plate_thickness_m'],
        original_connector_widths_preserved=True,full_acceptance_required=True)
    def check():
        if hasattr(model,'check_deadline'):model.check_deadline()
    check()
    # Oriented object boxes contain the ENTIRE nonconvex Minkowski skin,
    # including its 5 mm outer thickness. They also conservatively preserve
    # any extra stock inside those boxes; no contact-layer thinning occurs.
    box=G.mesh(G.box(model.source.bounds[0]-.005,model.source.bounds[1]+.005))
    regions=[G.solid(moved_mesh(box,M@P)) for M,P in zip(q.material,q.poses)]
    envelope=G.box(before.bounds[0]-.001,before.bounds[1]+.001)
    depth=model.policy['plate_thickness_m']
    for n,offset in planes(q.material):
        regions.append(envelope.trim_by_plane(n,offset/G.SCALE)
            .trim_by_plane(-n,(-offset-depth)/G.SCALE))
    protection=G.union(regions)
    protected=solid^protection
    # Subtract simple protective regions directly, rather than subtracting
    # two almost coincident complete fixture meshes.
    anchors=positive_parts(protected);posts=positive_parts(solid-protection)
    report.update(protected_component_count=len(anchors),available_connector_count=len(posts),
        protected_volume_cm3=G.volume(protected)*1e6)
    if not anchors or len(anchors)>512 or len(posts)>2048:
        return before,dict(report,reason='component_count_outside_construction_budget')
    parent=list(range(len(anchors)))
    def root(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    bounds=[G.mesh(a).bounds for a in anchors]
    # A 1 um overlap into the protected layer makes caps robust to Boolean
    # serialization. The final intersection with the ORIGINAL solid prevents
    # adding any connector material outside the proposed stock.
    overlap=1e-6;cube=G.box([-overlap]*3,[overlap]*3);edges=[]
    for index,post in enumerate(posts):
        check();b=G.mesh(post).bounds
        nearby=[k for k,a in enumerate(bounds)
            if np.all(np.maximum(a[0],b[0]-overlap)<np.minimum(a[1],b[1]+overlap))]
        if len(nearby)<2:continue
        expanded=post.minkowski_sum(cube)
        touching=[k for k in nearby if G.volume(anchors[k]^expanded)>1e-18]
        if len(touching)>1:
            edges.append((G.volume(post),index,touching,expanded))
    selected=[];chosen=[]
    while len({root(k) for k in parent})>1:
        check();eligible=[]
        for edge in edges:
            groups={root(k) for k in edge[2]}
            if len(groups)>1:eligible.append((edge[0]/(len(groups)-1),edge[1],edge,groups))
        if not eligible:return before,dict(report,reason='no_spanning_connector_set')
        _,index,edge,groups=min(eligible,key=lambda row:(row[0],row[1]))
        first=min(groups)
        for k in groups:parent[k]=first
        selected.append(index);chosen.append(edge[3])
    proposed=solid^(protection+G.union(chosen)) if chosen else protected
    if len(positive_parts(proposed))!=1:
        return before,dict(report,reason='spanning_graph_did_not_produce_one_actual_component')
    after=G.mesh(proposed);volume=float(after.volume*1e6)
    if not after.is_watertight or not after.is_winding_consistent or volume>=report['volume_before_cm3']-1e-5:
        return before,dict(report,reason='no_closed_smaller_connected_shape')
    report.update(applied=True,selected_connector_count=len(selected),selected_connector_indices=selected,
        volume_after_cm3=volume,removed_cm3=report['volume_before_cm3']-volume)
    return after,report
