"""Delete local Boolean remnants; certify the unchanged original exit paths."""
import numpy as np
import manifold3d as md
from whole_task.step3 import independent_geometry as G
from whole_task.step3.exit_geometry import continuous_checks
from .floors import connected_material, condition


def expand_prism(prism, clearance):
    """Exact convex Minkowski sum with a small axis-aligned clearance box."""
    points=G.mesh(prism).vertices
    offsets=np.array([[x,y,z] for x in (-clearance,clearance)
                     for y in (-clearance,clearance) for z in (-clearance,clearance)])
    return md.Manifold.hull_points((points[:,None]+offsets).reshape(-1,3)/G.SCALE)


def local_clearance(current,forbidden,clearance):
    """Expand the actual intrusion, retaining contacts elsewhere on the body.

    Expanding an entire original body to fix a tiny Boolean remnant would
    clear every valid contact face. This nonconvex Minkowski sum only changes
    the local material that actually intrudes into the forbidden geometry.
    """
    intrusion=current^forbidden
    if G.volume(intrusion)<=1e-20:return intrusion
    # Tangent contact sheets can survive in a native intersection with zero
    # thickness. Dilating them would erase valid contacts across the body.
    # A 10 nm opening removes those sheets before the local expansion. Every
    # unchanged original prism is still checked afterward at the same limit.
    opening=1e-8
    interior=intrusion.minkowski_difference(G.box([-opening]*3,[opening]*3))
    if G.volume(interior)<=1e-20:return interior
    radius=clearance+opening
    return interior.minkowski_sum(G.box([-radius]*3,[radius]*3))


def refine_candidate(model,candidate):
    tasks=candidate['tasks']
    paths=[(t['id'],t['body'],t['direction'],t['length']) for t in tasks]
    forbidden=[p for t in tasks for p in (G.solid(t['body']),t['work'])]
    original=G.solid(candidate['support'])
    # Operations may regenerate material; a subset of the temporary raw
    # candidate is not a physical design constraint. Keep the finite current
    # stock envelope and every actual exclusion/load guard instead. Near-
    # coincident serialized boundaries make raw-candidate subset Booleans an
    # unreliable extra restriction even after subtraction-only construction.
    envelope=candidate.get('parent',original)
    value=original
    history=[]
    # This changes the DESIGN by removing extra material, not the volume
    # tolerance or the original geometric definition of the extracted object.
    for round_index,clearance in enumerate((0.,1e-6,3e-6,1e-5,None)):
        actual=G.mesh(value)
        current=G.solid(actual)
        checks,bad=model.path_checks(current,paths)
        if all(row['passed'] for row in checks):
            actual,conditioning=condition(current,forbidden,envelope,model.tolerance)
            # Conditioning can change the saved boundary; certify exactly
            # that resulting boundary before any force acceptance.
            if not np.array_equal(actual.vertices,G.mesh(current).vertices) or not np.array_equal(actual.faces,G.mesh(current).faces):
                checks,bad=model.path_checks(G.solid(actual),paths)
                if not all(row['passed'] for row in checks):
                    value=G.solid(actual)
                    continue
            return actual,dict(continuous_checks=checks,local_clearance_history=history,
                conditioning=conditioning,removed_cm3=float((candidate['support'].volume-actual.volume)*1e6),
                original_paths_and_tolerance_unchanged=True,stock_bound='current finite regenerated stock envelope')
        if clearance is None or not bad:
            break
        cuts=[part if clearance==0 else local_clearance(current,part,clearance)
              for _,face,part in bad]
        value=connected_material(current-G.union(cuts))
        history.append(dict(round=round_index,extra_local_clearance_m=clearance,
                            cut_prism_count=len(cuts),failed_tasks=[r['id']for r in checks if not r['passed']]))
    raise RuntimeError('Original continuous exits remain unresolved after local subtraction: '+str(checks))
