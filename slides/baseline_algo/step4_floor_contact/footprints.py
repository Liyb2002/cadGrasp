"""Separate ground-bearing pads per support; no connector is constructed here.

Each support owns four small square bearing patches. Their convex hull carries
its pressure centre; the hull interior is NOT a plate or shared ground ring.
"""
from step1.needs import COORD
import numpy as np
from shapely.geometry import Polygon, box

FIXED_TEMPLATE = dict(pad_side_fraction=.025, height_fraction=.018,
                      initial_radius_fraction=.08, clearance_fraction=.005,
                      layout_increment_fraction=.035, maximum_layout_attempts=1000)


def fixed_layout(contacts, mesh):
    """Choose the deterministic geometry once, before looking at any reactions.

    The existing four-pad placement only grows enough to clear the object and
    earlier pads. Loads and Step 5 routing never resize or relocate these pads.
    """
    feet = serializable(design(contacts, np.full((0, len(contacts), 2), np.nan), mesh))
    for foot in feet:
        foot.update(height_m=FIXED_TEMPLATE['height_fraction']*float(mesh.extents.max()),
                    fixed_for_step5=True, layout_rule='fixed_contact_center_template')
    return feet


def rectangle(lo, hi):
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def design(contacts, pressure_centers, mesh, expansion=1.):
    length = float(mesh.extents.max())
    half_pad = .5*FIXED_TEMPLATE['pad_side_fraction']*length
    gap = FIXED_TEMPLATE['clearance_fraction']*length
    obstacle = box(*(COORD.floor(mesh.bounds[0])-gap), *(COORD.floor(mesh.bounds[1])+gap))
    occupied = []
    feet = []
    for j, contact in enumerate(contacts):
        data = pressure_centers[:, j]
        finite = data[np.isfinite(data).all(axis=1)]
        if not len(finite): finite = COORD.floor(contact['center_m'])[None]
        center = (finite.min(axis=0)+finite.max(axis=0))/2
        radii = np.maximum((finite.max(axis=0)-finite.min(axis=0))/2,
                           (FIXED_TEMPLATE['initial_radius_fraction']-.03)*length)
        radii = (radii+.03*length)*expansion
        for attempt in range(FIXED_TEMPLATE['maximum_layout_attempts']):
            corners = rectangle(center-radii, center+radii)
            pads = [rectangle(p-half_pad, p+half_pad) for p in corners]
            polygons = [Polygon(p) for p in pads]
            if all(not p.intersects(obstacle) and
                   all(p.distance(other) > gap for other in occupied) for p in polygons):
                break
            radii += FIXED_TEMPLATE['layout_increment_fraction']*length
        else:
            raise RuntimeError('No separated four-pad footprint in the finite layout search')
        occupied.extend(polygons)
        feet.append(dict(candidate_id=contact['candidate_id'], pads_xz_m=pads,
                         hull_xz_m=rectangle(center-radii-half_pad, center+radii+half_pad),
                         pad_side_m=2*half_pad, bearing_area_m2=16*half_pad**2,
                         center_xz_m=center, extent_xz_m=2*(radii+half_pad),
                         construction='four separately located bearing squares owned by this support'))
    return feet


def check(feet, mesh):
    groups = [(j, Polygon(p)) for j, foot in enumerate(feet) for p in foot['pads_xz_m']]
    overlap = max([a.intersection(b).area for k, (_, a) in enumerate(groups)
                   for _, b in groups[k+1:]], default=0.)
    envelope = box(*COORD.floor(mesh.bounds[0]), *COORD.floor(mesh.bounds[1]))
    clearance = min([p.distance(envelope) for _, p in groups], default=0.)
    return dict(passed=overlap == 0 and clearance > 0, maximum_pad_overlap_m2=overlap,
                object_projection_clearance_m=clearance,
                only_actual_pads_are_material=True, hull_interiors_are_not_material=True,
                connections_and_work_volume_clearance_deferred_to_step5=True)


def serializable(feet):
    return [{k: ([p.tolist() for p in v] if k == 'pads_xz_m' else
                 v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in foot.items()} for foot in feet]
