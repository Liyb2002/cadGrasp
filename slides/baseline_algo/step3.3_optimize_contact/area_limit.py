"""Every selected contact must exceed 0.5% of the object's full surface area."""
import math

MIN_AREA_FRACTION = 0.005


def radius_floor(lower, initial, tolerance, area_at_radius, minimum_area):
    """Return the feasible side of a bracket using actual, monotone patch area.

    Surface clipping and connected-component joins make pi*r*r unsuitable.
    The strict area inequality is preserved even when the area jumps.
    """
    if not (0 < lower <= initial and tolerance > 0
            and math.isfinite(minimum_area) and minimum_area > 0):
        raise ValueError('Expected positive radius bounds, tolerance and minimum area')
    if not area_at_radius(initial) > minimum_area:
        raise ValueError('Initial contact must exceed 0.5% of total object surface area')
    low, high = lower, initial
    iterations = 0
    if area_at_radius(low) > minimum_area:
        high = low
        low = None
    else:
        while high-low > tolerance:
            middle = (low+high)/2
            if area_at_radius(middle) > minimum_area:
                high = middle
            else:
                low = middle
            iterations += 1
    assert area_at_radius(high) > minimum_area
    return high, dict(minimum_area_fraction=MIN_AREA_FRACTION,
        minimum_area_m2=minimum_area, comparison='strictly_greater_than',
        reference='total_object_surface_area', minimum_radius_m=high,
        lower_infeasible_radius_m=low, boundary_area_m2=area_at_radius(high),
        radius_tolerance_m=tolerance, iterations=iterations)
