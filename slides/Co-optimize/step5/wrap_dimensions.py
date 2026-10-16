"""User dimensions in metres, anchored to the pre-change 4 mm foot band."""

PREVIOUS_FOOT_WIDTH_M = .004
INITIAL_FOOT_WIDTH_M = 2 * PREVIOUS_FOOT_WIDTH_M
MINIMUM_WIDTH_M = 2 * PREVIOUS_FOOT_WIDTH_M
INITIAL_FOOT_HEIGHT_M = .005
CONNECTOR_RADIUS_M = MINIMUM_WIDTH_M
# A pruning heuristic, not a minimum-width or strength certificate.
RETAINED_VOLUME_FRACTION = .55


def physical_component_count(value):
    """Count every positive material component, including tiny loose chips.

    Negative shells are internal cavities, not disconnected material. There
    is deliberately no positive-volume cutoff that could hide a small part.
    """
    return sum(float(part.volume()) > 0. for part in value.decompose())


def dimensions():
    return dict(width_rule_version=9,wall_thickness_constrained=False,
                previous_initial_foot_width_m=PREVIOUS_FOOT_WIDTH_M,
                initial_foot_width_m=INITIAL_FOOT_WIDTH_M,
                initial_foot_outward_offset_m=PREVIOUS_FOOT_WIDTH_M,
                initial_foot_inward_offset_m=INITIAL_FOOT_WIDTH_M-PREVIOUS_FOOT_WIDTH_M,
                minimum_width_m=None,minimum_width_enforced=False,
                width_goal_m=MINIMUM_WIDTH_M,
                retained_volume_fraction=RETAINED_VOLUME_FRACTION,
                initial_foot_height_m=INITIAL_FOOT_HEIGHT_M,
                nominal_connector_radius_m=CONNECTOR_RADIUS_M)
