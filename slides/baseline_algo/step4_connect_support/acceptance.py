"""Stage4 acceptance: geometry only; forces and torques belong to Stage3."""
import numpy as np
from shapely.geometry import MultiPoint

VOLUME_TOLERANCE_M3 = 8e-14


def pose_checks(check):
    return dict(
        floor=check['min_fixture_z_m'] >= -1e-9,
        contact_preservation=check['max_active_contact_gap_m'] < 1e-8,
        root_preservation=check['maximum_missing_head_cell_volume_m3'] <= VOLUME_TOLERANCE_M3,
        full_withdrawal=bool(check['withdrawal']['clear']),
        independent_head_withdrawal=bool(check['independent_head_withdrawal']['clear']))


def evaluate(case,report,mesh,working_surface):
    """Replay geometry aggregation against unchanged, hash-validated artifacts.

    Neither an old Step4 equilibrium boolean nor a Step3 boolean is promoted to
    a new force proof. Step3's original verdict stays separately visible.
    """
    from step4_connect_support.reseating import span_metrics
    body=report.get('construction',report)
    rows=[]
    for check,demands in zip(body['checks'],case.demands):
        actual=MultiPoint(np.asarray(check['actual_ground_hull_xy_m'])).convex_hull
        required=MultiPoint(np.asarray(demands)).convex_hull
        loss=float(required.difference(actual.buffer(1e-10)).area)
        conditions=pose_checks(check)
        conditions['ground_hull_coverage']=actual.area > 0 and loss <= 1e-12
        rows.append(dict(pose=check['pose'],passed=all(conditions.values()),
            checks=conditions,uncovered_demand_hull_area_m2=loss))
    if len(rows)!=len(case.poses):
        raise ValueError('Every pose needs its geometry evidence')
    solid=body['solid']
    topology=bool(solid['one_solid'] and solid['component_count']==1
        and mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0)
    b=np.asarray(report['placement']['bases']);o=np.asarray(report['placement']['offsets'])
    metrics=span_metrics(mesh.vertices,b,o)
    geometry_passed=all(r['passed'] for r in rows) and topology and working_surface['passed']
    failures=[dict(pose=r['pose'],check=k) for r in rows for k,v in r['checks'].items() if not v]
    if not topology:failures.append(dict(check='closed_connected_solid'))
    if not working_surface['passed']:failures.append(dict(check='working_surface'))
    return dict(complete=True,passed=bool(geometry_passed),
        geometry_excluding_size_passed=bool(geometry_passed),checks=rows,
        topology_passed=topology,working_surface_passed=working_surface['passed'],
        compactness_passed=None,size_limit_enforced=False,metrics=metrics,
        failures=failures,scope='Step4 geometry; force and torque acceptance belongs to Step3',
        force_torque_authority='step3',step4_force_torque_enforced=False,
        step4_force_torque_recomputed=False,step3_passed=bool(case.schedule['passed']),
        step3_covered_counts=case.schedule['covered_counts'],
        step3_verdict_modified=False,
        original_force_diagnostics_do_not_veto_step4=True)
