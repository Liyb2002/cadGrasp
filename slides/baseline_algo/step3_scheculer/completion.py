"""Require Step 3 coverage for certification; partial geometry is handled separately."""
import json

from step3_scheculer import contacts as I

from step1.cases import pose_name

MAX_CONTACTS = 3


class IncompleteSchedule(RuntimeError):
    pass


def require_passed(schedule):
    count = schedule.get('sample_count', 0)
    if not (schedule.get('complete') is True
            and schedule.get('status') == 'continuous_contact_model_verified'
            and count > 0 and schedule.get('covered_count') == count
            and schedule.get('continuous_domain_status') == 'verified'
            and schedule.get('continuous_coverage_proved') is True
            and schedule.get('round_limit') == MAX_CONTACTS
            and 0 <= schedule.get('contact_count', MAX_CONTACTS+1) <= MAX_CONTACTS
            and schedule.get('rest_equilibrium_verified') is True
            and schedule.get('passive_support_no_uplift_verified') is True
            and schedule.get('passive_support_constraint', {}).get('mode') == 'connected_massless_support_no_uplift'
            and schedule.get('passive_support_constraint', {}).get('enforced') is True
            and schedule.get('insertion_mode') == 'common_rigid_withdrawal_3d'
            and schedule.get('common_head_withdrawal_verified') is True
            and bool(schedule.get('common_withdrawal_directions', {}).get('ids'))
            and schedule.get('all_contact_areas_above_minimum') is True
            and (schedule.get('process_access_enforced') is False
                 or schedule.get('all_contact_heads_clear_of_work_volume') is True)
            and schedule.get('contact_heads_individually_insertable') is True):
        raise IncompleteSchedule(
            f"{schedule.get('object', 'object')}: Step 3 has not passed "
            f"({schedule.get('status', 'missing status')}, "
            f"{schedule.get('covered_count', 0)}/{count} loads); full load certification unavailable")
    return schedule


def read_passed(name):
    folder = I.OUTPUTS/name/pose_name()/'step3_scheculer'
    status = json.loads((folder/'status.json').read_text())
    path = folder/'schedule.json'
    if (status.get('complete') is not True
            or status.get('schedule_sha256') != I.sha256(path)):
        raise IncompleteSchedule(f'{name}: Step 3 result is unfinished or superseded; Step 4/5 cannot proceed')
    # Give an incomplete-result error before inspecting its older code hashes.
    require_passed(json.loads(path.read_text()))
    return require_passed(I.check_report(path))


def archive_downstream(name):
    """Invalidate downstream status in place; never create history directories."""
    root = I.OUTPUTS/name/pose_name()
    for stage in ('step4_floor_contact', 'step5_connect_support'):
        folder = root/stage
        if folder.exists():
            I.save(folder/'status.json', dict(
                object=name, complete=False, status='superseded_by_new_step3_search',
                reason='Previous geometry is retained but is not a current accepted solution.'))
    return None
