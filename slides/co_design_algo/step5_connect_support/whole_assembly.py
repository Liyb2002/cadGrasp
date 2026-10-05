"""Read fixed Step3 heads and extract the actual floor contact of one rigid support.

Direction/shape search, bearing and auditing live in belt_assembly.py and
direction_first.py. The retired independent-foot and early U-base searches
are no longer alternate entry points.
"""
import json
import numpy as np
from step1.needs import COORD, ContinuousNeeds, OUTPUTS, sha256
from step1.cases import pose_name
from step3_scheculer import contacts as I
from step2_local_support import insertion as D, work_volume as W, support_policy as POLICY
from step4_floor_contact import whole_assembly as F
from step5_connect_support import ground as G


def footprint(parts, pivot, required, scale):
    """Extract actual bottom triangles, including unexpected added ground material."""
    triangles = []
    for part in parts:
        on_floor = np.all(np.abs(part.triangles[:, :, 2]) <= scale*1e-10, axis=1)
        triangles.extend(part.triangles[on_floor])
    tri = np.asarray(triangles).reshape(-1, 3, 3)
    if not len(tri):
        return dict(passed=False, reason='no_actual_support_floor_faces'), tri
    points = np.vstack([COORD.floor(tri.reshape(-1, 3)), COORD.floor(pivot)])
    covered, hull = G.hull_coverage(required, points, scale*1e-9)
    maximum = float(np.max(required@hull.equations[:, :2].T+hull.equations[:, 2]))
    return dict(passed=bool(covered.all()), supplied_hull_xy_m=points[hull.vertices].tolist(),
        required_vertex_count=len(required), maximum_outside_distance_m=max(0., maximum),
        original_object_floor_point_included=True, floor_triangle_count=len(tri)), tri

def read_inputs(name):
    root = OUTPUTS/name/pose_name()
    schedule_path = root/'step3_scheculer/schedule.json'
    schedule = I.check_report(schedule_path)
    state = json.loads((schedule_path.parent/'status.json').read_text())
    assert state['complete'] and state['schedule_sha256'] == sha256(schedule_path)
    domain = ContinuousNeeds.read(root/'step_1_needs/needs.json')
    if not domain.mesh.is_watertight or not domain.mesh.is_winding_consistent:
        raise ValueError('One-body construction requires a closed outward-oriented object')
    path = schedule_path.parent/'final_contacts.npz'
    contacts = I.read_contacts(path)
    directions = I.check_report(schedule_path.parent/'insertion_directions.json')
    assert directions['selected_ids'] == schedule['selected_ids'] == [c['candidate_id'] for c in contacts]
    for contact, row in zip(contacts, directions['contacts']):
        assert row['geometry_signature'] == D.signature(contact, directions['normal_depth_m'])
    assert len(contacts) == len(directions['contacts'])
    floor = F.read(name); arrays = I.load_npz(F.output_folder(name)/floor['arrays_file'])
    audit_path = F.output_folder(name)/'audit.json'
    assert I.check_report(audit_path)['passed']
    work_path = root/W.STAGE/'work_volume.json'
    work = None
    if POLICY.ENFORCE_PROCESS_ACCESS:
        assert I.check_report(work_path.parent/'work_volume_audit.json')['passed']
        work = W.WorkVolume.read(work_path)
    paths = [schedule_path, schedule_path.parent/'status.json', path,
        schedule_path.parent/'insertion_directions.json', root/'step_1_needs/needs.json',
        F.output_folder(name)/'floor_contact.json', F.output_folder(name)/'floor_contact.npz', audit_path]
    if work is not None:
        paths += [work_path, work_path.parent/'work_volume.npz', work_path.parent/'work_volume_audit.json']
    return domain, contacts, schedule, arrays, directions, work, paths
