"""Prepare actual Pose1 construction snapshots and floor data for a storyboard."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import mapping, Point
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step5_connect_support import build_coupled_saddle as S
from step5_connect_support.build_shared_geometry import pack
from step5_connect_support.audit_local_roles import floor_contact, contact_vertices


def prepare(out):
    out = Path(out)
    report = json.loads((out/'report.json').read_text())
    replay = json.loads((out/'construction_sequence.json').read_text())
    assert report['complete'] and replay['complete']
    assert S.I.sha256(out/'report.json') == replay['original_report_sha256']
    for name, digest in report['artifacts'].items():
        assert S.I.sha256(out/name) == digest, name
    data, _ = json.JSONDecoder().raw_decode((out/'construction.html').read_text().split('const DATA=', 1)[1])
    assert [p['name'] for p in data['poses']] == ['pose_1','pose_3']
    task = S.read_task('B', 'pose_1', S.POSES)
    with np.load(out/'geometry.npz') as stored:
        basis, offset = stored['rotations'][0], stored['local_offsets_m'][0]

    def mesh(record):
        return trimesh.Trimesh(np.asarray(record['vertices']).reshape(-1,3),
                               np.asarray(record['faces']).reshape(-1,3), process=False)

    def floor_info(parts):
        # Union coplanar contact patches directly. Re-importing already split
        # Boolean increments into a new 3D CSG tree can perturb thin seams;
        # the figure displays the original triangles without such rebuilding.
        contact, stats = floor_contact(trimesh.util.concatenate(parts), basis, offset, 1e-9)
        vertices = contact_vertices(contact)
        points = np.vstack([vertices, np.asarray(task.floor).reshape(1,3)[:, :2]])
        hull = ConvexHull(points)
        outside = np.max(cloud@hull.equations[:, :2].T+hull.equations[:, 2],axis=1)>1e-9
        return dict(geometry=mapping(contact), statistics=stats,
                    support_hull_xy_m=points[hull.vertices].tolist(),
                    outside_original_sample_count=int(outside.sum()))

    cloud, _ = S.pressure_centers(task.targets/task.scale, task.domain.com)
    current = [mesh(h) for h in data['heads']]
    floor_stages = {}
    teal_floor_point = None
    for i, stage in enumerate(data['stages']):
        if stage.get('mesh'):
            current.append(mesh(stage['mesh']))
        if i in (6,10,14,18):
            floor_stages[str(i)] = floor_info(current)
        if i == 5:
            contact, _ = floor_contact(mesh(stage['mesh']),basis,offset,1e-9)
            polys = [contact] if contact.geom_type == 'Polygon' else [p for p in contact.geoms if p.geom_type=='Polygon']
            point = max(polys,key=lambda p:p.area).representative_point()
            teal_floor_point = [point.x,point.y,0.]
    final = mesh(data['fixture'])
    with np.load(out/'geometry.npz') as stored:
        np.testing.assert_array_equal(final.vertices,stored['vertices_m'])
        np.testing.assert_array_equal(final.faces,stored['faces'])
    actual_floor, _ = floor_contact(final,basis,offset,1e-9)
    displayed_floor, _ = floor_contact(trimesh.util.concatenate(current),basis,offset,1e-9)
    floor_error = float(actual_floor.symmetric_difference(displayed_floor).area)
    floor_distance = max(max(Point(p).distance(actual_floor) for p in contact_vertices(displayed_floor)),
                         max(Point(p).distance(displayed_floor) for p in contact_vertices(actual_floor)))
    assert floor_error<=1e-12 and floor_distance<=1e-9,(floor_error,floor_distance)
    assert floor_stages['6']['outside_original_sample_count']>0
    assert floor_stages['18']['outside_original_sample_count']==0
    design = json.loads((out/'design.json').read_text())
    feet = [dict(number=a['terminal']+1,body=a['head_body'],xy=a['polygon_xy_m'])
            for a in design['attachments'] if a['floor_pose']=='pose_1']
    source = (S.ROOT/report['source_schedule']).parent
    centers = []
    for pose in S.POSES:
        transform = next(t for t in report['task_fixture_transforms'] if t['pose']==pose)
        rotation,translation=np.asarray(transform['rotation']),np.asarray(transform['translation_m'])
        for contact in S.I.read_contacts(source/f'contacts_{pose}.npz'):
            center=np.asarray(contact['triangles_m']).reshape(-1,3).mean(axis=0)
            centers.append(((center-translation)@rotation).tolist())
    result = dict(pose='pose_1',original_sample_count=len(cloud),floor_demands_xy_m=cloud.tolist(),
        original_object_floor_point_m=np.asarray(task.floor).tolist(),floors=floor_stages,
        pose1_feet=feet,head_centers_m=centers,teal_initial_floor_point_m=teal_floor_point,
        teal_body=pack(trimesh.load(out/'body4.obj',process=False)),
        final_geometry_vertices_and_faces_exactly_match=True,
        source_replay_symmetric_difference_m3=replay['displayed_union_symmetric_difference_m3'],
        final_floor_union_symmetric_difference_m2=floor_error,final_floor_max_vertex_to_contact_distance_m=floor_distance,
        physical_artifacts_unchanged=True,
        scope='Fixed Pose1 display; the top-down panel shows the same floor. Support hulls are virtual, not material. No new loads or mechanics solve.',
        provenance=dict(inputs={str(out/'report.json'):S.I.sha256(out/'report.json'),
                                str(out/'construction.html'):S.I.sha256(out/'construction.html'),
                                **S.I.hashes(task.inputs)},code=S.I.hashes([Path(__file__)])))
    figures = out/'pose1_steps'
    figures.mkdir(parents=True, exist_ok=True)
    S.I.save(figures/'pose1_steps_data.json',result)
    print('Initial / after Pose1 feet / final CoP outside counts:',
          floor_stages['6']['outside_original_sample_count'],floor_stages['10']['outside_original_sample_count'],
          floor_stages['18']['outside_original_sample_count'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',nargs='?',type=Path,default=S.ROOT/'slides/baseline_algo/output/B/pose1+3/step5')
    prepare(parser.parse_args().output)
