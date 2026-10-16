"""Recompute BBX metrics and verify the single fixed image for every result."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import trimesh
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as E
from metrics import static_bounds,BOX_KEYS


def verify(name='B'):
    folder=E.directory(name);summary=E.read(folder/'summary.json');assert summary['complete']
    records=E.read(folder/'verification.json')
    for path,expected in records['protected_inputs'].items():assert E.sha(E.ROOT/path)==expected,path
    for path,expected in records['executed_code'].items():assert E.sha(folder/'executed_sources'/path)==expected,path
    body=trimesh.load(E.ROOT/'objects'/name/'mesh.stl',force='mesh',process=False);checked=0
    for row in summary['results']:
        path=E.report_path(name,row['id']);assert E.read(path)==row
        image=E.image_path(name,row['id']);assert E.sha(image)==row['image_sha256']
        assert row['relative_bbx_volume']['object_static']==1.
        with Image.open(image)as picture:
            assert picture.size==E.FIGURE_SIZE
            assert picture.info['evaluation_output_version']==E.VERSION
            assert json.loads(picture.info['relative_bbx_volume'])==row['relative_bbx_volume']
            assert json.loads(picture.info['bbx_increase_percent'])==row['bbx_increase_percent']
            assert json.loads(picture.info['presentation'])==row['presentation']
        display=row['presentation'];poses=[p['pose']for p in row['per_pose']]
        assert sorted(display['thumbnail_poses'])==sorted(poses)
        assert display['representative_pose']in poses and display['bbox_includes_all_task_poses']
        assert not display['volume_ratios_printed']and not display['litres_printed']
        assert display['column_percent_labels'][0]==''
        expected_text=[value for value in display['column_percent_labels']if value]
        if row['passed']:expected_text+=['Placement '+str(group['placement'])for group in display['groups']]
        assert display['visible_text']==expected_text
        assert display['annotation_policy']=='percent_and_placement_only'
        assert display['bbox_style']=='complete_wireframe_box'
        assert display['bbox_edges_per_main_column']==([12,12,12]if row['passed']else[12,0,0])
        assert display['main_instances']==([dict(objects=1,supports=0),dict(objects=1,supports=1),dict(objects=1,supports=1)]
            if row['passed']else[dict(objects=1,supports=0),dict(objects=0,supports=0),dict(objects=0,supports=0)])
        if row['passed']:
            source=E.ROOT/row['step4_source'];final=E.ROOT/row['step53_source']
            with np.load(source/'layout.npz')as arrays:layout={k:arrays[k].copy()for k in arrays.files}
            assert [g['poses']for g in display['groups']]==[s['poses']for s in row['fixture_states']]
            for group in display['groups']:
                for index in group['indices']:
                    np.testing.assert_allclose(layout['native_world'][layout['hosts'][index]],group['fixture_transform'],atol=1e-11,rtol=0)
            for key,label in zip(BOX_KEYS[1:],display['column_percent_labels'][1:]):
                assert label==f'{row["bbx_increase_percent"][key]:+.0f}%'
            nominal=trimesh.load(source/'support.obj',force='mesh',process=False)
            report=E.read(final/'data/report.json');full=trimesh.load(final/report['support'],force='mesh',process=False)
            actual=static_bounds(body.vertices,layout,nominal.vertices,full.vertices,[p['pose']for p in row['per_pose']])
            E.normalize(actual)
            for key in BOX_KEYS:
                np.testing.assert_allclose(actual[key]['dimensions_m'],row[key]['dimensions_m'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(actual['relative_bbx_volume'][key],row['relative_bbx_volume'][key],atol=1e-12,rtol=0)
            checked+=1
        else:
            assert display['column_percent_labels']==['','','']and not display['visible_text']
            assert row['with_full_support']is None and row['with_step4_support']is None
            extents=[]
            for pose in E.poses_for(row['id']):
                with np.load(E.ROOT/'objects'/name/'poses'/pose/'setup.npz')as arrays:
                    rotation=arrays['T_world_mesh'][:3,:3]
                extents.append(np.ptp(body.vertices@rotation.T,axis=0))
            np.testing.assert_allclose(np.max(extents,axis=0),row['object_static']['dimensions_m'],atol=1e-12,rtol=0)
        assert not(E.A.VISUALS/name/row['id']/'evaluation').exists()
        assert not(E.A.result_root(name)/row['id']/'evaluation').exists()
    assert not list((E.A.result_root(name)/'data').glob('evaluation_*'))
    records.update(passed=True,independent_metrics_recomputed=True,certified_groups_recomputed=checked,
        rendered_groups=len(summary['results']),normalized_object_baseline_verified=True,
        source_models_unchanged=True,one_fixed_image_per_case=True,no_experiment_evaluation_directories=True)
    records.update(single_pose_main_scenes_verified=True,all_task_pose_thumbnails_verified=True,
        actual_support_placement_groups_verified=True,percent_only_display_verified=True)
    records['only_percent_and_placement_text_verified']=True
    records['complete_wireframe_boxes_verified']=True
    E.save(folder/'verification.json',records)
    print('EVALUATION_VERIFIED',len(summary['results']),'images;',checked,'certified supports',flush=True)
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',nargs='?',default='B')
    verify(parser.parse_args().object)
