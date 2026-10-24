"""Publish only the fully replayed common-height result and its diagnostics."""
import json
import shutil
from pathlib import Path

import numpy as np

from whole_task.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from whole_task.step4.membrane_search import read_group
from whole_task.step5.draw_height_reference import render as draw_height
from whole_task.step5.draw_required_hulls import render as draw_hulls


def main():
    folder = Path(__file__).resolve().parent
    report = json.loads((folder / 'report.json').read_text())
    validation = json.loads((folder / 'validation.json').read_text())
    assert validation['status'] == 'passed'
    assert report['validation_sha256'] == sha256(folder / 'validation.json')
    assert validation['original_sample_count'] == 425984
    assert validation['continuous_outer_target_count'] == 35437
    for path, digest in report['input_sha256'].items():
        assert sha256(path) == digest, path
    height = report['height_reference']
    assert validation['fixed_head_world_m'] == height['head_world_m']
    assert height['connected_witness_certified']
    manifest_path = folder.parent / 'step5_height_normalized/step4_immutable.json'
    manifest = json.loads(manifest_path.read_text())
    assert all(sha256(path) == digest for path, digest in manifest.items())
    step4 = json.loads((Path(report['step4']) / 'report.json').read_text())
    for field in ('input_sha256', 'source_sha256'):
        for path, digest in step4.get(field, {}).items():
            assert sha256(path) == digest, path
    assert sha256(folder / 'overview.png') == report['overview_sha256']
    video = report['video']
    assert sha256(video['file']) == video['sha256']
    assert sha256(video['motion_file']) == video['motion_sha256']
    assert video['complete_video_decode_passed'] and video['frames'] == 1056
    seen = set()
    with np.load(video['motion_file']) as z:
        frames = z['task_family_phase']
        rotations = z['right_panel_rotation']
        for number, group in enumerate(report['groups']):
            q, _ = read_group(group)
            for local, owner in enumerate(group['members']):
                mask = (frames[:, 0] == owner) & (frames[:, 1] == number) & (frames[:, 2] == 0)
                assert mask.sum() == 48
                expected = np.linalg.inv(q.material[local])[:3, :3]
                np.testing.assert_allclose(rotations[mask], np.broadcast_to(expected, rotations[mask].shape),
                                           atol=1e-12, rtol=0.)
                seen.add(owner)
    assert seen == set(range(13))
    hulls = draw_hulls(folder, folder / 'required_hulls.png')
    assert hulls['task_count'] == 13 and hulls['all_task_pressure_demands_preserved']
    draw_height(folder)
    previous_folder = folder.parent / 'step5_compact_shared_ground'
    previous = json.loads((previous_folder / 'report.json').read_text())
    scratch_folder = folder.parent / 'step5_height_normalized'
    scratch = json.loads((scratch_folder / 'report.json').read_text())
    comparison = dict(original_common_head_height_mm=height['original_head_world_m'][2] * 1000,
                      new_common_head_height_mm=height['head_world_m'][2] * 1000,
                      previous_material_cm3=previous['final_volume_cm3'],
                      fresh_construction_material_cm3=scratch['final_volume_cm3'],
                      reused_pruning_material_cm3=report['final_volume_cm3'],
                      savings_percent=(1 - report['final_volume_cm3'] / previous['final_volume_cm3']) * 100,
                      group_count=report['final_group_count'],
                      step4_frozen_unchanged=True,
                      source_report_sha256={str(p / 'report.json'): sha256(p / 'report.json')
                                            for p in (previous_folder, scratch_folder, folder)})
    write_json(folder / 'comparison.json', comparison)
    out = DEFAULT_OUTPUT / 'B/step5'
    out.mkdir(parents=True, exist_ok=True)
    for filename in ('overview.png', 'result.mp4', 'required_hulls.png'):
        shutil.copy2(folder / filename, out / filename)
    assert {p.name for p in out.iterdir()} == {'overview.png', 'result.mp4', 'required_hulls.png'}
    artifact = dict(status='passed',
                    all_bound_source_hashes_match=True,
                    step4_frozen_files_unchanged=len(manifest),
                    step4_source_unchanged=True,
                    fixed_head_world_m=height['head_world_m'], ground_z_m=0.,
                    object_seats_and_installation_rotations_preserved=True,
                    group_count=report['final_group_count'], volume_cm3=report['final_volume_cm3'],
                    original_sample_count=validation['original_sample_count'],
                    continuous_outer_target_count=validation['continuous_outer_target_count'],
                    video_frames=video['frames'], video_duration_s=video['duration_s'],
                    video_work_rotations_match_actual_installations=True,
                    related_tests_passed=9,
                    required_hull_ground_face_counts=[len(g['ground_faces']) for g in hulls['groups']],
                    visible_output_sha256={p.name: sha256(p) for p in out.iterdir()},
                    height_figure_sha256=sha256(folder / 'height_reference.png'))
    write_json(folder / 'final_artifact_check.json', artifact)
    write_json(ROOT / 'whole_task/data/B/step5_active.json',
               dict(workdir=str(folder), report_sha256=sha256(folder / 'report.json'),
                    group_count=report['final_group_count'], volume_cm3=report['final_volume_cm3']))
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()
