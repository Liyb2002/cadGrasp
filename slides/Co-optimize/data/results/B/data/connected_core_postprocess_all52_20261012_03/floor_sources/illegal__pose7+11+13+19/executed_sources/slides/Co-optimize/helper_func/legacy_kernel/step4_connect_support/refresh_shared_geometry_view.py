"""Refresh presentation without rebuilding geometry or rerunning mechanics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT
from step3_scheculer import contacts as I


def head_regions(report, bases, offsets, margin=.004):
    """Display masks: each original patch's local bounds plus a 4 mm collar.

    These boxes only clip the rendered exterior surface. They add no material
    and do not change any contact, collision or bearing calculation.
    """
    source = (ROOT/report['source_schedule']).parent
    regions = []
    seen = {}
    for pose, basis, offset in zip(report['poses'], bases, offsets):
        for contact in I.read_contacts(source/f'contacts_{pose}.npz'):
            points = contact['triangles_m'].reshape(-1, 3)@basis+offset
            if contact['candidate_id'] in seen and report.get('physical_head_definition') in ('five_unique_heads_one_shared_patch', 'unique_registered_heads'):
                seen[contact['candidate_id']]['active_poses'].append(pose)
                continue
            regions.append(dict(id=contact['candidate_id'], pose=pose,
                                active_poses=[pose],
                                low=(points.min(0)-margin).tolist(),
                                high=(points.max(0)+margin).tolist()))
            seen[contact['candidate_id']] = regions[-1]
    return regions


def write_viewer(out, data):
    template = Path(__file__).with_name('shared_geometry_viewer.html').read_text()
    three = (ROOT/'slides/reuse/vendor/three.min.js').read_text()
    html = template.replace('__THREE__', three).replace('__DATA__', json.dumps(data, separators=(',', ':')))
    (Path(out)/'index.html').write_text(html)


def refresh(out):
    out = Path(out)
    report = json.loads((out/'report.json').read_text())
    I.check_hashes(report['provenance']['inputs'])
    for name, digest in report['artifacts'].items():
        if I.sha256(out/name) != digest:
            raise ValueError(f'Physical artifact changed: {name}')
    html = (out/'index.html').read_text()
    data, _ = json.JSONDecoder().raw_decode(html.split('const DATA=', 1)[1])
    with np.load(out/'geometry.npz') as geometry:
        data['head_regions'] = head_regions(report, geometry['rotations'], geometry['local_offsets_m'])
    report['presentation'] = dict(color_mode='colored_heads_neutral_bodies',
        head_neighborhood_margin_m=.004,
        colored_region='original contact patch local bounding box expanded by 4 mm, clipped to existing exterior faces',
        neutral_region='added connecting bodies, integral soles and heels outside the head neighborhood',
        physical_artifacts_unchanged=True, geometry_and_mechanics_rerun=False)
    # Preserve the sources of the original physical checks separately from
    # current presentation dependencies; no old check is claimed as a rerun.
    provenance = report['provenance']
    provenance.setdefault('geometry_generation_code', dict(provenance['code']))
    view_paths = [Path(__file__), Path(__file__).with_name('fixture_view.py'),
                  Path(__file__).with_name('shared_geometry_viewer.html'),
                  Path(__file__).with_name('export_shared_geometry.cjs')]
    provenance['presentation_code'] = I.hashes(view_paths)
    provenance['code'].update(provenance['presentation_code'])
    I.save(out/'report.json', report)
    data['report'] = report
    write_viewer(out, data)
    for name, digest in report['artifacts'].items():
        assert I.sha256(out/name) == digest, name
    print('Head colors refreshed; physical geometry and equilibrium artifacts unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', default=ROOT/'slides/baseline_algo/output/B/pose1+3/step4')
    refresh(parser.parse_args().output)
