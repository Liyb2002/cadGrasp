"""Draw the saved layout's reconstructed contact core, without force replay."""
from pathlib import Path
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    directory = Path(args.run).resolve()
    payload = json.loads((directory/'payload.json').read_text())
    tree = Path(payload['frozen'])
    sys.path[:0] = [str(tree), str(tree/'slides/Co-optimize/helper_func'),
                   str(tree/'slides/Co-optimize/step4.2')]
    import _bootstrap
    import artifact_store as A
    from co_common import ROOT, S, D, trimesh
    A.COOPT = ROOT/'slides/Co-optimize'
    A.RESULTS = A.COOPT/'data/results'
    A.VISUALS = A.COOPT/'output'
    A.INITIALIZATION_SOURCES = A.COOPT/'data/initialization_sources'
    from oracle_search import OracleModel
    from whole_pipeline import load_layout
    from connected_core import prepare
    from whole_step4_render import layers_for_pose, raster_tile
    from PIL import Image, ImageDraw, ImageFont
    out = directory/'B'/payload['id']/'step4/step4.2'/payload['options']['output_name']
    report = json.loads((out/'report.json').read_text())
    model = OracleModel(report['poses'],'B',demand_nodes=32,oracle_work_mode='normal')
    model.require_connected_core = True
    seed = A.RESULTS/'B'/payload['id']/'step3/step3.1/wrapped_support.obj'
    model.registered_seed = S.solid(trimesh.load(seed,force='mesh',process=False))
    def forbidden(*args, **kwargs):
        raise RuntimeError('Presentation must not rerun force acceptance')
    model.evaluate = forbidden
    layout = load_layout(out/'layout.npz')
    saved_geometry = out/'support_with_base.obj'
    if not saved_geometry.exists():
        saved_geometry = out/'support.obj'
    if saved_geometry.exists():
        core = trimesh.load(saved_geometry,force='mesh',process=False)
        record = report.get('connected_contact_core',{})
        reconstructed = False
    else:
        core, record = prepare(model, layout)
        reconstructed = True
    D.export_exact_obj(core, directory/'diagnostic_contact_core.obj')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    tiles = []
    for k, pose in enumerate(report['poses']):
        tile = raster_tile(layers_for_pose(model,layout,core,k),size=(520,440))
        panel = Image.new('RGB',(520,490),'white')
        panel.paste(tile,(0,50))
        failed = 32768-report['counts'][pose]
        ImageDraw.Draw(panel).text((16,12),pose+': '+str(failed)+' demands unmet',
            fill='#bb2a36' if failed else '#25664a',font=font)
        tiles.append(panel)
    image = Image.new('RGB',(520*5,490*2+60),'white')
    ImageDraw.Draw(image).text((16,14),
        payload['mode']+': '+('PASS' if report['passed'] else 'FAILED CANDIDATE')+
        ' | Saved layout; force acceptance not replayed',fill='#222222',font=font)
    for index,tile in enumerate(tiles):
        image.paste(tile,((index%5)*520,(index//5)*490+60))
    image.save(directory/'diagnostic_10poses.png')
    metadata = dict(complete=True,step4_passed=report['passed'],presentation_only=True,
        reconstructed_geometry=reconstructed,force_acceptance_replayed=False,layout_changed=False,
        geometry_role='retained contact core; no certified final base on a failed layout',
        connected_core=record,source_run=payload['options']['output_name'])
    (directory/'visualization.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    main()
