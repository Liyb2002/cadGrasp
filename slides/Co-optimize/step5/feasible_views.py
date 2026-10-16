"""Independent saved-model Evaluation and PNGs for the transactional Step5."""
from pathlib import Path
import argparse
import importlib.util
import sys

COOPT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COOPT/name) for name in ['helper_func', 'evaluation', 'step5.1', 'step5.3']]
import stage5_shared as C
from metrics import static_bounds
from feasible_pipeline import read, save, relative, digest


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--selection-summary', type=Path, required=True)
    parser.add_argument('--step51-experiment', required=True)
    parser.add_argument('--skip-images', action='store_true')
    parser.add_argument('--skip-evaluation', action='store_true')
    args = parser.parse_args()
    inputs = C.Inputs(args.object, args.selection_summary, args.step51_experiment)
    renderer = C.V.Renderer() if not args.skip_images else None
    R = load_module('feasible_final_renderer', COOPT/'step5.3/render.py') if renderer else None
    E = load_module('feasible_saved_evaluation', COOPT/'evaluation/run.py') if not args.skip_evaluation else None
    evaluation_renderer=(renderer or C.V.Renderer())if E else None
    results = []
    for case in inputs.cases:
        data = inputs.load(case)
        folder = inputs.base/case['id']/'step5/step5.3'/args.experiment
        report = read(folder/'data/report.json')
        final_path = folder/report['support']
        core_path = C.ROOT/report['step52_source']/'pruned_support.obj'
        for path in [folder/'data/report.json', final_path, core_path]:
            inputs.protected[relative(path)] = digest(path)
        final = C.trimesh.load(final_path, force='mesh', process=False)
        core = C.trimesh.load(core_path, force='mesh', process=False)
        with C.np.load(data['selected']/'layout.npz') as arrays:
            layout = {k:arrays[k].copy() for k in arrays.files}
        result = static_bounds(data['body'].vertices, layout, data['fixture'].vertices,
                               final.vertices, data['record']['poses'])
        crop_result = static_bounds(data['body'].vertices, layout, data['fixture'].vertices,
                                    core.vertices, data['record']['poses'])
        result.update(id=case['id'], step53_source=relative(folder),
                      bbx_trim_l=crop_result['with_full_support']['volume_litres'],
                      step4_full_bbx_l=report['baseline_bbx_l'],
                      material_full_cm3=abs(float(final.volume))*1e6,
                      saved_step4_fallback_used=report['saved_step4_fallback_used'])
        masks = folder/'force_certificate.npz'
        inputs.protected[relative(masks)] = digest(masks)
        with C.np.load(masks) as cert:
            counts = [int(cert[k].size) for k in cert.files]
            if len(counts) != case['pose_count'] or any(n != 32768 for n in counts):
                raise ValueError('Incomplete saved certificate: '+case['id'])
            if not all(cert[k].all() for k in cert.files):
                raise ValueError('Saved demand failure: '+case['id'])
        result['original_demands_passed'] = sum(counts)
        if renderer:
            images = folder/'images'
            images.mkdir(parents=True, exist_ok=False)
            R.reference(renderer, data, images/'overview.png', fixture=final)
            R.reference(renderer, data, images/'bbx_trim.png', fixture=core)
            intermediate_images=core_path.parent/'images'
            intermediate_images.mkdir(parents=True,exist_ok=False)
            R.reference(renderer,data,intermediate_images/'overview.png',fixture=core)
            tiles = []
            intermediate_tiles=[]
            for row in data['rows']:
                tile, _ = C.H.render_pose(renderer,
                    C.posed(data['body'], row['object_transform']),
                    C.posed(final, row['fixture_transform']), row['cloud'], row['color'],
                    images/(row['pose']+'.png'), hull=row['hull'], size=700)
                tiles.append(tile)
                intermediate_tile,_=C.H.render_pose(renderer,
                    C.posed(data['body'],row['object_transform']),
                    C.posed(core,row['fixture_transform']),row['cloud'],row['color'],
                    intermediate_images/(row['pose']+'.png'),hull=row['hull'],size=700)
                intermediate_tiles.append(intermediate_tile)
            C.V.sheet(tiles, images/'all_poses.png')
            C.V.sheet(intermediate_tiles,intermediate_images/'all_poses.png')
            baseline_path = C.ROOT/report['optional_operations'][0]['output']/'support.obj'
            baseline = C.trimesh.load(baseline_path, force='mesh', process=False)
            T = data['states'][0]
            posed = [C.posed(m, T) for m in [baseline, final]]
            camera = C.V.camera_for(C.np.vstack([m.vertices for m in posed]))
            panels = [renderer.render([C.V.CPU.placed(C.V.piece(m, C.V.BLUE))], camera, 800)
                      for m in posed]
            C.V.sheet(panels, images/'comparison.png')
            result['images'] = {p.name:dict(path=relative(p), sha256=digest(p))
                                for p in images.glob('*.png')}
        if E:
            result['step4_source']=relative(data['selected'])
            E.publish_case(args.object,result,data['body'],data['fixture'],final,layout,
                evaluation_renderer,inputs.protected)
        results.append(result)
        print('STEP5 SAVED EVALUATION',case['id'],round(result['with_full_support']['volume_litres'],4),flush=True)
    inputs.verify()
    if E:
        full=inputs.base/'data'/('step5_feasible_'+args.experiment)/'summary.json'
        selection=read(full)if full.exists()else read(args.selection_summary)
        for case in selection.get('results',selection.get('selected',[])):
            if not case['passed']:E.unavailable(args.object,case,inputs.mesh,evaluation_renderer)
        summary=E.write_summary(args.object,selection,full if full.exists()else args.selection_summary)
        if summary['complete']:E.clean_legacy(args.object)


if __name__ == '__main__':
    main()
