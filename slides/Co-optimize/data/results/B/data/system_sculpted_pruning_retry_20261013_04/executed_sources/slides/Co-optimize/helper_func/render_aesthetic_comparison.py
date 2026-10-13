"""Render old/new saved CAD solids with exactly the same camera per pair."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_dynamic_postprocess import ROOT,COOPT,BASE,sha,save
sys.path[:0]=[str(COOPT/x)for x in ['helper_func','step5.3','step5.1']]
import stage5_shared as C


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True)
    args=p.parse_args();system=BASE/'data'/('system_'+args.experiment)
    before={r['id']:r for r in json.loads((system/'previous_evaluation.json').read_text())['cases']}
    evaluation=json.loads((system/'evaluation_results.json').read_text());renderer=C.V.Renderer();pictures=[]
    for row in evaluation['cases']:
        case=dict(id=row['id'],passed=True,pose_count=row['object_pose_count'],experiment=row['source_experiment'])
        inputs=C.Inputs('B',system/'selection.json',Path(row['step51_source']).name);data=inputs.load(case)
        state=data['states'][0]
        polygons=[C.F.transform_points(C.F.transform_points(r['hull'],np.linalg.inv(r['fixture_transform'])),state)for r in data['rows']]
        for stage,key in [('step5.2','step52_source'),('step5.3','step53_source')]:
            folders=[ROOT/before[row['id']][key],ROOT/row[key]];meshes=[];sources=[]
            for folder in folders:
                report=json.loads((folder/'data/report.json').read_text());path=folder/report['support']
                meshes.append(C.posed(C.trimesh.load(path,force='mesh',process=False),state));sources.append(dict(path=C.relative(path),sha256=sha(path)))
            camera=C.V.camera_for(np.vstack([m.vertices for m in meshes]+polygons));images=[]
            for mesh in meshes:
                parts=[C.V.CPU.placed(C.V.piece(mesh,C.V.BLUE))]
                for poly,r in zip(polygons,data['rows']):parts.append(C.V.CPU.placed(C.H.outline_piece(poly,r['color'],camera,700)))
                images.append(renderer.render(parts,camera,700))
            fig,axes=plt.subplots(1,2,figsize=(12,6))
            for ax,image,title in zip(axes,images,['Previous grid pruning','Rounded block retention']):
                ax.imshow(np.asarray(image));ax.set_title(title);ax.axis('off')
            fig.suptitle(row['id']+' | '+stage+' | same camera, scale and BBX',fontsize=12)
            fig.tight_layout(rect=(0,0,1,.96));path=COOPT/'output/B'/row['id']/stage/'comparison.png'
            fig.savefig(path,dpi=150);plt.close(fig)
            pictures.append(dict(id=row['id'],stage=stage,path=C.relative(path),sha256=sha(path),
                camera=camera.tolist()if isinstance(camera,np.ndarray)else str(camera),sources=sources))
        inputs.verify()
    save(system/'visual_comparisons.json',dict(complete=True,groups=52,pictures=pictures,geometry_edited=False,
        force_solver_run=False,boolean_run=False,same_camera_within_each_pair=True))
    evaluation['pictures']=[r for r in evaluation['pictures']if not r['path'].endswith('/comparison.png')]+[
        {k:r[k]for k in ['id','path','sha256']}for r in pictures]
    save(system/'evaluation_results.json',evaluation)
    print('SAME CAMERA COMPARISONS',len(pictures),'PNG',flush=True)


if __name__=='__main__':main()
