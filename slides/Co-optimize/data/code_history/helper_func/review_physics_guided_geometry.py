"""Check trajectory-field direction sensitivities on saved real input geometry."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import _bootstrap
from co_common import *
from exit_clearance import ExitClearance
from physics_guided_geometry import SweepDistanceModel,tangent_frames


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--set',default='pose1+4+7+12+21+27')
    parser.add_argument('--directions',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    base=HERE/'output/B'/args.set
    groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    group=next(g for g in groups if g['id']==args.set)
    task,transform,mesh=state('B',group['poses'][0])
    z=np.load(base/'step3/step3.2/data/contacts.npz')
    points=z['triangles_mesh_m'].reshape(-1,3)
    normals=np.repeat(mesh.face_normals[z['source_faces']],3,axis=0)
    indices=np.linspace(0,len(points)-1,12).astype(int)
    directions=np.load(args.directions)['directions'];frames=tangent_frames(directions)
    model=SweepDistanceModel(ExitClearance(mesh),points[indices],normals[indices],.5,mesh.extents.max())
    values,jac=model.linearize(directions,frames)
    rows=[]
    for i in range(len(directions)):
        for axis in range(2):
            h=1e-4
            plus=directions[i]+h*frames[i,:,axis];plus/=np.linalg.norm(plus)
            minus=directions[i]-h*frames[i,:,axis];minus/=np.linalg.norm(minus)
            half=(model.distances(plus)-model.distances(minus))/(2*h)
            error=np.abs(half-jac[:,i,axis])
            rows.append(dict(pose=group['poses'][i],axis=axis,max_absolute_difference=float(error.max()),
                relative_norm_difference=float(np.linalg.norm(error)/max(np.linalg.norm(half),1e-8)),
                finite_difference_2e4=jac[:,i,axis].tolist(),finite_difference_1e4=half.tolist()))
    save(args.out,dict(set=args.set,ray_indices=indices.tolist(),checks=rows,
        policy='Real-input guidance derivative stability at 12 deterministic probes; not a final support acceptance or global smoothness claim',
        provenance=provenance([args.directions,base/'step3/step3.2/data/contacts.npz',model.field.path]+task.inputs,
            [Path(__file__),HERE/'helper_func/physics_guided_geometry.py',HERE/'helper_func/physics_guided_field.py',HERE/'helper_func/exit_clearance.py'])))
    print('Maximum relative norm difference:',max(row['relative_norm_difference'] for row in rows))


if __name__=='__main__':main()
