import json,hashlib,itertools
from pathlib import Path
import numpy as np,trimesh
root=Path('/home/yli581/Desktop/cadGrasp');raw=trimesh.load(root/'objects/B/mesh.stl',force='mesh')
for i in range(1,21):
 pose=f'pose_{i}';src=root/f'slides/DSL_algo/output/B/independent_poses/{pose}/step_1_needs/needs.json';d=json.loads(src.read_text());prov=d['provenance'];T=np.array(d['frame']['T_world_mesh']);mask=np.zeros(len(d['geometry']['faces']),bool);mask[d['geometry']['work_face_ids']]=True
 worlds=[raw.vertices@T[:3,:3].T+T[:3,3],trimesh.transform_points(raw.vertices,T)]
 pivots=[]
 for v in worlds:
  pt=v[np.argmin(v[:,2])].copy();pt[2]=0.;pivots.append(pt)
 for p in (root/'slides/DSL_algo/output/B').glob(f'*/step0_pose_selection/data/floor_contact_{pose}.npz'):
  with np.load(p) as z:
   if np.allclose(z['moment_origin_m'],d['frame']['moment_origin_m'],atol=1e-14,rtol=0):pivots.append(z['original_pivot_m'].copy())
 coms=[np.array(d['frame']['moment_origin_m']),T[:3,:3]@raw.center_mass+T[:3,3],trimesh.transform_points(raw.center_mass[None],T)[0]]
 target=root/f'slides/DSL_algo/output/B/independent_poses_gpu_v12/{pose}/input/setup.npz';target.parent.mkdir(parents=True,exist_ok=True)
 matched=False
 for pivot,com in itertools.product(pivots,coms):
  np.savez_compressed(target,object='B',pose_id=pose,T_world_mesh=T,com_m=com,work_faces=mask,floor_contact_m=pivot,mesh_sha256=prov['mesh_sha256'],poses_sha256=prov['poses_sha256'],K=.5,cone_half_deg=prov['setup_snapshot_cone_half_deg'],tip=-1)
  if hashlib.sha256(target.read_bytes()).hexdigest()==prov['setup_snapshot_sha256']:matched=True;break
 if not matched:target.unlink()
 print(pose,'EXACT MATCH' if matched else 'NOT MATCHED',flush=True)
