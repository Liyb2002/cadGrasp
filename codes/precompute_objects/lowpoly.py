"""Simplify objects with topology preservation and rebuild their exact input datasets."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
import pymeshlab
from codes.precompute_objects.run import precompute, refresh_registry
from codes.precompute_objects.work_regions import digest, write
from codes.precompute_objects.registry import active_objects


def split_large_faces(mesh, fraction=.005, cap=5000):
    """Conforming edge bisection; split both incident triangles to avoid T-junctions."""
    vertices=np.asarray(mesh.vertices).tolist();faces=np.asarray(mesh.faces).copy()
    total=mesh.area
    for _ in range(cap):
        v=np.asarray(vertices)
        triangles=v[faces]
        areas=.5*np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
        index=int(np.argmax(areas))
        if areas[index] <= fraction*total*(1+1e-12): break
        face=faces[index]
        lengths=np.linalg.norm(v[face]-v[np.roll(face,-1)],axis=1)
        k=int(np.argmax(lengths));a,b=int(face[k]),int(face[(k+1)%3])
        incident=np.flatnonzero(np.any(faces==a,axis=1)&np.any(faces==b,axis=1))
        if len(incident)!=2: raise ValueError('Edge bisection requires a closed two-manifold mesh')
        if len(faces)+len(incident)>cap: raise ValueError('Not enough triangle budget for work-patch discretization')
        mid=len(vertices);vertices.append(((v[a]+v[b])/2).tolist())
        added=[]
        for i in incident:
            old=faces[i].copy()
            for j in range(3):
                u,w,t=map(int,(old[j],old[(j+1)%3],old[(j+2)%3]))
                if {u,w}=={a,b}:
                    faces[i]=[u,mid,t];added.append([mid,w,t]);break
        faces=np.vstack([faces,added])
    else: raise ValueError('Local refinement did not converge')
    refined=trimesh.Trimesh(vertices,faces,process=False)
    np.testing.assert_allclose(refined.volume,mesh.volume,rtol=1e-10,atol=1e-15)
    return refined


def candidate(raw, cap=5000):
    if len(raw.faces)>cap:
        # Leave room for geometric-exact local splits of oversized planar triangles.
        for target in (cap-500,cap-1000,cap-1500):
            ms=pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(raw.vertices,raw.faces))
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target,preservetopology=True,
                preservenormal=True,optimalplacement=True,planarquadric=False,qualitythr=.3)
            current=ms.current_mesh()
            low=trimesh.Trimesh(current.vertex_matrix(),current.face_matrix(),process=True)
            try: return split_large_faces(low,cap=cap)
            except ValueError as error:
                if 'budget' not in str(error): raise
        raise ValueError('Could not fit conforming work mesh in triangle budget')
    return split_large_faces(raw,cap=cap)


def simplify(name, cap=5000, seed=20261003, budget=12000):
    began=time.monotonic();folder=ROOT/'objects'/name
    original=trimesh.load(folder/'mesh.stl',force='mesh')
    original_hash=digest(folder/'mesh.stl')
    meta=json.loads((folder/'meta.json').read_text())
    low=candidate(original,cap)
    with tempfile.TemporaryDirectory(prefix=f'cadgrasp_lowpoly_{name}_') as tmp:
        path=Path(tmp)/'mesh.stl';low.export(path)
        low=trimesh.load(path,force='mesh')
        if (not low.is_watertight or not low.is_winding_consistent or low.euler_number!=original.euler_number
                or len(low.split(only_watertight=False))!=len(original.split(only_watertight=False))):
            raise ValueError('Simplification changed topology or produced an open mesh')
        if len(low.faces)>cap: raise ValueError('Triangle budget exceeded')
        p=trimesh.sample.sample_surface(original,8000,seed=42)[0]
        q=trimesh.sample.sample_surface(low,8000,seed=43)[0]
        distances=np.r_[trimesh.proximity.closest_point(low,p)[1],trimesh.proximity.closest_point(original,q)[1]]
        scale=float(np.linalg.norm(original.extents))
        volume_change=float(low.volume/original.volume-1)
        if abs(volume_change)>.025 or np.quantile(distances,.99)>.003*scale or distances.max()>.01*scale:
            raise ValueError(f'Geometry error too large: volume={volume_change:.4%}, sampled max={distances.max()*1000:.3f}mm')
        row=dict(object=name,original_faces=len(original.faces),final_faces=len(low.faces),cap=cap,
            watertight=True,euler_number=low.euler_number,volume_change_pct=volume_change*100,
            sampled_p99_distance_mm=float(np.quantile(distances,.99)*1000),
            sampled_max_distance_mm=float(distances.max()*1000),distance_sample_count=16000,
            sampled_distance_is_not_hausdorff_bound=True,original_mesh_sha256=original_hash,
            mesh_sha256=digest(path),method='Topology-preserving quadric edge collapse + conforming local planar subdivision')
        print('LOWPOLY CANDIDATE',name,len(original.faces),'->',len(low.faces),row['sampled_p99_distance_mm'],flush=True)
        density=float(meta.get('density_kg_m3',1000.))
        low.density=density
        meta.update(n_vertices=len(low.vertices),n_faces=len(low.faces),watertight=True,
            extents_m=low.extents.tolist(),bounds_m=low.bounds.tolist(),volume_m3=float(low.volume),
            mass_kg=float(low.mass),com_mesh_frame=low.center_mass.tolist(),inertia_com=low.moment_inertia.tolist(),
            convex=bool(low.is_convex),hull_volume_ratio=float(low.volume/low.convex_hull.volume),lowpoly=row)
        # New mesh, metadata and every dependent input are staged and published together.
        result=precompute(name,seed,budget,mesh_source=path,mesh_metadata=meta)
    row.update(passed=True,pose_count=30,set_count=20,seconds=round(time.monotonic()-began,3))
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects',nargs='*');parser.add_argument('--faces',type=int,default=5000)
    parser.add_argument('--jobs',type=int,default=4);parser.add_argument('--seed',type=int,default=20261003)
    parser.add_argument('--budget',type=int,default=12000);args=parser.parse_args()
    if args.faces<2000: parser.error('Use at least 2000 triangles for this working-patch dataset')
    rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        jobs={pool.submit(simplify,name,args.faces,args.seed,args.budget):name for name in args.objects or active_objects()}
        for future in as_completed(jobs):
            name=jobs[future]
            try: rows.append(future.result())
            except Exception as error:
                rows.append(dict(object=name,passed=False,error=repr(error)))
                print('LOWPOLY FAILED',name,repr(error),flush=True)
            write(ROOT/'codes/precompute_objects/lowpoly_report.json',dict(complete=False,cases=rows))
    refresh_registry()
    write(ROOT/'codes/precompute_objects/lowpoly_report.json',dict(complete=True,passed=all(r['passed'] for r in rows),cases=rows))
    if any(not r['passed'] for r in rows):raise SystemExit(2)

if __name__=='__main__':main()
