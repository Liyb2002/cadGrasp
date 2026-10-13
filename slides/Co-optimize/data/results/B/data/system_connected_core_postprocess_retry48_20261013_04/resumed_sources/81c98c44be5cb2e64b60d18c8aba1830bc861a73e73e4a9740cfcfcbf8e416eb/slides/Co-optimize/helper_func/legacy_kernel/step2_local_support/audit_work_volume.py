"""Probe the saved visible families against the actual Step 1 ray predicate."""
from pathlib import Path
import numpy as np
from step2_local_support import work_volume as W


def run(path):
    path=Path(path);work=W.WorkVolume.read(path)
    points=[];rays=[];owners=[]
    for i in np.flatnonzero(work.certified_visible):
        triangle=work.triangles[i];center=triangle.mean(axis=0)
        positions=np.vstack([center,.999*triangle+.001*center])
        slopes=work.slopes[i,:work.slope_counts[i]];mean=slopes.mean(axis=0)
        which=np.linspace(0,len(slopes)-1,min(3,len(slopes)),dtype=int)
        directions=np.vstack([mean,.99*slopes[which]+.01*mean])
        vectors=work.normals[i]+directions[:,:1]*work.e1[i]+directions[:,1:]*work.e2[i]
        vectors/=np.linalg.norm(vectors,axis=1)[:,None]
        vectors=vectors[(vectors@work.normals[i])>=np.cos(np.deg2rad(work.half_angle_deg))-1e-12]
        for point in positions:
            points.extend([point+work.ray_offset*work.normals[i]]*len(vectors));rays.extend(vectors);owners.extend([i]*len(vectors))
    blocked=np.empty(0,bool)
    if points and work.object_mesh is not None:
        blocked=work.object_mesh.ray.intersects_any(np.asarray(points),np.asarray(rays))
    failures=[dict(cell_index=int(owners[i]),origin_m=np.asarray(points[i]).tolist(),direction=np.asarray(rays[i]).tolist()) for i in np.flatnonzero(blocked)[:20]]
    report=dict(complete=True,passed=not bool(np.any(blocked)),sampled_visible_rays_checked=len(points),
        blocked_probe_count=int(np.sum(blocked)),failures=failures,
        transition_cells=int(np.sum(~work.certified_visible)),
        scope='Independent numerical probes of analytically constructed visibility bounds; finite probes alone are not a continuous certificate',
        provenance=dict(inputs=W.I.hashes([path,path.parent/'work_volume.npz']),code=W.I.hashes([Path(__file__)])))
    W.I.save(path.parent/'work_volume_audit.json',report)
    if not report['passed']:raise RuntimeError(f'{len(failures)} visible-family ray probes failed; see work_volume_audit.json')
    print('  Step 2 visibility audit:',len(points),'rays; no blocked probes',flush=True)
    return report
