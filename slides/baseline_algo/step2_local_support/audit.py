"""Independently check the actual exported circles with clipped-edge topology."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.distance import pdist
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step2_local_support import circles as P,connectivity as C

from step1.cases import pose_name


def run(name):
    domain,data,report=P.read(name);mesh=domain.mesh
    out=P.OUTPUTS/name/pose_name()/P.W.STAGE
    work=P.W.WorkVolume.read(out/'work_volume.json') if P.POLICY.ENFORCE_PROCESS_ACCESS else None
    checker=P.WC.ContactClearance(mesh,report['normal_depth_m'],work)
    analyzer=P.WC.D.Analyzer(mesh,report['normal_depth_m'])
    object_clearance=P.G.Clearance(mesh,mesh.extents.max()*1e-9)
    target=P.AREA_FRACTION*mesh.area;checks=[];direction_checks=[]
    # Independent geometric formulation: signed XY area / true 3D area is n_z.
    # Check every candidate, including rejected ones; never use a mean normal.
    for index,row in enumerate(report['patches']):
        a,b=data.offsets[index:index+2];faces=np.unique(data.source_faces[a:b])
        triangles=mesh.triangles[faces]
        first=triangles[:,1]-triangles[:,0];second=triangles[:,2]-triangles[:,0]
        projected=first[:,0]*second[:,1]-first[:,1]*second[:,0]
        magnitude=np.linalg.norm(np.cross(first,second),axis=1)
        nz=projected/magnitude
        passed=bool(len(nz) and np.all(np.isfinite(nz)))
        assert passed==row['bearing_direction_check']['passed'],(name,row['id'],'normal direction mismatch')
        if data.valid[index]:assert passed,(name,row['id'],'invalid normal geometry')
        if len(nz):
            np.testing.assert_allclose(nz.max(),row['bearing_direction_check']['maximum_outward_normal_z'],atol=1e-14)
        direction_checks.append(dict(id=row['id'],passed=passed,source_faces_checked=len(faces),
                                     maximum_outward_normal_z=float(nz.max()) if len(nz) else None))
    for index in np.flatnonzero(data.valid):
        a,b=data.offsets[index:index+2];tri=data.triangles[a:b];source=data.source_faces[a:b]
        row=report['patches'][index]
        actual=float(P.areas(tri).sum())
        assert 0.<actual<=target*(1+P.AREA_REL_TOL)
        if row['area_status']=='target_reached':
            assert abs(actual-target)<=target*P.AREA_REL_TOL
        else:
            assert row['area_status'].startswith('smaller_') and actual<target
        assert abs(actual-row['area_m2'])<target*1e-10
        assert np.linalg.norm(tri-data.centers_m[index],axis=2).max()<=data.radius_m[index]*(1+1e-9)
        assert not np.isin(source,domain.work_ids).any()
        assert tri[:,:,2].min()>=P.FLOOR_CLEARANCE_M-1e-12
        oriented=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        assert np.all(np.einsum('ij,ij->i',oriented,mesh.face_normals[source]) >= -mesh.area*1e-14), 'Contact winding opposes its source surface'
        assert (mesh.face_normals[source]@mesh.face_normals[data.center_faces[index]]).min()>=np.cos(np.deg2rad(P.MAX_NORMAL_ANGLE_DEG))-P.NORMAL_DOT_TOL
        # Independent formulation: the maximum chord between unit normals
        # is 2*sin(wrap/2). This checks all pairs from the exported contact.
        normals=mesh.face_normals[np.unique(source)]
        chord2=float(pdist(normals,metric='sqeuclidean').max(initial=0.))
        wrap=float(np.degrees(2*np.arcsin(np.clip(np.sqrt(chord2)/2,0,1))))
        assert wrap<=P.MAX_WRAP_ANGLE_DEG+1e-7,(name,row['id'],'total wrap exceeds 90 degrees',wrap)
        assert row['wrap_limit_satisfied'] and abs(wrap-row['wrap_angle_degrees'])<1e-6
        if row['radius_shrunk_for_wrap']:
            assert row['radius_m']<row['radius_before_wrap_m']
            assert actual<=row['area_before_wrap_m2']*(1+1e-9)
        # Use barycentric containment. The closest-point routine's absolute
        # dot-product tolerance misclassifies small triangles in metre units.
        center=data.centers_m[index];length=mesh.extents.max();contained=False
        for triangle in tri[source==data.center_faces[index]]:
            uv=np.linalg.lstsq((triangle[1:]-triangle[0]).T/length,(center-triangle[0])/length,rcond=None)[0]
            weights=np.r_[1-uv.sum(),uv]
            if weights.min()>=-1e-9 and np.linalg.norm(weights@triangle-center)<length*1e-9:
                contained=True;break
        assert contained,(name,row['id'],'center missing from exported circle')
        parts,connectivity=C.components(mesh,tri,source)
        assert len(parts)==1,(name,row['id'],'disconnected exported circle')
        # Recheck actual exported heads against object and floor even when
        # the process-access constraint is omitted.
        heads=analyzer.heads(dict(triangles_m=tri,source_faces=source,triangle_areas_m2=P.areas(tri)))
        for head in heads:
            assert head.vertices[:,2].min()>=-object_clearance.tol,(name,row['id'],'head enters floor')
            assert object_clearance.obstruction(head.vertices)==-1,(name,row['id'],'head enters object')
        work_check=checker.check_parts(heads)
        assert work_check['passed'],(name,row['id'],'exported head fails work-volume clearance',work_check)
        checks.append(dict(id=row['id'],area_m2=actual,area_fraction=float(actual/mesh.area),
                           area_status=row['area_status'],wrap_angle_degrees=wrap,
                           object_and_floor_clearance_verified=True,work_volume_check=work_check,**connectivity))
    P.save(P.OUTPUTS/name/pose_name()/'step2_local_support/audit.json',dict(object=name,complete=True,passed=True,valid_circles_checked=len(checks),
           bearing_direction_checks=direction_checks,vertical_hemisphere_filter_enforced=False,
           process_access_enforced=P.POLICY.ENFORCE_PROCESS_ACCESS,
           checks=checks,circles_sha256=P.sha256(P.OUTPUTS/name/pose_name()/'step2_local_support/circles.npz'),
           code_sha256=P.sha256(__file__),connectivity_code_sha256=P.sha256(C.__file__),
           work_volume_provenance=P.W.I.hashes(([out/'work_volume.json',out/'work_volume.npz'] if work is not None else [])+[Path(P.WC.__file__),Path(P.POLICY.__file__)])))
    print(name,'audited',len(checks),'valid connected circles, total wrap <=90 degrees; no vertical hemisphere filter',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or P.OBJECTS:run(name)
