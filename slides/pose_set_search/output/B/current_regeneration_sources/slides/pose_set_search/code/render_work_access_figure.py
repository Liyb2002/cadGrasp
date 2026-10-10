"""One orbit sheet of the unchanged work exclusion with a round display cap.

The arbitrary finite excerpt is the original work volume intersected with one
sphere, rather than a different axial end plane for every source triangle.
Neither the saved fixture nor the unbounded search exclusion is modified.
"""
import argparse
import hashlib
from render_mesh_media import (C,HERE,Path,np,Image,Scene,WorkingSurface,WorkAccess,
    Panel,ELEVATION,GRAY,BLUE,FORBIDDEN,render_scene,transform_mesh,unpack_solid)
from case_sets import CASES,case_directory


def rounded_excerpt(body,work):
    access=WorkAccess(body,work.ids,work.access_definition['half_angle_deg'])
    center=np.average(body.triangles_center[work.ids],axis=0,weights=body.area_faces[work.ids])
    radius=float(body.extents.max())*.65
    sphere=C.trimesh.creation.icosphere(subdivisions=5,radius=radius)
    sphere.apply_translation(center)
    length=access.required_length(sphere.vertices)
    region=unpack_solid(access.solid(length)^C.S.solid(sphere))
    # This remains a subset of the exact conservative exclusion used in search.
    assert access.contains_points(region.vertices).all()
    assert np.linalg.norm(region.vertices-center,axis=1).max()<=radius+1e-10
    cap_minimum=float(np.min(np.einsum('fc,fc->f',sphere.face_normals,
                                      sphere.triangles_center-center)))
    return region,dict(truncation='original unbounded exclusion intersected with one sphere',
        center_object_m=center.tolist(),radius_m=radius,
        spherical_cap_subdivisions=5,maximum_cap_radial_error_m=radius-cap_minimum,
        source_cone_definition=access.definition(),subset_of_search_exclusion=True,
        axial_cap_beyond_display_sphere=True,unbounded_search_exclusion_unchanged=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=HERE/'output/B')
    parser.add_argument('--case',choices=list(CASES),default='pose1-10')
    parser.add_argument('--method',choices=['whole','incremental'],default='incremental')
    parser.add_argument('--pose',type=int,default=1)
    args=parser.parse_args()
    directory=case_directory(args.root,args.case)/args.method
    body=C.state('B',f'pose_{args.pose}')[2];scene=Scene(directory,body)
    k=scene.poses.index(f'pose_{args.pose}');work=scene.work[k]
    region,definition=rounded_excerpt(body,work)
    frame=scene.native[scene.final.hosts[k]];q=frame@scene.final.placements[k]
    region=transform_mesh(region,q)
    region.metadata.update(uniform_display_color=True,
        work_region_key=('round_excerpt',tuple(work.ids),q.tobytes()))
    posed_body=transform_mesh(body,q);support=transform_mesh(scene.support,frame)
    points=np.vstack([region.vertices,posed_body.vertices,support.vertices])
    azimuths=[-45+45*i for i in range(8)];size=(800,700)
    panels=[Panel(None,size=size,points=points,elevation=ELEVATION,azimuth=a) for a in azimuths]
    scale=min(panel.scale for panel in panels)
    sheet=Image.new('RGB',(4*size[0],2*size[1]),'white')
    for i,panel in enumerate(panels):
        panel.scale=scale
        layers=work.layers(q)+[(posed_body,GRAY),(support,BLUE),(region,FORBIDDEN,.28)]
        sheet.paste(render_scene(panel,layers),((i%4)*size[0],(i//4)*size[1]))
    path=args.root/'work_access_isometric.png';sheet.save(path)
    C.save(path.with_suffix('.json'),dict(complete=True,file=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),onscreen_text=False,
        pose=scene.poses[k],case=directory.parent.name,method=args.method,
        view_count=8,view_order='left to right, top then bottom; 45-degree orbit steps',
        elevation_degrees=ELEVATION,azimuth_degrees=azimuths,
        source_layout=str((directory/'sampled_layout.npz').relative_to(args.root)),
        support_sha256=hashlib.sha256((directory/'support.obj').read_bytes()).hexdigest(),
        object_to_world=q.tolist(),support_to_world=frame.tolist(),
        display_region=definition,geometry_and_search_unchanged=True,
        forbidden_region_color=FORBIDDEN,region_alpha=.28))
    print('ROUND WORK-ACCESS ORBIT FIGURE',path,flush=True)


if __name__=='__main__':main()
