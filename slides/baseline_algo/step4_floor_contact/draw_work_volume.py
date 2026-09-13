"""A smooth exterior illustration of the saved access volume, never query geometry."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull
import trimesh
from step2_local_support import render as R
from step2_local_support import work_volume as W

PURPLE=np.array([136.,101.,176.])
OUTLINE=np.array([91.,62.,130.])
SHELL_OPACITY=.40
BASIS=R.axes([.68, -1., .8])
CAP_RADIUS_FRACTION=.46
VIEWS=(('oblique',BASIS),('front',R.axes([0., -1., .30])),
       ('right',R.axes([1., 0., .30])),('back',R.axes([0., 1., .30])),
       ('left',R.axes([-1., 0., .30])),('top',R.axes([0., 0., 1.])))


def layer(triangles,colors,focus,width,size,unlit=(),basis=BASIS):
    if not len(triangles):
        return Image.new('RGB',(size,size),R.PAPER),np.full((size,size),-np.inf)
    image,ids=R.raster(triangles,colors,focus,basis,width,size,unlit=unlit)
    depth=np.full((size,size),-np.inf);yy,xx=np.nonzero(ids>=0)
    if len(xx):
        projected=R.project(triangles,focus,basis,width,size)
        a,b,c=np.moveaxis(projected[ids[yy,xx]],1,0)
        det=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
        u=((b[:,1]-c[:,1])*(xx+.5-c[:,0])+(c[:,0]-b[:,0])*(yy+.5-c[:,1]))/det
        v=((c[:,1]-a[:,1])*(xx+.5-c[:,0])+(a[:,0]-c[:,0])*(yy+.5-c[:,1]))/det
        depth[yy,xx]=u*a[:,2]+v*b[:,2]+(1-u-v)*c[:,2]
    return image,depth


def shell_mesh(work):
    """One shared spherical cap and a convex exterior, for illustration only.

    Every saved visibility family contributes source and exit points. A common
    sphere replaces the thousands of separate cone caps. Angular binning bounds
    display complexity; convexification intentionally fills small gaps. Neither
    operation changes work_volume.npz or the Step 5 collision domain.
    """
    triangles=work.original_triangles
    weights=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    center=np.average(triangles.mean(axis=1),weights=weights,axis=0)
    source_radius=float(np.linalg.norm(triangles-center,axis=2).max())
    radius=max(CAP_RADIUS_FRACTION*work.scale,1.08*source_radius)
    samples=[]
    for count in np.unique(work.slope_counts):
        selected=np.flatnonzero(work.slope_counts==count)
        slopes=work.slopes[selected,:count]
        slope_samples=np.concatenate([slopes,.5*(slopes+np.roll(slopes,-1,axis=1)),slopes.mean(axis=1,keepdims=True)],axis=1)
        rays=work.normals[selected,None]+slope_samples[:,:,:1]*work.e1[selected,None]+slope_samples[:,:,1:]*work.e2[selected,None]
        rays/=np.linalg.norm(rays,axis=2,keepdims=True)
        tri=work.triangles[selected]
        origins=np.concatenate([tri,tri.mean(axis=1,keepdims=True)],axis=1)
        delta=origins-center
        dot=np.einsum('fsi,fri->fsr',delta,rays)
        distance=-dot+np.sqrt(np.maximum(0.,dot**2+radius**2-np.sum(delta**2,axis=2)[:,:,None]))
        endpoints=origins[:,:,None]+distance[:,:,:,None]*rays[:,None]
        samples.append(endpoints.reshape(-1,3))
    if not samples:
        return trimesh.Trimesh(vertices=np.empty((0,3)),faces=np.empty((0,3),int),process=False),dict(center_m=center.tolist(),radius_m=radius)
    endpoints=np.vstack(samples)
    unit=(endpoints-center)/radius
    azimuth=np.arctan2(unit[:,1],unit[:,0]);latitude=np.arcsin(np.clip(unit[:,2],-1,1))
    # Average on the common sphere within small angular bins for a uniform mesh.
    bins=np.c_[np.floor((azimuth+np.pi)*96/(2*np.pi)),np.floor((latitude+np.pi/2)*48/np.pi)].astype(int)
    _,inverse=np.unique(bins,axis=0,return_inverse=True)
    sums=np.zeros((inverse.max()+1,3));np.add.at(sums,inverse,unit)
    sums/=np.linalg.norm(sums,axis=1,keepdims=True)
    cap=center+radius*sums
    # Preserve the actual source boundary; there are no individual cell caps.
    points=np.vstack([cap,work.triangles.reshape(-1,3)])
    points=np.unique(points,axis=0)
    hull=ConvexHull((points-center)/work.scale)
    faces=hull.simplices.copy()
    normals=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
    reverse=np.einsum('ij,ij->i',normals,hull.equations[:,:3])<0
    faces[reverse]=faces[reverse][:,::-1]
    mesh=trimesh.Trimesh(points,faces,process=False)
    mesh.remove_unreferenced_vertices()
    return mesh,dict(center_m=center.tolist(),radius_m=radius,common_top_z_m=float(center[2]+radius),
                     source_point_count=len(work.triangles)*3,cap_point_count=len(cap),face_count=len(faces))


def composite_shell(base,opaque_depth,shell_depth,focus,width,size,cap,basis=BASIS):
    """A single soft fill and silhouette, with no facet or internal-edge shading."""
    mask=np.isfinite(shell_depth)
    visible=mask&(shell_depth>opaque_depth+width*1e-8)
    # Analytic sphere lighting is continuous across all display triangles.
    yy,xx=np.indices(mask.shape)
    lateral_x=(xx+.5-size/2)*width/size
    lateral_y=-(yy+.5-size/2)*width/size
    safe_z=np.where(mask,shell_depth,0.)
    world=focus+lateral_x[:,:,None]*basis[0]+lateral_y[:,:,None]*basis[1]+safe_z[:,:,None]*basis[2]
    normal=world-np.asarray(cap['center_m'])
    normal/=np.maximum(np.linalg.norm(normal,axis=2,keepdims=True),1e-30)
    light=.75*basis[2]+.55*basis[1]-.25*basis[0];light/=np.linalg.norm(light)
    shade=.88+.12*np.maximum(0.,normal@light)
    paint=np.uint8(np.clip(PURPLE*shade[:,:,None],0,255))
    pixels=np.array(base,dtype=float)
    pixels[visible]=(1-SHELL_OPACITY)*pixels[visible]+SHELL_OPACITY*paint[visible]
    # Only the outer silhouette, not the triangle grid or object-occlusion edges.
    rim=(mask&~binary_erosion(mask,iterations=max(1,round(size/700))))&visible
    pixels[rim]=.45*pixels[rim]+.55*OUTLINE
    return Image.fromarray(np.uint8(np.clip(pixels,0,255)))


def contact_sheet(pictures,tile=720,gap=24,margin=20):
    sheet=Image.new('RGB',(3*tile+2*gap+2*margin,2*tile+gap+2*margin),R.PAPER)
    for index,picture in enumerate(pictures):
        sheet.paste(picture.resize((tile,tile),Image.Resampling.LANCZOS),
                    (margin+(index%3)*(tile+gap),margin+(index//3)*(tile+gap)))
    return sheet


def draw(domain,out,size=1400):
    out=Path(out);source=out.parent/W.STAGE;work=W.WorkVolume.read(source/'work_volume.json')
    print('  Drawing one smooth work-volume shell',flush=True)
    mesh,cap=shell_mesh(work)
    mesh.export(out/'work_volume_preview.stl',file_type='stl_ascii')
    # The new view has one envelope, without a second overlaid interior mesh.
    (out/'work_volume_visible_preview.stl').unlink(missing_ok=True)
    floor=R.floor_triangles(domain)
    points=np.vstack([domain.mesh.vertices,mesh.vertices,floor.reshape(-1,3)])
    # One world-space scale across all six cameras makes their sizes comparable.
    width=1.10*max(float(np.ptp(points@basis.T,axis=0)[:2].max()) for _,basis in VIEWS)
    colors=np.tile(R.GREY,(len(domain.mesh.faces),1));colors[domain.work_ids]=R.GREEN
    triangles=mesh.triangles
    view_dir=out/'work_volume_views';view_dir.mkdir(exist_ok=True)
    files=['work_volume.png','work_volume_only.png','work_volume_preview.stl']
    cameras=[];object_views=[];shell_views=[]
    for name,basis in VIEWS:
        pixels=size if name=='oblique' else min(size,1100)
        local=points@basis.T;lo,hi=local.min(axis=0),local.max(axis=0)
        focus=((lo+hi)/2)@basis
        opaque,depth=layer(np.concatenate([floor,domain.mesh.triangles]),
            np.concatenate([np.tile(R.FLOOR,(len(floor),1)),colors]),focus,width,pixels,unlit=range(len(floor)),basis=basis)
        empty,empty_depth=layer(floor,np.tile(R.FLOOR,(len(floor),1)),focus,width,pixels,unlit=range(len(floor)),basis=basis)
        _,shell_depth=layer(triangles,np.tile(PURPLE,(len(triangles),1)),focus,width,pixels,unlit=range(len(triangles)),basis=basis)
        obj=composite_shell(opaque,depth,shell_depth,focus,width,pixels,cap,basis)
        shell=composite_shell(empty,empty_depth,shell_depth,focus,width,pixels,cap,basis)
        obj_name=f'work_volume_views/{name}.png';shell_name=f'work_volume_views/{name}_only.png'
        obj.save(out/obj_name);shell.save(out/shell_name)
        if name=='oblique':
            obj.save(out/'work_volume.png');shell.save(out/'work_volume_only.png')
        object_views.append(obj);shell_views.append(shell);files.extend([obj_name,shell_name])
        cameras.append(dict(name=name,basis=basis.tolist(),focus_m=focus.tolist(),width_m=width,size_px=pixels,
                            object_image=obj_name,shell_only_image=shell_name))
    for filename,pictures in [('work_volume_multiview.png',object_views),('work_volume_only_multiview.png',shell_views)]:
        contact_sheet(pictures).save(out/filename);files.append(filename)
    W.I.save(out/'work_volume_views.json',dict(complete=True,text_in_figures=False,
        object_occludes_volume_behind_it=True,object_opacity=1.,shared_spherical_cap=cap,
        basis=BASIS.tolist(),focus_m=cameras[0]['focus_m'],width_m=width,cameras=cameras,
        sheet_order=[name for name,_ in VIEWS],sheet_layout='2 rows by 3 columns, row-major, no labels or grid',
        same_world_space_scale_across_views=True,
        color=PURPLE.tolist(),outline_color=OUTLINE.tolist(),shell_opacity=SHELL_OPACITY,
        individual_cell_caps_drawn=False,internal_edges_drawn=False,transition_cells_drawn_separately=False,
        preview_is_acceptance_geometry=False,depends_on_support_or_trajectory=False,
        preview_mesh_representation='Smooth convex exterior illustration of all saved families, truncated by one common sphere; angular resampling and convexification are display approximations, not a collision bound',
        provenance=dict(inputs=W.I.hashes([source/'work_volume.json',source/'work_volume.npz']),
                        code=W.I.hashes([Path(__file__),Path(R.__file__)])),
        artifacts={name:W.I.sha256(out/name) for name in files}))
    return files+['work_volume_views.json']
