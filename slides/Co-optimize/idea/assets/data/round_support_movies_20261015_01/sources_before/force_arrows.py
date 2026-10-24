"""Stable area-distributed illustrations of saved, individually legal loads."""
import numpy as np
from inputs import sha

ARROW_COUNT=18
LENGTH_M_PER_MG=.14


def project_points(scene,points,width,height):
    a,b=scene.camera
    eye=(a.pos.astype(float)+b.pos.astype(float))/2
    forward=a.forward.astype(float);up=a.up.astype(float);right=np.cross(forward,up)
    vectors=points-eye;depth=vectors@forward
    if np.min(depth)<=0:raise ValueError('Force arrow behind camera')
    focal=height*a.frustum_near/(a.frustum_top-a.frustum_bottom)
    pixels=np.column_stack([width/2+focal*(vectors@right)/depth,height/2-focal*(vectors@up)/depth])
    return pixels


def select(row,body,fixture,scene,translation,width,height):
    with np.load(row['load_samples']) as arrays:
        face_ids=arrays['work_face_index'].copy();parameters=arrays['parameters'].copy()
        points=arrays['pt_m'].copy();forces=arrays['force_push_mg'].copy()
        half_angle=float(arrays['cone_half_deg'])
    native=row['native_pose'];rotation=native[:3,:3]
    # The saved pairs are in the native setup's world frame, not mesh coordinates.
    points=(points-native[:3,3])@rotation;forces=forces@rotation
    magnitudes=np.linalg.norm(forces,axis=1)
    normals=body.face_normals[row['work_ids']][face_ids]
    directions=forces/np.maximum(magnitudes[:,None],1e-30)
    vertices=body.triangles[row['work_ids']][face_ids]
    uv=parameters[:,:2]
    reconstructed=vertices[:,0]+uv[:,0,None]*(vertices[:,1]-vertices[:,0])+uv[:,1,None]*(vertices[:,2]-vertices[:,0])
    surface_error=float(np.max(np.linalg.norm(reconstructed-points,axis=1)))
    cosine=np.einsum('ij,ij->i',directions,-normals)
    assert surface_error<1e-10
    assert np.min(cosine)>=np.cos(np.radians(half_angle))-1e-10
    assert np.max(np.abs(magnitudes-parameters[:,4]))<1e-10
    assert np.max(magnitudes)<=row['force_limit_mg']+1e-10
    pose=row['object'].copy();pose[:3,3]+=translation
    a,b=scene.camera;eye=(a.pos.astype(float)+b.pos.astype(float))/2
    local_eye=(eye-pose[:3,3])@pose[:3,:3]
    rng=np.random.default_rng(20261014+int(row['pose'].split('_')[-1]))
    candidate=rng.permutation(len(points))[:4096]
    facing=np.einsum('ij,ij->i',normals[candidate],local_eye-points[candidate])>0
    candidate=candidate[facing & (magnitudes[candidate]>=.06)]
    origins=points[candidate]+normals[candidate]*1e-5
    rays=local_eye-origins;rays/=np.linalg.norm(rays,axis=1)[:,None]
    visible=~body.ray.intersects_any(origins,rays)
    installed_fixture=fixture.copy();installed_fixture.apply_transform(np.linalg.inv(row['object'])@row['fixture'])
    visible &= ~installed_fixture.ray.intersects_any(origins,rays)
    candidate=candidate[visible]
    lengths=LENGTH_M_PER_MG*magnitudes[candidate]
    world=points[candidate]@pose[:3,:3].T+pose[:3,3]
    tails=(points[candidate]-directions[candidate]*lengths[:,None])@pose[:3,:3].T+pose[:3,3]
    pixels=project_points(scene,world,width,height);tail_pixels=project_points(scene,tails,width,height)
    readable=np.linalg.norm(pixels-tail_pixels,axis=1)>=9
    readable &= np.all((tail_pixels>10)&(tail_pixels<[width-10,height-10]),axis=1)
    candidate=candidate[readable];pixels=pixels[readable]
    if len(candidate)<ARROW_COUNT:
        raise ValueError(f'{row["pose"]}: only {len(candidate)} visible legal arrows')
    # Spread application points over the visible working area, rather than cluster them.
    picked=[int(np.argmin(np.linalg.norm(pixels-pixels.mean(0),axis=1)))]
    distance=np.linalg.norm(pixels-pixels[picked[0]],axis=1)
    for _ in range(ARROW_COUNT-1):
        index=int(np.argmax(distance));picked.append(index)
        distance=np.minimum(distance,np.linalg.norm(pixels-pixels[index],axis=1))
    selected=candidate[picked]
    output=dict(pose=row['pose'],sample_source=str(row['load_samples']),sample_source_sha256=sha(row['load_samples']),
        original_sample_ids=selected.tolist(),points_local_m=points[selected].tolist(),
        inward_directions_local=directions[selected].tolist(),magnitude_mg=magnitudes[selected].tolist(),
        display_length_m=(LENGTH_M_PER_MG*magnitudes[selected]).tolist(),
        work_face_indices=face_ids[selected].tolist(),cone_half_deg=half_angle,force_limit_mg=row['force_limit_mg'],
        angles_to_inward_normal_deg=np.degrees(np.arccos(np.clip(cosine[selected],-1,1))).tolist(),
        point_surface_max_error_m=surface_error,arrow_count=ARROW_COUNT,
        selection='Fixed farthest-point subset of visible saved paired legal loads',
        length_mapping='0.14 m display length per mg of the saved individual force magnitude',
        arrows_are_individual_legal_load_examples=True,simultaneous_total_load_certified=False)
    assert np.ptp(output['display_length_m'])>.01
    return output


def draw(ink,start,end,width):
    scale=width/1600;delta=end-start;length=np.linalg.norm(delta)
    if length<2:return
    color=(242,75,20);shaft=max(1,round(2*scale));halo=max(2,round(4*scale))
    ink.line([tuple(start),tuple(end)],fill='white',width=halo)
    ink.line([tuple(start),tuple(end)],fill=color,width=shaft)
    unit=delta/length;side=np.array([-unit[1],unit[0]])
    base=end-unit*min(8*scale,.42*length)
    ink.polygon([tuple(end),tuple(base+3*scale*side),tuple(base-3*scale*side)],fill=color)
