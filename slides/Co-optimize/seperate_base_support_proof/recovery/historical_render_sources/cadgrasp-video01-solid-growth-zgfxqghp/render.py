"""Append literal hull plates and support connectors to the approved video scene."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import time

ROOT=Path('/home/yli581/Desktop/cadGrasp')
PROOF=ROOT/'slides/Co-optimize/seperate_base_support_proof'
TEMP=Path(__file__).resolve().parent
sys.path.insert(0,str(PROOF))
import diagnose as D
import base_geometry as G
import numpy as np
import trimesh
from PIL import Image, ImageDraw

C=D.C
FPS,SECONDS,MAIN,SMALL=24,32,900,340
THICKNESS,RADIUS=.005,.003
HULL_ALPHA,EDGE_ALPHA=.17,.42
DEST=PROOF/'videos/01_hulls_and_support.mp4'
EXPECTED={'01_hulls_and_support.mp4','02_ground_collision.mp4',
          '03_lift_and_layout.mp4','04_work_cones.mp4'}


def smooth(t):
    t=float(np.clip(t,0,1))
    return t*t*(3-2*t)


def basis_for(angle):
    angle+=np.deg2rad(75.)
    c,s=np.cos(angle),np.sin(angle)
    rot=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    view=rot@(np.array([1.,-1.,1.])/np.sqrt(3))
    right=rot@(np.array([1.,1.,0.])/np.sqrt(2))
    return np.array([right,np.cross(view,right),view])


def disk(vertices,color):
    faces=np.array([[0,i,i+1] for i in range(1,len(vertices)-1)])
    result=C.V.piece(trimesh.Trimesh(vertices,faces,process=False),color,HULL_ALPHA)
    result['unlit']=True
    return result


def edge(vertices,color,camera,size):
    result=C.H.outline_piece(vertices,color,camera,size,half_width_pixels=.9)
    result['colors']=np.broadcast_to(C.V.CPU.rgb(color),result['triangles'].shape)
    result['opacity']=EDGE_ALPHA
    return result


def plate(hull,normal,height):
    n=len(hull)
    faces=[[0,i+1,i] for i in range(1,n-1)]
    faces.extend([[n,n+i,n+i+1] for i in range(1,n-1)])
    for i in range(n):
        j=(i+1)%n
        faces.extend([[i,j,n+j],[i,n+j,n+i]])
    mesh=trimesh.Trimesh(np.vstack([hull,hull+normal*height]),np.asarray(faces),process=False)
    if mesh.volume<0:mesh.invert()
    assert mesh.is_watertight
    return mesh


def rod(start,end):
    direction=end-start
    length=float(np.linalg.norm(direction))
    assert length>1e-10
    transform=trimesh.geometry.align_vectors([0,0,1],direction/length)
    transform[:3,3]=(start+end)/2
    return trimesh.creation.cylinder(radius=RADIUS,height=length,sections=20,transform=transform)


def prepare():
    inp=C.Inputs('B',D.BASE/'pose2+3+4+7/data/selection.json');inp.base=D.BASE
    data=inp.load(inp.cases[0]);rows=data['rows']
    assert [row['pose'] for row in rows]==['pose_2','pose_3','pose_4','pose_7']
    support=data['fixture']; old=G.solid(support)
    assert len(old.decompose())==len(support.split(only_watertight=False))==1
    assert {p.name for p in DEST.parent.iterdir()}==EXPECTED
    video_hashes={p.name:C.digest(p) for p in DEST.parent.iterdir()}
    sources=C.freeze(TEMP/'execution')
    shutil.copy2(__file__,TEMP/'execution/render.py');own_hash=C.digest(__file__)
    ref=rows[0]['fixture_transform']
    local_hulls=[];normals=[];plates=[];connectors=[];endpoints=[];checks=[]
    for row in rows:
        inverse=np.linalg.inv(row['fixture_transform'])
        hull=C.F.transform_points(row['hull'],inverse)
        normal=row['fixture_transform'][2,:3].copy()
        assert np.ptp(hull@normal)<1e-12
        assert np.max(np.abs(C.F.transform_points(hull,row['fixture_transform'])-row['hull']))<1e-12
        slab=plate(hull,normal,THICKNESS)
        value=G.solid(slab)
        center=D.Polygon(row['hull'][:,:2]).centroid
        target=C.F.transform_points(np.array([[center.x,center.y,THICKNESS/2]]),inverse)[0]
        closest,distance,face=trimesh.proximity.closest_point_naive(support,target[None])
        direction=(target-closest[0])/distance[0]
        root=closest[0]-direction*.002
        link=rod(root,target)
        link_value=G.solid(link)
        root_overlap=G.volume(link_value^old)
        plate_overlap=G.volume(link_value^value)
        assert root_overlap>1e-14 and plate_overlap>1e-14
        local_hulls.append(hull);normals.append(normal);plates.append(slab)
        connectors.append(link);endpoints.append((root,target))
        checks.append(dict(pose=row['pose'],gap_mm=float(distance[0])*1000,
                           rod_support_overlap_mm3=root_overlap*1e9,
                           rod_plate_overlap_mm3=plate_overlap*1e9,
                           plate_volume_cm3=G.volume(value)*1e6))
    combined=G.combine([old,*[G.solid(p) for p in plates],*[G.solid(p) for p in connectors]])
    assert len(combined.decompose())==1
    final=G.unpack(combined)
    assert final.is_watertight and len(final.split(only_watertight=False))==1
    final.export(TEMP/'constructed_support.obj')
    fixture=C.posed(support,ref)
    hulls=[C.F.transform_points(h,ref) for h in local_hulls]
    shown_normals=[ref[:3,:3]@n for n in normals]
    points=np.vstack([fixture.vertices,*hulls])
    focus=(points.min(0)+points.max(0))/2
    angles=np.linspace(0,2*np.pi,361)
    initial_span=2*max(float(np.abs((points-focus)@basis_for(a)[:2].T).max()) for a in angles)*1.12
    final_points=C.F.transform_points(final.vertices,ref)
    final_span=max(initial_span,2*max(float(np.abs((final_points-focus)@basis_for(a)[:2].T).max()) for a in angles)*1.12)
    assert np.allclose(basis_for(0),basis_for(2*np.pi),atol=1e-12)
    pose_geometries=[];initial_cameras=[];final_cameras=[]
    for row in rows:
        shown_support=C.posed(support,row['fixture_transform'])
        body=C.posed(data['body'],row['object_transform'])
        initial_camera=C.V.camera_for(np.vstack([shown_support.vertices,body.vertices,row['hull']]))
        final_camera=C.V.camera_for(np.vstack([C.F.transform_points(final.vertices,row['fixture_transform']),body.vertices]))
        pose_geometries.append((shown_support,body))
        initial_cameras.append(initial_camera);final_cameras.append(final_camera)
    initial_right_span=max(camera[2] for camera in initial_cameras)
    final_right_span=max(initial_right_span,max(camera[2] for camera in final_cameras))
    initial_cameras=[(f,b,initial_right_span) for f,b,_ in initial_cameras]
    final_cameras=[(f,b,final_right_span) for f,b,_ in final_cameras]
    renderer=C.V.Renderer()
    C.save(TEMP/'metadata.json',dict(case=data['case']['id'],poses=[r['pose'] for r in rows],
        old_support_components=1,constructed_support_components=1,watertight=True,
        thickness_mm=THICKNESS*1000,connector_diameter_mm=RADIUS*2000,connectors=checks,
        plane_opacity=HULL_ALPHA,outline_opacity=EDGE_ALPHA,duration_seconds=SECONDS,
        original_orbit_degrees=360,finished_orbit_degrees=360,text=False,grid=False,
        construction_scope='literal geometric demonstration; no floor, work, exit or load acceptance'))
    print(json.dumps(dict(case=data['case']['id'],components=1,plate_thickness_mm=5,
                         connector_gaps_mm=[round(x['gap_mm'],3) for x in checks],
                         hull_opacity=HULL_ALPHA,seconds=SECONDS)),flush=True)

    def stage(t):
        height=THICKNESS*smooth((t-16)/2.5)
        lengths=[smooth(t-19-i) for i in range(4)]
        zoom=smooth((t-16)/3)
        return height,lengths,zoom

    def pieces(hull_list,normal_list,support_mesh,camera,size,t,state):
        result=[C.V.piece(support_mesh,C.V.BLUE,.78 if size==MAIN else .82)]
        if t<2:return result
        if t<16:
            selected=range(4) if size==MAIN else [state]
            for i in selected:
                result.extend([disk(hull_list[i],rows[i]['color']),edge(hull_list[i],rows[i]['color'],camera,size)])
            return result
        height,lengths,_=stage(t)
        for i,(hull,normal) in enumerate(zip(hull_list,normal_list)):
            if height<1e-8:
                if size==MAIN or i==state:
                    result.extend([disk(hull,rows[i]['color']),edge(hull,rows[i]['color'],camera,size)])
            else:
                slab=plate(hull,normal,height)
                result.extend([C.V.piece(slab,rows[i]['color'],HULL_ALPHA),
                               edge(hull,rows[i]['color'],camera,size),
                               edge(hull+normal*height,rows[i]['color'],camera,size)])
        transform=ref if size==MAIN else rows[state]['fixture_transform']
        for (start,end),amount in zip(endpoints,lengths):
            if amount>1e-5:
                link=rod(start,start+(end-start)*amount)
                result.append(C.V.piece(C.posed(link,transform),C.V.BLUE,.82))
        return result

    right_cache={}
    def right_panel(t):
        cache_key=0 if t<2 else 1 if t<16 else 2 if t>=23 else None
        if cache_key in right_cache:return right_cache[cache_key]
        panel=Image.new('RGB',(700,900),'white')
        _,_,zoom=stage(t)
        for i,(row,(shown_support,body),before,after) in enumerate(zip(rows,pose_geometries,initial_cameras,final_cameras)):
            camera=((1-zoom)*before[0]+zoom*after[0],before[1],(1-zoom)*before[2]+zoom*after[2])
            native_hulls=[C.F.transform_points(h,row['fixture_transform']) for h in local_hulls]
            native_normals=[row['fixture_transform'][:3,:3]@n for n in normals]
            parts=pieces(native_hulls,native_normals,shown_support,camera,SMALL,t,i)
            parts.append(C.V.piece(body,C.V.GRAY,.18))
            tile=renderer.render([C.V.CPU.placed(p) for p in parts],camera,SMALL)
            panel.paste(tile,(7+(i%2)*346,55+(i//2)*450))
        if cache_key is not None:right_cache[cache_key]=panel
        return panel

    def frame(t):
        if 3<=t<15:angle=(t-3)/12*2*np.pi
        elif 23<=t<31:angle=(t-23)/8*2*np.pi
        else:angle=0
        _,_,zoom=stage(t)
        camera=(focus,basis_for(angle),(1-zoom)*initial_span+zoom*final_span)
        parts=pieces(hulls,shown_normals,fixture,camera,MAIN,t,None)
        main=renderer.render([C.V.CPU.placed(p) for p in parts],camera,MAIN)
        image=Image.new('RGB',(1600,900),'white');image.paste(main,(0,0));image.paste(right_panel(t),(900,0))
        draw=ImageDraw.Draw(image)
        draw.line((900,0,900,899),fill='#eef0f2',width=1)
        draw.line((901,450,1599,450),fill='#eef0f2',width=1)
        return image

    def verify():
        inp.verify()
        for path,expected in sources.items():
            assert C.digest(ROOT/path)==C.digest(TEMP/'execution/executed_sources'/path)==expected
        assert C.digest(__file__)==C.digest(TEMP/'execution/render.py')==own_hash
        assert {p.name:C.digest(p) for p in DEST.parent.iterdir()}==video_hashes
        print('Saved input, executed source and original video hashes all match.',flush=True)
    C.save(TEMP/'original_videos.json',video_hashes)
    return frame,verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--encode',action='store_true');args=parser.parse_args()
    frame,verify=prepare()
    if not args.encode:
        for name,t in [('thin',2.),('thickening',17.25),('plates',19.),('growing',21.5),('connected',23.),('connected_half',27.)]:
            frame(t).save(TEMP/(name+'.png'))
        verify();return
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r',str(FPS),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','18','-profile:v','main',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(TEMP/'video.mp4')],stdin=subprocess.PIPE)
    start=time.monotonic();cached={}
    try:
        for i in range(FPS*SECONDS):
            t=i/FPS
            key='empty' if t<2 else 'thin' if 2<=t<3 or 15<=t<16 else 'finished' if t>=31 else None
            if key in cached:rgb=cached[key]
            else:
                rgb=frame(t).tobytes()
                if key is not None:cached[key]=rgb
            encoder.stdin.write(rgb)
            if i%(FPS*2)==0:print(json.dumps(dict(frame=i,total=FPS*SECONDS,elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
        encoder.stdin.close();assert encoder.wait()==0
    except BaseException:
        encoder.kill();encoder.wait();raise
    verify();print('Finished encoding.',flush=True)


if __name__=='__main__':main()
