"""Eight-second simultaneous camera orbit and literal base growth, without text."""
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
from PIL import Image,ImageDraw

C=D.C
FPS,SECONDS,MAIN,SMALL=24,8,900,340
HEIGHT,RADIUS=.005,.003
ALPHA,OUTLINE_ALPHA=.17,.42
DEST=PROOF/'videos/01_hulls_and_support.mp4'
EXPECTED={'01_hulls_and_support.mp4','02_ground_collision.mp4',
          '03_lift_and_layout.mp4','04_work_cones.mp4'}


def smooth(value):
    value=float(np.clip(value,0,1))
    return value*value*(3-2*value)


def basis(angle):
    angle+=np.deg2rad(75.)
    c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    view=rotation@(np.array([1.,-1.,1.])/np.sqrt(3))
    right=rotation@(np.array([1.,1.,0.])/np.sqrt(2))
    return np.array([right,np.cross(view,right),view])


def slab(hull,normal,height):
    n=len(hull)
    faces=[[0,i+1,i] for i in range(1,n-1)]
    faces.extend([[n,n+i,n+i+1] for i in range(1,n-1)])
    for i in range(n):
        j=(i+1)%n
        faces.extend([[i,j,n+j],[i,n+j,n+i]])
    result=trimesh.Trimesh(np.vstack([hull,hull+normal*height]),np.asarray(faces),process=False)
    if result.volume<0:result.invert()
    assert result.is_watertight
    return result


def rod(start,end):
    direction=end-start
    length=float(np.linalg.norm(direction))
    transform=trimesh.geometry.align_vectors([0,0,1],direction/length)
    transform[:3,3]=(start+end)/2
    return trimesh.creation.cylinder(radius=RADIUS,height=length,sections=20,transform=transform)


def placed(piece,transform):
    return C.V.CPU.placed(piece,rotation=transform[:3,:3],point=transform[:3,3])


def outline(vertices,color,camera,size):
    result=C.H.outline_piece(vertices,color,camera,size,half_width_pixels=.9)
    result['colors']=np.broadcast_to(C.V.CPU.rgb(color),result['triangles'].shape)
    result['opacity']=OUTLINE_ALPHA
    return C.V.CPU.placed(result)


def prepare():
    inp=C.Inputs('B',D.BASE/'pose2+3+4+7/data/selection.json');inp.base=D.BASE
    data=inp.load(inp.cases[0]);rows=data['rows'];support=data['fixture']
    assert [r['pose'] for r in rows]==['pose_2','pose_3','pose_4','pose_7']
    assert {p.name for p in DEST.parent.iterdir()}==EXPECTED
    hashes={p.name:C.digest(p) for p in DEST.parent.iterdir()}
    sources=C.freeze(TEMP/'execution')
    shutil.copy2(__file__,TEMP/'execution/render.py');own_hash=C.digest(__file__)
    old=G.solid(support)
    assert len(old.decompose())==1
    hulls=[];normals=[];plates=[];endpoints=[];links=[]
    for row in rows:
        inverse=np.linalg.inv(row['fixture_transform'])
        hull=C.F.transform_points(row['hull'],inverse)
        normal=row['fixture_transform'][2,:3].copy()
        assert np.ptp(hull@normal)<1e-12
        assert np.max(np.abs(C.F.transform_points(hull,row['fixture_transform'])-row['hull']))<1e-12
        plate=slab(hull,normal,HEIGHT)
        center=D.Polygon(row['hull'][:,:2]).centroid
        target=C.F.transform_points(np.array([[center.x,center.y,HEIGHT/2]]),inverse)[0]
        closest,distance,_=trimesh.proximity.closest_point_naive(support,target[None])
        start=closest[0]-(target-closest[0])/distance[0]*.002
        link=rod(start,target)
        assert G.volume(G.solid(link)^old)>1e-14
        assert G.volume(G.solid(link)^G.solid(plate))>1e-14
        hulls.append(hull);normals.append(normal);plates.append(plate)
        endpoints.append((start,target));links.append(link)
    combined=G.combine([old,*[G.solid(p) for p in plates],*[G.solid(p) for p in links]])
    assert len(combined.decompose())==1
    final=G.unpack(combined)
    assert final.is_watertight and len(final.split(only_watertight=False))==1
    ref=rows[0]['fixture_transform']
    points=C.F.transform_points(np.vstack([support.vertices,*hulls]),ref)
    focus=(points.min(0)+points.max(0))/2
    all_points=C.F.transform_points(np.vstack([final.vertices,*hulls]),ref)
    span=2*max(float(np.abs((all_points-focus)@basis(a)[:2].T).max())
               for a in np.linspace(0,2*np.pi,361))*1.12
    assert np.allclose(basis(0),basis(2*np.pi),atol=1e-12)
    bodies=[];cameras=[]
    for row in rows:
        body=C.posed(data['body'],row['object_transform'])
        bodies.append(C.V.CPU.placed(C.V.piece(body,C.V.GRAY,.18)))
        camera=C.V.camera_for(np.vstack([C.F.transform_points(np.vstack([final.vertices,*hulls]),row['fixture_transform']),body.vertices]))
        cameras.append(camera)
    right_span=max(cam[2] for cam in cameras)
    cameras=[(f,b,right_span) for f,b,_ in cameras]
    renderer=C.V.Renderer()
    support_left=C.V.piece(support,C.V.BLUE,.78)
    support_right=C.V.piece(support,C.V.BLUE,.82)
    final_right=None

    def frame(t):
        nonlocal final_right
        angle=2*np.pi*min(t/7.75,1.)
        camera=(focus,basis(angle),span)
        height=HEIGHT*smooth(t/1.5)
        material=[]
        for hull,normal,row in zip(hulls,normals,rows):
            if height>1e-8:
                piece=C.V.piece(slab(hull,normal,height),row['color'],ALPHA)
            else:
                faces=np.array([[0,i,i+1] for i in range(1,len(hull)-1)])
                piece=C.V.piece(trimesh.Trimesh(hull,faces,process=False),row['color'],ALPHA)
                piece['unlit']=True
            material.append(piece)
        for i,(start,end) in enumerate(endpoints):
            fraction=smooth(t-1.5-i)
            if fraction>1e-5:
                material.append(C.V.piece(rod(start,start+(end-start)*fraction),C.V.BLUE,.82))

        def parts(transform,camera,size,support_piece):
            result=[placed(support_piece,transform),*[placed(p,transform) for p in material]]
            for hull,normal,row in zip(hulls,normals,rows):
                result.append(outline(C.F.transform_points(hull,transform),row['color'],camera,size))
                if height>1e-8:
                    result.append(outline(C.F.transform_points(hull+normal*height,transform),row['color'],camera,size))
            return result

        main=renderer.render(parts(ref,camera,MAIN,support_left),camera,MAIN)
        if t>=5.5 and final_right is not None:
            panel=final_right
        else:
            panel=Image.new('RGB',(700,900),'white')
            for i,(row,body,pose_camera) in enumerate(zip(rows,bodies,cameras)):
                pieces=parts(row['fixture_transform'],pose_camera,SMALL,support_right)
                pieces.append(body)
                tile=renderer.render(pieces,pose_camera,SMALL)
                panel.paste(tile,(7+(i%2)*346,55+(i//2)*450))
            if t>=5.5:final_right=panel
        image=Image.new('RGB',(1600,900),'white')
        image.paste(main,(0,0));image.paste(panel,(900,0))
        draw=ImageDraw.Draw(image)
        draw.line((900,0,900,899),fill='#eef0f2',width=1)
        draw.line((901,450,1599,450),fill='#eef0f2',width=1)
        return image

    def verify():
        inp.verify()
        for path,expected in sources.items():
            assert C.digest(ROOT/path)==C.digest(TEMP/'execution/executed_sources'/path)==expected
        assert C.digest(__file__)==C.digest(TEMP/'execution/render.py')==own_hash
        assert {p.name:C.digest(p) for p in DEST.parent.iterdir()}==hashes
        print('All saved input, source and original video hashes match.',flush=True)

    C.save(TEMP/'original_videos.json',hashes)
    C.save(TEMP/'metadata.json',dict(duration_seconds=8,frames=192,poses=[r['pose'] for r in rows],
        hulls_visible_at_start=4,hull_opacity=ALPHA,plate_thickness_mm=5,
        growth_starts_at=0,rods_grow_between_seconds=[1.5,5.5],orbit_degrees=360,
        rotation_during_growth=True,connected_components=1,text=False,grid=False))
    print('Prepared 8-second scene: all four hulls visible from frame 0; simultaneous orbit and growth.',flush=True)
    return frame,verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--encode',action='store_true');args=parser.parse_args()
    frame,verify=prepare()
    if not args.encode:
        for t,name in [(0,'start'),(1,'thickening'),(3,'growing'),(7.75,'finished')]:
            frame(t).save(TEMP/(name+'.png'))
        verify();return
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r',str(FPS),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','18','-profile:v','main',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(TEMP/'video.mp4')],stdin=subprocess.PIPE)
    start=time.monotonic()
    try:
        for i in range(FPS*SECONDS):
            encoder.stdin.write(frame(i/FPS).tobytes())
            if i%24==0:print(json.dumps(dict(frame=i,total=192,elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
        encoder.stdin.close();assert encoder.wait()==0
    except BaseException:
        encoder.kill();encoder.wait();raise
    verify();print('Encoding complete.',flush=True)


if __name__=='__main__':main()
