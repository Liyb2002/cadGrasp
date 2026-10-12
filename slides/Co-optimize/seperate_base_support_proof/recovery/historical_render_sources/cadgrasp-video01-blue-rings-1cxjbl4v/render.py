"""Blue hollow polygonal rims growing around the four existing demand hulls."""
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
from shapely.geometry import Polygon
from PIL import Image,ImageDraw

C=D.C
FPS,SECONDS,MAIN,SMALL=24,8,900,340
WIDTH,HEIGHT,RADIUS=.004,.005,.003
HULL_ALPHA,EDGE_ALPHA=.17,.42
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


def prism(xy):
    n=len(xy)
    vertices=np.c_[xy,np.zeros(n)]
    faces=[[0,i+1,i] for i in range(1,n-1)]
    faces.extend([[n,n+i,n+i+1] for i in range(1,n-1)])
    for i in range(n):
        j=(i+1)%n
        faces.extend([[i,j,n+j],[i,n+j,n+i]])
    mesh=trimesh.Trimesh(np.vstack([vertices,vertices+[0,0,HEIGHT]]),np.asarray(faces),process=False)
    if mesh.volume<0:mesh.invert()
    assert mesh.is_watertight
    return G.solid(mesh)


def rod(start,end):
    direction=end-start;length=float(np.linalg.norm(direction))
    transform=trimesh.geometry.align_vectors([0,0,1],direction/length)
    transform[:3,3]=(start+end)/2
    return trimesh.creation.cylinder(radius=RADIUS,height=length,sections=20,transform=transform)


def placed(piece,transform):
    return C.V.CPU.placed(piece,rotation=transform[:3,:3],point=transform[:3,3])


def outline(vertices,color,camera,size):
    result=C.H.outline_piece(vertices,color,camera,size,half_width_pixels=.9)
    result['colors']=np.broadcast_to(C.V.CPU.rgb(color),result['triangles'].shape)
    result['opacity']=EDGE_ALPHA
    return C.V.CPU.placed(result)


class Ring:
    def __init__(self,row,support):
        self.inverse=np.linalg.inv(row['fixture_transform'])
        xy=row['hull'][:,:2]
        direction=np.roll(xy,-1,axis=0)-xy
        outward=np.c_[direction[:,1],-direction[:,0]]/np.linalg.norm(direction,axis=1)[:,None]
        outer=np.array([np.linalg.solve(np.array([outward[i-1],outward[i]]),
                        np.array([outward[i-1]@xy[i],outward[i]@xy[i]])+WIDTH)
                        for i in range(len(xy))])
        inner_polygon,outer_polygon=Polygon(xy),Polygon(outer)
        assert inner_polygon.is_valid and outer_polygon.is_valid
        assert outer_polygon.covers(inner_polygon)
        band=outer_polygon.difference(inner_polygon)
        assert len(band.interiors)==1 and band.intersection(inner_polygon).area<1e-16
        middle=(xy+outer)/2
        candidates=[];indices=[]
        for i in range(len(xy)):
            for f in (0.,.25,.5,.75):
                candidates.append([*(middle[i]*(1-f)+middle[(i+1)%len(xy)]*f),HEIGHT/2])
                indices.append((i,f))
        common=C.F.transform_points(np.asarray(candidates),self.inverse)
        roots,distance,_=trimesh.proximity.closest_point_naive(support,common)
        chosen=int(np.argmin(distance));i,f=indices[chosen]
        self.end=common[chosen]
        self.start=roots[chosen]-(self.end-roots[chosen])/distance[chosen]*.002
        self.gap_mm=float(distance[chosen])*1000
        if f==0:
            self.inner=np.roll(xy,-i,axis=0)
            self.outer=np.roll(outer,-i,axis=0)
        else:
            order=[(i+1+j)%len(xy) for j in range(len(xy))]
            self.inner=np.vstack([xy[i]*(1-f)+xy[(i+1)%len(xy)]*f,xy[order]])
            self.outer=np.vstack([outer[i]*(1-f)+outer[(i+1)%len(xy)]*f,outer[order]])
        centers=(self.inner+self.outer)/2
        lengths=np.linalg.norm(np.roll(centers,-1,axis=0)-centers,axis=1)
        self.cumulative=np.r_[0,np.cumsum(lengths)]
        self.prefix=[G.md.Manifold()]
        for i in range(len(self.inner)):
            j=(i+1)%len(self.inner)
            cell=prism(np.array([self.inner[i],self.inner[j],self.outer[j],self.outer[i]]))
            self.prefix.append(self.prefix[-1]+cell)
        native=self.prefix[-1]
        assert len(native.decompose())==1
        assert np.isclose(G.volume(native),band.area*HEIGHT,rtol=1e-8)
        # A true hole: the new rim has zero material in the hull's interior.
        inner_solid=prism(xy)
        assert G.volume(native^inner_solid)<1e-14
        self.solid=G.transform(native,self.inverse)
        self.mesh=C.posed(G.unpack(native),self.inverse)
        self.link=rod(self.start,self.end)
        assert G.volume(G.solid(self.link)^self.solid)>1e-14

    def growing_mesh(self,amount):
        if amount>=1:return self.mesh
        distance=amount*self.cumulative[-1]
        i=min(int(np.searchsorted(self.cumulative,distance,side='right')-1),len(self.inner)-1)
        f=(distance-self.cumulative[i])/(self.cumulative[i+1]-self.cumulative[i])
        j=(i+1)%len(self.inner)
        part=self.prefix[i]
        if f>1e-8:
            xy=np.array([self.inner[i],(1-f)*self.inner[i]+f*self.inner[j],
                         (1-f)*self.outer[i]+f*self.outer[j],self.outer[i]])
            part=part+prism(xy)
        return C.posed(G.unpack(part),self.inverse)


def prepare():
    inp=C.Inputs('B',D.BASE/'pose2+3+4+7/data/selection.json');inp.base=D.BASE
    data=inp.load(inp.cases[0]);rows=data['rows'];support=data['fixture']
    assert [r['pose'] for r in rows]==['pose_2','pose_3','pose_4','pose_7']
    assert {p.name for p in DEST.parent.iterdir()}==EXPECTED
    hashes={p.name:C.digest(p) for p in DEST.parent.iterdir()}
    sources=C.freeze(TEMP/'execution')
    shutil.copy2(__file__,TEMP/'execution/render.py');own_hash=C.digest(__file__)
    old=G.solid(support);assert len(old.decompose())==1
    rings=[Ring(row,support) for row in rows]
    for ring in rings:assert G.volume(G.solid(ring.link)^old)>1e-14
    combined=G.combine([old,*[ring.solid for ring in rings],*[G.solid(ring.link) for ring in rings]])
    assert len(combined.decompose())==1
    final=G.unpack(combined)
    assert final.is_watertight and len(final.split(only_watertight=False))==1
    ref=rows[0]['fixture_transform']
    hulls=[C.F.transform_points(row['hull'],np.linalg.inv(row['fixture_transform'])) for row in rows]
    for hull,row in zip(hulls,rows):
        assert np.max(np.abs(C.F.transform_points(hull,row['fixture_transform'])-row['hull']))<1e-12
    points=C.F.transform_points(np.vstack([support.vertices,*hulls]),ref)
    focus=(points.min(0)+points.max(0))/2
    all_points=C.F.transform_points(np.vstack([final.vertices,*hulls]),ref)
    span=2*max(float(np.abs((all_points-focus)@basis(a)[:2].T).max()) for a in np.linspace(0,2*np.pi,361))*1.12
    assert np.allclose(basis(0),basis(2*np.pi),atol=1e-12)
    bodies=[];cameras=[]
    for row in rows:
        body=C.posed(data['body'],row['object_transform'])
        bodies.append(C.V.CPU.placed(C.V.piece(body,C.V.GRAY,.18)))
        cameras.append(C.V.camera_for(np.vstack([C.F.transform_points(np.vstack([final.vertices,*hulls]),row['fixture_transform']),body.vertices])))
    right_span=max(camera[2] for camera in cameras)
    cameras=[(f,b,right_span) for f,b,_ in cameras]
    renderer=C.V.Renderer()
    support_left=C.V.piece(support,C.V.BLUE,.78)
    support_right=C.V.piece(support,C.V.BLUE,.82)
    hull_pieces=[]
    for hull,row in zip(hulls,rows):
        faces=np.array([[0,i,i+1] for i in range(1,len(hull)-1)])
        value=C.V.piece(trimesh.Trimesh(hull,faces,process=False),row['color'],HULL_ALPHA)
        value['unlit']=True;hull_pieces.append(value)
    final_right=None

    def frame(t):
        nonlocal final_right
        camera=(focus,basis(2*np.pi*min(t/7.75,1.)),span)
        link_amount=smooth(t/1.2)
        ring_amount=smooth((t-1.2)/4.3)
        material=[]
        for ring in rings:
            if link_amount>1e-5:
                material.append(C.V.piece(rod(ring.start,ring.start+(ring.end-ring.start)*link_amount),C.V.BLUE,.82))
            if ring_amount>1e-5:
                material.append(C.V.piece(ring.growing_mesh(ring_amount),C.V.BLUE,.82))

        def parts(transform,camera,size,support_piece):
            result=[placed(support_piece,transform),*[placed(piece,transform) for piece in hull_pieces],
                    *[placed(piece,transform) for piece in material]]
            for hull,row in zip(hulls,rows):
                result.append(outline(C.F.transform_points(hull,transform),row['color'],camera,size))
            return result

        main=renderer.render(parts(ref,camera,MAIN,support_left),camera,MAIN)
        if t>=5.5 and final_right is not None:panel=final_right
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
        print('Saved input, source and original video hashes match.',flush=True)
    C.save(TEMP/'original_videos.json',hashes)
    C.save(TEMP/'metadata.json',dict(duration_seconds=8,frames=192,poses=[r['pose'] for r in rows],
        new_material_color=C.V.BLUE,hull_opacity=HULL_ALPHA,ring_width_mm=4,ring_height_mm=5,
        ring_interior_empty=True,ring_hull_interior_overlap_m3=0,connected_components=1,
        connector_gaps_mm=[ring.gap_mm for ring in rings],hulls_visible_from_first_frame=True,
        orbit_degrees=360,rotation_during_growth=True,text=False,grid=False))
    print(json.dumps(dict(rings=4,blue=C.V.BLUE,hollow=True,components=1,
                         connector_gaps_mm=[round(r.gap_mm,3) for r in rings])),flush=True)
    return frame,verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--encode',action='store_true');args=parser.parse_args()
    frame,verify=prepare()
    if not args.encode:
        for t,name in [(0,'start'),(3,'growing'),(5.5,'closed'),(7.75,'finished')]:frame(t).save(TEMP/(name+'.png'))
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
