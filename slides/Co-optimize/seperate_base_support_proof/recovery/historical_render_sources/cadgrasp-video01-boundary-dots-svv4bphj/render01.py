from pathlib import Path
import argparse,json,shutil,subprocess,time
import scene as S
from scene import C,np,Image,placed,compose,basis,smooth,GROW_START,MAIN,SMALL,FPS,SECONDS,TEMP,PROOF,ROOT

def prepare():
    scene=S.Scene();rows=scene.rows;renderer=scene.renderer
    directory=TEMP/'execution01';sources=C.freeze(directory)
    paths=[Path(__file__),Path(S.__file__),TEMP/'transparency.py',TEMP/'transparency.cpp']
    for path in paths:
        shutil.copy2(path,directory/path.name)
    own={p.name:C.digest(p) for p in paths}
    videos=PROOF/'videos';hashes={p.name:C.digest(p) for p in videos.iterdir()};assert set(hashes)==set(S.NAMES)
    C.save(TEMP/'original_videos.json',hashes)
    final_right=None
    def frame(t):
        nonlocal final_right
        camera=(scene.focus,basis(2*np.pi*min(t/7.75,1)),scene.span)
        link_amount=smooth(t-GROW_START);ring_amount=smooth((t-GROW_START-1)/4)
        material=[]
        for ring in scene.rings:
            if link_amount>1e-5:material.append(C.V.piece(S.rod(ring.start,ring.start+(ring.end-ring.start)*link_amount),C.V.BLUE,.82))
            if ring_amount>1e-5:material.append(C.V.piece(ring.mesh_at(ring_amount),C.V.BLUE,.82))
        def parts(T,camera,size):
            values=[placed(C.V.piece(scene.old,C.V.BLUE,.78 if size==MAIN else .82),T),*[placed(p,T) for p in material]]
            for hull,dots,boundary,platform,row in zip(scene.hulls,scene.dots,scene.boundary,scene.platforms,rows):
                if t<GROW_START:
                    values.append(placed(platform,T))
                    border=C.H.outline_piece(C.F.transform_points(hull,T),row['color'],camera,size,half_width_pixels=.8)
                    border['colors']=np.broadcast_to(C.V.CPU.rgb(row['color']),border['triangles'].shape);border['opacity']=.28
                    values.append(C.V.CPU.placed(border))
                scatter=C.V.points_piece(C.F.transform_points(dots,T),row['color'],camera,size,half_size_pixels=1.05 if size==MAIN else .65)
                scatter['opacity']=.8;values.append(C.V.CPU.placed(scatter))
                edge_points=C.V.points_piece(C.F.transform_points(boundary,T),row['color'],camera,size,half_size_pixels=1.4 if size==MAIN else .85)
                edge_points['opacity']=.95;values.append(C.V.CPU.placed(edge_points))
            return values
        main=renderer.render(parts(scene.ref,camera,MAIN),camera,MAIN)
        if t>=5.75 and final_right is not None:panel=final_right
        else:
            panel=Image.new('RGB',(700,900),'white')
            for i,(row,body,cam) in enumerate(zip(rows,scene.body_world,scene.cameras)):
                pieces=parts(row['fixture_transform'],cam,SMALL);pieces.append(C.V.CPU.placed(C.V.piece(body,C.V.GRAY,.18)))
                panel.paste(renderer.render(pieces,cam,SMALL),(7+(i%2)*346,55+(i//2)*450))
            if t>=5.75:final_right=panel
        return compose(main,panel)
    def verify():
        scene.verify()
        for path,h in sources.items():assert C.digest(ROOT/path)==C.digest(directory/'executed_sources'/path)==h
        for name,h in own.items():assert C.digest(TEMP/name)==C.digest(directory/name)==h
        assert {p.name:C.digest(p) for p in videos.iterdir()}==hashes
        print('Video 01: all input and source hashes match.',flush=True)
    C.save(TEMP/'metadata01.json',dict(boundary_points_per_pose=[len(p) for p in scene.boundary],
        boundary_spacing_max_mm=2,interior_original_points_per_pose=[len(p) for p in scene.dots],
        support_geometry_sha256=scene.support_hash,duration_seconds=8,text=False,grid=False))
    print('Added exact boundary dots: '+str([len(p) for p in scene.boundary]),flush=True)
    return frame,verify

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--encode',action='store_true');args=parser.parse_args()
    frame,verify=prepare()
    if not args.encode:
        for t,name in [(0,'start01'),(.75,'dots01'),(3,'growth01'),(7.75,'final01')]:frame(t).save(TEMP/(name+'.png'))
        verify();return
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900',
        '-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-profile:v','main',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(TEMP/'01.mp4')],stdin=subprocess.PIPE)
    started=time.monotonic()
    try:
        for i in range(FPS*SECONDS):
            encoder.stdin.write(frame(i/FPS).tobytes())
            if i%24==0:print(json.dumps(dict(video=1,frame=i,total=192,seconds=round(time.monotonic()-started,1))),flush=True)
        encoder.stdin.close();assert encoder.wait()==0
    except BaseException:encoder.kill();encoder.wait();raise
    verify()
if __name__=='__main__':main()
