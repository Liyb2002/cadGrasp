from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/yli581/Desktop/cadGrasp')
PROOF = ROOT/'slides/Co-optimize/seperate_base_support_proof'
TEMP = Path(__file__).resolve().parent
sys.path.insert(0, str(PROOF))
import diagnose as D
import base_geometry as G
import numpy as np
import trimesh
from PIL import Image, ImageDraw

C = D.C
FPS, SECONDS, MAIN, SMALL = 24, 16, 900, 340
DEST = PROOF/'videos/01_hulls_and_support.mp4'


def basis_for(angle):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    view = rotation @ (np.array([1.,-1.,1.])/np.sqrt(3))
    right = rotation @ (np.array([1.,1.,0.])/np.sqrt(2))
    return np.array([right, np.cross(view, right), view])


def disk(vertices, color):
    faces = np.array([[0,i,i+1] for i in range(1,len(vertices)-1)])
    value = C.V.piece(trimesh.Trimesh(vertices, faces, process=False), color, .34)
    value['unlit'] = True
    return value


def edge(vertices, color, camera, size):
    value = C.H.outline_piece(vertices, color, camera, size, half_width_pixels=.9)
    value['colors'] = np.broadcast_to(C.V.CPU.rgb(color), value['triangles'].shape)
    value['opacity'] = .84
    return value


def prepare():
    inp = C.Inputs('B', D.BASE/'pose2+3+4+7/data/selection.json')
    inp.base = D.BASE
    data = inp.load(inp.cases[0])
    assert [row['pose'] for row in data['rows']] == ['pose_2','pose_3','pose_4','pose_7']
    assert len(G.solid(data['fixture']).decompose()) == 1
    merged = data['fixture'].copy()
    merged.merge_vertices()
    assert len(merged.split(only_watertight=False)) == 1
    ref = data['rows'][0]['fixture_transform']
    fixture = C.posed(data['fixture'], ref)
    hulls = []
    normals = []
    for row in data['rows']:
        inv = np.linalg.inv(row['fixture_transform'])
        local = C.F.transform_points(row['hull'], inv)
        assert np.max(np.abs(C.F.transform_points(local, row['fixture_transform'])-row['hull'])) < 1e-12
        normal = row['fixture_transform'][2,:3]
        assert np.ptp(local@normal) < 1e-12
        hulls.append(C.F.transform_points(local, ref))
        normals.append(normal)
    points = np.vstack([fixture.vertices,*hulls])
    focus = (points.min(0)+points.max(0))/2
    projected_max = max(float(np.abs((points-focus)@basis_for(a)[:2].T).max())
                        for a in np.linspace(0,2*np.pi,361))
    span = 2*projected_max*1.12
    assert np.allclose(basis_for(0),basis_for(2*np.pi),atol=1e-12)
    pose_geometries = []
    cameras = []
    for row in data['rows']:
        support = C.posed(data['fixture'],row['fixture_transform'])
        body = C.posed(data['body'],row['object_transform'])
        camera = C.V.camera_for(np.vstack([support.vertices,body.vertices,row['hull']]))
        pose_geometries.append((support,body))
        cameras.append(camera)
    right_span = max(camera[2] for camera in cameras)
    cameras = [(focus,basis,right_span) for focus,basis,_ in cameras]
    renderer = C.V.Renderer()
    right = []
    for with_hulls in (False,True):
        panel = Image.new('RGB',(700,900),'white')
        for i,(row,(support,body),camera) in enumerate(zip(data['rows'],pose_geometries,cameras)):
            parts = [C.V.piece(support,C.V.BLUE,.82),C.V.piece(body,C.V.GRAY,.18)]
            if with_hulls:
                parts.extend([disk(row['hull'],row['color']),edge(row['hull'],row['color'],camera,SMALL)])
            tile = renderer.render([C.V.CPU.placed(p) for p in parts],camera,SMALL)
            panel.paste(tile,(7+(i%2)*346,55+(i//2)*450))
        right.append(panel)
    sources = C.freeze(TEMP/'execution')
    shutil.copy2(__file__,TEMP/'execution/render.py')
    own_digest = C.digest(__file__)
    others = {p.name:C.digest(p) for p in DEST.parent.iterdir() if p.name != DEST.name}
    metadata = dict(case=data['case']['id'],components=1,material_cm3=G.volume(G.solid(data['fixture']))*1e6,
                    poses=[row['pose'] for row in data['rows']],hull_vertices=[len(h) for h in hulls],
                    text=False,grid=False,duration_seconds=SECONDS,orbit_degrees=360,
                    minimum_plane_angle_degrees=min(float(np.degrees(np.arccos(np.clip(abs(np.dot(a,b)),-1,1))))
                                                    for i,a in enumerate(normals) for b in normals[i+1:]),
                    base_obstruction_proven=False)
    C.save(TEMP/'metadata.json',metadata)
    print(json.dumps(metadata),flush=True)
    def frame(angle, with_hulls):
        camera = (focus,basis_for(angle),span)
        parts = [C.V.piece(fixture,C.V.BLUE,.78)]
        if with_hulls:
            for hull,row in zip(hulls,data['rows']):
                parts.extend([disk(hull,row['color']),edge(hull,row['color'],camera,MAIN)])
        main = renderer.render([C.V.CPU.placed(p) for p in parts],camera,MAIN)
        image = Image.new('RGB',(1600,900),'white')
        image.paste(main,(0,0))
        image.paste(right[int(with_hulls)],(900,0))
        draw = ImageDraw.Draw(image)
        draw.line((900,0,900,899),fill='#eef0f2',width=1)
        draw.line((901,450,1599,450),fill='#eef0f2',width=1)
        return image
    def verify():
        inp.verify()
        for path,digest in sources.items():
            assert C.digest(ROOT/path) == digest
            assert C.digest(TEMP/'execution/executed_sources'/path) == digest
        assert C.digest(__file__) == own_digest == C.digest(TEMP/'execution/render.py')
        assert {p.name:C.digest(p) for p in DEST.parent.iterdir() if p.name != DEST.name} == others
    return frame, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encode', action='store_true')
    args = parser.parse_args()
    frame, verify = prepare()
    if not args.encode:
        frame(0,False).save(TEMP/'support.png')
        frame(0,True).save(TEMP/'hulls.png')
        frame(np.pi/2,True).save(TEMP/'quarter.png')
        frame(np.pi,True).save(TEMP/'half.png')
        verify()
        return
    encoder = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r',str(FPS),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','18','-profile:v','main',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(TEMP/'video.mp4')],stdin=subprocess.PIPE)
    start = time.monotonic()
    empty = frame(0,False).tobytes()
    full = frame(0,True).tobytes()
    try:
        for i in range(FPS*SECONDS):
            t = i/FPS
            if t < 2: rgb=empty
            elif t < 3 or t >= 15: rgb=full
            else: rgb=frame((t-3)/12*2*np.pi,True).tobytes()
            encoder.stdin.write(rgb)
            if i % (FPS*2) == 0:
                print(json.dumps(dict(frame=i,seconds=round(time.monotonic()-start,1))),flush=True)
        encoder.stdin.close()
        assert encoder.wait()==0
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    verify()
    print('Encoded; all saved inputs and executed sources unchanged.',flush=True)


if __name__ == '__main__':
    main()
