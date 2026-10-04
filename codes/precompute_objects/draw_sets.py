"""Render one sets.png per object, grouping the exact saved poses by task set."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'slides/baseline_algo'))
import numpy as np
from PIL import Image, ImageDraw
import trimesh
from codes.precompute_objects.registry import active_objects
from codes.precompute_objects.work_regions import digest
from step2_local_support import render as R

TILE = 208
CARD_W, CARD_H = 1320, 304
MARGIN, GAP = 36, 24


def render(name, folder=None):
    folder = Path(folder) if folder is not None else ROOT/'objects'/name
    data = json.loads((folder/'pose_sets.json').read_text())
    if data['pose_manifest_sha256'] != digest(folder/'poses.json'):
        raise ValueError('Stale pose-set manifest')
    manifest = json.loads((folder/'poses.json').read_text())
    used = {p for group in data['sets'] for p in group['poses']}
    rows = {p['pose_id']:p for p in manifest['poses']}
    raw = trimesh.load(folder/'mesh.stl', force='mesh')
    meshes = {0:raw}
    scenes = {}
    basis = R.axes([4., -6., 4.5])
    all_points = []
    for pose in sorted(used, key=lambda p:int(p.split('_')[1])):
        target = folder/'poses'/pose
        meta = json.loads((target/'setup.json').read_text())
        rounds = meta['uniform_subdivision_rounds']
        for level in range(1, rounds+1):
            if level not in meshes: meshes[level] = meshes[level-1].subdivide()
        mesh = meshes[rounds]
        with np.load(target/'setup.npz') as setup:
            T = setup['T_world_mesh'].copy()
            mask = setup['work_faces'].copy()
        np.testing.assert_array_equal(T, rows[pose]['T_world_mesh'])
        vertices = mesh.vertices@T[:3,:3].T+T[:3,3]
        scenes[pose] = (vertices[mesh.faces], mask)
        all_points.append(vertices)
    projected = np.concatenate(all_points)@basis.T
    low, high = projected.min(axis=0), projected.max(axis=0)
    focus = ((low+high)/2)@basis
    width = float((high-low)[:2].max())*1.17
    floor_radius = float(raw.extents.max())*.85
    corners = np.array([[-floor_radius,-floor_radius,-1e-7],
                        [ floor_radius,-floor_radius,-1e-7],
                        [ floor_radius, floor_radius,-1e-7],
                        [-floor_radius, floor_radius,-1e-7]])
    floor = corners[[[0,1,2],[0,2,3]]]
    tiles = {}
    for pose,(triangles,mask) in scenes.items():
        colors = np.tile([107.,136.,155.],(len(triangles),1))
        colors[mask] = [249.,156.,57.]
        image,_ = R.raster(np.concatenate([floor,triangles]),
            np.vstack([np.tile([242.,245.,247.],(2,1)),colors]),
            focus,basis,width,TILE,unlit=(0,1))
        tiles[pose] = image
    groups = sorted(data['sets'], key=lambda group:(group['size'],group['id']))
    height = 164+10*CARD_H+9*GAP+MARGIN
    sheet = Image.new('RGB',(2*CARD_W+2*MARGIN+GAP,height),'#f7f9fb')
    ink = ImageDraw.Draw(sheet)
    ink.text((MARGIN,26),f'{name} / SETS',font=R.font(42),fill='#233847')
    ink.text((MARGIN,86),'20 sets  |  4 sets per size: 2, 3, 4, 5, 6 poses',font=R.font(23),fill='#607383')
    ink.rounded_rectangle((sheet.width-347,44,sheet.width-324,67),radius=4,fill='#f99c39')
    ink.text((sheet.width-310,42),'Working surface',font=R.font(22),fill='#607383')
    for index,group in enumerate(groups):
        x = MARGIN+(index%2)*(CARD_W+GAP)
        y = 150+(index//2)*(CARD_H+GAP)
        ink.rounded_rectangle((x,y,x+CARD_W,y+CARD_H),radius=16,fill='white',outline='#dfe6eb',width=2)
        label = ' + '.join(p.split('_')[1] for p in group['poses'])
        ink.text((x+22,y+15),f"{group['size']} poses  /  {label}",font=R.font(25),fill='#304b5c')
        count = len(group['poses'])
        start = x+(CARD_W-count*TILE-(count-1)*8)//2
        for j,pose in enumerate(group['poses']):
            tx = start+j*(TILE+8)
            sheet.paste(tiles[pose],(tx,y+53))
            ink.text((tx+TILE/2,y+CARD_H-25),f"Pose {pose.split('_')[1]}",font=R.font(20),fill='#607383',anchor='mm')
    path = folder/'sets.png'
    sheet.save(path)
    print(path,flush=True)
    return str(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects',nargs='*')
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(render,args.objects or active_objects()))

if __name__=='__main__': main()
