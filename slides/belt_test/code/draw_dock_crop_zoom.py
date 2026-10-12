"""Same-view rerendered detail paired with archived shared-base scene."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import trimesh
from PIL import Image
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'slides/tools'))
import slide_scene as S
z=np.load(HERE/'original_geometry.npz')
def mesh(name):return trimesh.Trimesh(z[name+'_vertices'],z[name+'_faces'],process=False)
obj,blue,socket=[mesh(n) for n in ('object','blue','socket')]
# Use exactly the shared_base camera direction; recompute only focus and scale.
basis=S.R.axes(np.array([1.05,-1.25,.8]))
cam=S.Camera(z['port']-z['basis'][:,2]*.012,basis,.068,1250)
domain=SimpleNamespace(mesh=obj,work_ids=np.array([],dtype=int))
zoom,_,_=S.render(domain,parts=[(blue,(37,112,188)),(socket,(235,157,48))],cam=cam,ground=False)
source=Image.open(HERE.parent/'shared_base.png').convert('RGB')
left=source.crop((1080,400,1930,940));scale=1180/left.width
left=left.resize((1180,round(left.height*scale)),Image.Resampling.LANCZOS)
zoom=zoom.resize((1150,1150),Image.Resampling.LANCZOS)
canvas=Image.new('RGB',(2400,1250),'white');canvas.paste(left,(20,(1250-left.height)//2));canvas.paste(zoom,(1230,50))
p=HERE.parent/'dock_pressure_zoom.png';canvas.save(p);print(p)
