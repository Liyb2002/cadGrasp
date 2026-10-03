"""Pinned actual joint geometry; cutaway, hypothetical edge-contact highlights."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
HERE=Path(__file__).resolve().parent
z=np.load(HERE/'original_geometry.npz');B=z['basis'];port=z['port']
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
fig=plt.figure(figsize=(11,10),facecolor='white');ax=fig.add_subplot(111,projection='3d')
def local(name):return (z[name+'_vertices']-port)@B*1000
peg=local('peg');socket=local('socket')
tri=socket[z['socket_faces']];c=tri.mean(1)
# Remove front wall for visibility; keep original coordinates of every shown face.
tri=tri[c[:,1]>-4.51]
ax.add_collection3d(Poly3DCollection(tri,facecolors='#e4a048',edgecolors='none',linewidths=0,alpha=.22))
ax.add_collection3d(Poly3DCollection(peg[z['peg_faces']],facecolors='#378fbe',edgecolors='#245e80',linewidths=.8,alpha=.88))
# Highlight potential bearing strips ON actual inner x walls, not a computed map.
red=[]
for x,zlo,zhi in [(-9.49,-6.5,-5.01),(9.49,-26.19,-24.7)]:
    red.append(np.array([[x,-4.49,zlo],[x,4.49,zlo],[x,4.49,zhi],[x,-4.49,zhi]]))
ax.add_collection3d(Poly3DCollection(red,facecolors='#f20f28',edgecolors='#c10019',linewidths=1,alpha=1))



ax.view_init(elev=25,azim=-58)
ax.set(xlim=(-17,17),ylim=(-11,11),zlim=(-31,5));ax.set_box_aspect((34,22,36));ax.set_axis_off()
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
# Project edge markers as a presentation overlay so transparent walls cannot hide red.
from mpl_toolkits.mplot3d import proj3d
fig.canvas.draw()
for x,zz,tx,ty in [(-9.49,-5.75,.13,.77),(9.49,-25.45,.72,.27)]:
    xp,yp,_=proj3d.proj_transform(x,0,zz,ax.get_proj())
    a=proj3d.proj_transform(x,-4.4,zz,ax.get_proj());b=proj3d.proj_transform(x,4.4,zz,ax.get_proj())
    from matplotlib.lines import Line2D
    ax.add_artist(Line2D([a[0],b[0]],[a[1],b[1]],transform=ax.transData,color='#f01830',lw=7,solid_capstyle='round',zorder=100))
p=HERE.parent/'socket_pressure_3d_zoom.png';fig.savefig(p,dpi=200)
from PIL import Image, ImageChops
# Left is the existing dock-1 panel from shared_base.png, with its title cropped.
scene=Image.open(HERE.parent/'shared_base.png').convert('RGB').crop((1000,300,2000,1130))
zoom=Image.open(p).convert('RGB')
def trim(im):
    bbox=ImageChops.difference(im,Image.new('RGB',im.size,'white')).getbbox()
    return im.crop(bbox)
scene=trim(scene);zoom=trim(zoom)
canvas=Image.new('RGB',(2400,1300),'white')
for im,x,width,height in [(scene,50,1150,1100),(zoom,1250,1050,1150)]:
    im.thumbnail((width,height),Image.Resampling.LANCZOS)
    canvas.paste(im,(x+(width-im.width)//2,(1300-im.height)//2))
out=HERE.parent/'dock_pressure_zoom.png';canvas.save(out);print(out)
