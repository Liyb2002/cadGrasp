"""Final pose1 and continuous wrap sweep, four isometric camera quadrants."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'vis_func'))
from solid_render import depth_render
import render as R


def main():
    base=R.CO/'output/B/pose1+2+4+6';out=Path(__file__).resolve().parents[1]/'vis'
    from layout_geometry import Layout
    layout=Layout();obj=layout.obj;T=layout.transforms[0]
    record=R.json.loads((out/'animation.json').read_text())
    offset=np.asarray(record['states'][-1]['offsets_fixture_m'][0])
    width=float(np.linalg.norm(offset))
    offsets=np.zeros((4,3));offsets[0]=offset
    final_solid,_=layout.construct(offsets)
    support=R.S.unpack(final_solid);support.apply_transform(T)
    body=obj.copy();body.apply_translation(offset);body.apply_transform(T)
    bounds=np.vstack([support.vertices,body.vertices])*1000
    center=(bounds.min(0)+bounds.max(0))/2;radius=np.ptp(bounds,axis=0).max()*.55
    fig=plt.figure(figsize=(12,10),dpi=180,facecolor='white')
    light=np.array([1.,-1.,1.])+np.array([-.15,-.2,.65]);light/=np.linalg.norm(light)
    elevation=np.degrees(np.arctan(1/np.sqrt(2)))
    panels=[]
    for index,azimuth in enumerate([-45.,45.,135.,225.]):
        ax=fig.add_subplot(2,2,index+1,projection='3d',computed_zorder=False)
        triangles=[];colors=[]
        for mesh,color,alpha in [(body,'#a4a8ac',1.),(support,'#319cd7',1.)]:
            rgb=np.array(matplotlib.colors.to_rgb(color))
            shade=.58+.42*np.maximum(mesh.face_normals@light,0)
            triangles.append(mesh.triangles*1000)
            colors.append(np.c_[shade[:,None]*rgb,np.full(len(mesh.faces),alpha)])
        panels.append((ax,np.concatenate(triangles),np.concatenate(colors)))
        for axis,value in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(value-radius,value+radius)
        ax.view_init(elev=elevation,azim=azimuth);ax.set_proj_type('ortho')
        ax.set_box_aspect((1,1,1),zoom=1.25);ax.set_axis_off()
        ax.text2D(.05,.05,f'View {index+1}',transform=ax.transAxes,color='#46505a',fontsize=12)
    fig.suptitle('Pose 1 at final position | adjusted placement support',fontsize=17,y=.985)
    fig.text(.5,.025,f'Gray: pose 1   |   Blue: support   |   Translation: {width*1000:.1f} mm',ha='center',fontsize=12,color='#46505a')
    fig.subplots_adjust(left=0,right=1,bottom=.055,top=.955,wspace=0,hspace=0)
    fig.canvas.draw()
    for ax,triangles,colors in panels:depth_render(ax,triangles,colors)
    fig.savefig(out/'pose1_final_four_isometric.png',dpi=180,facecolor='white')
    for ext in ['pdf','svg']:(out/f'pose1_final_four_isometric.{ext}').unlink(missing_ok=True)
    plt.close(fig)
    print(out/'pose1_final_four_isometric.png')

if __name__=='__main__':main()
