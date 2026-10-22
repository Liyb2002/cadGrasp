"""Shared raster scene and the current hull-orbit/four-task presentation."""
import numpy as np
from PIL import Image
from whole_task.step1.needle_render import Renderer
from .video import raster as r


class Scene(Renderer):
    def shape(self,mesh,color,mask=None):
        shape=r.Shape(self.ctx,self.shader,mesh.vertices,mesh.faces,color,smooth=False,
            dynamic=mask is not None,roughness=.57,metallic=.04)
        if mask is not None:
            colors=np.tile(color,(len(mesh.faces),1));colors[mask]=r.PINK;shape.paint(colors)
        return shape

    def picture(self,items,points,direction,lines=None):
        direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
        right=np.cross([0,0,1],direction);right/=np.linalg.norm(right);up=np.cross(direction,right)
        basis=np.array([right,up,direction]);p=np.asarray(points)@basis.T
        low,high=p.min(0),p.max(0);target=((low+high)/2)@basis
        scale=max(high[0]-low[0],(high[1]-low[1])*self.width/self.height)*1.16
        vp=r.orthographic(scale,scale*self.height/self.width)@r.view_matrix(direction,target)
        self.shader['view_projection'].write(np.asarray(vp,dtype='f4').T.tobytes())
        self.shader['eye'].value=tuple(target+direction*4)
        key=np.array([-.65,-.9,1.45]);key/=np.linalg.norm(key)
        light=r.orthographic(scale*1.8,scale*1.8)@r.view_matrix(key,target)
        for shader in (self.shader,self.shadow):shader['light_projection'].write(np.asarray(light,dtype='f4').T.tobytes())
        ctx=self.ctx;ctx.enable(self.gl.DEPTH_TEST)
        self.studio.shadow_buffer.use();ctx.viewport=(0,0,1536,1536);self.studio.shadow_buffer.clear(depth=1)
        for shape,T in items:shape.shadow(ctx,self.shadow,T)
        self.studio.gbuffer.use();ctx.viewport=(0,0,*self.studio.size)
        self.studio.gbuffer.clear(1,1,1,1,depth=1);self.studio.shadow.use(3)
        for shape,T in items:shape.draw(self.shader,T)
        if lines is not None:
            self.lines['vp'].write(np.asarray(vp,dtype='f4').T.tobytes());self.lines['ink'].value=(.78,.55,.16)
            buffer=ctx.buffer(np.asarray(lines,dtype='f4').tobytes())
            vao=ctx.vertex_array(self.lines,[(buffer,'3f','position')]);ctx.line_width=1.6
            vao.render(self.gl.LINES);vao.release();buffer.release()
        return Image.frombytes('RGB',(self.width,self.height),self.studio.finish(vp,direction)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)



def render_movie(model,groups,path,motion_path,size=(1600,900),fps=24):
    from .hulls_presentation import render_movie as render_hulls
    return render_hulls(model,groups,path,motion_path,size=size,fps=fps)
