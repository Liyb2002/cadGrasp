"""Legacy shared-tendency and large coordinate proposal families, isolated from acceptance."""
import numpy as np
from scipy.optimize import linprog


def fibonacci(count):
    k=np.arange(count);z=1-2*(k+.5)/count;angle=k*np.pi*(3-np.sqrt(5))
    return np.c_[np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z]


def project_common(common,normals,lift=.01):
    directions=[]
    for normal in normals:
        d=common-min(common@normal,0.)*normal+lift*normal
        if np.linalg.norm(d)<1e-10:
            axis=np.eye(3)[np.argmin(abs(normal))];d=axis-(axis@normal)*normal+lift*normal
        directions.append(d/np.linalg.norm(d))
    return np.asarray(directions)


def floor_common(normals):
    lp=linprog([0,0,0,-1],A_ub=np.c_[-normals,np.ones(len(normals))],
        b_ub=np.zeros(len(normals)),bounds=[(-1,1)]*3+[(None,None)],method='highs')
    if lp.success and np.linalg.norm(lp.x[:3])>1e-8:return lp.x[:3]/np.linalg.norm(lp.x[:3])
    return None


def common_neighborhood(center,count=16):
    axis=np.eye(3)[np.argmin(abs(center))];u=np.cross(center,axis);u/=np.linalg.norm(u);v=np.cross(center,u)
    for angle in [20,10,5,2,.5]:
        for phase in np.linspace(0,2*np.pi,count,endpoint=False):
            a=np.cos(np.deg2rad(angle))*center+np.sin(np.deg2rad(angle))*(np.cos(phase)*u+np.sin(phase)*v)
            for lift in [.0001,.003,.01,.03]:yield a,lift


def independent_neighborhood(directions,normals,order):
    for k in order:
        center=directions[k];axis=np.eye(3)[np.argmin(abs(center))]
        u=np.cross(center,axis);u/=np.linalg.norm(u);v=np.cross(center,u)
        for angle in [20,40,10,5,2,.5]:
            for phase in np.linspace(0,2*np.pi,12,endpoint=False):
                candidate=directions.copy()
                candidate[k]=np.cos(np.deg2rad(angle))*center+np.sin(np.deg2rad(angle))*(np.cos(phase)*u+np.sin(phase)*v)
                if candidate[k]@normals[k]>=-1e-12:yield candidate,k
