"""Coupled object/fixture quasistatic equilibrium, no rigid weld assumption."""
import numpy as np
from scipy.optimize import linprog

def check(mesh,fixture,triangles,sources,contacts,normals,rotations,mu=.8):
    # Object mass is normalized to one. Equal uniform density fixes fixture
    # mass from material volume; this is an explicit pilot assumption.
    ratio=abs(fixture.volume)/abs(mesh.volume)
    columns=[]
    com=mesh.center_mass;fixture_com=fixture.center_mass
    for p,n in zip(triangles.reshape(-1,3),np.repeat(-mesh.face_normals[sources],3,axis=0)):
        columns.append(np.r_[n,np.cross(p-com,n),-n,np.cross(p-fixture_com,-n)])
    for p,n in zip(contacts,normals):
        axis=np.eye(3)[np.argmin(abs(n))];u=np.cross(n,axis);u/=np.linalg.norm(u);v=np.cross(n,u)
        for angle in np.linspace(0,2*np.pi,8,endpoint=False):
            force=n+mu*(u*np.cos(angle)+v*np.sin(angle))
            columns.append(np.r_[np.zeros(6),force,np.cross(p-fixture_com,force)])
    if not columns:return dict(passed=False,reason='no_contacts')
    A=np.asarray(columns).T;results=[]
    for R in rotations:
        up=R.T@np.array([0.,0.,1.]);target=np.r_[up,np.zeros(3),ratio*up,np.zeros(3)]
        solution=linprog(np.zeros(A.shape[1]),A_eq=A,b_eq=target,bounds=(0,None),method='highs')
        results.append(bool(solution.success and np.max(abs(A@solution.x-target))<1e-7))
    return dict(passed=all(results),orientation_results=results,fixture_object_mass_ratio=ratio,friction=mu,assumptions='equal uniform density; frictionless object/fixture; 8-ray gripper friction cones; quasistatic gravity only; no actuator force limit or acceleration proof')
