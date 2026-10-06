"""Continuous exit guidance, separate from exact construction acceptance.

A continuous trajectory queries a fixed trilinear object distance field at
material probes. Geometry sensitivities use central differences; normal
compatibility is analytic. No Boolean-surface derivative is assumed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from co_common import np, trimesh, S
from scipy.special import logsumexp, expit


def tangent_frames(directions):
    d = np.asarray(directions, float)
    if not np.isfinite(d).all() or not np.allclose(np.linalg.norm(d, axis=1), 1.):
        raise ValueError('directions must be finite unit vectors')
    axes = np.eye(3)[np.argmin(np.abs(d), axis=1)]
    u = np.cross(d, axes)
    u /= np.linalg.norm(u, axis=1)[:, None]
    return np.stack([u, np.cross(d, u)], axis=2)


def retract(origin, frames, coordinates):
    raw = origin + np.einsum('nki,ni->nk', frames, coordinates)
    return raw / np.linalg.norm(raw, axis=1)[:, None]


def retraction_jacobian(origin, frames, coordinates):
    raw = origin + np.einsum('nki,ni->nk', frames, coordinates)
    norm = np.linalg.norm(raw, axis=1)
    d = raw / norm[:, None]
    return np.einsum('nkl,nli->nki', np.eye(3)[None] - d[:,:,None]*d[:,None,:], frames) / norm[:,None,None]


def acquisition(directions, normals, width=.02):
    """Smooth positive maximum initial-motion violation and Cartesian Jacobian."""
    if width <= 0:
        raise ValueError('width must be positive')
    dots = normals @ directions.T
    z = dots / width
    logsum = logsumexp(z, axis=1)
    costs = width * np.logaddexp(0., logsum)
    probabilities = np.exp(z - logsum[:,None])
    jacobian = expit(logsum)[:,None,None] * probabilities[:,:,None] * normals[:,None,:]
    return costs, jacobian


class SweepDistanceModel:
    """Smooth trajectory acquisition field from a fixed object distance grid.

    phi(q,d) = tau log sum_t exp((sdf(q-t*d)+margin)/tau), with fixed
    time quadrature over the complete exit interval. Direction is continuous;
    field/time discretization is optimization guidance, not a sweep certificate.
    """
    def __init__(self, clearance, points, normals, length, extent, depth=.003, step=2e-4):
        from physics_guided_field import ObjectDistanceField
        self.clearance=clearance
        self.probes=np.asarray(points)+depth*np.asarray(normals)
        self.length,self.extent,self.step=length,extent,step
        self.field=ObjectDistanceField(clearance.mesh)
        self.margin=clearance.metadata['maximum_kernel_radius_m']
        self.times=np.linspace(0.,length,129)
        self.temperature=extent*.005
        self.evaluations=0

    def distances(self,direction):
        self.evaluations+=1
        positions=self.probes[:,None,:]-self.times[None,:,None]*direction
        values=self.field.sample(positions.reshape(-1,3)).reshape(len(self.probes),-1)
        # Unnormalized smooth maximum avoids suppressing short intercepted
        # features by their time measure. Its fixed bias is explicit guidance.
        return (self.temperature*logsumexp(values/self.temperature,axis=1)+self.margin)/self.extent

    def linearize(self,origin,frames):
        values=[];jacobian=[]
        for i,d in enumerate(origin):
            values.append(self.distances(d))
            gradients=[]
            for axis in range(2):
                plus=d+self.step*frames[i,:,axis];plus/=np.linalg.norm(plus)
                minus=d-self.step*frames[i,:,axis];minus/=np.linalg.norm(minus)
                gradients.append((self.distances(plus)-self.distances(minus))/(2*self.step))
            jacobian.append(np.column_stack(gradients))
        return np.column_stack(values),np.stack(jacobian,axis=1)


def distance_cost(values, jacobian, coordinates, width=.005):
    """Smooth all-pose sweep acquisition cost for a local distance model."""
    predicted = values + np.einsum('jni,ni->jn', jacobian, coordinates)
    z = predicted / width
    logsum = logsumexp(z, axis=1)
    weights = np.exp(z - logsum[:,None])
    costs = width*np.logaddexp(0., logsum)
    gradient = expit(logsum)[:,None,None]*weights[:,:,None]*jacobian
    return costs, gradient
