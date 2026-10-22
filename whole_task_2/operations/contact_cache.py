"""Reuse exact contact queries on the same oriented body and boundary mesh.

Support triangle order and cyclic vertex order carry no geometric meaning.
Coordinates are never rounded. Loads are always classified independently.
"""
from contextlib import contextmanager
from threading import RLock
import hashlib
import numpy as np
from whole_task_2.step3 import independent_physics as physics

_lock=RLock()


def canonical_triangles(triangles):
    # Canonicalize triangle vertices and triangle order, at full precision.
    first=np.lexsort((triangles[:,:,2],triangles[:,:,1],triangles[:,:,0]),axis=1)[:,0]
    order=(first[:,None]+np.arange(3))%3
    values=np.take_along_axis(triangles,order[:,:,None],axis=1).reshape(-1,9)
    values=values[np.lexsort(tuple(values[:,j] for j in range(8,-1,-1)))]
    return values


def boundary_key(triangles):
    return hashlib.sha256(np.ascontiguousarray(canonical_triangles(triangles)).tobytes()).hexdigest()


def query_key(body,support,excluded,com):
    values=canonical_triangles(support.triangles)
    digest=hashlib.sha256()
    for a in (body.vertices,body.faces,np.asarray(excluded),np.asarray(com),values):
        a=np.ascontiguousarray(a)
        digest.update(str((a.shape,a.dtype.str)).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def query_contacts(body,support,excluded,com,cache,primitive=None):
    key=query_key(body,support,excluded,com)
    if key in cache:return tuple(a.copy() for a in cache[key])
    value=(physics.contacts if primitive is None else primitive)(body,support,excluded,com)
    cache[key]=tuple(a.copy() for a in value)
    if len(cache)>96:cache.pop(next(iter(cache)))
    return value


@contextmanager
def cached_certificate_contacts(cache):
    """Run the unchanged certificate with a read-only exact-query cache."""
    with _lock:
        original=physics.contacts
        def cached(body,support,excluded,com):
            return query_contacts(body,support,excluded,com,cache,original)
        physics.contacts=cached
        try:yield
        finally:physics.contacts=original
