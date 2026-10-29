"""Reuse exact world sweep meshes across isolated proposal processes.

This stores construction geometry, never a force or exit certificate. The
body, direction, length, implementation and native version are in the key;
the immutable archive is independently hash checked before use.
"""
import hashlib
import json
from pathlib import Path
import tempfile
from importlib.metadata import version
import numpy as np
import trimesh


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sweep_mesh(folder,body,direction,length,build,implementation):
    key=hashlib.sha256(implementation.encode())
    for array in (body.vertices,body.faces,np.asarray(direction),np.asarray([length])):
        value=np.ascontiguousarray(array)
        key.update(str((value.dtype.str,value.shape)).encode());key.update(value.tobytes())
    key=key.hexdigest();folder=Path(folder);manifest=folder/(key+'.json')
    try:
        info=json.loads(manifest.read_text());path=folder/(info['mesh_sha256']+'.npz')
        if info['key']==key and digest(path)==info['mesh_sha256']:
            with np.load(path,allow_pickle=False) as z:
                return trimesh.Trimesh(z['vertices'],z['faces'],process=False)
    except (OSError,ValueError,KeyError):
        pass
    result=build()
    folder.mkdir(parents=True,exist_ok=True)
    temporary=None;meta=None
    try:
        with tempfile.NamedTemporaryFile(dir=folder,suffix='.npz',delete=False) as stream:
            temporary=Path(stream.name)
            np.savez_compressed(stream,vertices=result.vertices,faces=result.faces)
        checksum=digest(temporary);destination=folder/(checksum+'.npz')
        # Multiple workers may build the same geometry. The manifest points
        # at one immutable archive; it cannot observe a half-written pair.
        if not destination.exists() or digest(destination)!=checksum:
            temporary.replace(destination)
        with tempfile.NamedTemporaryFile(mode='w',dir=folder,suffix='.json',delete=False) as stream:
            meta=Path(stream.name);json.dump(dict(key=key,mesh_sha256=checksum),stream)
        meta.replace(manifest)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
        if meta is not None:meta.unlink(missing_ok=True)
    return result


def implementation_key():
    from subdivide_task.step3 import exit_geometry,independent_geometry
    return ':'.join([digest(__file__),digest(exit_geometry.__file__),
        digest(independent_geometry.__file__),version('manifold3d'),str(independent_geometry.SCALE)])
