"""Bounded exact-geometry cache; no discretization or motion approximation."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
from collections import OrderedDict
import hashlib
_original_sweep=S.swept_solid
_original_solid=S.solid
_entries=OrderedDict();_by_mesh={};stats=dict(hits=0,misses=0)
def cached_sweep(mesh,displacement,*,fan_in=8):
    h=hashlib.sha256();h.update(np.asarray(mesh.vertices,dtype=np.float64).tobytes());h.update(np.asarray(mesh.faces,dtype=np.int64).tobytes());h.update(np.asarray(displacement,dtype=np.float64).tobytes());h.update(str(fan_in).encode());key=h.digest()
    if key in _entries:
        stats['hits']+=1;_entries.move_to_end(key);return _entries[key][0]
    stats['misses']+=1;result=_original_sweep(mesh,displacement,fan_in=fan_in);solid=_original_solid(result);entry=(result,solid);_entries[key]=entry;_by_mesh[id(result)]=entry
    if len(_entries)>64:
        _,old=_entries.popitem(last=False);_by_mesh.pop(id(old[0]),None)
    return result
def cached_solid(mesh):
    entry=_by_mesh.get(id(mesh))
    return entry[1] if entry is not None and entry[0] is mesh else _original_solid(mesh)
def install():
    S.swept_solid=cached_sweep;S.solid=cached_solid
