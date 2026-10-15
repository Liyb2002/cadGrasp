"""Disposable, exact-input caches for one Step4 execution.

Only owned local files are loaded. Native Manifold values are never serialized:
their reconstructed boundary is not an equivalent Boolean operand. Completed
mesh boundaries, arrays, masks and query records can be stored losslessly.
"""
from collections import OrderedDict
from collections.abc import MutableMapping
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import time

import numpy as np

SCHEMA = 'step4-runtime-v1'
MAGIC = b'COOPT-RUNTIME-1\n'
MISSING = object()


def fingerprint(*values):
    return hashlib.sha256(pickle.dumps(values, protocol=5)).hexdigest()


def array_key(value):
    a = np.ascontiguousarray(value)
    return a.dtype.str, a.shape, hashlib.sha256(a.tobytes()).digest()


def mesh_key(mesh):
    return array_key(mesh.vertices), array_key(mesh.faces)


def layout_key(layout):
    """No new rounding or geometric equivalence assumption."""
    return (tuple(layout.active), array_key(layout.hosts),
            array_key(layout.placements), array_key(layout.directions),
            array_key(layout.state_offsets))


class NativeLRU(OrderedDict):
    """Bound native Boolean objects in memory; a miss recomputes the original."""
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.evictions = 0

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.capacity:
            self.popitem(last=False)
            self.evictions += 1


class EvidenceLRU(MutableMapping):
    """Keep live mutable evidence hot; persist its latest value on eviction."""
    def __init__(self, store, namespace, capacity=32):
        self.store=store;self.namespace=namespace;self.capacity=capacity
        self.memory=OrderedDict();self.disk_keys=set()

    def __getitem__(self,key):
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        if key not in self.disk_keys:raise KeyError(key)
        value=self.store.get(self.namespace,key)
        if value is MISSING:
            raise RuntimeError('Saved layout evidence is missing or corrupt')
        self[key]=value
        return value

    def __setitem__(self,key,value):
        self.memory[key]=value;self.memory.move_to_end(key)
        self.disk_keys.discard(key)
        while len(self.memory)>self.capacity:
            old,item=self.memory.popitem(last=False)
            self.store.put(self.namespace,old,item)
            self.disk_keys.add(old)

    def __delitem__(self,key):
        if key in self.memory:del self.memory[key]
        elif key in self.disk_keys:self.disk_keys.remove(key)
        else:raise KeyError(key)

    def __iter__(self):
        return iter(tuple(self.memory)+tuple(self.disk_keys))

    def __len__(self):return len(self.memory)+len(self.disk_keys)

    def __contains__(self,key):return key in self.memory or key in self.disk_keys

    def __getstate__(self):
        return {k:v for k,v in vars(self).items() if k!='store'}

    def __setstate__(self,state):
        vars(self).update(state);self.store=None


class SearchRuntime:
    def __init__(self, directory, scope, memory_bytes=32*1024**2):
        self.directory = Path(directory)
        self.scope = scope
        self.memory_bytes = int(memory_bytes)
        self.hot = OrderedDict()
        self.hot_bytes = 0
        self.stats = {}
        self.timings = {}
        self.restored_checkpoint = None
        self.started=time.monotonic()
        self.prior_elapsed=0.
        self.directory.mkdir(parents=True, exist_ok=True)
        marker = self.directory/'owner.json'
        expected = dict(schema=SCHEMA, scope=scope)
        if marker.exists():
            if json.loads(marker.read_text()) != expected:
                raise ValueError('Runtime cache source/input/configuration scope changed')
        elif any(self.directory.iterdir()):
            raise ValueError('Refusing to use an unowned runtime directory')
        else:
            marker.write_text(json.dumps(expected)+'\n')

    def _stats(self, namespace):
        return self.stats.setdefault(namespace, dict(memory_hits=0, disk_hits=0,
            misses=0, writes=0, corrupt=0, read_s=0., write_s=0.))

    def _path(self, namespace, key):
        return self.directory/namespace/(fingerprint(key)+'.bin')

    def _remember(self, token, value, size):
        old = self.hot.pop(token, None)
        if old is not None:
            self.hot_bytes -= old[1]
        if size > self.memory_bytes:
            return
        self.hot[token] = value, size
        self.hot_bytes += size
        while self.hot_bytes > self.memory_bytes:
            _, (_, amount) = self.hot.popitem(last=False)
            self.hot_bytes -= amount

    def get(self, namespace, key, default=MISSING):
        began = time.monotonic()
        stat = self._stats(namespace)
        path = self._path(namespace, key)
        token = str(path)
        if token in self.hot:
            value, _ = self.hot[token]
            self.hot.move_to_end(token)
            stat['memory_hits'] += 1
            return value
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            stat['misses'] += 1
            return default
        payload = raw[len(MAGIC)+32:]
        if (not raw.startswith(MAGIC) or
                hashlib.sha256(payload).digest() != raw[len(MAGIC):len(MAGIC)+32]):
            stat['corrupt'] += 1
            stat['misses'] += 1
            return default
        value = pickle.loads(payload)
        stat['disk_hits'] += 1
        stat['read_s'] += time.monotonic()-began
        self._remember(token, value, len(payload))
        return value

    def put(self, namespace, key, value):
        began = time.monotonic()
        stat = self._stats(namespace)
        payload = pickle.dumps(value, protocol=5)
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix='.writing-', dir=path.parent)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(MAGIC)
                stream.write(hashlib.sha256(payload).digest())
                stream.write(payload)
                stream.flush()
                if namespace=='checkpoint':os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self._remember(str(path), value, len(payload))
        stat['writes'] += 1
        stat['write_s'] += time.monotonic()-began

    @contextmanager
    def span(self, name):
        began = time.monotonic()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.)+time.monotonic()-began

    def report(self):
        return dict(schema=SCHEMA, scope=self.scope, cache=self.stats,
                    compute_seconds=self.timings, hot_bytes=self.hot_bytes,
                    memory_budget_bytes=self.memory_bytes,
                    elapsed_seconds_including_prior_fragments=self.prior_elapsed+time.monotonic()-self.started,
                    continued_from_checkpoint=self.restored_checkpoint is not None)

    def cleanup(self):
        marker = json.loads((self.directory/'owner.json').read_text())
        if marker != dict(schema=SCHEMA, scope=self.scope):
            raise ValueError('Runtime directory ownership changed')
        self.hot.clear()
        self.hot_bytes = 0
        shutil.rmtree(self.directory)


def execution_scope(model, options):
    """Bind all numerical sources and physical inputs before the first lookup."""
    coopt = Path(__file__).resolve().parents[1]
    sources = []
    for folder in ['helper_func', 'step4.2', 'step5', 'step5.2', 'step5.3', 'vis_func']:
        for path in (coopt/folder).rglob('*'):
            if (path.is_file() and path.suffix in ('.py', '.so') and
                    not set(path.relative_to(coopt/folder).parts).intersection(
                        {'data', 'output', '__pycache__', 'tests'})):
                sources.append((str(path.relative_to(coopt)), hashlib.sha256(path.read_bytes()).hexdigest()))
    inputs = [(str(p), hashlib.sha256(Path(p).read_bytes()).hexdigest())
              for p in sorted(set(model.inputs), key=str)]
    root=coopt.parents[1]
    external=[]
    for folder in ('codes/precompute_objects','slides/tools','slides/obj_supp'):
        for path in (root/folder).rglob('*'):
            if (path.is_file() and path.suffix in ('.py','.so') and
                    not set(path.relative_to(root/folder).parts).intersection(
                        {'data','output','__pycache__','tests'})):
                external.append((str(path.relative_to(root)),hashlib.sha256(path.read_bytes()).hexdigest()))
    physical_root=Path(os.environ.get('COOPT_REPO_ROOT',root)).resolve()
    for name in ('objects/r6_demand.py','objects/r6_quadratic.py'):
        path=physical_root/name
        if path.exists():external.append((name,hashlib.sha256(path.read_bytes()).hexdigest()))
    from importlib.metadata import version, PackageNotFoundError
    dependencies=[]
    for name in ('numpy','scipy','numba','trimesh','manifold3d','shapely','torch'):
        try:dependencies.append((name,version(name)))
        except PackageNotFoundError:dependencies.append((name,None))
    numerical_options = {k:v for k,v in options.items() if k not in
        ('jobs', 'sets', 'continue_checkpoint', 'runtime_cache', 'runtime_cache_memory_mb')}
    return fingerprint(SCHEMA, sorted(sources), sorted(set(external)),dependencies,sys.version,
                       inputs, numerical_options,
                       array_key(model.native), mesh_key(model.mesh))


def runtime(model):
    return getattr(model, 'search_runtime', None)
