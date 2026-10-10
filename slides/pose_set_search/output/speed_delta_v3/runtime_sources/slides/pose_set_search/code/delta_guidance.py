"""Persistent coverage/lock counts; candidate evaluation contains no solid Boolean.

States are immutable after construction. Trial states never mutate the committed
state. Translating a blocker updates its column AND its own contact row.
"""
from collections import OrderedDict
from dataclasses import dataclass
import time
from common import *
from volume_guidance import VolumeGuidance


@dataclass
class ContactState:
    layout: object
    coverage: dict
    locks: dict
    coverage_counts: dict
    lock_counts: dict
    available: dict


class ContactDelta:
    def __init__(self, model):
        self.model = model
        self.base = None
        self.cache = OrderedDict()
        self.pair_updates = 0

    def state(self, layout):
        key = layout.key()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        m, old = self.model, self.base
        before = set(old.layout.active) if old is not None else set()
        after = set(layout.active)
        moved = {k for k in before & after if not np.array_equal(
            old.layout.placements[k], layout.placements[k])}
        direction = {k for k in before & after if not np.array_equal(
            old.layout.directions[k], layout.directions[k])}
        coverage, locks = {}, {}
        coverage_counts, lock_counts, available = {}, {}, {}
        for owner in layout.active:
            new_row = owner not in before or owner in moved
            count = (np.zeros(len(m.points), np.int16) if new_row else
                     old.coverage_counts[owner].copy())
            blocked = (np.zeros(len(m.points), np.int16) if new_row else
                       old.lock_counts[owner].copy())
            for provider in before | after:
                pair = owner, provider
                replace_material = new_row or provider in moved or provider not in before or provider not in after
                replace_lock = replace_material or provider in direction
                if not new_row and provider in before:
                    if replace_material:
                        count -= old.coverage[pair]
                    if replace_lock:
                        blocked -= old.locks[pair]
                if provider not in after:
                    continue
                coverage[pair] = (m.material_at_points(layout, owner, provider)
                                  if replace_material else old.coverage[pair])
                locks[pair] = (m.locks(layout, owner, provider)
                               if replace_lock else old.locks[pair])
                if replace_material:
                    count += coverage[pair]
                if replace_lock:
                    blocked += locks[pair]
                    self.pair_updates += 1
            coverage_counts[owner], lock_counts[owner] = count, blocked
            available[owner] = ((count > 0) & (blocked == 0) &
                                np.isin(m.sources, m.allowed[owner]))
        state = ContactState(layout.copy(), coverage, locks, coverage_counts, lock_counts, available)
        self.cache[key] = state
        if self.base is None:
            self.base = state
        while len(self.cache) > 64:
            self.cache.popitem(last=False)
        return state

    def commit(self, layout):
        self.base = self.state(layout)


@dataclass
class VolumeState:
    layout: object
    parts: dict
    coverage_counts: np.ndarray
    lock_counts: np.ndarray
    material: np.ndarray
    volume_cm3: float


class VolumeDelta(VolumeGuidance):
    def __init__(self, model, layouts, power=15):
        super().__init__(model, layouts, power)
        self.base = None
        self.states = OrderedDict()
        self.sweep_cache = OrderedDict()
        self.epoch = 0
        self.changed_samples = 0
        self.seconds = 0.

    def ensure_bounds(self, layouts):
        began = time.monotonic()
        bounds = []
        for layout in layouts:
            for k in layout.active:
                points = C.transform_points(self.model.mesh.vertices, layout.placements[k])
                bounds.extend([points.min(0)-self.model.thickness, points.max(0)+self.model.thickness])
        bounds = np.asarray(bounds)
        low, high = bounds.min(0), bounds.max(0)
        if np.all(low >= self.low-1e-12) and np.all(high <= self.high+1e-12):
            self.seconds += time.monotonic()-began
            return
        # Rebase all candidates together on the SAME Sobol points. Old sample
        # volume numbers are guidance only, not comparable across epochs.
        old = self.base.layout.copy() if self.base is not None else None
        unit = (self.points-self.low)/(self.high-self.low)
        self.low, self.high = np.minimum(self.low,low), np.maximum(self.high,high)
        self.points = self.low+unit*(self.high-self.low)
        self.box_cm3 = float(np.prod(self.high-self.low)*1e6)
        self.cache.clear(); self.states.clear(); self.sweep_cache.clear()
        self.base = None
        self.epoch += 1
        if old is not None:
            self.base = self.state(old)
        self.seconds += time.monotonic()-began

    def sweep_mask(self, layout, k):
        q = layout.placements[k]
        d = q[:3, :3].T @ layout.directions[k]
        key = (np.round(q, 11).tobytes(), np.round(d, 11).tobytes())
        if key in self.sweep_cache:
            self.sweep_cache.move_to_end(key)
            return self.sweep_cache[key]
        points = C.transform_points(self.points, np.linalg.inv(q))
        # Bounding-box rejection is conservative, and includes both endpoints.
        bounds = self.model.mesh.bounds
        displacement = self.model.length*d
        low = bounds[0]+np.minimum(displacement, 0)
        high = bounds[1]+np.maximum(displacement, 0)
        ids = np.flatnonzero(np.all((points >= low) & (points <= high), axis=1))
        locked = np.zeros(len(points), bool)
        if len(ids):
            locations, rays, _ = self.model.ray.intersects_location(
                points[ids], np.tile(-d, (len(ids), 1)), multiple_hits=False)
            distance = (points[ids[rays]]-locations) @ d
            hit = rays[(distance >= 0) & (distance <= self.model.length)]
            locked[ids[hit]] = True
        self.sweep_cache[key] = locked
        while len(self.sweep_cache) > 256:
            self.sweep_cache.popitem(last=False)
        return locked

    def state(self, layout):
        key = layout.key()
        if key in self.states:
            self.states.move_to_end(key)
            return self.states[key]
        old = self.base
        before = set(old.layout.active) if old is not None else set()
        after = set(layout.active)
        count = (np.zeros(len(self.points), np.int16) if old is None else old.coverage_counts.copy())
        blocked = (np.zeros(len(self.points), np.int16) if old is None else old.lock_counts.copy())
        parts = {}
        for k in before | after:
            replace = k not in before or k not in after or not np.array_equal(
                old.layout.placements[k], layout.placements[k])
            replace_sweep = replace or not np.array_equal(old.layout.directions[k], layout.directions[k])
            if k in before:
                wrap, body, work, sweep = old.parts[k]
                if replace:
                    count -= wrap; blocked -= body; blocked -= work
                if replace_sweep:
                    blocked -= sweep
            if k not in after:
                continue
            if replace:
                wrap, body, work = (self.contains(layout, k, kind) for kind in ['wrap', 'body', 'work'])
                count += wrap; blocked += body; blocked += work
            if replace_sweep:
                sweep = self.sweep_mask(layout, k)
                blocked += sweep
            parts[k] = wrap, body, work, sweep
        material = (count > 0) & (blocked == 0)
        state = VolumeState(layout.copy(), parts, count, blocked, material,
                            float(material.mean()*self.box_cm3))
        if old is not None:
            self.changed_samples += int(np.count_nonzero(material != old.material))
        self.states[key] = state
        if self.base is None:
            self.base = state
        while len(self.states) > 64:
            self.states.popitem(last=False)
        return state

    def estimate(self, layout):
        began = time.monotonic()
        volume = self.state(layout).volume_cm3
        self.seconds += time.monotonic()-began
        return volume

    def delta(self, layout):
        trial = self.state(layout)
        added = trial.material & ~self.base.material
        removed = self.base.material & ~trial.material
        unit = self.box_cm3/len(self.points)
        return dict(added_cm3=float(added.sum()*unit), removed_cm3=float(removed.sum()*unit),
                    net_cm3=float((added.sum()-removed.sum())*unit), epoch=self.epoch)

    def commit(self, layout):
        self.base = self.state(layout)
