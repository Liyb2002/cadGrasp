"""Resume the same population at a committed round, including RNG and measure.

This is not a layout warm start. It is restricted to the connected-core
population, whose particle evidence consists of canonical meshes and arrays.
Native Boolean caches are deliberately rebuilt rather than round-tripped.
"""
from search_runtime import runtime, MISSING
import time

MODEL_FIELDS = ('adaptive_demands', 'quadrature_epoch', 'basis_caches',
    'adaptive_cache', 'connected_projection_cache', 'connected_contact_flags',
    'connected_core_cache', 'connected_core_repairs', 'force_cache', 'sample_cache',
    'proxy_force_cache', 'worst_cache', 'complete_sampled_states', 'lock_cache',
    'seed_cache', 'fixed_lock_cache', 'coordinate_cache', 'sample_calls',
    'proxy_calls', 'exact_calls', 'timing', 'gradient_seconds', 'adaptive_seconds')
SEARCH_FIELDS = ('events', 'process_rows', 'process_layouts', 'process_started',
    'frontier_statistics', 'material_selected', 'frontier_states',
    'frontier_witnesses', 'last_coordinate_selection', 'seat_hashes')


def state_without_model(value):
    return {k:v for k,v in vars(value).items() if k != 'model'}


def save_checkpoint(controller, phase, next_iteration, **extra):
    store = runtime(controller.model)
    if store is None:
        return
    m, s = controller.model, controller.search
    if not getattr(m, 'require_connected_core', False):
        raise ValueError('Exact population checkpoints require connected-core mode')
    state = dict(phase=phase, next_iteration=next_iteration, extra=extra,
        controller={k:v for k,v in vars(controller).items()
                    if k not in ('model', 'search', 'evaluator', '_resume') and not callable(v)},
        evaluator={k:v for k,v in vars(controller.evaluator).items()
                   if k not in ('model', 'search', 'builders')},
        model={k:getattr(m,k) for k in MODEL_FIELDS if hasattr(m,k)},
        search={k:getattr(s,k) for k in SEARCH_FIELDS if hasattr(s,k)},
        rng=s.rng.bit_generator.state,
        oracle={k:getattr(m.base_oracle,k) for k in
                ('cache','hulls','family_cache','constraint_cache','seconds','queries',
                 'component_queries','last_geometry','witness','core_clip_proposal',
                 'slice_cache','slice_cache_hits','slice_cache_misses',
                 'floor_rejections','slice_planes','equivalent_statistics')
                if hasattr(m.base_oracle,k) and k != 'constraint_cache'},
        elapsed_seconds=store.prior_elapsed+time.monotonic()-store.started,
        runtime_stats=store.stats,runtime_timings=store.timings,
        objects={k:state_without_model(getattr(m,k)) for k in
                 ('contact_delta','volume_delta','geometry_cache','demand_screen')
                 if getattr(m,k,None) is not None})
    # Keep large round states off the hot cache; they are for interruption only.
    store.put('checkpoint', 'latest', state)
    token = str(store._path('checkpoint','latest'))
    old = store.hot.pop(token, None)
    if old is not None:store.hot_bytes -= old[1]


def restore_checkpoint(controller):
    store=runtime(controller.model)
    if store is None or not getattr(controller.search,'continue_checkpoint',False):
        return None
    state=store.get('checkpoint','latest')
    if state is MISSING:
        raise ValueError('No intact same-execution population checkpoint')
    m,s=controller.model,controller.search
    store.prior_elapsed=state['elapsed_seconds'];store.started=time.monotonic()
    store.stats=state['runtime_stats'];store.timings=state['runtime_timings']
    vars(controller).update(state['controller'])
    if hasattr(controller.evaluated,'store'):controller.evaluated.store=store
    vars(controller.evaluator).update(state['evaluator'])
    vars(m).update(state['model'])
    vars(s).update(state['search'])
    s.rng.bit_generator.state=state['rng']
    vars(m.base_oracle).update(state['oracle'])
    for name,fields in state['objects'].items():
        if name=='volume_delta' and m.volume_delta is None:
            from whole_search.delta_guidance import VolumeDelta
            m.volume_delta=object.__new__(VolumeDelta)
            m.volume_delta.model=m
        vars(getattr(m,name)).update(fields)
    store.restored_checkpoint=state
    return state
