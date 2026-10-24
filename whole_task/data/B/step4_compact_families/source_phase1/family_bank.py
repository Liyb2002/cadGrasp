"""Recombine certified fixtures without freezing task membership too early.

The bank contains actual, immutable fixtures, not hypothetical task clusters.
An exact cover picks disjoint fixtures covering every original task once.
Changing the cover never modifies a mesh or assigns an untested task to it.
"""
from functools import lru_cache
import numpy as np


def mask(group):
    return sum(1 << int(i) for i in group['members'])


def register(state, group):
    bank=state.setdefault('family_bank',{})
    bank[group['sha256']]=group


def covers(bank,task_count,rng=None,forced=None):
    """Minimum-volume exact cover for each possible fixture count.

Random costs diversify *exploration* only. Published costs always use actual
mesh volumes. Keep three material variants per member set for exploration;
all variants and their immutable ancestry remain in the bank.
"""
    buckets={}
    for group in bank.values():buckets.setdefault(mask(group),[]).append(group)
    choices=[]
    for key,variants in sorted(buckets.items()):
        variants.sort(key=lambda g:(g['volume_cm3'],g['sha256']))
        if rng is None:g=variants[0];cost=g['volume_cm3']
        else:
            variants=variants[:3]
            g=variants[int(rng.integers(len(variants)))]
            cost=g['volume_cm3']*float(rng.uniform(.65,1.35))
        choices.append((key,g,cost))
    by_task=[[entry for entry in choices if entry[0]&(1<<i)] for i in range(task_count)]
    full=(1<<task_count)-1
    @lru_cache(None)
    def solve(remaining):
        if not remaining:return {0:(0.,())}
        first=(remaining&-remaining).bit_length()-1;result={}
        for bits,g,cost in by_task[first]:
            if bits&remaining!=bits:continue
            for count,(tail,groups) in solve(remaining^bits).items():
                count+=1;total=cost+tail
                if count not in result or total<result[count][0]:result[count]=(total,(g,)+groups)
        return result
    if forced is None:raw=solve(full)
    else:raw={n+1:(v+forced['volume_cm3'],(forced,)+gs)
        for n,(v,gs) in solve(full^mask(forced)).items()}
    return {n:list(value[1]) for n,value in raw.items()}


def best_cover(state,task_count):
    options=covers(state['family_bank'],task_count)
    eligible=[n for n in options if n<=2]
    n=(min(eligible,key=lambda n:sum(g['volume_cm3'] for g in options[n]))
       if eligible else min(options))
    return options[n]


def explore_cover(state,task_count,rng):
    """Replace a stale partition using certified alternatives from all samples."""
    merged=[g for g in state['family_bank'].values() if len(g['members'])>1]
    forced=None
    if merged and rng.random()<.65:forced=merged[int(rng.integers(len(merged)))]
    options=covers(state['family_bank'],task_count,rng=rng,forced=forced)
    counts=sorted(options)
    # Occasionally keep one additional independent fixture, allowing an old
    # family to be split into certified ancestors and paired differently.
    n=counts[min(int(rng.random()<.25),len(counts)-1)]
    return options[n]
