# Gradient objective audit (same 32 candidate starts)

Stalled incumbent: [492,32768,4101,32683]. Compared saved old surrogate descent and an initial reaction-reoptimized force-envelope prototype from identical raw candidate directions. Rank is minimum pose count then total count.

Old surrogate: 3 improved, 18 worsened, 11 unchanged. Initial force-envelope prototype: 5 improved, 1 worsened, 26 unchanged. Maximum analytic directional derivative relative error: 3.15e-8. Total comparison runtime: 18.23 seconds. Data: data/descent_comparison.json and data/run.log.

This prototype preceded trust-region overshoot/backtracking repair. The subsequent current force_descent_chain trial uses that repair and evaluates all raw/descent states; it is a different trajectory, not a same-start causal comparison of convergence. No complete fixture acceptance claimed.
