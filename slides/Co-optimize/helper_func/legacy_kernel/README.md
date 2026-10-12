# Relocated numerical dependencies

`slides/baseline_algo` was deleted while Co-optimize still imported its contact,
equilibrium, withdrawal, solid and rendering utilities. This directory retains
the 93 Python modules actually loaded by the current search/render entry points,
plus the renderer's C++ source. It does not restore the deleted experiment or its
outputs.

Sources come from the preserved `proof_cold_v1/executed_sources` snapshot, with
missing dependencies read from the repository's HEAD. `relocation_manifest.json`
records original paths and SHA256 values, and relocated SHA256 values. The sole
numerical-source relocation adjustment changes `step1/needs.py`'s repository-root
lookup for its new depth. Contact/force/geometry mathematics are preserved.

`helper_func/_bootstrap.py` selects this dependency root only when the original
baseline root is absent. Historical initialization verification still compares
original expected source hashes; relocation does not authorize ignoring them.
