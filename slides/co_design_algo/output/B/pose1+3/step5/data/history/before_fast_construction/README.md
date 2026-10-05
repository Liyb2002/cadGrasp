# Supporting files

The selected nearest-legal-floor result is `../overview.png` and `../shape.obj` (OBJ units: metres).

`report.json`, `geometry.npz`, `equilibrium.npz` and `independent_audit.json` hold the current geometry and verification. Replay with `audit_fixture.py --input <this data directory>`.

`generation_report.json` and `generation_audit.json` preserve the original generation evidence byte-for-byte. `layout.json` records all moves and original hashes. `files.json` maps the original logical filenames to their current locations relative to this directory.

`history/` contains the older loft experiment and reference models/reports. Their files and historical paths are retained unchanged; use the relocation mapping when interpreting old paths. No old pictures were regenerated.
