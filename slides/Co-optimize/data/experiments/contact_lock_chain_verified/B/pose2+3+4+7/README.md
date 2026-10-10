# Incremental contact-lock chain trial

One persistent chain; 10 rounds × 32 cheap candidates; top 1 finalist checked per round. Search runtime: **5.09 seconds**, excluding constructor. Final counts, pose 2/3/4/7: **[492, 32768, 4101, 32683]**. Unresolved; no final fixture or clearance acceptance is claimed.

Contact update uses per-pose locks and restores a contact only when all locks are cleared. Fixed seed contact vertices are checked using local normal compatibility and continuous reverse-ray obstruction. No swept support meshes or support Boolean operations occur in search. Whole-patch/contact-core/1% clearance certification is outside this contact-point model.

First 3 rounds improve; subsequent 7 rounds reject the checked candidate and preserve the incumbent. Ranking only one finalist by an approximate frozen surrogate can stagnate. Runtime does not prove convergence.

Records: data/report.json, data/candidate_timing.json, data/chain_trajectory.json, data/contact_locks.npz and data/run.log. Original loads and published outputs are preserved.
