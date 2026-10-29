# Asset provenance

- `assets/so101/so101_new_calib.urdf` and `assets/so101/assets/*.stl` are copied unchanged from this repository's `codes/simulation/assets/so101/`. Its recorded upstream is [TheRobotStudio/SO-ARM100, Simulation/SO101](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101), Apache License 2.0. The package includes `assets/so101/LICENSE`. Copyright and source metadata already present in the URDF are retained.
- `assets/B.stl` is the unchanged project input `objects/B/mesh.stl`. SHA256: `91691a4e136ea6b36ebbf888df95a8d5ea9eb3be2a3fdc28dbb9c71d7066f72b`. The project metadata records its source as `milmillin/passive-gripper data/stl/B.stl`. This bundle does not add a new license grant for that input; it is included for the user's project handoff.
- `inputs/system_single/info.json` is an unchanged copy of the project's already generated SO101 planning package. Its role is to retain the actual source placement and coordinate/unit declarations for the demonstration.
- Numerical and mesh-collision packages are installed separately from `requirements.txt`; none of their implementations are vendored here.
