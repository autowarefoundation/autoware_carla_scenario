# Protobuf definitions

`autoware_bridge/v0/autoware_bridge.proto` is the native `AutowareBridge` contract,
authored in this repository; its Python modules are generated into
`src/autoware_carla_scenario/autoware_bridge/_proto/`.

`t4_e2e_devkit/t4_scene.proto` is the T4Scene exchange contract of
tier4/e2e-devkit (TIER IV's repository, not public; contract 0.1.0), copied byte for
byte from its `proto/t4_scene.proto`; its Python modules are generated into
`src/autoware_carla_scenario/t4_scene/_proto/` for the trajectory log's T4Scene export
([docs/t4_scene.md](../docs/t4_scene.md)). It is compiled from its own directory so the
descriptor keeps the devkit's name, `t4_scene.proto`, and both bindings load into one
process. Update it by copying the devkit's file again; do not edit it here.

`pedestrian_motion/v0/pedestrian_motion.proto` is the `PedestrianMotion` contract
between `SmartPedestrianControllerAction` (the client) and a pedestrian motion
generator (the server, e.g. [carla_ardy_backend](https://github.com/hakuturu583/carla_ardy_backend)),
authored in this repository; its Python modules are generated into
`src/autoware_carla_scenario/pedestrian/_proto/`. A generator copies the file byte for
byte; change its contract version with every change.

`waymo_open_dataset/` is the Waymo Open Sim Agents Challenge submission contract,
`waymo_open_dataset/protos/sim_agents_submission.proto`, with the protos it imports
(`scenario.proto` for `Track.ObjectType`, and theirs), copied byte for byte from
[waymo-research/waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset)
at commit `99a4cb3ff07e2fe06c2ce73da001f850f628e45a` (`src/waymo_open_dataset/`, laid
out as there; Apache-2.0, its `LICENSE` beside them). Its Python modules are generated
into `src/autoware_carla_scenario/wosac/_proto/` for the trajectory log's WOSAC export
([docs/wosac.md](../docs/wosac.md)). It is compiled from that directory so every
descriptor keeps its upstream name and package (`waymo.open_dataset`), and the
`waymo-open-dataset` package's bindings load into one process beside these when its
protos are the same revision. Update it
by copying the files again; do not edit them here.

The alpasim `egodriver` contract (vendored from NVlabs/alpasim) lives with the package
that implements both of its ends,
[carla-driver-interface](https://github.com/hakuturu583/carla_driver_interface).

Regenerate every group with:

```bash
uv run python autoware_carla_scenario/scripts/compile_protos.py
```

`--check` verifies the committed output is up to date without writing anything; the same
check runs in `test_proto_generated.py`.
