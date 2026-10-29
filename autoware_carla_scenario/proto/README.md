# Protobuf definitions

`autoware_bridge/v0/autoware_bridge.proto` is the native `AutowareBridge` contract,
authored in this repository; its Python modules are generated into
`src/autoware_carla_scenario/autoware_bridge/_proto/`.

`t4_e2e_devkit/t4_scene.proto` is the T4Scene exchange contract of
[tier4/e2e-devkit](https://github.com/tier4/e2e-devkit) (contract 0.1.0), copied byte for
byte from its `proto/t4_scene.proto`; its Python modules are generated into
`src/autoware_carla_scenario/t4_scene/_proto/` for the trajectory log's T4Scene export
([docs/t4_scene.md](../docs/t4_scene.md)). It is compiled from its own directory so the
descriptor keeps the devkit's name, `t4_scene.proto`, and both bindings load into one
process. Update it by copying the devkit's file again; do not edit it here.

The alpasim `egodriver` contract (vendored from NVlabs/alpasim) lives with the package
that implements both of its ends,
[carla-driver-interface](https://github.com/hakuturu583/carla_driver_interface).

Regenerate every group with:

```bash
uv run python autoware_carla_scenario/scripts/compile_protos.py
```

`--check` verifies the committed output is up to date without writing anything; the same
check runs in `test_proto_generated.py`.
