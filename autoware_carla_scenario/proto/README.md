# Protobuf definitions

`autoware_bridge/v0/autoware_bridge.proto` is the native `AutowareBridge` contract,
authored in this repository; its Python modules are generated into
`src/autoware_carla_scenario/autoware_bridge/_proto/`.

The alpasim `egodriver` contract (vendored from NVlabs/alpasim) lives with the package
that implements both of its ends,
[carla-driver-interface](https://github.com/hakuturu583/carla_driver_interface).

Regenerate every group with:

```bash
uv run python autoware_carla_scenario/scripts/compile_protos.py
```

`--check` verifies the committed output is up to date without writing anything; the same
check runs in `test_proto_generated.py`.
