# autoware-carla-scenario-t4

Writes the runs of [autoware_carla_scenario](https://github.com/autowarefoundation/autoware_carla_scenario)
as [T4 datasets](https://github.com/tier4/t4-devkit), TIER IV's nuScenes-style
format: the ego's poses, the boxes of every vehicle and pedestrian, and the
sweeps and frames of the LiDARs and cameras the recorder puts on the ego.

The writer is Rust (`src/writer.rs`); `autoware_carla_scenario_t4.T4Writer` is
its PyO3 binding, one abi3 wheel for CPython 3.10 and later.  Sensor files are
encoded and written on worker threads while the simulation goes on.

```bash
pip install 'autoware-carla-scenario[t4]'
uv run scenario ... t4.enabled=true
```

See `docs/t4_dataset.md` in autoware_carla_scenario for the layout, the
frames and the coordinate conventions.

## Development

```bash
cargo test          # the writer
uv run pytest autoware_carla_scenario/test/test_t4_dataset.py   # from the repository root
```
