# autoware-carla-scenario-t4

Writes the runs of [autoware_carla_scenario](https://github.com/autowarefoundation/autoware_carla_scenario)
as [T4 datasets](https://github.com/tier4/t4-devkit), TIER IV's nuScenes-style
format: the ego's poses, the boxes of every vehicle and pedestrian, and the
sweeps and frames of the driver policy's LiDARs and cameras.

The writer is Rust (`src/writer.rs`); `autoware_carla_scenario_t4.T4Writer` is
its PyO3 binding, one abi3 wheel for CPython 3.10 and later.  It runs on every
key frame of a run, so the caller's thread only hands it the CARLA states (it
turns them into map-frame poses and boxes itself) and copies the sweeps into
reused buffers; worker threads write the files, count each box's points and
encode raw camera frames while the simulation goes on.

```bash
pip install 'autoware-carla-scenario[t4]'
uv run scenario ... t4.enabled=true   # the sensors are driver.lidars / driver.cameras
```

See `docs/t4_dataset.md` in autoware_carla_scenario for the layout, the
frames and the coordinate conventions.

## Development

```bash
cargo test          # the writer
uv run pytest autoware_carla_scenario/test/test_t4_dataset.py   # from the repository root
```
