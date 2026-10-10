# autoware-carla-scenario-t4

Write a T4 driving scene down as an
[autoware_carla_scenario](../autoware_carla_scenario) scenario.

A converted T4 scene (the format TIER IV's `e2e-devkit` reads: `derived/meta.json`,
`derived/scalars.npz`, `derived/frames.pack`) is read into the ego's drive and one
trajectory per road user, and written out as a scenario document: the ego starts
where the recording's ego did, and every road user is an entity with a
*Follow Trajectory* card replaying its track. Open it in the Scenario Editor to add
the conditions the test is about.

```bash
uv run t4-scenario /data/t4/.../scene_0001 \
    --lanelet2 map/lanelet2_map.osm --map-group my_map --map-name MyMap \
    --to-editor
```

See `docs/trajectory.md` in the framework for the format, the track association
and the replay.
