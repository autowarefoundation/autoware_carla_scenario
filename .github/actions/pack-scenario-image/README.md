# `pack-scenario-image`

Build a container image for a scenario package generated with `scenario-new`,
with its CARLA client -- typesafe_carla, compiled for CPython -- built in.

This directory is self-contained — `action.yml`, the `Dockerfile`, the
build-context assembler and the smoke test all live here — so the action is
meant to be referenced from other repositories:

```yaml
- uses: actions/checkout@v4

- uses: hakuturu583/autoware_carla_scenario/.github/actions/pack-scenario-image@master
  with:
    scenario-package-path: my_scenario_package
    image: ghcr.io/my-org/my-scenario
    push: "true"
```

Your repository needs to contain only your scenario package: the framework
comes from the ref you reference the action by. Push to a registry with
`docker/login-action` first; this action does not handle credentials.

The virtualenv ships as four layers -- the framework's dependency closure
(typesafe-carla included), typesafe_carla's compiled CPython package (a
~15 minute, ~8 GB compile at build time, so a container never does it), the
framework wheels, the scenario -- ordered so that rebuilding a scenario
against an unchanged lock and framework leaves the first three untouched, and
pulling the new image transfers only the last.

Inputs, outputs, sizing, and how to build the same image by hand are documented
in [`autoware_carla_scenario/docs/docker.md`](../../../autoware_carla_scenario/docs/docker.md).
