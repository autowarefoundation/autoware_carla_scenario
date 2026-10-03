# Autoware CARLA Scenario

A scenario-testing framework for validating [Autoware](https://www.autoware.org/)
on the [CARLA](https://carla.org/) simulator. A Hydra-driven scenario runner
loads an OpenDRIVE map into CARLA, drives an Autoware ego vehicle, evaluates
pass/fail conditions, and records video/JSON results. It ships a FastAPI viewer
for browsing runs and a FastAPI editor for authoring scenarios and exporting
them as reproducible packages.

The maps are typically produced by
[`autoware_lanelet2_to_opendrive`](https://github.com/hakuturu583/autoware_lanelet2_to_opendrive),
which converts Lanelet2 `.osm` maps to OpenDRIVE. This framework used to live in
that repository as a second workspace member, but it does not depend on the
converter: a Lanelet2 map and an OpenDRIVE map are loaded side by side as two
independent coordinate systems, related only through CARLA world coordinates.
Any OpenDRIVE file that describes the same place as the Lanelet2 map will do.

## Repository layout

```
.
├── autoware_carla_scenario/          # The framework package (uv workspace member)
├── carla_wheels/                     # Vendored CARLA Python wheels resolved by uv
├── data/                             # nishishinjuku fixture map used by the tests and examples
├── examples/scenario_package_template/  # A standalone scenario package to copy
├── .github/actions/                  # Composite actions (incl. scenario image packing)
├── pyproject.toml                    # uv workspace root
└── uv.lock
```

## Quick start

Python 3.10 to 3.12. Every dependency installs from a wheel or from git — no apt
packages, no C++ toolchain, no container. `git` has to be on `PATH`, because
uv clones the converter.

```bash
uv sync --dev --extra carla            # CARLA 0.10.0 (default, UE5)
uv sync --dev --extra carla-0-9-16     # CARLA 0.9.16 (legacy, UE4)
```

Run a scenario against the bundled nishishinjuku map. Its OpenDRIVE file,
`data/nishishinjuku_carla.xodr`, is committed next to the Lanelet2 one; it was
generated once with the converter's `convert` CLI, which this repository does
not install:

```bash
uv run scenario scenario=intersection_passing/straight map=nishishinjuku
```

Before it runs, every scenario is compiled with [Codon](https://github.com/exaloop/codon)
against a typed model of the framework, and one that does not type-check is
refused before CARLA starts. `uv run scenario-check` makes the same check without
running anything; see
[`docs/typecheck.md`](autoware_carla_scenario/docs/typecheck.md).

See [`autoware_carla_scenario/README.md`](autoware_carla_scenario/README.md) and
the documentation under [`autoware_carla_scenario/docs/`](autoware_carla_scenario/docs/)
for the full guide.

## Development

```bash
uv run pytest -n auto -o addopts= -m "not slow"   # what CI's fast job runs
uv run pre-commit run --all-files                 # what CI's lint job runs
```

## Releases

Releases are cut by `.github/workflows/release.yml` when a pull request is
merged into `master` with exactly one version bump label (`bump patch`,
`bump minor` or `bump major`). The workflow bumps `version` in
`autoware_carla_scenario/pyproject.toml`, re-locks, tags `v<version>`, creates
the GitHub Release with the documentation attached, and deploys the docs to
GitHub Pages. A merge without a label releases nothing.

If a labelled merge bumped the version but a later step failed, run the
workflow by hand (Actions → Release → Run workflow) with that version, e.g.
`2.63.0`. A manual run does not bump: it builds from the version's tag, or else
from its `chore: bump version to X` commit on `master`, so merges that landed
since do not leak in, and it reuses any tag, Release or asset the failed run
already created. Do not re-run the failed push run: it would bump a second time
from the old merge commit.

The version line continues from `autoware_lanelet2_to_opendrive`, where this
package started: it was split out at 2.62.0, so the first release here is
2.62.1. The workflow pushes with the `GH_PAT` repository secret.
