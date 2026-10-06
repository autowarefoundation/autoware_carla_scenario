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
uv sync --dev
```

The CARLA client is [typesafe_carla](https://github.com/hakuturu583/typesafe_carla)
(CARLA UE5: 0.10.0 and ue5-dev; Linux x86_64), imported as
`import typesafe_carla.carla as carla`. Its PyPI wheel carries the CPython
package prebuilt, so the sync above is the whole install; see
[installation](autoware_carla_scenario/docs/installation.md) for the cases that
build it instead. The official `carla` wheels, and CARLA 0.9.16 (UE4), are no
longer used.

Run a scenario against the bundled nishishinjuku map. Its OpenDRIVE file,
`data/nishishinjuku_carla.xodr`, is committed next to the Lanelet2 one; it was
generated once with the converter's `convert` CLI, which this repository does
not install:

```bash
uv run scenario scenario=intersection_passing/straight map=nishishinjuku
```

Before it runs, every scenario is compiled with [Codon](https://github.com/exaloop/codon)
against a typed model of the framework and typesafe_carla's CARLA API, and one that does
not type-check is refused before CARLA starts. `uv run scenario-check` makes the same check without
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

## Installation from PyPI

Released versions are published to PyPI as
[`autoware-carla-scenario`](https://pypi.org/project/autoware-carla-scenario/):

```bash
uv add autoware-carla-scenario          # or: pip install autoware-carla-scenario
uv add "autoware-carla-scenario[sumo]"  # with the SUMO traffic backend
```

## Releases

Releases are cut by [`.github/workflows/release.yml`](.github/workflows/release.yml),
modelled on [simple_lanelet2](https://github.com/hakuturu583/simple_lanelet2)'s.
When a pull request is merged into `master` with exactly one version bump label
(`bump patch`, `bump minor` or `bump major`), one run of the workflow:

1. bumps `version` in `autoware_carla_scenario/pyproject.toml` and re-locks
   (`tools/bump_version.py`), commits `chore(release): vX.Y.Z` to `master`,
   and creates the tag `vX.Y.Z` and the GitHub Release with generated notes;
2. re-runs the fast test suite on that commit and checks the tag matches the
   package version;
3. builds the wheel and sdist, checks their metadata, and installs the wheel
   clean;
4. publishes them to PyPI with Trusted Publishing (no API token is stored);
5. attaches the documentation as PDF and zip to the Release and deploys it to
   [GitHub Pages](https://autowarefoundation.github.io/autoware_carla_scenario/)
   with `mike`.

A merge without a label releases nothing. A hand-pushed `v*` tag is tested,
built, published and documented the same way, without a bump.

**Recovering a failed release.** Run the workflow by hand (Actions → Release →
Run workflow) on `master` with the version, e.g. `3.1.3`. It does not bump: it
builds from the version's tag, or else from its `chore(release): vX.Y.Z`
commit, so merges that landed since do not leak in, and it reuses any tag,
Release, PyPI file or docs version the failed run already made. Do not re-run
the failed push run: it would bump a second time. Run it with the version
empty for a dry run that tests and builds the selected ref and publishes
nothing.

**One-time setup** of the repository:

- PyPI → Publishing → add a (pending) trusted publisher: owner
  `autowarefoundation`, repository `autoware_carla_scenario`, workflow
  `release.yml`, environment `pypi`. Create the `pypi` environment under the
  repository's Settings → Environments.
- Create the `bump patch`, `bump minor` and `bump major` labels.
- Settings → Pages: deploy from the `gh-pages` branch (created by the first
  release).
- If `master` has a ruleset that github-actions[bot] cannot bypass, add a
  `GH_PAT` secret that can push to it; the workflow uses it for the bump
  commit only, and `GITHUB_TOKEN` otherwise.

The version line continues from `autoware_lanelet2_to_opendrive`, where this
package started: it was split out at 2.62.0, so the first release was 2.62.1.
Development moved from `hakuturu583/autoware_carla_scenario` to
`autowarefoundation/autoware_carla_scenario` after 3.1.2.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
