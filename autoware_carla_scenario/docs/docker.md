# Packing a scenario package into a container image

A scenario package created with `scenario-new` is a normal Python
distribution, so it can be shipped as a container image that carries the
framework, the scenario, its configs, and its CARLA client --
[typesafe_carla](https://github.com/hakuturu583/typesafe_carla), whose released
wheel carries its CPython package **prebuilt**, so nothing is compiled at build
time or in a container.

The image is built by the `pack-scenario-image` composite action. Everything it
needs lives in the action's own directory, so it is meant to be referenced from
other repositories:

```yaml
- uses: hakuturu583/autoware_carla_scenario/.github/actions/pack-scenario-image@master
```

The ref is both the action version and the framework version that ends up in
the image — the runner checks the whole repository out alongside the action,
and the framework packages are built from that copy. Pin a release tag instead
of `main` for reproducible framework versions. Your own repository does not
have to contain this framework, or even be checked out.

## What the image contains

Only installed code:

| Present | Absent |
| --- | --- |
| `/opt/venv` — the framework, the scenario package, their dependencies (typesafe-carla, with its prebuilt CPython package, and its Codon toolchain among them) | Source trees, `pyproject.toml`, tests, docs, git history |
| `ffmpeg`, when asked for | `uv`, wheels, build caches, a C compiler, any apt package at all |

The entry point is the `scenario` CLI and the working directory is `/work`,
where Hydra writes its run directory.

`/opt/venv` arrives as [four layers](#layer-layout), not one, so a client that
already holds another image built the same way pulls only what actually
differs.

## Stage layout

`.github/actions/pack-scenario-image/Dockerfile` runs three stages, each
handing the next strictly less than it was given:

```mermaid
flowchart LR
    A["wheelhouse<br/>(Alpine + uv)<br/>framework/ + scenario/ → wheels"]
    B["venv<br/>(Debian slim)<br/>four installs → /export/&lt;layer&gt;,<br/>then slimmed"]
    C["runtime<br/>(Debian slim)<br/>four COPY --link layers,<br/>no apt layer"]
    A -- "*.whl + requirements.txt" --> B
    B -- "/export/* only" --> C
```

`wheelhouse` is Alpine because everything it touches is pure Python — `uv
build` emits `py3-none-any` wheels for all three packages — so the stage needs
neither glibc nor a compiler, and nothing but the wheels survives it.

The runtime is **not** Alpine, and cannot be. `typesafe-carla` and its Codon
toolchain, `opencv-python-headless` and `simple-lanelet2` publish manylinux
wheels only. There is no musllinux wheel to fall back to, so a musl runtime
cannot install them at all. Debian slim is the smallest base that runs the
code.

Neither the `venv` stage nor the runtime compiles anything, and neither
installs a C compiler: the typesafe-carla wheel on PyPI carries typesafe_carla's
CPython package prebuilt (`typesafe_carla/carla/_prebuilt`, one build for every
Python 3.10+), so installing the wheel is all the client needs. typesafe_carla
falls back to compiling that package on its first import (15-50 minutes, ~14 GB
of memory, `cc`) only where no matching prebuilt one exists -- a typesafe-carla
installed from source, or a typesafe-carla-toolchain other than the one the
wheel was built with -- and the image rules that out: the build imports the
client from the finished virtualenv with `TYPESAFE_CARLA_PYCARLA_BUILD=0`, and
the runtime sets the same variable, so a prebuilt package that does not apply
is an `ImportError`, at build time or on the first run, rather than a
half-hour compile inside a container.

## Layer layout

An image is pulled as a stack of layers, and a client downloads only the ones
it does not already have. A virtualenv shipped as a single `COPY` is a single
layer, so changing one line of a scenario means pulling all ~400 MB of it
again. The Dockerfile therefore installs `/opt/venv` in four steps and ships
each step as its own layer, ordered from the input that changes least often to
the one that changes on every commit:

| # | Layer | Holds | Rebuilt when |
| --- | --- | --- | --- |
| 1 | `client` | The virtualenv, typesafe-carla with its prebuilt CPython package (`import typesafe_carla.carla as carla`) and its Codon toolchain (typesafe-carla-toolchain), at the versions `uv.lock` pins | `uv.lock` moves typesafe-carla or its toolchain, or `python-version` or the base image changes |
| 2 | `deps` | The rest of the framework's third-party dependency closure — OpenCV, scipy, numpy, the matplotlib stack, … | `uv.lock`, `python-version` or the base image changes |
| 3 | `framework` | `autoware-carla-scenario` | Framework code changes |
| 4 | `scenario` | The scenario wheel, plus any dependency it adds of its own | The scenario package changes |

(Before the move to typesafe_carla the dependency layer was ~400 MB and the
two framework and scenario layers ~2 MB and ~80 kB; the client layer, with
the Codon toolchain in it, has not been measured in an image yet.)

The client comes first because it depends on nothing else: layer 1 installs
typesafe-carla alone, constrained to the lock's versions, so its content is a
function of the typesafe-carla and typesafe-carla-toolchain versions and the
base image and nothing more. Two images built for the same client share that
layer whatever else they contain, and a lock change that leaves the client
alone moves only the dependency layer behind it.

So editing a scenario and rebuilding transfers the fourth layer. Bumping the
framework transfers the third and fourth. A lock change moves the dependency
layer, and the client layer only when it moves typesafe-carla or its
toolchain. The push is incremental for the same reason: a registry that
already holds a blob is sent the new layer and nothing else.

### What makes it hold

Layer reuse is by digest, and a digest covers file metadata as well as content,
so two things could quietly defeat the split:

- **Install timestamps.** The same wheel installed twice produces the same
  bytes at different times, and that alone is a different layer.
  `venv-layer.sh` therefore stamps every exported file with one fixed timestamp
  (`LAYER_MTIME`, 2020-01-01 by default) as it captures it, and re-stamps
  afterwards whatever the slim pass rewrote. A rebuild from a cold cache
  reproduces all four layers byte for byte: every one of them, the client
  included, is installed from wheels, and nothing in the image is compiled.
- **The layers underneath.** A layer can only be reused together with its whole
  parent chain, so anything that varies has to sit *behind* the virtualenv
  rather than in front of it. The runtime stage therefore copies the four
  layers first and only then creates the user and installs `ffmpeg`: `apt`
  hands out whatever build it has today, and `useradd` writes the current date
  into `/etc/shadow`, which would otherwise put a fresh digest on every layer
  behind it. The copies use `COPY --link`, so each layer is built against an
  empty base and stays the same blob wherever it lands.

`slim-venv.py` runs once per layer rather than once over the virtualenv, so
stripping a shared object rewrites it inside the layer that installed it
instead of duplicating a stripped copy into a later one. The build then
reassembles the four trees exactly as the runtime stage stacks them and imports
the native extension modules from the result, so a file lost or shadowed by the
split fails the build rather than the container.

### Sharing the build cache

Layers are about the pull; `cache-scope` is the same idea for the build. It
defaults to one scope per image, but images that share a lock and framework
are the same build up to their scenario layer, so pointing several of them at
one scope lets the later ones restore the client and dependency stages instead
of resolving and installing them again. The action reports the scope
it used, so a second build in the same workflow can just name the first's:

```yaml
- uses: ./.github/actions/pack-scenario-image
  id: first
  with:
    scenario-package-path: packages/first_scenario
    image: ghcr.io/my-org/first-scenario

- uses: ./.github/actions/pack-scenario-image
  with:
    scenario-package-path: packages/second_scenario
    image: ghcr.io/my-org/second-scenario
    cache-scope: ${{ steps.first.outputs.cache-scope }}
```

## Keeping the image small

Two things do the work, measured on a scaffolded package with the official
CARLA 0.10.0 client (before the move to typesafe_carla, whose Codon toolchain
adds to these totals):

| | virtualenv |
| --- | --- |
| Naive install | 565 MB |
| `opencv-python-headless` instead of `opencv-python` | 512 MB |
| …plus the `slim` pass | **413 MB** |

- **Headless OpenCV.** The framework's only OpenCV calls are `imencode` and
  `pointPolygonTest`; there is no `imshow` or `namedWindow` anywhere, so the GUI
  build's GTK/GStreamer payload is dead weight. Dropping it also removes the
  *last* external shared-library dependency: every `.so` in the venv now needs
  nothing beyond glibc, `libgcc_s`, `libstdc++` and `libz`, all of which the
  base image already has. The runtime therefore installs no apt packages and
  carries no apt layer unless `with-ffmpeg` is on.
- **The `slim` pass** (on by default, `slim: "false"` to disable) drops
  byte-code caches, vendored test suites, C headers, Cython sources and type
  stubs (29 MB), then runs `strip --strip-unneeded` over every bundled shared
  object (64 MB) — the manylinux wheels for scipy, numpy and OpenCV ship
  unstripped. Stripping leaves the dynamic symbol table intact.

    GNU binutils up to 2.40 — the version in Debian bookworm — rewrites some
    objects into a layout the kernel's ELF loader rejects with *"ELF load
    command address/offset not page-aligned"*; scipy's bundled OpenBLAS is one.
    So `slim-venv.py` re-reads every object it strips and restores the original
    unless each `PT_LOAD` segment still satisfies the loader's congruence rule.
    A newer binutils strips everything and restores nothing; on bookworm the
    one restored file costs about 2 MB. The build then imports the native
    extension modules and fails right there if anything did break.
- **Stage separation.** Wheels are built in a throwaway Alpine stage and the
  virtualenv in a throwaway Debian stage, so no source tree, wheel, build cache
  or copy of `uv` reaches the final image.

The [layer layout](#layer-layout) does not change any of these totals — the
same bytes are simply split across four layers instead of one, so that a
rebuild transfers only the layers that changed.

What is left is dominated by dependencies with no smaller variant: OpenCV,
scipy, numpy, and the matplotlib stack that `pyxodr` requires.

## Pinning the CARLA client — and everything else

The client is the `typesafe-carla` that `uv.lock` pins (CARLA UE5: the wheel's
LibCarla is built from a CARLA ref, which the build prints), with the CPython
package that wheel carries prebuilt, loaded before the image is finished. Nothing in the image
can change it afterwards. The default tags are `:latest` and the commit
(`:<sha>`); to see which client an image carries:

```bash
docker run --rm --entrypoint python ghcr.io/tier4/my-scenario:latest -c \
  "from importlib.metadata import version; from typesafe_carla import paths; print(version('typesafe-carla'), paths.native_info()['carla_git_ref'])"
```

The whole dependency tree is pinned, so the same ref rebuilt months
apart gives the same image: the build context carries `uv.lock`, the wheelhouse
stage exports it as a requirements file with `uv export --frozen`, and that
file is what constrains the client install, what the dependency layer
installs, and what constrains the two installs after it. This buys reproducibility rather than size — and, with the
timestamps normalised, it is what lets an unchanged dependency layer keep its
digest across rebuilds. Constraints only bound what is resolved, so a scenario
package that brings dependencies of its own still installs — they are the one
part that floats, and they land in the scenario layer.

## In a workflow

```yaml
jobs:
  pack:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Pack the scenario package
        uses: ./.github/actions/pack-scenario-image
        with:
          scenario-package-path: packages/my_scenario_package
          image: ghcr.io/${{ github.repository_owner }}/my-scenario
          push: "true"
```

The scenario package does not have to live in this repository — check it out
anywhere on the runner and point `scenario-package-path` at it.

### Inputs

`action.yml` carries the full list with defaults; these are the ones worth
explaining further:

| Input | Meaning |
| --- | --- |
| `scenario-package-path` | The generated package — the directory with its `pyproject.toml`. Required, and may sit outside this repository. |
| `framework-path` | uv workspace root supplying the framework. Defaults to the action's own checkout, which is what makes the pinned ref the framework version. Its members are read from `[tool.uv.workspace] members`, globs included. |
| `with-ffmpeg` | Installs ffmpeg for `CameraRecorder`. The only input that adds an apt layer. |
| `slim` | Strips the virtualenv (see above). On by default. |
| `platforms` | Target platforms. Empty (the default) builds for the runner's own: `linux/amd64` on an x86_64 runner, `linux/arm64` on an arm64 one such as `ubuntu-24.04-arm` — see [Architectures](#architectures). |
| `cache-scope` | Cache key namespace. Defaults to one scope per image and platform, so images built in the same workflow do not evict each other — give two images the same scope to [share it](#sharing-the-build-cache) instead. |

Outputs: `image-ref`, `tags`, `digest` (pushed images only), `cache-scope`.

### Architectures

typesafe-carla and its Codon toolchain ship wheels for Linux x86_64 and
aarch64, and so does every other native dependency of the image, so the action
packs for either. Run it on an arm64 runner to get an arm64 image:

```yaml
jobs:
  pack:
    strategy:
      matrix:
        runner: [ubuntu-latest, ubuntu-24.04-arm]
    runs-on: ${{ matrix.runner }}
```

Each leg builds, loads and smoke-tests its own image natively. To publish one
multi-architecture tag, push each leg under its own tag and join them with
`docker buildx imagetools create`. Cross-building on one runner
(`platforms: linux/amd64,linux/arm64` after `docker/setup-qemu-action`) also
works, but emulated, and a multi-platform build cannot be loaded, so it needs
`load: "false"` and `smoke-test: "false"`.

### The smoke test

With `smoke-test: "true"` the action runs its `smoke-test.py` inside the
freshly built image and checks that

1. typesafe_carla's prebuilt CPython package loads without compiling
   anything (`TYPESAFE_CARLA_PYCARLA_BUILD=0`), and the official `carla`
   package is not installed,
2. the scenario package's `autoware_carla_scenario.scenarios` entry point loads
   and registers at least one scenario,
3. Hydra discovers `AutowareScenarioSearchPathPlugin`, so the package's
   `conf/` directory reaches the config search path, and
4. `scenario --help` composes a config without a CARLA server.

Check 3 is worth its keep: `hydra_plugins` is a namespace package that an
editable install resolves via `src/`, so losing it from the wheel breaks every
`scenario=<name>/…` override in the image while leaving development untouched.

## Building by hand

```bash
ACTION=.github/actions/pack-scenario-image

# 1. Assemble a minimal build context (framework + scenario, nothing else).
"$ACTION/assemble-context.sh" \
  --scenario ../my_scenario_package \
  --out /tmp/scenario-ctx

# 2. Build. Nothing is compiled: the CARLA client's CPython package comes
#    prebuilt in the typesafe-carla wheel.
docker build \
  --file "$ACTION/Dockerfile" \
  --tag my-scenario:latest \
  /tmp/scenario-ctx

# 3. Verify.
docker run --rm -i --entrypoint python \
  my-scenario:latest - < "$ACTION/smoke-test.py"
```

## Running a scenario

The image holds the client, not the simulator, so point it at a CARLA server
and give it the map files:

```bash
docker run --rm \
  --network host \
  --volume "$PWD/maps:/maps:ro" \
  --volume "$PWD/outputs:/work/outputs" \
  --env NISHISHINJUKU_XODR_PATH=/maps/nishishinjuku_carla.xodr \
  --env NISHISHINJUKU_LANELET2_PATH=/maps/nishishinjuku.osm \
  my-scenario:latest \
  scenario=my_scenario/default map=nishishinjuku
```

Without `--network host`, pass the server address explicitly with
`server.host=<address> server.port=2000`. Hydra writes under the working
directory, so `/work` needs a volume for the results to outlive the container —
the image runs as UID 1000, which must be able to write to it.
