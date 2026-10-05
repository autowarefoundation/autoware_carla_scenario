# syntax=docker/dockerfile:1.7
#
# Container image for a scenario package generated with `scenario-new`.
# Build context: the trimmed tree that assemble-context.sh next to this file
# produces (`framework/` = the uv workspace, `scenario/` = the package).
#
# Wheels are built on Alpine (everything here is pure Python), installed into a
# throwaway Debian stage, and only the finished virtualenv reaches the runtime.
# The runtime cannot be Alpine -- typesafe-carla, its Codon toolchain,
# opencv-python-headless and simple-lanelet2 publish manylinux wheels only.
#
# The CARLA client is typesafe_carla (`import typesafe_carla.carla as carla`),
# whose released wheel carries its CPython package prebuilt: nothing is
# compiled here or in a container.
#
# The virtualenv does not reach the runtime as one lump. It is installed in
# four steps and shipped as four layers, ordered from the input that changes
# least often to the one that changes on every commit:
#
#   1. client     the virtualenv, typesafe-carla and its Codon toolchain,
#                 at the versions uv.lock pins           <- typesafe-carla
#   2. deps       the framework's third-party closure    <- uv.lock
#   3. framework  the workspace wheels                   <- framework sources
#   4. scenario   the scenario wheel and what it adds    <- scenario package
#
# So an image rebuilt for an unchanged client and framework reproduces layers
# 1-3 byte for byte, and a client that already holds any image built the same
# way pulls the scenario layer alone -- a few hundred kB rather than the whole
# virtualenv. venv-layer.sh explains how the split is made to hold across
# rebuilds.
#
# See autoware_carla_scenario/docs/docker.md for the reasoning and the sizes.

ARG PYTHON_VERSION=3.10
ARG UV_VERSION=0.9.7
ARG RUNTIME_BASE=python:${PYTHON_VERSION}-slim-bookworm


# ---------------------------------------------------------------------------
# Stage 1: wheelhouse -- Alpine, pure Python, no native toolchain
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-alpine AS wheelhouse

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# The framework is pinned by the action ref, so it changes far less often than
# the scenario package; building it first keeps its layer cached across
# scenario edits. `--all-packages` builds whatever workspace members the
# assembler copied, so the member list lives in exactly one place. The two
# wheel sets stay in separate directories because they are installed
# separately, into separate layers of the runtime image.
COPY framework/ framework/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --all-packages --out-dir /wheels/framework framework \
    && uv export --directory framework --frozen --no-dev --no-emit-workspace \
        --no-hashes -o /wheels/requirements.txt

COPY scenario/ scenario/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir /wheels/scenario scenario


# The uv binary is lifted out of its own distroless image rather than
# installed, so nothing about it lingers in a layer of the runtime.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv


# ---------------------------------------------------------------------------
# Stage 2: venv -- install /opt/venv in four steps, one per runtime layer
# ---------------------------------------------------------------------------
FROM ${RUNTIME_BASE} AS venv

# Set to anything but "true" to keep debug symbols and vendored test suites --
# useful when debugging a native crash inside the image with gdb.
ARG SLIM=true
# Timestamp stamped on every file that reaches the runtime image, in place of
# the time it happened to be installed. Layer digests are computed over file
# metadata as well as content, so without this an unchanged layer would be a
# fresh download on every rebuild. Declared without a value so that the default
# lives in venv-layer.sh alone; build args reach that script as environment.
ARG LAYER_MTIME

# `strip` comes from binutils. `git` is there for a framework the lock pins to
# a commit of its repository rather than to an index, which uv has to clone. Installed before anything volatile is copied in so the apt layer
# caches across every rebuild; the stage is discarded, so installing either
# costs the image nothing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends binutils ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv
COPY venv-layer.sh /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=wheelhouse /wheels/requirements.txt /wheels/requirements.txt

# --- layer 1: the virtualenv and the CARLA client --------------------------
# typesafe-carla (with its prebuilt CPython package) and its Codon toolchain,
# on their own and before anything else, at the versions the lock pins (the
# constraints file is the lock): they depend on nothing else, so the layer is
# a function of those two versions and the base image alone, and two images
# built for the same client share it whatever else they contain.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python /usr/local/bin/python3 /opt/venv \
    && uv pip install --python /opt/venv/bin/python \
        --constraint /wheels/requirements.txt \
        typesafe-carla \
    && sh /usr/local/bin/venv-layer.sh capture /opt/venv /export client

# --- layer 2: the framework's third-party dependency closure ---------------
# The requirements file is the workspace lock exported without its own members,
# so this installs exactly the third-party versions this repository tests --
# and changes only when uv.lock does, not when framework code does.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        --requirement /wheels/requirements.txt \
    && sh /usr/local/bin/venv-layer.sh capture /opt/venv /export deps

# --- layer 3: the framework wheels -----------------------------------------
# Their dependencies are already in place from the lock, so this adds the
# workspace packages and nothing else: a framework change moves a few hundred
# kB, not the dependency tree. The constraints keep a resolution that somehow
# is not satisfied from picking a version the lock does not name.
COPY --from=wheelhouse /wheels/framework /wheels/framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        --constraint /wheels/requirements.txt \
        --find-links /wheels/framework \
        /wheels/framework/*.whl \
    && sh /usr/local/bin/venv-layer.sh capture /opt/venv /export framework

# --- layer 4: the scenario package -----------------------------------------
# Constraints only bound what gets resolved, so a scenario package that pulls
# in extra dependencies of its own still installs -- they land here, in the
# layer that is expected to change anyway.
COPY --from=wheelhouse /wheels/scenario /wheels/scenario
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        --constraint /wheels/requirements.txt \
        --find-links /wheels/framework \
        /wheels/scenario/*.whl \
    && sh /usr/local/bin/venv-layer.sh capture /opt/venv /export scenario

# Drops what only a test runner, compiler or type checker reads, then strips
# the bundled shared objects. Run per layer rather than over the virtualenv, so
# that stripping a shared object rewrites it in the one layer that owns it
# instead of duplicating it into a later one. The script verifies each stripped
# object and puts the original back if binutils produced a layout the ELF
# loader would reject -- see its docstring; scipy's bundled OpenBLAS is one
# such object under the binutils in Debian bookworm.
COPY slim-venv.py /usr/local/bin/
RUN set -eu; \
    if [ "${SLIM}" = "true" ]; then \
        for layer in /export/*/; do \
            echo "== ${layer}"; \
            /opt/venv/bin/python /usr/local/bin/slim-venv.py "${layer}"; \
        done; \
    fi

# Reassemble the layers exactly as the runtime stage stacks them and check the
# result, so a file lost or shadowed by the split fails the build here rather
# than at run time. The reassembled tree is also the last chance to notice that
# the client's prebuilt CPython package does not load
# (TYPESAFE_CARLA_PYCARLA_BUILD=0: a missing or mismatched one is an
# ImportError here, not a 15-minute compile), or that slimming broke an
# extension module.
#
# It is built on a tmpfs over the virtualenv the installs left behind: at the
# path the venv expects, since that is what its scripts and pyvenv.cfg name,
# but without committing a third copy of it to this stage.
RUN --mount=type=tmpfs,target=/opt/venv \
    sh /usr/local/bin/venv-layer.sh normalize /export \
    && sh /usr/local/bin/venv-layer.sh stack /export /opt/venv \
    && TYPESAFE_CARLA_PYCARLA_BUILD=0 /opt/venv/bin/python -c "import typesafe_carla.carla as carla, cv2, lanelet2, numpy, scipy.interpolate, shapely; from importlib.metadata import version; from typesafe_carla import paths; carla.Location(1.0, 2.0, 3.0); print('typesafe-carla ' + version('typesafe-carla') + ' (CARLA ' + paths.native_info()['carla_git_ref'] + '); native extension modules import cleanly')"


# ---------------------------------------------------------------------------
# Stage 3: runtime -- the venv and nothing else
# ---------------------------------------------------------------------------
FROM ${RUNTIME_BASE} AS runtime

# ffmpeg is only invoked by CameraRecorder, and it is the largest thing that
# can be added to this image, so it stays opt-in. With it off the runtime
# installs no packages and carries no apt layer: after the move to
# opencv-python-headless the venv needs nothing beyond the glibc, libgcc_s,
# libstdc++ and libz the base image already has.
ARG WITH_FFMPEG=false

# The virtualenv comes first and the two RUN layers below follow it, because a
# pull can only reuse a layer whose whole parent chain it already has -- and
# both of those layers vary in ways their inputs do not: apt hands out whatever
# ffmpeg it has today, and `useradd` writes the current date into /etc/shadow.
# Ahead of the virtualenv, either would put a fresh digest on all ~400 MB
# behind it. Here, each costs only itself.
#
# --link is what makes the four layers portable between images: each is built
# against an empty base rather than against whatever the previous instruction
# left behind, so identical content is the same blob no matter what it lands on.
COPY --link --from=venv /export/client /opt/venv
COPY --link --from=venv /export/deps /opt/venv
COPY --link --from=venv /export/framework /opt/venv
COPY --link --from=venv /export/scenario /opt/venv

RUN set -eu; \
    if [ "${WITH_FFMPEG}" = "true" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends ffmpeg; \
        rm -rf /var/lib/apt/lists/*; \
    fi

# --user-group makes the matching group explicit rather than relying on
# Debian's USERGROUPS_ENAB default, which `install -o/-g` below depends on.
RUN useradd --create-home --user-group --uid 1000 scenario \
    && install -d -o scenario -g scenario /work

ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HYDRA_FULL_ERROR=1 \
    TYPESAFE_CARLA_PYCARLA_BUILD=0

LABEL org.opencontainers.image.title="autoware_carla_scenario scenario package" \
      org.opencontainers.image.description="A generated scenario package with its CARLA client (typesafe_carla)" \
      org.opencontainers.image.source="https://github.com/hakuturu583/autoware_carla_scenario" \
      io.autoware.carla-scenario.carla-client="typesafe-carla"

USER scenario
# Hydra writes its run directory relative to the working directory, so mount a
# volume here to keep scenario outputs after the container exits.
WORKDIR /work

ENTRYPOINT ["scenario"]
CMD ["--help"]
