# Backends a Run Starts

Some parts of a scenario run on more than CARLA. One example is a pedestrian
whose motion a model generates (`SmartPedestrianControllerAction`, which uses
the ARDY container). Another is a renderer served by a process of its own. Such
a part declares on its class the backends it needs, with `@need_prepare`:

```python
from autoware_carla_scenario.prepare.declare import need_prepare
from autoware_carla_scenario.pedestrian.ardy import ArdyBackend


@need_prepare(ArdyBackend)
class SmartPedestrianControllerAction(BaseAction): ...
```

The scenario queue then starts what its scenarios need before they run, and
stops it when the queue stops.

## When a backend starts

The queue has a preparer, `prepare.pool.BackendPool`, which `scenario-run`
builds from `backends`. The preparer looks at each scenario before it runs:

- **When the queue starts**, before the CARLA server, it starts what the queue's
  scenarios declare.
  - The scenario class's own `@need_prepare` counts.
  - So do the declarations on the classes that a declarative (YAML) scenario's
    actions build. These are read from the compiled document without CARLA.
  - On SLURM, the worker does the same for its task's cases once its CARLA is
    up.
- **After each scenario's `setup()`, before its run ticks**, it starts whatever
  the registered actions need that is not running yet. This covers a Python
  scenario whose actions only exist once `setup()` has built them.

Each backend starts at most once per queue. The next scenario that needs it
finds it running. An action reaches the backend it needs with
`prepare.pool.running_backend(ArdyBackend)`.

An action that runs without a queue starts its backend itself the first time it
fires, with that backend's defaults, and stops it when the process exits.

## Resources

Before starting anything, the preparer adds up what the backends it is about to
start take, as each one declares in `prepare.Resources`:

- GPU memory on one GPU, in MiB;
- host memory, in MiB.

It then compares that with what the host has free:

- the free memory of each GPU this process may use, from `nvidia-smi`, limited
  by `CUDA_VISIBLE_DEVICES` (SLURM sets it for a job's GPUs);
- `MemAvailable`, the host's own, since a container runs outside the job that
  starts it.

If anything falls short, the run fails with `InsufficientResourcesError` before
any backend starts. The message names what each backend needs and what is
free:

```text
InsufficientResourcesError: the host cannot run the backends this run needs --
ardy: 24000 MiB of GPU memory needed, 9120 MiB free on the roomiest GPU (0)
(needed: ardy (24000 MiB GPU, 16000 MiB host)). Free the memory, or lower what
a backend is said to take (backends.<name>.gpu_memory_mb / memory_mb).
```

Each backend is placed on the GPU with the most room left, and starts on that
GPU. A container set to one GPU (`gpus: device=N`) is checked on that GPU, and
one with `gpus: none` needs no GPU at all.

The check is a snapshot of what is free when the backends start. It cannot see
what others are about to take:

- CARLA, when the queue starts it, starts after the backends. Its GPU memory
  must be left free, beyond what the backends take.
- Another run on the same host that has started a backend that is still
  loading has not taken its GPU memory yet. Two runs that prepare at the same
  time can both pass the check and run out of GPU memory when they load.
- `CUDA_VISIBLE_DEVICES` is read as `nvidia-smi`'s indices or UUIDs. Where
  SLURM renumbers a job's GPUs from 0, set `gpus: device=N` to the host's index
  instead.

## Containers

`prepare.docker.DockerBackend` runs a backend as a container on this host. It:

- checks that docker is usable, and warns when it has no nvidia runtime;
- pulls the image if the host lacks it (`pull: missing`, the default; also
  `always` or `never`);
- runs the container on its GPU, with its port published on `127.0.0.1` at a
  free port;
- follows the container's log into the run's output directory
  (`backend_<name>.log`);
- waits until the service answers, for up to `startup_timeout_s`.

A container that exits while starting is kept until its logs are read into the
error, and then removed; every container is removed when its run ends.

It labels each container with the host and process that started it. A container
left behind by a killed run is removed the next time a backend of its name
starts. Only that one is removed: two runs on one host, such as two SLURM jobs,
each keep their own.

## Settings

Each backend reads its own block under `backends` of the config, and every key
is optional:

```yaml
backends:
  ardy:
    image: ghcr.io/hakuturu583/carla_ardy_backend:0.1.0
    model: soma             # the ARDY checkpoint
    acceleration: none      # none, compile
    gpus: auto              # the GPU the preparer chose; or all, device=N
    hf_home: ~/.cache/huggingface
    gpu_memory_mb: 24000
    memory_mb: 16000
    startup_timeout_s: 1800 # a first start downloads ~20 GB of weights
    timeout_s: 120          # one generation step
    pull: missing
    docker_args: []
```

```bash
scenario-run scenario=... backends.ardy.image=ghcr.io/hakuturu583/carla_ardy_backend:0.2.0
```

The ARDY text encoder is Llama-3-8B, which is gated on Hugging Face. The
container gets the host's Hugging Face cache, where the token and the weights
are, at `hf_home`. It also gets `HF_TOKEN` when this process has one.

## Adding a backend

A backend class has two parts:

- a `backend_name`, which is its settings' key under `backends`;
- a `from_settings(settings)` class method that builds it.

What it builds implements `prepare.Backend`: `name`, `resources()`,
`start(log_dir, gpu)`, `address()` and `close()`. For a container, subclass
`prepare.docker.DockerBackend`, build its `DockerSettings`, and implement
`ready(timeout_s)` to say when the service answers. Then put
`@need_prepare(YourBackend)` on the action, scenario or renderer that needs it.
