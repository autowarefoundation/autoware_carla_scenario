# Crash Reports

When a run raises or the process crashes, `scenario` writes a **crash
report**: everything it takes to see what went wrong and to run the same case
again. It is built on the [Sentry SDK](https://docs.sentry.io/platforms/python/)
(`sentry-sdk`), which captures the error as a Sentry event. Nothing leaves the
machine unless you set a DSN (see [Sending reports](#sending-reports)).

## Where reports go

Each report is a directory under `crash_reports/` in the run's output
directory:

```text
outputs/2026-11-22/10-15-00/crash_reports/20261122T011500Z-RuntimeError-1a2b3c4d/
├── event.json      # the Sentry event: exception, every frame and its local variables,
│                   # breadcrumbs (the recent log records), installed packages
├── traceback.txt   # the traceback, chained exceptions included
├── log.txt         # the last 5000 log records of the process
├── context.json    # command line, overrides, output directory, versions, environment
├── config.yaml     # the run's composed config (a run through Hydra)
└── repro.sh        # the command that runs the same case again
```

| Run | Reports go to |
|---|---|
| A single run, or a `--multirun` case | `crash_reports/` in that run's or case's directory |
| A batch (a scenario glob) | `crash_reports/` in the batch's directory |
| Anything before that (composing the config, ...) | `$SCENARIO_CRASH_REPORT_DIR`, else `outputs/crash_reports/` |

`repro.sh` changes to the directory the run started in, exports the
environment variables that affect a run (`CARLA*`, `SCENARIO*`, `ROS_*`, ...;
anything whose name looks like a secret is left out), and runs `scenario` with
the case's own overrides. Run again, the case writes to an output directory of
its own. Extra arguments are passed through:

```bash
sh outputs/.../crash_reports/20261122T011500Z-RuntimeError-1a2b3c4d/repro.sh
```

## What is reported

- A run that raised: a single run, or a `--multirun` case that Hydra's basic
  sweeper goes on from.
- Every failed attempt the scenario queue retries
  (`server.cooldown_max_retries`). A batch goes on from a case that raised and
  keeps only one line of it in its result; the report keeps the rest.
- Anything else `scenario` raises before it exits.

An exception is reported once: the queue's "failed after N attempts" error,
raised from the last attempt's, does not produce a second report.

A refusal of the configuration (a bad value, a scenario the
[static check](typecheck.md) refuses, a missing extra) is for you to fix, not a
crash, so it is not reported.

## Native crashes

A `SIGSEGV` in the CARLA client or an abort ends the process before Python can
write anything. So once the run's output directory is known, `scenario`
prepares a report there in advance, `crash_reports/native-<time>-<pid>/`, with
`context.json`, `config.yaml` and `repro.sh`. If the process dies of a fatal
signal, Python's `faulthandler` writes the stack of every thread to that
directory's `faulthandler.log`.

When the run ends normally, the directory is removed. If one is left with an
empty `faulthandler.log`, the process was killed without a chance to write
anything (`SIGKILL`, the OOM killer). Hydra's own log of the run, in the run's
output directory, has the log records that came before.

## Sending reports

Set `SENTRY_DSN` to a Sentry project's DSN, and every report is sent there as
well, with its files as attachments. Sentry's `SENTRY_ENVIRONMENT` and
`SENTRY_RELEASE` apply as usual. The release defaults to
`autoware-carla-scenario@<version>`.

!!! warning
    The event includes local variables and the run's config. Only set a DSN
    for a project you trust with them.

## Settings

| Variable | Effect |
|---|---|
| `SCENARIO_CRASH_REPORT=0` | Turns crash reports off (`false`, `no` and `off` also work) |
| `SCENARIO_CRASH_REPORT_DIR` | Where reports go before a run's directory is known; native-crash reports are also prepared there from the start |
| `SENTRY_DSN` | Also send each report to this Sentry project |
