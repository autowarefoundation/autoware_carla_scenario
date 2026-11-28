# Crash Reports

When a run raises or the process crashes, `scenario` writes a **crash
report**: everything it takes to see what went wrong and to run the same case
again. It is built on the [Sentry SDK](https://github.com/getsentry/sentry-python)
(`sentry-sdk`, MIT-licensed), which captures the error as a Sentry event. The
SDK is used as a library only: reports are written to files, and nothing leaves
the machine unless you set `SCENARIO_CRASH_REPORT_DSN` (see
[Sending reports](#sending-reports)).

## Where reports go

Each report is a directory under `crash_reports/` in the run's output
directory:

```text
outputs/2026-11-22/10-15-00/crash_reports/20261122T011500Z-RuntimeError-1a2b3c4d/
├── event.json      # the Sentry event: exception, every frame and its local variables,
│                   # breadcrumbs (the recent log records), installed packages
├── traceback.txt   # the traceback, chained exceptions included
├── log.txt         # the last log records of the process (5000 by default)
├── context.json    # command line, overrides, output directory, versions, environment,
│                   # the environment pin, the map's commit
├── config.yaml     # the case's composed config (none outside a case)
└── repro.sh        # the command that runs the same case again
```

| Run | Reports go to |
|---|---|
| A single run, or a `--multirun` case | `crash_reports/` in that run's or case's directory |
| A batch (a scenario glob) | `crash_reports/` in the batch's directory |
| A SLURM task (`launcher=slurm`) | `crash_reports/` in the batch's directory, from every task |
| Anything before that (composing the config, ...) | `$SCENARIO_CRASH_REPORT_DIR`, else `outputs/crash_reports/` |

`repro.sh` changes to the directory the run started in, exports the
environment variables that affect a run (`CARLA*`, `SCENARIO*`, `ROS_*`, ...;
anything whose name looks like a secret is left out), and runs `scenario` with
the case's own overrides. Run again, the case writes to an output directory of
its own. A SLURM case's runs locally, without the batch's `launcher=` and
`launcher.*` overrides, and with `launcher=local` when the case's config
selects SLURM (a launcher its scenario's config sets, which a single run
refuses); a report of a SLURM task that failed outside any case
runs the task's worker again. A report written outside any case (a batch's,
before its cases are known) runs the command line again, without its
`hydra.run.dir=` and `hydra.sweep.dir=`: the batch's directory is no longer
empty, and a batch refuses one that is not, so the rerun writes to a new one.
Extra arguments are passed through:

```bash
sh outputs/.../crash_reports/20261122T011500Z-RuntimeError-1a2b3c4d/repro.sh
```

The environment `repro.sh` exports is the one `scenario` started with, read
before the framework sets any variable of its own. It never exports the
variables the framework sets for a run's processes
(`AUTOWARE_CARLA_SCENARIO_SWEEPER_OWNS_RETRIES`,
`AUTOWARE_CARLA_SCENARIO_HEAVY_METRICS_DECIDED`), nor `SLURM_*`; in a SLURM
task it also leaves out `AUTOWARE_CARLA_SCENARIO_TYPECHECK` (the worker turns
the static check off, the submitting process having run it) and
`CUDA_VISIBLE_DEVICES` (the GPUs SLURM gave the task). `context.json` keeps
them all.

### The map's commit

When the map was read from `map.source`, `context.json` names the commit it
resolved to (`"map": {"source": ..., "commit": ..., "pinned": ...}`). A
branch or a tag moves, so a case's `repro.sh` reads the map at that commit: it
runs with `map.source='<source>@<commit>'` in place of the case's own
`map.source=`, and says so in a comment. Leave that override out to read the
ref's current tip instead. A report written outside any case names the commit
in a comment only. A source that names a commit already is run as it is.

### The environment pin

A run that raised writes no `{ScenarioName}_requirements.lock` nor
`_environment.json` (they are written with a verdict; see
[Usage](usage.md)), so `context.json` names the
environment in brief under `environment_pin`: the lock's id, and the
framework's and the scenario's package with their version, how they were
installed, and the commit and uncommitted changes (`dirty`) of a checkout.
It is `null` when the environment cannot be read; the report is written all
the same.

### Credentials

A URL may carry credentials, a map source's most often
(`map.source=https://oauth2:TOKEN@host/o/r`). Everything a report writes or
sends shows them redacted, as the rest of the framework does
(`https://oauth2:***@host/o/r`; see [Maps](maps.md)): `context.json`,
`config.yaml`, `log.txt`, `traceback.txt`, every string of `event.json` (the
local variables, the command line, the breadcrumbs) and `repro.sh` too. Only
the userinfo of a URL is redacted, not a token in its query. A `repro.sh`
that holds a redacted URL does not run as it is: a comment at its top says so.
Put the credentials back (`***`) before running it, or pass the source again
as an extra argument, which comes after the script's own and wins:

```bash
sh .../repro.sh "map.source='https://oauth2:$TOKEN@host/o/r@<commit>'"
```

## What is reported

- A run that raised: a single run, or a `--multirun` case that Hydra's basic
  sweeper goes on from.
- Every failed attempt the scenario queue retries
  (`server.cooldown_max_retries`). A batch goes on from a case that raised and
  keeps only one line of it in its result; the report keeps the rest.
- A SLURM task that raised, every failed attempt of its cases, and anything
  else its worker raises. The task's status names the report of a task that
  failed (`crash_report`; see [Running a Batch on SLURM](slurm.md)).
- Anything else `scenario` raises before it exits.

An exception is reported once: the queue's "failed after N attempts" error,
raised from the last attempt's, does not produce a second report.

A refusal of the configuration (a bad value, a scenario the
[static check](typecheck.md) refuses, a missing extra) is for you to fix, not a
crash, so it is not reported.

## Native crashes

A `SIGSEGV` in the CARLA client or an abort ends the process before Python can
write anything. So once the run's output directory is known, `scenario`
prepares a report there in advance, `crash_reports/native-<time>-<host>-<pid>/`
(the host because the tasks of a SLURM batch share its directory), with
`context.json`, `config.yaml` and `repro.sh`. If the process dies of a fatal
signal, Python's `faulthandler` writes the stack of every thread to that
directory's `faulthandler.log`.

When the run ends normally, the directory is removed, and `crash_reports/`
with it when preparing the report made it and nothing else is in it: a
directory `SCENARIO_CRASH_REPORT_DIR` names is never removed. If one is left with an
empty `faulthandler.log`, the process was killed without a chance to write
anything (`SIGKILL`, the OOM killer). Hydra's own log of the run, in the run's
output directory, has the log records that came before.

## Sending reports

Reports are sent only when `SCENARIO_CRASH_REPORT_DSN` is set to a project's
DSN. Each report is then also sent there, with its files as attachments. The
Sentry SDK's own `SENTRY_DSN` and `SENTRY_SPOTLIGHT` are ignored, so a shell
that sets them for another project sends nothing from `scenario`. `SENTRY_ENVIRONMENT` and `SENTRY_RELEASE` apply
as usual. The release defaults to `autoware-carla-scenario@<version>`.

Any server that speaks Sentry's protocol accepts the reports. The recommended
one is [GlitchTip](https://glitchtip.com/), an open-source (MIT) error tracker
you host yourself, which takes events from the Sentry SDK unchanged:

1. Run GlitchTip (its [installation guide](https://glitchtip.com/documentation/install)
   covers Docker Compose and Helm).
2. Create an organization and a project in it, and copy the project's DSN.
3. Export it where `scenario` runs:

    ```bash
    export SCENARIO_CRASH_REPORT_DSN="https://<key>@glitchtip.example.org/<project-id>"
    ```

The client side is open source either way. Only the server you send to
changes: sentry.io is a hosted service, and Sentry's own self-hosted server is
under the Functional Source License, which is not an open-source license.

!!! warning
    The event includes local variables and the run's config. Only set a DSN
    for a project you trust with them.

## Settings

| Variable | Effect |
|---|---|
| `SCENARIO_CRASH_REPORT=0` | Turns crash reports off (`false`, `no` and `off` also work) |
| `SCENARIO_CRASH_REPORT_DIR` | Where reports go before a run's directory is known; native-crash reports are also prepared there from the start |
| `SCENARIO_CRASH_REPORT_DSN` | Also send each report to this project (a Sentry DSN, such as GlitchTip's); nothing is sent without it |
| `SCENARIO_CRASH_REPORT_ITEMS` | What a report holds ([items](#choosing-what-a-report-holds)); default `all` |
| `SCENARIO_CRASH_REPORT_SEND_ITEMS` | What of a report is sent with a DSN; default: what it holds |
| `SCENARIO_CRASH_REPORT_MAX_FILE_SIZE` | The largest `log.txt`, `traceback.txt` and `config.yaml` of a report (`1MiB`, `512KB`, `1048576`; `0` for no limit, a size under a byte is refused); default `1MiB` |
| `SCENARIO_CRASH_REPORT_MAX_REPORTS` | How many reports one process writes at most (`0` for no limit); default `20` |
| `SCENARIO_CRASH_REPORT_LOG_LINES` | How many log records `log.txt` keeps; default `5000` |

A value that cannot be read is reported as a warning, and its default is
used. A crash report never stops a run.

## Choosing what a report holds

Every report holds the exception with its stack, `traceback.txt` and
`context.json` (the command line, overrides, versions). Beyond those,
`SCENARIO_CRASH_REPORT_ITEMS` chooses among these items:

| Item | What it adds |
|---|---|
| `locals` | the local variables of every frame, in `event.json`; one whose name looks like a secret (`password`, `token`, `api_key`, ...) shows as `[Filtered]` |
| `log` | `log.txt`, and the log records as the event's breadcrumbs |
| `config` | `config.yaml`, the run's composed config |
| `environment` | the environment variables that affect a run, in `context.json` and as `export` lines in `repro.sh` |
| `packages` | the installed packages and their versions, in `event.json` |
| `repro` | `repro.sh` |

The value is a comma-separated list of items, `all`, `none`, or `all` with
some items removed by a leading `-` (a list that starts with a removal, such
as `-locals`, starts from `all`):

```bash
export SCENARIO_CRASH_REPORT_ITEMS=all,-locals          # everything but local variables
export SCENARIO_CRASH_REPORT_SEND_ITEMS=log,config,repro  # write all, send less
```

`SCENARIO_CRASH_REPORT_SEND_ITEMS` takes the same form and chooses what is
sent. Only what a report holds can be sent. The files on disk are written
whole either way, so you can keep everything locally and send a narrower
report. For example, leaving out `environment` from the sent items drops it
from the sent `context.json` and sends a `repro.sh` written without the
`export` lines.

## Size limits

The defaults keep a report useful while bounding what a failing batch can
write:

- **`MAX_FILE_SIZE` (1 MiB):** applies to `log.txt`, `traceback.txt` and
  `config.yaml`. A longer file is cut, with a line saying how many bytes were
  dropped. `log.txt` and `traceback.txt` keep their end, where the error is;
  `config.yaml` keeps its start. At the default length, `log.txt` fits in this
  limit, and a config is a few kilobytes. `context.json` and `repro.sh` are
  small and never cut: cut, they would be neither JSON nor a script. The
  Sentry SDK bounds `event.json` itself: it shortens long values and caps how
  much of each frame it keeps.
- **`MAX_REPORTS` (20):** applies to each process. A batch whose every case
  fails on every attempt would otherwise write a report per attempt. After the
  limit, one warning says so, and nothing more is written or sent. The limit
  counts every report the process writes, so in a batch the first failures use
  it up. Native-crash reports are not counted.
- **`LOG_LINES` (5000):** applies to `log.txt`, which keeps the records at
  `INFO` and above. 5000 records is a few minutes of a run's log, including
  the start of the case.
