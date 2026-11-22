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
├── context.json    # command line, overrides, output directory, versions, environment
├── config.yaml     # the run's composed config (a run through Hydra)
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
`launcher.*` overrides; a report of a SLURM task that failed outside any case
runs the task's worker again. Extra arguments are passed through:

```bash
sh outputs/.../crash_reports/20261122T011500Z-RuntimeError-1a2b3c4d/repro.sh
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

When the run ends normally, the directory is removed. If one is left with an
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
| `SCENARIO_CRASH_REPORT_MAX_FILE_SIZE` | The largest file of a report (`1MiB`, `512KB`, `1048576`; `0` for no limit); default `1MiB` |
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
| `locals` | the local variables of every frame, in `event.json` |
| `log` | `log.txt`, and the log records as the event's breadcrumbs |
| `config` | `config.yaml`, the run's composed config |
| `environment` | the environment variables that affect a run, in `context.json` and as `export` lines in `repro.sh` |
| `packages` | the installed packages and their versions, in `event.json` |
| `repro` | `repro.sh` |

The value is a comma-separated list of items, `all`, `none`, or `all` with
some items removed by a leading `-`:

```bash
export SCENARIO_CRASH_REPORT_ITEMS=all,-locals          # everything but local variables
export SCENARIO_CRASH_REPORT_SEND_ITEMS=log,config,repro  # write all, send less
```

`SCENARIO_CRASH_REPORT_SEND_ITEMS` takes the same form and chooses what is
sent. Only what a report holds can be sent. The files on disk are written
whole either way, so you can keep everything locally and send a narrower
report. For example, leaving out `environment` from the sent items drops it
from the sent `context.json` and drops the `export` lines from the sent
`repro.sh`.

## Size limits

The defaults keep a report useful while bounding what a failing batch can
write:

- **`MAX_FILE_SIZE` (1 MiB):** applies to each file of a report except
  `event.json`. A longer file is cut, with a line saying how many bytes were
  dropped. `log.txt` and `traceback.txt` keep their end, where the error is.
  Other files keep their start. At the default length, `log.txt` fits in this
  limit, and a config is a few kilobytes. The Sentry SDK bounds `event.json`
  itself: it shortens long values and caps how much of each frame it keeps.
- **`MAX_REPORTS` (20):** applies to each process. A batch whose every case
  fails on every attempt would otherwise write a report per attempt. After the
  limit, one warning says so, and nothing more is written or sent. The limit
  counts every report the process writes, so in a batch the first failures use
  it up. Native-crash reports are not counted.
- **`LOG_LINES` (5000):** applies to `log.txt`. 5000 records is a few minutes
  of a run's log, including the start of the case.
