"""A run's metrics in its Rerun recording.

A run judged on a :class:`~autoware_carla_scenario.conditions.MetricCondition`
writes the metrics it computed once it was over into its trajectory log, as
``metric`` records (docs/trajectory_log.md), and each metric condition's NG
frames as ``condition`` records, which the converter draws in its Conditions
view.  A converter that draws ``metric`` records too (one whose schema has
them) puts the metrics in itself, with a Metrics view.  For one that does not
know them yet, once it has converted the log, :func:`merge_metrics` writes
their values as scalars,
``metrics/<key>``, on the converter's timelines (``frame``, ``sim_time``) and
recording id (the stem), and merges them into ``{stem}.rrd`` (``rerun rrd
merge``, as the sensors are, :mod:`.rerun_lidar`).  A time series view of
``/metrics`` plots them.

Needs the Rerun SDK of the converter's minor version, which the ``rerun``
extra installs; without it the recording is made without the metrics, and a
warning says so.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "METRICS_ENTITY",
    "METRICS_SUFFIX",
    "converter_draws_metrics",
    "merge_metrics",
    "metric_records",
]

logger = logging.getLogger(__name__)

#: The entity the metrics are recorded under: ``metrics/<key>``.
METRICS_ENTITY = "metrics"
#: The scratch recording the metrics are written to before the merge.
METRICS_SUFFIX = ".metrics.rrd"


def metric_records(trajectory: Path) -> list[dict[str, Any]]:
    """The ``metric`` records of the log *trajectory*, in its order."""
    records: list[dict[str, Any]] = []
    with Path(trajectory).open("rb") as lines:
        for line in lines:
            # Cheap to skip: every other record is most of the file.
            if b'"metric"' not in line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("type") == "metric":
                records.append(record)
    return records


def converter_draws_metrics(converter: Any) -> bool:
    """Whether the *converter* package draws ``metric`` records itself (its
    vendored schema has them), with its own Metrics view."""
    return "metric" in list(getattr(converter, "RECORD_TYPES", ()))


def merge_metrics(
    trajectory: Path,
    recording: Path,
    label: str,
    sdk: Any = None,
    converter: Any = None,
) -> bool:
    """Write the metrics of the log *trajectory* into its *recording*, in
    place; whether there were any to write.  Nothing when the *converter*
    that made the recording drew them itself
    (:func:`converter_draws_metrics`).

    Never raises: a recording without its metrics is a recording still, and
    what went wrong is logged.
    """
    if converter is not None and converter_draws_metrics(converter):
        return False
    try:
        records = metric_records(trajectory)
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning(
            "[%s] reading the metrics of %s failed: %s", label, trajectory, exc
        )
        return False
    if not records:
        return False
    try:
        rr = sdk if sdk is not None else _sdk()
    except ImportError:
        logger.warning(
            "[%s] %s holds metrics, but the Rerun SDK, which puts them in the "
            "recording, is not installed: pip install 'autoware-carla-scenario[rerun]'",
            label,
            trajectory,
        )
        return False
    from .rerun_lidar import APPLICATION_ID, merge_into  # noqa: PLC0415

    scratch = Path(recording).with_name(f"{label}{METRICS_SUFFIX}")
    try:
        _write(rr, scratch, label, APPLICATION_ID, records)
        merge_into(Path(recording), scratch)
    except Exception:  # noqa: BLE001 - the recording stays, without them
        logger.exception("[%s] putting the metrics into %s failed", label, recording)
        scratch.unlink(missing_ok=True)
        return False
    logger.info("[%s] Metrics recorded in %s", label, recording)
    return True


def _sdk() -> Any:
    import rerun  # noqa: PLC0415

    return rerun


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write(
    rr: Any,
    path: Path,
    recording_id: str,
    application_id: str,
    records: list[dict[str, Any]],
) -> None:
    stream = rr.RecordingStream(application_id, recording_id=recording_id)
    stream.save(str(path))
    try:
        for record in records:
            stream.set_time("frame", sequence=int(record["frame"]))
            stream.set_time("sim_time", duration=float(record["t"]))
            for key, value in dict(record.get("values", {})).items():
                number = _finite(value)
                if number is not None:
                    stream.log(f"{METRICS_ENTITY}/{key}", rr.Scalars(number))
    finally:
        stream.flush()
        stream.disconnect()
