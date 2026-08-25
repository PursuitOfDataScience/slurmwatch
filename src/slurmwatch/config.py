from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass

# A zero/near-zero interval would busy-loop the collector on the compute node
# being monitored, so every path that sets an interval floors it here.
#
# 0.1s, not the old 0.05: `--interval 0.001` is one misplaced character away from
# the intended `0.1`, and it used to be accepted silently — the loop then ran at
# the floor, ~19 snapshots a SECOND, writing a --log file a thousand times faster
# than asked while re-reading the cgroup that often. Nothing in a live view is
# meaningfully faster than 10 Hz, so this bounds the pathological end without
# touching any real usage (the defaults are 0.5 / 1.0). SW-13.
MIN_INTERVAL = 0.1
# The floor when sampling comes from `sstat` instead of the local cgroup: each
# sample is a subprocess AND a slurmdbd query, so the polite rate is an order of
# magnitude slower than reading files on the node. (The collector separately caches
# an sstat sample for 5s, so this bounds the snapshot/log rate rather than the
# database load.)
MIN_REMOTE_INTERVAL = 1.0
# Ceiling on the refresh interval too (like MAX_HISTORY_SECONDS): a huge finite
# SLURMWATCH_POLL_INTERVAL (e.g. 1e9) passes from_env but would freeze the refresh
# for ~decades. One hour is far longer than any live view needs.
MAX_INTERVAL = 3_600.0

# The trend history is a rolling window; cap it so a huge (but finite) value can't
# size the dashboard's history deque past sys.maxsize — deque(maxlen=…) then raises
# OverflowError and breaks every UI update (#54) — and so a large value can't grow
# the deques without bound. One day of history is far more than any live trend needs.
MAX_HISTORY_SECONDS = 86_400

_TRUE_VALUES = {"1", "true", "yes", "on", "y", "t"}
_FALSE_VALUES = {"0", "false", "no", "off", "n", "f", ""}


# Variables whose unusable value has already been reported, so a reader called once
# per frame can't turn one stale setting into a stream of identical lines.
_warned_env_vars: set[str] = set()


def warn_unusable_env(var: str, value: str, reason: str, using: str) -> None:
    """Say ONCE that an environment value couldn't be used, and what happened instead.

    Tolerating a bad value is often the right call — a stale variable from a site
    module file or a `.bashrc` carried between clusters shouldn't stop a monitor
    from running — but it must not be a SILENT call. These knobs are set far from
    where they take effect, so the person who set the value is usually not the
    person reading the output, and "it fell back and said nothing" is
    indistinguishable from "the variable works". SW-17's framing: the complaint was
    never "always fall back", it was falling back silently.
    """
    if var in _warned_env_vars:
        return
    _warned_env_vars.add(var)
    logging.getLogger("slurmwatch").warning(
        "Ignoring %s=%r (%s); using %s", var, value, reason, using
    )


def _parse_bool(value: str) -> bool:
    """Parse a boolean env value, accepting the common spellings.

    Accepts on/off, y/n, t/f, yes/no in addition to 1/0/true/false so that a
    natural value like ``SLURMWATCH_ASCII=on`` isn't silently read as False
    (B-P14). An unrecognized value raises ValueError rather than defaulting.
    """
    v = value.strip().lower()
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    raise ValueError(value)


# A CSV dialect for pipes: RFC 4180's CRLF is correct for a file a spreadsheet will
# open, and wrong for `slurmwatch --once --format csv | awk -F,`, where the trailing
# \r ends up inside the last field and silently breaks a comparison against it. The
# stdlib offers no dialect that is both LF-terminated and minimally quoted ("unix" is
# LF but QUOTE_ALL, which makes every field a quoted string for the reader), so
# register one. SW-31.
SHELL_CSV_DIALECT = "slurmwatch"
csv.register_dialect(
    SHELL_CSV_DIALECT,
    delimiter=",",
    quoting=csv.QUOTE_MINIMAL,
    lineterminator="\n",
    doublequote=True,
)


def resolve_csv_dialect(configured: str, *, to_regular_file: bool) -> str:
    """Turn a configured dialect (possibly "auto") into a concrete dialect name.

    Auto means: a real file gets "excel" (CRLF, what a spreadsheet expects), and
    anything else — a pipe, a terminal, /dev/stdout, a fifo — gets the LF dialect.
    The distinction is the same one --log already draws when it decides what to
    claim: only a regular file is a file.
    """
    if configured != "auto":
        return configured
    return "excel" if to_regular_file else SHELL_CSV_DIALECT


@dataclass
class SlurmwatchConfig:
    # What the user literally asked for on the CLI, before any floor moved it. The
    # floors are applied in two places — `clamp()` right after the override, then the
    # per-path `_apply_sampling_floor` — so by the time the second one runs the value
    # has already been raised and it can neither tell that it happened nor quote the
    # number that was typed. Kept so the notice says "--interval 0.001 raised to …"
    # instead of misquoting its own clamped value (or staying silent).
    requested_interval: float | None = None
    poll_interval: float = 0.5
    oom_warning_threshold: float = 0.85
    oom_critical_threshold: float = 0.90
    headless_interval: float = 1.0
    # "auto" (the default) resolves by DESTINATION, not by taste: a pipe gets
    # SHELL_CSV_DIALECT (LF, minimal quoting) because `awk`/`cut`/`while read` treat a
    # trailing \r as part of the last field, and a regular .csv gets "excel" (CRLF)
    # because RFC 4180 is what a spreadsheet expects. Neither stdlib dialect can serve
    # both: "excel" leaves the \r, "unix" quotes every field. Set
    # SLURMWATCH_CSV_DIALECT to override either way. SW-31.
    csv_dialect: str = "auto"
    # SLURMWATCH_MOUSE. A field rather than a bare os.environ read at app-launch
    # time, so a bad value is caught by from_env's bool validator with the same
    # message every other knob gets — `MOUSE=7` used to be silently False while
    # `ASCII=maybe` was rejected (SW-17).
    mouse: bool = False
    ascii_mode: bool = False
    history_seconds: int = 60
    # Effective-cores / allocated-cores ratio below which CPU is flagged
    # underused (SLURMWATCH_CPU_UNDERUSE). Kept lenient by default so a normally
    # bursty job doesn't flap between "healthy" and "underused"; raise it for a
    # stricter efficiency bar.
    cpu_underuse_threshold: float = 0.15
    gpu_idle_threshold: float = 5.0

    def clamp(self) -> None:
        """Re-apply the interval/history floors after any override.

        Called by :meth:`from_env` and again after CLI flags mutate a config, so
        that ``--interval 0.0001`` can't slip under the floor that ``from_env``
        enforces (B-P1).
        """
        self.poll_interval = min(max(self.poll_interval, MIN_INTERVAL), MAX_INTERVAL)
        self.headless_interval = min(max(self.headless_interval, MIN_INTERVAL), MAX_INTERVAL)
        # Floor AND ceiling: a huge SLURMWATCH_HISTORY_SECONDS (e.g. 1e19) is a
        # finite float that passes from_env but would size deque(maxlen=…) past
        # sys.maxsize and raise OverflowError on the first UI update (#54).
        self.history_seconds = min(max(self.history_seconds, 1), MAX_HISTORY_SECONDS)

    def validate(self) -> None:
        """Reject nonsensical thresholds and an unknown CSV dialect (C3).

        The OOM thresholds must be fractions in (0, 1] with warning <= critical
        (or the guard's meaning inverts); the CPU-underuse ratio must be in
        [0, 1] and the GPU-idle percent in [0, 100] (out-of-range values produce
        nonsensical verdicts); and the CSV dialect must be one Python knows, so a
        bad name fails here with a clear message rather than as a raw csv.Error
        deep in the output path.
        """
        for env_var, value in (
            ("SLURMWATCH_OOM_WARN", self.oom_warning_threshold),
            ("SLURMWATCH_OOM_CRIT", self.oom_critical_threshold),
        ):
            if not (0.0 < value <= 1.0):
                raise ValueError(
                    f"Invalid value for {env_var}: {value!r} "
                    "(expected a fraction in (0, 1], e.g. 0.9)"
                )
        if self.oom_warning_threshold > self.oom_critical_threshold:
            raise ValueError(
                f"SLURMWATCH_OOM_WARN ({self.oom_warning_threshold}) must be <= "
                f"SLURMWATCH_OOM_CRIT ({self.oom_critical_threshold}); if you raised "
                "OOM_WARN above the default 0.90, set OOM_CRIT to match."
            )
        if not (0.0 <= self.cpu_underuse_threshold <= 1.0):
            raise ValueError(
                f"Invalid value for SLURMWATCH_CPU_UNDERUSE: {self.cpu_underuse_threshold!r} "
                "(expected a ratio in [0, 1], e.g. 0.15)"
            )
        if not (0.0 <= self.gpu_idle_threshold <= 100.0):
            raise ValueError(
                f"Invalid value for SLURMWATCH_GPU_IDLE_PCT: {self.gpu_idle_threshold!r} "
                "(expected a percent in [0, 100], e.g. 5)"
            )
        # The CLI's own --interval rejects a non-positive value (argparse
        # _positive_float), so the env must too: SLURMWATCH_POLL_INTERVAL=-5 used to
        # be accepted and quietly clamped to the floor, i.e. the same value was an
        # error as a flag and fine as an environment variable. Validation runs
        # BEFORE clamp() for exactly this reason (SW-17).
        for env_var, value in (
            ("SLURMWATCH_POLL_INTERVAL", self.poll_interval),
            ("SLURMWATCH_HEADLESS_INTERVAL", self.headless_interval),
        ):
            if value <= 0:
                raise ValueError(
                    f"Invalid value for {env_var}: {value!r} "
                    "(expected a positive number of seconds, e.g. 0.5)"
                )
        if self.history_seconds < 1:
            raise ValueError(
                f"Invalid value for SLURMWATCH_HISTORY_SECONDS: {self.history_seconds!r} "
                "(expected a positive number of seconds, e.g. 60)"
            )
        if self.csv_dialect != "auto" and self.csv_dialect not in csv.list_dialects():
            raise ValueError(
                f"Invalid value for SLURMWATCH_CSV_DIALECT: {self.csv_dialect!r} "
                f"(expected 'auto' or one of {sorted(csv.list_dialects())})"
            )

    @classmethod
    def from_env(cls) -> SlurmwatchConfig:
        kwargs: dict[str, object] = {}
        env_map = {
            "SLURMWATCH_POLL_INTERVAL": "poll_interval",
            "SLURMWATCH_OOM_WARN": "oom_warning_threshold",
            "SLURMWATCH_OOM_CRIT": "oom_critical_threshold",
            "SLURMWATCH_HEADLESS_INTERVAL": "headless_interval",
            "SLURMWATCH_CSV_DIALECT": "csv_dialect",
            "SLURMWATCH_ASCII": "ascii_mode",
            "SLURMWATCH_HISTORY_SECONDS": "history_seconds",
            "SLURMWATCH_CPU_UNDERUSE": "cpu_underuse_threshold",
            "SLURMWATCH_GPU_IDLE_PCT": "gpu_idle_threshold",
            "SLURMWATCH_MOUSE": "mouse",
        }
        float_fields = {
            "poll_interval",
            "oom_warning_threshold",
            "oom_critical_threshold",
            "headless_interval",
            "cpu_underuse_threshold",
            "gpu_idle_threshold",
        }
        int_fields = {"history_seconds"}
        bool_fields = {"ascii_mode", "mouse"}
        for env_var, field_name in env_map.items():
            val = os.environ.get(env_var)
            if val is None:
                continue
            if field_name in bool_fields:
                # A bool toggle needs its own message: a bad true/false value is
                # not a "finite number" problem (C4).
                try:
                    kwargs[field_name] = _parse_bool(val)
                except ValueError:
                    raise ValueError(
                        f"Invalid value for {env_var}: {val!r} "
                        "(expected a boolean, e.g. true/false, on/off, 1/0)"
                    ) from None
                continue
            if field_name in float_fields or field_name in int_fields:
                try:
                    num = float(val)
                    # 'inf'/'nan' parse fine but defeat the min-interval clamp
                    # (max(nan, 0.05) is nan) and crash/hang downstream, and
                    # int(float('inf')) raises OverflowError — so reject
                    # non-finite input here as a plain bad value.
                    if not math.isfinite(num):
                        raise ValueError(val)
                except (ValueError, OverflowError):
                    raise ValueError(
                        f"Invalid value for {env_var}: {val!r} (expected a finite number)"
                    ) from None
                kwargs[field_name] = int(num) if field_name in int_fields else num
                continue
            # String fields (csv_dialect): validated in validate().
            kwargs[field_name] = val
        config = cls(**kwargs)  # type: ignore[arg-type]
        # An interval set through the environment is just as much a REQUEST as one
        # typed as --interval, and the notice that a floor overrode it keys off this
        # field — so record it here too, or `SLURMWATCH_POLL_INTERVAL=0.2` off-node
        # gets silently raised to 1s with nothing said.
        if "poll_interval" in kwargs:
            config.requested_interval = float(kwargs["poll_interval"])  # type: ignore[arg-type]
        # validate() first: clamp() would raise a negative interval to the floor and
        # a negative history to 1, so validating afterwards could never see the value
        # the user actually set (SW-17).
        config.validate()
        config.clamp()
        return config
