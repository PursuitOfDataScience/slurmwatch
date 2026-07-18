from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

# A zero/near-zero interval would busy-loop the collector on the compute node
# being monitored, so every path that sets an interval floors it here.
MIN_INTERVAL = 0.05
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


@dataclass
class SlurmwatchConfig:
    poll_interval: float = 0.5
    oom_warning_threshold: float = 0.85
    oom_critical_threshold: float = 0.90
    headless_interval: float = 1.0
    csv_dialect: str = "excel"
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
        if self.csv_dialect not in csv.list_dialects():
            raise ValueError(
                f"Invalid value for SLURMWATCH_CSV_DIALECT: {self.csv_dialect!r} "
                f"(expected one of {sorted(csv.list_dialects())})"
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
        bool_fields = {"ascii_mode"}
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
        config.clamp()
        config.validate()
        return config
