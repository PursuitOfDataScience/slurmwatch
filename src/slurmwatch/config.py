from __future__ import annotations

import math
import os
from dataclasses import dataclass

# A zero/near-zero interval would busy-loop the collector on the compute node
# being monitored, so every path that sets an interval floors it here.
MIN_INTERVAL = 0.05

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
        self.poll_interval = max(self.poll_interval, MIN_INTERVAL)
        self.headless_interval = max(self.headless_interval, MIN_INTERVAL)
        self.history_seconds = max(self.history_seconds, 1)

    def validate(self) -> None:
        """Reject nonsensical OOM thresholds (B-P14).

        Both must be fractions in (0, 1] and the warning must not sit above the
        critical threshold, or the guard's meaning inverts.
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
                "SLURMWATCH_OOM_WARN "
                f"({self.oom_warning_threshold}) must be <= SLURMWATCH_OOM_CRIT "
                f"({self.oom_critical_threshold})"
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
            try:
                if field_name in float_fields:
                    num = float(val)
                    # 'inf'/'nan' parse fine but defeat the min-interval clamp
                    # below (max(nan, 0.05) is nan) and crash/hang downstream,
                    # so reject non-finite input here as a plain bad value.
                    if not math.isfinite(num):
                        raise ValueError(val)
                    kwargs[field_name] = num
                elif field_name in int_fields:
                    num = float(val)
                    # int(float('inf')) raises OverflowError (not ValueError),
                    # so guard finiteness before the int() cast.
                    if not math.isfinite(num):
                        raise ValueError(val)
                    kwargs[field_name] = int(num)
                elif field_name in bool_fields:
                    kwargs[field_name] = _parse_bool(val)
                else:
                    kwargs[field_name] = val
            except (ValueError, OverflowError):
                raise ValueError(
                    f"Invalid value for {env_var}: {val!r} (expected a finite number)"
                ) from None
        config = cls(**kwargs)  # type: ignore[arg-type]
        config.clamp()
        config.validate()
        return config
