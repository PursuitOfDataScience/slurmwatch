from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SlurmwatchConfig:
    poll_interval: float = 0.5
    oom_warning_threshold: float = 0.85
    oom_critical_threshold: float = 0.90
    headless_interval: float = 1.0
    csv_dialect: str = "excel"
    ascii_mode: bool = False
    history_seconds: int = 60
    cpu_underuse_threshold: float = 0.5
    gpu_idle_threshold: float = 5.0

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
                    kwargs[field_name] = float(val)
                elif field_name in int_fields:
                    kwargs[field_name] = int(float(val))
                elif field_name in bool_fields:
                    kwargs[field_name] = val.lower() in ("1", "true", "yes")
                else:
                    kwargs[field_name] = val
            except ValueError:
                raise ValueError(
                    f"Invalid value for {env_var}: {val!r} (expected a number)"
                ) from None
        config = cls(**kwargs)  # type: ignore[arg-type]
        # A zero/negative interval would busy-loop the collector on the
        # compute node being monitored.
        config.poll_interval = max(config.poll_interval, 0.05)
        config.headless_interval = max(config.headless_interval, 0.05)
        config.history_seconds = max(config.history_seconds, 1)
        return config
