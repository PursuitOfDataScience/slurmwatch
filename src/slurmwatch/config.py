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
    gpu_idle_minutes: int = 5

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
            "SLURMWATCH_GPU_IDLE_MIN": "gpu_idle_minutes",
        }
        float_fields = {
            "poll_interval",
            "oom_warning_threshold",
            "oom_critical_threshold",
            "headless_interval",
            "cpu_underuse_threshold",
            "gpu_idle_threshold",
        }
        int_fields = {"history_seconds", "gpu_idle_minutes"}
        bool_fields = {"ascii_mode"}
        for env_var, field_name in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                if field_name in float_fields:
                    kwargs[field_name] = float(val)
                elif field_name in int_fields:
                    kwargs[field_name] = int(val)
                elif field_name in bool_fields:
                    kwargs[field_name] = val.lower() in ("1", "true", "yes")
                else:
                    kwargs[field_name] = val
        return cls(**kwargs)  # type: ignore[arg-type]
