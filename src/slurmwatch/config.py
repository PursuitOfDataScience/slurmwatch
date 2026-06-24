from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SlurmwatchConfig:
    poll_interval: float = 0.5
    oom_warning_threshold: float = 0.85
    oom_critical_threshold: float = 0.90
    cgroup_scan_timeout: float = 2.0
    slurm_cmd_timeout: int = 15
    csv_header_interval: int = 3600
    log_buffering: int = 10_000
    gpu_temp_threshold_celsius: float = 85.0
    gpu_power_threshold_watts: float = 300.0
    headless_interval: float = 1.0
