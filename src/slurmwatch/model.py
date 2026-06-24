from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class CpuMetrics:
    cores_allocated: int
    usage_ns: int
    usage_percent: float

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


@dataclass
class MemoryMetrics:
    current_bytes: int
    limit_bytes: int
    peak_bytes: int
    usage_percent: float
    oom_guard_warning: bool
    oom_guard_critical: bool

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


@dataclass
class GpuMetrics:
    index: int
    uuid: str
    name: str
    utilization_percent: float
    memory_used_bytes: int
    memory_total_bytes: int
    memory_utilization_percent: float
    power_watts: float
    temperature_celsius: float
    throttling: bool

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


@dataclass
class TelemetrySnapshot:
    timestamp: float
    job_id: int
    step_id: int | None
    hostname: str
    elapsed_seconds: int
    cpu: CpuMetrics
    memory: MemoryMetrics
    gpus: list[GpuMetrics] = field(default_factory=list)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["gpus"] = [g.to_dict() for g in self.gpus]
        return json.dumps(payload, default=str)

    def to_csv_row(self) -> str:
        cols = [
            f"{self.timestamp:.3f}",
            str(self.job_id),
            self.hostname,
            str(self.elapsed_seconds),
            str(self.cpu.cores_allocated),
            f"{self.cpu.usage_percent:.2f}",
            str(self.memory.current_bytes),
            str(self.memory.limit_bytes),
            f"{self.memory.usage_percent:.2f}",
            str(len(self.gpus)),
        ]
        for gpu in self.gpus:
            cols.extend(
                [
                    f"{gpu.utilization_percent:.2f}",
                    str(gpu.memory_used_bytes),
                    str(gpu.memory_total_bytes),
                    f"{gpu.memory_utilization_percent:.2f}",
                    f"{gpu.power_watts:.1f}",
                    f"{gpu.temperature_celsius:.1f}",
                    str(gpu.throttling).lower(),
                ]
            )
        return ",".join(cols)

    @classmethod
    def csv_header(cls, max_gpus: int = 8) -> str:
        cols = [
            "timestamp",
            "job_id",
            "hostname",
            "elapsed_seconds",
            "cpu_cores",
            "cpu_percent",
            "mem_current_bytes",
            "mem_limit_bytes",
            "mem_percent",
            "gpu_count",
        ]
        for i in range(max_gpus):
            cols.extend(
                [
                    f"gpu_{i}_util_percent",
                    f"gpu_{i}_mem_used_bytes",
                    f"gpu_{i}_mem_total_bytes",
                    f"gpu_{i}_mem_percent",
                    f"gpu_{i}_power_watts",
                    f"gpu_{i}_temp_celsius",
                    f"gpu_{i}_throttling",
                ]
            )
        return ",".join(cols)


@dataclass
class JobContext:
    job_id: int
    username: str
    partition: str
    nodelist: str
    hostname: str
    cpus_allocated: int
    mem_limit_bytes: int
    gpu_count_requested: int
    gpu_indices: list[int]
    step_id: int | None = None
    uid: int | None = None
    cgroup_v2_path: str | None = None
    cgroup_v1_mem_path: str | None = None
    cgroup_v1_cpu_path: str | None = None
    job_start_time: float | None = None
