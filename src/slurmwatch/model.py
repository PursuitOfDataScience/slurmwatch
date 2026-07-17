from __future__ import annotations

import json
import os
import socket
from dataclasses import asdict, dataclass, field
from typing import Any


def short_host(host: str) -> str:
    """A hostname reduced to a comparable short form (domain stripped, lower-cased).

    A node's own ``gethostname`` and Slurm's ``NodeName`` can differ by case or a
    kept domain suffix on some clusters; comparing the short forms makes "is this
    the node I mean?" robust to that (used to identify the local node and its
    index in the resolved nodelist).
    """
    return host.split(".")[0].strip().lower()


def local_node_name() -> str:
    """This host's Slurm node name (short form), for matching against a NodeList.

    Prefer ``$SLURMD_NODENAME`` — Slurm's authoritative NodeName, exported into
    every batch/step task — over the OS hostname, so identity still works on the
    clusters that use the documented ``NodeName``≠``NodeHostname`` alias split
    (where ``gethostname`` returns a name that appears in no NodeList). Falls back
    to the short OS hostname when it's unset (login nodes, or outside a step).
    """
    return short_host(os.environ.get("SLURMD_NODENAME") or socket.gethostname())


@dataclass
class CpuMetrics:
    cores_allocated: int
    usage_ns: int
    usage_percent: float
    effective_cores: float = 0.0
    # The most cores ever busy at once since monitoring began — a high-water mark
    # for right-sizing --cpus-per-task (there's no kernel counter for this, so the
    # collector tracks it as a monotonic running max).
    peak_effective_cores: float = 0.0

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
    working_set_bytes: int = 0
    cache_bytes: int = 0

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
    process_utilization_percent: float = 0.0
    process_memory_bytes: int = 0
    # False when NVML couldn't read device-wide utilization (e.g. a MIG slice
    # where the rate APIs return NOT_SUPPORTED); the active/idle heuristic then
    # falls back to VRAM occupancy instead of scoring the device idle (B-P3).
    utilization_available: bool = True

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


@dataclass
class TelemetrySnapshot:
    timestamp: float
    job_id: str
    step_id: str | None
    hostname: str
    elapsed_seconds: int
    cpu: CpuMetrics
    memory: MemoryMetrics
    gpus: list[GpuMetrics] = field(default_factory=list)
    node_count: int = 1
    node_index: int = 0
    gpu_count_requested: int = 0
    gpu_active_count: int = 0
    # True when the sample is a job-wide sstat estimate collected off the compute
    # node (no cgroups / NVML reachable), not live per-node telemetry. Memory is a
    # lifetime peak (MaxRSS), CPU is an average, and neither can be attributed to a
    # single node — so consumers must not read the memory figure as a live per-node
    # "current" or drive a (never-clearing) OOM alarm off it (#34, #35).
    remote: bool = False

    def to_json(self) -> str:
        payload = asdict(self)
        payload["gpus"] = [g.to_dict() for g in self.gpus]
        return json.dumps(payload, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TelemetrySnapshot:
        """Reconstruct a snapshot from a ``to_json`` payload.

        Used by the node switcher to turn another node's ``--once --json`` output
        back into a snapshot. Unknown keys are ignored so a small version skew
        between nodes can't crash the parse.
        """

        def _only(cls_: Any, src: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in src.items() if k in cls_.__dataclass_fields__}

        return cls(
            timestamp=float(d["timestamp"]),
            job_id=str(d["job_id"]),
            step_id=(None if d.get("step_id") is None else str(d["step_id"])),
            hostname=str(d["hostname"]),
            elapsed_seconds=int(d["elapsed_seconds"]),
            cpu=CpuMetrics(**_only(CpuMetrics, d["cpu"])),
            memory=MemoryMetrics(**_only(MemoryMetrics, d["memory"])),
            gpus=[GpuMetrics(**_only(GpuMetrics, g)) for g in d.get("gpus", [])],
            node_count=int(d.get("node_count", 1)),
            node_index=int(d.get("node_index", 0)),
            gpu_count_requested=int(d.get("gpu_count_requested", 0)),
            gpu_active_count=int(d.get("gpu_active_count", 0)),
            remote=bool(d.get("remote", False)),
        )

    @classmethod
    def from_json(cls, text: str) -> TelemetrySnapshot:
        return cls.from_dict(json.loads(text))

    _GPU_COLS = 12
    # A CSV file has one fixed header, so per-GPU detail needs a fixed column
    # count. The caller sizes it to the job's actual GPU count via ``max_gpus``
    # (``--once``/``--log`` pass ``max(len(gpus), gpu_count_requested)``), so a
    # 16-GPU node or a many-slice MIG config isn't silently clipped at 8 (#38).
    # This default is only the fallback for a bare ``to_csv_row()``/``csv_header()``
    # call. The ``gpu_count`` column always reports the *real* device count, so if
    # a row ever carries more GPUs than ``max_gpus`` groups (e.g. the default was
    # used), ``gpu_count`` exceeds the number of ``gpu_<N>_*`` groups present and
    # signals the truncation rather than hiding it.
    _CSV_MAX_GPUS = 8

    def to_csv_row(self, max_gpus: int | None = None) -> list[str]:
        if max_gpus is None:
            max_gpus = self._CSV_MAX_GPUS
        cols: list[str] = [
            f"{self.timestamp:.3f}",
            self.job_id,
            self.hostname,
            str(self.elapsed_seconds),
            str(self.cpu.cores_allocated),
            f"{self.cpu.usage_percent:.2f}",
            f"{self.cpu.effective_cores:.2f}",
            str(self.memory.current_bytes),
            str(self.memory.limit_bytes),
            str(self.memory.working_set_bytes),
            str(self.memory.cache_bytes),
            f"{self.memory.usage_percent:.2f}",
            str(self.memory.peak_bytes),
            str(int(self.memory.oom_guard_warning)),
            str(int(self.memory.oom_guard_critical)),
            # The real device count — never capped. With max_gpus sized to fit it
            # equals the number of gpu_<N>_* groups; if it exceeds them it flags
            # that the row was truncated (#38).
            str(len(self.gpus)),
            str(self.gpu_count_requested),
            str(self.gpu_active_count),
            str(self.node_count),
            str(self.node_index),
            str(int(self.remote)),
        ]
        for i in range(max_gpus):
            if i < len(self.gpus):
                gpu = self.gpus[i]
                cols.extend(
                    [
                        str(gpu.index),
                        gpu.uuid,
                        gpu.name,
                        f"{gpu.utilization_percent:.2f}",
                        str(gpu.memory_used_bytes),
                        str(gpu.memory_total_bytes),
                        f"{gpu.memory_utilization_percent:.2f}",
                        f"{gpu.power_watts:.1f}",
                        f"{gpu.temperature_celsius:.1f}",
                        "1" if gpu.throttling else "0",
                        f"{gpu.process_utilization_percent:.2f}",
                        str(gpu.process_memory_bytes),
                    ]
                )
            else:
                cols.extend([""] * self._GPU_COLS)
        return cols

    @classmethod
    def csv_header(cls, max_gpus: int | None = None) -> list[str]:
        if max_gpus is None:
            max_gpus = cls._CSV_MAX_GPUS
        cols = [
            "timestamp",
            "job_id",
            "hostname",
            "elapsed_seconds",
            "cpu_cores",
            "cpu_percent",
            "cpu_effective_cores",
            "mem_current_bytes",
            "mem_limit_bytes",
            "mem_working_set_bytes",
            "mem_cache_bytes",
            "mem_percent",
            "mem_peak_bytes",
            "mem_oom_warning",
            "mem_oom_critical",
            "gpu_count",
            "gpu_count_requested",
            "gpu_active_count",
            "node_count",
            "node_index",
            "remote",
        ]
        for i in range(max_gpus):
            cols.extend(
                [
                    f"gpu_{i}_index",
                    f"gpu_{i}_uuid",
                    f"gpu_{i}_name",
                    f"gpu_{i}_util_percent",
                    f"gpu_{i}_mem_used_bytes",
                    f"gpu_{i}_mem_total_bytes",
                    f"gpu_{i}_mem_percent",
                    f"gpu_{i}_power_watts",
                    f"gpu_{i}_temp_celsius",
                    f"gpu_{i}_throttling",
                    f"gpu_{i}_proc_util_percent",
                    f"gpu_{i}_proc_mem_bytes",
                ]
            )
        return cols


@dataclass
class JobContext:
    job_id: str
    username: str
    partition: str
    nodelist: str
    hostname: str
    cpus_allocated: int
    mem_limit_bytes: int
    gpu_count_requested: int
    gpu_indices: list[int]
    gpu_uuids: list[str] = field(default_factory=list)
    step_id: str | None = None
    uid: int | None = None
    cgroup_v2_path: str | None = None
    cgroup_v1_mem_path: str | None = None
    cgroup_v1_cpu_path: str | None = None
    job_start_time: float | None = None
    job_state: str | None = None
    # The job's wall-clock time limit in seconds (Slurm TimeLimit), or None when
    # unset / UNLIMITED. Used to show how long the job can still run.
    time_limit_seconds: int | None = None
    nodelist_resolved: list[str] = field(default_factory=list)
    min_memory_node: int = 0
    tres: str = ""
    # Job provenance parsed from the same `scontrol show job -d` record — shown
    # in the dashboard's JOB card so "what exactly is this job" is answerable.
    # Empty string / None when the field wasn't present.
    account: str = ""
    qos: str = ""
    command: str = ""
    work_dir: str = ""
    # Resolved stdout / stderr log paths (scontrol StdOut / StdErr, with %j etc.
    # already substituted) so the card can point the user straight at their logs.
    # Slurm merges the two by default, so they are frequently equal.
    std_out: str = ""
    std_err: str = ""
    submit_time: float | None = None
    # The underlying numeric Slurm JobId (array tasks / het components have
    # their own, distinct from the user-facing "12345_3" / "123+1" form). Needed
    # by tools like `srun --jobid=` that only accept the numeric id.
    raw_job_id: str = ""
    # For an array task, the array's base JobId and this task's index (scontrol
    # ArrayJobId / ArrayTaskId); both empty for a non-array job. The user-facing
    # "<base>_<task>" is job_id — these let the UI show the array membership as a
    # fact and correct a bare-base label to the task actually resolved.
    array_job_id: str = ""
    array_task_id: str = ""
    # True when the job's cgroups are not on this host (e.g. running from a
    # login node): usage is sourced remotely via sstat instead of cgroups.
    remote: bool = False
