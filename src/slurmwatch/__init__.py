from ._version import resolve as _resolve_version
from .collector import TelemetryCollector
from .config import SlurmwatchConfig
from .exceptions import (
    CgroupAccessError,
    CgroupNotFoundError,
    CgroupPermissionError,
    JobNotFoundError,
    JobNotRunningError,
    SlurmCommandError,
    SlurmwatchError,
)
from .model import (
    CpuMetrics,
    GpuInterconnect,
    GpuMetrics,
    JobContext,
    MemoryMetrics,
    NodeFabric,
    TelemetrySnapshot,
)
from .slurm import resolve_current_jobs, resolve_job_context


def __getattr__(name: str) -> str:
    """Resolve ``__version__`` on first access, not at import.

    Reading it costs an ``importlib.metadata`` import (~34 ms, and email/zipfile
    with it) that every other entry point pays for and none of them use.
    """
    if name == "__version__":
        return _resolve_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CgroupAccessError",
    "CgroupNotFoundError",
    "CgroupPermissionError",
    "CpuMetrics",
    "GpuInterconnect",
    "NodeFabric",
    "GpuMetrics",
    "JobContext",
    "JobNotFoundError",
    "JobNotRunningError",
    "MemoryMetrics",
    "SlurmCommandError",
    "SlurmwatchConfig",
    "SlurmwatchError",
    "TelemetryCollector",
    "TelemetrySnapshot",
    "__version__",
    "resolve_current_jobs",
    "resolve_job_context",
]
