from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import socket
import time
from pathlib import Path

from .config import SlurmwatchConfig
from .model import CpuMetrics, GpuMetrics, JobContext, MemoryMetrics, TelemetrySnapshot

logger = logging.getLogger(__name__)


class TelemetryCollector:
    def __init__(
        self,
        job_ctx: JobContext,
        config: SlurmwatchConfig | None = None,
    ) -> None:
        self.job_ctx = job_ctx
        self.config = config or SlurmwatchConfig()
        self._queue: asyncio.Queue[TelemetrySnapshot] = asyncio.Queue(maxsize=32)
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]

        self._prev_cpu_ns: int | None = None
        self._prev_timestamp: float | None = None
        self._nvml_initialized = False
        self._nvml_handles: list[object] = []
        self._hostname = job_ctx.hostname or socket.gethostname().split(".")[0]
        self._mock = os.environ.get("SLURMWATCH_MOCK") == "1"
        self._mock_start = time.monotonic() if self._mock else 0.0

    async def start(self) -> None:
        self._nvml_initialized = self._init_nvml()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._shutdown_nvml()

    def stop_sync(self) -> None:
        self._stop_event.set()

    async def next_snapshot(self) -> TelemetrySnapshot:
        return await self._queue.get()

    def _init_nvml(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                pynvml.nvmlShutdown()
                logger.info("No NVIDIA devices detected by NVML")
                return False

            visible_indices = self.job_ctx.gpu_indices
            if not visible_indices:
                for idx in range(device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        self._nvml_handles.append(handle)
                    except pynvml.NVMLError:
                        continue
            else:
                for idx in visible_indices:
                    if idx >= device_count:
                        continue
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        self._nvml_handles.append(handle)
                    except pynvml.NVMLError:
                        continue

            logger.info(
                "NVML initialized: %d/%d GPUs visible",
                len(self._nvml_handles),
                device_count,
            )
            return True

        except ImportError:
            logger.info("pynvml not installed; GPU monitoring disabled")
            return False
        except Exception as exc:
            logger.warning("NVML init failed: %s", exc)
            return False

    def _shutdown_nvml(self) -> None:
        if not self._nvml_initialized:
            return
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            pass
        self._nvml_handles.clear()

    async def _run_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                snapshot = await self._collect_snapshot()
                try:
                    self._queue.put_nowait(snapshot)
                except asyncio.QueueFull:
                    try:
                        self._queue.get_nowait()
                        self._queue.put_nowait(snapshot)
                    except asyncio.QueueEmpty:
                        pass
                await asyncio.sleep(self.config.poll_interval)
        except asyncio.CancelledError:
            pass

    async def _collect_snapshot(self) -> TelemetrySnapshot:
        now = time.time()
        cpu = await self._collect_cpu(now)
        mem = await self._collect_memory()
        gpus = await self._collect_gpus()
        elapsed = 0
        if self.job_ctx.job_start_time is not None:
            elapsed = int(now - self.job_ctx.job_start_time)

        return TelemetrySnapshot(
            timestamp=now,
            job_id=self.job_ctx.job_id,
            step_id=self.job_ctx.step_id,
            hostname=self._hostname,
            elapsed_seconds=elapsed,
            cpu=cpu,
            memory=mem,
            gpus=gpus,
        )

    async def _collect_cpu(self, now: float) -> CpuMetrics:
        cores = self.job_ctx.cpus_allocated or 1
        if self._mock:
            elapsed = time.monotonic() - self._mock_start
            pct = 30 + 40 * (0.5 + 0.5 * math.sin(elapsed * 0.4))
            return CpuMetrics(
                cores_allocated=cores,
                usage_ns=int(pct * cores * 10_000_000 * max(elapsed, 0.1)),
                usage_percent=round(pct, 1),
            )
        usage_ns = self._read_cpu_ns()
        usage_pct = 0.0

        if usage_ns is not None and self._prev_cpu_ns is not None:
            dt = now - (self._prev_timestamp or now)
            if dt > 0:
                delta_ns = usage_ns - self._prev_cpu_ns
                max_possible_ns = dt * cores * 1_000_000_000
                raw_pct = (delta_ns / max_possible_ns) * 100.0 if max_possible_ns > 0 else 0.0
                usage_pct = min(100.0, raw_pct)

        self._prev_cpu_ns = usage_ns
        self._prev_timestamp = now

        return CpuMetrics(
            cores_allocated=cores,
            usage_ns=usage_ns or 0,
            usage_percent=round(usage_pct, 1),
        )

    def _read_cpu_ns(self) -> int | None:
        ctx = self.job_ctx
        if ctx.cgroup_v2_path:
            val = _read_cgroup_field(Path(ctx.cgroup_v2_path) / "cpu.stat", "usage_usec")
            if val is not None:
                return val * 1000
        if ctx.cgroup_v1_cpu_path:
            val = _read_int_file(Path(ctx.cgroup_v1_cpu_path) / "cpuacct.usage")
            return val
        return None

    async def _collect_memory(self) -> MemoryMetrics:
        ctx = self.job_ctx
        limit_bytes = ctx.mem_limit_bytes
        if self._mock:
            elapsed = time.monotonic() - self._mock_start
            pct = min(88, 25 + (elapsed / 11) * 63)
            current = int(pct / 100 * limit_bytes)
            peak = min(int(1.05 * current), limit_bytes)
            return MemoryMetrics(
                current_bytes=current,
                limit_bytes=limit_bytes,
                peak_bytes=peak,
                usage_percent=round(pct, 1),
                oom_guard_warning=pct >= 85,
                oom_guard_critical=pct >= 90,
            )
        current_bytes = 0
        peak_bytes = 0

        if ctx.cgroup_v2_path:
            current_bytes = _read_int_file(Path(ctx.cgroup_v2_path) / "memory.current") or 0
            peak_bytes = _read_int_file(Path(ctx.cgroup_v2_path) / "memory.peak") or 0
            raw_max = _read_cgroup_raw(Path(ctx.cgroup_v2_path) / "memory.max")
            if raw_max is not None and raw_max.strip() != "max":
                with contextlib.suppress(ValueError):
                    limit_bytes = int(raw_max.strip())

        elif ctx.cgroup_v1_mem_path:
            v1 = Path(ctx.cgroup_v1_mem_path)
            current_bytes = _read_int_file(v1 / "memory.usage_in_bytes") or 0
            peak_bytes = _read_int_file(v1 / "memory.max_usage_in_bytes") or 0
            limit_bytes = _read_int_file(v1 / "memory.limit_in_bytes") or limit_bytes

        usage_pct = 0.0
        if limit_bytes > 0:
            usage_pct = (current_bytes / limit_bytes) * 100.0

        return MemoryMetrics(
            current_bytes=current_bytes,
            limit_bytes=limit_bytes,
            peak_bytes=peak_bytes,
            usage_percent=round(usage_pct, 1),
            oom_guard_warning=usage_pct >= self.config.oom_warning_threshold * 100,
            oom_guard_critical=usage_pct >= self.config.oom_critical_threshold * 100,
        )

    async def _collect_gpus(self) -> list[GpuMetrics]:
        if self._mock:
            elapsed = time.monotonic() - self._mock_start
            return [
                GpuMetrics(
                    index=i,
                    uuid=f"GPU-demo-{i}",
                    name="NVIDIA A100-SXM4-80GB",
                    utilization_percent=round(
                        30 + 50 * (0.5 + 0.5 * math.sin(elapsed * 0.3 + i * 1.5)), 1
                    ),
                    memory_used_bytes=int(
                        (0.4 + 0.3 * (0.5 + 0.5 * math.sin(elapsed * 0.2 + i))) * 80 * 1024**3
                    ),
                    memory_total_bytes=80 * 1024**3,
                    memory_utilization_percent=round(
                        40 + 40 * (0.5 + 0.5 * math.sin(elapsed * 0.2 + i)), 1
                    ),
                    power_watts=round(
                        200 + 80 * (0.5 + 0.5 * math.sin(elapsed * 0.25 + i)), 1
                    ),
                    temperature_celsius=round(
                        55 + 20 * (0.5 + 0.5 * math.sin(elapsed * 0.15 + i)), 1
                    ),
                    throttling=False,
                )
                for i in range(4)
            ]
        if not self._nvml_initialized:
            return []
        import pynvml

        metrics: list[GpuMetrics] = []
        for handle in self._nvml_handles:
            try:
                idx = pynvml.nvmlDeviceGetIndex(handle)
                uuid = (
                    pynvml.nvmlDeviceGetUUID(handle).decode()
                    if isinstance(pynvml.nvmlDeviceGetUUID(handle), bytes)
                    else pynvml.nvmlDeviceGetUUID(handle)
                )
                name = (
                    pynvml.nvmlDeviceGetName(handle).decode()
                    if isinstance(pynvml.nvmlDeviceGetName(handle), bytes)
                    else pynvml.nvmlDeviceGetName(handle)
                )

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                mem_util_pct = 0.0
                if mem_info.total > 0:
                    mem_util_pct = (mem_info.used / mem_info.total) * 100.0

                power_w = 0.0
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_w = power_mw / 1000.0
                except pynvml.NVMLError:
                    pass

                temp_c = 0.0
                with contextlib.suppress(pynvml.NVMLError):
                    temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                throttling = self._check_gpu_throttling(handle)

                metrics.append(
                    GpuMetrics(
                        index=idx,
                        uuid=uuid,
                        name=name,
                        utilization_percent=round(util.gpu, 1),
                        memory_used_bytes=mem_info.used,
                        memory_total_bytes=mem_info.total,
                        memory_utilization_percent=round(mem_util_pct, 1),
                        power_watts=round(power_w, 1),
                        temperature_celsius=round(temp_c, 1),
                        throttling=throttling,
                    )
                )
            except Exception as exc:
                logger.debug("GPU metric collection failed for handle %s: %s", handle, exc)
                continue

        return metrics

    def _check_gpu_throttling(self, handle: object) -> bool:
        try:
            import pynvml

            temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            if temp_c >= self.config.gpu_temp_threshold_celsius:
                return True

            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w = power_mw / 1000.0
            if power_w >= self.config.gpu_power_threshold_watts:
                return True

            try:
                throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                if throttle_reasons:
                    sw_power_cap = pynvml.nvmlClocksThrottleReasonSwPowerCap
                    hw_thermal = pynvml.nvmlClocksThrottleReasonHwThermal
                    sw_thermal = pynvml.nvmlClocksThrottleReasonSwThermal
                    if throttle_reasons & (sw_power_cap | hw_thermal | sw_thermal):
                        return True
            except (pynvml.NVMLError, AttributeError):
                pass

        except Exception:
            pass
        return False

    @property
    def queue(self) -> asyncio.Queue[TelemetrySnapshot]:
        return self._queue


def _read_int_file(path: Path) -> int | None:
    try:
        data = path.read_text().strip()
        return int(data)
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        return None


def _read_cgroup_field(path: Path, key: str) -> int | None:
    try:
        data = path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None

    for line in data.split("\n"):
        line = line.strip()
        if line.startswith(key + " "):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    return None


def _read_cgroup_raw(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
