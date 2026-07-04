from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .config import SlurmwatchConfig
from .model import CpuMetrics, GpuMetrics, JobContext, MemoryMetrics, TelemetrySnapshot

if TYPE_CHECKING:
    from .slurm import RemoteUsage

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
        self._task: asyncio.Task[None] | None = None

        self._prev_cpu_ns: int | None = None
        self._prev_timestamp: float | None = None
        self._nvml_initialized = False
        self._nvml_shutdown_done = False
        self._nvml_handles: list[object] = []
        self._nvml_handle_info: dict[int, tuple[str, str]] = {}
        # For a remote (login-node) view, job_ctx.hostname is *this* host, not
        # where the job runs — report the job's actual node instead.
        if job_ctx.remote and job_ctx.nodelist_resolved:
            self._hostname = job_ctx.nodelist_resolved[0]
        else:
            self._hostname = job_ctx.hostname or socket.gethostname().split(".")[0]
        self._mock = os.environ.get("SLURMWATCH_MOCK") == "1"
        self._remote = job_ctx.remote
        self._mock_start = time.monotonic() if self._mock else 0.0
        self._peak_mem_running: int = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        # Remote sstat sampling is throttled (Slurm samples every ~30s and
        # each call is an RPC to the controller).
        self._remote_cache: tuple[float, RemoteUsage] | None = None
        self._remote_min_interval = 5.0

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        assert self._loop is not None
        if self._mock or self._remote:
            # Mock synthesizes data; remote has no local GPUs to query.
            self._nvml_initialized = False
        else:
            self._nvml_initialized = await self._loop.run_in_executor(None, self._init_nvml)
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self._shutdown_nvml()

    def stop_sync(self) -> None:
        self._stop_event.set()
        self._shutdown_nvml_sync()

    def _shutdown_nvml_sync(self) -> None:
        if not self._nvml_initialized or self._nvml_shutdown_done:
            return
        self._nvml_shutdown_done = True
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            pass
        self._nvml_handles.clear()
        self._nvml_handle_info.clear()

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

            visible_uuids = self.job_ctx.gpu_uuids
            visible_indices = self.job_ctx.gpu_indices
            if not visible_uuids and not visible_indices and self.job_ctx.gpu_count_requested == 0:
                # CPU-only job on a GPU node: attaching to every device would
                # display other users' workloads as this job's.
                pynvml.nvmlShutdown()
                logger.info("Job requested no GPUs; GPU monitoring disabled")
                return False
            if visible_uuids:
                for uuid_str in visible_uuids:
                    handle = self._handle_by_uuid(pynvml, uuid_str, device_count)
                    if handle is not None:
                        self._nvml_handles.append(handle)
                        self._cache_gpu_info(pynvml, handle)
            elif not visible_indices:
                for idx in range(device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        self._nvml_handles.append(handle)
                        self._cache_gpu_info(pynvml, handle)
                    except pynvml.NVMLError:
                        continue
            else:
                all_handles: list[object] = []
                for idx in range(device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        all_handles.append(handle)
                    except pynvml.NVMLError:
                        continue

                def _pci_bus_id_key(h: object) -> str:
                    try:
                        info = pynvml.nvmlDeviceGetPciInfo(h)
                        bid = info.busId
                        return bid.decode() if isinstance(bid, bytes) else bid
                    except Exception:
                        return ""

                all_handles.sort(key=_pci_bus_id_key)

                if device_count == len(visible_indices):
                    # ConstrainDevices: NVML already exposes only the job's GPUs,
                    # renumbered 0..N-1, so the node-global IDX list (e.g. [1] on
                    # a device NVML now calls 0) won't map. Every visible device
                    # belongs to the job, so attach them all.
                    for handle in all_handles:
                        self._nvml_handles.append(handle)
                        self._cache_gpu_info(pynvml, handle)
                else:
                    for ordinal in visible_indices:
                        if ordinal < len(all_handles):
                            handle = all_handles[ordinal]
                            self._nvml_handles.append(handle)
                            self._cache_gpu_info(pynvml, handle)

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

    def _cache_gpu_info(self, _pynvml: object, handle: object) -> None:
        import pynvml as nv

        try:
            idx = nv.nvmlDeviceGetIndex(handle)
            raw_uuid = nv.nvmlDeviceGetUUID(handle)
            uuid = raw_uuid.decode() if isinstance(raw_uuid, bytes) else raw_uuid
            raw_name = nv.nvmlDeviceGetName(handle)
            name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
            self._nvml_handle_info[idx] = (uuid, name)
        except nv.NVMLError:
            pass

    def _handle_by_uuid(self, _pynvml: object, uuid_str: str, device_count: int) -> object | None:
        import pynvml as nv

        with contextlib.suppress(Exception):
            handle: object = nv.nvmlDeviceGetHandleByUUID(uuid_str.encode())
            return handle
        for idx in range(device_count):
            try:
                handle = nv.nvmlDeviceGetHandleByIndex(idx)
                raw = nv.nvmlDeviceGetUUID(handle)
                this_uuid = raw.decode() if isinstance(raw, bytes) else raw
                if this_uuid == uuid_str:
                    return handle
            except nv.NVMLError:
                continue
        return None

    async def _shutdown_nvml(self) -> None:
        if not self._nvml_initialized or self._nvml_shutdown_done:
            return
        self._nvml_shutdown_done = True
        try:
            import pynvml

            if self._loop is not None:
                await self._loop.run_in_executor(None, pynvml.nvmlShutdown)
            else:
                pynvml.nvmlShutdown()
        except Exception:
            pass
        self._nvml_handles.clear()
        self._nvml_handle_info.clear()

    async def _run_loop(self) -> None:
        try:
            loop = self._loop
            assert loop is not None
            # Prime the CPU counter so the first snapshot (the only one
            # --once ever sees) reports a real delta instead of 0%, and let
            # a measurable window elapse so that delta isn't noise.
            await loop.run_in_executor(None, self._prime_cpu_baseline)
            if not self._mock:
                await asyncio.sleep(min(self.config.poll_interval, 0.2))
            while not self._stop_event.is_set():
                try:
                    snapshot = await loop.run_in_executor(None, self._collect_snapshot_sync)
                except Exception:
                    # One bad cycle (cgroup vanished mid-read, NVML hiccup)
                    # must not permanently end telemetry.
                    logger.exception("Snapshot collection failed; retrying")
                    await asyncio.sleep(self.config.poll_interval)
                    continue
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

    def _prime_cpu_baseline(self) -> None:
        if self._mock or self._remote:
            return
        usage_ns = self._read_cpu_ns(self._get_job_pids())
        if usage_ns is not None:
            self._prev_cpu_ns = usage_ns
            self._prev_timestamp = time.time()

    def _collect_snapshot_sync(self) -> TelemetrySnapshot:
        now = time.time()
        if self._remote:
            cpu, mem = self._collect_remote(now)
            gpus: list[GpuMetrics] = []
        else:
            # Enumerate the job's PIDs once; CPU (on clusters without a
            # cpuacct cgroup) and GPU attribution both need them.
            job_pids = set() if self._mock else self._get_job_pids()
            cpu = self._collect_cpu(now, job_pids)
            mem = self._collect_memory()
            gpus = self._collect_gpus(job_pids)
        elapsed = 0
        if self.job_ctx.job_start_time is not None:
            elapsed = int(now - self.job_ctx.job_start_time)

        node_count = max(len(self.job_ctx.nodelist_resolved), 1)
        node_index = 0
        hostname = socket.gethostname().split(".")[0]
        for i, node in enumerate(self.job_ctx.nodelist_resolved):
            if node == hostname:
                node_index = i
                break

        idle_threshold = self.config.gpu_idle_threshold
        active_gpus = sum(1 for g in gpus if _gpu_is_active(g, idle_threshold))

        return TelemetrySnapshot(
            timestamp=now,
            job_id=self.job_ctx.job_id,
            step_id=self.job_ctx.step_id,
            hostname=self._hostname,
            elapsed_seconds=elapsed,
            cpu=cpu,
            memory=mem,
            gpus=gpus,
            node_count=node_count,
            node_index=node_index,
            gpu_count_requested=self.job_ctx.gpu_count_requested,
            gpu_active_count=active_gpus,
        )

    def _collect_remote(self, now: float) -> tuple[CpuMetrics, MemoryMetrics]:
        """Build CPU/memory metrics from sstat when off the compute node.

        CPU is the average utilization since the job started (cumulative CPU
        time / elapsed / cores); memory is the peak RSS Slurm has sampled.
        """
        from .slurm import resolve_remote_usage

        cached = self._remote_cache
        if cached is not None and (now - cached[0]) < self._remote_min_interval:
            usage = cached[1]
        else:
            usage = resolve_remote_usage(self.job_ctx.job_id)
            self._remote_cache = (now, usage)

        ctx = self.job_ctx
        cores = ctx.cpus_allocated or 1
        node_count = max(len(ctx.nodelist_resolved), 1)
        elapsed = now - ctx.job_start_time if ctx.job_start_time else 0.0

        usage_pct = 0.0
        effective = 0.0
        if usage.cpu_seconds > 0 and elapsed > 0:
            # sstat CPU-seconds are job-wide (summed over all nodes), but cores
            # is per-node; divide to an approximate per-node average and clamp
            # so effective cores never exceed the node's allocation.
            effective = min((usage.cpu_seconds / elapsed) / node_count, float(cores))
            usage_pct = max(0.0, min(100.0, effective / cores * 100.0))
        cpu = CpuMetrics(
            cores_allocated=cores,
            usage_ns=int(usage.cpu_seconds * 1_000_000_000),
            usage_percent=round(usage_pct, 1),
            effective_cores=round(effective, 1),
        )

        limit = ctx.mem_limit_bytes
        rss = usage.rss_bytes
        mem_pct = (rss / limit * 100.0) if limit > 0 else 0.0
        mem = MemoryMetrics(
            current_bytes=rss,
            limit_bytes=limit,
            peak_bytes=rss,
            usage_percent=round(mem_pct, 1),
            oom_guard_warning=mem_pct >= self.config.oom_warning_threshold * 100,
            oom_guard_critical=mem_pct >= self.config.oom_critical_threshold * 100,
            working_set_bytes=rss,
            cache_bytes=0,
        )
        return cpu, mem

    def _collect_cpu(self, now: float, job_pids: set[int] | None = None) -> CpuMetrics:
        cores = self.job_ctx.cpus_allocated or 1
        if self._mock:
            elapsed = time.monotonic() - self._mock_start
            pct = 30 + 40 * (0.5 + 0.5 * math.sin(elapsed * 0.4))
            effective = pct * cores / 100.0
            return CpuMetrics(
                cores_allocated=cores,
                usage_ns=int(pct * cores * 10_000_000 * max(elapsed, 0.1)),
                usage_percent=round(pct, 1),
                effective_cores=round(effective, 1),
            )
        usage_ns = self._read_cpu_ns(job_pids)
        usage_pct = 0.0

        if usage_ns is not None and self._prev_cpu_ns is not None:
            dt = now - (self._prev_timestamp or now)
            if dt > 0:
                # Clamp the delta: the /proc fallback can shrink when a
                # process exits between samples.
                delta_ns = max(0, usage_ns - self._prev_cpu_ns)
                max_possible_ns = dt * cores * 1_000_000_000
                raw_pct = (delta_ns / max_possible_ns) * 100.0 if max_possible_ns > 0 else 0.0
                usage_pct = max(0.0, min(100.0, raw_pct))

        self._prev_cpu_ns = usage_ns
        self._prev_timestamp = now

        effective = usage_pct * cores / 100.0

        return CpuMetrics(
            cores_allocated=cores,
            usage_ns=usage_ns or 0,
            usage_percent=round(usage_pct, 1),
            effective_cores=round(effective, 1),
        )

    def _read_cpu_ns(self, job_pids: set[int] | None = None) -> int | None:
        """Cumulative CPU time (ns) for the job.

        Prefers the cgroup accounting controllers (which also capture children
        that have already exited), then falls back to summing /proc/<pid>/stat
        for the job's live PIDs — needed on clusters that constrain jobs with
        the cpuset controller but create no per-job cpuacct/cpu cgroup.
        """
        ctx = self.job_ctx
        if ctx.cgroup_v2_path:
            val = _read_cgroup_field(Path(ctx.cgroup_v2_path) / "cpu.stat", "usage_usec")
            if val is not None:
                return val * 1000
        if ctx.cgroup_v1_cpu_path:
            val = _read_int_file(Path(ctx.cgroup_v1_cpu_path) / "cpuacct.usage")
            if val is not None:
                return val
        if job_pids:
            return _proc_cpu_ns(job_pids)
        return None

    def _collect_memory(self) -> MemoryMetrics:
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
                working_set_bytes=current,
                cache_bytes=0,
            )
        current_bytes = 0
        peak_bytes = 0
        working_set_bytes = 0
        cache_bytes = 0

        if ctx.cgroup_v2_path:
            v2 = Path(ctx.cgroup_v2_path)
            current_bytes = _read_int_file(v2 / "memory.current") or 0

            peak_bytes = _read_int_file(v2 / "memory.peak") or 0
            if peak_bytes == 0:
                peak_bytes = self._peak_mem_running
                if current_bytes > self._peak_mem_running:
                    self._peak_mem_running = current_bytes
                    peak_bytes = current_bytes

            raw_max = _read_cgroup_raw(v2 / "memory.max")
            if raw_max is not None and raw_max.strip() != "max":
                with contextlib.suppress(ValueError):
                    limit_bytes = int(raw_max.strip())

            stat = _read_cgroup_raw(v2 / "memory.stat")
            if stat:
                inactive_file = 0
                slab_reclaimable = 0
                for line in stat.split("\n"):
                    line = line.strip()
                    if line.startswith("inactive_file "):
                        with contextlib.suppress(ValueError, IndexError):
                            inactive_file = int(line.split()[1])
                    elif line.startswith("slab_reclaimable "):
                        with contextlib.suppress(ValueError, IndexError):
                            slab_reclaimable = int(line.split()[1])
                working_set_bytes = max(0, current_bytes - inactive_file)
                cache_bytes = inactive_file + slab_reclaimable

            if limit_bytes == 0 or limit_bytes > 10**16:
                limit_bytes = _read_meminfo_total()

        elif ctx.cgroup_v1_mem_path:
            v1 = Path(ctx.cgroup_v1_mem_path)
            current_bytes = _read_int_file(v1 / "memory.usage_in_bytes") or 0
            peak_bytes = _read_int_file(v1 / "memory.max_usage_in_bytes") or 0
            if peak_bytes == 0:
                peak_bytes = self._peak_mem_running
                if current_bytes > self._peak_mem_running:
                    self._peak_mem_running = current_bytes
                    peak_bytes = current_bytes
            limit_bytes = _read_int_file(v1 / "memory.limit_in_bytes") or limit_bytes
            if limit_bytes == 0 or limit_bytes > 10**16:
                limit_bytes = _read_meminfo_total()
            # memory.usage_in_bytes counts reclaimable page cache, so subtract
            # the inactive file cache to get the working set that actually
            # drives OOM pressure (mirrors the v2 branch). v1 memory.stat uses
            # hierarchical total_* keys.
            working_set_bytes = current_bytes
            stat = _read_cgroup_raw(v1 / "memory.stat")
            if stat:
                inactive_file = 0
                total_cache = 0
                for line in stat.split("\n"):
                    line = line.strip()
                    if line.startswith("total_inactive_file "):
                        with contextlib.suppress(ValueError, IndexError):
                            inactive_file = int(line.split()[1])
                    elif line.startswith("total_cache "):
                        with contextlib.suppress(ValueError, IndexError):
                            total_cache = int(line.split()[1])
                working_set_bytes = max(0, current_bytes - inactive_file)
                cache_bytes = total_cache

        # A job's cgroup memory.max can be the whole node's RAM when the cluster
        # doesn't RAM-constrain the cgroup (ConstrainRAMSpace=no). The meaningful
        # ceiling the user cares about is then the memory Slurm actually reserved,
        # so never report a limit larger than the allocation.
        alloc = ctx.mem_limit_bytes
        if alloc > 0:
            limit_bytes = alloc if limit_bytes <= 0 else min(limit_bytes, alloc)

        if limit_bytes == 0:
            limit_bytes = _read_meminfo_total()

        usage_pct = 0.0
        if limit_bytes > 0:
            usage_pct = (current_bytes / limit_bytes) * 100.0

        ws_for_guard = working_set_bytes or current_bytes
        ws_pct = 0.0
        if limit_bytes > 0:
            ws_pct = (ws_for_guard / limit_bytes) * 100.0

        return MemoryMetrics(
            current_bytes=current_bytes,
            limit_bytes=limit_bytes,
            peak_bytes=peak_bytes,
            usage_percent=round(usage_pct, 1),
            oom_guard_warning=ws_pct >= self.config.oom_warning_threshold * 100,
            oom_guard_critical=ws_pct >= self.config.oom_critical_threshold * 100,
            working_set_bytes=working_set_bytes or current_bytes,
            cache_bytes=cache_bytes,
        )

    def _collect_gpus(self, job_pids: set[int] | None = None) -> list[GpuMetrics]:
        if self._mock:
            elapsed = time.monotonic() - self._mock_start
            total = 80 * 1024**3
            gpus: list[GpuMetrics] = []
            for i in range(4):
                used = int((0.4 + 0.3 * (0.5 + 0.5 * math.sin(elapsed * 0.2 + i))) * total)
                gpus.append(
                    GpuMetrics(
                        index=i,
                        uuid=f"GPU-demo-{i}",
                        name="NVIDIA A100-SXM4-80GB",
                        utilization_percent=round(
                            30 + 50 * (0.5 + 0.5 * math.sin(elapsed * 0.3 + i * 1.5)), 1
                        ),
                        memory_used_bytes=used,
                        memory_total_bytes=total,
                        memory_utilization_percent=round(used / total * 100.0, 1),
                        power_watts=round(200 + 80 * (0.5 + 0.5 * math.sin(elapsed * 0.25 + i)), 1),
                        temperature_celsius=round(
                            55 + 20 * (0.5 + 0.5 * math.sin(elapsed * 0.15 + i)), 1
                        ),
                        throttling=False,
                        process_utilization_percent=round(
                            30 + 50 * (0.5 + 0.5 * math.sin(elapsed * 0.3 + i * 1.5)), 1
                        ),
                        process_memory_bytes=int(used * 0.9),
                    )
                )
            return gpus
        if not self._nvml_initialized:
            return []
        import pynvml

        if job_pids is None:
            job_pids = self._get_job_pids()

        metrics: list[GpuMetrics] = []
        for handle in self._nvml_handles:
            try:
                idx = pynvml.nvmlDeviceGetIndex(handle)
                uuid, name = self._nvml_handle_info.get(idx, ("", ""))

                # Guard each sub-query individually: e.g. utilization rates
                # raise NOT_SUPPORTED on MIG devices, but memory, power, and
                # temperature are still worth reporting.
                util_pct = 0.0
                with contextlib.suppress(pynvml.NVMLError):
                    util_pct = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)

                mem_used = 0
                mem_total = 0
                with contextlib.suppress(pynvml.NVMLError):
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used = mem_info.used
                    mem_total = mem_info.total

                mem_util_pct = 0.0
                if mem_total > 0:
                    mem_util_pct = (mem_used / mem_total) * 100.0

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

                process_util = 0.0
                process_mem = 0
                if job_pids:
                    # usedGpuMemory is None (not missing) when NVML reports
                    # NVML_VALUE_NOT_AVAILABLE, e.g. on MIG devices.
                    try:
                        running_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        for proc in running_procs:
                            if proc.pid in job_pids:
                                process_mem += getattr(proc, "usedGpuMemory", 0) or 0
                    except (pynvml.NVMLError, AttributeError):
                        pass
                    try:
                        graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                        for proc in graphics_procs:
                            if proc.pid in job_pids:
                                process_mem += getattr(proc, "usedGpuMemory", 0) or 0
                    except (pynvml.NVMLError, AttributeError):
                        pass
                    try:
                        proc_util = pynvml.nvmlDeviceGetProcessUtilization(
                            handle, int((time.time() - 2) * 1e6)
                        )
                        # The job's share of the device is the SUM over its
                        # processes; the API may return several time-window
                        # samples per pid, so keep only the newest per pid.
                        latest: dict[int, tuple[int, float]] = {}
                        for p in proc_util:
                            if p.pid not in job_pids:
                                continue
                            ts = getattr(p, "timeStamp", 0)
                            if p.pid not in latest or ts >= latest[p.pid][0]:
                                latest[p.pid] = (ts, float(p.smUtil))
                        if latest:
                            process_util = min(100.0, sum(sm for _, sm in latest.values()))
                    except (pynvml.NVMLError, AttributeError):
                        pass

                metrics.append(
                    GpuMetrics(
                        index=idx,
                        uuid=uuid,
                        name=name,
                        utilization_percent=round(util_pct, 1),
                        memory_used_bytes=mem_used,
                        memory_total_bytes=mem_total,
                        memory_utilization_percent=round(mem_util_pct, 1),
                        power_watts=round(power_w, 1),
                        temperature_celsius=round(temp_c, 1),
                        throttling=throttling,
                        process_utilization_percent=round(process_util, 1),
                        process_memory_bytes=process_mem,
                    )
                )
            except Exception as exc:
                logger.debug("GPU metric collection failed for handle %s: %s", handle, exc)
                continue

        return metrics

    def _get_job_pids(self) -> set[int]:
        pids: set[int] = set()
        ctx = self.job_ctx

        def _read_procs(cg_path: Path) -> None:
            # On cgroup v2 processes live only in leaf cgroups
            # (job_X/step_Y/user/task_Z), so walk every descendant. The tree
            # can vanish mid-walk when the job ends, hence the broad OSError
            # guards.
            files = [cg_path / "cgroup.procs"]
            with contextlib.suppress(OSError):
                files.extend(cg_path.rglob("cgroup.procs"))
            for procs_file in files:
                data = _read_cgroup_raw(procs_file)
                if data:
                    for token in data.split():
                        if token.isdigit():
                            pids.add(int(token))

        if ctx.cgroup_v2_path:
            _read_procs(Path(ctx.cgroup_v2_path))
        if ctx.cgroup_v1_cpu_path:
            _read_procs(Path(ctx.cgroup_v1_cpu_path))
        if ctx.cgroup_v1_mem_path:
            _read_procs(Path(ctx.cgroup_v1_mem_path))
        # Never count the monitor itself as job workload — it shares the job's
        # cgroup when launched inside the allocation (e.g. after an srun hop).
        pids.discard(os.getpid())
        return pids

    def _check_gpu_throttling(self, handle: object) -> bool:
        try:
            import pynvml

            def _const(*names: str) -> int:
                for name in names:
                    value = getattr(pynvml, name, None)
                    if isinstance(value, int):
                        return value
                return 0

            try:
                throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                if throttle_reasons:
                    # Constant names differ across pynvml releases
                    # (ThrottleReason* vs the newer EventReason* spellings).
                    throttle_mask = (
                        _const(
                            "nvmlClocksThrottleReasonSwPowerCap",
                            "nvmlClocksEventReasonSwPowerCap",
                        )
                        | _const(
                            "nvmlClocksThrottleReasonHwThermalSlowdown",
                            "nvmlClocksEventReasonHwThermalSlowdown",
                        )
                        | _const(
                            "nvmlClocksThrottleReasonSwThermalSlowdown",
                            "nvmlClocksEventReasonSwThermalSlowdown",
                        )
                        | _const(
                            "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",
                            "nvmlClocksEventReasonHwPowerBrakeSlowdown",
                        )
                        | _const(
                            "nvmlClocksThrottleReasonHwSlowdown",
                            "nvmlClocksEventReasonHwSlowdown",
                        )
                    )
                    if throttle_reasons & throttle_mask:
                        return True
            except (pynvml.NVMLError, AttributeError):
                pass
        except Exception:
            pass
        return False

    @property
    def queue(self) -> asyncio.Queue[TelemetrySnapshot]:
        return self._queue


_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _gpu_is_active(g: GpuMetrics, idle_threshold: float) -> bool:
    """Whether the job is actively using this GPU.

    Prefer the job's per-process utilization, but fall back to device
    utilization when the job holds VRAM on it — per-process sampling
    (nvmlDeviceGetProcessUtilization) is optional and frequently returns
    nothing on a single poll or unsupported driver, and a missing sample must
    not flip a clearly-busy GPU to IDLE.
    """
    if g.process_utilization_percent > idle_threshold:
        return True
    return g.process_memory_bytes > 0 and g.utilization_percent > idle_threshold


def _parse_stat_cpu_ticks(data: str) -> int:
    """utime + stime (clock ticks) from the contents of /proc/<pid>/stat.

    The comm field (2nd) may contain spaces and parentheses, so the fields
    after it are located relative to the final ')'.
    """
    rparen = data.rfind(")")
    if rparen == -1:
        return 0
    fields = data[rparen + 1 :].split()
    # After comm, fields are: state(0) ppid(1) ... utime(11) stime(12) ...
    if len(fields) < 13:
        return 0
    try:
        return int(fields[11]) + int(fields[12])
    except ValueError:
        return 0


def _read_pid_cpu_ticks(pid: int) -> int:
    """utime + stime (in clock ticks) for a PID from /proc/<pid>/stat."""
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return 0
    return _parse_stat_cpu_ticks(data)


def _proc_cpu_ns(pids: set[int]) -> int:
    """Cumulative CPU time (ns) summed over live PIDs, via /proc/<pid>/stat."""
    total_ticks = sum(_read_pid_cpu_ticks(pid) for pid in pids)
    return total_ticks * 1_000_000_000 // _CLK_TCK


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


def _read_meminfo_total() -> int:
    try:
        data = Path("/proc/meminfo").read_text()
        for line in data.split("\n"):
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        pass
    return 0
