from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from .config import SlurmwatchConfig
from .model import (
    CpuMetrics,
    GpuInterconnect,
    GpuMetrics,
    JobContext,
    MemoryMetrics,
    TelemetrySnapshot,
    local_node_name,
    short_host,
)

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
        # /proc CPU fallback (no cpuacct cgroup): a monotonic accumulator of CPU
        # ticks plus each PID's last-seen ticks, so a child that exits between
        # polls doesn't erase its work from the running total (which would make a
        # busy job read 0% — the counter must only ever climb, like the cgroup).
        self._proc_cpu_seen: dict[int, int] = {}
        self._proc_cpu_accum_ticks: int = 0
        # PID -> consecutive polls it's been absent from the sampled set, so a
        # briefly-missed live PID isn't forgotten (and re-added whole) on return.
        self._proc_cpu_absent: dict[int, int] = {}
        self._nvml_initialized = False
        # NVML is functional on this node (init OK + devices present), regardless of
        # whether the job's own GPUs attach — the signal for the "no GPU telemetry
        # here" vs "GPU held by srun" message distinction (F3).
        self._nvml_functional = False
        self._nvml_shutdown_done = False
        self._nvml_handles: list[object] = []
        self._nvml_handle_info: dict[int, tuple[str, str]] = {}
        # Index cached per handle, aligned with _nvml_handles, so a transient
        # nvmlDeviceGetIndex failure mid-collection doesn't drop the whole GPU
        # for that cycle (B-P7).
        self._nvml_indices: list[int] = []
        # GPU interconnect (NVLink/PCIe topology). The wiring is fixed for the job,
        # so probe it once and cache the static part; only live throughput is
        # recomputed each frame. ``_interconnect_probed`` guards the one-time build
        # (so a node with no NVLink doesn't re-probe every cycle). ``_nvlink_prev``
        # holds each device's last (timestamp, rx_kib, tx_kib) counter reading to
        # turn the cumulative NVLink byte counters into a live MiB/s rate.
        self._interconnect_static: GpuInterconnect | None = None
        self._interconnect_probed = False
        self._nvlink_prev: dict[int, tuple[float, int, int]] = {}
        # Serializes every NVML call so a shutdown can never run concurrently
        # with an in-flight _collect_gpus in the executor thread (B-C2).
        self._nvml_lock = threading.Lock()
        # The executor future for the collection currently in flight; awaited
        # (bounded) on stop() so teardown doesn't race a running collection.
        self._inflight_collect: asyncio.Future[TelemetrySnapshot] | None = None
        # For a remote (login-node) view, job_ctx.hostname is *this* host, not
        # where the job runs — report the job's actual node instead.
        if job_ctx.remote and job_ctx.nodelist_resolved:
            self._hostname = job_ctx.nodelist_resolved[0]
        else:
            self._hostname = job_ctx.hostname or local_node_name()
        self._mock = os.environ.get("SLURMWATCH_MOCK") == "1"
        self._remote = job_ctx.remote
        # Login-node-hop contention detector (best-effort): only the hop's own
        # monitor step (env set by cli._hop_to_compute_node) scans the job's PIDs
        # for a stalled launcher, so a normal on-node run pays nothing for it.
        self.launcher_present: bool = False
        self._detect_launchers = os.environ.get("SLURMWATCH_MONITOR_STEP") == "1"
        self._mock_start = time.monotonic() if self._mock else 0.0
        self._peak_mem_running: int = 0
        # Peak cores ever busy at once — a monotonic running max for right-sizing
        # --cpus-per-task. Unlike the memory peak (which _collect_memory reads from
        # the cgroup's own lifetime counter), there is no kernel counter for a
        # concurrent-core peak, so this one covers only the current sw session.
        self._peak_effective_cores: float = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        # Remote sstat sampling is throttled (Slurm samples every ~30s and
        # each call is an RPC to the controller).
        # (cache_ts, usage, elapsed_at_sample): the elapsed is frozen alongside the
        # usage so remote "avg cores" (cpu_seconds/elapsed) doesn't slide downward
        # between throttled samples or during a transient sstat outage (N9).
        self._remote_cache: tuple[float, RemoteUsage, float] | None = None
        self._remote_min_interval = 5.0
        # Job-liveness recheck: resolve_job_context runs once, so a job that ends
        # while attached would otherwise freeze the dashboard at its last numbers
        # with an ever-climbing elapsed (#28). A DEDICATED task re-asks Slurm on a
        # throttle and latches `_job_ended` so the TUI can show a banner and the
        # headless logger can exit. It's separate from the snapshot loop on
        # purpose: a single squeue can take >15s on a busy controller, and running
        # it inline would stall the live telemetry feed for that whole time. Local
        # on-node view only (remote/sstat can't tell "ended" from "not yet
        # sampled"); mock runs forever.
        self._job_ended = False
        self._liveness_min_interval = 15.0
        self._liveness_task: asyncio.Task[None] | None = None

    @property
    def job_ended(self) -> bool:
        """True once Slurm reports the monitored job is no longer running (#28)."""
        return self._job_ended

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        assert self._loop is not None
        if self._mock or self._remote:
            # Mock synthesizes data; remote has no local GPUs to query.
            self._nvml_initialized = False
        else:
            self._nvml_initialized = await self._loop.run_in_executor(None, self._init_nvml)
        self._task = asyncio.create_task(self._run_loop())
        # Poll job liveness so any live view can announce "JOB ENDED" and stop.
        # This must run for the remote (login-node) dashboard too (A1): the earlier
        # "remote can't tell ended from not-yet-sampled" reasoning conflated two
        # things — is_job_active polls Slurm job STATE (squeue/sacct), which is
        # independent of whether sstat has sampled usage. Without it the remote
        # dashboard never latches job_ended and retries srun against a dead job
        # forever. Only a demo (synthetic data, runs forever) skips it.
        if not self._mock:
            self._liveness_task = asyncio.create_task(self._liveness_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        # Capture the in-flight collection BEFORE cancelling the task. Cancelling
        # unwinds the poll loop, whose `finally` sets self._inflight_collect =
        # None, so reading it *after* the await would always see None and silently
        # skip the graceful wait below (C1).
        fut = self._inflight_collect
        if self._liveness_task is not None:
            self._liveness_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._liveness_task
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        # Cancelling the task doesn't stop the executor thread it left running,
        # so let that collection finish (bounded) before we shut NVML down; the
        # NVML lock guarantees mutual exclusion, this just makes teardown
        # graceful and avoids an unretrieved-exception warning (B-C2).
        if fut is not None and not fut.done():
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(fut), timeout=2.0)
        await self._shutdown_nvml()

    def stop_sync(self) -> None:
        self._stop_event.set()
        self._shutdown_nvml_sync()

    def _nvml_shutdown_locked(self) -> None:
        """Call nvmlShutdown while holding the NVML lock (bounded).

        Serializes with any in-flight _collect_gpus so we never call into NVML
        concurrently with it (B-C2). The acquire is bounded so a wedged
        collection can't hang teardown; if it can't be acquired we shut down
        anyway (the process is on its way out).
        """
        import pynvml

        acquired = self._nvml_lock.acquire(timeout=2.0)
        try:
            pynvml.nvmlShutdown()
        finally:
            if acquired:
                self._nvml_lock.release()

    def _shutdown_nvml_sync(self) -> None:
        if not self._nvml_initialized or self._nvml_shutdown_done:
            return
        self._nvml_shutdown_done = True
        with contextlib.suppress(Exception):
            self._nvml_shutdown_locked()
        self._nvml_handles.clear()
        self._nvml_handle_info.clear()
        self._nvml_indices.clear()

    async def next_snapshot(self) -> TelemetrySnapshot:
        return await self._queue.get()

    @property
    def is_mock(self) -> bool:
        """Whether this collector synthesizes demo data (SLURMWATCH_MOCK / --demo)."""
        return self._mock

    def mock_snapshot_for_node(self, node: str) -> TelemetrySnapshot:
        """A synthesized snapshot stamped for ``node`` — for demo node-switching.

        There is no real cluster in --demo mode, so switching nodes can't srun into
        another host; synthesize that node's frame locally instead, keeping the
        switch instant and free of the (meaningless) "still reaching / unreachable"
        watchdog on a fake node.
        """
        return self._collect_snapshot_sync(node_override=node)

    def _init_nvml(self) -> bool:
        # A CPU-only job never needs NVML, so don't even load it: otherwise a node
        # without the NVIDIA driver emits a scary "NVML Shared Library Not Found"
        # line for a job that wasn't using a GPU in the first place.
        ctx = self.job_ctx
        if not ctx.gpu_uuids and not ctx.gpu_indices and ctx.gpu_count_requested == 0:
            logger.info("Job requested no GPUs; GPU monitoring disabled")
            return False

        try:
            import pynvml
        except ImportError:
            logger.info("pynvml not installed; GPU monitoring disabled")
            return False

        try:
            pynvml.nvmlInit()
        except Exception as exc:
            # No NVIDIA driver / NVML library on this node (a login node, a
            # CPU-only node) is a normal condition, not a fault — note it quietly
            # at INFO instead of a loud WARNING with a cryptic library error. A
            # genuine, unexpected NVML failure still warns.
            if type(exc).__name__ in ("NVMLError_LibraryNotFound", "NVMLError_DriverNotLoaded"):
                logger.info("No NVIDIA driver on this node; GPU monitoring off")
            else:
                logger.warning("NVML init failed: %s", exc)
            return False

        # NVML is live from here on. Mark it initialized *now* so that cleanup
        # runs even if the awaiting task is cancelled before start() records the
        # return value (B-C5) or if the enumeration below raises (B-P6). Any
        # early return that decides GPU monitoring is off shuts NVML back down.
        self._nvml_initialized = True

        try:
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                logger.info("No NVIDIA devices detected by NVML")
                self._shutdown_nvml_sync()
                return False

            # NVML works here (init succeeded, devices present) even if the job's own
            # GPUs turn out not to be attachable below (F3).
            self._nvml_functional = True

            # The CPU-only case (no uuids/indices and 0 GPUs requested) returned
            # before NVML was ever initialised, so here the job wants GPUs.
            visible_uuids = self.job_ctx.gpu_uuids
            visible_indices = self.job_ctx.gpu_indices
            if visible_uuids:
                for uuid_str in visible_uuids:
                    handle = self._handle_by_uuid(pynvml, uuid_str, device_count)
                    if handle is not None:
                        self._attach_handle(pynvml, handle)
            elif not visible_indices:
                # No specific indices/UUIDs resolved, but the job did request
                # GPUs (the CPU-only case returned early above). Enumerate the
                # node's devices; if the job asked for fewer than the node has,
                # attaching every device would show other users' GPUs on a
                # shared node, so cap to the requested count in PCI-bus order.
                all_handles: list[object] = []
                for idx in range(device_count):
                    try:
                        all_handles.append(pynvml.nvmlDeviceGetHandleByIndex(idx))
                    except pynvml.NVMLError:
                        continue
                want = self.job_ctx.gpu_count_requested
                if want and want < len(all_handles):
                    all_handles.sort(key=self._pci_bus_id_key)
                    all_handles = all_handles[:want]
                for handle in all_handles:
                    self._attach_handle(pynvml, handle)
            else:
                all_handles = []
                for idx in range(device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        all_handles.append(handle)
                    except pynvml.NVMLError:
                        continue

                all_handles.sort(key=self._pci_bus_id_key)

                if device_count == len(visible_indices):
                    # ConstrainDevices: NVML already exposes only the job's GPUs,
                    # renumbered 0..N-1, so the node-global IDX list (e.g. [1] on
                    # a device NVML now calls 0) won't map. Every visible device
                    # belongs to the job, so attach them all.
                    for handle in all_handles:
                        self._attach_handle(pynvml, handle)
                else:
                    for ordinal in visible_indices:
                        if ordinal < len(all_handles):
                            handle = all_handles[ordinal]
                            self._attach_handle(pynvml, handle)

            logger.info(
                "NVML initialized: %d/%d GPUs visible",
                len(self._nvml_handles),
                device_count,
            )
            return True

        except Exception as exc:
            # nvmlInit() succeeded but enumeration failed; shut NVML back down so
            # it isn't left initialized (B-P6).
            logger.warning("NVML device enumeration failed: %s", exc)
            self._shutdown_nvml_sync()
            return False

    def _pci_bus_id_key(self, handle: object) -> str:
        """PCI bus id for a handle, used to order NVML devices deterministically.

        NVML's per-index order is not guaranteed to match CUDA's PCI-bus order,
        so sorting by bus id gives a stable, CUDA-ordinal-comparable sequence.
        """
        import pynvml as nv

        try:
            info = nv.nvmlDeviceGetPciInfo(handle)
            bid = info.busId
            return bid.decode() if isinstance(bid, bytes) else bid
        except Exception:
            return ""

    def _attach_handle(self, _pynvml: object, handle: object) -> None:
        """Record a handle plus its cached index/uuid/name, kept aligned.

        _nvml_handles and _nvml_indices are appended together so that a later,
        transient nvmlDeviceGetIndex failure during collection can fall back to
        the index cached here instead of dropping the GPU (B-P7).
        """
        import pynvml as nv

        self._nvml_handles.append(handle)
        idx = -1
        uuid = ""
        name = ""
        try:
            idx = nv.nvmlDeviceGetIndex(handle)
            raw_uuid = nv.nvmlDeviceGetUUID(handle)
            uuid = raw_uuid.decode() if isinstance(raw_uuid, bytes) else raw_uuid
            raw_name = nv.nvmlDeviceGetName(handle)
            name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
        except nv.NVMLError:
            pass
        self._nvml_indices.append(idx)
        if idx >= 0:
            self._nvml_handle_info[idx] = (uuid, name)

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
            if self._loop is not None:
                await self._loop.run_in_executor(None, self._nvml_shutdown_locked)
            else:
                self._nvml_shutdown_locked()
        except Exception:
            pass
        self._nvml_handles.clear()
        self._nvml_handle_info.clear()
        self._nvml_indices.clear()

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
                    self._inflight_collect = loop.run_in_executor(None, self._collect_snapshot_sync)
                    snapshot = await self._inflight_collect
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # One bad cycle (cgroup vanished mid-read, NVML hiccup)
                    # must not permanently end telemetry.
                    logger.exception("Snapshot collection failed; retrying")
                    await asyncio.sleep(self.config.poll_interval)
                    continue
                finally:
                    self._inflight_collect = None
                self._enqueue(snapshot)
                await asyncio.sleep(self.config.poll_interval)
        except asyncio.CancelledError:
            pass

    async def _liveness_loop(self) -> None:
        """Poll Slurm for job liveness on its own cadence and latch ``_job_ended``.

        Runs separately from the snapshot loop so a slow squeue (>15s on a busy
        controller) never stalls the live telemetry feed (#28). Waits a full
        interval first (the job was just resolved as running), then rechecks;
        stops at the first ``False`` (ended) — an unknown result (Slurm slow or
        unreachable) is ignored so a transient failure can't tear down a live
        dashboard. Each check runs in the executor so it never blocks the loop.
        """
        from .slurm import is_job_active

        loop = self._loop
        assert loop is not None
        job_id = self.job_ctx.raw_job_id or self.job_ctx.job_id
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._liveness_min_interval)
                return  # stop requested during the wait
            except asyncio.TimeoutError:
                pass  # interval elapsed -> time for a check
            try:
                active = await loop.run_in_executor(None, is_job_active, job_id)
            except Exception:
                logger.debug("Liveness check failed; will retry", exc_info=True)
                continue
            if active is False:
                logger.info("Job %s is no longer running; telemetry stopped.", self.job_ctx.job_id)
                self._job_ended = True
                return

    def _enqueue(self, snapshot: TelemetrySnapshot) -> None:
        """Put a snapshot on the bounded queue, dropping the oldest if full.

        The dashboard consumes at its own pace; when it stalls the queue fills,
        and the freshest sample matters most, so evict the oldest rather than
        block or discard the new one.
        """
        try:
            self._queue.put_nowait(snapshot)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self._queue.put_nowait(snapshot)

    def _prime_cpu_baseline(self) -> None:
        if self._mock or self._remote:
            return
        usage_ns = self._read_cpu_ns(self._get_job_pids())
        if usage_ns is not None:
            self._prev_cpu_ns = usage_ns
            self._prev_timestamp = time.monotonic()

    def _collect_snapshot_sync(self, node_override: str | None = None) -> TelemetrySnapshot:
        now = time.time()
        if self._remote:
            cpu, mem = self._collect_remote(now)
            gpus: list[GpuMetrics] = []
            self.launcher_present = False
        else:
            # Enumerate the job's PIDs once; CPU (on clusters without a
            # cpuacct cgroup) and GPU attribution both need them.
            job_pids = set() if self._mock else self._get_job_pids()
            cpu = self._collect_cpu(job_pids)
            mem = self._collect_memory()
            self._apply_peaks(cpu, mem)
            gpus = self._collect_gpus(job_pids)
            # Is a new srun/mpirun the user just started stuck behind our own
            # held step? Only the monitor step scans, never in mock mode.
            self.launcher_present = (
                self._detect_launchers and not self._mock and _any_launcher_pid(job_pids)
            )
        elapsed = 0
        if self.job_ctx.job_start_time is not None:
            # Clamp to >= 0: a just-started job with compute-node clock skew can make
            # now < job_start_time, which otherwise rendered "ran -1:59:56" / "-0%"
            # on the dashboard and wrote a negative elapsed_seconds to CSV (N10).
            elapsed = max(0, int(now - self.job_ctx.job_start_time))

        node_count = max(len(self.job_ctx.nodelist_resolved), 1)
        if node_override is not None:
            # Demo/mock: synthesize a frame stamped for THIS node so switching is
            # instant (no real cluster to srun into).
            stamp_host = node_override
            node_index = next(
                (
                    i
                    for i, n in enumerate(self.job_ctx.nodelist_resolved)
                    if short_host(n) == short_host(node_override)
                ),
                0,
            )
        else:
            stamp_host = self._hostname
            node_index = 0
            local = local_node_name()
            for i, n in enumerate(self.job_ctx.nodelist_resolved):
                if short_host(n) == local:
                    node_index = i
                    break

        idle_threshold = self.config.gpu_idle_threshold
        active_gpus = sum(1 for g in gpus if _gpu_is_active(g, idle_threshold))

        # Only a multi-GPU node has an interconnect to report; a CPU-only, single-GPU,
        # or off-node (sstat) sample leaves it None.
        interconnect = None
        if not self._remote and len(gpus) > 1:
            interconnect = self._collect_interconnect(gpus)

        return TelemetrySnapshot(
            timestamp=now,
            job_id=self.job_ctx.job_id,
            step_id=self.job_ctx.step_id,
            hostname=stamp_host,
            elapsed_seconds=elapsed,
            cpu=cpu,
            memory=mem,
            gpus=gpus,
            node_count=node_count,
            node_index=node_index,
            gpu_count_requested=self.job_ctx.gpu_count_requested,
            gpu_active_count=active_gpus,
            remote=self._remote,
            gpu_monitoring_available=self._nvml_functional,
            interconnect=interconnect,
        )

    def _collect_remote(self, now: float) -> tuple[CpuMetrics, MemoryMetrics]:
        """Build CPU/memory metrics from sstat when off the compute node.

        CPU is the average utilization since the job started (cumulative CPU
        time / elapsed / cores); memory is the peak RSS Slurm has sampled
        (``MaxRSS``) — a lifetime high-water mark, not a live "current". Because
        it can only ever climb, it must NOT drive the OOM warning/critical guard:
        a job that briefly spiked and then dropped would otherwise show a red
        "near limit" banner that can never clear (#34). The snapshot is tagged
        ``remote=True`` so the UI labels this bar "peak" (not "used") and readers
        of the structured output know it's a job-wide estimate, not per-node
        telemetry (#35).
        """
        from .slurm import resolve_remote_usage

        ctx = self.job_ctx
        node_count = max(len(ctx.nodelist_resolved), 1)

        cached = self._remote_cache
        if cached is not None and (now - cached[0]) < self._remote_min_interval:
            usage, sample_elapsed = cached[1], cached[2]
        else:
            # resolve_remote_usage returns per-node estimates (sstat totals are
            # job-wide; it scales by an estimated per-node task count). Query with
            # the raw numeric JobId, not the user-facing form: `sstat -j 12345` and
            # `sstat -j 12345_3` both expand to EVERY running task of an array, so
            # their steps get summed and CPU time is over-reported N-fold; only the
            # underlying numeric JobId scopes the sample to this one task (#30).
            # raw_job_id is unset for a demo/mock context (where sstat isn't
            # called anyway), so fall back to job_id there.
            sstat_id = self.job_ctx.raw_job_id or self.job_ctx.job_id
            fresh = resolve_remote_usage(sstat_id, node_count)
            elapsed_now = now - ctx.job_start_time if ctx.job_start_time else 0.0
            if not fresh.sampled and cached is not None and cached[1].sampled:
                # sstat failed transiently (a busy controller) — keep the last REAL
                # sample AND the elapsed it was computed against, so remote "avg
                # cores" (cpu_seconds/elapsed) holds steady instead of decaying every
                # frame while the numerator is frozen but a now-based elapsed keeps
                # growing (N9). Bump the timestamp so we retry after the interval, not
                # every frame, and never cache the failed reading.
                usage, sample_elapsed = cached[1], cached[2]
                self._remote_cache = (now, usage, sample_elapsed)
            else:
                usage, sample_elapsed = fresh, elapsed_now
                self._remote_cache = (now, fresh, elapsed_now)

        cores = ctx.cpus_allocated or 1
        # The elapsed captured WITH the sample, not a fresh now-based one: the average
        # is cpu_seconds/elapsed as of the sample, so both only move together when a
        # new sample lands — no per-frame decay between samples / during an outage (N9).
        elapsed = sample_elapsed

        usage_pct = 0.0
        effective = 0.0
        if usage.cpu_seconds > 0 and elapsed > 0:
            # cpu_seconds is already a per-node estimate; clamp so effective
            # cores never exceed the node's allocation.
            effective = min(usage.cpu_seconds / elapsed, float(cores))
            usage_pct = max(0.0, min(100.0, effective / cores * 100.0))
        cpu = CpuMetrics(
            cores_allocated=cores,
            usage_ns=int(usage.cpu_seconds * 1_000_000_000),
            usage_percent=round(usage_pct, 1),
            effective_cores=round(effective, 1),
        )

        limit = ctx.mem_limit_bytes
        rss = usage.rss_bytes  # sstat MaxRSS: a lifetime peak, not a live current
        # Clamp like the on-node path (F1): rss is MaxRSS x tasks_per_node, which can
        # exceed the limit for a memory-imbalanced step, yielding an impossible
        # >100% in --json / the remote summary.
        mem_pct = min(100.0, rss / limit * 100.0) if limit > 0 else 0.0
        mem = MemoryMetrics(
            current_bytes=rss,
            limit_bytes=limit,
            peak_bytes=rss,
            usage_percent=round(mem_pct, 1),
            # A monotonic high-water mark must never drive the OOM guard: it would
            # latch a red "near limit" alarm that can't clear after the job's real
            # RSS drops (#34). The peak fraction is still shown honestly in the row.
            oom_guard_warning=False,
            oom_guard_critical=False,
            working_set_bytes=rss,
            cache_bytes=0,
        )
        return cpu, mem

    def _apply_peaks(self, cpu: CpuMetrics, mem: MemoryMetrics) -> None:
        """Fold the CPU high-water mark into a freshly-collected local snapshot and
        keep the memory peak self-consistent.

        Memory peak is the number a user sizes ``--mem`` against, so it must be the
        job's TRUE lifetime maximum — not just what we happened to see since
        attaching. ``_collect_memory`` already reports that from the cgroup's own
        lifetime counter (v1 ``memory.max_usage_in_bytes`` / v2 ``memory.peak``),
        which survives sw restarts and covers the whole job even on a late attach;
        on the rare kernel exposing neither it falls back to a running max of usage.
        We do NOT recompute it here — folding in the smaller live working set would
        drag a late-attached job's peak DOWN below its real high-water mark — we
        only ensure it never reads below the current usage, so ``max >= used`` holds.

        CPU peak = the most cores ever busy at once. No kernel counter exists for
        it, so it is a monotonic running max over the current sw session."""
        if not self._mock:
            # Mock keeps _collect_memory's demo peak (a little headroom over "used"),
            # so the demo GIF still shows a peak bar distinct from the used bar.
            mem.peak_bytes = max(mem.peak_bytes, mem.current_bytes)

        if cpu.effective_cores > self._peak_effective_cores:
            self._peak_effective_cores = cpu.effective_cores
        cpu.peak_effective_cores = round(self._peak_effective_cores, 1)

    def _collect_cpu(self, job_pids: set[int] | None = None) -> CpuMetrics:
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

        # Use a MONOTONIC clock for the rate window: a wall-clock (time.time())
        # step backward from an NTP correction would give dt<=0 and drop the
        # sample to 0% (or a huge spike on a forward jump).
        mono = time.monotonic()
        if usage_ns is not None and self._prev_cpu_ns is not None:
            dt = mono - (self._prev_timestamp or mono)
            if dt > 0:
                # Clamp the delta: the /proc fallback can shrink when a
                # process exits between samples.
                delta_ns = max(0, usage_ns - self._prev_cpu_ns)
                max_possible_ns = dt * cores * 1_000_000_000
                raw_pct = (delta_ns / max_possible_ns) * 100.0 if max_possible_ns > 0 else 0.0
                usage_pct = max(0.0, min(100.0, raw_pct))

        self._prev_cpu_ns = usage_ns
        self._prev_timestamp = mono

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
            return self._accumulated_proc_cpu_ns(job_pids)
        return None

    def _accumulated_proc_cpu_ns(self, pids: set[int]) -> int:
        """Monotonic cumulative CPU time (ns) over the job's PIDs, via /proc.

        Summing utime+stime over only the *currently-live* PIDs is non-monotonic:
        when a busy child exits between two polls its accumulated ticks vanish, the
        delta goes negative, and the caller's ``max(0, …)`` clamp turns a fully
        busy interval into 0% — badly wrong for jobs that churn short-lived
        children (``make -j``, shell pipelines, per-file loops) on a cpuset-only
        cluster with no cpuacct cgroup. Instead accumulate the *forward* delta of
        each PID and keep an exited PID's contribution in the running total, so the
        value only ever increases (mirroring the cgroup counter).
        """
        for pid in pids:
            cur = _read_pid_cpu_ticks(pid)
            if cur <= 0:
                continue
            prev = self._proc_cpu_seen.get(pid, 0)
            # cur >= prev: normal forward progress. cur < prev: the PID number was
            # reused by a new process — count its ticks as fresh (from 0).
            self._proc_cpu_accum_ticks += cur - prev if cur >= prev else cur
            self._proc_cpu_seen[pid] = cur
            self._proc_cpu_absent.pop(pid, None)
        # A PID absent from THIS poll is not dropped right away: a still-live PID can
        # briefly fall out of the sampled set (an enumeration race, a one-off
        # /proc/<pid>/stat read miss), and forgetting its last tick count would make
        # it look brand-new next poll and re-add its whole history — a spurious CPU
        # spike (a double-count). Keep the value; only evict once it's been gone long
        # enough to be certainly dead, which merely bounds memory (its ticks already
        # live in the accumulator, so eviction never changes the total).
        for pid in list(self._proc_cpu_seen):
            if pid in pids:
                continue
            gone = self._proc_cpu_absent.get(pid, 0) + 1
            if gone >= _PROC_CPU_EVICT_POLLS:
                del self._proc_cpu_seen[pid]
                self._proc_cpu_absent.pop(pid, None)
            else:
                self._proc_cpu_absent[pid] = gone
        return self._proc_cpu_accum_ticks * 1_000_000_000 // _CLK_TCK

    def _proc_rss_bytes(self) -> int:
        """Sum resident memory across the job's processes from /proc.

        Fallback for a cgroup with no memory controller delegated (no
        memory.current) so MEM isn't a misleading 0 (F4) — the memory analogue of
        the CPU /proc fallback. Best-effort: pids that vanish mid-read are skipped.
        statm's resident field counts shared pages too, so this can slightly
        over-report, but it beats reporting nothing.
        """
        total = 0
        for pid in self._get_job_pids():
            try:
                resident = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
            except (OSError, ValueError, IndexError):
                continue
            total += resident * _PAGE_SIZE
        return total

    def _collect_memory(self) -> MemoryMetrics:
        ctx = self.job_ctx
        limit_bytes = ctx.mem_limit_bytes
        if self._mock:
            elapsed = time.monotonic() - self._mock_start
            # Climb to a healthy, well-utilised ~72% and plateau — deliberately
            # BELOW the 85% OOM-warn threshold so the demo/showcase never trips a
            # (false) amber "MEMORY nn% of limit" alarm on a job that's perfectly fine.
            pct = min(72, 25 + (elapsed / 11) * 47)
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
            current_raw = _read_int_file(v2 / "memory.current")
            if current_raw is None:
                # The cgroup exists but has no memory controller delegated (e.g.
                # task/cgroup without ConstrainRAMSpace): memory.current is absent.
                # Sum the job's process RSS from /proc so MEM isn't a misleading 0
                # (F4) — the memory analogue of the CPU /proc fallback. Because the
                # discovered cgroup now *succeeds*, we'd otherwise never fall back to
                # sstat, so a 0 here would stick.
                current_bytes = self._proc_rss_bytes()
                working_set_bytes = current_bytes
            else:
                current_bytes = current_raw

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
                working_set_bytes, cache_bytes = _working_set_from_stat(stat, current_bytes, "")

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
            # memory.usage_in_bytes counts reclaimable page cache; subtract the
            # file-backed cache to get the working set that drives OOM pressure.
            # v1 memory.stat uses hierarchical total_* keys.
            working_set_bytes = current_bytes
            stat = _read_cgroup_raw(v1 / "memory.stat")
            if stat:
                working_set_bytes, cache_bytes = _working_set_from_stat(
                    stat, current_bytes, "total_"
                )

        # `limit_bytes` currently holds the cgroup's enforced limit (memory.max /
        # limit_in_bytes), which is where the kernel actually OOM-kills.
        cgroup_limit = limit_bytes

        # Report and guard against the memory Slurm allocated (what the user
        # requested and what accounting shows), not the raw cgroup limit — that
        # can be the whole node's RAM (ConstrainRAMSpace=no), which is confusing
        # ("196 of 200 GiB requested"). But the job dies at the cgroup limit, so
        # when that limit is *below* the allocation (a tighter enforced cap), it
        # is the real ceiling: use the smaller of the two so the OOM guard can't
        # under-warn against a too-generous allocation figure (F5).
        alloc = ctx.mem_limit_bytes
        if alloc > 0 and cgroup_limit > 0:
            limit_bytes = min(alloc, cgroup_limit)
        elif alloc > 0:
            limit_bytes = alloc

        if limit_bytes == 0:
            limit_bytes = _read_meminfo_total()

        usage_pct = 0.0
        if limit_bytes > 0:
            # Clamp: current_bytes includes reclaimable page cache, which can push
            # RSS+cache above the cgroup limit and yield an impossible >100% "used"
            # in --json/--once output. The OOM guards below use the working set.
            usage_pct = min(100.0, (current_bytes / limit_bytes) * 100.0)

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
        # Hold the NVML lock for the whole sweep so nvmlShutdown (teardown) can
        # never run concurrently with these calls (B-C2).
        with self._nvml_lock:
            for pos, handle in enumerate(self._nvml_handles):
                try:
                    # nvmlDeviceGetIndex can raise transiently; fall back to the
                    # index cached at attach time so a single hiccup doesn't drop
                    # the whole GPU (and flicker the device count) for the cycle
                    # (B-P7).
                    cached_idx = self._nvml_indices[pos] if pos < len(self._nvml_indices) else -1
                    try:
                        idx = pynvml.nvmlDeviceGetIndex(handle)
                    except pynvml.NVMLError:
                        idx = cached_idx if cached_idx >= 0 else pos
                    uuid, name = self._nvml_handle_info.get(idx, ("", ""))

                    # Guard each sub-query individually: e.g. utilization rates
                    # raise NOT_SUPPORTED on MIG devices, but memory, power, and
                    # temperature are still worth reporting. Track whether the
                    # utilization read actually succeeded so a MIG device the job
                    # is using isn't scored as idle purely because util is
                    # unreadable (B-P3).
                    util_pct = 0.0
                    util_available = True
                    try:
                        util_pct = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    except pynvml.NVMLError:
                        util_available = False

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
                        temp_c = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )

                    throttling = self._check_gpu_throttling(handle)

                    process_util = 0.0
                    process_mem = 0
                    if job_pids:
                        # A PID can appear in both the compute and graphics
                        # process lists (e.g. a CUDA+OpenGL app); key the memory
                        # by PID and take the max so it's counted once, not
                        # doubled (B-P5). usedGpuMemory is None (not missing)
                        # when NVML reports NVML_VALUE_NOT_AVAILABLE, e.g. on MIG.
                        mem_by_pid: dict[int, int] = {}
                        for getter in (
                            "nvmlDeviceGetComputeRunningProcesses",
                            "nvmlDeviceGetGraphicsRunningProcesses",
                        ):
                            with contextlib.suppress(pynvml.NVMLError, AttributeError):
                                for proc in getattr(pynvml, getter)(handle):
                                    if proc.pid in job_pids:
                                        used = getattr(proc, "usedGpuMemory", 0) or 0
                                        mem_by_pid[proc.pid] = max(
                                            mem_by_pid.get(proc.pid, 0), used
                                        )
                        process_mem = sum(mem_by_pid.values())
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
                            utilization_available=util_available,
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

    # -- GPU interconnect (NVLink / PCIe topology) ---------------------------

    def _collect_interconnect(self, gpus: list[GpuMetrics]) -> GpuInterconnect | None:
        """The GPU↔GPU interconnect for this multi-GPU node: NVLink generation and
        speed, the pairwise NVLink/PCIe topology matrix, and live NVLink traffic.

        The wiring is fixed for the life of the job, so the matrix + speed are
        probed once and cached; only throughput is re-read each frame. Every NVML
        call is guarded — a PCIe-only box, or a driver that doesn't expose NVLink,
        degrades to a PCIe topology (or None), never a crash.
        """
        if self._mock:
            return self._mock_interconnect(gpus)
        if not self._nvml_initialized:
            return None
        import pynvml

        # One lock hold for the whole sweep so nvmlShutdown can't run mid-probe
        # (B-C2); the helpers below assume the caller holds it and never re-acquire.
        with self._nvml_lock:
            if not self._interconnect_probed:
                self._interconnect_probed = True
                try:
                    self._interconnect_static = self._build_topology()
                except Exception as exc:
                    logger.debug("interconnect topology probe failed: %s", exc)
                    self._interconnect_static = None
            static = self._interconnect_static
            if static is None:
                return None
            # Attach fresh live traffic to a copy so cached snapshots keep the rate
            # they were built with (the static object is shared across frames). Read
            # whichever fabric connects the GPUs: NVLink counters and/or the live
            # PCIe meter.
            nv_rx: list[float] = []
            nv_tx: list[float] = []
            if static.fabric in ("nvlink", "mixed"):
                nv_rx, nv_tx = self._nvlink_throughput(pynvml, static.devices)
            pcie_rx: list[float] = []
            pcie_tx: list[float] = []
            if static.fabric in ("pcie", "mixed"):
                pcie_rx, pcie_tx = self._pcie_throughput(pynvml, static.devices)
            return replace(
                static,
                nvlink_rx_gbps=nv_rx,
                nvlink_tx_gbps=nv_tx,
                pcie_rx_gbps=pcie_rx,
                pcie_tx_gbps=pcie_tx,
            )

    def _build_topology(self) -> GpuInterconnect | None:
        """Probe the static NVLink/PCIe wiring once. Caller holds the NVML lock."""
        import pynvml as nv

        # Order the job's handles by PCI bus id so the matrix rows/cols are stable
        # and comparable to CUDA ordinals (matching _init_nvml's ordering).
        entries: list[tuple[int, object, str]] = []
        for pos, handle in enumerate(self._nvml_handles):
            idx = self._nvml_indices[pos] if pos < len(self._nvml_indices) else pos
            entries.append((idx, handle, self._pci_bus_id_key(handle)))
        entries.sort(key=lambda e: (e[2] or "", e[0]))
        if len(entries) < 2:
            return None
        devices = [e[0] for e in entries]
        handles = [e[1] for e in entries]
        # Key by the domain-normalized bus id so a link's remote endpoint matches one
        # of our devices even if NVML formats the two PCI-info structs' busId with a
        # different domain width (e.g. "0000:07:..." vs "00000000:07:...").
        pos_by_bus = {_norm_bus(e[2]): i for i, e in enumerate(entries) if e[2]}
        n = len(entries)

        # nvlink[i][j] = NVLinks from device i whose remote end is our device j.
        nvlink = [[0] * n for _ in range(n)]
        switch_links = [0] * n  # links terminating on an NVSwitch (all-to-all fabric)
        active_links = [0] * n  # total active NVLinks on device i
        version = 0
        for i, handle in enumerate(handles):
            for link in range(nv.NVML_NVLINK_MAX_LINKS):
                try:
                    state = nv.nvmlDeviceGetNvLinkState(handle, link)
                except (nv.NVMLError, AttributeError):
                    # NOT_SUPPORTED on a PCIe-only card, or the whole API missing.
                    continue
                if state != nv.NVML_FEATURE_ENABLED:
                    continue
                active_links[i] += 1
                if version == 0:
                    with contextlib.suppress(nv.NVMLError, AttributeError):
                        version = int(nv.nvmlDeviceGetNvLinkVersion(handle, link))
                remote_is_switch = False
                with contextlib.suppress(nv.NVMLError, AttributeError):
                    rtype = nv.nvmlDeviceGetNvLinkRemoteDeviceType(handle, link)
                    remote_is_switch = rtype == nv.NVML_NVLINK_DEVICE_TYPE_SWITCH
                if remote_is_switch:
                    switch_links[i] += 1
                    continue
                with contextlib.suppress(nv.NVMLError, AttributeError):
                    rbus = self._decode(nv.nvmlDeviceGetNvLinkRemotePciInfo(handle, link).busId)
                    j = pos_by_bus.get(_norm_bus(rbus))
                    if j is not None and j != i:
                        nvlink[i][j] += 1

        nvswitch = any(switch_links)
        speed_mbps, fv_link_count = self._nvlink_speed(nv, handles[0])
        links_per_gpu = max([*active_links, fv_link_count], default=0)
        link_speed_gbps = speed_mbps / 1000.0 if speed_mbps else _NVLINK_GEN_GBPS.get(version, 0.0)
        per_gpu_gbps = links_per_gpu * link_speed_gbps * 2  # bidirectional aggregate

        matrix = [["self"] * n for _ in range(n)]
        any_nv = any_pcie = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                direct = max(nvlink[i][j], nvlink[j][i])
                if direct == 0 and nvswitch and switch_links[i] and switch_links[j]:
                    # All-to-all through the NVSwitch fabric: the full link budget
                    # is available between any pair (as nvidia-smi topo -m reports).
                    direct = min(switch_links[i], switch_links[j])
                if direct > 0:
                    matrix[i][j] = f"NV{direct}"
                    any_nv = True
                else:
                    matrix[i][j] = self._pcie_class(nv, handles[i], handles[j])
                    any_pcie = True

        fabric = "nvlink" if any_nv and not any_pcie else "mixed" if any_nv else "pcie"
        return GpuInterconnect(
            fabric=fabric,
            nvlink_version=version,
            links_per_gpu=links_per_gpu if any_nv else 0,
            link_speed_gbps=round(link_speed_gbps, 1) if any_nv else 0.0,
            per_gpu_gbps=round(per_gpu_gbps, 1) if any_nv else 0.0,
            nvswitch=nvswitch,
            devices=devices,
            matrix=matrix,
        )

    def _pcie_class(self, nv: object, h1: object, h2: object) -> str:
        """nvidia-smi-style PCIe path label between two devices (PIX…SYS)."""
        with contextlib.suppress(Exception):
            lvl = nv.nvmlDeviceGetTopologyCommonAncestor(h1, h2)  # type: ignore[attr-defined]
            return _TOPO_LABEL.get(lvl, "?")
        return "?"

    def _nvlink_speed(self, nv: object, handle: object) -> tuple[int, int]:
        """(per-link MB/s common, active link count) from field values; 0 if absent."""
        speed = count = 0
        with contextlib.suppress(Exception):
            vals = nv.nvmlDeviceGetFieldValues(  # type: ignore[attr-defined]
                handle,
                [nv.NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON, nv.NVML_FI_DEV_NVLINK_LINK_COUNT],  # type: ignore[attr-defined]
            )
            for v in vals:
                if v.nvmlReturn != 0:  # not NVML_SUCCESS
                    continue
                val = int(_field_value(v))
                if v.fieldId == nv.NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON:  # type: ignore[attr-defined]
                    speed = val
                elif v.fieldId == nv.NVML_FI_DEV_NVLINK_LINK_COUNT:  # type: ignore[attr-defined]
                    count = val
        return speed, count

    def _nvlink_throughput(self, nv: object, devices: list[int]) -> tuple[list[float], list[float]]:
        """Live per-device NVLink (RX, TX) in GB/s from the cumulative DATA counters.

        The THROUGHPUT_DATA_* fields are cumulative KiB, so a rate is the delta
        between two reads over the elapsed time; the first read seeds the baseline
        and yields 0. Returns empty lists when the counters aren't readable (older
        driver, no permission, PCIe-only) so the UI can hide the live line."""
        rx_out: list[float] = []
        tx_out: list[float] = []
        now = time.time()
        got_any = False
        for idx in devices:
            handle = self._handle_for_device(idx)
            rx_kib = tx_kib = -1
            if handle is not None:
                with contextlib.suppress(Exception):
                    vals = nv.nvmlDeviceGetFieldValues(  # type: ignore[attr-defined]
                        handle,
                        [
                            nv.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX,  # type: ignore[attr-defined]
                            nv.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,  # type: ignore[attr-defined]
                        ],
                    )
                    for v in vals:
                        if v.nvmlReturn != 0:
                            continue
                        val = int(_field_value(v))
                        if v.fieldId == nv.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX:  # type: ignore[attr-defined]
                            rx_kib = val
                        elif v.fieldId == nv.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX:  # type: ignore[attr-defined]
                            tx_kib = val
            rx_rate = tx_rate = 0.0
            if rx_kib >= 0 and tx_kib >= 0:
                got_any = True
                prev = self._nvlink_prev.get(idx)
                if prev is not None:
                    dt = now - prev[0]
                    if dt > 0:
                        # KiB delta over dt → GB/s (decimal): *1024 bytes /dt /1e9.
                        # Clamp deltas at 0 so a counter reset (e.g. driver reload)
                        # reads as a lull, not a huge negative spike.
                        rx_rate = max(rx_kib - prev[1], 0) * 1024.0 / dt / 1e9
                        tx_rate = max(tx_kib - prev[2], 0) * 1024.0 / dt / 1e9
                self._nvlink_prev[idx] = (now, rx_kib, tx_kib)
            # 3 decimals (≈1 MB/s), not 1: the UI sums these across devices, so
            # rounding each to 0.1 GB/s first would lose real aggregate traffic
            # (three GPUs at 0.04 GB/s each → 0.0+0.0+0.0 instead of 0.1). The
            # display rounds the sum to 0.1 GB/s.
            rx_out.append(round(rx_rate, 3))
            tx_out.append(round(tx_rate, 3))
        if not got_any:
            return [], []
        return rx_out, tx_out

    def _pcie_throughput(self, nv: object, devices: list[int]) -> tuple[list[float], list[float]]:
        """Live per-device PCIe (RX, TX) in GB/s from ``nvmlDeviceGetPcieThroughput``.

        That call is already a live rate (measured over a ~20ms window, reported in
        KB/s), so no delta/state is needed — it covers all of the device's PCIe
        traffic (host↔GPU plus any peer-to-peer over PCIe). Returns empty lists when
        it isn't readable so the UI can hide the line."""
        rx_out: list[float] = []
        tx_out: list[float] = []
        got_any = False
        for idx in devices:
            handle = self._handle_for_device(idx)
            rx_kbps = tx_kbps = -1
            if handle is not None:
                with contextlib.suppress(Exception):
                    rx_kbps = int(
                        nv.nvmlDeviceGetPcieThroughput(handle, nv.NVML_PCIE_UTIL_RX_BYTES)  # type: ignore[attr-defined]
                    )
                    tx_kbps = int(
                        nv.nvmlDeviceGetPcieThroughput(handle, nv.NVML_PCIE_UTIL_TX_BYTES)  # type: ignore[attr-defined]
                    )
            if rx_kbps >= 0 and tx_kbps >= 0:
                got_any = True
                # KB/s → GB/s (decimal): *1000 bytes /1e9 = /1e6. 3 decimals (not 1)
                # so the per-device values stay accurate when the UI sums them across
                # devices — see _nvlink_throughput; the display rounds the sum to 0.1.
                rx_out.append(round(rx_kbps / 1e6, 3))
                tx_out.append(round(tx_kbps / 1e6, 3))
            else:
                rx_out.append(0.0)
                tx_out.append(0.0)
        if not got_any:
            return [], []
        return rx_out, tx_out

    def _handle_for_device(self, idx: int) -> object | None:
        for pos, di in enumerate(self._nvml_indices):
            if di == idx and pos < len(self._nvml_handles):
                return self._nvml_handles[pos]
        return None

    @staticmethod
    def _decode(raw: object) -> str:
        return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)

    def _mock_interconnect(self, gpus: list[GpuMetrics]) -> GpuInterconnect:
        """A DGX-style 4×A100 NVSwitch fabric for the demo (NVLink 3, 12 links,
        600 GB/s), with gently varying live traffic (GB/s) so the fabric line moves."""
        devices = [g.index for g in gpus]
        n = len(devices)
        matrix = [["self" if i == j else "NV12" for j in range(n)] for i in range(n)]
        elapsed = time.monotonic() - self._mock_start
        rx = [round(60 + 180 * (0.5 + 0.5 * math.sin(elapsed * 0.3 + i)), 1) for i in range(n)]
        tx = [round(45 + 160 * (0.5 + 0.5 * math.cos(elapsed * 0.25 + i)), 1) for i in range(n)]
        return GpuInterconnect(
            fabric="nvlink",
            nvlink_version=3,
            links_per_gpu=12,
            link_speed_gbps=25.0,
            per_gpu_gbps=600.0,
            nvswitch=True,
            devices=devices,
            matrix=matrix,
            nvlink_rx_gbps=rx,
            nvlink_tx_gbps=tx,
        )

    @property
    def queue(self) -> asyncio.Queue[TelemetrySnapshot]:
        return self._queue


_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

# NVML topology-common-ancestor level → nvidia-smi topo -m PCIe path label, fastest
# (PIX, single PCIe bridge) to slowest (SYS, across NUMA/QPI/UPI). NVML_TOPOLOGY_CPU
# shares NODE's value (40); both mean "same NUMA node, different host bridge".
_TOPO_LABEL = {0: "self", 10: "PIX", 20: "PXB", 30: "PHB", 40: "NODE", 50: "SYS"}

# Per-link, one-direction GB/s by NVLink generation — a fallback used only when the
# driver doesn't expose the exact per-link speed (SPEED_MBPS_COMMON). 1=P100, 2=V100,
# 3=A100, 4=H100/H200, 5=B200.
_NVLINK_GEN_GBPS = {1: 20.0, 2: 25.0, 3: 25.0, 4: 25.0, 5: 50.0}


def _norm_bus(busid: str) -> str:
    """A PCI bus id reduced to its ``bus:device.function`` tail, lower-cased.

    NVML's ``nvmlDeviceGetPciInfo`` and ``nvmlDeviceGetNvLinkRemotePciInfo`` both
    fill a ``busId`` like ``00000000:07:00.0``, but the domain width can differ
    between calls/drivers; dropping the domain makes "is this link's far end one of
    my GPUs?" a reliable match within a node (domains beyond 0 are vanishingly rare
    on GPU hosts).
    """
    parts = busid.strip().lower().split(":")
    return ":".join(parts[-2:]) if len(parts) >= 2 else busid.strip().lower()


def _field_value(v: object) -> float:
    """Read the active member of an nvmlFieldValue's tagged union by its valueType."""
    import pynvml as nv

    t = v.valueType  # type: ignore[attr-defined]
    val = v.value  # type: ignore[attr-defined]
    if t == nv.NVML_VALUE_TYPE_DOUBLE:
        return float(val.dVal)
    if t == nv.NVML_VALUE_TYPE_UNSIGNED_INT:
        return float(val.uiVal)
    if t == nv.NVML_VALUE_TYPE_UNSIGNED_LONG:
        return float(val.ulVal)
    if t == nv.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
        return float(val.ullVal)
    if t == nv.NVML_VALUE_TYPE_SIGNED_LONG_LONG:
        return float(val.sllVal)
    return 0.0


# How many consecutive polls a PID may be absent from the sampled set before the
# /proc CPU accumulator forgets its last-seen tick count. A still-live PID can
# briefly drop out (enumeration race, a transient stat read miss); until then we
# keep its value so a reappearance computes a correct delta instead of re-adding
# its whole history. Once it's been gone this long it's certainly dead, so we
# evict it purely to bound memory (its ticks already live in the running total).
_PROC_CPU_EVICT_POLLS = 8


def _working_set_from_stat(stat: str, current_bytes: int, prefix: str) -> tuple[int, int]:
    """(working_set, reclaimable_file_cache) from a cgroup memory.stat.

    File-backed page cache (inactive_file + active_file) is clean and reclaimed
    by the kernel before it OOM-kills a job, so it's excluded from the working
    set that drives the OOM guard (leaving anonymous + shmem). cgroup v1 uses
    hierarchical ``total_``-prefixed keys; v2 keys have no prefix.
    """
    keys = {f"{prefix}inactive_file": 0, f"{prefix}active_file": 0}
    for line in stat.split("\n"):
        parts = line.split()
        if len(parts) >= 2 and parts[0] in keys:
            with contextlib.suppress(ValueError):
                keys[parts[0]] = int(parts[1])
    reclaimable = keys[f"{prefix}inactive_file"] + keys[f"{prefix}active_file"]
    return max(0, current_bytes - reclaimable), reclaimable


def _gpu_is_active(g: GpuMetrics, idle_threshold: float) -> bool:
    """Whether the job is actively using this GPU.

    Prefer the job's per-process utilization. Per-process sampling
    (nvmlDeviceGetProcessUtilization) is optional and frequently returns
    nothing on a single poll or unsupported driver, so fall back to device
    utilization — but only when the job is the GPU's primary tenant (holds the
    majority of the used VRAM). That avoids crediting another user's load on a
    shared, non-isolated GPU while still catching a busy GPU the job owns.
    """
    if g.process_utilization_percent > idle_threshold:
        return True
    if not g.utilization_available:
        # Device-wide utilization couldn't be read (e.g. a MIG slice, where the
        # rate APIs return NOT_SUPPORTED). Without a util reading, "0%" is
        # meaningless, so fall back to VRAM occupancy. Per-process VRAM
        # (process_memory_bytes) is *also* frequently NOT_AVAILABLE on MIG, so
        # don't rely on it alone — that made an actively-used slice read as "idle"
        # (crit) whenever NVML withheld both signals (#36). Fall back to the
        # slice's own used VRAM too: on a MIG device that memory is isolated to
        # this slice, so it's a clean "in use" signal. Only a slice with no
        # readable activity at all (no util, no process VRAM, no used VRAM) is
        # scored inactive. This branch never runs for a normal shared GPU (there
        # utilization_available is True), so it can't over-credit another tenant.
        return g.process_memory_bytes > 0 or g.memory_used_bytes > 0
    if g.utilization_percent > idle_threshold and g.memory_used_bytes > 0:
        if g.process_memory_bytes > 0:
            # Per-process VRAM is readable: require the job to own the majority of
            # it, so we don't credit another user's load on a shared, non-isolated
            # GPU.
            return g.process_memory_bytes >= 0.5 * g.memory_used_bytes
        # Per-process VRAM is 0 = NVML withheld it (containerized jobs where PIDs
        # are namespaced, vGPU, or NO_PERMISSION on the process APIs), not a truly
        # idle job — a genuinely idle job wouldn't peg device utilization. We can't
        # judge ownership, so score a pegged GPU with used VRAM as active. On the
        # common cgroup-isolated (ConstrainDevices) GPU it's the job's anyway;
        # this avoids false "GPU IDLE" on a fully-busy GPU (the container case).
        return True
    return False


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


# Top-level MPI/srun launcher *clients* — the processes that block at step
# creation when a monitor step already holds the allocation's cores. The
# per-node daemons a *running* step spawns (hydra_pmi_proxy / orted / prted) are
# deliberately excluded: seeing those means a step already launched, so the job
# isn't stuck. The kernel caps comm at 15 chars, so keep the names short.
_LAUNCHER_COMMS = frozenset(
    {"srun", "mpirun", "mpiexec", "mpiexec.hydra", "mpirun.hydra", "orterun", "ibrun", "prun"}
)


def _read_pid_comm(pid: int) -> str:
    """The process name (comm) for a PID from /proc/<pid>/comm ('' on error)."""
    try:
        return Path(f"/proc/{pid}/comm").read_text(errors="replace").strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return ""


def _any_launcher_pid(pids: set[int]) -> bool:
    """True if any PID looks like a top-level MPI/srun launcher client (see _LAUNCHER_COMMS)."""
    return any(_read_pid_comm(pid) in _LAUNCHER_COMMS for pid in pids)


def _read_pid_cpu_ticks(pid: int) -> int:
    """utime + stime (in clock ticks) for a PID from /proc/<pid>/stat."""
    try:
        data = Path(f"/proc/{pid}/stat").read_text(errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return 0
    return _parse_stat_cpu_ticks(data)


def _proc_cpu_ns(pids: set[int]) -> int:
    """Cumulative CPU time (ns) summed over live PIDs, via /proc/<pid>/stat."""
    total_ticks = sum(_read_pid_cpu_ticks(pid) for pid in pids)
    return total_ticks * 1_000_000_000 // _CLK_TCK


def _read_int_file(path: Path) -> int | None:
    try:
        data = path.read_text(errors="replace").strip()
        return int(data)
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        return None


def _read_cgroup_field(path: Path, key: str) -> int | None:
    try:
        data = path.read_text(errors="replace").strip()
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
        return path.read_text(errors="replace").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _read_meminfo_total() -> int:
    try:
        data = Path("/proc/meminfo").read_text(errors="replace")
        for line in data.split("\n"):
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        pass
    return 0
