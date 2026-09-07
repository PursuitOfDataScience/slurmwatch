from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import re
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from .aio import join_bounded
from .config import SlurmwatchConfig
from .model import (
    CpuMetrics,
    GpuInterconnect,
    GpuMetrics,
    JobContext,
    MemoryMetrics,
    NodeFabric,
    TelemetrySnapshot,
    local_node_name,
    short_host,
)

if TYPE_CHECKING:
    from .slurm import RemoteUsage

logger = logging.getLogger(__name__)

# The NVIDIA driver exposes one directory per physical GPU here, readable from
# *any* process on the node: it is plain procfs, so it survives the cgroup device
# ACL that hides /dev/nvidiaN from a step which was allocated no GPU. That makes it
# the one way to tell "this node has no NVIDIA GPU" (a CPU node, an AMD node) apart
# from "this node has NVIDIA GPUs that this step may not open" — two situations NVML
# reports identically, as nvmlInit() succeeds through /dev/nvidiactl and then
# enumerates zero devices. Without the distinction a monitor step on a GPU node
# blamed a missing driver for what is really Slurm withholding the devices.
_NVIDIA_PROC_GPUS = Path("/proc/driver/nvidia/gpus")


# The kernel exposes every RDMA HCA here, world-readable, with no device to open —
# so a monitor can read the node's inter-NODE fabric regardless of what Slurm did
# or didn't allocate it. This is the number that explains a slow multi-node step:
# gradient all-reduce crosses this fabric, and NVML's PCIe counters never see it
# (GPUDirect RDMA moves data GPU->NIC without appearing as host PCIe traffic).
_IB_SYSFS = Path("/sys/class/infiniband")

# port_xmit_data/port_rcv_data are counted in units of FOUR OCTETS per the
# InfiniBand spec (IBTA vol1, "PortCounters"), not bytes. Reading them as bytes
# under-reports the fabric by exactly 4x.
_IB_COUNTER_OCTETS = 4

# How long teardown will wait to JOIN a cancelled task before giving up on it.
# cancel() only unwinds a coroutine at a suspension point, and both of this
# collector's tasks spend their time inside run_in_executor — a thread that has
# already started is not cancellable, so `cancel(); await task` lasts exactly as
# long as the shell-out it is sitting in (squeue for liveness, sstat/ssh/NVML for a
# collection). A busy controller is the condition this tool exists to watch, and
# stop() runs on the Ctrl-C path, where "slow" is indistinguishable from "hung".
# See aio.join_bounded for why the wait is asyncio.wait and not wait_for.
_TEARDOWN_JOIN_SECONDS = 2.0


class _IbPort(NamedTuple):
    """One active RDMA port, with counters already converted to bytes."""

    device: str
    port: str
    rx_bytes: int
    tx_bytes: int
    rate_gbps: float
    rate_label: str
    kind: str


def _ib_ports(root: Path | None = None) -> list[_IbPort]:
    """Every ACTIVE RDMA port on this node: counters, link rate, and transport.

    Returns one entry per active port with ``rx_octets``/``tx_octets`` already
    converted to bytes. Ports that are down are skipped — their counters are stale
    and would dilute the rate. Never raises: an absent ``/sys/class/infiniband`` (a
    node with no HCA, or a non-Linux host) is simply "no fabric".
    """
    root = _IB_SYSFS if root is None else root
    out: list[_IbPort] = []
    try:
        devices = sorted(p.name for p in root.iterdir())
    except OSError:
        return out

    def _read(path: Path) -> str:
        try:
            return path.read_text(errors="replace").strip()
        except OSError:
            return ""

    for dev in devices:
        ports_dir = root / dev / "ports"
        try:
            ports = sorted(p.name for p in ports_dir.iterdir())
        except OSError:
            continue
        for port in ports:
            pdir = ports_dir / port
            # "4: ACTIVE" — anything else (DOWN, INIT, ARMED) has stale counters.
            if "ACTIVE" not in _read(pdir / "state").upper():
                continue
            rx_raw, tx_raw = (
                _read(pdir / "counters" / "port_rcv_data"),
                _read(pdir / "counters" / "port_xmit_data"),
            )
            if not rx_raw.isdigit() or not tx_raw.isdigit():
                continue
            # "100 Gb/sec (2X HDR)" -> 100.0, keeping the HCA's own words for the UI.
            rate_label = _read(pdir / "rate")
            rate_gbps = 0.0
            head = rate_label.split()[0] if rate_label else ""
            with contextlib.suppress(ValueError):
                rate_gbps = float(head)
            link_layer = _read(pdir / "link_layer") or ""
            out.append(
                _IbPort(
                    device=dev,
                    port=port,
                    rx_bytes=int(rx_raw) * _IB_COUNTER_OCTETS,
                    tx_bytes=int(tx_raw) * _IB_COUNTER_OCTETS,
                    rate_gbps=rate_gbps,
                    rate_label=rate_label,
                    kind="RoCE" if link_layer.lower().startswith("ether") else "InfiniBand",
                )
            )
    return out


def _common_gpu_model(models: list[str]) -> str:
    """One label for a node's GPUs: the shared model, or "" if they disagree.

    Nodes are almost always homogeneous, so collapsing to a single name keeps the
    UI short. A genuinely mixed node returns "" rather than picking a device
    arbitrarily and mislabelling the rest.
    """
    distinct = {m for m in models if m}
    return distinct.pop() if len(distinct) == 1 else ""


def _nvidia_node_gpu_models(root: Path | None = None) -> list[str]:
    """Model name of every NVIDIA GPU physically present on this node.

    One entry per device in PCI-address order, e.g.
    ``["NVIDIA A100-PCIE-40GB", ...]``; ``[]`` when the node has no NVIDIA driver
    (so the caller can keep saying "no NVIDIA GPU here" when that is the truth).
    Never raises — an unreadable or absent procfs is simply "nothing known".

    ``root`` defaults to :data:`_NVIDIA_PROC_GPUS` but is resolved at call time, not
    bound as a default argument: a default would freeze the path at import and make
    the module constant unpatchable, so every test would silently read the *real*
    /proc of whatever machine it ran on and pass or fail by accident.
    """
    root = _NVIDIA_PROC_GPUS if root is None else root
    models: list[str] = []
    try:
        entries = sorted(p.name for p in root.iterdir())
    except OSError:
        return models
    for name in entries:
        model = ""
        try:
            with open(root / name / "information", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    # "Model: \t NVIDIA A100-PCIE-40GB"
                    key, _, value = line.partition(":")
                    if key.strip() == "Model":
                        model = value.strip()
                        break
        except OSError:
            model = ""
        # A device with an unreadable `information` still counts as a present GPU;
        # the count is what the "held elsewhere" message leans on, not the name.
        models.append(model)
    return models


def _vary_mock_for_node(
    cpu: CpuMetrics, mem: MemoryMetrics, gpus: list[GpuMetrics], node_index: int
) -> tuple[CpuMetrics, MemoryMetrics, list[GpuMetrics]]:
    """Give each --demo node its own numbers, so switching nodes visibly does something.

    The per-node mock frames were byte-identical apart from the hostname: every node
    read 50.0% CPU and 25.0% memory. The node switcher is a whole feature — a banner, a
    watchdog, digit-and-Enter selection, arrow keys — and in the only mode where anyone
    can try it without a multi-node allocation, pressing the key changed nothing on
    screen. That is indistinguishable from a switch that silently failed, which is the
    exact confusion the switch banner exists to prevent.

    Deterministic, and node 0 is left EXACTLY as it was: the factor is 1.0 there, so
    every existing expectation about the demo's primary node still holds and only the
    nodes a switch reaches differ. Values only — no field appears or disappears, which
    is the rule SW-81 settled about what a demo may vary.
    """
    # Not merely a fast path, and not redundant either: the factor happens to be
    # exactly 1.0 at index 0 today, so a mutant that deletes this line passes. Kept
    # because it makes "the primary node is untouched" true BY CONSTRUCTION rather
    # than by an arithmetic coincidence that a future change to the factor could break.
    if node_index <= 0:
        return cpu, mem, gpus
    factor = 1.0 - 0.17 * (node_index % 5)  # 0.83, 0.66, 0.49, 0.32, then repeats
    cpu = replace(
        cpu,
        usage_percent=round(cpu.usage_percent * factor, 1),
        effective_cores=round(cpu.effective_cores * factor, 1),
        peak_effective_cores=round(max(cpu.peak_effective_cores * factor, 0.0), 1),
    )
    scaled_current = int(mem.current_bytes * factor)
    scaled_working_set = int(mem.working_set_bytes * factor)
    mem = replace(
        mem,
        current_bytes=scaled_current,
        working_set_bytes=scaled_working_set,
        usage_percent=round(mem.usage_percent * factor, 1),
        working_set_percent=round(mem.working_set_percent * factor, 1),
        # The two memory peaks were the only figures on the card that did not
        # vary, so a reader pressing a digit saw every number move except these
        # -- the exact "changed nothing on screen" confusion this function
        # exists to remove, surviving on two fields. The CPU half above already
        # scales its peak. Measured before the fix on the mock's own frame:
        # node 4 read `used 5.12 GiB` against `peak 16.80 GiB`, a 3.3x ratio
        # where the mock generates 1.05x, and both peaks were byte-identical
        # across all five nodes.
        #
        # Floored at the scaled reading so `peak >= used` still holds -- the
        # invariant `_apply_peaks` states and enforces for `peak_bytes`, and
        # which every producer upholds. Integer truncation is what makes the
        # floor necessary rather than decorative: at factor 0.32 a peak only
        # 1.05x above the reading can truncate to the reading or below it.
        peak_bytes=max(int(mem.peak_bytes * factor), scaled_current),
        peak_working_set_bytes=max(int(mem.peak_working_set_bytes * factor), scaled_working_set),
    )
    gpus = [
        replace(
            g,
            utilization_percent=round(g.utilization_percent * factor, 1),
            process_utilization_percent=round(g.process_utilization_percent * factor, 1),
            memory_used_bytes=int(g.memory_used_bytes * factor),
            memory_utilization_percent=round(g.memory_utilization_percent * factor, 1),
        )
        for g in gpus
    ]
    return cpu, mem, gpus


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
        # The largest total ever EMITTED. The /proc accumulator is deliberately not
        # monotone — SW-24's fix hands a process's ticks back when an in-job ancestor
        # will re-report them, which is what makes the RATE exact — but `usage_ns` is
        # published as a cumulative counter, and a consumer differencing it (SU
        # accounting, an exact-total window) must never see a decrease. Measured on a
        # live 100-sample log: one hand-back dropped the column by 118.7 CPU-seconds
        # in a single row. The rate keeps using the raw accumulator; only the reported
        # total is clamped monotone.
        self._reported_cpu_ns: int = 0
        self._prev_timestamp: float | None = None
        # WHICH counter produced _prev_cpu_ns. _read_cpu_ns can answer from three
        # mutually non-comparable sources (v2 cpu.stat, v1 cpuacct.usage, the /proc
        # accumulator), so a rate is only meaningful between two reads of the SAME one:
        # differencing a 1002 s cgroup counter against a 20 s /proc accumulator once
        # produced effective_cores=982 on a 4-core job, which then latched into
        # peak_effective_cores for the rest of the session.
        self._cpu_source: str | None = None
        self._prev_cpu_source: str | None = None
        # /proc CPU fallback (no cpuacct cgroup): a monotonic accumulator of CPU
        # ticks plus each PID's last-seen ticks, so a child that exits between
        # polls doesn't erase its work from the running total (which would make a
        # busy job read 0% — the counter must only ever climb, like the cgroup).
        # The job's physical CPU ceiling from its cpuset, resolved once (see
        # _cpu_affinity_ceiling). _UNSET distinguishes "not looked up yet" from a
        # looked-up "no constraint" (None).
        self._cpu_ceiling_cached: int | None | object = _UNSET
        self._proc_cpu_seen: dict[int, int] = {}
        # PID -> whether an ancestor INSIDE the job will re-report its CPU once it
        # exits (i.e. it was forked by a job process, not reparented to init). Decided
        # at first sighting, because by exit time the parent is usually gone too.
        self._proc_cpu_from_job: dict[int, bool] = {}
        self._proc_cpu_accum_ticks: int = 0
        self._nvml_initialized = False
        # NVML is functional on this node (init OK + devices present), regardless of
        # whether the job's own GPUs attach — the signal for the "no GPU telemetry
        # here" vs "GPU held by srun" message distinction (F3).
        self._nvml_functional = False
        # WHY GPU telemetry is missing, when it is. "" means nothing to explain
        # (NVML is working, or the job asked for no GPU). The UI needs the reason,
        # not just the boolean: "no_driver" is the user's problem to ignore, while
        # "devices_denied" means the GPUs are right there but Slurm gave this
        # monitor step none of them — a completely different sentence to show.
        self._gpu_unavailable_reason: str = ""
        # The node's physical NVIDIA GPUs (model per device), from procfs rather
        # than NVML so it is still known when the device ACL blinds NVML.
        self._gpu_node_models: list[str] = []
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
        # Last (timestamp, rx_bytes, tx_bytes) summed over the node's ACTIVE RDMA
        # ports, to turn cumulative fabric counters into a live rate. None until the
        # first sample, so the first frame reports "rate not known yet" rather than
        # publishing a fake 0.0 as if the fabric were measured idle.
        self._fabric_prev: tuple[float, int, int] | None = None
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
        # Running max of every memory peak REPORTED, so the figure can never step
        # backwards when the kernel counter stops answering (see _collect_memory).
        self._peak_mem_running: int = 0
        # Running max of the working set (cache-EXCLUDED) — the honest --mem sizing
        # peak, since the cgroup's own peak counter is cache-inclusive (see P2 /
        # _advance_working_set_peak).
        self._peak_working_set: int = 0
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
        # (last_touched, usage, sample_elapsed, measured_at). measured_at is when the
        # underlying sstat query actually succeeded — distinct from the first element,
        # which the transient-failure branch below bumps to pace retries. Keeping them
        # apart is what lets a row say how old its measurement really is.
        self._remote_cache: tuple[float, RemoteUsage, float, float] | None = None
        self._usage_age_seconds = 0.0
        # On-node the cgroup is read every sample, so a snapshot is always a
        # measurement; off-node it depends on whether Slurm has sampled yet.
        self._usage_sampled = True
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
        # Both joins are BOUNDED (see _TEARDOWN_JOIN_SECONDS): an uncancellable
        # executor thread must not be able to hold teardown open indefinitely.
        if self._liveness_task is not None:
            self._liveness_task.cancel()
            await join_bounded(self._liveness_task, _TEARDOWN_JOIN_SECONDS)
        if self._task is not None:
            self._task.cancel()
            await join_bounded(self._task, _TEARDOWN_JOIN_SECONDS)
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

        # The job wants a GPU, so learn what the node physically has before touching
        # NVML: one procfs listing, done once, and it is the only GPU fact that stays
        # readable no matter how NVML fails below.
        self._gpu_node_models = _nvidia_node_gpu_models()

        try:
            import pynvml
        except ImportError:
            logger.info("pynvml not installed; GPU monitoring disabled")
            self._gpu_unavailable_reason = "no_pynvml"
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
                self._gpu_unavailable_reason = "no_driver"
            else:
                logger.warning("NVML init failed: %s", exc)
                self._gpu_unavailable_reason = "nvml_error"
            return False

        # NVML is live from here on. Mark it initialized *now* so that cleanup
        # runs even if the awaiting task is cancelled before start() records the
        # return value (B-C5) or if the enumeration below raises (B-P6). Any
        # early return that decides GPU monitoring is off shuts NVML back down.
        self._nvml_initialized = True

        try:
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                # nvmlInit() got this far through /dev/nvidiactl, which Slurm leaves
                # open to every step, so "0 devices" does NOT mean "no GPU here": on
                # a GPU node it usually means the device cgroup denied /dev/nvidiaN
                # because this step was allocated no GPU (ConstrainDevices=yes and
                # the job's own steps hold them all). procfs still lists the physical
                # cards, so use it to say which of the two happened.
                if self._gpu_node_models:
                    logger.info(
                        "NVML enumerated 0 of the node's %d NVIDIA GPUs; "
                        "this step was allocated none (device cgroup denies them)",
                        len(self._gpu_node_models),
                    )
                    self._gpu_unavailable_reason = "devices_denied"
                else:
                    logger.info("No NVIDIA devices detected by NVML")
                    self._gpu_unavailable_reason = "no_devices"
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
            return self._decode(bid)
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
            # `self._decode`, not a bare `.decode()`. A strict decode of an NVML
            # string raises `UnicodeDecodeError`, which is not an `nv.NVMLError` and
            # so escapes the handler below -- after `_nvml_handles.append(handle)`
            # above and before `_nvml_indices.append(idx)` following, leaving the two
            # lists permanently misaligned. That alignment is the whole point of this
            # method (B-P7): a later index lookup would then read another GPU's
            # cached index. The class already has the tolerant decoder; three other
            # sites in this file open-coded the strict one instead.
            uuid = self._decode(raw_uuid)
            raw_name = nv.nvmlDeviceGetName(handle)
            name = self._decode(raw_name)
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
                this_uuid = self._decode(raw)
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
                # Bounded like the joins above: the lock acquire inside is already
                # capped, but the SUBMISSION can queue behind a saturated default
                # executor, and this runs while the process is trying to exit.
                # Leaving NVML un-shut-down at exit costs nothing; hanging does.
                await join_bounded(
                    self._loop.run_in_executor(None, self._nvml_shutdown_locked),
                    _TEARDOWN_JOIN_SECONDS,
                )
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
            self._prev_cpu_source = self._cpu_source

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
            gpus = self._collect_gpus(job_pids)
            # Is a new srun/mpirun the user just started stuck behind our own
            # held step? Only the monitor step scans, never in mock mode.
            self.launcher_present = (
                self._detect_launchers and not self._mock and _any_launcher_pid(job_pids)
            )
        # Both paths: a peak must never read below the current value it sits beside.
        self._apply_peaks(cpu, mem)
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
        nvml_ok = self._nvml_functional or (self._mock and bool(gpus))
        active_gpus: int | None = sum(1 for g in gpus if _gpu_is_active(g, idle_threshold))
        if not gpus and not nvml_ok and self.job_ctx.gpu_count_requested > 0:
            # Summing an empty list yields 0, and 0 here reads as "your GPUs are idle"
            # — the one number a right-sizing consumer must not be handed when the
            # devices could not be opened at all (the off-node/devices_denied norm).
            active_gpus = None

        # Only a multi-GPU node has an interconnect to report; a CPU-only, single-GPU,
        # or off-node (sstat) sample leaves it None.
        interconnect = None
        if not self._remote and len(gpus) > 1:
            interconnect = self._collect_interconnect(gpus)

        if node_override is not None and self._mock:
            # Demo only: make a switch visible. See _vary_mock_for_node.
            cpu, mem, gpus = _vary_mock_for_node(cpu, mem, gpus, node_index)
        return TelemetrySnapshot(
            timestamp=now,
            job_id=self.job_ctx.job_id,
            job_name=self.job_ctx.job_name,
            # The denominator for elapsed_seconds, which the payload was missing.
            time_limit_seconds=self.job_ctx.time_limit_seconds,
            partition=self.job_ctx.partition,
            array_job_id=self.job_ctx.array_job_id,
            array_task_id=self.job_ctx.array_task_id,
            owner=self.job_ctx.username,
            account=self.job_ctx.account,
            qos=self.job_ctx.qos,
            step_id=self.job_ctx.step_id,
            hostname=stamp_host,
            elapsed_seconds=elapsed,
            cpu=cpu,
            memory=mem,
            gpus=gpus,
            node_count=node_count,
            node_index=node_index,
            gpu_count_requested=self.job_ctx.gpu_count_requested,
            # 0.0 on-node: the cgroup is re-read every sample.
            usage_age_seconds=self._usage_age_seconds if self._remote else 0.0,
            usage_sampled=self._usage_sampled if self._remote else True,
            gpu_active_count=active_gpus,
            remote=self._remote,
            # Says these figures were simulated, at the top level of the payload the
            # machine paths emit (SW-30).
            mock=self._mock,
            # --demo has no NVML, but it DOES put four fully-populated devices in
            # this payload; leaving the flag False made every consumer that gates on
            # it read the demo as GPU-less while the GPUs sat right beside it (SW-5).
            gpu_monitoring_available=nvml_ok,
            gpu_unavailable_reason=self._gpu_unavailable_reason,
            gpu_node_count=len(self._gpu_node_models),
            gpu_node_model=_common_gpu_model(self._gpu_node_models),
            gpu_allocated_indices=list(self.job_ctx.gpu_indices),
            interconnect=interconnect,
            # Node-wide, and only meaningful ON the node — an off-node sstat estimate
            # has no local sysfs to read.
            # time.monotonic(), NOT the wall-clock `now` above: `now` is right for
            # elapsed (it is compared against Slurm's start time) and wrong for a rate
            # window, where an NTP step would divide a real byte delta by a fraction of
            # a second. The CPU and NVLink rates already read monotonic for this reason;
            # this one was measuring throughput against a clock that can jump.
            fabric=None if self._remote else self._collect_fabric(time.monotonic()),
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
            self._usage_age_seconds = max(0.0, now - cached[3])
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
                # measured_at is carried over, NOT bumped: the retry clock moves, the
                # measurement does not, so its age keeps growing as it should.
                self._remote_cache = (now, usage, sample_elapsed, cached[3])
                self._usage_age_seconds = max(0.0, now - cached[3])
            else:
                usage, sample_elapsed = fresh, elapsed_now
                self._remote_cache = (now, fresh, elapsed_now, now)
                self._usage_age_seconds = 0.0

        cores = ctx.cpus_allocated or 1
        # The elapsed captured WITH the sample, not a fresh now-based one: the average
        # is cpu_seconds/elapsed as of the sample, so both only move together when a
        # new sample lands — no per-frame decay between samples / during an outage (N9).
        elapsed = sample_elapsed

        usage_pct = 0.0
        effective = 0.0
        if usage.cpu_seconds > 0 and elapsed > 0:
            # cpu_seconds is already a per-node estimate. effective_cores is left
            # UNCAPPED so an over-subscribed job on a ConstrainCores=no node still
            # shows it used more than allocated; only the bar percent is clamped to
            # [0,100] (A3).
            #
            # This no longer "matches the on-node path", as this comment used to claim:
            # SW-25 gave that path a cap at the job's cpuset width, because a cpuset is
            # a HARD kernel limit and exceeding it can only be a sampling artifact.
            # There is deliberately no equivalent here, and the reason is evidence: off
            # the node no cpuset is readable, so an over-report cannot be distinguished
            # from real over-subscription — and capping at `cores` on a guess would
            # erase the "raise --cpus-per-task" signal that this figure exists to give.
            # The cost is that a record can still pair a clamped 100.0% with, say,
            # 8.2/8 (the shape SW-25 removed on-node); reconciling that by capping
            # without evidence would trade a visible inconsistency for a silent lie.
            effective = usage.cpu_seconds / elapsed
            usage_pct = max(0.0, min(100.0, effective / cores * 100.0))
        cpu = CpuMetrics(
            cores_allocated=cores,
            usage_ns=int(usage.cpu_seconds * 1_000_000_000),
            usage_percent=round(usage_pct, 1),
            effective_cores=round(effective, 1),
            source="sstat",
        )

        self._usage_sampled = usage.sampled
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
            # Evaluated, not hardcoded False. `sw <jobid>` from a login node is the
            # primary documented workflow, and the guard exists to warn BEFORE the
            # kill — so a job creeping to 89% of its --mem used to report
            # `oom_guard_*: false` all the way into the OOM, even with the
            # thresholds lowered to 30%. Reaching the working guard meant knowing to
            # `srun --overlap` onto the node first, which is the knowledge this tool
            # exists to spare people. SW-15.
            #
            # #34's concern was that MaxRSS only climbs, so a guard on it latches an
            # alarm that can't clear. It stands, and this is why the reading is
            # LABELLED a peak everywhere it is shown (the row's bar says "peak", the
            # snapshot says source="sstat"): "this job came within 10% of its limit"
            # stays true after the fact, and raising --mem stays the right advice.
            # It is tempting to call this a LOWER bound on the cgroup's fraction —
            # MaxRSS excludes cache and kernel memory — and this comment used to,
            # concluding the guard could fire late but never falsely early. Measured
            # off-node against a live 4-GPU job on an A100 node, that is wrong:
            # JobAcctGather builds MaxRSS by SUMMING each process's RSS, so a page
            # shared between processes is counted once per process. MaxRSS read
            # 61.34 GB against the same job's cgroup lifetime peak of 54.27 GB —
            # which INCLUDES cache — and an anonymous working set of 32.65 GB: 13%
            # above the former, 1.88x the latter. So the guard fired at 87.9% of
            # --mem while the cgroup's own peak fraction was 79.3%. It errs in BOTH
            # directions, and every surface that shows this number now says so
            # rather than sending the reader off to raise --mem. SW-15 keeps the
            # guard; what was wrong was the one-sidedness claimed for it.
            oom_guard_warning=mem_pct >= self.config.oom_warning_threshold * 100,
            oom_guard_critical=mem_pct >= self.config.oom_critical_threshold * 100,
            working_set_bytes=rss,
            cache_bytes=0,
            # Off-node has only sstat MaxRSS (no cache breakdown), so the RSS peak
            # is the best working-set peak we can offer.
            peak_working_set_bytes=rss,
            working_set_percent=round(mem_pct, 1),
            # MaxRSS is a job-lifetime high-water (that is the whole reason it never
            # falls), so this one IS a lifetime figure — a cache-EXCLUDED one.
            peak_is_lifetime=True,
            # Say which reading this is: every field above is derived from ONE
            # MaxRSS high-water, so `current_bytes == peak_bytes` always, neither
            # falls when the job's memory drops, and the 0 cache is "sstat doesn't
            # report cache", not "this job has none" (SW-3).
            source="sstat",
            cache_measured=False,
        )
        return cpu, mem

    def _apply_peaks(self, cpu: CpuMetrics, mem: MemoryMetrics) -> None:
        """Fold the CPU high-water mark into a freshly-collected snapshot and keep the
        memory peak self-consistent.

        Memory peak is the number a user sizes ``--mem`` against, so it must be the
        job's TRUE lifetime maximum — not just what we happened to see since
        attaching. ``_collect_memory`` already reports that from the cgroup's own
        lifetime counter (v1 ``memory.max_usage_in_bytes`` / v2 ``memory.peak``),
        which survives sw restarts and covers the whole job even on a late attach;
        where the kernel exposes neither it falls back to a running max of usage and
        says so via ``peak_is_lifetime=False``. That is NOT rare, as this comment
        used to imply: cgroup v2 only gained ``memory.peak`` in kernel 5.19, so every
        v2 cluster on an older kernel (RHEL/Rocky 9 ships 5.14) takes the fallback.
        We do NOT recompute it here — folding in the smaller live working set would
        drag a late-attached job's peak DOWN below its real high-water mark — we
        only ensure it never reads below the current usage, so ``max >= used`` holds.

        CPU peak = the most cores ever busy at once. No kernel counter exists for
        it, so it is a monotonic running max over the current sw session.

        Runs for the OFF-NODE (sstat) path too. There ``effective_cores`` is an
        average since job start rather than an instantaneous rate, so its running max
        is a weaker figure — but leaving the field at 0.0 beside a non-zero
        ``effective_cores`` broke the ``peak >= current`` invariant every other peak
        upholds, and a ``--json`` consumer reading ``peak_effective_cores`` off-node
        got "this job used no CPU". The memory path already does exactly this (it
        reports MaxRSS as both current and peak off-node). The TUI still suppresses
        the redundant "· peak" suffix off-node, matching how it treats the MEM row."""
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
                source="mock",
            )
        usage_ns = self._read_cpu_ns(job_pids)
        usage_pct = 0.0
        effective = 0.0

        # Use a MONOTONIC clock for the rate window: a wall-clock (time.time())
        # step backward from an NTP correction would give dt<=0 and drop the
        # sample to 0% (or a huge spike on a forward jump).
        mono = time.monotonic()
        source = self._cpu_source
        if (
            usage_ns is not None
            and self._prev_cpu_ns is not None
            and source == self._prev_cpu_source
        ):
            dt = mono - (self._prev_timestamp or mono)
            # Require a MINIMUM dt (a fraction of the poll interval, capped at 0.1s)
            # before trusting the rate. effective_cores is now uncapped (A3), so an
            # anomalously short window — two rapid collects, a sub-interval frame,
            # /proc-read jitter — would turn a normal CPU delta into a huge rate that
            # then latches PERMANENTLY into the monotonic peak_effective_cores. Below
            # the floor, leave the baseline untouched so the delta accumulates into
            # the next adequately-spaced frame instead of being measured over a bogus
            # window; this frame just reads 0 and self-corrects next tick.
            min_dt = min(0.1, 0.5 * self.config.poll_interval)
            if dt >= min_dt:
                # Clamp the delta: the /proc fallback can shrink when a
                # process exits between samples.
                delta_ns = max(0, usage_ns - self._prev_cpu_ns)
                # effective_cores = the CPU-time rate (cores actually busy), from the
                # RAW delta. Left UNCAPPED against `cores`: on a ConstrainCores=no node
                # a job can run on MORE cores than allocated, and capping at `cores`
                # would erase that over-subscription (the "raise --cpus-per-task"
                # signal). Only the bar percent is clamped against it (A3).
                #
                # It IS capped at the number of CPUs the job may physically run on,
                # when the kernel says there is a hard limit. `8.2 of 8` on a job
                # confined to `cpuset.cpus=14-21` is not over-subscription, it is a
                # sampling artifact — and it contradicted `usage_percent`, which is
                # clamped, so one --json record said 100.0% and 8.2/8 at once (SW-25).
                effective = delta_ns / (dt * 1_000_000_000)
                ceiling = self._cpu_affinity_ceiling(job_pids)
                if ceiling is not None:
                    effective = min(effective, float(ceiling))
                max_possible_ns = dt * cores * 1_000_000_000
                raw_pct = (delta_ns / max_possible_ns) * 100.0 if max_possible_ns > 0 else 0.0
                usage_pct = max(0.0, min(100.0, raw_pct))
                self._prev_cpu_ns = usage_ns
                self._prev_timestamp = mono
                self._prev_cpu_source = source
        elif usage_ns is not None:
            # First sample, or the counter SOURCE changed (a cgroup read failed and the
            # /proc accumulator answered instead, or vice versa). The two are not
            # comparable, so re-seed against the new one and emit 0 for this frame
            # rather than differencing them.
            self._prev_cpu_ns = usage_ns
            self._prev_timestamp = mono
            self._prev_cpu_source = source
        # else: NO source was readable this frame. Deliberately leave the baseline
        # (and its source) untouched — every counter it can come from is monotonic, so
        # the next successful read measures a correct multi-interval rate. Overwriting
        # _prev_cpu_ns with None here cost TWO consecutive frames of 0% on a fully busy
        # job: this one, and then the next, which found no baseline to difference against.

        if usage_ns is not None:
            self._reported_cpu_ns = max(self._reported_cpu_ns, usage_ns)
        return CpuMetrics(
            cores_allocated=cores,
            # usage_ns is a monotonic cumulative counter, enforced HERE rather than
            # assumed: a failed read reports the last known value rather than a
            # literal 0, and a raw accumulator that stepped backwards (the SW-24
            # hand-back) reports its previous high instead. Either way a consumer
            # differencing the column never sees a fake reset. The overstatement
            # after a hand-back is bounded by the ticks handed back and closes as
            # soon as the ancestor re-reports them.
            usage_ns=self._reported_cpu_ns,
            usage_percent=round(usage_pct, 1),
            effective_cores=round(effective, 1),
            # Whichever of the three counters answered this frame ("v2"/"v1"/"proc"),
            # so a consumer can tell a cgroup figure from a /proc PID-sum.
            source=self._cpu_source or "",
        )

    def _cpu_affinity_ceiling(self, job_pids: set[int] | None) -> int | None:
        """How many CPUs this job may physically use, or ``None`` if unconstrained.

        Read from a job process's CPU affinity, which is exactly what the cpuset
        controller enforces — no path discovery, and identical on cgroup v1 and v2.
        ``None`` when the affinity covers the whole node (no cpuset confinement), so
        an over-subscribed job on a ConstrainCores=no node still reports more cores
        than allocated, which is the signal the uncapped figure exists for.

        Cached: a job's cpuset does not change while it runs, and this must not add a
        syscall per PID per frame.
        """
        cached = self._cpu_ceiling_cached
        if cached is not _UNSET:
            return cached if isinstance(cached, int) else None
        ceiling: int | None = None
        node_cpus = os.cpu_count() or 0
        for pid in sorted(job_pids or ()):
            try:
                allowed = len(os.sched_getaffinity(pid))
            except (OSError, ProcessLookupError, AttributeError):
                continue
            if allowed > 0 and (node_cpus == 0 or allowed < node_cpus):
                ceiling = allowed
            break
        self._cpu_ceiling_cached = ceiling
        return ceiling

    def _read_cpu_ns(self, job_pids: set[int] | None = None) -> int | None:
        """Cumulative CPU time (ns) for the job.

        Prefers the cgroup accounting controllers (which also capture children
        that have already exited), then falls back to summing /proc/<pid>/stat
        for the job's live PIDs — needed on clusters that constrain jobs with
        the cpuset controller but create no per-job cpuacct/cpu cgroup.

        Records WHICH of the three counters answered in ``self._cpu_source``, because
        their values are not comparable to each other — see ``_collect_cpu``.
        """
        ctx = self.job_ctx
        if ctx.cgroup_v2_path:
            val = _read_cgroup_field(Path(ctx.cgroup_v2_path) / "cpu.stat", "usage_usec")
            if val is not None:
                self._cpu_source = "v2"
                return val * 1000
        if ctx.cgroup_v1_cpu_path:
            val = _read_int_file(Path(ctx.cgroup_v1_cpu_path) / "cpuacct.usage")
            if val is not None:
                self._cpu_source = "v1"
                return val
        if job_pids:
            self._cpu_source = "proc"
            return self._accumulated_proc_cpu_ns(job_pids)
        self._cpu_source = None
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

        Each PID contributes its own CPU **and** the CPU of children it has already
        reaped (``cutime``/``cstime``), because on this kind of cluster that is where
        most of the work ends up: an R worker `system()`-ing a 1 s shell 1,400 times
        books ~99% of the job's CPU in reaped children (SW-24).

        Reading both introduces a double-count the naive version misses, and this is
        the part that has to be exact: a child's CPU is credited directly while it
        lives, and then AGAIN when its parent reaps it and the parent's ``cutime``
        jumps. So when a PID leaves, the ticks credited for it are removed from the
        accumulator **if its parent is inside this job** — the parent's ``cutime``
        now carries them (POSIX: a reaped child's whole subtree total lands there).
        Where the parent is NOT in the job — an orphaned PSOCK worker reparented to
        PID 1, which is the other half of this same workload — nothing else will ever
        report that CPU, so it is kept. Net effect at the transition is zero either
        way, and no interval can lose or duplicate a generation.

        The parent test is recorded at FIRST SIGHTING, not at exit, and that detail
        is what makes it work on the real workload: `timeout 1 bash -c …` means the
        child's parent usually exits in the same interval the child does, so asking
        "is the parent alive now?" answered no and the subtraction never fired —
        measured +23.6% over ground truth before this was corrected. Each generation
        hands its ticks up the chain to whichever ancestor is still in the job.
        """
        for pid in pids:
            sample = _read_pid_cpu(pid)
            if sample is None:
                continue
            cur = sample.own + sample.children
            # Whether this process's CPU will be re-reported by an ancestor once it
            # exits. Decided at FIRST sighting, while the parent is still observable
            # — by exit time the parent is usually gone too.
            if pid not in self._proc_cpu_from_job:
                self._proc_cpu_from_job[pid] = sample.ppid > 1 and (
                    sample.ppid in pids or sample.ppid in self._proc_cpu_seen
                )
            if cur <= 0:
                continue
            prev = self._proc_cpu_seen.get(pid, 0)
            # cur >= prev: normal forward progress. cur < prev: the PID number was
            # reused by a new process — count its ticks as fresh (from 0).
            self._proc_cpu_accum_ticks += cur - prev if cur >= prev else cur
            self._proc_cpu_seen[pid] = cur
        # Forget a PID only once it is TRULY gone from /proc — not merely absent from
        # this poll's sample. A still-live PID can briefly fall out of the sampled set
        # (an enumeration race, a one-off /proc/<pid>/stat read miss); dropping its
        # last tick count then would make it look brand-new on return and re-add its
        # whole history — a spurious ~2x CPU spike (a double-count). Keying eviction on
        # the PID's /proc entry rather than a timeout can never evict a live PID, so
        # that double-count is impossible; a dead PID's ticks already live in the
        # accumulator (eviction never changes the total), and if its number is later
        # reused the ``cur < prev`` reset above counts the new process from zero. This
        # still bounds memory: an exited PID is dropped on the very next poll.
        for pid in list(self._proc_cpu_seen):
            if pid in pids or _pid_alive(pid):
                continue
            credited = self._proc_cpu_seen.pop(pid)
            # Hand the ticks back to whoever will now report them. An ancestor inside
            # the job absorbs this process's whole subtree into its own cutime/cstime,
            # so keeping our copy would count it twice; an orphan's parent is outside
            # the job (PID 1), so our copy is the only record that will ever exist.
            if self._proc_cpu_from_job.pop(pid, False):
                self._proc_cpu_accum_ticks = max(0, self._proc_cpu_accum_ticks - credited)
        for pid in list(self._proc_cpu_from_job):
            if pid not in self._proc_cpu_seen and pid not in pids and not _pid_alive(pid):
                del self._proc_cpu_from_job[pid]
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

    def _advance_working_set_peak(self, working_set_bytes: int) -> int:
        """Monotonic running max of the working set (cache-EXCLUDED) — the --mem number.

        The cgroup's own peak counter (``peak_bytes``) is CACHE-INCLUSIVE, so for a
        job that streams/mmaps a big dataset it reads far above the anonymous
        high-water mark and pushes the user to over-request. The kernel exposes no
        cache-excluded lifetime peak, so this is a running max over the current sw
        session (like the CPU peak); ``peak_bytes`` stays as the cache-inclusive
        lifetime total for reference.
        """
        if working_set_bytes > self._peak_working_set:
            self._peak_working_set = working_set_bytes
        return self._peak_working_set

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
                # From the config, not a hardcoded 85/90: three sites computed this
                # guard and only one honoured SLURMWATCH_OOM_WARN/_CRIT, so a user
                # who lowered the thresholds had them silently ignored here (SW-15,
                # secondary). The demo's own curve tops out below the default warn
                # threshold on purpose, so this stays quiet by default.
                oom_guard_warning=pct >= self.config.oom_warning_threshold * 100,
                oom_guard_critical=pct >= self.config.oom_critical_threshold * 100,
                working_set_bytes=current,
                cache_bytes=0,
                peak_working_set_bytes=peak,
                working_set_percent=round(pct, 1),
                source="mock",
                cache_measured=False,
                peak_is_lifetime=True,
            )
        current_bytes = 0
        peak_bytes = 0
        working_set_bytes = 0
        cache_bytes = 0
        # Only a kernel counter earns the "lifetime" claim; the /proc fallback below
        # (no cgroup delegated at all) never does.
        peak_is_lifetime = False
        # Same rule for the cache figure (SW-3): `cache_bytes: 0` is a MEASUREMENT
        # only where memory.stat answered. It does not on two paths that reach here
        # — a cgroup with no memory controller delegated (the /proc-RSS fallback
        # below), and a cgroup that vanished mid-sample because the job ended — and
        # both used to publish the untouched 0 as measured, which the TUI renders as
        # "0.0 B" of reclaimable cache and CSV/JSON as mem_cache_measured=1. Off-node
        # sstat already says False for exactly this reason.
        cache_measured = False
        # And the same rule for the FIGURE's provenance. `MemoryMetrics.source` says
        # "WHERE these numbers came from", and CpuMetrics.source already distinguishes
        # the /proc PID-sum from the cgroup counter because the two "are not comparable
        # to each other" — yet the memory analogue of that very fallback
        # (_proc_rss_bytes, F4) published its statm sum as `source: "cgroup"`. It is not
        # a cgroup reading and it does not mean the same thing: statm's resident field
        # counts shared pages, so it over-reports, and it sees only PIDs alive at this
        # instant. Same vocabulary as the CPU side ("proc"), so a --json/CSV consumer
        # sizing --mem can tell which counter answered.
        mem_source = "cgroup"

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
                mem_source = "proc"
            else:
                current_bytes = current_raw

            peak_bytes = _read_int_file(v2 / "memory.peak") or 0
            peak_is_lifetime = peak_bytes > 0
            if peak_bytes == 0:
                # cgroup v2 gained memory.peak in kernel 5.19; RHEL/Rocky 9 ships
                # 5.14, so on a large share of clusters there is no counter to read
                # and this becomes a running max since monitoring began. Same field,
                # weaker claim — say which, so no surface calls it a lifetime peak.
                peak_bytes = self._peak_mem_running
                if current_bytes > self._peak_mem_running:
                    self._peak_mem_running = current_bytes
                    peak_bytes = current_bytes

            raw_max = _read_cgroup_raw(v2 / "memory.max")
            enforced_v2 = False
            if raw_max is not None and raw_max.strip() != "max":
                with contextlib.suppress(ValueError):
                    limit_bytes = int(raw_max.strip())
                    enforced_v2 = True
            if not enforced_v2:
                # No enforced cgroup cap ("max", absent, or unparseable) — so the kernel
                # OOM-kills at NODE RAM, not at the Slurm request. Say so explicitly:
                # otherwise `limit_bytes` keeps ctx.mem_limit_bytes, `cgroup_limit` below
                # silently becomes the ALLOCATION, and the OOM guard measures the job
                # against its own request — re-introducing the false "near limit, raise
                # --mem" critical that P3 removed. v1 gets this right only by accident:
                # its unlimited value is a huge sentinel that the `> 10**16` branch below
                # converts to MemTotal, while v2 spells unlimited as the string "max",
                # which that numeric check can never catch. Same physical situation, so
                # both branches must reach the same guard basis.
                limit_bytes = _read_meminfo_total()

            stat = _read_cgroup_raw(v2 / "memory.stat")
            if stat:
                working_set_bytes, cache_bytes = _working_set_from_stat(stat, current_bytes, "")
                cache_measured = True

            if limit_bytes == 0 or limit_bytes > 10**16:
                limit_bytes = _read_meminfo_total()

        elif ctx.cgroup_v1_mem_path:
            v1 = Path(ctx.cgroup_v1_mem_path)
            # Same gap as v2 above (F4): the discovered v1 memory cgroup can have no
            # memory controller delegated, so memory.usage_in_bytes is absent. Fall
            # back to /proc RSS rather than let MEM stick at 0. (working_set_bytes
            # is re-derived from current_bytes below regardless, so it doesn't need
            # setting here too.)
            current_raw = _read_int_file(v1 / "memory.usage_in_bytes")
            if current_raw is None:
                current_bytes = self._proc_rss_bytes()
                mem_source = "proc"
            else:
                current_bytes = current_raw
            peak_bytes = _read_int_file(v1 / "memory.max_usage_in_bytes") or 0
            peak_is_lifetime = peak_bytes > 0
            if peak_bytes == 0:
                # Absent when the memcg was built without the usage-history counters
                # (or the file is unreadable): same running-max fallback as v2.
                peak_bytes = self._peak_mem_running
                if current_bytes > self._peak_mem_running:
                    self._peak_mem_running = current_bytes
                    peak_bytes = current_bytes
            raw_limit = _read_int_file(v1 / "memory.limit_in_bytes")
            # "No enforced cap was READ" is not "the cap is the allocation". The old
            # `... or limit_bytes` substituted ctx.mem_limit_bytes when the file was
            # absent or unreadable, so `cgroup_limit` below silently became the
            # ALLOCATION and the OOM guard measured the job against its own REQUEST —
            # the exact false "near limit, raise --mem" critical that P3 removed and
            # that the v2 branch above spells out it must avoid ("Same physical
            # situation, so both branches must reach the same guard basis"). Measured:
            # a 7.5 GiB working set in an 8 GiB allocation reported
            # oom_guard_critical=True here while the identical v2 shape (memory.max
            # absent) correctly reported False. Unreadable joins the two values that
            # already meant unlimited (0 and the v1 sentinel), so all three land on
            # node RAM — where the kernel actually OOM-kills when nothing caps the
            # cgroup. The DISPLAYED limit is unchanged: min(alloc, cgroup_limit) below
            # still reports the allocation.
            if raw_limit is None or raw_limit == 0 or raw_limit > 10**16:
                limit_bytes = _read_meminfo_total()
            else:
                limit_bytes = raw_limit
            # memory.usage_in_bytes counts reclaimable page cache; subtract the
            # file-backed cache to get the working set that drives OOM pressure.
            # v1 memory.stat uses hierarchical total_* keys.
            working_set_bytes = current_bytes
            stat = _read_cgroup_raw(v1 / "memory.stat")
            if stat:
                working_set_bytes, cache_bytes = _working_set_from_stat(
                    stat, current_bytes, "total_"
                )
                cache_measured = True

        # A peak may never read BELOW a peak already reported. Both branches above
        # fall back to a running max when the kernel counter (v1
        # memory.max_usage_in_bytes / v2 memory.peak) is unreadable, but that running
        # max is fed only by `current`, so the first frame whose counter read fails
        # rebuilt the peak from scratch and threw the kernel's high-water away. The
        # trigger is not exotic: when the job ends mid-sample the whole cgroup goes at
        # once, `current` collapses to the /proc fallback's 0, and a 16 GiB lifetime
        # peak was published as 0 — lower than `peak_working_set_bytes`, whose own
        # running max had correctly kept it, and impossible for a cache-INCLUSIVE
        # figure. --once and --log both write that frame. Feed every peak reported
        # into the running max and floor the reading with it; `peak_is_lifetime` is
        # left to say whether a kernel counter answered THIS frame.
        if peak_bytes > self._peak_mem_running:
            self._peak_mem_running = peak_bytes
        peak_bytes = self._peak_mem_running

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
        # The OOM guards measure against the cgroup's REAL enforced limit — where the
        # kernel actually OOM-kills — NOT the reported `limit_bytes`. `limit_bytes` is
        # min(alloc, cgroup) so the dashboard reads "used / requested"; on a
        # ConstrainRAMSpace=no node the cgroup limit is the whole node's RAM, so
        # limit_bytes collapses to the allocation and a job that merely exceeds its
        # *request* (but is nowhere near the node-RAM ceiling, so cannot be
        # OOM-killed) would otherwise trip a false "near limit, raise --mem" critical.
        # Guarding against cgroup_limit keeps the OOM alarm honest; the display % above
        # still uses the allocation. cgroup_limit is 0 only when no limit was readable,
        # then fall back to limit_bytes.
        guard_limit = cgroup_limit if cgroup_limit > 0 else limit_bytes
        guard_pct = (ws_for_guard / guard_limit) * 100.0 if guard_limit > 0 else 0.0
        # Working set as a percent of the ALLOCATION (what the TUI MEM gauge shows),
        # clamped — emitted so a --json/CSV consumer sizing --mem gets the
        # cache-EXCLUDED figure, not only the cache-inclusive usage_percent (Note 1).
        ws_pct = min(100.0, ws_for_guard / limit_bytes * 100.0) if limit_bytes > 0 else 0.0

        return MemoryMetrics(
            current_bytes=current_bytes,
            limit_bytes=limit_bytes,
            peak_bytes=peak_bytes,
            usage_percent=round(usage_pct, 1),
            oom_guard_warning=guard_pct >= self.config.oom_warning_threshold * 100,
            oom_guard_critical=guard_pct >= self.config.oom_critical_threshold * 100,
            working_set_bytes=working_set_bytes or current_bytes,
            cache_bytes=cache_bytes,
            source=mem_source,
            cache_measured=cache_measured,
            peak_working_set_bytes=self._advance_working_set_peak(ws_for_guard),
            working_set_percent=round(ws_pct, 1),
            peak_is_lifetime=peak_is_lifetime,
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
                        # The real A100-SXM4-80GB enforced cap, so --demo shows the
                        # "used / cap W" headroom figure a real device does (and the
                        # README GIF advertises) instead of a bare "240 W".
                        power_limit_watts=400.0,
                        temperature_celsius=round(
                            55 + 20 * (0.5 + 0.5 * math.sin(elapsed * 0.15 + i)), 1
                        ),
                        throttling=False,
                        process_utilization_percent=round(
                            30 + 50 * (0.5 + 0.5 * math.sin(elapsed * 0.3 + i * 1.5)), 1
                        ),
                        process_memory_bytes=int(used * 0.9),
                        cuda_ordinal=i,
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
                    util_supported = True
                    try:
                        util_pct = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    except pynvml.NVMLError as exc:
                        # A failed util read is either genuinely unsupported (a MIG
                        # slice → NOT_SUPPORTED, persistent) or a transient hiccup on
                        # a util-capable device. Only the former lets the active
                        # heuristic fall back to device-wide VRAM; a transient failure
                        # on a shared GPU must keep the majority-owner guard so another
                        # tenant's VRAM isn't scored as this job's activity (A7).
                        util_available = False
                        not_supported = getattr(pynvml, "NVML_ERROR_NOT_SUPPORTED", 3)
                        if getattr(exc, "value", None) == not_supported:
                            util_supported = False

                    mem_used = 0
                    mem_total = 0
                    mem_available = True
                    try:
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        mem_used = mem_info.used
                        mem_total = mem_info.total
                    except pynvml.NVMLError:
                        # 0 is a plausible VRAM reading, so record that this one is not a
                        # reading at all. Otherwise the activity heuristic's
                        # `memory_used_bytes > 0` veto scores a busy GPU as idle.
                        mem_available = False

                    mem_util_pct = 0.0
                    if mem_total > 0:
                        mem_util_pct = (mem_used / mem_total) * 100.0

                    power_w = 0.0
                    power_available = True
                    try:
                        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                        power_w = power_mw / 1000.0
                    except pynvml.NVMLError:
                        power_available = False

                    # The enforced power cap, so the UI/JSON can show headroom-to-cap
                    # (a GPU pegged at its cap is well-utilised, not sick) — it's also
                    # the context for a benign SwPowerCap "throttle". 0 if unreadable.
                    power_limit_w = 0.0
                    with contextlib.suppress(pynvml.NVMLError, AttributeError):
                        power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0

                    temp_c = 0.0
                    temp_available = True
                    try:
                        temp_c = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    except pynvml.NVMLError:
                        # 0 degrees C would otherwise render as a real reading.
                        temp_available = False

                    throttling, throttle_reasons = self._check_gpu_throttling(handle)

                    process_util = 0.0
                    process_mem = 0
                    # No PIDs to attribute anything to is not "the job used 0%" —
                    # there was nothing to ask about.
                    process_util_available = bool(job_pids)
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
                            # NOT_SUPPORTED (MIG, old driver), NO_PERMISSION, or the
                            # symbol is absent: the 0.0 below is not a reading.
                            process_util_available = False

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
                            process_utilization_available=process_util_available,
                            utilization_available=util_available,
                            utilization_supported=util_supported,
                            memory_available=mem_available,
                            power_available=power_available,
                            temperature_available=temp_available,
                            power_limit_watts=round(power_limit_w, 1),
                            throttle_reasons=throttle_reasons,
                            # Position in _nvml_handles IS the CUDA ordinal: every
                            # _init_nvml path attaches the job's devices in the order
                            # the job's own CUDA_VISIBLE_DEVICES exposes them (or, when
                            # NVML already shows only the job's GPUs, in PCI-bus order,
                            # which is that same order). Taken from `pos`, not from the
                            # length of `metrics`, so a device dropped by the guard
                            # below can't shift the ordinals of the ones after it.
                            cuda_ordinal=pos,
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

    def _check_gpu_throttling(self, handle: object) -> tuple[bool, list[str]]:
        """``(throttling, reasons)`` from NVML's current clock-throttle bitmask.

        ``throttling`` is True when any *meaningful* reason bit is set; ``reasons``
        names them so a ``--json`` consumer can tell a benign power cap
        (``sw_power_cap`` — the ideal steady state of a power-limited GPU) apart from
        a thermal or hardware slowdown. Benign bits (GpuIdle, ApplicationsClocksSetting,
        SyncBoost, DisplayClockSetting) are excluded. The TUI surfaces none of this.
        """
        reasons: list[str] = []
        try:
            import pynvml

            def _const(*names: str) -> int:
                for name in names:
                    value = getattr(pynvml, name, None)
                    if isinstance(value, int):
                        return value
                return 0

            # (label, constant spellings) — names differ across pynvml releases
            # (ThrottleReason* vs the newer EventReason*). Same 5 bits as before.
            reason_bits = (
                (
                    "sw_power_cap",
                    ("nvmlClocksThrottleReasonSwPowerCap", "nvmlClocksEventReasonSwPowerCap"),
                ),
                (
                    "hw_thermal",
                    (
                        "nvmlClocksThrottleReasonHwThermalSlowdown",
                        "nvmlClocksEventReasonHwThermalSlowdown",
                    ),
                ),
                (
                    "sw_thermal",
                    (
                        "nvmlClocksThrottleReasonSwThermalSlowdown",
                        "nvmlClocksEventReasonSwThermalSlowdown",
                    ),
                ),
                (
                    "hw_power_brake",
                    (
                        "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",
                        "nvmlClocksEventReasonHwPowerBrakeSlowdown",
                    ),
                ),
                (
                    "hw_slowdown",
                    ("nvmlClocksThrottleReasonHwSlowdown", "nvmlClocksEventReasonHwSlowdown"),
                ),
            )
            try:
                bits = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                if bits:
                    for label, names in reason_bits:
                        if bits & _const(*names):
                            reasons.append(label)
            except (pynvml.NVMLError, AttributeError):
                pass
        except Exception:
            pass
        return bool(reasons), reasons

    # -- GPU interconnect (NVLink / PCIe topology) ---------------------------

    def _collect_fabric(self, mono: float) -> NodeFabric | None:
        """The node's inter-node fabric and its live throughput.

        ``mono`` must come from a MONOTONIC clock: this is a rate window, and a
        wall-clock step (NTP correction, leap second, VM migration) would divide a
        real byte delta by a fraction of a second and publish a throughput far above
        the link's ceiling. The rate has no high-side clamp, so there is nothing else
        to catch it.

        Rates come from a delta of the cumulative port counters, so the first frame
        reports ``rates_known=False``. A counter that goes BACKWARDS (HCA reset, or
        a port bounced) clamps to 0 rather than producing a wild negative or a
        nonsense spike. A window shorter than ``min_dt`` is not measured at all and
        the baseline is KEPT, so the delta lands in the next properly spaced frame
        instead of being divided by a sliver — the same floor the CPU rate uses.
        """
        if self._mock:
            return self._mock_fabric()
        ports = _ib_ports()
        if not ports:
            self._fabric_prev = None
            return None
        rx_total = sum(p.rx_bytes for p in ports)
        tx_total = sum(p.tx_bytes for p in ports)
        # Report the per-port link rate, not the sum: "100 Gb/s" is what an operator
        # recognises, and summing ports would imply a single stream can use it all.
        rates = [p.rate_gbps for p in ports if p.rate_gbps > 0]
        fabric = NodeFabric(
            ports=len(ports),
            link_rate_gbps=max(rates) if rates else 0.0,
            # Summed, because rx/tx below are summed across the same ports.
            link_rate_total_gbps=sum(rates),
            kind=ports[0].kind,
            rate_label=ports[0].rate_label,
        )
        prev = self._fabric_prev
        dt = mono - prev[0] if prev is not None else 0.0
        min_dt = min(0.1, 0.5 * self.config.poll_interval)
        usable = prev is not None and dt >= min_dt
        if prev is None or usable:
            # Only move the baseline when this frame either seeds it or consumes it;
            # replacing it on a too-short window would discard the bytes measured
            # since the last usable frame.
            self._fabric_prev = (mono, rx_total, tx_total)
        if usable and prev is not None:
            # bytes/s -> Gbit/s (decimal), matching how the fabric is specced and
            # how link_rate_gbps reads, so a rate can be compared to the ceiling.
            fabric.rx_gbps = round(max(rx_total - prev[1], 0) * 8.0 / dt / 1e9, 3)
            fabric.tx_gbps = round(max(tx_total - prev[2], 0) * 8.0 / dt / 1e9, 3)
            fabric.rates_known = True
        return fabric

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
            # Fall back to `pos` when the cached index is ABSENT *or* unknown (-1), the
            # same rule _collect_gpus applies. Taking a cached -1 at face value put -1
            # into `devices`, and _handle_for_device(-1) then matched the first such
            # device for every one of them: one GPU's NVLink counters were read twice
            # (once under its sibling's name) and the other's never at all, while the
            # grid header printed "CUDA -1" and the model lookup missed.
            cached = self._nvml_indices[pos] if pos < len(self._nvml_indices) else -1
            idx = cached if cached >= 0 else pos
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
        link_state_readable = False
        for i, handle in enumerate(handles):
            for link in range(nv.NVML_NVLINK_MAX_LINKS):
                try:
                    state = nv.nvmlDeviceGetNvLinkState(handle, link)
                except (nv.NVMLError, AttributeError):
                    # NOT_SUPPORTED on a PCIe-only card, or the whole API missing.
                    continue
                # The state itself was readable, whatever it says. Distinguishing that
                # from "the API told us nothing" is what lets a genuinely DOWN link be
                # reported as down instead of being papered over by the present-count.
                link_state_readable = True
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
        # Prefer the MEASURED count of enabled links. NVML_FI_DEV_NVLINK_LINK_COUNT is
        # documented as "Number of NVLinks PRESENT on the device", so folding it in with
        # max() made a degraded fabric advertise its spec bandwidth: an H200 with one
        # link down reported "18 links · 900 GB/s per GPU" while the topology grid on the
        # same screen said NV17. Fall back to the present-count only when no per-link
        # state was readable at all, where it is the sole evidence available.
        links_per_gpu = max(active_links, default=0) if link_state_readable else fv_link_count
        # Generation and the per-link-speed fallback come from the device MODEL, not
        # from `version` (NVML's driver-internal link-version code, which reads 7 on a
        # live H200 where the enum's 7 means NVLink 5) — see _NVLINK_MODEL_SPEC. NVML's
        # own measured per-link speed still wins when the driver exposes it; it's
        # NOT_SUPPORTED on driver 535, which is why the fallback has to be right.
        generation, spec_gbps = _nvlink_model_spec(
            self._nvml_handle_info.get(devices[0], ("", ""))[1]
        )
        logger.debug(
            "NVML nvlink version code %s; model-derived generation %s", version, generation
        )
        link_speed_gbps = speed_mbps / 1000.0 if speed_mbps else spec_gbps
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
            nvlink_version=generation,
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

    def _nvlink_counters(self, nv: object, handle: object) -> tuple[int, int]:
        """Cumulative (RX, TX) NVLink data in KiB for one device, summed over ALL links.

        ``(-1, -1)`` when the counters aren't readable at all. Prefers the documented
        all-links aggregate scope (``_NVLINK_SCOPE_ALL``); a driver that rejects that
        scope falls back to summing the per-link scopes in a single call, so an older
        stack still reports the whole fabric instead of link 0 alone.
        """
        rx_fid = nv.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX  # type: ignore[attr-defined]
        tx_fid = nv.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX  # type: ignore[attr-defined]
        rx = tx = -1
        with contextlib.suppress(Exception):
            for v in nv.nvmlDeviceGetFieldValues(  # type: ignore[attr-defined]
                handle, [(rx_fid, _NVLINK_SCOPE_ALL), (tx_fid, _NVLINK_SCOPE_ALL)]
            ):
                if v.nvmlReturn != 0:
                    continue
                if v.fieldId == rx_fid:
                    rx = int(_field_value(v))
                elif v.fieldId == tx_fid:
                    tx = int(_field_value(v))
        if rx >= 0 and tx >= 0:
            return rx, tx
        # The aggregate scope is unsupported here: ask for every link explicitly in one
        # call and sum what comes back. Links are contiguous from 0 and an absent one
        # simply reports non-SUCCESS, so the successful values ARE the whole fabric.
        max_links = int(getattr(nv, "NVML_NVLINK_MAX_LINKS", 18) or 18)
        per_rx = per_tx = -1
        with contextlib.suppress(Exception):
            requests = [(fid, link) for link in range(max_links) for fid in (rx_fid, tx_fid)]
            for v in nv.nvmlDeviceGetFieldValues(handle, requests):  # type: ignore[attr-defined]
                if v.nvmlReturn != 0:
                    continue
                val = int(_field_value(v))
                if v.fieldId == rx_fid:
                    per_rx = val if per_rx < 0 else per_rx + val
                elif v.fieldId == tx_fid:
                    per_tx = val if per_tx < 0 else per_tx + val
        if per_rx >= 0 and per_tx >= 0:
            return per_rx, per_tx
        return -1, -1

    def _nvlink_throughput(self, nv: object, devices: list[int]) -> tuple[list[float], list[float]]:
        """Live per-device NVLink (RX, TX) in GB/s from the cumulative DATA counters.

        The THROUGHPUT_DATA_* fields are cumulative KiB, so a rate is the delta
        between two reads over the elapsed time. Returns empty lists when the counters
        aren't readable (older driver, no permission, PCIe-only) so the UI can hide the
        live line — AND on the read that merely seeds the baseline, because there a rate
        is not yet knowable: emitting 0.0 there would claim an idle fabric, which is the
        one thing this line must never say wrongly (the same "displays zero when it
        isn't zero" trap B1 fixed in the formatter). That case is not hypothetical —
        ``--once`` takes exactly one sample, so it reported ``0.0`` on a live H200 job
        whose counters had already carried 90 GB across the links.

        The counters are read across ALL links (see ``_nvlink_counters``), not link 0."""
        rx_out: list[float] = []
        tx_out: list[float] = []
        # MONOTONIC clock for the rate window, like the CPU path (a wall-clock
        # time.time() step — NTP correction, leap second, VM migration — would give
        # a tiny/negative dt and, since this rate has no high-side clamp, an
        # arbitrarily large bogus throughput on the next sample) (A1).
        now = time.monotonic()
        rated = 0
        for idx in devices:
            handle = self._handle_for_device(idx)
            rx_kib, tx_kib = self._nvlink_counters(nv, handle) if handle is not None else (-1, -1)
            rx_rate = tx_rate = 0.0
            if rx_kib >= 0 and tx_kib >= 0:
                prev = self._nvlink_prev.get(idx)
                if prev is not None:
                    dt = now - prev[0]
                    if dt > 0:
                        rated += 1
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
        # EVERY device must have a rate, not merely one of them. The lists are summed
        # across devices, so a device that only just seeded its baseline (or whose
        # counters failed this frame) would otherwise contribute a hard 0.0 to that sum
        # while its siblings kept the list non-empty — understating the fabric (half the
        # real traffic on a symmetric 2-GPU ring) and showing that GPU as idle. "Not yet
        # knowable" is per-device, so suppress the whole line until it is knowable for
        # all of them, exactly as the seeding read already does for the single-device case.
        if rated < len(devices):
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
        rated = 0
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
                rated += 1
                # KB/s → GB/s (decimal): *1000 bytes /1e9 = /1e6. 3 decimals (not 1)
                # so the per-device values stay accurate when the UI sums them across
                # devices — see _nvlink_throughput; the display rounds the sum to 0.1.
                rx_out.append(round(rx_kbps / 1e6, 3))
                tx_out.append(round(tx_kbps / 1e6, 3))
            else:
                rx_out.append(0.0)
                tx_out.append(0.0)
        # Same rule as the NVLink path: a device whose meter is unreadable must not
        # contribute a hard 0.0 to a sum the UI presents as the whole bus, so require
        # EVERY device to have been read rather than merely one of them (the old
        # `got_any` published the fake zero alongside its readable siblings).
        if rated < len(devices):
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

    def _mock_fabric(self) -> NodeFabric:
        """A 200 Gb/s HDR InfiniBand link for the demo, with traffic that breathes.

        Synthesized rather than read, for the same reason the interconnect is: demo
        mode must not present the RECORDING HOST's real counters as the fake job's.
        Without this the row either vanished (a login node has no HCA) or published
        somebody's actual fabric traffic — including into the README GIF, which is
        rendered from demo mode.
        """
        elapsed = time.monotonic() - self._mock_start
        # An all-reduce pattern: mostly busy, dipping between steps, so the "% of
        # link" figure moves through the range a real training job walks.
        rx = round(90 + 85 * (0.5 + 0.5 * math.sin(elapsed * 0.35)), 2)
        tx = round(85 + 80 * (0.5 + 0.5 * math.cos(elapsed * 0.3)), 2)
        return NodeFabric(
            ports=1,
            link_rate_gbps=200.0,
            link_rate_total_gbps=200.0,
            kind="InfiniBand",
            rate_label="200 Gb/sec (4X HDR)",
            rx_gbps=rx,
            tx_gbps=tx,
            rates_known=True,
        )

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


# Sentinel for "this cache slot has never been filled", so a genuine None (meaning
# "looked up, and there is no constraint") is not re-resolved on every frame.
_UNSET = object()

_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

# The NVLink THROUGHPUT_DATA_* field values are PER-LINK counters selected by
# nvmlFieldValue_t.scopeId, and nvml.h documents UINT_MAX as "aggregate value summed
# up across all links for the specified counter type in fieldId". pynvml only sets
# scopeId when the caller passes a (fieldId, scopeId) TUPLE — a bare int leaves it at
# the ctypes default 0, which silently answers for link 0 alone. That read the fabric
# at 1/links_per_gpu of its real rate (18x under-report on an 18-link H200), i.e. a
# saturated NVLink ring looked nearly idle — so always request this scope explicitly.
_NVLINK_SCOPE_ALL = 0xFFFFFFFF

# NVML topology-common-ancestor level → nvidia-smi topo -m PCIe path label, fastest
# (PIX, single PCIe bridge) to slowest (SYS, across NUMA/QPI/UPI). NVML_TOPOLOGY_CPU
# shares NODE's value (40); both mean "same NUMA node, different host bridge".
_TOPO_LABEL = {0: "self", 10: "PIX", 20: "PXB", 30: "PHB", 40: "NODE", 50: "SYS"}

# NVLink generation and per-link one-direction GB/s by DEVICE MODEL, keyed on the
# model token in NVML's product name.
#
# Deliberately NOT derived from nvmlDeviceGetNvLinkVersion: that call returns a
# driver-internal code, not the marketing generation, and its numbering is not stable
# across driver branches. Measured on a live 3x H200 node (driver 535.216.03) it
# returns 7 on every link — while the nvmlNvlinkVersion_enum added in CUDA 12.7
# defines 7 as NVLINK_VERSION_5_0 (Blackwell, 50 GB/s per link) and 6 as 4_0. An H200
# is Hopper, NVLink 4, and nvidia-smi nvlink -s reports 26.562 GB/s per link there, so
# reading the code as a generation prints a wrong number ("NVLink 7") and mapping it
# to a speed the way hwloc does would claim double this card's real per-link rate.
# The model, by contrast, pins both facts unambiguously.
#
# The link COUNT still comes from NVML (measured per device), so the aggregate below
# is spec-exact for every entry: P100 4x20x2 = 160, V100 6x25x2 = 300,
# A100 12x25x2 = 600, H100/H200 18x25x2 = 900, B200 18x50x2 = 1800 GB/s.
# Cards absent from the table (workstation/consumer parts with bridge NVLink, or a
# GPU newer than this build) report no generation and no speed rather than a guess.
_NVLINK_MODEL_SPEC: dict[str, tuple[int, float]] = {
    "P100": (1, 20.0),
    "V100": (2, 25.0),
    "A100": (3, 25.0),
    "H100": (4, 25.0),
    "H200": (4, 25.0),
    "GH200": (4, 25.0),
    "B100": (5, 50.0),
    "B200": (5, 50.0),
    "GB200": (5, 50.0),
}


def _nvlink_model_spec(name: str) -> tuple[int, float]:
    """``(NVLink generation, per-link one-direction GB/s)`` for an NVML product name.

    Matches on whole tokens, so ``GH200`` can't be mistaken for ``H200`` (nor
    ``GB200`` for ``B200``) and a name that merely contains a model string doesn't
    match. ``(0, 0.0)`` when the model isn't in the table, which the caller renders
    as a bare "NVLink" with no generation and no bandwidth.
    """
    for token in re.split(r"[^0-9A-Za-z]+", name.upper()):
        spec = _NVLINK_MODEL_SPEC.get(token)
        if spec is not None:
            return spec
    return 0, 0.0


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


def _pid_alive(pid: int) -> bool:
    """Whether ``/proc/<pid>`` still exists (the PID is live or an unreaped zombie).

    Used by the /proc CPU accumulator to evict only genuinely-dead PIDs, so a
    still-live PID that briefly fell out of one enumeration is never forgotten and
    re-counted from scratch. Split out (module scope) so it's mockable in tests.
    """
    return Path(f"/proc/{pid}").exists()


def _working_set_from_stat(stat: str, current_bytes: int, prefix: str) -> tuple[int, int]:
    """(working_set, reclaimable_file_cache) from a cgroup memory.stat.

    File-backed page cache (inactive_file + active_file) is clean and reclaimed
    by the kernel before it OOM-kills a job, so it's excluded from the working set
    that drives the OOM guard. The working set is computed as ``current − file
    cache``, so it retains everything else — anonymous, shmem, AND kernel memory
    (slab/pagetables, ~0.5%); that's deliberately conservative (it can only
    over-, never under-, state the OOM-relevant footprint). cgroup v1 uses
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
        if not g.utilization_supported:
            # Device-wide utilization is genuinely UNSUPPORTED (a MIG slice, where
            # the rate APIs return NOT_SUPPORTED). Without a util reading, "0%" is
            # meaningless, so fall back to VRAM occupancy. Per-process VRAM
            # (process_memory_bytes) is *also* frequently NOT_AVAILABLE on MIG, so
            # don't rely on it alone — that made an actively-used slice read as
            # "idle" (crit) whenever NVML withheld both signals (#36). A MIG slice's
            # used VRAM is isolated to this job, so it's a clean "in use" signal.
            # Only a slice with no readable activity at all is scored inactive.
            return g.process_memory_bytes > 0 or g.memory_used_bytes > 0
        # Util IS supported but this poll's read failed transiently. We can't see
        # device util this frame, so DON'T credit device-wide VRAM — on a shared,
        # non-isolated GPU it may be another tenant's. Score active only if the job
        # positively owns the majority of the used VRAM, matching the guard on the
        # normal util path below (A7).
        return g.process_memory_bytes > 0 and g.process_memory_bytes >= 0.5 * g.memory_used_bytes
    # The `memory_used_bytes > 0` conjunct is a sanity check that a busy-looking device
    # really has something resident — but only when VRAM was actually READ. When the VRAM
    # query itself failed, that 0 is not a measurement, and vetoing on it scored a GPU
    # reporting 99% utilization as IDLE (amber "idle", gpu_active_count=0). That is
    # reachable and persistent, not a transient: nvidia-ml-py >= 11.510 raises
    # FunctionNotFound from nvmlDeviceGetMemoryInfo_v2 against a pre-510 driver.
    if g.utilization_percent > idle_threshold and (
        g.memory_used_bytes > 0 or not g.memory_available
    ):
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


class _PidCpu(NamedTuple):
    """One process's CPU as /proc/<pid>/stat reports it, in clock ticks."""

    own: int  # utime + stime — this process's own CPU
    children: int  # cutime + cstime — the CPU of children it has already REAPED
    ppid: int  # parent, so a dying child's ticks can be reconciled (see below)


def _parse_stat_cpu(data: str) -> _PidCpu | None:
    """Own and reaped-children CPU (clock ticks) plus ppid, from /proc/<pid>/stat.

    The comm field (2nd) may contain spaces and parentheses, so the fields after it
    are located relative to the final ')'.

    ``cutime``/``cstime`` are where the kernel books a child's CPU once its parent
    reaps it, and reading only ``utime``/``stime`` is why a job whose work happens in
    short-lived children (an R worker `system()`-ing a shell in a loop, `make -j`, a
    per-file pipeline) read as little as 1% of its true load on a cluster with no
    per-job cpuacct cgroup: each child's CPU vanished the moment it was reaped, and a
    child born and reaped between two polls was never seen at all. SW-24.
    """
    rparen = data.rfind(")")
    if rparen == -1:
        return None
    fields = data[rparen + 1 :].split()
    # After comm: state(0) ppid(1) … utime(11) stime(12) cutime(13) cstime(14)
    if len(fields) < 15:
        return None
    try:
        return _PidCpu(
            own=int(fields[11]) + int(fields[12]),
            children=int(fields[13]) + int(fields[14]),
            ppid=int(fields[1]),
        )
    except ValueError:
        return None


def _parse_stat_cpu_ticks(data: str) -> int:
    """Total CPU (own + reaped children) in clock ticks, or 0 if unreadable."""
    parsed = _parse_stat_cpu(data)
    return 0 if parsed is None else parsed.own + parsed.children


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


def _read_pid_cpu(pid: int) -> _PidCpu | None:
    """Per-process CPU (own, reaped children, ppid) for a PID, or None if gone."""
    try:
        data = Path(f"/proc/{pid}/stat").read_text(errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    return _parse_stat_cpu(data)


def _read_pid_cpu_ticks(pid: int) -> int:
    """Total CPU (own + reaped children) in clock ticks for a PID."""
    parsed = _read_pid_cpu(pid)
    return 0 if parsed is None else parsed.own + parsed.children


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
