from __future__ import annotations

import json
import math
import os
import socket
from dataclasses import asdict, dataclass, field
from typing import Any

from .units import format_cores, printable_text


def _csv_text(value: str) -> str:
    """A free-form text field made safe to open in a spreadsheet.

    The ``csv`` module quotes delimiters, quotes and newlines, but quoting does NOT stop
    Excel / LibreOffice / Sheets from EVALUATING a cell whose text begins ``= + - @`` (or
    a lone tab/CR) as a formula. A job name is arbitrary user text — ``sbatch -J
    '=cmd|"/bin/sh"!A1'`` is a live DDE cell, not a label — so prefix a single quote,
    the conventional "treat this as text" marker, and leave ``--json`` untouched.
    """
    # Control characters first: the terminal is the third interpreter of this field
    # (see units.printable_text), and after this pass a leading tab/CR cannot occur —
    # they arrive as the two printable characters ``\`` and ``t``.
    value = printable_text(value)
    return "'" + value if value[:1] in ("=", "+", "-", "@") else value


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/Infinity) with None.

    Lets ``to_json`` pass ``allow_nan=False`` (spec-compliant JSON that ``jq`` and
    other RFC-8259 parsers accept) without crashing a long ``--log`` run if a stray
    non-finite metric ever appears — unreachable today (the collector's divisions
    are guarded), so this is purely defensive.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


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
    # WHICH counter produced usage_ns: "v2"/"v1" (the cgroup's own, which also
    # captures children that already exited), "proc" (a sum over the job's live PIDs,
    # the only option on a cluster that constrains with cpuset but creates no per-job
    # cpuacct), "sstat" (off-node) or "mock". `_read_cpu_ns`'s docstring has
    # always said these "are not comparable to each other" — and MemoryMetrics has
    # published its `source` since SW-3 for exactly that reason, while this one stayed
    # internal. Measured on a live reservation: the "proc" sum read 127,053 CPU-s
    # where `sacct TotalCPU` said 612 s and `sstat AveCPU` said 00:00.000, because
    # jobacct_gather polls the step's task tree and this counts every PID in the
    # cgroup. A consumer that cannot see which counter answered cannot reconcile that.
    source: str = ""

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


# The tail of the CPU-underuse advice, shared so the dashboard's insight line and
# the plain-text summary cannot drift apart. It leads with the scheduling argument
# on purpose: "would schedule faster" is what actually moves someone to shrink a
# request, where "you are wasting cores" does not.
CPU_UNDERUSE_ADVICE = "a smaller --cpus-per-task would schedule faster and free the rest"


def cpu_underuse_subject(cpu: CpuMetrics) -> str:
    """The clause both surfaces put in front of :data:`CPU_UNDERUSE_ADVICE`.

    The advice *tail* has been shared since SW-18, and the dashboard says so at
    its call site: "this line and the plain-text summary's cannot drift; only
    the ink on the flag differs."  That was true of the tail alone -- the
    subject half was spelled once in ``cli.py`` and once in ``tui.py``, and it
    is the half that actually drifted: one wrote the core figure with a private
    formatter that dropped a trailing ``.0`` and the other with ``:.1f``, so
    one job read ``1 of 8`` on the card and ``~1.0 of 8`` in the summary.
    Sharing the formatter fixed that instance; sharing the sentence is what
    makes the comment true.

    Markup is the caller's business, which is why this returns the words only.
    """
    return (
        f"only ~{format_cores(cpu.effective_cores)} of {cpu.cores_allocated} cores are doing work"
    )


def cpu_ratio(cpu: CpuMetrics) -> float:
    """Busy cores as a fraction of allocated; 0 when nothing is allocated."""
    if cpu.cores_allocated <= 0:
        return 0.0
    return cpu.effective_cores / cpu.cores_allocated


def cpu_is_underused(cpu: CpuMetrics, threshold: float) -> bool:
    """Whether this job is holding materially more cores than it is using.

    A single-core allocation can't be "underused". Lives here rather than in the
    TUI because the degraded plain-text summary has exactly the same inputs and
    should reach the same verdict — it didn't, so the readers who CANNOT get the
    live dashboard (a cluster that forbids step creation, or a redirect) were the
    only ones not told they had asked for 8x what they use. SW-18.
    """
    return cpu.cores_allocated > 1 and cpu_ratio(cpu) < threshold


@dataclass
class MemoryMetrics:
    current_bytes: int
    limit_bytes: int
    # The job's peak TOTAL footprint — normally the cgroup's own high-water counter
    # (v1 memory.max_usage_in_bytes / v2 memory.peak), which survives sw restarts and
    # a late attach. It is CACHE-INCLUSIVE (anon + page cache + kmem), so for a
    # cache/mmap-heavy job it reads well above the anonymous high-water mark. Size
    # --mem against peak_working_set_bytes instead; this stays as the total.
    # Always >= current_bytes. Check `peak_is_lifetime` before calling it a lifetime
    # figure: where the kernel exposes no counter this is a since-attach running max.
    peak_bytes: int
    usage_percent: float
    oom_guard_warning: bool
    oom_guard_critical: bool
    working_set_bytes: int = 0
    cache_bytes: int = 0
    # The number to size --mem against: the running max of the working set
    # (anon + shmem + kmem, CACHE-EXCLUDED) since monitoring began. The kernel
    # exposes no cache-excluded lifetime peak, so unlike peak_bytes this is a
    # since-session figure (like the CPU peak); peak_bytes remains the
    # cache-inclusive lifetime total for reference.
    peak_working_set_bytes: int = 0
    # The working set as a percent of the limit (cache-EXCLUDED), clamped to 100.
    # This is what the TUI shows as the MEM gauge; emitted here too so a --json/CSV
    # consumer sizing --mem sees the same working-set figure, not only the
    # cache-INCLUSIVE `usage_percent` (which can read far higher for a mmap-heavy
    # job and drive an over-request).
    working_set_percent: float = 0.0
    # WHERE these numbers came from — "cgroup" (the memcg's own counter), "proc" (a
    # sum of /proc/<pid>/statm over the job's live PIDs, the only option when no
    # memory controller is delegated; it counts shared pages, so it over-reports, and
    # it sees nothing of processes that already exited), "sstat" (off the node) or
    # "mock" (--demo). Same vocabulary as CpuMetrics.source, for the same reason: the
    # counters are not comparable. Off-node the fields mean different things under
    # the same names: `current_bytes` is sstat's MaxRSS, i.e. a lifetime HIGH-WATER
    # that never falls, `peak_bytes` is a copy of it, and there is no cache
    # breakdown at all. `remote` on the snapshot said the reading was off-node but
    # not that the SEMANTICS changed, so a consumer sizing --mem off `peak_bytes`
    # could not tell a real high-water from a copy of one instantaneous sample.
    # SW-3.
    source: str = "cgroup"
    # False when nothing measured the page cache, so `cache_bytes: 0` must not be
    # read as "this job has no reclaimable cache" — off-node, sstat reports no
    # cache at all. "Not measured" and "measured zero" are different claims.
    cache_measured: bool = True
    # Is `peak_bytes` a KERNEL lifetime counter, or our own running max?
    #
    # It is a lifetime figure when the kernel handed us one: v1
    # `memory.max_usage_in_bytes`, v2 `memory.peak`, or sstat's MaxRSS. But v2 only
    # gained `memory.peak` in kernel 5.19, so on a cgroup-v2 cluster running an
    # older kernel (RHEL/Rocky 9 ships 5.14 — a large share of clusters) there is no
    # counter to read and `peak_bytes` becomes a running max of `memory.current`
    # taken since monitoring began. Same field, different meaning: no pre-session
    # history, and a restart resets it.
    #
    # Defaults to False on the SW-3 principle — a payload that does not state its
    # provenance must not have provenance invented for it. Every code path that
    # really did read a kernel counter says so explicitly.
    peak_is_lifetime: bool = False

    @property
    def no_limit_set(self) -> bool:
        """Whether `usage_percent`/`working_set_percent` are a ratio of nothing.

        `limit_bytes == 0` means "no limit is enforced" — the meaning
        `_parse_mem_to_bytes` returns `None` for an unreadable spelling to protect
        (SW-12) — and both human surfaces then refuse to show a percentage at all:
        the MEM row drops its bar for "26.0 GiB · no limit set" because "a 'used 0%'
        bar would contradict the GiB in use", and the plain summary prints "peak 26.0
        GiB (no limit set)". A property rather than two copies of `limit_bytes <= 0`,
        for the reason SW-4 gives about the gauge and the summary: the CSV row and
        the JSON payload have to answer this the same way or one of the two formats
        drifts back to publishing the bare zero.

        Normal off-node, where it is the only reachable spelling: `_collect_remote`
        copies `ctx.mem_limit_bytes` straight through, so a job submitted with no
        `--mem` on a cluster with no DefMemPerCPU has no limit in every row. On-node
        a missing cgroup cap falls back to node RAM instead.
        """
        return self.limit_bytes <= 0

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
    # False ONLY when the device-util rate API is genuinely unsupported
    # (NVMLError_NotSupported — e.g. a MIG slice), vs. a transient read failure
    # that leaves utilization_available False but utilization_supported True. The
    # active/idle heuristic uses this to fall back to device-wide VRAM ONLY for a
    # MIG slice (where that VRAM is isolated to the job) — a transient failure on a
    # shared GPU keeps the majority-owner guard so it can't credit another tenant's
    # VRAM as this job's activity (A7).
    utilization_supported: bool = True
    # The enforced power cap (W), 0 when unreadable. Shown as "used / cap W" so
    # headroom-to-cap is visible; a GPU pegged at its cap is well-utilised, not sick.
    power_limit_watts: float = 0.0
    # The specific active throttle reasons (e.g. ["sw_power_cap"]) behind
    # ``throttling``, so a --json consumer can tell a benign power cap (the ideal
    # steady state of a power-limited GPU) apart from a thermal/hardware slowdown.
    # The TUI intentionally surfaces neither as a status word.
    throttle_reasons: list[str] = field(default_factory=list)
    # False when NVML couldn't read VRAM / power / temperature. Each of those fields
    # initialises to 0, and 0 is a perfectly plausible MEASUREMENT, so without a flag a
    # failed read was presented as fact: a MIG slice (where the rate APIs and often the
    # power/temp APIs return NOT_SUPPORTED) rendered as an unpowered, below-freezing card
    # at "0 W · 0°C (32°F)", and an unreadable VRAM read made `_gpu_is_active` score a
    # 99%-busy GPU as IDLE — because the activity heuristic vetoes on
    # ``memory_used_bytes > 0`` — which also zeroed gpu_active_count and reported
    # "0% HBM" on a full card, the exact figure a user sizes their batch size from.
    # Same role as ``utilization_available``, which exists for precisely this reason.
    memory_available: bool = True
    power_available: bool = True
    temperature_available: bool = True
    # Whether the job's OWN share of the device was measurable.
    # nvmlDeviceGetProcessUtilization is optional: it raises NOT_SUPPORTED on MIG
    # slices and old drivers, and NO_PERMISSION where the process APIs are
    # restricted, leaving `process_utilization_percent` at 0.0 — indistinguishable
    # from "this job used none of the GPU". Same role as the four flags above; a
    # right-sizing consumer reading the 0 as a measurement would advise dropping a
    # GPU the job is actually using.
    process_utilization_available: bool = True
    # The device's CUDA ordinal — the number the JOB'S OWN CODE addresses it by
    # (``cuda:0``) — as distinct from ``index``, which is NVML's device index (what
    # ``nvidia-smi`` prints). They're equal on a cluster with device-cgroup isolation
    # (``ConstrainDevices=yes``), because NVML then exposes only the job's GPUs,
    # renumbered from 0. WITHOUT that isolation NVML sees the whole node, so a job
    # holding the node's GPUs 2 and 3 has ordinals 0 and 1 against indices 2 and 3 —
    # and labelling its devices "CUDA 2"/"CUDA 3" would name numbers its code never
    # uses. -1 when unknown (a remote node running a build that predates this field),
    # in which case the UI falls back to ``index``.
    cuda_ordinal: int = -1

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


@dataclass
class NodeFabric:
    """The node's inter-NODE network (InfiniBand / RoCE) and how hard it is working.

    Distinct from :class:`GpuInterconnect`, which is strictly INTRA-node (how this
    node's GPUs reach each other). For a multi-node job the number that actually
    explains a slow step is usually this one: gradient all-reduce crosses the
    fabric, and NVML's PCIe counters never see it (with GPUDirect RDMA the transfer
    goes GPU→NIC and may not appear as host PCIe traffic at all).

    ``rx_gbps``/``tx_gbps`` are live rates derived from the port counters, which are
    **node-wide**: on a shared node another job's traffic is included, so the UI must
    not present this as the job's own. ``ports`` counts the ACTIVE ports summed.
    """

    ports: int = 0
    link_rate_gbps: float = 0.0  # per active port, one direction, as the HCA reports
    # Every active port's rate SUMMED — the node's actual ceiling, and the only
    # honest denominator for a traffic figure that is itself summed across ports.
    # Dividing the sum by ONE port's rate let a busy 2-HCA node report "180% of
    # 100 Gb/s link", a number that cannot be true. link_rate_gbps stays as the
    # per-port figure because "100 Gb/s" is what an operator recognises.
    link_rate_total_gbps: float = 0.0
    kind: str = ""  # "InfiniBand" / "RoCE" / "" when unknown
    rate_label: str = ""  # the HCA's own words, e.g. "100 Gb/sec (2X HDR)"
    rx_gbps: float = 0.0
    tx_gbps: float = 0.0
    # False until two samples exist (the counters are cumulative, so a rate needs a
    # delta). Keeps a first frame from publishing a fake 0.0 as a measurement.
    rates_known: bool = False

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


@dataclass
class GpuInterconnect:
    """How the job's GPUs on one node are wired to each other.

    Only meaningful for a multi-GPU job — a single device has nothing to
    interconnect — so the collector populates it only when >1 of the job's GPUs
    are visible on the node. The wiring (NVLink generation, per-link speed, the
    pairwise topology) is fixed for the life of the job, so it's probed once and
    reused; only ``rx_mibps``/``tx_mibps`` (live NVLink traffic) change per frame.

    ``matrix`` is the symmetric device-by-device grid ``nvidia-smi topo -m``
    prints: ``matrix[i][j]`` describes the path between ``devices[i]`` and
    ``devices[j]`` — ``"self"`` on the diagonal, ``"NV<k>"`` for k NVLinks, or a
    PCIe class (``PIX``/``PXB``/``PHB``/``NODE``/``SYS``, fastest→slowest).
    """

    # Overall fabric wiring the GPUs together, worst-case across pairs:
    # "nvlink" (every pair has NVLink), "mixed" (some NVLink, some PCIe),
    # "pcie" (no NVLink between any pair), or "unknown" (couldn't probe).
    fabric: str = "unknown"
    # NVLink generation (2=V100, 3=A100, 4=H100/H200, 5=B200); 0 when unknown. Derived
    # from the device MODEL, not from nvmlDeviceGetNvLinkVersion — that call returns a
    # driver-internal code whose numbering isn't the marketing generation (a live H200
    # reports 7 on driver 535, where CUDA 12.7's enum defines 7 as NVLink 5.0).
    nvlink_version: int = 0
    links_per_gpu: int = 0  # active NVLinks on a typical device
    link_speed_gbps: float = 0.0  # per link, one direction
    per_gpu_gbps: float = 0.0  # aggregate bidirectional NVLink bandwidth per device
    nvswitch: bool = False  # links terminate on NVSwitch(es) → all-to-all fabric
    devices: list[int] = field(default_factory=list)  # device indices, in matrix order
    matrix: list[list[str]] = field(default_factory=list)  # symmetric NxN topology cells
    # Live per-device data-transfer rate in GB/s (decimal, to match the GB/s speeds
    # above), aligned with ``devices``. NVLink comes from the fabric throughput
    # counters; PCIe from the live PCIe meter (host↔GPU plus any P2P over PCIe).
    # Each list is empty when that link's counters aren't readable (older driver, no
    # permission, or that fabric isn't present).
    nvlink_rx_gbps: list[float] = field(default_factory=list)
    nvlink_tx_gbps: list[float] = field(default_factory=list)
    pcie_rx_gbps: list[float] = field(default_factory=list)
    pcie_tx_gbps: list[float] = field(default_factory=list)

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
    # How old the CPU/memory measurement in this row is, in seconds. 0.0 on-node,
    # where every sample re-reads the cgroup. Off-node it matters: sstat is queried at
    # most every 5s (and Slurm samples it far less often than that), so a 1s --log
    # off-node writes four re-serialisations of one measurement for every fresh
    # query — measured 17 of 21 consecutive rows byte-identical in cpu+mem, then a
    # 5x jump. A consumer CANNOT recover that by diffing rows, because an unchanged
    # cpu_usage_ns is also exactly what an idle job produces: without this field
    # "stale repeat" and "did no work" are the same row. -1.0 when a payload came
    # from a build that did not report it (same convention as cuda_ordinal).
    usage_age_seconds: float = 0.0
    # Whether the CPU/memory figures in this row are a MEASUREMENT at all. True
    # on-node, where the cgroup is always readable. Off-node it can be False: Slurm's
    # accounting samples roughly every 30s, so a young job (or one on a site where
    # sstat is unavailable) has no sample yet — and every metric then reads 0. The
    # plain-text summary has always said "usage not yet sampled by Slurm" for that
    # state; the machine payload published `usage_ns: 0`, `limit_bytes: 0` and
    # `source: "sstat"` instead, which a right-sizing consumer reads as "this job uses
    # nothing" and acts on by shrinking --mem and --cpus-per-task to the floor. Same
    # defect as an unread GPU reported as 0% (see gpu_active_count), on the two fields
    # that matter most. Absent from an older build's payload reads as True: unlike
    # usage_age_seconds there is nothing to re-derive it from, and marking every row
    # from an older node unsampled would be its own lie.
    usage_sampled: bool = True
    # How many of the job's GPUs are doing work. None — not 0 — when NOTHING could be
    # read: this is a SUM over `gpus`, so an unreadable device set collapses to a bare
    # 0 that a consumer cannot tell apart from "read all four, all four idle". The
    # zero is exactly the figure a right-sizing script acts on, and acting on it means
    # advising the user to drop GPUs their job is busy using — the failure the
    # per-metric `*_available` flags on GpuMetrics exist to prevent, and the one
    # `gpu_count`'s CSV note describes for a non-NVIDIA node. Off-node this is the
    # NORMAL case, not an edge one: a monitor step beside a job holding every GPU it
    # was given is denied /dev/nvidiaN ("devices_denied"), so `--once` against any
    # ordinary multi-GPU job published "0 of 4 active" about cards it never saw.
    # Stays 0 when the job asked for no GPU at all — "none active" is then a fact.
    gpu_active_count: int | None = 0
    # The job's name (sbatch -J), carried beside job_id so a log or a --json capture
    # says WHICH experiment it measured, not just which numeric record. "" when Slurm
    # reports none. Free-form user text: escape it before rendering, and note that a
    # CSV consumer gets it quoted by the csv module like any other field.
    job_name: str = ""
    # The job's wall-clock limit, the DENOMINATOR for `elapsed_seconds`. Both the
    # dashboard's time-budget line and the foreign-job payload have carried it, but the
    # telemetry payload did not — so `--once --json` reported "this job has run 282279
    # seconds" with nothing to compare it against, while the SAME machine surface for
    # somebody ELSE's job did include the limit. "How much of my wall-clock budget is
    # gone" is the most common reason to look at a running job at all, and it was the
    # one arithmetic a consumer could not do. None when the job has no limit (or Slurm
    # reports UNLIMITED).
    time_limit_seconds: int | None = None
    # Identity a consumer needs to ATTRIBUTE a row, all of it already resolved into
    # JobContext and all of it already on a sibling surface: the foreign-job payload
    # carries `owner` and `partition`, and the JOB card shows account/QOS to a human.
    # The telemetry payload carried none of them, so a `--log` file accumulating rows
    # across jobs — or a pipeline over `--once --json` — could not group by the two
    # axes every cluster reports on (partition, account) or say whose job a row was.
    # "" when Slurm reports nothing.
    partition: str = ""
    owner: str = ""
    account: str = ""
    qos: str = ""
    # An array task's two halves, so a log can be grouped by the ARRAY rather than by
    # each task's composite id. `job_id` carries `12345_3` and a consumer could split
    # it, but the foreign payload states both explicitly and this one did not — the
    # same asymmetry as the four fields above. "" for a job that is not an array task.
    array_job_id: str = ""
    array_task_id: str = ""
    # True when the sample is a job-wide sstat estimate collected off the compute
    # node (no cgroups / NVML reachable), not live per-node telemetry. Memory is a
    # lifetime peak (MaxRSS), CPU is an average, and neither can be attributed to a
    # single node — so consumers must not read the memory figure as a live per-node
    # "current" or drive a (never-clearing) OOM alarm off it (#34, #35).
    remote: bool = False
    # True when these numbers were SIMULATED (`--demo` / SLURMWATCH_MOCK=1), not
    # measured. The flag is documented and the dashboard is obviously a demo to a
    # human, but `--once --json` is the form a pipeline consumes with nobody reading
    # it — and the env var is a documented equivalent of the flag, so a wrapper or a
    # leftover export can turn simulated figures into ingested "measurement". The
    # memory block already said `source: "mock"`, but nested one level down, where a
    # consumer reading `cpu` or `gpus` never looks. Top level, so it cannot be missed.
    # SW-30.
    mock: bool = False
    # False when NVML/pynvml couldn't be brought up on this node at all (no NVIDIA
    # driver, pynvml not installed, or 0 devices) — distinct from NVML working but
    # the job's own GPUs not being visible to the monitor. Lets the UI tell "no GPU
    # telemetry here" apart from the genuine "GPU held by your srun step" case (F3).
    gpu_monitoring_available: bool = True
    # WHY GPU telemetry is missing, so a consumer can act on the cause instead of
    # guessing it from a bare False. "" = nothing to explain. "no_pynvml" /
    # "no_driver" / "nvml_error" / "no_devices" = there is genuinely nothing to read
    # here. "devices_denied" = the node HAS NVIDIA GPUs but this process was given
    # none of them (Slurm's ConstrainDevices denies /dev/nvidiaN to a step allocated
    # no GPU), which is the common case for a monitor step beside a job that holds
    # all its GPUs — and which used to be reported as a missing driver.
    gpu_unavailable_reason: str = ""
    # The NVIDIA GPUs physically present on this node, from procfs (which the device
    # cgroup does not hide). Lets the UI name the hardware — "2 of the node's 4 x
    # A100" — even in the "devices_denied" case where NVML can read nothing at all.
    gpu_node_count: int = 0
    gpu_node_model: str = ""
    # The node-global GPU indices Slurm allocated to this job on THIS node (Slurm's
    # ``GRES=gpu:2(IDX:0,2)``). Already resolved for attaching NVML handles; carried
    # into the snapshot so the "can't read them" path can still answer the question
    # the user actually has — did my job get its GPUs on this node, and which ones.
    gpu_allocated_indices: list[int] = field(default_factory=list)
    # How the job's GPUs are wired to each other (NVLink/PCIe topology + live
    # traffic). Populated only for a multi-GPU node — None for CPU-only, single-GPU,
    # or off-node (sstat) samples, where there's no interconnect to report.
    interconnect: GpuInterconnect | None = None
    # The node's inter-NODE fabric (InfiniBand/RoCE). None when the node has no
    # such HCA, or off-node where there is nothing local to read.
    fabric: NodeFabric | None = None

    def active_gpu_count(self) -> int | None:
        """Active devices, or None when nothing could be read — DERIVED, not trusted.

        The producer already sets the field to None in that case, but a snapshot can
        also be assembled by hand, replayed from a log line, or forwarded by a node
        running a build that predates the distinction, and each of those can carry a
        summed 0. Deriving it here means the JSON, the CSV and the reader agree
        whatever built the object.
        """
        if not self.gpus and not self.gpu_monitoring_available and self.gpu_count_requested > 0:
            return None
        return self.gpu_active_count

    def to_json(self) -> str:
        payload = asdict(self)
        payload["gpus"] = [g.to_dict() for g in self.gpus]
        payload["gpu_active_count"] = self.active_gpu_count()
        if self.memory.no_limit_set:
            # `null`, the JSON spelling of the empty CSV cell — see `to_csv_row`. Both
            # formats have to withhold it or the documented "one dataset, two
            # vocabularies" rule stops holding for `memory.usage_percent` /
            # `mem_percent`, and JSON is the DEFAULT for `--log` (only a `.csv`
            # extension picks the other). Safe to replay: `from_dict`'s `_only`
            # already coerces a null in a numeric field back to 0.0, and a snapshot
            # rebuilt from this line still carries `limit_bytes: 0`, so every renderer
            # takes the "no limit set" branch that never shows a percent anyway.
            payload["memory"]["usage_percent"] = None
            payload["memory"]["working_set_percent"] = None
        # allow_nan=False keeps output spec-compliant (jq rejects NaN/Infinity);
        # _json_safe sanitizes any stray non-finite first so it can't raise.
        return json.dumps(_json_safe(payload), default=str, allow_nan=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TelemetrySnapshot:
        """Reconstruct a snapshot from a ``to_json`` payload.

        Used by the node switcher to turn another node's ``--once --json`` output
        back into a snapshot. Unknown keys are ignored so a small version skew
        between nodes can't crash the parse.
        """

        def _only(cls_: Any, src: dict[str, Any]) -> dict[str, Any]:
            # Coerce a null in a numeric field to zero. to_json maps a non-finite float to
            # `null` so the line stays valid RFC-8259, but feeding that back into a
            # non-optional float/int raised TypeError — and
            # remote.parse_snapshot_line swallows any exception as "unparseable", so the
            # node switcher silently showed that node as producing NO data at all, with no
            # diagnostic. Zeroing the one unrepresentable metric keeps the frame.
            out: dict[str, Any] = {}
            for k, v in src.items():
                fld = cls_.__dataclass_fields__.get(k)
                if fld is None:
                    continue  # unknown key: version skew between nodes
                # ...and a non-finite float the same way, for the same reason. The
                # producer's half of this contract (`_json_safe`) maps NaN/Infinity to
                # `null` precisely so the line stays RFC-8259, and the `null` branch
                # above is what catches it -- but `json.loads` is more permissive than
                # the RFC the producer targets and reads a bare `NaN`/`Infinity` token
                # as a real float. A build that predates `allow_nan=False` emits exactly
                # that (`json.dumps` defaults to `allow_nan=True`), so the mixed-version
                # hop this method exists to survive is the one case that got through:
                # `nan` reached the TUI, where `_labeled_bar` printed "nan%" beside a
                # bar clamped to empty and `_area_chart` raised
                # "cannot convert float NaN to integer". Both halves now agree that an
                # unrepresentable metric is zero here.
                if str(fld.type) in ("int", "float") and (
                    v is None or (isinstance(v, float) and not math.isfinite(v))
                ):
                    v = 0 if str(fld.type) == "int" else 0.0
                out[k] = v
            return out

        # A field the payload does not MENTION must not be filled in with the
        # flattering default. These three say whether a figure beside them is a
        # measurement, and their dataclass defaults are written for the collector
        # (where the answer is yes). A node streaming from an older build omits
        # them, and taking "cgroup"/measured/available on faith would present that
        # node's sstat reading as live cgroup data, its unmeasured 0 cache as "no
        # cache", and an unreadable GPU share as 0%. Absent means unknown here.
        def _unstated(src: dict[str, Any], key: str, unknown: Any) -> dict[str, Any]:
            out = dict(src)
            if key not in src:
                out[key] = unknown
            return out

        mem_raw = _unstated(d["memory"], "source", "unknown")
        mem_raw = _unstated(mem_raw, "cache_measured", False)
        mem_raw = _unstated(mem_raw, "peak_is_lifetime", False)
        gpus_raw = [_unstated(g, "process_utilization_available", False) for g in d.get("gpus", [])]

        # An unreadable device set means the active count is UNKNOWN, not zero. A build
        # that predates that distinction sends a summed 0 here, so re-derive it rather
        # than trust it: a mixed-version hop (the node runs the site's module, the
        # login side a newer wheel) must not be less honest than a matched pair.
        active_raw = d.get("gpu_active_count", 0)
        gpus_unreadable = (
            not gpus_raw
            and not bool(d.get("gpu_monitoring_available", True))
            and int(d.get("gpu_count_requested", 0)) > 0
        )
        active_count = None if active_raw is None or gpus_unreadable else int(active_raw)

        ic_raw = d.get("interconnect")
        interconnect = (
            GpuInterconnect(**_only(GpuInterconnect, ic_raw)) if isinstance(ic_raw, dict) else None
        )
        fab_raw = d.get("fabric")

        return cls(
            timestamp=float(d["timestamp"]),
            job_id=str(d["job_id"]),
            # .get: a node running a build from before the field existed omits it.
            job_name=str(d.get("job_name", "")),
            step_id=(None if d.get("step_id") is None else str(d["step_id"])),
            hostname=str(d["hostname"]),
            # Clamped here as well as at the producer (N10, collector.py:806).
            # The producer computes `max(0, now - job_start_time)` precisely so
            # compute-node clock skew never renders "ran -1:59:56" / "-0%" on the
            # dashboard, but a build predating that clamp streams the raw negative
            # and this took it verbatim -- and version skew across the node hop is
            # what this method exists to survive. Measured with a 1h limit:
            # elapsed=-7196 rendered "-200%" with "02:59:56 left of 01:00:00
            # limit", a remaining that triples the limit it is measured against.
            elapsed_seconds=max(0, int(d["elapsed_seconds"])),
            time_limit_seconds=(
                None if d.get("time_limit_seconds") is None else int(d["time_limit_seconds"])
            ),
            partition=str(d.get("partition", "")),
            array_job_id=str(d.get("array_job_id", "")),
            array_task_id=str(d.get("array_task_id", "")),
            owner=str(d.get("owner", "")),
            account=str(d.get("account", "")),
            qos=str(d.get("qos", "")),
            cpu=CpuMetrics(**_only(CpuMetrics, d["cpu"])),
            memory=MemoryMetrics(**_only(MemoryMetrics, mem_raw)),
            gpus=[GpuMetrics(**_only(GpuMetrics, g)) for g in gpus_raw],
            node_count=int(d.get("node_count", 1)),
            node_index=int(d.get("node_index", 0)),
            gpu_count_requested=int(d.get("gpu_count_requested", 0)),
            # Absent => the far side could not tell us, which is not the same as
            # "fresh"; claiming 0.0 there would invent the guarantee this exists for.
            usage_age_seconds=float(d.get("usage_age_seconds", -1.0)),
            usage_sampled=bool(d.get("usage_sampled", True)),
            gpu_active_count=active_count,
            remote=bool(d.get("remote", False)),
            # Absent on a build that had no marker; False is the safe reading, and a
            # mock payload from such a build still carries memory.source == "mock".
            mock=bool(d.get("mock", False)),
            gpu_monitoring_available=bool(d.get("gpu_monitoring_available", True)),
            gpu_unavailable_reason=str(d.get("gpu_unavailable_reason", "")),
            gpu_node_count=int(d.get("gpu_node_count", 0)),
            gpu_node_model=str(d.get("gpu_node_model", "")),
            gpu_allocated_indices=[int(i) for i in d.get("gpu_allocated_indices", [])],
            interconnect=interconnect,
            fabric=(
                NodeFabric(**_only(NodeFabric, fab_raw)) if isinstance(fab_raw, dict) else None
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> TelemetrySnapshot:
        return cls.from_dict(json.loads(text))

    _GPU_COLS = 21
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
        # Empty (not 0) when there is no HCA or no rate is known yet: a hard 0 would
        # read as a measured idle fabric, a different claim from "not known".
        fab = self.fabric
        ic = self.interconnect
        cols: list[str] = [
            f"{self.timestamp:.3f}",
            self.job_id,
            _csv_text(self.job_name),
            self.hostname,
            str(self.elapsed_seconds),
            "" if self.time_limit_seconds is None else str(self.time_limit_seconds),
            self.partition,
            self.owner,
            self.account,
            self.qos,
            self.array_job_id,
            self.array_task_id,
            str(self.cpu.cores_allocated),
            # The cumulative CPU-time counter, exported so a consumer can difference it
            # instead of integrating a rounded percentage — for a window average that
            # ignores per-frame sampling jitter, or a total for right-sizing.
            #
            # NOT an accounting figure, and the earlier note here said "SU accounting",
            # which is wrong twice over. SUs are billed on ALLOCATED core-time
            # (`AllocCPUS x Elapsed`, Slurm's `CPUTime`), which does not depend on what
            # the job actually used — that is the whole point of right-sizing. And this
            # counter can disagree with Slurm's own MEASURED figure by orders of
            # magnitude, because the two count different things: measured live on this
            # node, 53834744 read 127,053 CPU-seconds here against `sacct TotalCPU`
            # 612 s and `sstat AveCPU` 00:00.000, because jobacct_gather polls the
            # step's TASK TREE while this sums every PID in the job's cgroup — and on a
            # reservation whose processes joined the cgroup outside that tree, Slurm
            # sees almost none of them. Being the higher number is the point of a
            # node-local monitor; pointing a reader at accounting for corroboration was
            # not.
            str(self.cpu.usage_ns),
            self.cpu.source,
            f"{self.cpu.usage_percent:.2f}",
            f"{self.cpu.effective_cores:.2f}",
            f"{self.cpu.peak_effective_cores:.2f}",
            str(self.memory.current_bytes),
            str(self.memory.limit_bytes),
            str(self.memory.working_set_bytes),
            str(self.memory.cache_bytes),
            self.memory.source,
            str(int(self.memory.cache_measured)),
            # Empty, never "0.00", when nothing caps this job's memory — the same
            # convention `time_limit_seconds` above uses for the wall-clock
            # denominator ("empty when the job has no limit") and the fabric rates
            # below use for an unknown rate. The screen this record is written beside
            # refuses the percentage outright and prints "no limit set" instead; the
            # record published `mem_percent=0.00` next to `mem_current_bytes=
            # 27917287424`, which is "this job used 0% of its memory" about a job
            # holding 26 GiB — the one figure a right-sizing consumer acts on, and the
            # bare-zero-as-measurement failure `mem_cache_measured` and
            # `gpu_active_count` already exist to prevent. SW-3.
            "" if self.memory.no_limit_set else f"{self.memory.usage_percent:.2f}",
            "" if self.memory.no_limit_set else f"{self.memory.working_set_percent:.2f}",
            str(self.memory.peak_bytes),
            str(int(self.memory.peak_is_lifetime)),
            str(self.memory.peak_working_set_bytes),
            str(int(self.memory.oom_guard_warning)),
            str(int(self.memory.oom_guard_critical)),
            # The real device count — never capped. With max_gpus sized to fit it
            # equals the number of gpu_<N>_* groups; if it exceeds them it flags
            # that the row was truncated (#38).
            str(len(self.gpus)),
            str(self.gpu_count_requested),
            f"{self.usage_age_seconds:.2f}",
            str(int(self.usage_sampled)),
            # "" (unknown), never a summed 0, when the devices could not be opened.
            "" if self.active_gpu_count() is None else str(self.active_gpu_count()),
            str(self.node_count),
            str(self.node_index),
            str(int(self.remote)),
            str(int(self.mock)),
            # Tells "this tool cannot see this vendor's GPUs" apart from "this job has
            # none". --json carried it and the TUI acts on it, but CSV — the DEFAULT
            # format for --once — showed only gpu_count=0, so on a ROCm/oneAPI node a
            # right-sizing script read a measured zero and advised dropping the GPUs.
            str(int(self.gpu_monitoring_available)),
            # The CAUSE, beside the boolean: a right-sizing script that sees
            # gpu_monitoring_available=0 cannot otherwise tell "this node has no GPU,
            # drop the request" from "the GPUs are allocated and busy, I just could
            # not read them from here" — opposite advice from the same row.
            self.gpu_unavailable_reason,
            str(self.gpu_node_count),
            self.gpu_node_model,
            ";".join(str(i) for i in self.gpu_allocated_indices),
            # The inter-node fabric, in CSV too. --once DEFAULTS to CSV, so a
            # multi-node right-sizing sweep that logs it would otherwise see no
            # network at all and conclude the job is compute-bound while its
            # all-reduce is pinned at 95% of the link — the same JSON-only blind
            # spot that once hid gpu_monitoring_available from CSV consumers.
            fab.kind if fab else "",
            f"{fab.link_rate_gbps:g}" if fab else "",
            f"{fab.link_rate_total_gbps:g}" if fab else "",
            str(fab.ports) if fab else "",
            f"{fab.rx_gbps:g}" if fab and fab.rates_known else "",
            f"{fab.tx_gbps:g}" if fab and fab.rates_known else "",
            # The GPU interconnect, reduced to what a table can carry: the fabric
            # kind, its per-GPU ceiling, and traffic SUMMED across devices. A
            # right-sizing sweep needs "is this job interconnect-bound", which these
            # answer; the NxN topology matrix and per-device lists are not
            # table-shaped and stay --json-only rather than being flattened badly.
            ic.fabric if ic else "",
            f"{ic.per_gpu_gbps:g}" if ic else "",
            f"{sum(ic.nvlink_rx_gbps):g}" if ic and ic.nvlink_rx_gbps else "",
            f"{sum(ic.nvlink_tx_gbps):g}" if ic and ic.nvlink_tx_gbps else "",
            f"{sum(ic.pcie_rx_gbps):g}" if ic and ic.pcie_rx_gbps else "",
            f"{sum(ic.pcie_tx_gbps):g}" if ic and ic.pcie_tx_gbps else "",
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
                        f"{gpu.power_limit_watts:.1f}",
                        f"{gpu.temperature_celsius:.1f}",
                        "1" if gpu.throttling else "0",
                        f"{gpu.process_utilization_percent:.2f}",
                        str(gpu.process_memory_bytes),
                        "1" if gpu.utilization_available else "0",
                        "1" if gpu.utilization_supported else "0",
                        "1" if gpu.process_utilization_available else "0",
                        "1" if gpu.memory_available else "0",
                        "1" if gpu.power_available else "0",
                        "1" if gpu.temperature_available else "0",
                        str(gpu.cuda_ordinal),
                        # WHY it is throttling. A CSV consumer saw only throttling=1 and
                        # could not tell a benign sw_power_cap (the ideal steady state of
                        # a power-limited GPU) from a thermal or hardware slowdown.
                        ";".join(gpu.throttle_reasons),
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
            # Beside the id, so a log says which experiment it measured.
            "job_name",
            "hostname",
            "elapsed_seconds",
            # The denominator for the column above; empty when the job has no limit.
            "time_limit_seconds",
            # Attribution: group a multi-job log by the axes a cluster reports on.
            "partition",
            "owner",
            "account",
            "qos",
            "array_job_id",
            "array_task_id",
            "cpu_cores",
            "cpu_usage_ns",
            # Which counter answered; see CpuMetrics.source.
            "cpu_source",
            "cpu_percent",
            "cpu_effective_cores",
            # The high-water mark to size --cpus-per-task against. CSV carried both
            # memory peaks but no CPU peak, so a CSV consumer couldn't do the
            # right-sizing --json consumers could (the same gap that left
            # working_set_percent out of CSV).
            "cpu_peak_effective_cores",
            "mem_current_bytes",
            "mem_limit_bytes",
            "mem_working_set_bytes",
            "mem_cache_bytes",
            # Beside the figures they qualify: WHERE the memory reading came from
            # ("cgroup"/"sstat"/"mock"), and whether anything measured the cache at
            # all — off-node nothing does, and `mem_cache_bytes=0` there is "not
            # measured", not "no cache". Off-node `mem_current_bytes` is also
            # sstat's MaxRSS, a high-water that never falls, so a consumer sizing
            # --mem needs to know which reading it holds. SW-3.
            "mem_source",
            "mem_cache_measured",
            "mem_percent",
            "mem_working_set_percent",
            "mem_peak_bytes",
            # 1 = a kernel lifetime counter, 0 = a running max since sw attached
            # (cgroup v2 before kernel 5.19 exposes no memory.peak).
            "mem_peak_is_lifetime",
            "mem_peak_working_set_bytes",
            "mem_oom_warning",
            "mem_oom_critical",
            "gpu_count",
            "gpu_count_requested",
            "usage_age_seconds",
            "usage_sampled",
            "gpu_active_count",
            "node_count",
            "node_index",
            "remote",
            # 1 = simulated (--demo / SLURMWATCH_MOCK), not measured (SW-30).
            "mock",
            "gpu_monitoring_available",
            "gpu_unavailable_reason",
            "gpu_node_count",
            "gpu_node_model",
            "gpu_allocated_indices",
            "fabric_kind",
            "fabric_link_rate_gbps",
            # The per-port rate AND the node's aggregate: on a multi-HCA node the
            # traffic columns are summed across ports, so only the aggregate can be
            # divided into them.
            "fabric_link_rate_total_gbps",
            "fabric_ports",
            "fabric_rx_gbps",
            "fabric_tx_gbps",
            "gpu_interconnect",
            "gpu_interconnect_per_gpu_gbps",
            "gpu_nvlink_rx_gbps",
            "gpu_nvlink_tx_gbps",
            "gpu_pcie_rx_gbps",
            "gpu_pcie_tx_gbps",
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
                    f"gpu_{i}_power_limit_watts",
                    f"gpu_{i}_temp_celsius",
                    f"gpu_{i}_throttling",
                    f"gpu_{i}_proc_util_percent",
                    f"gpu_{i}_proc_mem_bytes",
                    f"gpu_{i}_util_available",
                    f"gpu_{i}_util_supported",
                    # Whether gpu_<i>_proc_util_percent is a measurement at all.
                    f"gpu_{i}_proc_util_available",
                    # 0 is a plausible VRAM / power / temperature reading, so these say
                    # whether the neighbouring number is a measurement at all.
                    f"gpu_{i}_mem_available",
                    f"gpu_{i}_power_available",
                    f"gpu_{i}_temp_available",
                    # The CUDA ordinal, alongside the NVML index in gpu_<i>_index.
                    # The GROUP number i is positional and so normally the ordinal
                    # too, but a device dropped from one frame shifts the groups —
                    # this column says which device the row really describes.
                    f"gpu_{i}_cuda_ordinal",
                    f"gpu_{i}_throttle_reasons",
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
    # {node: allocated GPU indices} for the WHOLE job, from `scontrol show job -d`.
    # Lets the GPU view answer "which GPUs did my job get, on every node" in one
    # place — the only GPU fact available when the job holds every GPU and no
    # monitor step can read utilization. Empty for a CPU-only job.
    gpu_indices_by_node: dict[str, list[int]] = field(default_factory=dict)
    min_memory_node: int = 0
    tres: str = ""
    # The SHARED-GPU GRES the job asked for, spelled as Slurm records it —
    # ``"shard:2"`` / ``"mps:100"`` — and ``""`` when it asked for none.
    #
    # Deliberately NOT folded into ``gpu_count_requested``, and deliberately not a
    # number: ``gres/shard`` and ``gres/mps`` allocate a FRACTION of a device, so
    # neither figure is a device count (two shards can be two slices of one physical
    # GPU, and ``mps:100`` is 100% of one). It exists because the alternative was
    # worse than saying nothing: with only a count, every renderer read the
    # uncountable request as a measured zero and positively stated "no GPUs
    # requested by this job" about a job that had asked for one (D18).
    gpu_fraction_request: str = ""
    # Job provenance parsed from the same `scontrol show job -d` record — shown
    # in the dashboard's JOB card so "what exactly is this job" is answerable.
    # Empty string / None when the field wasn't present.
    #
    # job_name is the ``sbatch -J`` / ``--job-name`` label (scontrol JobName): the
    # one field the USER chose, and so the fastest way to tell which of several
    # running jobs you're looking at — an id answers "which record", a name answers
    # "which experiment". Free-form, so every render path must escape it.
    job_name: str = ""
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
    # The compact scontrol NodeList string as read (e.g. "cn[001-500]"), kept
    # alongside the expanded `nodelist` so displays stay human-sized on wide jobs
    # (B3). Empty when unknown (mock/foreign paths that only have the node list).
    nodelist_compact: str = ""

    @property
    def nodelist_display(self) -> str:
        """A human-sized nodelist for display: the compact scontrol form
        (``cn[001-500]``) when available, else the expanded comma list. Code that
        needs individual nodes must use ``nodelist_resolved``, not this."""
        return self.nodelist_compact or self.nodelist
