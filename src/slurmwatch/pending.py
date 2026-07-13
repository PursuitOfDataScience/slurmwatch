"""Insight for a PENDING Slurm job: why it's waiting, when it will start, and
where in the cluster it could run.

slurmwatch's live telemetry only applies to a RUNNING job — a pending job has no
node, no cgroup, no metrics. But the user's real questions while a job sits in the
queue are "why is it stuck?", "when will it start?", and "would a different
partition run it sooner?". This module answers those from ``scontrol`` (the job's
Reason + the scheduler's estimated StartTime) and ``sinfo`` (cluster-wide free
capacity), reusing slurm.py's robust parsers. It never touches the running-job
path, so live monitoring is completely unaffected (#60).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .exceptions import JobNotFoundError, JobNotPendingError, SlurmCommandError
from .model import short_host
from .slurm import (
    _is_mock,
    _parse_gpu_count,
    _parse_leading_int,
    _parse_mem_to_bytes,
    _parse_scontrol_field,
    _parse_slurm_duration,
    _parse_tres_gpus,
    _run_slurm_cmd,
)


def _scontrol_time(raw: str | None) -> float | None:
    """Parse a ``scontrol`` ``YYYY-MM-DDTHH:MM:SS`` timestamp to epoch seconds.

    Returns ``None`` for the ``Unknown``/``N/A`` sentinels scontrol prints for an
    unset time (e.g. StartTime before the backfill scheduler has placed the job)."""
    if not raw or raw in ("Unknown", "N/A"):
        return None
    try:
        return time.mktime(time.strptime(raw, "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, OSError):
        return None


# Slurm job states that count as "waiting in the queue" for this view. RUNNING /
# COMPLETING etc. are handled by the live dashboard, not here.
_PENDING_STATES = frozenset({"PENDING"})

_RUNNING_QUEUE_STATES = frozenset({"RUNNING", "CONFIGURING", "COMPLETING"})
_PENDING_QUEUE_STATES = frozenset({"PENDING", "SUSPENDED"})


@dataclass
class PendingJob:
    """A queued job's request and the scheduler's view of why/when it will run."""

    job_id: str
    raw_job_id: str
    name: str
    username: str
    partition: str
    qos: str
    account: str
    reason: str
    submit_time: float | None
    start_time_estimate: float | None  # scontrol StartTime (backfill estimate) or None
    priority: int | None
    req_cpus: int
    req_nodes: int
    req_mem_bytes: int
    req_gpus: int
    req_gpu_type: str
    time_limit_seconds: int | None
    # Whether the job asked for whole nodes (`--exclusive`). Then only fully-idle
    # nodes can take it — a partition's partially-used (mixed) nodes don't count.
    exclusive: bool = False


@dataclass
class PartitionResources:
    """A partition's current free capacity, aggregated from ``sinfo``."""

    name: str
    available: bool  # partition AVAIL == "up"
    total_nodes: int = 0
    idle_nodes: int = 0
    mix_nodes: int = 0
    cpus_idle: int = 0
    cpus_total: int = 0
    gpu_types: list[str] = field(default_factory=list)
    timelimit_seconds: int | None = None
    is_current: bool = False
    # Whether the partition has GPUs at all — tracked independently of gpu_types
    # because many clusters report GPUs untyped in `sinfo %G` (e.g. `gpu:4`), so
    # gpu_types can be empty while GPUs are present (#60 review).
    has_gpus: bool = False
    # The largest per-node memory in the partition (bytes), from `sinfo %m`; 0 if
    # unknown. Memory is a per-node scheduling constraint, so a job's per-node
    # request must fit the biggest node.
    max_node_mem_bytes: int = 0

    @property
    def free_nodes(self) -> int:
        """Nodes that could take work now (fully idle + partially free)."""
        return self.idle_nodes + self.mix_nodes


# Plain-English translations for the Slurm Reason codes users hit most. Anything
# not matched exactly falls through to prefix heuristics in :func:`explain_reason`.
_REASON_EXPLANATIONS = {
    "Resources": "Waiting for enough free nodes/CPUs/GPUs to become available.",
    "Priority": "Queued behind higher-priority jobs — it will run once it reaches the front.",
    "Dependency": "Waiting on another job it depends on to finish.",
    "DependencyNeverSatisfied": (
        "A dependency can never be satisfied — this job won't start (consider cancelling it)."
    ),
    "ReqNodeNotAvail": "Requested nodes are unavailable (down, drained, reserved, or powered off).",
    "Reservation": "Waiting for its reservation window to begin.",
    "ReservationDeleted": "Its reservation was deleted — it may never start as requested.",
    "BeginTime": "Held until its scheduled begin time (submitted with --begin).",
    "JobHeldUser": "Held by you — release it with `scontrol release <jobid>`.",
    "JobHeldAdmin": "Held by an administrator — contact support to release it.",
    "PartitionTimeLimit": "Requested time exceeds the partition's limit — lower --time.",
    "PartitionNodeLimit": "Requested node count exceeds the partition's limit.",
    "PartitionDown": "The partition is down.",
    "PartitionInactive": "The partition is inactive.",
    "NodeDown": "A required node is down.",
    "Cleaning": "A previous job is still being cleaned up on the target nodes.",
    "None": "Being scheduled now — no blocking reason reported.",
}


def explain_reason(reason: str) -> str:
    """Translate a Slurm Reason code into a plain-English explanation."""
    r = (reason or "").strip()
    if not r or r in ("None", "(null)", "N/A"):
        return "Being scheduled now — no blocking reason reported."
    if r in _REASON_EXPLANATIONS:
        return _REASON_EXPLANATIONS[r]
    low = r.lower()
    # Many limit reasons are QOS*/Assoc*/Grp* variants; group them sensibly.
    if low.startswith("qos") or "qos" in low:
        return "A QOS limit is capping your usage (running jobs / CPUs / GPUs / time)."
    if low.startswith("assoc") or "account" in low:
        return "An account/association limit is capping your usage."
    if "grp" in low and ("cpu" in low or "gres" in low or "node" in low or "mem" in low):
        return "A group resource limit (CPUs/GPUs/nodes/memory) has been reached."
    if "depend" in low:
        return "Waiting on a job dependency."
    if "reservation" in low or "resv" in low:
        return "Related to a reservation window."
    # Node availability is checked BEFORE the generic "partition" catch-all: the
    # common free-text reason "Nodes required for job are DOWN, DRAINED or reserved
    # for jobs in higher priority partitions" contains the word "partitions" and
    # would otherwise be mislabelled a partition limit (#60 review).
    if (
        "nodenotavail" in low
        or "nodedown" in low
        or "nodefail" in low
        or "drain" in low
        or "down" in low
        or "reserved" in low
    ):
        return "Requested nodes are currently unavailable (down, drained, or reserved)."
    if "partition" in low:
        return "A partition limit or state is blocking it."
    if "prolog" in low or "cleaning" in low:
        return "The target nodes are still being prepared/cleaned."
    return f"Slurm is holding it with reason '{r}'."


def _gpu_type_from_gres(gres: str) -> str:
    """The GPU model from a GRES/TRES value ("" if untyped).

    Accepts BOTH the per-node ``Gres``/``TresPerNode`` colon form (``gpu:a100:2``)
    AND the TRES equals form (``gres/gpu:a100=2``) that a job-level ``--gpus=a100:2``
    request produces — the latter was previously missed, so a typed per-job GPU
    request lost its type and got a wrong "fits" verdict (#60 review)."""
    m = re.search(r"gpu:([a-zA-Z0-9._-]+)[:=]\d+", gres or "")
    if m and m.group(1).lower() not in ("gpu", "mps", "shard"):
        return m.group(1).replace("_", "-")
    return ""


def _select_pending_record(output: str) -> str:
    """Pick the record to describe from ``scontrol show job`` output.

    Arrays/het jobs return several records; prefer a PENDING one (that's what this
    view is about), else fall back to the first record so the caller can report a
    clear "not pending" state rather than crashing.
    """
    records = [r for r in re.split(r"\n\s*\n", output) if "JobId=" in r]
    if not records:
        return output
    for r in records:
        if (_parse_scontrol_field(r, "JobState") or "").upper() in _PENDING_STATES:
            return r
    return records[0]


def resolve_pending_job(job_id: str) -> PendingJob:
    """Resolve a queued job's request + scheduler estimate via ``scontrol``.

    Raises :class:`JobNotFoundError` if the job doesn't exist and
    :class:`JobNotPendingError` if it exists but isn't PENDING (so the caller can
    fall back to the normal running/ended handling).
    """
    if _is_mock():
        return _mock_pending_job(job_id)

    try:
        output = _run_slurm_cmd(["scontrol", "show", "job", job_id])
    except SlurmCommandError as exc:
        raise JobNotFoundError(f"Job {job_id} not found") from exc

    record = _select_pending_record(output)
    state = (_parse_scontrol_field(record, "JobState") or "").upper()
    if state not in _PENDING_STATES:
        raise JobNotPendingError(f"Job {job_id} is in state '{state or 'UNKNOWN'}', not PENDING.")

    username = _parse_scontrol_field(record, "UserId") or ""
    username = username.split("@")[0].split("(")[0] if username else ""

    def _clean(fieldname: str) -> str:
        val = _parse_scontrol_field(record, fieldname) or ""
        return "" if val in ("(null)", "(none)", "N/A", "Unknown") else val

    req_cpus = _parse_leading_int(_parse_scontrol_field(record, "NumCPUs"))
    req_nodes = max(_parse_leading_int(_parse_scontrol_field(record, "NumNodes")), 1)

    req_tres = (
        _parse_scontrol_field(record, "ReqTRES") or _parse_scontrol_field(record, "TRES") or ""
    )

    # Requested memory (whole-job total): prefer the TRES `mem=` token; else scale
    # the per-node / per-cpu minimums to a total — MinMemoryNode is memory PER NODE
    # and MinMemoryCPU is PER CPU, so they must be multiplied by the node / CPU
    # count to be comparable with the TRES total (#60 review).
    req_mem_bytes = 0
    for token in req_tres.split(","):
        token = token.strip()
        if token.startswith("mem="):
            req_mem_bytes = _parse_mem_to_bytes(token.split("=", 1)[1])
            break
    if req_mem_bytes == 0:
        node_mem = _parse_mem_to_bytes(_clean("MinMemoryNode"))
        if node_mem > 0:
            req_mem_bytes = node_mem * req_nodes
        else:
            cpu_mem = _parse_mem_to_bytes(_clean("MinMemoryCPU"))
            if cpu_mem > 0:
                req_mem_bytes = cpu_mem * max(req_cpus, 1)

    # Requested GPUs: the job-wide TRES count, then a per-node Gres/TresPerNode.
    req_gpus = _parse_tres_gpus(req_tres)
    gres_fields = " ".join(_parse_scontrol_field(record, f) or "" for f in ("TresPerNode", "Gres"))
    if req_gpus == 0:
        req_gpus = _parse_gpu_count(gres_fields)
    req_gpu_type = _gpu_type_from_gres(gres_fields) or _gpu_type_from_gres(req_tres)

    priority_raw = _parse_scontrol_field(record, "Priority")
    priority = _parse_leading_int(priority_raw) if priority_raw else None

    # `--exclusive` (whole-node) requests show as OverSubscribe=NO/EXCLUSIVE; then
    # only fully-idle nodes can host the job, so the fit check must not count a
    # partition's partially-used nodes.
    oversub = (_parse_scontrol_field(record, "OverSubscribe") or "").upper()
    exclusive = oversub in ("NO", "EXCLUSIVE")

    time_limit_str = _parse_scontrol_field(record, "TimeLimit") or ""
    time_limit_seconds: int | None = None
    if time_limit_str and time_limit_str.upper() not in ("UNLIMITED", "PARTITION_LIMIT", "N/A"):
        secs = _parse_slurm_duration(time_limit_str)
        if secs > 0:
            time_limit_seconds = int(secs)

    return PendingJob(
        job_id=job_id,
        raw_job_id=_parse_scontrol_field(record, "JobId") or job_id,
        name=_clean("JobName") or _clean("Name"),
        username=username,
        partition=_parse_scontrol_field(record, "Partition") or "unknown",
        qos=_clean("QOS"),
        account=_clean("Account"),
        reason=_parse_scontrol_field(record, "Reason") or "",
        submit_time=_scontrol_time(_parse_scontrol_field(record, "SubmitTime")),
        start_time_estimate=_scontrol_time(_parse_scontrol_field(record, "StartTime")),
        priority=priority,
        req_cpus=req_cpus,
        req_nodes=req_nodes,
        req_mem_bytes=req_mem_bytes,
        req_gpus=req_gpus,
        req_gpu_type=req_gpu_type,
        time_limit_seconds=time_limit_seconds,
        exclusive=exclusive,
    )


def _parse_cpu_state(cpus_field: str) -> tuple[int, int]:
    """(idle, total) CPUs from an ``sinfo %C`` ``allocated/idle/other/total`` value."""
    parts = cpus_field.strip().split("/")
    if len(parts) != 4:
        return 0, 0
    try:
        return int(parts[1]), int(parts[3])
    except ValueError:
        return 0, 0


def _resolve_accessible_partitions(job_account: str) -> set[str] | None:
    """Names of partitions ``job_account`` may submit to, per Slurm's per-partition
    ``AllowAccounts``/``DenyAccounts``.

    Private (per-PI) partitions restrict ``AllowAccounts`` to an account the job
    doesn't hold, so listing them as places to "requeue" is noise and misleading —
    the user can't actually move there. ``None`` when it can't be determined (then
    the caller must not filter, so a parsing gap never hides real options).
    """
    if not job_account:
        return None
    try:
        out = _run_slurm_cmd(["scontrol", "-o", "show", "partition"])
    except Exception:
        return None
    ok: set[str] = set()
    for line in out.splitlines():
        name = _parse_scontrol_field(line, "PartitionName")
        if not name:
            continue
        allow = (_parse_scontrol_field(line, "AllowAccounts") or "ALL").strip()
        deny = (_parse_scontrol_field(line, "DenyAccounts") or "").strip()
        allow_set = {a.strip() for a in allow.split(",") if a.strip()}
        deny_set = {a.strip() for a in deny.split(",") if a.strip()}
        allowed = allow.upper() == "ALL" or job_account in allow_set
        denied = deny.lower() not in ("", "(null)") and job_account in deny_set
        if allowed and not denied:
            ok.add(name)
    return ok or None


def resolve_cluster_partitions(
    current_partition: str = "", job_account: str = ""
) -> list[PartitionResources]:
    """Per-partition free capacity across the cluster, from ``sinfo``.

    One ``PartitionResources`` per partition, aggregating every node-state line:
    idle/mix node counts, idle & total CPUs, GPU types, and the time limit. The
    job's current partition is flagged. When ``job_account`` is given, partitions
    the account can't submit to (private per-PI ones) are dropped — the current
    partition is always kept. Returns ``[]`` if ``sinfo`` is unavailable.
    """
    if _is_mock():
        return _mock_partitions(current_partition)

    try:
        # %m (per-node memory, MB) lets us reject a partition no node of which can
        # hold the job's per-node memory request.
        out = _run_slurm_cmd(["sinfo", "-h", "-o", "%R|%a|%D|%t|%C|%G|%l|%m"])
    except SlurmCommandError:
        return []

    # A job can be submitted to several partitions (`sbatch -p a,b`), so
    # current_partition may be a comma-list — match ANY of them (#audit3-4).
    cur = {short_host(p) for p in current_partition.split(",") if p.strip()}
    parts: dict[str, PartitionResources] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split("|")
        if len(fields) < 6:
            continue
        name = fields[0].strip().rstrip("*")
        avail = fields[1].strip().lower()
        nnodes = _parse_leading_int(fields[2])
        state = fields[3].strip().lower()
        cpus_field = fields[4].strip()
        gres = fields[5].strip()
        timelimit = fields[6].strip() if len(fields) > 6 else ""
        mem_field = fields[7].strip() if len(fields) > 7 else ""

        p = parts.get(name)
        if p is None:
            p = PartitionResources(
                name=name,
                available=(avail == "up"),
                is_current=short_host(name) in cur,
            )
            parts[name] = p
        # A partition line is 'up' if any of its state lines report up.
        p.available = p.available or avail == "up"
        p.total_nodes += nnodes
        # sinfo appends flag chars to the base state (idle*, mix~, alloc#, ...).
        base = re.sub(r"[^a-z]", "", state)
        if base.startswith("idle"):
            p.idle_nodes += nnodes
        elif base.startswith("mix"):
            p.mix_nodes += nnodes
        idle_cpus, total_cpus = _parse_cpu_state(cpus_field)
        p.cpus_idle += idle_cpus
        p.cpus_total += total_cpus
        if gres and gres.lower() not in ("(null)", "null", ""):
            # Any gpu:... entry means the partition has GPUs, even when it's the
            # untyped `gpu:N` form (common on real clusters) that carries no model.
            if re.search(r"(?:^|,)\s*(?:gres/)?gpu[:=]", gres, re.IGNORECASE):
                p.has_gpus = True
            for m in re.finditer(r"gpu:([a-zA-Z0-9._-]+)[:=]\d+", gres):
                gt = m.group(1).replace("_", "-")
                if gt.lower() not in ("gpu", "mps", "shard") and gt not in p.gpu_types:
                    p.gpu_types.append(gt)
        node_mem = _parse_leading_int(mem_field)
        if node_mem > 0:
            p.max_node_mem_bytes = max(p.max_node_mem_bytes, node_mem * 1024**2)
        if p.timelimit_seconds is None and timelimit and timelimit not in ("infinite", "n/a"):
            secs = _parse_slurm_duration(timelimit)
            if secs > 0:
                p.timelimit_seconds = int(secs)

    # Drop partitions the job's account can't use (private per-PI ones), so the
    # WHERE list is only places the user could actually requeue to. Always keep the
    # current partition. If access can't be determined, don't filter (show all).
    accessible = _resolve_accessible_partitions(job_account) if job_account else None
    values = [
        p for p in parts.values() if accessible is None or p.is_current or p.name in accessible
    ]

    # Current partition first, then the ones with the most free capacity.
    return sorted(
        values,
        key=lambda p: (not p.is_current, -(p.idle_nodes + p.mix_nodes), -p.cpus_idle),
    )


def partition_fits_now(job: PendingJob, part: PartitionResources) -> bool:
    """Heuristic: could ``job`` plausibly start in ``part`` right now?

    A coarse "worth trying" signal, not a scheduling guarantee: the partition is
    up, has at least the requested nodes free (idle+mix), enough idle CPUs, a node
    big enough for the per-node memory request, and — if GPUs are requested — has
    GPUs (and offers the requested *type* when both the job and the partition name
    a type). It intentionally can't see QOS/account limits, exact idle-GPU counts,
    or per-node CPU placement, so it's presented to the user as an estimate.
    """
    if not part.available:
        return False
    # A whole-node (`--exclusive`) job can only use fully-idle nodes; a normal job
    # can also land on partially-free (mixed) nodes.
    avail_nodes = part.idle_nodes if job.exclusive else part.free_nodes
    if job.req_nodes > avail_nodes:
        return False
    if job.req_cpus > part.cpus_idle:
        return False
    # Memory is a per-node constraint: the job's per-node share must fit the
    # partition's biggest node. Only reject when we actually know the node size.
    if job.req_mem_bytes > 0 and part.max_node_mem_bytes > 0:
        per_node_req = job.req_mem_bytes / max(job.req_nodes, 1)
        if per_node_req > part.max_node_mem_bytes:
            return False
    if job.req_gpus > 0:
        # The partition must have GPUs at all. gpu_types may be empty even when it
        # does (untyped `gpu:N` clusters), so trust has_gpus OR a parsed type.
        if not (part.has_gpus or part.gpu_types):
            return False
        # Only enforce a type match when BOTH sides name a type; an untyped
        # partition (no model reported) can't be excluded on type.
        if (
            job.req_gpu_type
            and part.gpu_types
            and job.req_gpu_type.lower() not in {g.lower() for g in part.gpu_types}
        ):
            return False
    return True


def resolve_priority_rank(partition: str, priority: int | None) -> tuple[int, int] | None:
    """The job's ``(rank, total_pending)`` among a partition's pending jobs by priority.

    ``rank`` is 1-based, highest-priority first (rank 1 = next in line). Used to
    quantify a ``Priority`` wait ("#312 of 453") so "queued behind higher-priority
    jobs" is a concrete position, not just a phrase. ``None`` when the priority is
    unknown or ``squeue`` can't be read.
    """
    if priority is None:
        return None
    if _is_mock():
        return 4, 5
    try:
        out = _run_slurm_cmd(["squeue", "-h", "-p", partition, "-t", "PD", "-o", "%Q"])
    except Exception:
        return None  # best-effort context; never let a squeue hiccup break the view
    prios: list[int] = []
    for tok in out.split():
        tok = tok.strip()
        if tok.lstrip("-").isdigit():
            prios.append(int(tok))
    if not prios:
        return None
    ahead = sum(1 for p in prios if p > priority)
    return ahead + 1, len(prios)


def resolve_queue_counts(partition: str) -> tuple[int, int]:
    """(running, pending) job counts on ``partition`` for queue-pressure context."""
    if _is_mock():
        return 12, 5
    try:
        out = _run_slurm_cmd(["squeue", "-h", "-p", partition, "-o", "%T"])
    except SlurmCommandError:
        return 0, 0
    running = pending = 0
    for line in out.splitlines():
        state = line.strip().upper()
        if not state:
            continue
        if state in _RUNNING_QUEUE_STATES:
            running += 1
        elif state in _PENDING_QUEUE_STATES:
            pending += 1
    return running, pending


# ---------------------------------------------------------------------------
# Mock data (SLURMWATCH_MOCK) so the pending view is demoable/testable with no
# cluster — mirrors the shape of real scontrol/sinfo output.
# ---------------------------------------------------------------------------


def _mock_pending_job(job_id: str) -> PendingJob:
    # A job queued on a busy partition that would fit elsewhere right now — so the
    # demo shows the payoff: "your partition is full, but gpu-a100 has room now".
    now = time.time()
    return PendingJob(
        job_id=job_id,
        raw_job_id=job_id.split("_")[0] if "_" in job_id else job_id,
        name="train",
        username="demo",
        partition="gpu-shared",
        qos="normal",
        account="rcc-staff",
        reason="Resources",
        submit_time=now - 5400,  # queued 1.5h ago
        start_time_estimate=now + 3600,  # scheduler estimate: ~1h out
        priority=10432,
        req_cpus=16,
        req_nodes=1,
        req_mem_bytes=64 * 1024**3,
        req_gpus=2,
        req_gpu_type="a100",
        time_limit_seconds=24 * 3600,
    )


def _mock_partitions(current_partition: str = "") -> list[PartitionResources]:
    cur = short_host(current_partition) if current_partition else "gpu-shared"
    gib = 1024**3

    def _p(name: str, **kw: object) -> PartitionResources:
        return PartitionResources(name=name, available=True, **kw)  # type: ignore[arg-type]

    raw = [
        _p(
            "cpu-shared",
            total_nodes=100,
            idle_nodes=40,
            mix_nodes=20,
            cpus_idle=1280,
            cpus_total=3200,
            timelimit_seconds=2 * 3600,
            max_node_mem_bytes=192 * gib,
        ),
        _p(
            "gpu-shared",
            total_nodes=10,
            mix_nodes=1,
            cpus_idle=4,
            cpus_total=160,
            gpu_types=["a100", "v100"],
            has_gpus=True,
            timelimit_seconds=4 * 3600,
            max_node_mem_bytes=256 * gib,
        ),
        _p(
            "gpu-a100",
            total_nodes=8,
            idle_nodes=3,
            mix_nodes=1,
            cpus_idle=96,
            cpus_total=256,
            gpu_types=["a100"],
            has_gpus=True,
            timelimit_seconds=12 * 3600,
            max_node_mem_bytes=256 * gib,
        ),
        _p(
            "gpu-highend",
            total_nodes=4,
            mix_nodes=1,
            cpus_idle=8,
            cpus_total=128,
            gpu_types=["h100"],
            has_gpus=True,
            timelimit_seconds=24 * 3600,
            max_node_mem_bytes=512 * gib,
        ),
        _p(
            "debug",
            total_nodes=2,
            idle_nodes=2,
            cpus_idle=16,
            cpus_total=16,
            timelimit_seconds=3600,
            max_node_mem_bytes=32 * gib,
        ),
    ]
    for p in raw:
        p.is_current = short_host(p.name) == cur
    return sorted(
        raw, key=lambda p: (not p.is_current, -(p.idle_nodes + p.mix_nodes), -p.cpus_idle)
    )
