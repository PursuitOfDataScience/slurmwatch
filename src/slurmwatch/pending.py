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

import contextlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import NamedTuple

from .exceptions import JobNotFoundError, JobNotPendingError, SlurmCommandError
from .model import short_host
from .slurm import (
    _is_missing_job_error,
    _is_mock,
    _owner_from_record,
    _parse_gpu_count,
    _parse_leading_int,
    _parse_mem_to_bytes,
    _parse_scontrol_field,
    _parse_slurm_duration,
    _parse_tres_gpus,
    _run_slurm_cmd,
    current_username,
)
from .units import format_bytes

# Cap the WHERE table (both the TUI's PendingView and the plain-text CLI report
# share this) so a pathological (unfiltered) partition list can't flood the
# screen/terminal — but high enough that a normal account's access-filtered set
# shows in full (no silly "... and 1 more"). Current partition + fits always kept.
_MAX_WHERE_ROWS = 24


def _scontrol_time(raw: str | None) -> float | None:
    """Parse a ``scontrol`` ``YYYY-MM-DDTHH:MM:SS`` timestamp to epoch seconds.

    Returns ``None`` for the ``Unknown``/``N/A`` sentinels scontrol prints for an
    unset time (e.g. StartTime before the backfill scheduler has placed the job)."""
    if not raw or raw in ("Unknown", "N/A"):
        return None
    try:
        return time.mktime(time.strptime(raw, "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, OSError, OverflowError):
        return None


# Slurm job states that count as "waiting in the queue" for this view. RUNNING /
# COMPLETING etc. are handled by the live dashboard, not here.
_PENDING_STATES = frozenset({"PENDING"})

# Transient active states (RESIZING/SIGNALING/REQUEUED) count as running too, so a
# mid-transition job isn't dropped from the "N running" context — they previously
# fell into neither bucket and silently under-counted the queue depth (A6).
_RUNNING_QUEUE_STATES = frozenset(
    {"RUNNING", "CONFIGURING", "COMPLETING", "RESIZING", "SIGNALING", "REQUEUED"}
)
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
    # The largest per-node CPU count in the partition (`sinfo %c`); 0 if unknown.
    # Like memory, a job's per-node CPU share must fit one node — the cluster-wide
    # idle-core sum alone would falsely pass a job needing more cores than any node.
    max_node_cpus: int = 0
    # The largest per-node GPU count in the partition (summed across a node's
    # `sinfo %G` gpu entries); 0 if unknown. A job's per-node GPU request must fit
    # one node — without this a multi-GPU job "fits now" on GPU-poor nodes (M4).
    max_node_gpus: int = 0
    # Capacity restricted to FULLY-IDLE nodes, for exclusive / GPU jobs that need a
    # whole idle node (they can't use the free cores on a busy mix node). The
    # general fields above span mix nodes too, so for such a job they would falsely
    # pass it against a big node that isn't actually idle (M5).
    idle_node_cpus: int = 0
    max_idle_node_cpus: int = 0
    max_idle_node_mem_bytes: int = 0
    # The biggest node the partition CONTAINS, counted from every node line `sinfo`
    # reports for it whatever state that node is in. The two `max_node_*` fields above
    # deliberately span schedulable (idle/mix) nodes only, because they also stand in
    # for "is there a free node big enough" — but that makes them a reading of this
    # minute's occupancy, and `fit_blocker` was using them to decide "node too small",
    # a claim about HARDWARE that `blocker_is_permanent` then reports as forever.
    # Measured live: `cobey-hm` owns a 64-CPU / 2,063,994 MB node that was ALLOC, so
    # its schedulable maxima read 48 CPU / 768,000 MB and a `--cpus-per-task=64` job
    # was told "node too small" — permanently unrunnable — against a node that exists
    # and is merely busy (same for `lgagliardi-ld` and `andrewferguson-gpu`). These
    # fields keep the label honest: too small for the hardware is permanent, too big
    # for what is free right now is "no room", which clears by itself. 0 if unknown.
    max_config_node_cpus: int = 0
    max_config_node_mem_bytes: int = 0
    # Free GPUs on each schedulable node, from `sinfo -N -O Gres,GresUsed`. Empty
    # when that query is unavailable, which is what `gpu_detail` distinguishes:
    # an empty list means "unknown", not "no free GPUs".
    #
    # Before this existed, a GPU job was measured against fully-IDLE nodes only,
    # because the aggregate `sinfo %G` reports *configured* GRES with no way to
    # tell what is in use. That is safe but wrong in the direction that matters:
    # on a cluster whose GPU partition was idle=0/mix=11, it reported zero
    # available nodes for every GPU job while 95 GPUs sat free cluster-wide, all
    # of them on mix nodes. `GresUsed` is a per-node field and gives the real
    # figure, so mix nodes can now be counted for exactly the GPUs they have left.
    free_gpus_per_node: list[int] = field(default_factory=list)
    gpu_detail: bool = False
    # Allocatable MEMORY on each schedulable node (MB), from the same per-node query:
    # `Memory - AllocMem`. Empty when that query is unavailable, which `mem_detail`
    # distinguishes exactly as `gpu_detail` does for GPUs.
    #
    # `sinfo`'s aggregate `%m` is a node's CONFIGURED memory, so the WHERE table
    # measured a job's per-node request against hardware already carrying someone
    # else's job -- while the CPU half of the same row had used idle `%C` all along.
    # Measured live on `vitelli-amd` (2 nodes, neither idle, largest mix node holding
    # 6,160 MB free of 250,000 MB configured): a 240 GiB/node request came back with a
    # BLANK blocker, i.e. "plausibly fits", carrying the copy-pasteable requeue command
    # that goes with it -- which would have made the job pend for ever and lose the
    # priority it had accrued. D7.
    free_node_mem_mb: list[int] = field(default_factory=list)
    mem_detail: bool = False
    # Whether this list was filtered by what the user may actually SUBMIT to, not
    # just by capacity. False when the association list couldn't be read, and then
    # the WHERE table must claim only "has room", never "can run now" — a partition
    # can have idle nodes and still reject the job (SW-2).
    assoc_verified: bool = True

    @property
    def free_nodes(self) -> int:
        """Nodes that could take work now (fully idle + partially free)."""
        return self.idle_nodes + self.mix_nodes

    @property
    def gpus_free(self) -> int:
        """Allocatable GPUs across the partition; 0 when unknown (see gpu_detail)."""
        return sum(self.free_gpus_per_node)

    @property
    def max_node_gpus_free(self) -> int:
        """Most GPUs free on any one node; 0 when unknown (see gpu_detail)."""
        return max(self.free_gpus_per_node, default=0)

    @property
    def max_free_node_mem_bytes(self) -> int:
        """Most memory free on any one node (bytes); 0 when unknown (see mem_detail)."""
        return max(self.free_node_mem_mb, default=0) * 1024**2

    def nodes_with_free_gpus(self, per_node: int) -> int:
        """Schedulable nodes with at least ``per_node`` GPUs free."""
        if per_node <= 0:
            return len(self.free_gpus_per_node)
        return sum(1 for free in self.free_gpus_per_node if free >= per_node)


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
    # Measured on the live queue: these four are all present here and every one of
    # them fell through to a heuristic that said something false or unhelpful.
    "JobArrayTaskLimit": (
        "The array is at its concurrent-task limit (the %N in --array) — "
        "earlier tasks must finish before this one starts."
    ),
    "BadConstraints": (
        "No node satisfies the requested --constraint/features as submitted — "
        "resubmit with a constraint this cluster can meet."
    ),
    "InvalidAccount": (
        "The account isn't valid here — resubmit with a valid -A/--account "
        "(this is not a usage limit; waiting won't clear it)."
    ),
    "InvalidQOS": ("The QOS isn't valid for this account/partition — resubmit with a valid --qos."),
    # Free text, spaces and all, as Slurm 25.11 reports it: the launch failed, Slurm
    # requeued the job and then HELD it, so it will sit there until released. Three
    # live jobs on the second cluster were getting the generic "Slurm is holding it
    # with reason '...'", which does not say that a release is what unblocks it.
    "launch failed requeued held": (
        "Its launch failed, so Slurm requeued and HELD it — it won't start until "
        "released (`scontrol release <jobid>`); check the node/prolog for why the "
        "launch failed first."
    ),
}


def _asciify(text: str) -> str:
    """Fold the few decorative Unicode glyphs to ASCII (for --ascii terminals)."""
    return (
        text.replace("—", "-")
        .replace("–", "-")
        .replace("·", "-")
        .replace("…", "...")
        .replace("→", "->")
        .replace("▸", ">")
    )


def explain_reason(reason: str, ascii_mode: bool = False, job_id: str = "") -> str:
    """Translate a Slurm Reason code into a plain-English explanation."""
    msg = _explain_reason(reason)
    # Paste-ready when the id is known. The table is keyed by REASON, so it can only
    # carry a `<jobid>` placeholder — but every caller has the job in hand, and a
    # command that needs editing before it runs is the weaker half of the class SW-32
    # opened: `scontrol release <jobid>` verbatim answers "too few arguments".
    if job_id:
        msg = msg.replace("<jobid>", job_id)
    return _asciify(msg) if ascii_mode else msg


#: Slurm's several spellings of "nothing is blocking this job".
#:
#: Answered before `_REASON_EXPLANATIONS` is consulted, which is why `"None"` is
#: NOT a key there: it was, with this same sentence as its value, and the entry
#: was unreachable -- the early return below already catches it, so the dict copy
#: could have drifted to say anything without a reader ever seeing it.
_NO_BLOCKING_REASON = "Being scheduled now — no blocking reason reported."
#: `""` is included via the falsy test; these are the literal strings Slurm emits.
_NOT_A_REASON = ("None", "(null)", "N/A")


def _explain_reason(reason: str) -> str:
    r = (reason or "").strip()
    if not r or r in _NOT_A_REASON:
        return _NO_BLOCKING_REASON
    if r in _REASON_EXPLANATIONS:
        return _REASON_EXPLANATIONS[r]
    low = r.lower()
    # PER-JOB before per-user. Slurm's own naming carries the distinction and the
    # advice inverts on it: `QOSMaxWallDurationPerJobLimit` means THIS REQUEST is too
    # big for the limit — permanent until resubmitted — while
    # `QOSMaxCpuPerUserLimit` means your other running jobs are using the allowance
    # and waiting genuinely helps. Both matched the same "qos" heuristic, so a job
    # whose --time exceeded its QOS was told "a QOS limit is capping your usage",
    # i.e. wait for your own jobs to finish, which can never work. Measured on the
    # live queue: 14 jobs sitting on QOSMaxWallDurationPerJobLimit right now. The
    # partition twin of this, PartitionTimeLimit, has always said "lower --time".
    if "perjob" in low:
        return (
            "The request exceeds a per-JOB limit of this QOS/account — lower the "
            "request (--time / --cpus / --nodes); waiting won't help."
        )
    # Many limit reasons are QOS*/Assoc*/Grp* variants; group them sensibly.
    if low.startswith("qos") or "qos" in low:
        return "A QOS limit is capping your usage (jobs / CPUs / GPUs / memory / time / billing)."
    if low.startswith("assoc") or "account" in low:
        return "An account/association limit is capping your usage."
    # A Max*/Grp* limit scoped to a user or group, with neither prefix in its name
    # (Slurm 25.11's `MaxBillingPerUser`, `MaxCpuPerUser`, ...).
    if _is_scoped_limit(low):
        return (
            "A per-user/group limit is capping your usage — it frees up as your other jobs finish."
        )
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
        # Only an explicit "invalid job id" means the job is gone. Any other error
        # (timeout, socket, controller unreachable) is transient — re-raise it so
        # the pending view keeps its last state instead of falsely announcing the
        # job "started" and freezing (PendingScreen._refresh swallows it and keeps
        # the view; the periodic refresh recovers when the controller does).
        if _is_missing_job_error(exc):
            raise JobNotFoundError(f"Job {job_id} not found") from exc
        raise

    record = _select_pending_record(output)
    state = (_parse_scontrol_field(record, "JobState") or "").upper()
    if state not in _PENDING_STATES:
        raise JobNotPendingError(f"Job {job_id} is in state '{state or 'UNKNOWN'}', not PENDING.")

    # Name from the uid Slurm printed, so a node that can't resolve the uid doesn't
    # label the user's own pending job "nobody" (SW-1) — and read it defensively,
    # since a newline in a job name can forge a UserId line of its own.
    username, _uid = _owner_from_record(record)

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
            req_mem_bytes = _parse_mem_to_bytes(token.split("=", 1)[1]) or 0
            break
    if req_mem_bytes == 0:
        node_mem = _parse_mem_to_bytes(_clean("MinMemoryNode")) or 0
        if node_mem > 0:
            req_mem_bytes = node_mem * req_nodes
        else:
            cpu_mem = _parse_mem_to_bytes(_clean("MinMemoryCPU")) or 0
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


def _user_groups(username: str) -> set[str] | None:
    """The Unix group names ``username`` belongs to (primary + supplementary).

    ``None`` if they can't be resolved, so a group-restricted partition can't be
    judged and the caller stays conservative. Unix-only (grp/pwd); this is a Slurm
    tool, so that's a given.
    """
    if not username:
        return None
    try:
        import grp
        import pwd

        pw = pwd.getpwnam(username)
        names: set[str] = set()
        for gid in {pw.pw_gid, *os.getgrouplist(username, pw.pw_gid)}:
            with contextlib.suppress(KeyError, OSError):
                names.add(grp.getgrgid(gid).gr_name)
        return names or None
    except Exception:
        return None


def _csv_set(value: str | None) -> set[str]:
    """A Slurm comma-list field as a set of trimmed tokens ('', '(null)' -> empty)."""
    val = (value or "").strip()
    if val.lower() in ("", "(null)"):
        return set()
    return {t.strip() for t in val.split(",") if t.strip()}


def _resolve_accessible_partitions(job_account: str, username: str = "") -> set[str] | None:
    """Names of partitions this job may actually submit to, per each partition's
    ``AllowAccounts``/``DenyAccounts`` (the job's account) and ``AllowGroups``/
    ``DenyGroups`` (the owner's Unix groups) — the two access gates a user can't
    change by editing the job.

    Private (per-PI) partitions restrict one of these, so listing them as places to
    "requeue" is misleading — the user can't move there. QOS gating is intentionally
    NOT applied: a partition change can carry a QOS change, so QOS isn't a hard
    barrier. ``None`` when it can't be determined at all (then the caller must not
    filter, so a parsing gap never hides real options). A partition whose group
    restriction can't be evaluated (owner groups unknown) is excluded — better to
    omit than to recommend a requeue that Slurm will reject.
    """
    groups = _user_groups(username)
    # An unknown account does not make the GROUP gate unknowable — the two are
    # independent, and a cluster running without slurmdbd accounting has no Account on
    # its jobs at all. Bailing out here listed every group-restricted partition on such
    # a site as somewhere to requeue, which is the same "recommend a move Slurm will
    # reject" this function exists to prevent; the account dimension simply goes
    # unfiltered (never hiding a real option), while the one we CAN evaluate is applied.
    if not job_account and groups is None:
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
        # Account gate — skipped entirely when the account is unknown, so an
        # account-restricted partition stays listed rather than being hidden on a guess.
        if job_account:
            allow_acct = (_parse_scontrol_field(line, "AllowAccounts") or "ALL").strip()
            if allow_acct.upper() != "ALL" and job_account not in _csv_set(allow_acct):
                continue
            if job_account in _csv_set(_parse_scontrol_field(line, "DenyAccounts")):
                continue
        # Group gate (against the owner's Unix groups).
        allow_grp = (_parse_scontrol_field(line, "AllowGroups") or "ALL").strip()
        if allow_grp.upper() != "ALL":
            grp_set = _csv_set(allow_grp)
            if groups is None or not (groups & grp_set):
                continue
        deny_grp = _csv_set(_parse_scontrol_field(line, "DenyGroups"))
        if deny_grp and (groups is None or (groups & deny_grp)):
            # Can't verify a group-denied partition (owner groups unknown) → exclude
            # it rather than recommend a requeue Slurm may reject.
            continue
        ok.add(name)
    # Return the set as-is, even when empty: an empty set means the job reaches no
    # partition other than its current one (which the caller always keeps), so it
    # shows only that. `ok or None` used to collapse empty -> None, which the caller
    # reads as "couldn't determine -> show all" and leaks private per-PI partitions.
    # The genuine can't-determine paths return None above.
    return ok


class _Associations(NamedTuple):
    """Which partitions a (user, account) pair is associated with in Slurm."""

    partitions: frozenset[str]
    # An association row with an EMPTY Partition field grants the account every
    # partition, so nothing should be filtered out on its behalf.
    unrestricted: bool


def _resolve_associated_partitions(username: str, job_account: str) -> _Associations | None:
    """Partitions the (user, account) pair actually holds a Slurm ASSOCIATION for.

    The gate a private partition hides behind is usually NOT its own
    ``AllowAccounts``/``AllowGroups`` — on a real 28-partition cluster those all
    read ``ALL`` — but the association list ``sacctmgr`` keeps. Submitting without
    one fails with "Invalid account or account/partition combination specified", so
    a WHERE table built from capacity alone marked private per-PI partitions "YES ▸
    can run now" and ``sbatch --test-only`` rejected them. SW-2.

    ``None`` when the answer cannot be determined — no ``sacctmgr``, a query the
    site denies, no rows at all, or no row for the job's own account (which means
    our reading is off, not that the user may go nowhere). The caller must then
    neither filter nor claim a partition can run the job.
    """
    if not username or not job_account:
        return None
    try:
        out = _run_slurm_cmd(
            ["sacctmgr", "-nP", "show", "assoc", f"user={username}", "format=Account,Partition"]
        )
    except Exception:
        return None
    names: set[str] = set()
    unrestricted = False
    saw_rows = False
    want = job_account.strip().lower()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split("|")
        saw_rows = True
        if fields[0].strip().lower() != want:
            continue
        partition = fields[1].strip() if len(fields) > 1 else ""
        if partition:
            names.add(partition)
        else:
            unrestricted = True
    if not saw_rows or not (names or unrestricted):
        return None
    return _Associations(frozenset(names), unrestricted)


# A node's GRES string: sum every `gpu[:type]:N`, ignoring the `(IDX:0-3)` suffix
# `GresUsed` appends and skipping mps/shard. Shared with the aggregate parser so
# the configured and in-use sides are counted the same way.
_GPU_COUNT_RE = re.compile(r"(?:^|,)\s*(?:gres/)?gpu(?::[a-zA-Z0-9._-]+)?[:=](\d+)", re.IGNORECASE)


def _sum_gres_gpus(gres: str) -> int:
    """Total GPUs in a `sinfo` Gres / GresUsed value; 0 when it names none."""
    if not gres or gres.strip().lower() in ("(null)", "null", "n/a", ""):
        return 0
    return sum(int(n) for n in _GPU_COUNT_RE.findall(gres))


def _fetch_free_gpus_by_partition() -> tuple[
    dict[str, list[int]], bool, dict[str, list[int]], bool
]:
    """Per-partition list of free GPUs on each schedulable node, and whether the
    query itself worked.

    ``sinfo``'s aggregate ``%G`` reports *configured* GRES only, which is why the
    rest of this module could not tell a mix node's free GPUs from its total and
    fell back to counting fully-idle nodes. The node-centric ``GresUsed`` field
    (long-form ``-O`` only — there is no ``%`` short code) closes that gap:
    ``Gres - GresUsed`` per node is the allocatable figure.

    A node appears once per partition it belongs to, which is what we want — the
    same node contributes its free GPUs to each partition that can schedule it.

    The second element separates "the query failed / is unsupported" from "the query
    worked and this partition has no schedulable GPU node". Only the FORMER is
    unknown; conflating them let a partition whose GPU nodes were all allocated look
    like missing data, and the caller then fell back to its idle GPU-less node count
    (see ``resolve_cluster_partitions``).
    """
    if _is_mock():
        return {}, False, {}, False
    try:
        out = _run_slurm_cmd(
            [
                "sinfo",
                "-a",
                "-h",
                "-N",
                # Each field carries an explicit "|" suffix. sinfo's -O output is
                # fixed-WIDTH, not delimited: it truncates a value to the width and pads
                # only UP TO it, so a value within one char of the width leaves 0-1
                # spaces and merges with the next field. Splitting on runs of whitespace
                # then silently yielded 3 fields instead of 4 and GresUsed read as "",
                # i.e. "every GPU on this node is free" for a node whose GPUs were all
                # allocated. A printable separator makes the boundaries unambiguous
                # regardless of value length.
                # `Memory` and `AllocMem` ride along on this query rather than
                # earning a second one: it already visits every node line. Their
                # difference is the allocatable memory the aggregate `%m` cannot
                # express. `MemSpecLimit` is deliberately absent -- measured on this
                # controller (Slurm 20.11.8) it is not a valid `-O` field at all:
                # `sinfo: error: Invalid job format specification: MemSpecLimit` on
                # stderr, rc=0, and the column renders EMPTY, which is precisely the
                # unsupported-field failure the GresUsed comment below describes.
                "-O",
                "Partition:40|,StateLong:20|,Gres:60|,GresUsed:60|,Memory:20|,AllocMem:20|",
            ]
        )
    except SlurmCommandError:
        return {}, False, {}, False

    free: dict[str, list[int]] = {}
    free_mem: dict[str, list[int]] = {}
    read_an_alloc_mem = False
    # Did we see a schedulable GPU node at all, and did ANY of them yield a GresUsed
    # value? sinfo does not fail on an `-O` field it does not know: it warns on stderr,
    # exits 0, and renders that column EMPTY (measured on Slurm 20.11.8 —
    # `-O ...,NoSuchField:60|` printed 1,246 rows, rc=0). Every GPU node is then
    # skipped below, the dict comes back empty, and the caller reads that as a KNOWN
    # "no free GPUs" for every GPU partition on the cluster — 20 of them here, while
    # ~226 GPUs were actually free. So a total absence of readings is the query being
    # unsupported, not a fact about occupancy.
    saw_gpu_node = False
    read_a_gres_used = False
    for line in out.splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split("|")]
        # 4 real fields plus the trailing separator's empty tail. Fewer means the line
        # is malformed, and guessing at a missing GresUsed is what caused the over-report.
        if len(fields) < 4:
            continue
        name = fields[0].rstrip("*")
        state = fields[1].lower()
        gres_total, gres_used = fields[2], fields[3]
        # The four GPU fields stay the requirement; memory is read only when the
        # line carries it. Requiring six SKIPPED a shorter line outright and took
        # the GPU reads down with it -- caught by seven existing tests whose
        # fixtures are four-field node lines, and whose expectations were the
        # correct ones. A real unsupported `-O` field still renders its separators
        # (`...|250000||`), so this controller emits six either way.
        mem_total, mem_alloc = (fields[4], fields[5]) if len(fields) >= 6 else ("", "")
        # Same schedulability rule as the aggregate pass: only idle/mix nodes can
        # take work, and the flag suffixes mark nodes that will not.
        if any(flag in state for flag in ("*", "$", "%", "@", "!")):
            continue
        base = re.sub(r"[^a-z]", "", state)
        if not base.startswith(("idle", "mix")):
            continue
        # Memory FIRST: the GPU reads below `continue` past every node without GPUs,
        # and a node without GPUs still has memory. Recording it after them collected
        # free memory for GPU nodes only -- which on this cluster is 464 of 1,248 node
        # lines, and none at all on the CPU partitions where the over-report was found.
        if mem_total.isdigit() and mem_alloc.isdigit():
            read_an_alloc_mem = True
            free_mem.setdefault(name, []).append(max(0, int(mem_total) - int(mem_alloc)))
        total = _sum_gres_gpus(gres_total)
        if total <= 0:
            continue
        saw_gpu_node = True
        # This node HAS GPUs, so an empty GresUsed is a read we did not get, not a
        # genuine zero: skip the node rather than donating its whole GPU count to the
        # partition's free pool (which advises a requeue onto capacity that cannot run).
        if not gres_used:
            continue
        read_a_gres_used = True
        used = _sum_gres_gpus(gres_used)
        free.setdefault(name, []).append(max(0, min(total, total - used)))
    # An unsupported `-O` field renders EMPTY rather than failing (see above), so a
    # total absence of readings is the field being unavailable -- not a cluster whose
    # every node is full. `isdigit` above is what separates the two: `AllocMem` renders
    # `0` for an empty node and `` for a field this Slurm does not know.
    mem_detail = read_an_alloc_mem
    if saw_gpu_node and not read_a_gres_used:
        # Not one GPU node's GresUsed was readable: the field itself is unavailable, so
        # report UNKNOWN and let the caller keep its conservative idle-node fallback.
        # A single unreadable node beside readable ones is still just that one node.
        return {}, False, free_mem, mem_detail
    return free, True, free_mem, mem_detail


def resolve_cluster_partitions(
    current_partition: str = "", job_account: str = "", job_username: str = ""
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
        # %m (per-node memory, MB) and %c (per-node CPUs) let us reject a partition
        # no single node of which can hold the job's per-node request.
        # -a: include hidden partitions (a job can be pending in one); the
        # account/group filter below still prunes ones the user can't use.
        # -e: list each distinct node configuration on its own line instead of
        # collapsing heterogeneous nodes to one "+"-suffixed representative value,
        # so the per-node max %m/%c is real, not one node's figure (F1/F4).
        out = _run_slurm_cmd(["sinfo", "-a", "-e", "-h", "-o", "%R|%a|%D|%t|%C|%G|%l|%m|%c"])
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
        cpus_per_node_field = fields[8].strip() if len(fields) > 8 else ""

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
        idle_cpus, total_cpus = _parse_cpu_state(cpus_field)
        p.cpus_total += total_cpus
        # The partition's wall-clock ceiling is CONFIGURATION — it does not depend on
        # what state its nodes are in — so read it before the schedulability filter
        # below. Read after it, a partition every one of whose node lines is flagged
        # (live here: `climate`, 48 nodes, and `climate-build`, 2, are entirely
        # `drain*`) never learned its limit, and a job asking for more time than the
        # partition allows got the TRANSIENT "no room" instead of the PERMANENT "time
        # limit" — a verdict decided by unrelated node health, which is the SW-28
        # argument, and the transient answer invites a wait that can never end.
        if p.timelimit_seconds is None and timelimit and timelimit not in ("infinite", "n/a"):
            secs = _parse_slurm_duration(timelimit)
            if secs > 0:
                p.timelimit_seconds = int(secs)
        # The biggest node the partition CONTAINS is configuration too, so like the
        # wall-clock ceiling above it is read before the schedulability filter — a
        # node does not stop being 64 cores wide because someone else is using it.
        # This only ever softens a verdict: `fit_blocker` consults it to tell "no node
        # here is that big" (permanent) apart from "no node that big is free right
        # now" (transient), and the request must exceed BOTH to keep the permanent
        # label, so nothing that used to fit can start failing.
        cfg_mem = _parse_leading_int(mem_field)
        cfg_cpus = _parse_leading_int(cpus_per_node_field)
        if cfg_mem > 0:
            p.max_config_node_mem_bytes = max(p.max_config_node_mem_bytes, cfg_mem * 1024**2)
        if cfg_cpus > 0:
            p.max_config_node_cpus = max(p.max_config_node_cpus, cfg_cpus)
        # sinfo appends flag chars to the base state (idle*, mix~, idle$, ...). Some
        # mark nodes that won't take a normal job, so their "idle" cores must not
        # count as free capacity — otherwise the partition reads "FITS NOW" and gets
        # recommended for requeue when nothing can actually be placed: * not
        # responding, $ in a maintenance reservation, % powering down, @ pending
        # reboot, ! pending power-down. ~ (power-save) and # (powering-up) are kept —
        # they will run.
        if any(flag in state for flag in ("*", "$", "%", "@", "!")):
            continue
        base = re.sub(r"[^a-z]", "", state)
        # Free cores and biggest-node sizes count only on schedulable (idle/mix)
        # nodes: a RESERVED (resv) or PLANNED (plnd, backfill-held) node reports its
        # cores as idle in %C but can't take a normal job, so including it produced a
        # false "FITS NOW" on the plain-CPU path (F2).
        schedulable = base.startswith(("idle", "mix"))
        if base.startswith("idle"):
            p.idle_nodes += nnodes
        elif base.startswith("mix"):
            p.mix_nodes += nnodes
        if schedulable:
            p.cpus_idle += idle_cpus
        if gres and gres.lower() not in ("(null)", "null", ""):
            # Any gpu:... entry means the partition has GPUs, even when it's the
            # untyped `gpu:N` form (common on real clusters) that carries no model.
            if re.search(r"(?:^|,)\s*(?:gres/)?gpu[:=]", gres, re.IGNORECASE):
                p.has_gpus = True
            for m in re.finditer(r"gpu:([a-zA-Z0-9._-]+)[:=]\d+", gres):
                gt = m.group(1).replace("_", "-")
                if gt.lower() not in ("gpu", "mps", "shard") and gt not in p.gpu_types:
                    p.gpu_types.append(gt)
            # Per-node GPU count = the sum of this node's gpu counts (typed and
            # untyped) so `--gpus-per-node=N` can be checked against a real node size
            # (M4). Matches gpu:4, gpu:a100:4, gres/gpu:2, ...; skips mps/shard.
            line_gpus = sum(
                int(n)
                for n in re.findall(
                    r"(?:^|,)\s*(?:gres/)?gpu(?::[a-zA-Z0-9._-]+)?[:=](\d+)", gres, re.IGNORECASE
                )
            )
            if line_gpus > 0:
                p.max_node_gpus = max(p.max_node_gpus, line_gpus)
        node_mem = _parse_leading_int(mem_field)
        node_cpus = _parse_leading_int(cpus_per_node_field)
        if schedulable and node_mem > 0:
            p.max_node_mem_bytes = max(p.max_node_mem_bytes, node_mem * 1024**2)
        if schedulable and node_cpus > 0:
            p.max_node_cpus = max(p.max_node_cpus, node_cpus)
        # Idle-only capacity: an exclusive/GPU job needs a WHOLE idle node, so track
        # the fully-idle nodes' cores/mem apart from the mix-inclusive totals above,
        # which would otherwise pass such a job against a busy big node (M5).
        if base.startswith("idle"):
            p.idle_node_cpus += idle_cpus
            if node_cpus > 0:
                p.max_idle_node_cpus = max(p.max_idle_node_cpus, node_cpus)
            if node_mem > 0:
                p.max_idle_node_mem_bytes = max(p.max_idle_node_mem_bytes, node_mem * 1024**2)

    # Real free-GPU counts, which the aggregate query above cannot supply.
    free_by_partition, gpu_query_ok, free_mem_by_partition, mem_query_ok = (
        _fetch_free_gpus_by_partition()
    )
    for name, p in parts.items():
        if mem_query_ok:
            # Same shape as the GPU block below: a partition the query VISITED and
            # found no schedulable node in gets an empty list, which is a known "no
            # free memory" rather than missing data. The idle/mix node-count test
            # upstream already answers "no room" there, so this only has to avoid
            # claiming a figure it does not have.
            p.free_node_mem_mb = free_mem_by_partition.get(name, [])
            p.mem_detail = True
        free_list = free_by_partition.get(name)
        if free_list is not None:
            p.free_gpus_per_node = free_list
            p.gpu_detail = True
        elif gpu_query_ok and p.has_gpus:
            # The query SUCCEEDED and named no schedulable GPU node in this partition —
            # so "no free GPUs" is a known fact, not missing data. Marking it unknown
            # instead made a GPU job fall back to counting the partition's idle
            # GPU-LESS nodes, which reported "FITS NOW" plus a copy-pasteable requeue
            # command for a partition with zero free GPUs (revived the very over-report
            # the per-node free-GPU pass was added to fix).
            p.free_gpus_per_node = []
            p.gpu_detail = True

    # Drop partitions the job's account can't use (private per-PI ones), so the
    # WHERE list is only places the user could actually requeue to. Always keep the
    # current partition. If access can't be determined, don't filter (show all).
    # Called even with no account: the group gate inside does not need one, and a
    # cluster without accounting has no Account on its jobs at all. The helper returns
    # None when nothing is knowable, which is what "do not filter" is spelled as.
    accessible = _resolve_accessible_partitions(job_account, job_username)
    # ...and by the association list, which is the gate that actually rejects a
    # submission on most sites (SW-2). Every row records whether that check was
    # possible, so the table can hedge its verdict when it wasn't.
    assoc = _resolve_associated_partitions(job_username, job_account) if job_account else None
    values: list[PartitionResources] = []
    for p in parts.values():
        p.assoc_verified = assoc is not None
        if p.is_current:
            values.append(p)  # where the job already is, always shown
            continue
        if accessible is not None and p.name not in accessible:
            continue
        if assoc is not None and not assoc.unrestricted and p.name not in assoc.partitions:
            continue
        values.append(p)

    # Current partition first, then the ones with the most free capacity.
    return sorted(
        values,
        key=lambda p: (not p.is_current, -(p.idle_nodes + p.mix_nodes), -p.cpus_idle),
    )


# Pending reasons where moving to another partition can't make the job start —
# it isn't waiting on capacity, so a "requeue here" tip would be wrong. "assoc"
# catches account/association limits (AssocGrp*/AssocMax*), which are account-
# scoped and partition-independent. (QOS limits are deliberately NOT here: a
# partition change can carry a QOS change, so requeuing can help.)
# Usage caps: the job is priority-ordered and capacity is available, but an
# association or QOS limit on CPUs/nodes/jobs is withholding it. `assoc` was in the
# non-capacity list and `qos` was not, so two structurally identical families were
# classified oppositely — `AssocMaxCpuPerJobLimit` got "a partition change won't help"
# while `QOSMaxNodePerUserLimit` got "broadwl has room, requeue there", which
# misdiagnoses a cap as a shortage. Both are handled as caps now, and the tip says what
# is actually true without guessing the site's configuration (SW-29, follow-up).
_USAGE_CAP_REASONS = ("assoc", "qos")

# A limit scoped to an ACCOUNT / USER / GROUP is a usage cap even when its reason name
# carries neither the `Assoc` nor the `QOS` prefix. Slurm 25.11 on a second cluster
# reports `MaxBillingPerAccount` for 21 live jobs: the explainer already called it
# "an account limit is capping your usage" (it matches the "account" branch) while
# `is_usage_capped` said False, so the same screen offered "requeue to a partition with
# room" for a job that is not short of room at all — the two halves disagreeing about
# the same reason. `MaxCpuPerUser`, `MaxNodePerAccount`, `MaxJobsPerAccount` and the
# rest of that family have the same shape, so match the SCOPE rather than the prefix.
_SCOPED_LIMIT_SUFFIXES = ("peraccount", "peruser", "pergroup")

# Reasons that say the job is INVALID as submitted, not waiting for anything. Room is
# not what they lack and no limit is withholding them, so both the "your usage is
# capped" tip and the "partition X has room" table are the wrong remedy — the answer is
# always to resubmit with different arguments. Kept explicit because the substring
# heuristics get them wrong in opposite directions: `InvalidQOS` contains "qos" and was
# called a usage cap, while `InvalidAccount` and `BadConstraints` matched nothing and
# were offered a partition with room. All three are live on this cluster.
_INVALID_REQUEST_REASONS = ("invalidaccount", "invalidqos", "badconstraints")

_NON_CAPACITY_REASONS = ("dependency", "held", "begintime", "reservation", *_USAGE_CAP_REASONS)


def _is_scoped_limit(reason_lower: str) -> bool:
    """A Max*/Grp* limit scoped to an account, user or group (not to one job)."""
    return any(suffix in reason_lower for suffix in _SCOPED_LIMIT_SUFFIXES)


# Blocked (not capacity/priority) waits: the job isn't being priority-scheduled,
# so a queue position and a "calculating" start estimate are meaningless. (Assoc
# limits are NOT here — such a job is still priority-ordered, just usage-capped.)
_HELD_LIKE_REASONS = ("dependency", "held", "begintime", "reservation")


def is_held_like(reason: str) -> bool:
    """True when the job is blocked (held / dependency / begin-time / reservation)
    rather than queued for capacity/priority."""
    r = (reason or "").strip().lower()
    return any(tok in r for tok in _HELD_LIKE_REASONS)


def is_usage_capped(reason: str) -> bool:
    """True when an association/QOS usage limit is withholding the job.

    Distinct from held-like: such a job IS priority-ordered, so the queue position and
    the start estimate stay meaningful — but "partition X has room" is the wrong
    remedy, because room is not what it lacks.

    A per-JOB limit is NOT a usage cap, even though its reason string also starts with
    QOS/Assoc: nothing about the user's current usage is withholding the job, the
    request itself is too large, and the tip this drives ("your usage is capped —
    other jobs must finish first") describes an event that would change nothing.
    Slurm names them apart, so key on that rather than on the QOS prefix.
    """
    r = (reason or "").strip().lower()
    if "perjob" in r or any(tok in r for tok in _INVALID_REQUEST_REASONS):
        return False
    return any(tok in r for tok in _USAGE_CAP_REASONS) or _is_scoped_limit(r)


def request_must_change(reason: str) -> bool:
    """True when the job's own REQUEST, not the queue, is what stops it — so Slurm
    will never hand it a start time as submitted.

    Two families, both live on this controller: a per-JOB limit
    (``QOSMaxWallDurationPerJobLimit`` and friends — the request exceeds the ceiling)
    and an invalid request (``InvalidAccount`` / ``InvalidQOS`` / ``BadConstraints``).
    Neither is ever planned by the backfill scheduler, so ``StartTime`` is ``N/A`` and
    stays ``N/A`` — measured on the live queue: every one of the 15
    ``QOSMaxWallDurationPerJobLimit`` jobs and the one ``InvalidAccount`` job reports
    no estimate, and job 47297644 has been PENDING on that reason for 166 days.

    Answering "when will it start?" with "calculating… (the scheduler estimates a
    start once the job has waited a few minutes)" is then a promise about an event
    that cannot happen, printed directly under a Why line that already says "waiting
    won't help" — the same self-contradiction SW-29 removed from the WHERE table, in
    the line above it. Held-like waits are deliberately NOT included: those have
    their own wording, and a ``--begin`` job really does have a future start time.
    """
    r = (reason or "").strip().lower()
    return "perjob" in r or any(t in r for t in _INVALID_REQUEST_REASONS)


# Reasons meaning the job is not in the priority line and will not rejoin it on its
# own: a hold needs `scontrol release`, a dead dependency will never clear, and an
# invalid or oversized request has to be resubmitted. Deliberately NARROWER than
# `is_held_like`, which also covers a plain `Dependency` or `BeginTime` wait -- those
# DO rejoin the line by themselves, at a priority that has been ageing the whole
# time, so they stay counted.
_NEVER_COMPETING_REASONS = ("held", "neversatisfied")


def _never_competes(reason: str) -> bool:
    """True when this pending row will never be weighed against another job.

    Used only for the ``#N of M`` queue position, where such a row inflates M -- and,
    because these are the OLDEST jobs on the queue and priority ages, usually N too.
    Measured on the live controller (1,582 pending rows): 159
    ``DependencyNeverSatisfied``, 27 ``JobHeldUser`` and 15 ``request_must_change``
    rows, 12.7% of the queue, and they are not stragglers -- 43900631 has been
    ``DependencyNeverSatisfied`` for 239 days, 47297644
    ``QOSMaxWallDurationPerJobLimit`` for 166, 53473823 ``JobHeldUser`` for 15. On
    caslake this moves a real job from "#1078 of 1252" to "#931 of 1105": 147 fewer
    jobs said to be ahead of it, none of which the scheduler will ever start.

    ``DependencyNeverSatisfied`` counts as never because this controller leaves those
    jobs queued indefinitely instead of killing them (``DependencyParameters =
    (null)``, i.e. no ``kill_invalid_depend``).
    """
    r = (reason or "").strip().lower()
    return any(tok in r for tok in _NEVER_COMPETING_REASONS) or request_must_change(r)


def _raw_job_number(job_id: str | None) -> int | None:
    """The numeric job id behind ``12345``, ``12345_7`` or ``12345_[1-9%2]``.

    ``None`` when there is no plain number to compare (a het-job ``12345+0``, a mock
    id, an absent field) -- callers must then skip the tie-break rather than guess.
    A PENDING array task reports the ARRAY job id here, which is the same number
    ``squeue %i`` prints for the un-launched row (verified: ``scontrol show job
    53302068_1`` -> ``JobId=53302068``), so the two sides compare.
    """
    if not job_id:
        return None
    base = job_id.strip().split("_", 1)[0]
    return int(base) if base.isdigit() else None


def capacity_is_irrelevant(reason: str) -> bool:
    """Whether the "where is there room?" question is beside the point for this job.

    SW-29 stopped answering it 51 times for a held / dependency / begin-time job,
    whose screen contradicted itself: a closing tip saying "it isn't waiting on free
    capacity" above a table of partitions that all said "FITS NOW". The same
    contradiction reappeared for two families that are not holds:

    * a per-JOB limit (`QOSMaxWallDurationPerJobLimit` and friends) — the request is
      too large for the limit, so no amount of free capacity starts it as submitted;
    * an invalid request (`InvalidAccount`, `InvalidQOS`, `BadConstraints`) — nothing
      is waiting for anything.

    A usage cap is deliberately NOT included: such a job is priority-ordered and will
    run when the user's other jobs finish, so the room figures remain real context —
    that decision has its own test and this must not quietly reverse it.
    """
    return is_held_like(reason) or request_must_change(reason)


def requeue_could_help(reason: str) -> bool:
    """Whether requeuing to a partition with room could plausibly start the job.

    False for holds/dependencies/begin-time/reservation waits (a partition change
    won't clear those); True for capacity/priority waits (incl. an empty reason).
    """
    r = (reason or "").strip().lower()
    if r in ("", "none", "(null)"):
        return True
    if any(tok in r for tok in _INVALID_REQUEST_REASONS) or _is_scoped_limit(r):
        return False
    return not any(tok in r for tok in _NON_CAPACITY_REASONS)


def format_gpu_types(
    types: list[str], width: int, ascii_mode: bool = False, has_gpus: bool = False
) -> str:
    """A GPU-type list trimmed to ``width`` cells on WHOLE items (never a dangling
    ", "), with an ellipsis when some are dropped. When there are no *typed* models
    but the partition does have GPUs (the common untyped ``gpu:N`` form), show
    "GPU" rather than the no-GPU placeholder — else a partition recommended for a
    GPU job would look like it has none."""
    if not types:
        if has_gpus:
            return "GPU"
        return "-" if ascii_mode else "—"  # em dash
    ell = "..." if ascii_mode else "…"  # ellipsis
    kept: list[str] = []
    for t in types:
        if len(", ".join([*kept, t])) <= width:
            kept.append(t)
        else:
            break
    s = ", ".join(kept)
    if len(kept) < len(types):
        # Some types were dropped — always show the indicator (never a silent drop).
        # Make room for the ellipsis by dropping trailing kept items if it won't fit.
        while kept and len(", ".join(kept)) + len(ell) > width:
            kept.pop()
        s = (", ".join(kept) + ell) if kept else ell
    return s


# Blockers that WAITING can clear, as opposed to ones that need a different request.
# A node's size, a partition's wall-clock ceiling and its hardware type do not change
# because you waited; free cores and busy GPUs do.
_TRANSIENT_BLOCKERS = frozenset({"", "no room", "GPUs busy", "down"})


def blocker_is_permanent(blocker: str) -> bool:
    """Whether ``blocker`` describes something no amount of queueing will fix.

    Drives the closing tip: with every partition permanently blocked, "it will start
    once resources free up" is a promise about an event that cannot happen — measured
    on a `--cpus-per-task=999` job against a cluster whose largest node has 64 CPUs.
    SW-28.
    """
    return bool(blocker) and blocker not in _TRANSIENT_BLOCKERS


def largest_node_cpus(parts: list[PartitionResources]) -> int:
    """The biggest single node anywhere, for saying WHY nothing can hold it.

    The CONFIGURED width wherever it is known: ``max_node_cpus`` spans schedulable
    nodes only, so a busy big node would make this understate the hardware — and the
    figure is quoted as a claim about what the cluster OWNS. It is also the one
    ``fit_blocker`` decides "node too small" against, so the tip and the verdict cite
    the same number.
    """
    return max((p.max_config_node_cpus or p.max_node_cpus for p in parts), default=0)


def _slurm_duration(seconds: int) -> str:
    """``seconds`` in the ``D-HH:MM:SS`` / ``H:MM:SS`` shape Slurm itself prints.

    A wall-clock ceiling quoted at the user has to be comparable to what they typed
    in ``--time`` and to what ``sinfo %l`` shows, so it is rendered Slurm's way
    rather than as "4h" or "14400s".
    """
    days, rem = divmod(max(int(seconds), 0), 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{hours}:{mins:02d}:{secs:02d}"


def permanent_blocker_note(job: PendingJob, parts: list[PartitionResources]) -> str:
    """The parenthetical for the "no partition can ever hold this request" tip: the
    figure that actually binds, or "" when nothing measured here bears on the blocker.

    The tip used to append ``(largest node: N CPU)`` whenever that number existed,
    because it was the only figure to hand. But a permanent blocker is just as often
    `time limit`, `no GPU`, `no <type>` or `too few GPUs`, and then a CPU count
    explains nothing. Measured live on a 9-partition account view whose blockers were
    all `no GPU` / `too few GPUs`: a `--gres=gpu:16 --cpus-per-task=1` job was told
    "no partition on this cluster can ever hold this request (largest node: 128 CPU)"
    — a reason about cores, quoted at a job that asked for one core.

    This tip is the strongest claim the screen makes, so a figure is named only where
    the request provably exceeds it, and nothing is named otherwise: the rule here is
    never to assert what was not measured, and a tip with no parenthetical is honest
    where a wrong one is not.

    ONE figure, and the ORDER IS PART OF THE CONTRACT. More than one shape can bind at
    once (`--gres=gpu:8 --cpus-per-task=999` exceeds both), and a mixed blocker set is
    the normal case rather than the exception — measured live on the `beagle3-users`
    account view (6 partitions), a `--gres=gpu:8 --cpus-per-task=1` job is blocked
    `no GPU` in five of them and `too few GPUs` in the sixth. What makes that safe is
    that every branch below tests the request against the maximum over the WHOLE list,
    so a branch fires only when the figure is exceeded on every partition the user can
    reach: each one is then a complete and sufficient reason on its own, and a
    one-line tip needs exactly one. The fixed order (CPU, RAM, GPU, wall-clock) is so
    the choice is made by this function and not by partition order or the hour.
    """
    per_node_cpus = -(-job.req_cpus // max(job.req_nodes, 1))
    biggest_cpus = largest_node_cpus(parts)
    if biggest_cpus > 0 and per_node_cpus > biggest_cpus:
        return f"largest node: {biggest_cpus} CPU"
    # Memory is the other per-node shape that can make a request permanently
    # unrunnable — one `node too small` label covers both — so name RAM when RAM is
    # what no node can hold.
    biggest_mem = max(
        (p.max_config_node_mem_bytes or p.max_node_mem_bytes for p in parts), default=0
    )
    per_node_mem = job.req_mem_bytes / max(job.req_nodes, 1)
    if job.req_mem_bytes > 0 and biggest_mem > 0 and per_node_mem > biggest_mem:
        return f"largest node: {format_bytes(biggest_mem)} RAM"
    # GPUs per node: the `too few GPUs` blocker, and the one worth naming most — a job
    # refused for asking 8 GPUs where the widest node has 4 is owed that 4, and the
    # tip was saying nothing at all. `max_node_gpus` is the figure `fit_blocker`
    # decides `too few GPUs` against, so verdict and reason cite one number, and it is
    # summed from every node line whatever its ALLOC state (unlike `max_node_cpus`,
    # which needed `max_config_node_cpus` to stop reporting this minute's occupancy as
    # hardware) — so there is no schedulable-vs-configured split to correct here.
    #
    # A GPU-LESS partition in the list does not weaken the claim: it cannot supply 8
    # either, so the maximum over the list is still the widest GPU node reachable, and
    # this is what resolves the mixed `no GPU` + `too few GPUs` set. 0 means no GPU
    # node was seen (or `sinfo` gave no %G), and then there is no figure to name.
    biggest_gpus = max((p.max_node_gpus for p in parts), default=0)
    if job.req_gpus > 0 and biggest_gpus > 0 and _per_node_gpus(job) > biggest_gpus:
        return f"largest node: {biggest_gpus} GPU"
    # Wall clock: the `time limit` blocker. The binding figure is the LONGEST ceiling
    # any listed partition allows, and `None` is Slurm's unlimited (or an unreadable
    # %l), so one such partition makes a "max wall-clock" claim false however short
    # the rest are — the figure is named only when every partition reported a finite
    # one. Not reproducible on this cluster: all 87 partitions are MaxTime=UNLIMITED,
    # so this branch is covered against partition rows with a ceiling set by hand.
    limits = [p.timelimit_seconds for p in parts]
    if job.time_limit_seconds is not None and limits and all(x is not None for x in limits):
        longest = max(x for x in limits if x is not None)
        if job.time_limit_seconds > longest:
            return f"max wall-clock: {_slurm_duration(longest)}"
    return ""


def _per_node_gpus(job: PendingJob) -> int:
    """The job's GPU request per node (ceiling division)."""
    return -(-job.req_gpus // max(job.req_nodes, 1))


def available_node_count(job: PendingJob, part: PartitionResources) -> int:
    """Nodes in ``part`` that could actually host ``job`` right now.

    An ``--exclusive`` job takes the whole machine, so only a fully-IDLE node
    counts.

    A GPU job used to be treated the same way, for a reason that no longer holds:
    the aggregate ``sinfo %G`` reports configured GRES, so a mixed node's free
    GPUs were unknowable and counting it would have claimed room that might not
    exist. ``GresUsed`` gives that figure per node, so when we have it
    (``gpu_detail``) a GPU job is measured against the nodes that really have
    enough GPUs left — mixed ones included. Without it we keep the old
    conservative fallback rather than guess.

    A plain CPU job can always land on a mixed node, so it counts idle + mixed.
    """
    if job.exclusive:
        return part.idle_nodes
    if job.req_gpus > 0:
        if part.gpu_detail:
            return part.nodes_with_free_gpus(_per_node_gpus(job))
        return part.idle_nodes
    return part.free_nodes


def capacity_cell(free: int, total: int) -> str:
    """A free-capacity figure with the denominator it is a fraction OF: ``free/total``.

    An "idle cores" cell reading 240 is the same three characters for a 256-core
    partition, which is 94% free and about to take the job, and for a 3200-core one,
    which is 7% free and will not — the two rows rendered identically, so the reader
    could not tell a nearly-empty queue from a busy one, and the requeue tip (which
    ranks partitions by ABSOLUTE free cores, `resolve_cluster_partitions`' sort key)
    looked arbitrary beside them. ``total_nodes`` / ``cpus_total`` were already summed
    from `sinfo` for exactly this and then read by nothing (D17), so the denominator
    costs no new query.

    ``total`` of 0 means unknown — a caller-built ``PartitionResources``, or an `sinfo`
    that gave no %D/%C — and an invented denominator ("240/0", or worse "240/240")
    would be a stronger claim than the bare figure, not a weaker one. Then show the
    numerator alone.
    """
    return f"{free}/{total}" if total > 0 else str(free)


def _shape_blocker(need: float, config_max: float) -> str:
    """Label a per-node request that no FREE node can hold: is it the hardware or the
    hour? Permanent only when even the biggest node the partition owns is too small.

    ``config_max`` of 0 means unknown (a caller-built ``PartitionResources``, or an
    ``sinfo`` that gave no %m/%c), and then the permanent label stands — that is what
    it has always been, so an unreadable field can't quietly soften a real verdict.
    """
    return "no room" if config_max > 0 and need <= config_max else "node too small"


def fit_blocker(job: PendingJob, part: PartitionResources) -> str:
    """Why ``job`` can't start in ``part`` right now — "" if it plausibly fits.

    Returns a short label ("down" / "no room" / "node too small" / "time limit" /
    "no GPU" / "no <type>") so the WHERE table can say *why* a partition with free
    capacity still can't take the job (a GPU/type/walltime/node-size mismatch),
    instead of the misleading catch-all "no room". A coarse estimate, not a
    scheduling guarantee (it can't see QOS/account limits or exact idle GPUs).
    """
    if not part.available:
        return "down"
    # An exclusive or GPU job needs a WHOLE idle node, so measure it against
    # idle-only capacity; a plain CPU job can also use the free cores on a mix node,
    # so it uses the mix-inclusive totals. Without this split, an exclusive job was
    # passed against free cores on busy mix nodes / a big node that isn't idle (M5).
    # A GPU job is only pinned to whole idle nodes when we cannot see free GPUs;
    # with `gpu_detail` it can use a mix node's spare cores and memory like any
    # other job, so measuring it against idle-only capacity would now under-report.
    whole_node = job.exclusive or (job.req_gpus > 0 and not part.gpu_detail)
    # `... or <total>`: the idle-only figure is 0 only when there are no idle nodes
    # at all (then the node-count check below already returns "no room"), so the
    # fallback never masks the M5 fix — it just keeps the mix-inclusive behavior for
    # a plain CPU job and for partitions with no idle-node detail.
    cpus_avail = (part.idle_node_cpus or part.cpus_idle) if whole_node else part.cpus_idle
    max_node_cpus = (
        (part.max_idle_node_cpus or part.max_node_cpus) if whole_node else part.max_node_cpus
    )
    # The memory half now matches the CPU half above: a job that can use a mix node's
    # spare cores is measured against that node's spare MEMORY, not against the memory
    # it was configured with. `%m` is configured size, so before this a partition whose
    # nodes were all busy still advertised their full size -- see
    # `PartitionResources.free_node_mem_mb` for the live measurement. A whole-node job
    # keeps the idle-only figure, which is already free memory by definition.
    max_node_mem = (
        (part.max_idle_node_mem_bytes or part.max_node_mem_bytes)
        if whole_node
        else (
            part.max_free_node_mem_bytes
            if part.mem_detail and part.free_node_mem_mb
            else part.max_node_mem_bytes
        )
    )
    # PERMANENT shape mismatches before transient scarcity — no amount of waiting
    # changes the size of a node. These two used to sit AFTER the aggregate test, which
    # shadowed them: `req_cpus > cpus_avail` is true for any partition with fewer than
    # req_cpus idle cores in TOTAL, so a `--cpus-per-task=999` job (max node on the
    # cluster: 64) was labelled "no room" — transient scarcity — everywhere except the
    # three partitions that happened to have >999 cores idle at that instant, which
    # fell through and got the right answer. The verdict therefore depended on an
    # unrelated coincidence, and the same command an hour later relabelled a partition
    # with nothing about the job or the hardware having changed. SW-28.
    #
    # ...but "the size of a node" has to mean the size of a node the partition OWNS,
    # not the size of the biggest one that happens to be free. Both `max_node_*`
    # figures above are read from schedulable (idle/mix) lines only, so a partition
    # whose large nodes were all ALLOC reported small ones — and the permanent label
    # then blamed hardware for this minute's occupancy, the very inversion the
    # paragraph above exists to prevent. Measured live: `cobey-hm` holds a 64-CPU /
    # 2,063,994 MB node that was ALLOC, and a `--cpus-per-task=64` job there got
    # "node too small" with `blocker_is_permanent` True. `max_config_node_*` counts
    # every node line, so the request must be too big for the hardware AND for what is
    # free to keep the permanent label; too big only for what is free is "no room",
    # which clears by itself.
    #
    # Per-node CPU: the job's per-node share must fit ONE node (an idle-core sum can
    # pass a job needing more cores than any single node has).
    per_node_cpus = -(-job.req_cpus // max(job.req_nodes, 1))
    if max_node_cpus > 0 and per_node_cpus > max_node_cpus:
        return _shape_blocker(per_node_cpus, part.max_config_node_cpus)
    # Per-node memory: the biggest (idle, for a whole-node job) node must hold the
    # per-node share.
    per_node_mem = job.req_mem_bytes / max(job.req_nodes, 1)
    if job.req_mem_bytes > 0 and max_node_mem > 0 and per_node_mem > max_node_mem:
        return _shape_blocker(per_node_mem, part.max_config_node_mem_bytes)
    # A partition whose max wall time is shorter than the job's would reject it.
    if (
        job.time_limit_seconds is not None
        and part.timelimit_seconds is not None
        and job.time_limit_seconds > part.timelimit_seconds
    ):
        return "time limit"
    if job.req_gpus > 0:
        # gpu_types may be empty even when GPUs exist (untyped `gpu:N`), so trust
        # has_gpus OR a parsed type.
        if not (part.has_gpus or part.gpu_types):
            return "no GPU"
        # Enforce a type match only when BOTH sides name a type.
        if (
            job.req_gpu_type
            and part.gpu_types
            and job.req_gpu_type.lower() not in {g.lower() for g in part.gpu_types}
        ):
            return f"no {job.req_gpu_type}"
        # Per-node GPU count: the biggest node must hold the per-node GPU share, or a
        # multi-GPU-per-node job "fits now" on GPU-poor nodes and just sits PENDING —
        # the tool's headline use case, so the wrong recommendation is user-facing (M4).
        per_node_gpus = _per_node_gpus(job)
        if part.max_node_gpus > 0 and per_node_gpus > part.max_node_gpus:
            return "too few GPUs"
    # ---- transient scarcity, only once nothing permanent rules the partition out ----
    # Measured on a 7-partition cluster: a 1-GPU job listed the three GPU-LESS
    # partitions as `standard: no GPU`, `highmem: no GPU` and `test: no room` — the
    # last only because `test` happened to have no fully idle node that minute. Same
    # hardware, same job, two different verdicts, and the transient one ("no room")
    # invites a wait that can never end. `cron` got "time limit" for the same
    # accidental reason: it had a free node, so it reached the check. The permanent
    # tests above therefore run FIRST for every partition, which is the SW-28
    # argument (a verdict must not depend on unrelated cluster load) applied to the
    # GPU/walltime mismatches SW-28 left behind the aggregate test.
    #
    # "GPUs busy" is itself transient (free GPUs come and go), so it belongs here and
    # not above the shape tests: a partition whose nodes are too small for the job
    # should say so rather than blame this minute's GPU occupancy.
    if job.req_gpus > 0 and part.gpu_detail and part.max_node_gpus_free < _per_node_gpus(job):
        return "GPUs busy"
    if job.req_nodes > available_node_count(job, part) or job.req_cpus > cpus_avail:
        return "no room"
    return ""


def partition_fits_now(job: PendingJob, part: PartitionResources) -> bool:
    """Whether ``job`` could plausibly start in ``part`` right now (see fit_blocker)."""
    return fit_blocker(job, part) == ""


def _split_partitions(partition: str) -> list[str]:
    """Split a scontrol ``Partition=`` value into its individual partitions.

    A job submitted with ``sbatch -p a,b`` carries a comma-list; each partition is
    an independent queue, so context that compares priorities or counts pending
    depth must treat them separately, not pool them (P4).
    """
    return [p.strip() for p in partition.split(",") if p.strip()]


# partition -> the QOS names an association allows there. The empty-string key holds
# CLUSTER-level associations, whose QOS apply whatever partition the job sits in.
AssocTable = dict[str, list[str]]


_ASSOC_QOS_CACHE: dict[str, AssocTable | None] = {}


def resolve_user_associations(username: str = "") -> AssocTable | None:
    """What the user's Slurm associations allow, partition by partition.

    ``None`` means UNKNOWN — no ``sacctmgr``, no permission, or nothing parsable.
    That is deliberately distinct from ``{}`` ("read it, the user has no
    associations"): advice is softened when the table is unknown and a partition is
    withheld only when it is known to be unusable. Guessing in either direction was
    the SW-32 failure.

    Two shapes are in the wild and both are handled, because they disagree about
    what a row means:

    * partition-keyed (measured on midway2) — ``build|build``,
      ``broadwl|broadwl,broadwl-large,debug``: the QOS a job gets depends on which
      partition it was submitted to.
    * cluster-level (measured on midway3) — ``|aaz,astroplasmas,build,test,...``
      with an EMPTY partition field: one association, one big QOS list, no
      per-partition keying.

    A parser that assumed either shape alone would read the other as "no
    associations at all" and, on the SW-32 path, silently stop offering every
    partition.
    """
    if _is_mock():
        return {"": ["normal"], "gpu": ["gpu"]}
    # Read ONCE per process, like `acct_gather_disabled`. This is account
    # configuration in the Slurm DB, not job state: it does not change while a
    # dashboard is open. And it was being paid for repeatedly — the pending view
    # asks for it from BOTH `cli._run_pending` and `tui.PendingApp`, so the
    # identical `sacctmgr` ran twice back-to-back (222 ms each, measured) on the
    # first paint and twice more on every 10 s refresh.
    who = username or current_username()
    if who in _ASSOC_QOS_CACHE:
        return _ASSOC_QOS_CACHE[who]
    try:
        out = _run_slurm_cmd(
            [
                "sacctmgr",
                "-nP",
                "show",
                "assoc",
                f"user={who}",
                "format=Partition,QOS",
            ]
        )
    except Exception:
        # NOT cached: a transient sacctmgr failure must not latch "unknown" for the
        # life of the process, or one hiccup permanently softens every later answer.
        return None  # unknown, NOT empty
    table: AssocTable = {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        part, _, qos = line.partition("|")
        names = [q.strip() for q in qos.split(",") if q.strip()]
        if not names:
            continue
        table.setdefault(part.strip(), [])
        for n in names:
            if n not in table[part.strip()]:
                table[part.strip()].append(n)
    result = table or None
    _ASSOC_QOS_CACHE[who] = result
    return result


def qos_for_partition(partition: str, assoc: AssocTable | None) -> str | None:
    """The QOS to request when moving a job into ``partition``, if it is knowable.

    A move that changes only the partition keeps the old QOS, and where QOS names
    track partition names — which is how both clusters measured here are set up — the
    destination rejects it: the job goes from ``Resources``, which clears by itself,
    to ``InvalidQOS``, which does not (SW-32).

    Prefer a QOS named after the partition, since that is the convention that creates
    the problem in the first place. Otherwise take the only candidate. With several
    unrelated names there is nothing to choose between them, so return ``None`` and
    let the caller say so rather than pick one and be wrong.
    """
    if assoc is None:
        return None
    candidates = assoc.get(partition) or assoc.get("") or []
    if partition in candidates:
        return partition
    return candidates[0] if len(candidates) == 1 else None


def partition_allowed_by_assoc(partition: str, assoc: AssocTable | None) -> bool:
    """Could this user submit to ``partition`` at all, as far as the table says?

    Unknown table -> True: a partition that would work must not be withheld because
    ``sacctmgr`` was unreadable. Known table -> the partition must be named by a
    partition-keyed row, or a cluster-level row must exist (which applies everywhere).
    """
    if assoc is None:
        return True
    if partition in assoc:
        return True
    return "" in assoc


def partition_move_command(job_id: str, partition: str, assoc: AssocTable | None) -> str:
    """The ``scontrol update`` line offered as the requeue remedy.

    Carries ``QOS=`` when it is knowable, because without it the command makes the
    job strictly worse (SW-32).
    """
    cmd = f"scontrol update JobId={job_id} Partition={partition}"
    qos = qos_for_partition(partition, assoc)
    return f"{cmd} QOS={qos}" if qos else cmd


def partition_move_caveat(
    partition: str, assoc: AssocTable | None, ascii_mode: bool = False
) -> str:
    """The one-line hedge to print when the command cannot be complete.

    Empty when the command is self-sufficient. `fit_blocker`'s own docstring has
    always said the estimate "can't see QOS/account limits"; SW-32 was that hedge
    never reaching the screen the command was printed on.

    Folds through :func:`_asciify` when asked, exactly as :func:`explain_reason`
    does and for the same reason: **the plain report has no whole-surface fold.**
    It folds token by token (its own `dash`/`dot`/`dots`) and helper by helper, so
    an unfolded string reaches the screen intact -- the tip read "has room for this
    request now -" while the hedge printed directly under it still carried an em
    dash, on a terminal that had asked for none, against that function's own
    promise of "no stray Unicode".

    The dashboard needs nothing from this: `PendingView.render` ends in
    ``_asciify(out) if ascii_mode else out`` (`tui.py`), folding everything it has
    built. Measured, not assumed -- handing the view an unfolded caveat still
    renders ASCII-clean -- so its call site deliberately does NOT pass the flag,
    and there is a control pinning that asymmetry.
    """
    if qos_for_partition(partition, assoc) is not None:
        return ""
    msg = (
        "check your QOS for it first — the QOS moves with the job, not the partition"
        if assoc is None
        else "add QOS=<name> if that partition needs its own — the QOS does not move with it"
    )
    return _asciify(msg) if ascii_mode else msg


def resolve_priority_rank(
    partition: str, priority: int | None, job_id: str | None = None
) -> tuple[int, int] | None:
    """The job's ``(rank, total_pending)`` among a partition's pending jobs by priority.

    ``rank`` is 1-based, highest-priority first (rank 1 = next in line). Used to
    quantify a ``Priority`` wait ("#312 of 453") so "queued behind higher-priority
    jobs" is a concrete position, not just a phrase. ``None`` when the priority is
    unknown or ``squeue`` can't be read.

    For a partition comma-list (``sbatch -p a,b``) the job sits in each queue
    independently, so the rank is computed PER partition — priorities are only
    comparable within one partition — and the BEST (the queue it's nearest the front
    of, where it will start first) is returned. A single pooled ``squeue -p a,b``
    would mix non-comparable cross-partition priorities and double-count the job (P4).

    Two things keep the pair honest about the line the scheduler actually forms.

    ``job_id`` makes the position a POINT instead of a set. Counting only
    ``priority > mine`` leaves every job at EXACTLY my priority neither ahead nor
    behind, so a whole tie run is handed one "#N": on the live queue 758 of the 887
    jobs that get shown a rank sit in such a run — a median 7 others on the same
    number, worst case 102, all 103 of them told "#1078 of 1252". Slurm does not
    leave that order open — ``slurmctld`` sorts the queue by priority descending and
    breaks a tie by ASCENDING job id — so the seat is knowable, and is used whenever
    a ``job_id`` is supplied. Checked against this controller rather than taken on
    faith: across the 245 equal-priority groups among caslake jobs that queued over
    an hour, the lower job id started first in 99.8% of 2.9M within-group pairs
    (median per-group concordance 1.000). The residue is ``sched/backfill`` slotting
    a short job in early, which no ordering can predict.

    ``M`` counts only jobs that can be weighed against this one (`_never_competes`):
    a hold, a dead dependency or a request that must change is queued but not in
    line, and dropping those took a live caslake job from "#1078 of 1252" to "#931 of
    1105". They are dropped from BOTH sides — a job that will never start cannot be
    "ahead" either — and the caller's own row is always kept, so "of M" never
    excludes the job whose position it describes.
    """
    if priority is None:
        return None
    if _is_mock():
        return 4, 5
    me = _raw_job_number(job_id)
    best: tuple[int, int] | None = None
    for part in _split_partitions(partition):
        try:
            # -a for the same reason every sinfo call here passes it: a job can be
            # pending in a partition flagged Hidden=YES, and `squeue` WITHOUT -a
            # ("-a, --all: display jobs in hidden partitions") returns NOTHING for
            # one -- not even the caller's own job. An empty queue then reads as
            # "no other pending jobs", i.e. rank #1, for a job that may be #400.
            out = _run_slurm_cmd(["squeue", "-a", "-h", "-p", part, "-t", "PD", "-o", "%Q|%i|%r"])
        except Exception:
            continue  # best-effort context; never let a squeue hiccup break the view
        # Line-oriented, with the reason LAST: a reason carries spaces and commas
        # ("ReqNodeNotAvail, UnavailableNodes:midway3-[0440-0441]" is live on this
        # queue), so a whitespace split would shatter one row into bogus tokens and a
        # fixed field count would drop it. A priority-only line still parses, so a
        # controller that will not give the wider -o degrades to the plain count
        # rather than to no rank at all.
        rows: list[tuple[int, int | None, str]] = []
        for line in out.splitlines():
            tok, _, rest = line.strip().partition("|")
            tok = tok.strip()
            if not tok.lstrip("-").isdigit():
                continue
            row_id, _, reason = rest.partition("|")
            rows.append((int(tok), _raw_job_number(row_id), reason.strip()))
        rows = [
            row for row in rows if not _never_competes(row[2]) or (me is not None and row[1] == me)
        ]
        if not rows:
            continue
        ahead = sum(
            1
            for prio, row_id, _ in rows
            if prio > priority
            or (prio == priority and me is not None and row_id is not None and row_id < me)
        )
        # Clamp: if the job left the PD set between the scontrol read and this squeue
        # snapshot (it started/was held/cancelled), or its priority drifted above the
        # stale scontrol value, its own entry is absent from `rows` — making
        # ahead == len(rows) and the rank exceed the total. Never render an impossible
        # "#N of M" with N > M.
        rank = (min(ahead + 1, len(rows)), len(rows))
        if best is None or rank[0] < best[0]:
            best = rank
    return best


def resolve_queue_counts(partition: str) -> tuple[int, int] | None:
    """(running, pending) DISTINCT job counts on ``partition`` for queue-pressure context.

    Returns ``None`` when squeue can't be read (timeout / controller busy) so the
    caller shows "unavailable" rather than a fabricated, self-contradictory
    "0 running · 0 pending" for a partition that provably holds at least the user's
    own pending job.

    squeue lists a job pending in several partitions (``sbatch -p a,b``) once PER
    partition, so counts are deduplicated by job id; summing the raw lines would
    double-count every such job and inflate the pending depth (P4).
    """
    if _is_mock():
        return 12, 5
    try:
        # -a: include jobs in Hidden=YES partitions. Without it squeue lists none of
        # them and this returns a (0, 0) that the docstring above exists to rule out
        # -- an empty result is indistinguishable from a genuinely empty queue, so the
        # SlurmCommandError guard cannot catch it. -a does not widen the -p filter.
        out = _run_slurm_cmd(["squeue", "-a", "-h", "-p", partition, "-o", "%i|%T"])
    except SlurmCommandError:
        return None
    states: dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        job_id, _, state = line.partition("|")
        job_id, state = job_id.strip(), state.strip().upper()
        if job_id and state:
            states[job_id] = state
    running = pending = 0
    for state in states.values():
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
        # Generic, like every other value in this fixture: a real site's
        # allocation name has no business shipping in --demo (SW-6).
        account="demo-alloc",
        reason="Resources",
        submit_time=now - 5400,  # queued 1.5h ago
        start_time_estimate=now + 3600,  # scheduler estimate: ~1h out
        priority=10432,
        req_cpus=16,
        req_nodes=1,
        req_mem_bytes=64 * 1024**3,
        req_gpus=2,
        req_gpu_type="a100",
        # 8h fits the gpu-a100 demo target (12h limit) but not the shorter
        # partitions — keeps the "gpu-a100 has room" story coherent with the
        # partition time-limit check.
        time_limit_seconds=8 * 3600,
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
