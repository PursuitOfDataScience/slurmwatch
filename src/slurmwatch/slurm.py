from __future__ import annotations

import contextlib
import logging
import os
import pwd
import re
import socket
import subprocess
import time
from pathlib import Path

from .exceptions import (
    CgroupNotFoundError,
    CgroupPermissionError,
    JobNotFoundError,
    JobNotRunningError,
    SlurmCommandError,
)
from .model import JobContext, local_node_name, short_host

SLURM_CMD_TIMEOUT = 15
_CGROUP_V2_BASE = Path("/sys/fs/cgroup")
_MOCK_ENV_VAR = "SLURMWATCH_MOCK"
_MAX_SANE_CPU_SECONDS = 3650 * 86400  # 10 years; guards against NO_VAL sentinels
# Cap total hostlist expansion so a corrupt/garbage NodeList (e.g. cn[1-1e8] or a
# cartesian blow-up a[1-10000]b[1-10000]) can't exhaust memory / hang. Far above
# any real allocation, so a legitimate nodelist is never truncated (#audit3-10).
_MAX_HOSTLIST_NODES = 65536
# Cap GPU IDX-range expansion the same way — far above any real node's GPU count,
# so a crafted `IDX:0-<huge>` can't exhaust memory, but a real list never truncates.
_MAX_GPU_IDX = 4096

logger = logging.getLogger("slurmwatch")

# A het job's scontrol records carry JobId=<leader>+<component>, e.g. 12345+0.
_HET_JOBID_RE = re.compile(r"\bJobId=(\S+\+\d+)")


def _count_het_components(scontrol_output: str) -> int:
    """How many distinct het-job components are in a ``scontrol show job`` dump.

    >1 means the job is heterogeneous (one record per component); slurmwatch
    monitors only the selected component, so the caller warns.
    """
    return len({m.group(1) for m in _HET_JOBID_RE.finditer(scontrol_output)})


def _is_mock() -> bool:
    return os.environ.get(_MOCK_ENV_VAR) == "1"


def _run_slurm_cmd(cmd: list[str], timeout: int = SLURM_CMD_TIMEOUT) -> str:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            # A job name/comment can carry non-UTF-8 bytes; without errors= the
            # strict decode raises UnicodeDecodeError and breaks every Slurm call.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise SlurmCommandError(f"Slurm binary not found: {cmd[0]}. Is Slurm installed?") from exc
    except subprocess.TimeoutExpired as exc:
        raise SlurmCommandError(f"Command {' '.join(cmd)} timed out after {timeout}s") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise SlurmCommandError(
            f"Command {' '.join(cmd)} failed (rc={result.returncode}): {stderr}"
        )
    return result.stdout


def _parse_mem_to_bytes(mem_str: str) -> int:
    mem_str = mem_str.strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if mem_str.isdigit():
        return int(mem_str)
    for suffix, mult in multipliers.items():
        if mem_str.endswith(suffix):
            try:
                return int(float(mem_str[:-1]) * mult)
            except ValueError:
                pass
    try:
        return int(float(mem_str))
    except ValueError:
        return 0


def _expand_range_group(content: str) -> list[str]:
    """Expand the inside of one bracket group into padded strings.

    Handles the forms Slurm emits: single values ('007'), ranges ('001-003'),
    and stepped ranges ('001-007:2' -> 001,003,005,007), comma-joined
    ('001-003,007'). An unparseable or reversed range is kept verbatim rather
    than silently dropped, so a malformed hostlist never quietly collapses the
    node count (#39)."""
    out: list[str] = []
    for rng in content.split(","):
        rng = rng.strip()
        if not rng:
            continue
        if "-" not in rng:
            out.append(rng)
            continue
        start_str, rest = rng.split("-", 1)
        # A stepped range is 'start-end:step' (Slurm's hostlist syntax).
        step_str = "1"
        if ":" in rest:
            end_str, step_str = rest.split(":", 1)
        else:
            end_str = rest
        try:
            pad = len(start_str)
            start_n, end_n, step_n = int(start_str), int(end_str), int(step_str)
        except ValueError:
            out.append(rng)  # not numeric (e.g. 'a-c') -> keep verbatim
            continue
        if step_n < 1 or start_n > end_n:
            out.append(rng)  # bad step or reversed -> keep verbatim, don't drop
            continue
        for i in range(start_n, end_n + 1, step_n):
            if len(out) >= _MAX_HOSTLIST_NODES:  # bound a pathological range (#audit3-10)
                return out
            out.append(str(i).zfill(pad))
    return out


def _expand_hostlist_part(part: str) -> list[str]:
    """Expand one hostlist element, including multi-dimensional bracket groups.

    Slurm hostlists can carry more than one bracket group per element
    (``rack[1-2]node[3-4]`` → rack1node3, rack1node4, rack2node3, rack2node4);
    the earlier single-trailing-bracket parser silently dropped the extra
    dimensions and undercounted nodes (B-P12). Splits the element into literal
    and bracket tokens in order, then takes the cartesian product.
    """
    tokens: list[tuple[str, str]] = []  # (kind, value); kind is "lit" or "brk"
    i = 0
    n = len(part)
    while i < n:
        if part[i] == "[":
            j = part.find("]", i)
            if j == -1:
                tokens.append(("lit", part[i:]))
                break
            tokens.append(("brk", part[i + 1 : j]))
            i = j + 1
        else:
            j = part.find("[", i)
            if j == -1:
                tokens.append(("lit", part[i:]))
                break
            tokens.append(("lit", part[i:j]))
            i = j

    result = [""]
    for kind, value in tokens:
        options = [value] if kind == "lit" else _expand_range_group(value)
        # Build the product incrementally and stop AT the cap, so a cartesian
        # blow-up (a[1-10000]b[1-10000] = 100M) can't materialise the full list
        # before a post-hoc size check (#audit3-10).
        combined: list[str] = []
        for prefix in result:
            for opt in options:
                if len(combined) >= _MAX_HOSTLIST_NODES:
                    return combined
                combined.append(prefix + opt)
        result = combined
    return result


def _parse_nodelist(nodelist: str) -> list[str]:
    if not nodelist or nodelist == "(null)":
        return []

    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in nodelist:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))

    nodes: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            nodes.extend(_expand_hostlist_part(part))
        if len(nodes) >= _MAX_HOSTLIST_NODES:  # bound total expansion (#audit3-10)
            return nodes[:_MAX_HOSTLIST_NODES]
    return nodes


def resolve_current_jobs(username: str | None = None) -> list[dict[str, object]]:
    if _is_mock():
        return [
            {
                "job_id": "12345",
                "state": "R",
                "partition": "gpu-highend",
                "name": "train",
                "nodes": "4",
                "wall_time": "2:00:00",
                "time_limit": "4:00:00",
                "reason": "None",
            },
        ]
    if username is None:
        username = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    # Pipe-delimited so job names with spaces don't shift columns. The job name
    # (%j) is the only free-form field and is placed *last* so that a literal
    # '|' inside it is absorbed by the final split() field instead of shifting
    # every column after it (B-P10). Every field before it (id/state/partition/
    # nodes/times/reason) is machine-generated and pipe-free.
    output = _run_slurm_cmd(["squeue", "-u", username, "-h", "-o", "%i|%t|%P|%D|%M|%l|%R|%j"])
    jobs: list[dict[str, object]] = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 7)
        parts = [p.strip() for p in parts]
        # Include running AND pending jobs so the picker offers both (a pending
        # pick routes to the why/when/where view). Other transient states
        # (completing/configuring) aren't monitorable, so they're left out.
        if len(parts) >= 2 and parts[1] in ("R", "PD"):
            job: dict[str, object] = {"job_id": parts[0], "state": parts[1]}
            if len(parts) > 2:
                job["partition"] = parts[2]
            if len(parts) > 3:
                job["nodes"] = parts[3]
            if len(parts) > 4:
                job["wall_time"] = parts[4]
            if len(parts) > 5:
                job["time_limit"] = parts[5]
            if len(parts) > 6:
                job["reason"] = parts[6]
            if len(parts) > 7:
                job["name"] = parts[7]
            jobs.append(job)
    return jobs


def _sacct_final_state(job_id: str) -> tuple[str, str] | None:
    """``(State, End)`` from the accounting DB for a job no longer in the controller.

    Returns ``None`` if sacct has no record (the job truly never existed) or the
    query fails. ``-X`` limits to the top-level job (one row, no steps); a state
    like ``CANCELLED by 1234`` is reduced to its first word.
    """
    if _is_mock():
        return None
    try:
        out = _run_slurm_cmd(["sacct", "-n", "-P", "-X", "-j", job_id, "--format=State,End"])
    except SlurmCommandError:
        return None
    for line in out.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        state = parts[0].strip().split(" ")[0] if parts[0].strip() else ""
        end = parts[1].strip() if len(parts) > 1 else ""
        if state:
            return state, end
    return None


def resolve_job_context(
    job_id: str,
    step_id: str | None = None,
) -> JobContext:
    if _is_mock():
        return _make_mock_job_context(job_id, step_id)

    try:
        # -d adds the per-node allocation detail lines (GRES=gpu:N(IDX:...)),
        # the only node-global source of allocated GPU indices.
        output = _run_slurm_cmd(["scontrol", "show", "job", "-d", job_id])
    except SlurmCommandError as exc:
        # scontrol purges finished jobs after MinJobAge, but the accounting DB
        # keeps them: distinguish a job that *finished* (still in sacct) from one
        # that never existed, so we don't tell the user a completed job "does not
        # exist" (which reads like a typo'd id).
        finished = _sacct_final_state(job_id)
        if finished is not None:
            state, end = finished
            when = f", ended {end}" if end and end not in ("Unknown", "") else ""
            raise JobNotRunningError(
                f"Job {job_id} has finished (State: {state}{when}). "
                "slurmwatch shows live telemetry for running jobs only."
            ) from exc
        raise JobNotFoundError(f"Job {job_id} not found") from exc

    hostname = local_node_name()
    record = _select_job_record(output, hostname)

    # Het jobs return one record per component (JobId=<leader>+N); we resolve only
    # the selected component, so its sibling components' nodes/GPUs aren't shown.
    # Warn rather than silently misrepresent the job's scope (full het aggregation
    # is not implemented).
    het = _count_het_components(output)
    if het > 1:
        logger.warning(
            "Job %s is heterogeneous (%d components); slurmwatch is showing only "
            "one component's node(s) — the other components aren't monitored.",
            job_id,
            het,
        )

    job_state = _parse_scontrol_field(record, "JobState")
    if job_state and job_state.upper() not in ("RUNNING", "CONFIGURING", "COMPLETING"):
        raise JobNotRunningError(
            f"Job {job_id} is in state '{job_state}'. Only running jobs can be monitored."
        )

    username = _parse_scontrol_field(record, "UserId") or ""
    username = username.split("@")[0].split("(")[0] if username else ""

    partition = _parse_scontrol_field(record, "Partition") or "unknown"
    nodelist_raw = _parse_scontrol_field(record, "NodeList") or ""
    cpus = _parse_leading_int(_parse_scontrol_field(record, "NumCPUs"))
    num_nodes = max(_parse_leading_int(_parse_scontrol_field(record, "NumNodes")), 1)

    uid = _resolve_uid(username)
    resolved_nodes = _parse_nodelist(nodelist_raw)

    # The node whose per-node detail we read. On the compute node that's this
    # host; viewed off-node (login node / --once / --log) the host is in no
    # detail line, so scope to the node the collector will actually represent —
    # nodelist[0], the hop/stream/remote-summary target — instead of falling
    # back to a job-wide // NumNodes average that matches no real node on a
    # heterogeneous allocation (#31).
    if _host_in_nodelist(hostname, resolved_nodes) or not resolved_nodes:
        detail_host = hostname
    else:
        detail_host = resolved_nodes[0]

    tres = _parse_scontrol_field(record, "TRES") or ""
    alloc_tres = _parse_scontrol_field(record, "AllocTRES") or ""
    tres_str = alloc_tres or tres

    mem_bytes = 0
    gpu_count = 0
    if tres_str:
        for token in tres_str.split(","):
            token = token.strip()
            if token.startswith("mem="):
                mem_bytes = _parse_mem_to_bytes(token.split("=", 1)[1])
        gpu_count = _parse_tres_gpus(tres_str)

    min_memory_node = 0
    min_mem_str = _parse_scontrol_field(record, "MinMemoryNode") or ""
    if min_mem_str:
        min_memory_node = _parse_mem_to_bytes(min_mem_str)

    # slurmwatch monitors one node, so limits must be node-local. Prefer the
    # exact per-node figures on the `scontrol -d` detail line (CPU_IDs / Mem) for
    # the target node; fall back to the job-wide totals only when the detail line
    # is absent (B-P4). The fallback rounds UP (ceil): on a job that doesn't
    # divide evenly (30 CPUs over 4 nodes) it matches the largest real node
    # instead of truncating to 7 and inflating every % against a too-small limit,
    # which could trip a false OOM-critical (#32).
    node_cpus, node_mem = _parse_node_detail(record, detail_host)
    if node_cpus > 0:
        cpus = node_cpus
    elif num_nodes > 1:
        cpus = max(-(-cpus // num_nodes), 1)
    if node_mem > 0:
        mem_bytes = node_mem
    elif num_nodes > 1:
        mem_bytes = min_memory_node if min_memory_node > 0 else -(-mem_bytes // num_nodes)
    if mem_bytes == 0:
        mem_bytes = min_memory_node

    # Per-node GPU count: prefer the exact IDX list on the target node's detail
    # line (node-local and precise), so an uneven GPU-per-node allocation shows
    # the right number instead of a job-wide // NumNodes average that contradicts
    # the GPU rows actually rendered (#33). Fall back to a per-node Gres field,
    # then to division.
    detail_gpu_indices = _parse_gres_idx(record, detail_host)
    per_node_gpus = 0
    for field in ("TresPerNode", "Gres"):
        gres = _parse_scontrol_field(record, field) or ""
        per_node_gpus = _parse_gpu_count(gres)
        if per_node_gpus:
            break
    if detail_gpu_indices:
        gpu_count = len(detail_gpu_indices)
    elif num_nodes > 1:
        # TRES gres/gpu=N is the job-wide total; the panel wants this node's.
        gpu_count = per_node_gpus if per_node_gpus else -(-gpu_count // num_nodes)
    elif gpu_count == 0:
        gpu_count = per_node_gpus

    def _parse_scontrol_time(field: str) -> float | None:
        raw = _parse_scontrol_field(record, field)
        if not raw or raw in ("Unknown", "N/A"):
            return None
        try:
            return time.mktime(time.strptime(raw, "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, OSError):
            return None

    job_start_time = _parse_scontrol_time("StartTime")
    submit_time = _parse_scontrol_time("SubmitTime")

    # Job provenance from the same record — for the dashboard's JOB card. These
    # are single-token scontrol fields (Command is the script path; args aren't
    # in this field), so the existing key=value parser captures them cleanly.
    # scontrol prints "(null)" for an unset field (e.g. Command on an interactive
    # salloc job) — normalize that (and N/A/Unknown) to "" so the card omits the
    # line instead of showing a useless "(null)".
    def _clean_field(field: str) -> str:
        val = _parse_scontrol_field(record, field) or ""
        return "" if val in ("(null)", "(none)", "N/A", "Unknown") else val

    account = _clean_field("Account")
    qos = _clean_field("QOS")
    command = _clean_field("Command")
    work_dir = _clean_field("WorkDir")
    std_out = _clean_field("StdOut")
    std_err = _clean_field("StdErr")

    # TimeLimit is 'D-HH:MM:SS' / 'HH:MM:SS' — or 'UNLIMITED'/'Partition_Limit'
    # when there's no fixed wall-clock cap (leave it None then).
    time_limit_str = _parse_scontrol_field(record, "TimeLimit") or ""
    time_limit_seconds: int | None = None
    if time_limit_str and time_limit_str.upper() not in ("UNLIMITED", "PARTITION_LIMIT", "N/A"):
        secs = _parse_slurm_duration(time_limit_str)
        if secs > 0:
            time_limit_seconds = int(secs)

    ctx = JobContext(
        job_id=job_id,
        username=username,
        partition=partition,
        nodelist=",".join(resolved_nodes) if resolved_nodes else nodelist_raw,
        hostname=hostname,
        cpus_allocated=cpus,
        mem_limit_bytes=mem_bytes,
        gpu_count_requested=gpu_count,
        gpu_indices=[],
        gpu_uuids=[],
        step_id=step_id,
        uid=uid,
        job_start_time=job_start_time,
        job_state=job_state,
        time_limit_seconds=time_limit_seconds,
        nodelist_resolved=resolved_nodes,
        min_memory_node=min_memory_node,
        tres=tres_str,
        account=account,
        qos=qos,
        command=command,
        work_dir=work_dir,
        std_out=std_out,
        std_err=std_err,
        submit_time=submit_time,
    )

    # Cgroups are named after the task's raw JobId (array tasks and het
    # components have their own), not the user-facing 12345_3 / 123+1 form.
    raw_job_id = _parse_scontrol_field(record, "JobId") or job_id
    # Persist it before the possible remote early-return: `srun --jobid=` only
    # accepts this numeric id, so the login-node hop needs it too.
    ctx.raw_job_id = raw_job_id
    try:
        cgroup_paths = _discover_cgroup_paths(raw_job_id, uid, step_id)
    except CgroupNotFoundError:
        # Not on the job's compute node (e.g. a login node): fall back to
        # remote usage via sstat instead of erroring. GPU utilization is
        # unavailable this way, but memory and CPU are.
        ctx.remote = True
        return ctx
    ctx.cgroup_v2_path = str(cgroup_paths.get("v2")) if cgroup_paths.get("v2") else None
    ctx.cgroup_v1_mem_path = str(cgroup_paths.get("v1_mem")) if cgroup_paths.get("v1_mem") else None
    ctx.cgroup_v1_cpu_path = str(cgroup_paths.get("v1_cpu")) if cgroup_paths.get("v1_cpu") else None

    job_pids = _cgroup_pids(
        [p for p in (cgroup_paths.get("v2"), cgroup_paths.get("v1_cpu")) if p is not None]
    )
    gpu_indices, gpu_uuids = _resolve_gpu_indices(record, hostname, job_pids)
    ctx.gpu_indices = gpu_indices
    ctx.gpu_uuids = gpu_uuids
    if ctx.gpu_count_requested == 0 and gpu_indices:
        ctx.gpu_count_requested = len(gpu_indices)

    return ctx


class RemoteUsage:
    """Live job usage sampled remotely via sstat (works from any node)."""

    def __init__(self, rss_bytes: int, cpu_seconds: float, sampled: bool) -> None:
        self.rss_bytes = rss_bytes
        self.cpu_seconds = cpu_seconds
        self.sampled = sampled  # False until Slurm has taken its first sample


def _parse_slurm_duration(text: str) -> float:
    """Parse a Slurm duration ('D-HH:MM:SS', 'HH:MM:SS', 'MM:SS.mmm') to seconds."""
    text = text.strip()
    if not text:
        return 0.0
    days = 0
    if "-" in text:
        day_str, _, text = text.partition("-")
        with contextlib.suppress(ValueError):
            days = int(day_str)
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return 0.0
    seconds = 0.0
    for n in nums:
        seconds = seconds * 60 + n
    return days * 86400 + seconds


def resolve_remote_usage(job_id: str, node_count: int = 1) -> RemoteUsage:
    """Query sstat for a running job's per-node peak RSS and CPU time.

    sstat totals are job-wide, but slurmwatch compares against per-node limits,
    so each step is scaled by an estimated per-node task count
    (max(1, NTasks // node_count)). Using at least one task means a concentrated
    step (NTasks < nodes, or a single-task head step) reports its real
    single-node footprint rather than being diluted by node_count. Returns zeros
    with sampled=False when Slurm has not yet produced a sample.
    """
    if _is_mock():
        return RemoteUsage(rss_bytes=32 * 1024**3, cpu_seconds=3600.0, sampled=True)
    try:
        output = _run_slurm_cmd(
            [
                "sstat",
                "--allsteps",
                "--noheader",
                "-P",
                "-j",
                job_id,
                "--format=JobID,MaxRSS,AveCPU,NTasks",
            ]
        )
    except SlurmCommandError:
        return RemoteUsage(rss_bytes=0, cpu_seconds=0.0, sampled=False)

    peak_rss = 0
    cpu_seconds = 0.0
    sampled = False
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        fields = line.split("|")
        if len(fields) < 4:
            continue
        job_field, max_rss, ave_cpu, ntasks = fields[0], fields[1], fields[2], fields[3]
        # Scope to the requested job. `sstat -j <ArrayJobId>` widens to EVERY
        # running array task (the representative task's raw id equals the
        # ArrayJobId), so summing every row would over-count CPU N-fold — the #30
        # fix's blind spot for the base task. Keep only steps whose base id matches.
        if job_field.strip().split(".")[0] != job_id:
            continue
        if not max_rss.strip():
            # No RSS sample -> no valid CPU sample either (skips extern step).
            continue
        sampled = True
        try:
            tasks = int(ntasks.strip())
        except ValueError:
            tasks = 1
        tasks = max(tasks, 1)
        # MaxRSS/AveCPU are single-task figures; scale by the per-node task count
        # for a per-node total (exact for balanced tasks such as MPI ranks). The
        # floor of 1 keeps a concentrated step from being diluted below its real
        # single-node footprint.
        tasks_per_node = max(1, tasks // node_count)
        peak_rss = max(peak_rss, _parse_mem_to_bytes(max_rss) * tasks_per_node)
        step_cpu = _parse_slurm_duration(ave_cpu)
        # Steps Slurm hasn't sampled report a NO_VAL sentinel
        # (e.g. AveCPU "213503982334-14:25:51"); ignore anything absurd.
        if step_cpu >= _MAX_SANE_CPU_SECONDS:
            continue
        cpu_seconds += step_cpu * tasks_per_node
    return RemoteUsage(rss_bytes=peak_rss, cpu_seconds=cpu_seconds, sampled=sampled)


# States in which a job is still on a compute node and worth monitoring. SUSPENDED
# and STOPPED are held allocations (gang scheduling, PreemptMode=SUSPEND, `scontrol
# suspend`) that resume on the SAME node, so they are NOT ended. Anything else
# (COMPLETED, FAILED, CANCELLED, TIMEOUT, ...) means the job has left the node.
_ACTIVE_JOB_STATES = frozenset(
    {"RUNNING", "COMPLETING", "CONFIGURING", "RESIZING", "SIGNALING", "SUSPENDED", "STOPPED"}
)
# States where the job is back in the queue but NOT finished — it will run again
# under the same JobId (preemption with PreemptMode=REQUEUE, `scontrol requeue`,
# NODE_FAIL with --requeue, a held requeue). Treating these as "ended" would tear
# the dashboard down on a job that's merely waiting to resume, so they count as
# alive (there's just no node telemetry until it runs again). PENDING covers a
# requeued job that has settled back to the pending queue.
_REQUEUED_JOB_STATES = frozenset(
    {"PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD"}
)


def is_job_active(job_id: str) -> bool | None:
    """Whether ``job_id`` is still on a node, for a mid-flight liveness recheck.

    ``True`` = still allocated (running or a resumable suspend); ``False`` =
    gone/terminal (so the dashboard can show "job ended" and stop); ``None`` =
    couldn't tell, so the caller must NOT treat it as ended — a transient squeue
    hiccup or a slow controller should never tear down a live dashboard.

    ``squeue`` only lists active jobs, so an empty result means the job has left
    the queue (ended). A purged job is instead *rejected* with "Invalid job id
    specified"; that exact message is the only failure taken as ended — every
    other error (timeouts, socket errors, controller unreachable) is unknown, so
    we never mistake a slow/flaky controller for a finished job. Pass the raw
    numeric JobId: ``squeue -j 12345`` and ``12345_3`` both widen to the whole
    array, but any still-active task keeps it True regardless.
    """
    if _is_mock():
        return True
    try:
        output = _run_slurm_cmd(["squeue", "-h", "-j", job_id, "-o", "%T"])
    except SlurmCommandError as exc:
        # Only an explicit "invalid/unknown job id" means the job is truly gone.
        # Anything else (timeout, "Socket timed out on send/recv", "Unable to
        # contact slurm controller") is a transient/infra failure -> unknown.
        msg = str(exc).lower()
        if "invalid job id" in msg or "invalid job" in msg:
            return False
        return None
    states = [line.strip().upper() for line in output.strip().split("\n") if line.strip()]
    if not states:
        return False  # not listed among active jobs -> ended
    # A requeued/preempted job is still queued (will rerun under the same id), so
    # it's alive even though it's momentarily off the node — don't declare "ended".
    return any(state in _ACTIVE_JOB_STATES or state in _REQUEUED_JOB_STATES for state in states)


def _host_in_nodelist(hostname: str, nodes: list[str]) -> bool:
    """Whether ``hostname`` is one of ``nodes``, tolerant of case and domain.

    A node's own ``gethostname`` and Slurm's ``NodeName`` can differ by case or a
    kept domain suffix on some clusters (e.g. ``gpu01`` vs ``gpu01.cluster.edu``).
    Comparing the short forms on *both* sides keeps host matching working there
    (#29); an exact ``in`` test silently failed, discarding the per-node CPU/mem/
    GPU detail, picking the wrong array-task record, and mis-flagging a live local
    job as remote. ``short_host`` also lower-cases, so it covers case mismatch.
    This matches how ``collector.py`` and ``tui.py`` already resolve the local
    node, so the whole codebase agrees on what "this host" means.

    Slurm ``NodeName`` values are unique short names within a cluster, so the
    short forms don't collide in practice. The only shapes that could over-match
    are degenerate (two nodes named identically apart from their domain, or
    bare-IP node names collapsing to a first octet) and are not produced by a
    normal single-domain Slurm config."""
    target = short_host(hostname)
    return any(short_host(n) == target for n in nodes)


def _select_job_record(output: str, hostname: str) -> str:
    """Pick the right record when scontrol returns several (job arrays).

    Prefers a RUNNING record whose nodelist contains this host, then any
    RUNNING record, then the first record.
    """
    records = [r for r in re.split(r"\n\s*\n", output) if "JobId=" in r]
    if len(records) <= 1:
        return records[0] if records else output
    running = [
        r
        for r in records
        if (_parse_scontrol_field(r, "JobState") or "").upper()
        in ("RUNNING", "CONFIGURING", "COMPLETING")
    ]
    for record in running:
        nodes = _parse_nodelist(_parse_scontrol_field(record, "NodeList") or "")
        if _host_in_nodelist(hostname, nodes):
            return record
    return running[0] if running else records[0]


def _parse_leading_int(value: str | None) -> int:
    if not value:
        return 0
    m = re.match(r"\d+", value)
    return int(m.group(0)) if m else 0


def _parse_tres_gpus(tres_str: str) -> int:
    """GPU count from a TRES string.

    Only `gres/gpu=N` and typed `gres/gpu:type=N` count; `gres/gpumem=...` and
    `gres/gpuutil=...` share the prefix but are different TRES. The generic
    entry is the total; typed entries are summed only when it is absent.
    """
    generic: int | None = None
    typed_total = 0
    for token in tres_str.split(","):
        m = re.match(r"gres/gpu(?::([^=]+))?=(\d+)$", token.strip())
        if not m:
            continue
        if m.group(1) is None:
            generic = int(m.group(2))
        else:
            typed_total += int(m.group(2))
    return generic if generic is not None else typed_total


# A ``key=`` token: a key of any non-space, non-'=' chars (so colon/slash keys
# like ``AllocNode:Sid`` and ``gres/gpu`` are single tokens), at a start-of-line
# or whitespace boundary. Each field's value runs up to the NEXT such token.
_SCONTROL_KEY_RE = re.compile(r"(?:^|\s)([^\s=]+)=")

# Free-form fields whose value is user-controlled and may itself contain a
# ``<space>Key=value`` sequence. Once one starts, the rest of ITS line is its
# value — so a job name like ``x Partition=[/]`` can't shadow the real
# ``Partition=`` on a later line, corrupt the reported field, or (before the
# panels escaped) smuggle markup that crashes the TUI (#audit3-9). Each is the
# last field on its own line in ``scontrol show job`` output.
_SCONTROL_FREE_TEXT_KEYS = frozenset(
    {"JobName", "Name", "Command", "WorkDir", "Comment", "StdOut", "StdErr", "StdIn"}
)


def _parse_scontrol_field(output: str, field: str) -> str | None:
    """Value of ``field`` from ``scontrol`` key=value output.

    A value may contain spaces (Command, WorkDir, JobName, Comment, Std* paths),
    so it must not be truncated at the first space (#37); it runs from just after
    ``field=`` to the next ``key=`` token (or end of line). Splitting on the next
    key token — rather than a lazy ``.*?`` regex — keeps values with spaces intact
    AND correctly bounds a field whose neighbour has a colon/slash key: real
    scontrol prints ``Partition=gpu AllocNode:Sid=login1:42`` on one line, and a
    ``\\w+=`` lookahead would over-capture that whole tail into Partition. It is
    also linear, avoiding the O(n^2) backtracking a lazy regex has on long input.
    ``key=`` inside a comma-joined value (``TRES=cpu=4,mem=8G``) isn't matched: it
    follows ``=``/``,``, not a whitespace/line boundary.
    """
    for line in output.split("\n"):
        tokens = list(_SCONTROL_KEY_RE.finditer(line))
        # Once a free-text key starts, the rest of the line is ITS value — drop
        # any later ``key=`` tokens so an embedded ``Partition=…`` inside a job
        # name can't be read as a real field (#audit3-9).
        for j, m in enumerate(tokens):
            if m.group(1) in _SCONTROL_FREE_TEXT_KEYS:
                tokens = tokens[: j + 1]
                break
        for i, m in enumerate(tokens):
            if m.group(1) == field:
                start = m.end()
                end = tokens[i + 1].start() if i + 1 < len(tokens) else len(line)
                return line[start:end].strip()
    return None


def _parse_gpu_count(gres: str) -> int:
    """GPU count from a Gres/TresPerNode value like 'gpu:2' or 'gres/gpu:a100:2'."""
    if not gres:
        return 0
    total = 0
    for part in gres.split(","):
        part = part.strip()
        gpu_match = re.match(r"(?:gres/)?gpu(?::[\w.\-]+)?:(\d+)", part)
        if gpu_match:
            total += int(gpu_match.group(1))
    return total


def _resolve_uid(username: str) -> int | None:
    try:
        return pwd.getpwnam(username).pw_uid
    except (KeyError, OSError):
        return None


def _split_cuda_visible(cuda_visible: str) -> tuple[list[int], list[str]]:
    """Split a CUDA_VISIBLE_DEVICES value into integer ordinals and UUID/MIG tokens."""
    idxs: list[int] = []
    uuids: list[str] = []
    for tok in cuda_visible.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            idxs.append(int(tok))
        except ValueError:
            uuids.append(tok)
    return idxs, uuids


_GRES_IDX_RE = re.compile(r"gpu[^(,]*\(IDX:([0-9,\-]+)\)")


def _resolve_gpu_indices(
    record: str, hostname: str, job_pids: list[int]
) -> tuple[list[int], list[str]]:
    """Resolve which node-local GPUs belong to the job.

    Priority: the IDX list from `scontrol show job -d` (node-global, exact),
    then CUDA_VISIBLE_DEVICES read from the job's own processes, then this
    process's environment. Integer ordinals from process environments are a
    last resort: with ConstrainDevices they are renumbered relative to the
    job's device cgroup and may not match node-global indices. UUID/MIG
    tokens are absolute, so they are kept whenever found.
    """
    indices = _parse_gres_idx(record, hostname)

    # Union CUDA_VISIBLE_DEVICES across the job's processes: with per-task GPU
    # binding each rank sees only its own device(s), so keeping just the first
    # PID's value under-reports the node's allocation (B-P13).
    ordinals: list[int] = []
    uuids: list[str] = []
    for pid in job_pids[:8]:
        env = _read_pid_environ(pid)
        cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "")
        if not cuda_visible:
            continue
        o, u = _split_cuda_visible(cuda_visible)
        for ordinal in o:
            if ordinal not in ordinals:
                ordinals.append(ordinal)
        for uuid in u:
            if uuid not in uuids:
                uuids.append(uuid)

    if not (indices or ordinals or uuids):
        env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if env_gpus:
            ordinals, uuids = _split_cuda_visible(env_gpus)

    if indices:
        return indices, uuids
    return sorted(ordinals), uuids


def _parse_node_detail(record: str, hostname: str) -> tuple[int, int]:
    """(cpus, mem_bytes) for this host from the ``scontrol show job -d`` detail.

    The detail output carries per-node lines such as
    ``Nodes=cn[001-002] CPU_IDs=0-15 Mem=64000 GRES=gpu:a100:2(IDX:0-1)``. The
    CPU_IDs count and Mem are the exact per-node allocation — more precise than
    dividing the job-wide totals by the node count. ``Mem`` is megabytes when
    unsuffixed. Returns (0, 0) when no single line can be attributed to this
    host (same fallback rule as :func:`_parse_gres_idx`)."""
    matching: list[str] = []
    fallback: list[str] = []
    for line in record.split("\n"):
        if "CPU_IDs=" not in line:
            continue
        nodes_str = _parse_scontrol_field(line, "Nodes") or ""
        if _host_in_nodelist(hostname, _parse_nodelist(nodes_str)):
            matching.append(line)
        else:
            fallback.append(line)
    lines = matching or (fallback if len(fallback) == 1 else [])
    if not lines:
        return 0, 0
    line = lines[0]

    cpu_ids = _parse_scontrol_field(line, "CPU_IDs") or ""
    cpus = len(_expand_idx_list(cpu_ids)) if cpu_ids else 0

    mem_str = _parse_scontrol_field(line, "Mem") or ""
    mem_bytes = 0
    if mem_str:
        if mem_str[-1:].isalpha():
            mem_bytes = _parse_mem_to_bytes(mem_str)
        else:
            # An unsuffixed Mem in the -d node detail is in megabytes.
            mem_bytes = _parse_leading_int(mem_str) * 1024**2
    return cpus, mem_bytes


def _parse_gres_idx(record: str, hostname: str) -> list[int]:
    """Extract this node's allocated GPU indices from `scontrol show job -d`.

    The detail output contains per-node lines like
    `Nodes=cn[001-002] CPU_IDs=0-15 Mem=64000 GRES=gpu:a100:2(IDX:0-1)`.
    """
    matching_lines: list[str] = []
    fallback_lines: list[str] = []
    for line in record.split("\n"):
        if "IDX:" not in line or "GRES" not in line:
            continue
        nodes_str = _parse_scontrol_field(line, "Nodes") or ""
        node_names = _parse_nodelist(nodes_str)
        if _host_in_nodelist(hostname, node_names):
            matching_lines.append(line)
        else:
            fallback_lines.append(line)
    # If no line names this host (single-node job or hostname mismatch),
    # only trust the detail when there is exactly one allocation line.
    lines = matching_lines or (fallback_lines if len(fallback_lines) == 1 else [])

    indices: list[int] = []
    for line in lines:
        for idx_list in _GRES_IDX_RE.findall(line):
            indices.extend(_expand_idx_list(idx_list))
    return sorted(set(indices))


def _expand_idx_list(idx_list: str) -> list[int]:
    """Expand an IDX range list like '0-1,3' into [0, 1, 3].

    Capped like the NodeList expansion (#audit3-10): this parses untrusted
    ``scontrol show job -d`` text, and a crafted GPU ``IDX:0-2000000000`` (e.g. via
    a hostile JobName that the raw GRES regex doesn't field-shadow) would otherwise
    materialize billions of ints and exhaust memory / hang the monitor. No real
    node has more GPUs than this bound, so a legitimate list never hits it.
    """
    out: list[int] = []
    for rng in idx_list.split(","):
        rng = rng.strip()
        if not rng:
            continue
        if "-" in rng:
            start_str, _, end_str = rng.partition("-")
            with contextlib.suppress(ValueError):
                start, end = int(start_str), int(end_str)
                # Bound the span before materializing range() (a huge end would OOM).
                end = min(end, start + _MAX_GPU_IDX)
                out.extend(range(start, end + 1))
        else:
            with contextlib.suppress(ValueError):
                out.append(int(rng))
        if len(out) >= _MAX_GPU_IDX:
            return out[:_MAX_GPU_IDX]
    return out


def _cgroup_pids(paths: list[Path]) -> list[int]:
    """Union of PIDs from cgroup.procs files anywhere under the given cgroups.

    On cgroup v2 processes live only in leaf cgroups (job_X/step_Y/user/task_Z),
    so every descendant must be visited.
    """
    pids: set[int] = set()
    for base in paths:
        files = [base / "cgroup.procs"]
        with contextlib.suppress(OSError):
            files.extend(base.rglob("cgroup.procs"))
        for procs_file in files:
            try:
                data = procs_file.read_text()
            except OSError:
                continue
            for token in data.split():
                if token.isdigit():
                    pids.add(int(token))
    return sorted(pids)


def _read_pid_environ(pid: int) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        data = Path(f"/proc/{pid}/environ").read_bytes()
        for entry in data.split(b"\x00"):
            if not entry:
                continue
            if b"=" in entry:
                key, _, val = entry.partition(b"=")
                env[key.decode("utf-8", errors="replace")] = val.decode("utf-8", errors="replace")
    except (PermissionError, FileNotFoundError, OSError):
        pass
    return env


def _make_mock_job_context(
    job_id: str,
    step_id: str | None = None,
) -> JobContext:
    hostname = socket.gethostname().split(".")[0]
    # Node 1 must be *this* host. The dashboard serves the node it runs on from
    # the local collector and streams every other node over srun; a mock nodelist
    # of purely fictional names therefore matched no local node, so `--demo`
    # selected an unreachable node[0] and sat on "awaiting telemetry…" forever
    # while the mock collector's frames went nowhere (#27). Keeping the remaining
    # names fictional still exercises the node switcher in the demo. Any filler
    # name that collides with this host's own is dropped (a machine actually
    # called cn-002 would otherwise be listed twice in the switcher), so the demo
    # always shows exactly four distinct nodes.
    filler = [
        n for n in ("cn-002", "cn-003", "cn-004", "cn-005") if short_host(n) != short_host(hostname)
    ]
    resolved_nodes = [hostname, *filler[:3]]
    return JobContext(
        job_id=job_id,
        username="demo",
        partition="gpu-highend",
        nodelist=",".join(resolved_nodes),
        hostname=hostname,
        cpus_allocated=16,
        mem_limit_bytes=64 * 1024**3,
        gpu_count_requested=4,
        gpu_indices=[0, 1, 2, 3],
        step_id=step_id or "0",
        uid=1001,
        job_start_time=time.time() - 7200,
        time_limit_seconds=24 * 3600,
        nodelist_resolved=resolved_nodes,
        job_state="RUNNING",
        tres="cpu=16,mem=64G,gres/gpu=4",
        account="rcc-staff",
        qos="normal",
        command="/home/demo/proj/train.py",
        work_dir="/home/demo/proj/runs/2026-07",
        std_out=f"/home/demo/proj/runs/2026-07/logs/train-{job_id}.out",
        std_err=f"/home/demo/proj/runs/2026-07/logs/train-{job_id}.err",
        submit_time=time.time() - 7500,
    )


def _cgroup_name_matches_job(name: str, base_job_id: str) -> bool:
    """Whether a cgroup directory name belongs to ``base_job_id``.

    Matches ``job_<id>`` only at a numeric boundary, so ``job_123`` no longer
    matches ``job_1234``/``job_12345`` and attaches to the wrong job (B-P11).
    Still tolerates suffixed forms such as ``job_123.scope`` or ``job_123_0``.
    """
    target = f"job_{base_job_id}"
    pos = name.find(target)
    if pos == -1:
        return False
    after = name[pos + len(target) :]
    return not after[:1].isdigit()


def _discover_cgroup_paths(
    job_id: str,
    uid: int | None = None,
    step_id: str | None = None,
) -> dict[str, Path | None]:
    result: dict[str, Path | None] = {"v2": None, "v1_mem": None, "v1_cpu": None}

    base_job_id = job_id.split("_")[0] if "_" in job_id else job_id

    if detect_cgroup_version() == 2:
        v2_candidates = [
            _CGROUP_V2_BASE / "system.slice" / "slurmstepd.scope" / f"job_{base_job_id}",
        ]
        if step_id is not None:
            step_path = (
                _CGROUP_V2_BASE
                / "system.slice"
                / "slurmstepd.scope"
                / f"job_{base_job_id}"
                / f"step_{step_id}"
            )
            v2_candidates.insert(0, step_path)
            v2_candidates.append(
                _CGROUP_V2_BASE / "system.slice" / "slurmstepd.scope" / f"step_{step_id}",
            )

        for path in v2_candidates:
            if path.exists():
                result["v2"] = path
                break

        if result["v2"] is None:
            # Fallback scan for a job cgroup whose exact path didn't match a
            # candidate above (e.g. a `.scope`-suffixed or otherwise non-standard
            # name). Slurm's job cgroups live *inside* slurmstepd.scope, so scan
            # there first; older layouts put them directly under system.slice, so
            # scan that too (F3 — the fallback previously only looked in
            # system.slice and never descended into slurmstepd.scope).
            scope = _CGROUP_V2_BASE / "system.slice" / "slurmstepd.scope"
            for parent in (scope, _CGROUP_V2_BASE / "system.slice"):
                try:
                    children = list(parent.iterdir())
                except (PermissionError, FileNotFoundError, NotADirectoryError):
                    continue
                for child in children:
                    if _cgroup_name_matches_job(child.name, base_job_id):
                        result["v2"] = child
                        break
                if result["v2"] is not None:
                    break

    v1_mem_base = _CGROUP_V2_BASE / "memory"
    v1_cpu_base = _CGROUP_V2_BASE / "cpuacct"

    if uid is not None:
        for base, key in [(v1_mem_base, "v1_mem"), (v1_cpu_base, "v1_cpu")]:
            if not base.exists():
                continue
            paths_to_check = [
                base / "slurm" / f"uid_{uid}" / f"job_{base_job_id}",
            ]
            if step_id is not None:
                paths_to_check.insert(
                    0,
                    base / "slurm" / f"uid_{uid}" / f"job_{base_job_id}" / f"step_{step_id}",
                )
            for path in paths_to_check:
                if path.exists():
                    result[key] = path
                    break

    if result["v2"] is None and result["v1_mem"] is None and result["v1_cpu"] is None:
        if uid is not None:
            raise CgroupNotFoundError(
                f"No cgroup hierarchy found for job {job_id} (uid={uid}). "
                "This host may not be a Slurm compute node, or the job's cgroups "
                "have been cleaned up. Try running from within the job allocation."
            )
        raise CgroupNotFoundError(
            f"No cgroup hierarchy found for job {job_id}. "
            "Unable to determine UID for path resolution."
        )

    if result["v2"] is not None:
        try:
            _check_cgroup_readable(result["v2"])
        except PermissionError as exc:
            raise CgroupPermissionError(
                f"Cgroup path {result['v2']} exists but is not readable. "
                "Try running slurmwatch from within a Slurm job allocation."
            ) from exc

    return result


def _check_cgroup_readable(path: Path) -> None:
    if not path.is_dir():
        return
    try:
        next(iter(path.iterdir()))
    except PermissionError:
        raise
    except StopIteration:
        pass


def detect_cgroup_version() -> int:
    if (_CGROUP_V2_BASE / "cgroup.controllers").exists():
        return 2
    return 1
