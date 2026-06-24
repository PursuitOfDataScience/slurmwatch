from __future__ import annotations

import contextlib
import os
import pwd
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

from .exceptions import (
    CgroupNotFoundError,
    CgroupPermissionError,
    JobNotFoundError,
    JobNotRunningError,
    LoginNodeError,
    SlurmCommandError,
)
from .model import JobContext

SLURM_CMD_TIMEOUT = 15
_CGROUP_V2_BASE = Path("/sys/fs/cgroup")
_MOCK_ENV_VAR = "SLURMWATCH_MOCK"


def _is_mock() -> bool:
    return os.environ.get(_MOCK_ENV_VAR) == "1"


def _run_slurm_cmd(cmd: list[str], timeout: int = SLURM_CMD_TIMEOUT) -> str:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
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
        bracket_match = re.match(r"^([a-zA-Z0-9_-]+)\[([^\]]+)\]$", part)
        if bracket_match:
            prefix = bracket_match.group(1)
            ranges_content = bracket_match.group(2)
            for rng in ranges_content.split(","):
                rng = rng.strip()
                if "-" in rng:
                    start_str, end_str = rng.split("-", 1)
                    try:
                        pad = len(start_str)
                        start_n, end_n = int(start_str), int(end_str)
                        for i in range(start_n, end_n + 1):
                            nodes.append(f"{prefix}{str(i).zfill(pad)}")
                    except ValueError:
                        nodes.append(f"{prefix}{rng}")
                else:
                    nodes.append(f"{prefix}{rng}")
        else:
            nodes.append(part)
    return nodes


def resolve_current_jobs(username: str | None = None) -> list[dict[str, object]]:
    if _is_mock():
        return [
            {
                "job_id": "12345",
                "state": "R",
                "partition": "gpu-highend",
                "name": "train",
                "nodes": "3",
                "wall_time": "2:00:00",
                "time_limit": "4:00:00",
                "reason": "None",
            },
        ]
    if username is None:
        username = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    output = _run_slurm_cmd(["squeue", "-u", username, "-h", "-o", "%i %t %P %j %D %M %l %R"])
    jobs: list[dict[str, object]] = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 7)
        if len(parts) >= 2 and parts[1] == "R":
            job: dict[str, object] = {"job_id": parts[0], "state": parts[1]}
            if len(parts) > 2:
                job["partition"] = parts[2]
            if len(parts) > 3:
                job["name"] = parts[3]
            if len(parts) > 4:
                job["nodes"] = parts[4]
            if len(parts) > 5:
                job["wall_time"] = parts[5]
            if len(parts) > 6:
                job["time_limit"] = parts[6]
            if len(parts) > 7:
                job["reason"] = parts[7]
            jobs.append(job)
    return jobs


def resolve_job_context(
    job_id: str,
    step_id: str | None = None,
) -> JobContext:
    if _is_mock():
        return _make_mock_job_context(job_id, step_id)

    try:
        output = _run_slurm_cmd(["scontrol", "show", "job", job_id])
    except SlurmCommandError as exc:
        raise JobNotFoundError(f"Job {job_id} not found") from exc

    job_state = _parse_scontrol_field(output, "JobState")
    if job_state and job_state.upper() not in ("RUNNING", "CONFIGURING", "COMPLETING"):
        raise JobNotRunningError(
            f"Job {job_id} is in state '{job_state}'. Only running jobs can be monitored."
        )

    username = _parse_scontrol_field(output, "UserId") or ""
    username = username.split("@")[0].split("(")[0] if username else ""

    partition = _parse_scontrol_field(output, "Partition") or "unknown"
    nodelist_raw = _parse_scontrol_field(output, "NodeList") or ""
    cpus = int(_parse_scontrol_field(output, "NumCPUs") or "0")

    hostname = socket.gethostname().split(".")[0]
    uid = _resolve_uid(username)
    resolved_nodes = _parse_nodelist(nodelist_raw)

    mem_bytes, gpu_count = 0, 0
    tres = _parse_scontrol_field(output, "TRES") or ""
    alloc_tres = _parse_scontrol_field(output, "AllocTRES") or ""
    tres_str = alloc_tres or tres

    if tres_str:
        for token in tres_str.split(","):
            token = token.strip()
            if token.startswith("mem="):
                mem_bytes = _parse_mem_to_bytes(token.split("=", 1)[1])
            elif token.startswith("gres/gpu"):
                parts = token.split("=", 1)
                if len(parts) > 1:
                    gpu_count = int(parts[1])
            elif token == "gres/gpu":
                gpu_count = 1

    if mem_bytes == 0:
        mem_str = _parse_scontrol_field(output, "Mem") or ""
        if mem_str:
            mem_bytes = _parse_mem_to_bytes(mem_str)
    if mem_bytes == 0:
        mem_per_node = _parse_scontrol_field(output, "MinMemoryNode") or ""
        if mem_per_node:
            mem_bytes = _parse_mem_to_bytes(mem_per_node)

    if gpu_count == 0:
        gres = _parse_scontrol_field(output, "GRES") or ""
        gpu_count = _parse_gpu_count(gres)

    gpu_indices = _resolve_gpu_indices(output, uid)

    min_memory_node = 0
    min_mem_str = _parse_scontrol_field(output, "MinMemoryNode") or ""
    if min_mem_str:
        min_memory_node = _parse_mem_to_bytes(min_mem_str)

    start_time_str = _parse_scontrol_field(output, "StartTime")
    job_start_time: float | None = None
    if start_time_str and start_time_str not in ("Unknown", "N/A"):
        try:
            start_time_struct = time.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S")
            job_start_time = time.mktime(start_time_struct)
        except (ValueError, OSError):
            pass

    ctx = JobContext(
        job_id=job_id,
        username=username,
        partition=partition,
        nodelist=",".join(resolved_nodes) if resolved_nodes else nodelist_raw,
        hostname=hostname,
        cpus_allocated=cpus,
        mem_limit_bytes=mem_bytes,
        gpu_count_requested=gpu_count,
        gpu_indices=gpu_indices,
        step_id=step_id,
        uid=uid,
        job_start_time=job_start_time,
        job_state=job_state,
        nodelist_resolved=resolved_nodes,
        min_memory_node=min_memory_node,
        tres=tres_str,
    )

    try:
        cgroup_paths = _discover_cgroup_paths(job_id, uid, step_id)
    except CgroupNotFoundError as exc:
        if shutil.which("squeue") or shutil.which("sacct"):
            raise LoginNodeError(
                f"This appears to be a Slurm login node rather than a compute node.\n"
                f"Job {job_id} runs on {nodelist_raw} but its cgroups are not "
                f"present on this host ({hostname}).\n"
                f"Use 'srun --jobid {job_id} --overlap slurmwatch' to attach "
                f"to the compute node."
            ) from exc
        raise
    ctx.cgroup_v2_path = str(cgroup_paths.get("v2")) if cgroup_paths.get("v2") else None
    ctx.cgroup_v1_mem_path = str(cgroup_paths.get("v1_mem")) if cgroup_paths.get("v1_mem") else None
    ctx.cgroup_v1_cpu_path = str(cgroup_paths.get("v1_cpu")) if cgroup_paths.get("v1_cpu") else None

    return ctx


_SCONTROL_FIELD_RE = re.compile(r"(?:^|\s)(\w+)=(\S+)")


def _parse_scontrol_field(output: str, field: str) -> str | None:
    for line in output.split("\n"):
        for m in _SCONTROL_FIELD_RE.finditer(line):
            if m.group(1) == field:
                return m.group(2)
    return None


def _parse_gpu_count(gres: str) -> int:
    if not gres:
        return 0
    total = 0
    for part in gres.split(","):
        part = part.strip()
        gpu_match = re.match(r"gpu(?::\w+)?:(\d+)", part)
        if gpu_match:
            total += int(gpu_match.group(1))
    return total


def _resolve_uid(username: str) -> int | None:
    try:
        return pwd.getpwnam(username).pw_uid
    except (KeyError, OSError):
        return None


def _resolve_gpu_indices(scontrol_output: str, uid: int | None = None) -> list[int]:
    gpu_indices: list[int] = []

    if uid is not None:
        pids = _find_job_pids(uid)
        for pid in pids[:5]:
            env = _read_pid_environ(pid)
            if env:
                cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "")
                if cuda_visible:
                    try:
                        gpu_indices = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
                        return gpu_indices
                    except ValueError:
                        pass

    gres_detail = _parse_scontrol_field(scontrol_output, "GresDetail") or ""
    if gres_detail:
        for part in gres_detail.split(","):
            part = part.strip()
            m = re.search(r"gpu:(\w+):(\d+)", part)
            if m:
                with contextlib.suppress(ValueError):
                    gpu_indices.append(int(m.group(2)))

    env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env_gpus:
        try:
            return [int(x.strip()) for x in env_gpus.split(",") if x.strip()]
        except ValueError:
            pass

    return gpu_indices


def _find_job_pids(uid: int) -> list[int]:
    pids: list[int] = []
    try:
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                status = (proc / "status").read_text()
                for line in status.split("\n"):
                    if line.startswith("Uid:"):
                        parts = line.split()
                        if len(parts) > 1 and int(parts[1]) == uid:
                            pids.append(int(proc.name))
                            break
            except (PermissionError, FileNotFoundError, ValueError, OSError):
                continue
    except PermissionError:
        pass
    return pids


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
    return JobContext(
        job_id=job_id,
        username="demo",
        partition="gpu-highend",
        nodelist="cn-[001-004]",
        hostname=hostname,
        cpus_allocated=16,
        mem_limit_bytes=64 * 1024**3,
        gpu_count_requested=4,
        gpu_indices=[0, 1, 2, 3],
        step_id=step_id or "0",
        uid=1001,
        job_start_time=time.time() - 7200,
    )


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
            try:
                for child in (_CGROUP_V2_BASE / "system.slice").iterdir():
                    if f"job_{base_job_id}" in child.name:
                        result["v2"] = child
                        break
            except (PermissionError, FileNotFoundError):
                pass

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
