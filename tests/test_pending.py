from __future__ import annotations

import asyncio
import time

import pytest
from rich.text import Text

import slurmwatch.cli as cli
from slurmwatch import pending
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.exceptions import (
    JobNotFoundError,
    JobNotPendingError,
    JobNotRunningError,
    SlurmCommandError,
)
from slurmwatch.pending import (
    PartitionResources,
    PendingJob,
    explain_reason,
    fit_blocker,
    partition_fits_now,
    resolve_cluster_partitions,
    resolve_pending_job,
    resolve_priority_rank,
    resolve_queue_counts,
)

# A realistic `scontrol show job` record for a pending GPU job.
_PENDING_RECORD = (
    "JobId=12345 JobName=train\n"
    "   UserId=demo(1001) GroupId=demo(1001)\n"
    "   Priority=10432 Nice=0 Account=rcc-staff QOS=normal\n"
    "   JobState=PENDING Reason=Resources Dependency=(null)\n"
    "   SubmitTime=2024-01-15T10:00:00 EligibleTime=2024-01-15T10:00:00\n"
    "   StartTime=2024-01-15T11:00:00 EndTime=Unknown\n"
    "   Partition=gpu-highend AllocNode:Sid=login1:42\n"
    "   NumNodes=1 NumCPUs=16 NumTasks=1 CPUs/Task=1\n"
    "   TimeLimit=1-00:00:00\n"
    "   TRES=cpu=16,mem=64G,node=1,billing=16,gres/gpu=2\n"
    "   Gres=gpu:a100:2\n"
)


class TestExplainReason:
    @pytest.mark.parametrize(
        ("reason", "needle"),
        [
            ("Resources", "free nodes"),
            ("Priority", "higher-priority"),
            ("Dependency", "depends on"),
            ("ReqNodeNotAvail", "unavailable"),
            ("QOSMaxJobsPerUserLimit", "QOS limit"),
            ("QOSGrpCpuLimit", "QOS limit"),
            ("AssocMaxCpuPerUserLimit", "account/association"),
            ("BeginTime", "begin time"),
            ("", "Being scheduled"),
            ("None", "Being scheduled"),
        ],
    )
    def test_known_and_prefix_reasons(self, reason: str, needle: str) -> None:
        assert needle.lower() in explain_reason(reason).lower()

    def test_unknown_reason_is_surfaced_verbatim(self) -> None:
        assert "Wibble" in explain_reason("Wibble")

    def test_ascii_mode_folds_unicode(self) -> None:
        # "Priority" explanation has an em-dash; ascii_mode must fold it to ASCII.
        out = explain_reason("Priority", ascii_mode=True)
        out.encode("ascii")  # raises if any glyph leaked
        assert "—" not in out and "-" in out

    def test_nodes_down_free_text_is_not_mislabelled_a_partition_limit(self) -> None:
        # #60 review: this common free-text reason contains "partitions" and used
        # to be mislabelled a partition limit; it's really node availability.
        msg = explain_reason(
            "Nodes required for job are DOWN, DRAINED or reserved for jobs in "
            "higher priority partitions"
        )
        assert "unavailable" in msg.lower()
        assert "partition limit" not in msg.lower()


class TestGpuTypeFromGres:
    def test_typed_colon_form(self) -> None:
        assert pending._gpu_type_from_gres("gpu:a100:2") == "a100"

    def test_typed_tres_equals_form(self) -> None:
        # #60 review: a job-level --gpus=a100:2 records the type only as the TRES
        # equals form gres/gpu:a100=2, which must still yield the type.
        assert pending._gpu_type_from_gres("cpu=16,mem=64G,gres/gpu:a100=2") == "a100"

    def test_untyped_and_empty(self) -> None:
        assert pending._gpu_type_from_gres("gpu:2") == ""
        assert pending._gpu_type_from_gres("gres/gpu=2") == ""
        assert pending._gpu_type_from_gres("") == ""

    def test_typed_gpu_request_via_gpus_flag_keeps_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end: a --gpus=a100:2 pending record (type only in ReqTRES `=` form,
        # empty Gres/TresPerNode) resolves with req_gpu_type="a100", not "".
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        rec = (
            "JobId=555 JobName=t\n   JobState=PENDING Reason=Resources\n"
            "   Partition=gpu NumNodes=1 NumCPUs=8\n"
            "   ReqTRES=cpu=8,mem=32G,node=1,gres/gpu=2,gres/gpu:a100=2\n"
            "   TresPerNode=(null) Gres=(null)\n"
        )
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: rec)
        job = resolve_pending_job("555")
        assert job.req_gpus == 2 and job.req_gpu_type == "a100"


class TestResolvePendingJob:
    def _no_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)

    def test_parses_a_pending_record(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_mock(monkeypatch)
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: _PENDING_RECORD)
        job = resolve_pending_job("12345")
        assert job.job_id == "12345"
        assert job.partition == "gpu-highend"
        assert job.reason == "Resources"
        assert job.account == "rcc-staff" and job.qos == "normal"
        assert job.username == "demo"
        assert job.priority == 10432
        assert job.req_cpus == 16 and job.req_nodes == 1
        assert job.req_mem_bytes == 64 * 1024**3
        assert job.req_gpus == 2 and job.req_gpu_type == "a100"
        assert job.time_limit_seconds == 24 * 3600
        assert job.submit_time is not None and job.start_time_estimate is not None

    def test_min_memory_per_cpu_is_scaled_to_total(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # #60 review: when TRES has no mem= token, MinMemoryCPU is PER CPU — it must
        # be multiplied by NumCPUs to get the whole-job total (not stored verbatim).
        self._no_mock(monkeypatch)
        rec = (
            "JobId=7 JobName=t\n   JobState=PENDING Reason=Resources\n"
            "   Partition=cpu NumNodes=1 NumCPUs=16\n"
            "   TRES=cpu=16,node=1,billing=16\n"
            "   MinMemoryCPU=4G MinMemoryNode=0\n"
        )
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: rec)
        job = resolve_pending_job("7")
        assert job.req_mem_bytes == 16 * 4 * 1024**3  # 4G/cpu x 16 cpus

    def test_min_memory_node_is_scaled_by_node_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_mock(monkeypatch)
        rec = (
            "JobId=8 JobName=t\n   JobState=PENDING Reason=Resources\n"
            "   Partition=cpu NumNodes=4 NumCPUs=32\n"
            "   TRES=cpu=32,node=4,billing=32\n"
            "   MinMemoryNode=32G\n"
        )
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: rec)
        job = resolve_pending_job("8")
        assert job.req_mem_bytes == 4 * 32 * 1024**3  # 32G/node x 4 nodes

    def test_running_job_raises_not_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_mock(monkeypatch)
        rec = _PENDING_RECORD.replace("JobState=PENDING", "JobState=RUNNING")
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: rec)
        with pytest.raises(JobNotPendingError):
            resolve_pending_job("12345")

    def test_missing_job_raises_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_mock(monkeypatch)

        def _boom(cmd: list[str]) -> str:
            raise SlurmCommandError("Invalid job id specified")

        monkeypatch.setattr(pending, "_run_slurm_cmd", _boom)
        with pytest.raises(JobNotFoundError):
            resolve_pending_job("99999")

    def test_transient_failure_is_retryable_not_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A busy/unreachable controller (timeout, NOT "invalid job id") must not be
        # reported as JobNotFoundError — that made the pending view falsely announce
        # the job "started" and freeze. Re-raise the (retryable) SlurmCommandError.
        self._no_mock(monkeypatch)

        def _boom(cmd: list[str]) -> str:
            raise SlurmCommandError("Command scontrol show job 12345 timed out after 15s")

        monkeypatch.setattr(pending, "_run_slurm_cmd", _boom)
        with pytest.raises(SlurmCommandError) as ei:
            resolve_pending_job("12345")
        assert not isinstance(ei.value, JobNotFoundError)

    def test_array_prefers_the_pending_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_mock(monkeypatch)
        running = _PENDING_RECORD.replace("JobState=PENDING", "JobState=RUNNING")
        both = running + "\n\n" + _PENDING_RECORD
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: both)
        job = resolve_pending_job("12345")
        assert job.reason == "Resources"  # the pending record was selected


class TestResolveClusterPartitions:
    def test_aggregates_sinfo_states(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        # 8-field format: %R|%a|%D|%t|%C|%G|%l|%m (mem MB is the last column).
        sinfo = (
            "gpu-a100|up|3|idle|0/96/0/96|gpu:a100:8|12:00:00|257000\n"
            "gpu-a100|up|1|mix|16/16/0/32|gpu:a100:8|12:00:00|257000\n"
            "gpu-a100|up|4|alloc|128/0/0/128|gpu:a100:8|12:00:00|257000\n"
            "cpu|up|10|idle|0/320/0/320|(null)|1-00:00:00|192000\n"
        )
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: sinfo)
        parts = {p.name: p for p in resolve_cluster_partitions("gpu-a100")}
        a = parts["gpu-a100"]
        assert a.total_nodes == 8
        assert a.idle_nodes == 3 and a.mix_nodes == 1
        assert a.free_nodes == 4
        assert a.cpus_idle == 112 and a.cpus_total == 256  # 96+16 idle, 96+32+128 total
        assert a.gpu_types == ["a100"]
        assert a.has_gpus is True
        assert a.max_node_gpus == 8  # gpu:a100:8 -> 8 GPUs/node (M4)
        assert a.idle_node_cpus == 96  # only the fully-idle nodes' cores, not mix (M5)
        assert a.max_node_mem_bytes == 257000 * 1024**2
        assert a.is_current is True
        assert a.timelimit_seconds == 12 * 3600
        assert parts["cpu"].gpu_types == [] and parts["cpu"].has_gpus is False
        assert parts["cpu"].is_current is False

    def test_heterogeneous_partition_uses_largest_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # F1/F4: sinfo runs with -e (heterogeneous nodes on separate lines) and -a
        # (hidden partitions). The per-node max must be the LARGER config.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        captured: dict[str, list[str]] = {}
        sinfo = (
            # %R|%a|%D|%t|%C|%G|%l|%m|%c
            "caslake|up|10|idle|0/480/0/480|(null)|1-00:00:00|192000|48\n"
            "caslake|up|4|idle|0/256/0/256|(null)|1-00:00:00|256000|64\n"
        )

        def _cmd(cmd: list[str]) -> str:
            captured["cmd"] = cmd
            return sinfo

        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)
        p = resolve_cluster_partitions("caslake")[0]
        assert p.max_node_cpus == 64  # the 64-core config, not the 48-core one
        assert p.max_node_mem_bytes == 256000 * 1024**2
        assert "-e" in captured["cmd"] and "-a" in captured["cmd"]

    def test_reserved_and_flagged_nodes_excluded_from_free_capacity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # F2/F3: a RESERVED (resv) node and a maintenance-flagged (idle$) node report
        # idle cores in %C but can't take a normal job, so they must not inflate free
        # cores or the per-node max — else the partition reads a false "FITS NOW".
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        sinfo = (
            "p|up|1|idle|0/16/0/16|(null)|1-00:00:00|64000|16\n"  # schedulable
            "p|up|1|resv|0/128/0/128|(null)|1-00:00:00|512000|128\n"  # reserved: out
            "p|up|1|idle$|0/64/0/64|(null)|1-00:00:00|256000|64\n"  # maint flag: out
        )
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: sinfo)
        p = resolve_cluster_partitions("p")[0]
        assert p.cpus_idle == 16  # only the schedulable idle node's cores
        assert p.max_node_cpus == 16  # not the 128-core reserved node
        assert p.max_node_mem_bytes == 64000 * 1024**2
        assert p.total_nodes == 3  # totals still count every node

    def test_untyped_gpu_partition_is_marked_has_gpus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #60 review (high): many clusters report GPUs untyped as `gpu:4`; the
        # partition must still register as having GPUs so GPU jobs aren't hidden.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(
            pending,
            "_run_slurm_cmd",
            lambda cmd: "gpu|up|11|mix|276/252/0/528|gpu:4|infinite|515000\n",
        )
        p = resolve_cluster_partitions("gpu")[0]
        assert p.has_gpus is True
        assert p.gpu_types == []  # untyped: no model reported
        assert p.cpus_idle == 252

    def test_multi_partition_current_is_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # audit-3 #4: `sbatch -p a,b` -> Partition="partA,partB". EVERY listed
        # partition must be flagged current (so none is dropped from the table),
        # not compared as a single "parta,partb" string that matches nothing.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        sinfo = (
            "partA|up|2|idle|0/64/0/64|(null)|1:00:00|192000\n"
            "partB|up|2|idle|0/64/0/64|(null)|1:00:00|192000\n"
            "partC|up|2|idle|0/64/0/64|(null)|1:00:00|192000\n"
        )
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: sinfo)
        parts = {p.name: p for p in resolve_cluster_partitions("partA,partB")}
        assert parts["partA"].is_current and parts["partB"].is_current
        assert parts["partC"].is_current is False

    def test_unavailable_when_sinfo_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)

        def _boom(cmd: list[str]) -> str:
            raise SlurmCommandError("down")

        monkeypatch.setattr(pending, "_run_slurm_cmd", _boom)
        assert resolve_cluster_partitions("x") == []

    _SINFO = (
        "caslake|up|10|idle|0/480/0/480|(null)|1-00:00:00|192000\n"
        "test|up|5|idle|0/240/0/240|(null)|1-00:00:00|192000\n"
        "pi-secret|up|4|idle|0/192/0/192|(null)|1-00:00:00|192000\n"
    )
    _PARTS = (
        "PartitionName=caslake AllowGroups=ALL AllowAccounts=ALL AllowQos=caslake\n"
        "PartitionName=test AllowGroups=ALL AllowAccounts=rcc-staff AllowQos=test\n"
        "PartitionName=pi-secret AllowGroups=ALL AllowAccounts=pi-secret AllowQos=x\n"
    )

    def _dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)

        def fake(cmd: list[str]) -> str:
            if cmd[0] == "sinfo":
                return self._SINFO
            if cmd[0] == "scontrol":
                return self._PARTS
            raise SlurmCommandError("?")

        monkeypatch.setattr(pending, "_run_slurm_cmd", fake)

    def test_filters_partitions_the_account_cannot_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._dispatch(monkeypatch)
        names = {p.name for p in resolve_cluster_partitions("caslake", "rcc-staff")}
        assert names == {"caslake", "test"}  # ALL + rcc-staff allowed; pi-secret dropped

    def test_no_account_shows_all_partitions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._dispatch(monkeypatch)
        # Without an account we can't filter, so nothing is hidden (private included).
        names = {p.name for p in resolve_cluster_partitions("caslake")}
        assert "pi-secret" in names

    def test_current_partition_is_always_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._dispatch(monkeypatch)
        # Even if the account check would exclude it, the job's own partition stays.
        names = {p.name for p in resolve_cluster_partitions("pi-secret", "rcc-staff")}
        assert "pi-secret" in names

    def test_non_responding_nodes_are_not_counted_as_free(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A '*' (NOT RESPONDING) flag means the node takes no new work; it must not
        # inflate free-node/idle-core counts (which fed a false "FITS NOW").
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        sinfo = (
            "gpu|up|4|idle*|0/64/0/64|gpu:v100:2|1-00:00:00|192000\n"
            "gpu|up|2|idle|0/32/0/32|gpu:v100:2|1-00:00:00|192000\n"
        )
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: sinfo)
        p = resolve_cluster_partitions("gpu")[0]
        assert p.idle_nodes == 2 and p.cpus_idle == 32  # the 4 idle* nodes excluded
        assert p.total_nodes == 6  # total capacity still counts them

    def test_group_restricted_partition_excluded_when_user_not_in_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        sinfo = (
            "open|up|4|idle|0/64/0/64|(null)|1-00:00:00|192000\n"
            "labonly|up|4|idle|0/64/0/64|(null)|1-00:00:00|192000\n"
        )
        parts = (
            "PartitionName=open AllowGroups=ALL AllowAccounts=ALL\n"
            "PartitionName=labonly AllowGroups=pilab AllowAccounts=ALL\n"
        )
        monkeypatch.setattr(
            pending, "_run_slurm_cmd", lambda cmd: sinfo if cmd[0] == "sinfo" else parts
        )
        monkeypatch.setattr(pending, "_user_groups", lambda u: {"other"})  # not in pilab
        names = {p.name for p in resolve_cluster_partitions("open", "acct", "someone")}
        assert names == {"open"}  # labonly (AllowGroups=pilab) hidden
        # A member of pilab sees it.
        monkeypatch.setattr(pending, "_user_groups", lambda u: {"pilab"})
        names = {p.name for p in resolve_cluster_partitions("open", "acct", "member")}
        assert names == {"open", "labonly"}

    def test_zero_accessible_shows_only_current_not_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A job whose account reaches NO listed partition (besides its current one)
        # must show only its current one — not fall back to showing ALL, which leaked
        # private per-PI partitions ("ok or None" collapsed empty -> None).
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        sinfo = (
            "mypart|up|4|idle|0/64/0/64|(null)|1-00:00:00|192000\n"
            "pi-a|up|4|idle|0/64/0/64|(null)|1-00:00:00|192000\n"
            "pi-b|up|4|idle|0/64/0/64|(null)|1-00:00:00|192000\n"
        )
        parts = (
            "PartitionName=mypart AllowGroups=ALL AllowAccounts=other-acct\n"
            "PartitionName=pi-a AllowGroups=ALL AllowAccounts=pi-a\n"
            "PartitionName=pi-b AllowGroups=ALL AllowAccounts=pi-b\n"
        )
        monkeypatch.setattr(
            pending, "_run_slurm_cmd", lambda cmd: sinfo if cmd[0] == "sinfo" else parts
        )
        monkeypatch.setattr(pending, "_user_groups", lambda u: {"grp"})
        names = {p.name for p in resolve_cluster_partitions("mypart", "myacct", "someone")}
        assert names == {"mypart"}  # only current; pi-a / pi-b not leaked


class TestPartitionFits:
    def _job(self, **kw: object) -> PendingJob:
        base: dict[str, object] = {
            "job_id": "1",
            "raw_job_id": "1",
            "name": "j",
            "username": "u",
            "partition": "p",
            "qos": "",
            "account": "",
            "reason": "Resources",
            "submit_time": None,
            "start_time_estimate": None,
            "priority": None,
            "req_cpus": 8,
            "req_nodes": 1,
            "req_mem_bytes": 0,
            "req_gpus": 0,
            "req_gpu_type": "",
            "time_limit_seconds": None,
        }
        base.update(kw)
        return PendingJob(**base)  # type: ignore[arg-type]

    def test_cpu_job_fits_idle_partition(self) -> None:
        p = PartitionResources("p", True, idle_nodes=2, cpus_idle=64)
        assert partition_fits_now(self._job(req_cpus=8), p) is True

    def test_not_enough_cpus(self) -> None:
        p = PartitionResources("p", True, idle_nodes=2, cpus_idle=4)
        assert partition_fits_now(self._job(req_cpus=8), p) is False

    def test_down_partition_never_fits(self) -> None:
        p = PartitionResources("p", False, idle_nodes=9, cpus_idle=999)
        assert partition_fits_now(self._job(req_cpus=1), p) is False

    def test_gpu_type_must_match(self) -> None:
        p = PartitionResources("p", True, idle_nodes=2, cpus_idle=64, gpu_types=["h100"])
        j = self._job(req_gpus=2, req_gpu_type="a100")
        assert partition_fits_now(j, p) is False
        p.gpu_types = ["a100", "h100"]
        assert partition_fits_now(j, p) is True

    def test_gpu_job_needs_some_gpu(self) -> None:
        p = PartitionResources("p", True, idle_nodes=2, cpus_idle=64, gpu_types=[])
        assert partition_fits_now(self._job(req_gpus=1), p) is False

    def test_untyped_gpu_partition_fits_a_gpu_job(self) -> None:
        # #60 review (high): a partition with GPUs but no reported model (has_gpus
        # True, gpu_types []) must still be considered for a GPU job — not hidden.
        p = PartitionResources("p", True, idle_nodes=3, cpus_idle=96, has_gpus=True)
        assert partition_fits_now(self._job(req_gpus=1), p) is True
        # A specific type request can't be excluded when the partition is untyped.
        assert partition_fits_now(self._job(req_gpus=1, req_gpu_type="a100"), p) is True

    def test_memory_that_no_node_can_hold_does_not_fit(self) -> None:
        # #60 review: a per-node memory request larger than the biggest node is a
        # hard no — don't recommend requeuing there.
        p = PartitionResources(
            "p", True, idle_nodes=4, cpus_idle=256, max_node_mem_bytes=128 * 1024**3
        )
        assert partition_fits_now(self._job(req_cpus=8, req_mem_bytes=900 * 1024**3), p) is False
        assert partition_fits_now(self._job(req_cpus=8, req_mem_bytes=64 * 1024**3), p) is True

    def test_memory_unknown_is_not_rejected(self) -> None:
        # When node memory is unknown (max_node_mem_bytes=0) we can't reject on it.
        p = PartitionResources("p", True, idle_nodes=4, cpus_idle=256, max_node_mem_bytes=0)
        assert partition_fits_now(self._job(req_cpus=8, req_mem_bytes=900 * 1024**3), p) is True

    def test_exclusive_job_needs_fully_idle_nodes(self) -> None:
        # A partition with 1 empty + 5 partially-used nodes: a normal 2-node job fits
        # (free_nodes=6), but an --exclusive one needs 2 fully-empty nodes (idle=1).
        p = PartitionResources("p", True, idle_nodes=1, mix_nodes=5, cpus_idle=256)
        assert partition_fits_now(self._job(req_nodes=2), p) is True
        assert partition_fits_now(self._job(req_nodes=2, exclusive=True), p) is False
        # Enough empty nodes → the exclusive job fits.
        p2 = PartitionResources("p", True, idle_nodes=2, mix_nodes=5, cpus_idle=256)
        assert partition_fits_now(self._job(req_nodes=2, exclusive=True), p2) is True

    def test_gpu_job_needs_a_fully_idle_node(self) -> None:
        # All GPUs busy on mixed nodes (no empty node): sinfo can't show idle-GPU
        # count, so a GPU request must NOT be judged to fit there.
        busy = PartitionResources("g", True, idle_nodes=0, mix_nodes=4, cpus_idle=40, has_gpus=True)
        assert partition_fits_now(self._job(req_gpus=1), busy) is False
        free = PartitionResources("g", True, idle_nodes=2, mix_nodes=4, cpus_idle=40, has_gpus=True)
        assert partition_fits_now(self._job(req_gpus=1), free) is True

    def test_partition_time_limit_shorter_than_job_does_not_fit(self) -> None:
        p = PartitionResources("p", True, idle_nodes=4, cpus_idle=256, timelimit_seconds=3600)
        assert partition_fits_now(self._job(time_limit_seconds=7200), p) is False
        assert partition_fits_now(self._job(time_limit_seconds=1800), p) is True
        # Unknown limits can't reject.
        q = PartitionResources("q", True, idle_nodes=4, cpus_idle=256, timelimit_seconds=None)
        assert partition_fits_now(self._job(time_limit_seconds=999999), q) is True

    def test_per_node_cpu_request_larger_than_any_node_does_not_fit(self) -> None:
        # 4 idle nodes of 4 cores each = 16 idle cluster-wide, but a 1-node 16-CPU
        # job needs 16 cores on ONE node — no node has that, so it must not fit.
        p = PartitionResources("p", True, idle_nodes=4, cpus_idle=16, max_node_cpus=4)
        assert partition_fits_now(self._job(req_nodes=1, req_cpus=16), p) is False
        assert partition_fits_now(self._job(req_nodes=1, req_cpus=4), p) is True
        # Unknown per-node size can't reject.
        q = PartitionResources("q", True, idle_nodes=4, cpus_idle=16, max_node_cpus=0)
        assert partition_fits_now(self._job(req_nodes=1, req_cpus=16), q) is True


class TestFitBlocker:
    def _job(self, **kw: object) -> PendingJob:
        base: dict[str, object] = {
            "job_id": "1",
            "raw_job_id": "1",
            "name": "j",
            "username": "u",
            "partition": "p",
            "qos": "",
            "account": "",
            "reason": "Resources",
            "submit_time": None,
            "start_time_estimate": None,
            "priority": None,
            "req_cpus": 4,
            "req_nodes": 1,
            "req_mem_bytes": 0,
            "req_gpus": 0,
            "req_gpu_type": "",
            "time_limit_seconds": None,
        }
        base.update(kw)
        return PendingJob(**base)  # type: ignore[arg-type]

    def test_reports_specific_blocker(self) -> None:
        big = PartitionResources("p", True, idle_nodes=4, cpus_idle=256, max_node_mem_bytes=0)
        assert pending.fit_blocker(self._job(), big) == ""  # fits
        assert pending.fit_blocker(self._job(), PartitionResources("p", False)) == "down"
        assert pending.fit_blocker(self._job(req_cpus=999), big) == "no room"
        # GPU job on a GPU-less partition -> "no GPU", not "no room".
        assert pending.fit_blocker(self._job(req_gpus=1), big) == "no GPU"
        # Wrong GPU type.
        gpu = PartitionResources(
            "p", True, idle_nodes=4, cpus_idle=256, gpu_types=["v100"], has_gpus=True
        )
        assert pending.fit_blocker(self._job(req_gpus=1, req_gpu_type="a100"), gpu) == "no a100"
        # Walltime too long.
        short = PartitionResources("p", True, idle_nodes=4, cpus_idle=256, timelimit_seconds=60)
        assert pending.fit_blocker(self._job(time_limit_seconds=999), short) == "time limit"
        # Per-node too small.
        small = PartitionResources("p", True, idle_nodes=4, cpus_idle=256, max_node_cpus=2)
        assert pending.fit_blocker(self._job(req_cpus=8), small) == "node too small"

    def test_gpu_count_per_node_blocks(self) -> None:
        def _gpu_part(n_gpus: int) -> PartitionResources:
            return PartitionResources(
                "g",
                True,
                idle_nodes=2,
                cpus_idle=64,
                idle_node_cpus=64,
                max_idle_node_cpus=32,
                has_gpus=True,
                max_node_gpus=n_gpus,
            )

        # M4: 8 GPUs/node on a partition of 4-GPU nodes must not "fit now" — it would
        # sit PENDING forever after the recommended requeue.
        assert fit_blocker(self._job(req_gpus=8, req_nodes=1), _gpu_part(4)) == "too few GPUs"
        assert fit_blocker(self._job(req_gpus=8, req_nodes=1), _gpu_part(8)) == ""  # fits
        assert fit_blocker(self._job(req_gpus=8, req_nodes=1), _gpu_part(0)) == ""  # count unknown
        # A multi-node request spreads the per-node share: 8 GPUs over 2 nodes = 4/node.
        assert fit_blocker(self._job(req_gpus=8, req_nodes=2), _gpu_part(4)) == ""

    def test_exclusive_job_uses_idle_only_capacity(self) -> None:
        # M5: one idle 16-core node + busy 64-core mix nodes. cpus_idle/max_node_cpus
        # are inflated by the mix nodes, but an exclusive job needs a WHOLE idle node —
        # measure it against idle-only capacity, or it falsely "fits now".
        tight = PartitionResources(
            "p",
            True,
            idle_nodes=1,
            mix_nodes=3,
            cpus_idle=200,  # inflated by free cores on busy mix nodes
            max_node_cpus=64,  # a busy 64-core node
            idle_node_cpus=16,
            max_idle_node_cpus=16,  # the only fully-idle node is 16-core
        )
        j = self._job(req_cpus=48, req_nodes=1, exclusive=True)
        assert fit_blocker(j, tight) in ("no room", "node too small")  # NOT "" (false fit)
        big = PartitionResources(
            "p",
            True,
            idle_nodes=1,
            mix_nodes=3,
            cpus_idle=200,
            max_node_cpus=64,
            idle_node_cpus=64,
            max_idle_node_cpus=64,  # a fully-idle 64-core node
        )
        assert fit_blocker(self._job(req_cpus=48, req_nodes=1, exclusive=True), big) == ""


class TestRequeueCouldHelp:
    @pytest.mark.parametrize("reason", ["Resources", "Priority", "", "None", "QOSMaxCpuPerJob"])
    def test_capacity_reasons_allow_requeue(self, reason: str) -> None:
        assert pending.requeue_could_help(reason) is True

    @pytest.mark.parametrize(
        "reason",
        [
            "Dependency",
            "DependencyNeverSatisfied",
            "JobHeldUser",
            "JobHeldAdmin",
            "BeginTime",
            "Reservation",
            "AssocGrpCpuLimit",
            "AssocMaxWallDurationPerJobLimit",
        ],
    )
    def test_non_capacity_reasons_block_requeue(self, reason: str) -> None:
        assert pending.requeue_could_help(reason) is False


class TestIsHeldLike:
    @pytest.mark.parametrize("reason", ["JobHeldUser", "Dependency", "BeginTime", "Reservation"])
    def test_blocked_reasons(self, reason: str) -> None:
        assert pending.is_held_like(reason) is True

    @pytest.mark.parametrize("reason", ["Resources", "Priority", "", "AssocGrpCpuLimit"])
    def test_scheduled_reasons(self, reason: str) -> None:
        # Assoc-limited jobs are still priority-ordered → NOT held-like.
        assert pending.is_held_like(reason) is False


class TestFormatGpuTypes:
    def test_placeholder_when_empty(self) -> None:
        assert pending.format_gpu_types([], 12) == "—"
        assert pending.format_gpu_types([], 12, ascii_mode=True) == "-"

    def test_untyped_but_present_shows_GPU(self) -> None:
        # Untyped gpu:N partition (has_gpus, no models) shows "GPU", not the no-GPU
        # placeholder, so a partition recommended for a GPU job doesn't look empty.
        assert pending.format_gpu_types([], 12, has_gpus=True) == "GPU"

    def test_truncates_on_whole_items_no_dangling_comma(self) -> None:
        # 'a100, v100, h100' is 16 > 12; keep whole items + ellipsis, never 'v100, '.
        out = pending.format_gpu_types(["a100", "v100", "h100"], 12)
        assert out == "a100, v100…" and ", …" not in out and not out.endswith(", ")

    def test_ascii_ellipsis(self) -> None:
        # width 13 leaves room for the 3-char "..." after "a100, v100" (10).
        out = pending.format_gpu_types(["a100", "v100", "h100"], 13, ascii_mode=True)
        out.encode("ascii")  # no unicode ellipsis under ascii
        assert out.endswith("...")

    def test_never_silently_drops(self) -> None:
        # width 12 ascii: the 3-char "..." doesn't fit after "a100, v100" (10+3>12),
        # so an item is dropped to make room — truncation is never silent/indicator-less.
        out = pending.format_gpu_types(["a100", "v100", "h100"], 12, ascii_mode=True)
        assert out.endswith("...") and "v100" not in out


class TestPriorityRank:
    def test_rank_from_squeue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        # Pending priorities on the partition; ours is 500 → 2 ahead → rank 3 of 5.
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: "900\n700\n500\n300\n100\n")
        assert resolve_priority_rank("p", 500) == (3, 5)

    def test_rank_multipartition_returns_best(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # sbatch -p a,b: rank is computed per partition (priorities compare only
        # within one) and the BEST position — the queue the job starts from first —
        # is returned, not a pooled mix of non-comparable priorities (P4).
        def _fake(cmd: list[str]) -> str:
            part = cmd[cmd.index("-p") + 1]
            # In 'a' the job (prio 500) is behind 3; in 'b' behind only 1.
            return "900\n800\n700\n500\n" if part == "a" else "600\n500\n"

        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_run_slurm_cmd", _fake)
        # a: 3 ahead -> #4 of 4; b: 1 ahead -> #2 of 2. Best (nearest the front) = (2, 2).
        assert resolve_priority_rank("a,b", 500) == (2, 2)

    def test_rank_none_when_priority_unknown(self) -> None:
        assert resolve_priority_rank("p", None) is None

    def test_rank_none_when_squeue_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)

        def _boom(cmd: list[str]) -> str:
            raise SlurmCommandError("down")

        monkeypatch.setattr(pending, "_run_slurm_cmd", _boom)
        assert resolve_priority_rank("p", 500) is None

    def test_rank_clamped_when_job_absent_from_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the job left the PD set between the scontrol read and this squeue
        # snapshot, its priority isn't in `prios`; the rank must never exceed the
        # total (no impossible "#4 of 3").
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: "900\n700\n600\n")
        rank = resolve_priority_rank("p", 100)  # below all three, absent from the set
        assert rank is not None
        n, total = rank
        assert total == 3
        assert n <= total


class TestQueueCounts:
    def test_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        out = "1|RUNNING\n2|RUNNING\n3|PENDING\n4|COMPLETING\n5|PENDING\n6|SUSPENDED\n"
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: out)
        # 2 RUNNING + 1 COMPLETING running; 2 PENDING + 1 SUSPENDED pending.
        assert resolve_queue_counts("p") == (3, 3)

    def test_counts_dedupes_multipartition_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A job pending in several partitions (sbatch -p a,b) is listed once PER
        # partition by squeue; dedup by job id counts it once, not twice (P4).
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        out = "7|PENDING\n7|PENDING\n8|RUNNING\n"
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: out)
        assert resolve_queue_counts("a,b") == (1, 1)

    def test_unavailable_is_none_not_fabricated_zeros(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A busy/unreachable controller must yield None ("unavailable"), never a
        # fabricated (0, 0) that reads as a genuinely empty partition.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)

        def _boom(cmd: list[str]) -> str:
            raise SlurmCommandError("Socket timed out on send/recv operation")

        monkeypatch.setattr(pending, "_run_slurm_cmd", _boom)
        assert resolve_queue_counts("p") is None


class TestMockData:
    def test_mock_tells_a_where_could_it_run_story(self) -> None:
        # The demo data is self-consistent: the current partition is full but an
        # alternative fits — so the feature's payoff is visible offline.
        job = pending._mock_pending_job("777")
        parts = pending._mock_partitions(job.partition)
        current = next(p for p in parts if p.is_current)
        alts = [p for p in parts if partition_fits_now(job, p) and not p.is_current]
        assert partition_fits_now(job, current) is False
        assert any(p.name == "gpu-a100" for p in alts)


# ---------------------------------------------------------------------------
# CLI routing + text summary
# ---------------------------------------------------------------------------


class TestCliRouting:
    def test_resolve_routes_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _running_raises(job_id: str) -> object:
            raise JobNotRunningError("Job 1 is in state 'PENDING'.")

        pend = pending._mock_pending_job("1")
        monkeypatch.setattr(cli, "resolve_job_context", _running_raises)
        monkeypatch.setattr(cli, "resolve_pending_job", lambda job_id: pend)
        ctx, got = cli._resolve_running_or_pending("1")
        assert ctx is None and got is pend

    def test_completed_job_still_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A genuinely non-runnable state (not pending) keeps the clean exit(1).
        def _running_raises(job_id: str) -> object:
            raise JobNotRunningError("Job 1 is in state 'COMPLETED'.")

        def _not_pending(job_id: str) -> object:
            raise JobNotPendingError("not pending")

        monkeypatch.setattr(cli, "resolve_job_context", _running_raises)
        monkeypatch.setattr(cli, "resolve_pending_job", _not_pending)
        with pytest.raises(SystemExit) as exc:
            cli._resolve_running_or_pending("1")
        assert exc.value.code == 1

    def test_once_pending_reports_on_stderr_and_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # #60 review: --once is machine-oriented, so a queued job must keep stdout
        # clean (no prose for a jq/CSV reader) — report on STDERR and exit 1.
        def _running_raises(job_id: str) -> object:
            raise JobNotRunningError("Job 777 is in state 'PENDING'.")

        monkeypatch.setattr(cli, "resolve_job_context", _running_raises)
        monkeypatch.setattr(cli, "resolve_pending_job", pending._mock_pending_job)
        monkeypatch.setattr(
            cli, "resolve_cluster_partitions", lambda p, a="", u="": pending._mock_partitions(p)
        )
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda p: (12, 5))
        with pytest.raises(SystemExit) as exc:
            cli._run_once("777", SlurmwatchConfig())
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""  # stdout stays clean for machine consumers
        assert "PENDING" in captured.err and "Why" in captured.err and "Where" in captured.err
        assert "gpu-a100" in captured.err and "FITS NOW" in captured.err
        assert "scontrol update JobId=777 Partition=gpu-a100" in captured.err

    def test_demo_pending_sentinel_routes_to_pending_view(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #60 review: `slurmwatch --demo pending` must reach the pending view even
        # though mock resolve_job_context always returns a RUNNING job.
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        ctx, pend = cli._resolve_running_or_pending("pending")
        assert ctx is None and pend is not None and pend.reason == "Resources"

    def test_headless_pending_writes_no_log(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from pathlib import Path

        def _running_raises(job_id: str) -> object:
            raise JobNotRunningError("Job 777 is in state 'PENDING'.")

        monkeypatch.setattr(cli, "resolve_job_context", _running_raises)
        monkeypatch.setattr(cli, "resolve_pending_job", pending._mock_pending_job)
        monkeypatch.setattr(
            cli, "resolve_cluster_partitions", lambda p, a="", u="": pending._mock_partitions(p)
        )
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda p: (0, 0))
        log = Path(str(tmp_path)) / "out.csv"
        cli._run_headless("777", SlurmwatchConfig(), str(log))
        assert not log.exists()  # nothing logged for a queued job
        err = capsys.readouterr().err
        assert "PENDING" in err and "nothing to log yet" in err


# ---------------------------------------------------------------------------
# TUI view
# ---------------------------------------------------------------------------


class TestPendingTui:
    def _view(self) -> object:
        from slurmwatch.tui import PendingView

        job = pending._mock_pending_job("777")
        v = PendingView()
        v.job = job
        v.partitions = pending._mock_partitions(job.partition)
        v.queue_running, v.queue_pending = 12, 5
        v.config = SlurmwatchConfig()
        return v

    def test_render_has_all_sections_and_tip(self) -> None:
        plain = Text.from_markup(self._view().render()).plain  # type: ignore[attr-defined]
        assert "Why It's Waiting" in plain
        assert "When It Might Start" in plain
        assert "Where It Could Run" in plain
        assert "PENDING" in plain and "Resources" in plain
        assert "estimated start" in plain
        # The request lives in the WHY section now (to read against WHERE capacity).
        assert "requested" in plain and "16 CPU" in plain
        assert "gpu-a100" in plain and "YES" in plain
        assert "scontrol update JobId=777 Partition=gpu-a100" in plain

    def test_render_no_data(self) -> None:
        from slurmwatch.tui import PendingView

        assert "resolving" in PendingView().render()

    def test_render_is_pure_ascii_under_ascii_mode(self) -> None:
        # --ascii exists for non-UTF-8 terminals; every glyph in the pending view
        # (separators, dashes, spinner, gpu placeholder/ellipsis) must be ASCII.
        from slurmwatch.tui import PendingView

        now = time.time()
        job = PendingJob(
            job_id="1",
            raw_job_id="1",
            name="j",
            username="u",
            partition="cur",
            qos="q",
            account="a",
            reason="Dependency",
            submit_time=now - 60,
            start_time_estimate=None,
            priority=100,
            req_cpus=8,
            req_nodes=2,
            req_mem_bytes=8 * 1024**3,
            req_gpus=1,
            req_gpu_type="a100",
            time_limit_seconds=3600,
            exclusive=True,
        )
        v = PendingView()
        v.job = job
        v.queue_rank = (3, 9)
        v.config = SlurmwatchConfig(ascii_mode=True)
        v.partitions = [
            PartitionResources(
                "cur", True, idle_nodes=0, mix_nodes=2, cpus_idle=8, is_current=True
            ),
            PartitionResources(
                "big",
                True,
                idle_nodes=5,
                cpus_idle=99,
                gpu_types=["a100", "v100", "h100"],
                has_gpus=True,
            ),
        ]
        v.render().encode("ascii")  # raises UnicodeEncodeError if any glyph leaked

    def test_where_escapes_gpu_type_with_bracket(self) -> None:
        # Completeness #3: a GPU type string containing '[' must be escaped before it
        # reaches Textual's markup parser, or PendingView.render() crashes.
        from slurmwatch.tui import PendingView

        job = pending._mock_pending_job("777")
        job.req_gpus = 1
        v = PendingView()
        v.job = job
        v.config = SlurmwatchConfig()
        v.partitions = [
            PartitionResources(
                "gpu", True, idle_nodes=2, cpus_idle=8, gpu_types=["a[100"], has_gpus=True
            ),
        ]
        Text.from_markup(v.render())  # MarkupError here if the '[' wasn't escaped

    def test_calculating_shown_when_no_estimate(self) -> None:
        from slurmwatch.tui import PendingView

        job = pending._mock_pending_job("777")
        job.start_time_estimate = None
        v = PendingView()
        v.job = job
        v.config = SlurmwatchConfig()
        plain = Text.from_markup(v.render()).plain
        assert "calculating" in plain and "not yet estimated" not in plain

    def test_slightly_past_estimate_reads_imminent_not_calculating(self) -> None:
        # Backfill stamps StartTime at its last cycle, so an imminent job's estimate
        # is often a few seconds in the past — must read "imminent", not "calculating".
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = pending._mock_pending_job("777")
        v.job.start_time_estimate = time.time() - 30  # 30s in the past
        v.config = SlurmwatchConfig()
        plain = Text.from_markup(v._when(v.job, False)).plain
        assert "imminent" in plain and "calculating" not in plain

    def test_priority_rank_is_displayed(self) -> None:
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = pending._mock_pending_job("777")
        v.queue_rank = (3, 5)
        v.config = SlurmwatchConfig()
        plain = Text.from_markup(v.render()).plain
        assert "#3 of 5" in plain and "2 higher-priority jobs ahead of yours" in plain

    def test_current_partition_never_shows_fits_now(self) -> None:
        # The current partition is where the job is PENDING, so even with abundant
        # capacity it must read "waiting", never a self-contradictory "YES/fits now".
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = pending._mock_pending_job("777")
        v.config = SlurmwatchConfig()
        v.partitions = [
            PartitionResources("mypart", True, idle_nodes=50, cpus_idle=9999, is_current=True)
        ]
        lines = Text.from_markup(v.render()).plain.splitlines()
        cur = next(ln for ln in lines if "(current)" in ln)
        assert "waiting" in cur and "YES" not in cur

    def test_where_columns_align_across_magnitudes(self) -> None:
        # Right-aligned numeric columns keep the status marker in line whether a row
        # has 646 or 10458 idle cores (the misalignment the user hit).
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = pending._mock_pending_job("777")
        v.config = SlurmwatchConfig()
        # has_gpus so the mock (GPU) job fits and shows the YES marker to align on.
        v.partitions = [
            PartitionResources("small", True, idle_nodes=6, cpus_idle=646, has_gpus=True),
            PartitionResources("big", True, idle_nodes=300, cpus_idle=10458, has_gpus=True),
        ]
        yes_rows = [ln for ln in Text.from_markup(v.render()).plain.splitlines() if "YES" in ln]
        assert len(yes_rows) == 2
        assert len({ln.index("YES") for ln in yes_rows}) == 1  # marker aligned in every row

    @pytest.mark.asyncio
    async def test_pending_app_mounts_and_refreshes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from slurmwatch.tui import PendingApp, PendingView

        job = pending._mock_pending_job("777")
        monkeypatch.setattr("slurmwatch.tui.resolve_pending_job", lambda jid: job)
        monkeypatch.setattr(
            "slurmwatch.tui.resolve_cluster_partitions",
            lambda p, a="", u="": pending._mock_partitions(p),
        )
        monkeypatch.setattr("slurmwatch.tui.resolve_queue_counts", lambda p: (12, 5))
        app = PendingApp(job)
        async with app.run_test(size=(110, 40)) as pilot:
            for _ in range(30):
                await pilot.pause()
                await asyncio.sleep(0.03)
                if app.screen.query_one(PendingView).partitions:
                    break
            plain = Text.from_markup(app.screen.query_one(PendingView).render()).plain
            assert "Where It Could Run" in plain and "gpu-a100" in plain

    @pytest.mark.asyncio
    async def test_pending_screen_notes_when_job_no_longer_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the job leaves the queue (it started), the screen says so and stops.
        from textual.widgets import Static

        from slurmwatch.tui import PendingApp, PendingScreen

        job = pending._mock_pending_job("777")

        def _now_running(jid: str) -> object:
            raise JobNotPendingError("started")

        monkeypatch.setattr("slurmwatch.tui.resolve_pending_job", _now_running)
        app = PendingApp(job)
        async with app.run_test(size=(110, 40)) as pilot:
            scr = app.screen
            assert isinstance(scr, PendingScreen)
            for _ in range(30):
                await pilot.pause()
                await asyncio.sleep(0.03)
                if scr._done:
                    break
            assert scr._done is True
            # The notice is revealed (its exact rendered text is a Textual-version
            # detail; _done + display is the observable contract).
            assert scr.query_one("#pending-notice", Static).display is True
