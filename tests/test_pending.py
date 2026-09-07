from __future__ import annotations

import asyncio
import contextlib
import csv
import json
import time
from collections.abc import Callable
from typing import Any

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
    available_node_count,
    blocker_is_permanent,
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

        calls: list[list[str]] = []

        def _cmd(cmd: list[str]) -> str:
            calls.append(cmd)
            captured["cmd"] = cmd
            return sinfo

        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)
        p = resolve_cluster_partitions("caslake")[0]
        assert p.max_node_cpus == 64  # the 64-core config, not the 48-core one
        assert p.max_node_mem_bytes == 256000 * 1024**2
        # Two sinfo calls now: the aggregate capacity query and the node-centric
        # free-GPU one. The -e/-a assertion is about the aggregate query, so pick it
        # out by its `-o` format rather than assuming which ran last.
        aggregate = next(c for c in calls if "-o" in c)
        assert "-e" in aggregate and "-a" in aggregate

    def test_free_gpus_parsed_from_node_level_gres_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `sinfo -O` pads only UP TO each field's width, so the query asks for an
        # explicit "|" suffix per field and this fake mirrors that. Value shapes are
        # from a live Slurm 20.11 cluster: untyped `gpu:4`, typed `gpu:a30:4`, and the
        # `(IDX:...)` suffix GresUsed appends.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        aggregate = "gpu|up|3|mix|12/36/0/48|gpu:4|1-00:00:00|192000|48\n"
        nodes = (
            "gpu|mixed|gpu:4|gpu:1(IDX:0)|\n"
            "gpu|mixed|gpu:4|gpu:4(IDX:0-3)|\n"
            "gpu|idle|gpu:a30:4|gpu:a30:0(IDX:N/A)|\n"
            # Excluded: drained, and a non-GPU node.
            "gpu|drained|gpu:4|gpu:0(IDX:N/A)|\n"
            "cpu|idle|(null)|(null)|\n"
        )

        def _cmd(cmd: list[str]) -> str:
            return nodes if "-N" in cmd else aggregate

        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)
        p = resolve_cluster_partitions("gpu")[0]
        assert p.gpu_detail is True
        # 4-1=3, 4-4=0, 4-0=4 → 7 free; the drained node contributes nothing.
        assert sorted(p.free_gpus_per_node) == [0, 3, 4]
        assert p.gpus_free == 7
        assert p.max_node_gpus_free == 4

    def test_gpu_detail_absent_when_node_query_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An unsupported/failing GresUsed query must leave gpu_detail False so the
        # conservative idle-node fallback stays in force — never read as "0 free".
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        aggregate = "gpu|up|3|mix|12/36/0/48|gpu:4|1-00:00:00|192000|48\n"

        def _cmd(cmd: list[str]) -> str:
            if "-N" in cmd:
                raise SlurmCommandError("sinfo: invalid field")
            return aggregate

        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)
        p = resolve_cluster_partitions("gpu")[0]
        assert p.gpu_detail is False
        assert p.free_gpus_per_node == []
        assert p.gpus_free == 0  # "unknown", and gpu_detail is how callers tell

    def test_free_gpu_fields_are_delimited_not_width_separated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `sinfo -O` truncates each value to its field width and pads only UP TO it, so a
        # value within a char of the width leaves no separating run of spaces and merges
        # with the next field. Splitting on whitespace then produced 3 fields instead of
        # 4, GresUsed read as "", and a node with every GPU allocated was reported as
        # fully free. Verified against live sinfo: narrow widths print "testmixgpu:4 gpu:4".
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        aggregate = "gpu|up|1|mix|4/44/0/48|gpu:4|1-00:00:00|192000|48\n"
        captured: list[list[str]] = []
        # A GRES string long enough to fill the width, exactly the case that used to
        # swallow the GresUsed field beside it.
        long_gres = "gpu:a100:4(S:0-1),mps:a100:400(S:0-1),shard:a100:32,nvme:1600"
        nodes = f"gpu|mixed|{long_gres}|gpu:a100:4(IDX:0-3)|\n"

        def _cmd(cmd: list[str]) -> str:
            captured.append(cmd)
            return nodes if "-N" in cmd else aggregate

        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)
        p = resolve_cluster_partitions("gpu")[0]
        node_query = next(c for c in captured if "-N" in c)
        assert "|" in node_query[node_query.index("-O") + 1], "fields must be delimited"
        # All 4 GPUs allocated -> 0 free, NOT 4 free.
        assert p.free_gpus_per_node == [0]
        assert p.gpus_free == 0

    def test_free_gpus_skip_a_node_whose_gresused_is_unreadable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A GPU node whose GresUsed came back empty is a read we did not get. Treating it
        # as a genuine zero would donate the node's whole GPU count to the free pool and
        # advise a requeue onto capacity that cannot actually run the job.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        aggregate = "gpu|up|2|mix|4/44/0/48|gpu:4|1-00:00:00|192000|48\n"
        nodes = "gpu|mixed|gpu:4||\ngpu|idle|gpu:4|gpu:1(IDX:0)|\n"

        def _cmd(cmd: list[str]) -> str:
            return nodes if "-N" in cmd else aggregate

        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)
        p = resolve_cluster_partitions("gpu")[0]
        # Only the readable node contributes (4-1=3); the blank one is skipped, not 4.
        assert p.free_gpus_per_node == [3]

    def test_free_gpus_ignore_flagged_non_schedulable_nodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The `* $ % @ !` suffixes mark non-responding / reserved / powering-down /
        # pending-reboot / power-save nodes. `base = re.sub(r"[^a-z]", "", state)` strips
        # them BEFORE the idle/mix test, so the flag check is the only thing keeping them
        # out — and a mutation sweep showed deleting it left the whole suite green.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        aggregate = "gpu|up|3|mix|12/36/0/48|gpu:4|1-00:00:00|192000|48\n"
        nodes = (
            "gpu|idle|gpu:4|gpu:0(IDX:N/A)|\n"  # schedulable: 4 free
            "gpu|idle*|gpu:4|gpu:0(IDX:N/A)|\n"  # non-responding
            "gpu|mixed$|gpu:4|gpu:1(IDX:0)|\n"  # reserved / maintenance
        )

        def _cmd(cmd: list[str]) -> str:
            return nodes if "-N" in cmd else aggregate

        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)
        p = resolve_cluster_partitions("gpu")[0]
        assert p.free_gpus_per_node == [4], "only the unflagged node is schedulable"
        assert p.gpus_free == 4  # not 11

    def test_gpu_detail_is_a_known_zero_when_no_gpu_node_is_schedulable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A successful query that names no schedulable GPU node in this partition means
        # "no free GPUs" as a FACT. Marking it unknown instead made a GPU job fall back
        # to the partition's idle GPU-LESS node count, printing "FITS NOW" and a requeue
        # command for a partition with zero free GPUs.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        # 40 idle nodes, all of them CPU-only; the partition's GPU nodes are allocated.
        aggregate = "gpu|up|40|idle|0/1920/0/1920|gpu:4|1-00:00:00|192000|48\n"
        nodes = "gpu|allocated|gpu:4|gpu:4(IDX:0-3)|\n"

        def _cmd(cmd: list[str]) -> str:
            return nodes if "-N" in cmd else aggregate

        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)
        p = resolve_cluster_partitions("gpu")[0]
        assert p.gpu_detail is True, "the query worked, so zero is known"
        assert p.free_gpus_per_node == []
        job = PendingJob(
            job_id="1",
            raw_job_id="1",
            name="j",
            username="u",
            partition="gpu",
            qos="",
            account="",
            reason="Priority",
            submit_time=None,
            start_time_estimate=None,
            priority=None,
            req_cpus=1,
            req_nodes=1,
            req_mem_bytes=0,
            req_gpus=1,
            req_gpu_type="",
            time_limit_seconds=None,
        )
        assert fit_blocker(job, p) != "", "a GPU job must not be told this partition fits"
        assert partition_fits_now(job, p) is False

    def test_a_gresused_column_empty_on_every_gpu_node_is_unknown_not_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # sinfo does NOT fail on an `-O` field it does not know: measured on Slurm
        # 20.11.8, `-O "Partition:40|,StateLong:20|,Gres:60|,NoSuchField:60|"` exits 0,
        # puts "Invalid job format specification" on stderr only, and renders the
        # unknown column EMPTY — 1,246 rows of it. Every GPU node is then skipped as
        # "unreadable", the dict comes back empty, and `gpu_query_ok` is still True, so
        # the caller recorded a KNOWN zero: on this cluster that was "GPUs busy" for all
        # 20 GPU partitions at once while ~226 GPUs were free. Not one reading means the
        # FIELD is missing, so the answer is unknown and the idle-node fallback stands.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        aggregate = "gpu|up|4|idle|0/192/0/192|gpu:4|1-00:00:00|192000|48\n"
        nodes = "gpu|idle|gpu:4||\ngpu|mixed|gpu:4||\n"

        def _cmd(cmd: list[str]) -> str:
            return nodes if "-N" in cmd else aggregate

        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)
        p = resolve_cluster_partitions("gpu")[0]
        assert p.gpu_detail is False, "no GresUsed was readable at all -> unknown"
        assert p.free_gpus_per_node == []
        # …and the conservative fallback (fully-idle nodes) still lets a GPU job through
        # instead of the false cluster-wide "GPUs busy".
        job = self._gpu_job()
        assert fit_blocker(job, p) == ""

    def test_one_unreadable_node_beside_readable_ones_is_still_just_that_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Control for the test above: the "no readings at all" rule must not swallow the
        # per-node skip. With even ONE GresUsed value read, the query DID work, so the
        # partition keeps `gpu_detail` and the blank node simply contributes nothing —
        # exactly the behaviour test_free_gpus_skip_a_node_whose_gresused_is_unreadable
        # pins. Otherwise a single flaky node would discard the whole cluster's detail.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        aggregate = "gpu|up|4|idle|0/192/0/192|gpu:4|1-00:00:00|192000|48\n"
        nodes = "gpu|idle|gpu:4||\ngpu|mixed|gpu:4|gpu:1(IDX:0)|\n"

        def _cmd(cmd: list[str]) -> str:
            return nodes if "-N" in cmd else aggregate

        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)
        p = resolve_cluster_partitions("gpu")[0]
        assert p.gpu_detail is True, "one reading proves the field exists"
        assert p.free_gpus_per_node == [3]

    def _gpu_job(self) -> PendingJob:
        return PendingJob(
            job_id="1",
            raw_job_id="1",
            name="j",
            username="u",
            partition="gpu",
            qos="",
            account="",
            reason="Resources",
            submit_time=None,
            start_time_estimate=None,
            priority=None,
            req_cpus=1,
            req_nodes=1,
            req_mem_bytes=0,
            req_gpus=1,
            req_gpu_type="",
            time_limit_seconds=None,
        )

    def test_gres_used_larger_than_gres_clamps_to_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Defensive: a malformed pair must never yield a negative free count that
        # would make `nodes_with_free_gpus` behave oddly.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        aggregate = "gpu|up|1|mix|4/44/0/48|gpu:4|1-00:00:00|192000|48\n"
        nodes = "gpu|mixed|gpu:2|gpu:9(IDX:0-8)|\n"

        def _cmd(cmd: list[str]) -> str:
            return nodes if "-N" in cmd else aggregate

        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)
        p = resolve_cluster_partitions("gpu")[0]
        assert p.free_gpus_per_node == [0]

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

    def test_time_limit_is_read_even_when_every_node_line_is_flagged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A partition's wall-clock ceiling is CONFIGURATION, not a property of node
        # health, but it used to be parsed after the flagged-state `continue` — so a
        # partition drained end to end (live on this cluster: `climate`, 48 nodes, and
        # `climate-build`, 2, are entirely `drain*`) never learned its limit. A job asking
        # for more time than the partition allows then got the TRANSIENT "no room"
        # instead of the PERMANENT "time limit": a verdict decided by unrelated node
        # health (the SW-28 argument), and one that invites a wait that can never end.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        sinfo = "short|up|10|drain*|0/480/0/480|(null)|1:00:00|192000|48\n"
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: sinfo)
        p = resolve_cluster_partitions("short")[0]
        assert p.timelimit_seconds == 3600
        job = self._two_hour_job()
        assert fit_blocker(job, p) == "time limit"
        assert blocker_is_permanent(fit_blocker(job, p)) is True

    def test_a_flagged_node_still_contributes_no_capacity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Control: reading the limit before the schedulability filter must not let the
        # flagged node's cores, node counts or per-node sizes leak into the free-capacity
        # figures — that is what the same `continue` exists for. `idle*` (idle but not
        # responding), not `drain*`, because only there is the flag check load-bearing:
        # `drain` fails the idle/mix test on its own, so a `drain*` fixture stays green
        # even with the `continue` deleted.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        sinfo = "short|up|10|idle*|0/480/0/480|(null)|1:00:00|192000|48\n"
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: sinfo)
        p = resolve_cluster_partitions("short")[0]
        assert (p.idle_nodes, p.mix_nodes, p.cpus_idle) == (0, 0, 0)
        assert (p.max_node_cpus, p.max_node_mem_bytes) == (0, 0)
        assert p.total_nodes == 10 and p.cpus_total == 480  # totals count every node
        # A job that fits the limit is still blocked, and by scarcity, not the clock.
        short_enough = self._two_hour_job(time_limit_seconds=1800)
        assert fit_blocker(short_enough, p) == "no room"

    def _two_hour_job(self, **kw: object) -> PendingJob:
        base: dict[str, object] = {
            "job_id": "1",
            "raw_job_id": "1",
            "name": "j",
            "username": "u",
            "partition": "short",
            "qos": "",
            "account": "",
            "reason": "Resources",
            "submit_time": None,
            "start_time_estimate": None,
            "priority": None,
            "req_cpus": 1,
            "req_nodes": 1,
            "req_mem_bytes": 0,
            "req_gpus": 0,
            "req_gpu_type": "",
            "time_limit_seconds": 7200,
        }
        base.update(kw)
        return PendingJob(**base)  # type: ignore[arg-type]

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

    def test_the_group_gate_applies_when_the_account_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A site with no accounting has no Account on its jobs — the groups are still
        knowable, and they are a gate the user cannot edit around.

        Bailing out on an empty account listed every group-restricted partition as
        somewhere to requeue, which is the same "recommend a move Slurm will reject"
        this filter exists to prevent. The account dimension goes unfiltered (so a
        parsing gap still never hides a real option); the dimension that CAN be
        evaluated is applied.
        """
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        sinfo = (
            "open|up|4|idle|0/64/0/64|(null)|1-00:00:00|192000\n"
            "labonly|up|4|idle|0/64/0/64|(null)|1-00:00:00|192000\n"
            "pi-secret|up|4|idle|0/64/0/64|(null)|1-00:00:00|192000\n"
        )
        parts = (
            "PartitionName=open AllowGroups=ALL AllowAccounts=ALL\n"
            "PartitionName=labonly AllowGroups=pilab AllowAccounts=ALL\n"
            "PartitionName=pi-secret AllowGroups=ALL AllowAccounts=pi-secret\n"
        )
        monkeypatch.setattr(
            pending, "_run_slurm_cmd", lambda cmd: sinfo if cmd[0] == "sinfo" else parts
        )
        monkeypatch.setattr(pending, "_user_groups", lambda u: {"other"})
        names = {p.name for p in resolve_cluster_partitions("open", "", "someone")}
        assert "labonly" not in names, "a group we are not in is still a hard barrier"
        # ...and the ACCOUNT-restricted one stays, because we cannot judge it.
        assert names == {"open", "pi-secret"}, names

    def test_an_unevaluable_group_restriction_is_excluded_not_recommended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the owner's groups can't be resolved, omit the partition.

        The docstring has always promised this — "better to omit than to recommend a
        requeue that Slurm will reject" — and nothing tested it, so a sweep that let an
        unevaluable restriction through went unnoticed. Groups fail to resolve for real
        reasons: an LDAP/SSSD hiccup, a container without the group database, a
        username that isn't a local account.
        """
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
        monkeypatch.setattr(pending, "_user_groups", lambda u: None)  # cannot resolve
        names = {p.name for p in resolve_cluster_partitions("open", "acct", "someone")}
        assert names == {"open"}, names
        # A DenyGroups partition is treated the same way, for the same reason.
        parts_deny = (
            "PartitionName=open AllowGroups=ALL AllowAccounts=ALL\n"
            "PartitionName=nope AllowGroups=ALL DenyGroups=banned AllowAccounts=ALL\n"
        )
        sinfo_deny = sinfo.replace("labonly", "nope")
        monkeypatch.setattr(
            pending,
            "_run_slurm_cmd",
            lambda cmd: sinfo_deny if cmd[0] == "sinfo" else parts_deny,
        )
        names = {p.name for p in resolve_cluster_partitions("open", "acct", "someone")}
        assert names == {"open"}, names

    def test_nothing_knowable_still_means_no_filtering(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No account AND no resolvable groups: filter on nothing, hide nothing."""
        monkeypatch.setattr(pending, "_user_groups", lambda u: None)
        assert pending._resolve_accessible_partitions("", "") is None
        assert pending._resolve_accessible_partitions("", "nobody") is None

    def test_a_known_account_is_unaffected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verified against the live cluster too: with the real account this returns the
        same 10 partitions it did before the change, beagle3 among them."""
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        sinfo = (
            "open|up|4|idle|0/64/0/64|(null)|1-00:00:00|192000\n"
            "mine|up|4|idle|0/64/0/64|(null)|1-00:00:00|192000\n"
            "theirs|up|4|idle|0/64/0/64|(null)|1-00:00:00|192000\n"
        )
        parts = (
            "PartitionName=open AllowGroups=ALL AllowAccounts=ALL\n"
            "PartitionName=mine AllowGroups=ALL AllowAccounts=myacct,other\n"
            "PartitionName=theirs AllowGroups=ALL AllowAccounts=other\n"
        )
        monkeypatch.setattr(
            pending, "_run_slurm_cmd", lambda cmd: sinfo if cmd[0] == "sinfo" else parts
        )
        monkeypatch.setattr(pending, "_user_groups", lambda u: {"grp"})
        names = {p.name for p in resolve_cluster_partitions("open", "myacct", "someone")}
        assert names == {"open", "mine"}, names

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

    def test_gpu_job_needs_a_fully_idle_node_without_gres_detail(self) -> None:
        # The FALLBACK path (gpu_detail=False): with no GresUsed data, a mixed node's
        # free-GPU count is unknowable, so a GPU request must not be judged to fit
        # there. This is deliberately conservative — see the gpu_detail tests below
        # for what happens once the real figure is available.
        busy = PartitionResources("g", True, idle_nodes=0, mix_nodes=4, cpus_idle=40, has_gpus=True)
        assert busy.gpu_detail is False
        assert partition_fits_now(self._job(req_gpus=1), busy) is False
        free = PartitionResources("g", True, idle_nodes=2, mix_nodes=4, cpus_idle=40, has_gpus=True)
        assert partition_fits_now(self._job(req_gpus=1), free) is True

    def test_gpu_job_fits_a_mixed_node_that_has_free_gpus(self) -> None:
        # The bug this fixes: idle_nodes=0 but four mix nodes each holding 2 free
        # GPUs. The old code reported zero available nodes and "no room"; the real
        # answer is that 8 GPUs are allocatable right now.
        p = PartitionResources(
            "g",
            True,
            idle_nodes=0,
            mix_nodes=4,
            cpus_idle=40,
            has_gpus=True,
            max_node_gpus=4,
            free_gpus_per_node=[2, 2, 2, 2],
            gpu_detail=True,
        )
        assert p.gpus_free == 8
        assert p.max_node_gpus_free == 2
        assert available_node_count(self._job(req_gpus=1), p) == 4
        assert partition_fits_now(self._job(req_gpus=1), p) is True
        # Two GPUs on one node still fits; three does not — no node has three free.
        assert partition_fits_now(self._job(req_gpus=2), p) is True
        assert fit_blocker(self._job(req_gpus=3), p) == "GPUs busy"

    def test_gpu_job_does_not_fit_when_every_gpu_is_allocated(self) -> None:
        # The other half: mix nodes exist, but every GPU on them is taken. Measured
        # live on a cluster whose gpu partition was idle=0/mix=11 with 0 of 44 free.
        p = PartitionResources(
            "g",
            True,
            idle_nodes=0,
            mix_nodes=11,
            cpus_idle=88,
            has_gpus=True,
            max_node_gpus=4,
            free_gpus_per_node=[0] * 11,
            gpu_detail=True,
        )
        assert p.gpus_free == 0
        assert available_node_count(self._job(req_gpus=1), p) == 0
        # Named cause, not the catch-all "no room" — the partition has plenty of
        # free cores, so "no room" would send the user looking in the wrong place.
        assert fit_blocker(self._job(req_gpus=1), p) == "GPUs busy"

    def test_multi_node_gpu_job_counts_nodes_with_enough_free_each(self) -> None:
        # 2 nodes × 2 GPUs each: only the nodes with >=2 free count toward req_nodes.
        p = PartitionResources(
            "g",
            True,
            idle_nodes=0,
            mix_nodes=3,
            cpus_idle=96,
            has_gpus=True,
            max_node_gpus=4,
            free_gpus_per_node=[4, 2, 1],
            gpu_detail=True,
        )
        assert p.nodes_with_free_gpus(2) == 2
        assert partition_fits_now(self._job(req_nodes=2, req_gpus=4), p) is True
        assert partition_fits_now(self._job(req_nodes=3, req_gpus=6), p) is False

    def test_exclusive_gpu_job_still_needs_a_fully_idle_node(self) -> None:
        # gpu_detail must not weaken --exclusive: free GPUs on a busy node are no
        # use to a job that demands the whole machine.
        p = PartitionResources(
            "g",
            True,
            idle_nodes=0,
            mix_nodes=4,
            cpus_idle=40,
            has_gpus=True,
            max_node_gpus=4,
            free_gpus_per_node=[4, 4, 4, 4],
            gpu_detail=True,
        )
        assert available_node_count(self._job(req_gpus=1, exclusive=True), p) == 0
        assert partition_fits_now(self._job(req_gpus=1, exclusive=True), p) is False

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
    @pytest.mark.parametrize("reason", ["Resources", "Priority", "", "None"])
    def test_capacity_reasons_allow_requeue(self, reason: str) -> None:
        assert pending.requeue_could_help(reason) is True

    @pytest.mark.parametrize(
        "reason",
        [
            "QOSMaxCpuPerJob",
            "QOSMaxNodePerUserLimit",
            "AssocMaxCpuPerJobLimit",
            "AssocGrpCpuLimit",
        ],
    )
    def test_usage_caps_are_one_family_now(self, reason: str) -> None:
        """`assoc` was in the non-capacity list and `qos` was not, so two structurally
        identical usage-cap families were classified oppositely: the QOS one got
        "partition X has room — requeue there", which misdiagnoses a cap as a
        shortage, on the most common reason code on the reporting cluster (23 live
        jobs). Both are caps; neither gets the capacity suggestion.

        This claim is about REQUEUE, which is what that round was about, and it holds
        for all four. Whether each is a USAGE cap is a separate question, split out
        below: two of these four are per-JOB limits, where nothing about the user's
        usage is in the way."""
        assert pending.requeue_could_help(reason) is False
        # ...and they stay priority-ordered, so the estimate and queue position remain.
        assert pending.is_held_like(reason) is False

    @pytest.mark.parametrize(
        ("reason", "capped"),
        [
            # Per-USER / group: your other jobs are using the allowance, so waiting
            # for them genuinely helps and the cap tip is the right thing to say.
            ("QOSMaxNodePerUserLimit", True),
            ("AssocGrpCpuLimit", True),
            ("QOSMaxCpuPerUserLimit", True),
            # Per-JOB: THIS request is too big. No other job of yours is in the way,
            # so "a limit is capping your usage — other jobs must finish first"
            # describes an event that would change nothing. Slurm names them apart.
            ("QOSMaxCpuPerJob", False),
            ("AssocMaxCpuPerJobLimit", False),
            ("QOSMaxWallDurationPerJobLimit", False),
        ],
    )
    def test_a_per_job_limit_is_not_a_usage_cap(self, reason: str, capped: bool) -> None:
        """Measured on the live queue: 14 jobs on QOSMaxWallDurationPerJobLimit, all
        told "a QOS limit is capping your usage (running jobs / CPUs / GPUs / time)"
        — i.e. wait for your own jobs — when the fix is to lower --time. The partition
        twin of that reason, PartitionTimeLimit, has always said "lower --time"."""
        assert pending.is_usage_capped(reason) is capped

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


class TestReasonsMeasuredOnASecondCluster:
    """Harvested from a DIFFERENT cluster: Slurm 25.11, cgroup v2, 112 pending jobs.

    Two of its eleven distinct reasons were mishandled, and both are the shape round
    66 found — a substring rule deciding meaning — but neither string exists on the
    first cluster, so only running there could surface them.
    """

    @pytest.mark.parametrize(
        "reason",
        [
            "MaxBillingPerAccount",  # 21 live jobs on that cluster
            "MaxBillingPerUser",
            "MaxCpuPerUser",
            "MaxNodePerAccount",
            "MaxJobsPerAccount",
            "GrpBillingPerGroup",
        ],
    )
    def test_a_limit_scoped_to_an_account_or_user_is_a_usage_cap(self, reason: str) -> None:
        """`MaxBillingPerAccount` carries neither the `Assoc` nor the `QOS` prefix the
        token set matched on, so `is_usage_capped` said False while the EXPLAINER said
        "an account limit is capping your usage" — the same screen disagreeing with
        itself, and offering "requeue to a partition with room" for a job that is not
        short of room. Match the SCOPE (per-account / per-user / per-group), not the
        prefix."""
        assert pending.is_usage_capped(reason) is True
        assert pending.requeue_could_help(reason) is False
        # A cap is not a hold: the job stays priority-ordered, so its table stays.
        assert pending.capacity_is_irrelevant(reason) is False

    def test_a_per_job_limit_is_still_not_a_scoped_cap(self) -> None:
        """The round-66 rule has to win: PerJob means the request is too big."""
        assert pending.is_usage_capped("QOSMaxCpuPerJobLimit") is False
        assert pending.capacity_is_irrelevant("QOSMaxCpuPerJobLimit") is True

    def test_the_launch_failure_reason_says_a_release_is_needed(self) -> None:
        """Free text, spaces and all, as Slurm 25.11 reports it (3 live jobs).

        It fell to the generic "Slurm is holding it with reason '...'", which does not
        say the thing the reader needs: the job is HELD after a failed launch and sits
        there until released.
        """
        text = pending.explain_reason("launch failed requeued held")
        assert "scontrol release" in text, text
        assert "holding it with reason" not in text, text
        # Already classified as held-like, so the capacity table stays suppressed.
        assert pending.is_held_like("launch failed requeued held") is True
        assert pending.capacity_is_irrelevant("launch failed requeued held") is True

    def test_the_qos_wording_covers_billing_and_memory(self) -> None:
        """That cluster's caps are billing-based; the old parenthetical listed only
        jobs / CPUs / GPUs / time, so a billing cap read as if it were about none of
        the things actually limiting it."""
        text = pending.explain_reason("QOSMaxBillingPerUser")
        assert "billing" in text, text


class TestReasonsMeasuredOnTheLiveQueue:
    """Every distinct pending reason on a real 2896-job queue, run through the
    explainer. Four of the fourteen were wrong or useless, and all four were live.

    The heuristics are substring rules — `"qos" in reason` decides the whole meaning —
    so the failures were not random: a per-JOB limit and a per-USER cap are opposite
    situations that share a prefix, and `InvalidQOS` (invalid request) shares it with
    both.
    """

    @pytest.mark.parametrize(
        ("reason", "must_contain", "must_not_contain"),
        [
            # 14 live jobs. "A QOS limit is capping your usage" told them to wait for
            # their own jobs to finish; the fix is to lower --time. Its partition twin
            # PartitionTimeLimit has always said "lower --time".
            ("QOSMaxWallDurationPerJobLimit", "per-JOB", "capping your usage"),
            # 2 live jobs, previously "Slurm is holding it with reason 'BadConstraints'".
            ("BadConstraints", "--constraint", "holding it with reason"),
            # 1 live job, previously "An account/association limit is capping your
            # usage" — it is not a limit at all, the account is invalid.
            ("InvalidAccount", "--account", "capping your usage"),
            # 6 live jobs, previously the generic fallback. Self-resolving, and the
            # %N throttle is the thing to explain.
            ("JobArrayTaskLimit", "--array", "holding it with reason"),
            # The sibling of InvalidAccount, which the "qos" heuristic called a cap.
            ("InvalidQOS", "--qos", "capping your usage"),
        ],
    )
    def test_the_explanation_says_the_true_thing(
        self, reason: str, must_contain: str, must_not_contain: str
    ) -> None:
        text = pending.explain_reason(reason)
        assert must_contain in text, text
        assert must_not_contain not in text, text

    @pytest.mark.parametrize(
        "reason", ["Priority", "Resources", "Dependency", "QOSMaxCpuPerUserLimit", "None"]
    )
    def test_the_reasons_that_were_right_stay_right(self, reason: str) -> None:
        """The complement: five of the live fourteen were already correct."""
        text = pending.explain_reason(reason)
        assert "holding it with reason" not in text, text
        assert text.endswith(".")

    @pytest.mark.parametrize(
        ("reason", "irrelevant"),
        [
            ("QOSMaxWallDurationPerJobLimit", True),
            ("AssocMaxCpuPerJobLimit", True),
            ("InvalidAccount", True),
            ("BadConstraints", True),
            ("JobHeldUser", True),
            # A usage cap KEEPS its table: the job is priority-ordered and will run
            # when the user's other jobs finish, so the room figures are real context.
            # That was a deliberate decision with its own test; this must not reverse it.
            ("QOSMaxCpuPerUserLimit", False),
            ("AssocGrpCpuLimit", False),
            ("Priority", False),
            ("Resources", False),
        ],
    )
    def test_capacity_is_irrelevant_matches_what_the_tip_says(
        self, reason: str, irrelevant: bool
    ) -> None:
        """The screen used to contradict itself: a tip saying "it isn't waiting on free
        capacity" above a table of partitions answering the capacity question. SW-29
        fixed that for holds; a per-job limit and an invalid request produced it again.
        """
        assert pending.capacity_is_irrelevant(reason) is irrelevant

    @staticmethod
    def _parts() -> list[PartitionResources]:
        return [
            PartitionResources(
                "build", True, idle_nodes=4, cpus_idle=128, max_node_cpus=48, is_current=True
            ),
            PartitionResources("broadwl", True, idle_nodes=53, cpus_idle=1101, max_node_cpus=48),
        ]

    @pytest.mark.parametrize("reason", ["InvalidAccount", "InvalidQOS", "BadConstraints"])
    def test_an_invalid_request_is_not_offered_a_partition_with_room(self, reason: str) -> None:
        """ "Requeue to a partition with room" is noise for a job that is invalid.

        The substring rules got these wrong in opposite directions, so both verdicts
        need pinning: `InvalidQOS` contains "qos" and was called a usage cap, while
        `InvalidAccount` and `BadConstraints` matched nothing and were handed the
        capacity suggestion. Room is not what any of the three lacks.
        """
        assert pending.requeue_could_help(reason) is False
        assert pending.is_usage_capped(reason) is False

    def test_both_renderers_suppress_the_table_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One predicate, two renderers — the recurring half-fix in this exercise.

        The resolvers are PATCHED, not ambient: the suppression branch needs a
        non-empty partition list, which resolving for real needs Slurm on PATH. The
        first version of this test passed on the cluster and skipped its branch
        everywhere else — the same trap as round 62, and I had written the note about
        it before repeating it.
        """
        import io
        from contextlib import redirect_stdout

        from rich.text import Text

        from slurmwatch import cli
        from slurmwatch.cli import _print_pending_summary
        from slurmwatch.tui import PendingView

        parts = self._parts()
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: parts)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        job = pending._mock_pending_job("777")
        job.reason = "QOSMaxWallDurationPerJobLimit"
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_pending_summary(job, stream=buf)
        text_report = buf.getvalue()
        assert "capacity is not the constraint" in text_report, text_report
        assert "FITS NOW" not in text_report, text_report

        view = PendingView()
        view.job = job
        view.config = SlurmwatchConfig()
        view.resolved = True
        view.partitions = parts
        card = Text.from_markup(view.render()).plain
        assert "capacity is not the constraint" in card, card
        assert "YES" not in card, card

    def test_a_usage_cap_still_gets_its_table_in_both(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The complement, and the decision this must not reverse."""
        import io
        from contextlib import redirect_stdout

        from rich.text import Text

        from slurmwatch import cli
        from slurmwatch.cli import _print_pending_summary
        from slurmwatch.tui import PendingView

        parts = self._parts()
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: parts)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        job = pending._mock_pending_job("777")
        job.reason = "QOSMaxCpuPerUserLimit"
        job.req_gpus = 0
        job.req_cpus = 4
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_pending_summary(job, stream=buf)
        assert "capacity is not the constraint" not in buf.getvalue(), buf.getvalue()

        view = PendingView()
        view.job = job
        view.config = SlurmwatchConfig()
        view.resolved = True
        view.partitions = parts
        card = Text.from_markup(view.render()).plain
        assert "capacity is not the constraint" not in card, card


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
        seen: list[str] = []

        def _fake(cmd: list[str]) -> str:
            part = cmd[cmd.index("-p") + 1]
            seen.append(part)
            # In 'a' the job (prio 500) is behind 3; in 'b' behind only 1. A POOLED
            # `squeue -p a,b` would really return the UNION of both queues (and list
            # the job once per partition), so model that too — otherwise a pooling
            # regression could coincidentally return the same answer as one queue.
            return {
                "a": "900\n800\n700\n500\n",
                "b": "600\n500\n",
                "a,b": "900\n800\n700\n500\n600\n500\n",
            }[part]

        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_run_slurm_cmd", _fake)
        # a: 3 ahead -> #4 of 4; b: 1 ahead -> #2 of 2. Best (nearest the front) = (2, 2).
        assert resolve_priority_rank("a,b", 500) == (2, 2)
        # Each partition was queried SEPARATELY — priorities only compare within one,
        # and the pooled query above would have answered "#5 of 6".
        assert seen == ["a", "b"]

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


class TestPriorityRankIsTheLineSlurmForms:
    """The "#N of M" pair must describe the queue the controller sorts, not a rounder one.

    Two live findings drive this, both measured on midway3 (Slurm 20.11.8) rather
    than argued:

    * A tie run is handed ONE number. 603 of the 806 caslake jobs that get shown a
      rank sat in a run of equal priority; the largest run was 116 jobs at priority
      131765, every one of them told "#421 of 1238". On beagle3 a 66-job run at
      priority 1106212 was told "#19 of 93" from front to back -- the last member is
      really 83rd. Slurm breaks the tie by ASCENDING job id, and the live queue
      agrees: over the 245 equal-priority groups among caslake jobs that had queued
      more than an hour, the lower job id started first in 99.8% of 2.9M
      within-group pairs (median per-group concordance 1.000).
    * M counted rows that are never weighed against anything. 12.7% of the 1,582
      pending rows were a hold, a dead dependency, or a request that must change --
      and being the oldest jobs on the queue, they age ABOVE most callers, so they
      inflated N as well: a real caslake job read "#822 of 1238" and is "#676 of
      1092".
    """

    # Shaped exactly like `squeue -a -h -p <part> -t PD -o "%Q|%i|%r"`: priority,
    # job id, reason. Three of the four rows above the tie run are queued but not in
    # line; the run itself is five jobs at one priority, of which 202 is the caller.
    _QUEUE = (
        "900|100|Priority\n"
        "800|101|DependencyNeverSatisfied\n"
        "800|102|JobHeldUser\n"
        "800|103|QOSMaxWallDurationPerJobLimit\n"
        "500|200|Priority\n"
        "500|201|Priority\n"
        "500|202|Priority\n"
        "500|203|Priority\n"
        "500|204|Priority\n"
        "100|300|Priority\n"
    )

    # No ties, nothing held: the queue the control below pins.
    _PLAIN = (
        "900|100|Priority\n700|101|Priority\n500|102|Priority\n300|103|Resources\n100|104|None\n"
    )

    @staticmethod
    def _feed(monkeypatch: pytest.MonkeyPatch, out: str) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: out)

    def test_tie_run_gets_distinct_seats_in_job_id_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Five jobs at priority 500. `sum(p > priority)` puts all five on the same
        # seat, so four of them read a position they do not hold; Slurm orders them
        # by ascending job id, so they occupy five CONSECUTIVE seats. Asserted as
        # offsets from the front of the run, which is what the tie-break alone
        # decides -- the denominator fix shifts the whole run and must not be able
        # to make this pass.
        self._feed(monkeypatch, self._QUEUE)
        ranks = []
        for job_id in ("200", "201", "202", "203", "204"):
            rank = resolve_priority_rank("p", 500, job_id)
            assert rank is not None
            ranks.append(rank[0])
        assert len(set(ranks)) == 5, f"a tie run must not share one seat: {ranks}"
        assert [r - ranks[0] for r in ranks] == [0, 1, 2, 3, 4]

    def test_never_competing_rows_leave_the_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Job 200 is the FRONT of the tie run, so the tie-break adds nothing and this
        # isolates the denominator: the held, never-satisfiable and must-change rows
        # go from both sides at once. Only job 100 is genuinely ahead of it.
        self._feed(monkeypatch, self._QUEUE)
        assert resolve_priority_rank("p", 500, "200") == (2, 7)

    def test_own_never_competing_row_is_still_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A `QOSMaxWallDurationPerJobLimit` job is NOT held-like, so it is still shown
        # a rank (only `is_held_like` suppresses that). Its own row must survive the
        # filter or "of M" would exclude the very job the position describes -- M is 8
        # here, one more than the 7 its two never-competing neighbours leave behind.
        self._feed(monkeypatch, self._QUEUE)
        rank = resolve_priority_rank("p", 800, "103")
        assert rank == (2, 8)

    def test_control_no_ties_no_holds_reproduces_todays_numbers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL: passes BEFORE and AFTER both fixes.

        Neither correction may move a queue that has nothing for it to correct. With
        every priority distinct and no row out of the line, the answer has to be the
        pre-fix one -- otherwise a corrected count is indistinguishable from a
        changed one. Pinned both with and without a ``job_id``, since the tie-break
        is what the id switches on.
        """
        self._feed(monkeypatch, self._PLAIN)
        assert resolve_priority_rank("p", 500, "102") == (3, 5)
        assert resolve_priority_rank("p", 500) == (3, 5)
        # Front and back of the same queue, still untouched.
        assert resolve_priority_rank("p", 900, "100") == (1, 5)
        assert resolve_priority_rank("p", 100, "104") == (5, 5)
        # And the bare `-o "%Q"` shape the older tests record: a row with no id and
        # no reason must still count, so a controller that will not give the wider
        # format degrades to today's plain answer instead of losing the rank.
        self._feed(monkeypatch, "900\n700\n500\n300\n100\n")
        assert resolve_priority_rank("p", 500, "102") == (3, 5)

    def test_reason_field_keeps_its_spaces_and_commas(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Recorded verbatim from the live queue: the reason is the last field for
        # exactly this row, whose commas and spaces a whitespace split would shatter
        # into tokens that then read as garbage priorities.
        self._feed(
            monkeypatch,
            "900|100|ReqNodeNotAvail, UnavailableNodes:midway3-[0440-0441]\n500|101|Priority\n",
        )
        assert resolve_priority_rank("p", 500, "101") == (2, 2)

    def test_array_row_and_task_id_compare_as_the_same_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A pending array task reports the ARRAY id from scontrol (`scontrol show job
        # 53302068_1` -> `JobId=53302068`) and squeue prints `53302068_[1-5]` for the
        # un-launched row, so the two must resolve to one number and the job must not
        # count its own array as sitting ahead of it.
        self._feed(monkeypatch, "500|900_[1-5]|Priority\n500|901|Priority\n")
        assert resolve_priority_rank("p", 500, "900_1") == (1, 2)
        assert pending._raw_job_number("900_[1-5]") == pending._raw_job_number("900_1") == 900
        assert pending._raw_job_number("900+0") is None

    def test_both_renderers_hand_over_the_job_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Without the id the tie-break cannot fire, and it would fire nowhere in
        # production while every unit test above still passed. Both renderers own a
        # copy of this call, so pin both: the CLI report and the TUI screen.
        import io

        import slurmwatch.tui as tui_mod
        from slurmwatch import cli

        seen: list[tuple[object, ...]] = []

        def _spy(*args: object) -> tuple[int, int]:
            seen.append(args)
            return (3, 9)

        job = pending._mock_pending_job("777")
        monkeypatch.setattr(cli, "resolve_priority_rank", _spy)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda part: (1, 2))
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a: [])
        monkeypatch.setattr(cli, "resolve_user_associations", lambda user: None)
        cli._print_pending_summary(job, stream=io.StringIO())
        assert seen == [(job.partition, job.priority, job.raw_job_id)]

        seen.clear()
        monkeypatch.setattr(tui_mod, "resolve_priority_rank", _spy)
        monkeypatch.setattr(tui_mod, "resolve_queue_counts", lambda part: (1, 2))
        monkeypatch.setattr(tui_mod, "resolve_cluster_partitions", lambda *a: [])
        monkeypatch.setattr(tui_mod, "resolve_pending_job", lambda jid: job)
        app = tui_mod.PendingApp(job)

        async def _drive() -> None:
            async with app.run_test(size=(110, 40)) as pilot:
                for _ in range(40):
                    await pilot.pause()
                    if seen:
                        break
                    await asyncio.sleep(0.03)

        asyncio.run(_drive())
        assert seen and seen[0] == (job.partition, job.priority, job.raw_job_id)


class TestQueueCounts:
    def test_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        out = "1|RUNNING\n2|RUNNING\n3|PENDING\n4|COMPLETING\n5|PENDING\n6|SUSPENDED\n"
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: out)
        # 2 RUNNING + 1 COMPLETING running; 2 PENDING + 1 SUSPENDED pending.
        assert resolve_queue_counts("p") == (3, 3)

    def test_counts_include_transient_active_states(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A6: RESIZING/SIGNALING/REQUEUED are active states squeue returns; they must
        # count as running, not vanish from the "N running · M pending" context.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        out = "1|RUNNING\n2|RESIZING\n3|SIGNALING\n4|REQUEUED\n5|PENDING\n"
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: out)
        assert resolve_queue_counts("p") == (4, 1)  # 4 running-ish, 1 pending

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


class TestHiddenPartitionQueueContext:
    """The queue-context helpers must pass ``squeue -a`` or a hidden partition reads empty.

    Slurm's own ``squeue --help`` (20.11.8): "-a, --all  display jobs in hidden
    partitions". Without it the controller drops every job whose partition carries
    ``Hidden=YES`` -- including the caller's OWN pending job -- so both helpers below
    receive an empty string that is indistinguishable from an empty queue. Every
    ``sinfo`` call in the module already passes ``-a`` for exactly this reason.

    The repro is driven through RECORDED output, not live: on the cluster this was
    written against, ``test``/``climate``/``climate-build`` are the ``Hidden=YES``
    partitions and all three held zero jobs, and the probing account holds
    ``AdminLevel=Operator``, which Slurm exempts from the hidden-partition filter
    server-side (``squeue`` returned an identical 5290 jobs with and without ``-a``).
    So the filter could not be demonstrated live from that account.
    """

    # Recorded verbatim from `squeue -a -h -p bigmem -o "%i|%T"` on Slurm 20.11.8:
    # 6 PENDING + 5 RUNNING.
    _ROWS = (
        "57071245|PENDING\n"
        "57071242|PENDING\n"
        "57071227|PENDING\n"
        "57070813|PENDING\n"
        "57070695|PENDING\n"
        "57070569|PENDING\n"
        "57075002|RUNNING\n"
        "57023337|RUNNING\n"
        "57072993|RUNNING\n"
        "57070351|RUNNING\n"
        "57070352|RUNNING\n"
    )
    # Recorded verbatim from `squeue -a -h -p bigmem -t PD -o "%Q"` (same snapshot).
    _PRIOS = "139886\n139886\n139885\n139885\n139885\n139885\n"

    @staticmethod
    def _hidden(recorded: str) -> tuple[Callable[[list[str]], str], list[list[str]]]:
        """A controller that answers only when ``-a`` is passed, as a hidden partition does."""
        calls: list[list[str]] = []

        def _cmd(cmd: list[str]) -> str:
            calls.append(cmd)
            return recorded if "-a" in cmd else ""

        return _cmd, calls

    def test_queue_counts_sees_a_hidden_partition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        cmd, calls = self._hidden(self._ROWS)
        monkeypatch.setattr(pending, "_run_slurm_cmd", cmd)
        # Without -a this is (0, 0) -- the fabricated, self-contradictory zero that
        # resolve_queue_counts' docstring exists to rule out, and one the
        # SlurmCommandError guard cannot catch because squeue exits 0 (verified:
        # `squeue -a -h -p climate-build -o "%i|%T"` printed nothing, rc=0).
        assert resolve_queue_counts("climate") == (5, 6)
        assert "-a" in calls[0]

    def test_priority_rank_sees_a_hidden_partition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        cmd, calls = self._hidden(self._PRIOS)
        monkeypatch.setattr(pending, "_run_slurm_cmd", cmd)
        # Two jobs at 139886 sit ahead of ours at 139885 -> #3 of 6. Without -a the
        # priority list is empty and the rank silently disappears (None), so a
        # Priority wait on a hidden partition loses its "#N of M" position entirely.
        assert resolve_priority_rank("climate", 139885) == (3, 6)
        assert "-a" in calls[0]

    def test_control_visible_partition_answer_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL: passes BEFORE and AFTER the fix.

        A non-hidden partition returns the same rows either way (verified live:
        `squeue -h -p amd` and `squeue -a -h -p amd` both returned 180 jobs with the
        same %P set), so ``-a`` must not change the answer, widen the ``-p`` filter,
        drop ``-t PD``, or double-count anything.
        """
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        seen: list[list[str]] = []

        def _cmd(cmd: list[str]) -> str:
            seen.append(cmd)
            return self._PRIOS if "-t" in cmd else self._ROWS

        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)
        assert resolve_queue_counts("bigmem") == (5, 6)
        assert resolve_priority_rank("bigmem", 139885) == (3, 6)
        assert len(seen) == 2
        for cmd in seen:
            assert cmd[0] == "squeue"
            assert "-h" in cmd
            assert cmd[cmd.index("-p") + 1] == "bigmem"
        assert [c[c.index("-t") + 1] for c in seen if "-t" in c] == ["PD"]

    def test_control_empty_visible_queue_is_still_zeros_not_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL: passes BEFORE and AFTER. An empty (rc=0) queue stays (0, 0)."""
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: "")
        assert resolve_queue_counts("p") == (0, 0)
        assert resolve_priority_rank("p", 500) is None


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
        # clean (no prose for a jq/CSV reader) — report on STDERR and exit 1. SW-27's
        # fifth outcome refines "clean" to "parseable, not empty": the facts go to
        # stdout in the requested format, the prose stays on stderr.
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
        rows = list(csv.DictReader(captured.out.splitlines()))
        assert len(rows) == 1, captured.out
        row = rows[0]
        assert row["telemetry_unavailable_reason"] == "job_pending"
        assert "Why" not in captured.out, "no prose on the machine channel"
        # The REQUEST is the useful content for a queued job, and it is what the
        # foreign schema's "requested, not used" fields are for. A poller deciding
        # whether to wait needs the shape of what it asked for.
        job = pending._mock_pending_job("777")
        assert row["state"] == "PENDING"
        assert row["cpus_allocated"] == str(job.req_cpus)
        assert row["gpu_count_requested"] == str(job.req_gpus)
        assert row["mem_limit_bytes"] == str(job.req_mem_bytes)
        assert row["time_limit_seconds"] == str(job.time_limit_seconds)
        assert row["partition"] == job.partition and row["owner"] == job.username
        assert row["reason"], "and why it is waiting, as data not prose"
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

    def test_headless_pending_writes_only_the_facts_row(
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
        # Non-zero: no file AND exit 0 reads exactly like a finished recording.
        with pytest.raises(SystemExit) as exc:
            cli._run_headless("777", SlurmwatchConfig(), str(log))
        assert exc.value.code == 1
        # SW-27's fifth outcome: one facts row rather than no file, matching the
        # foreign-job branch. No TELEMETRY row is written (the measured columns are
        # empty), and the non-zero exit still says "there is no recording here" — which
        # is what the old "no file AND exit 0 reads like a finished recording" note was
        # protecting, and the exit code carries that on its own.
        rows = list(csv.DictReader(log.read_text().splitlines()))
        assert len(rows) == 1, rows
        assert rows[0]["telemetry_unavailable_reason"] == "job_pending"
        assert rows[0]["state"] == "PENDING"
        assert rows[0]["cpu_percent"] == "", "never a measured-looking zero"
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

    def test_verdict_column_hedges_when_permission_was_not_checked(self) -> None:
        """SW-2, TUI side (the cli renderer has its own twin of this): the column
        may only claim "can run now" when the association list was readable —
        otherwise it measured room, and a partition with room can still reject the
        job. Both renderers hedge or neither is honest."""
        view = self._view()
        checked = Text.from_markup(view.render()).plain  # type: ignore[attr-defined]
        assert "can run now?" in checked
        for part in view.partitions:  # type: ignore[attr-defined]
            part.assoc_verified = False
        hedged = Text.from_markup(view.render()).plain  # type: ignore[attr-defined]
        assert "has room now?" in hedged
        assert "can run now?" not in hedged

    def test_partitions_default_is_not_shared_across_instances(self) -> None:
        # A bare `partitions: list[...] = []` class attribute would hand every
        # instance the SAME list object — harmless only as long as every write
        # rebinds (`self.partitions = [...]`) rather than mutates in place.
        # Mutating one instance's default must never leak into another's.
        from slurmwatch.tui import PendingView

        a, b = PendingView(), PendingView()
        assert a.partitions is not b.partitions
        a.partitions.append(PartitionResources("x", True))
        assert b.partitions == []

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

    def test_where_tip_omitted_when_the_current_partition_can_hold_the_job(self) -> None:
        # The tip's condition tested `fits`, which is deliberately forced False for the
        # CURRENT partition (so the table never prints a contradictory "FITS NOW
        # (current)"). That made it a tautology: the claim below was emitted no matter
        # what, directly contradicting the free-node and idle-core columns beside it. A
        # job can sit PENDING on Reason=Priority while its own partition has ample room.
        from slurmwatch.tui import PendingView

        job = PendingJob(
            job_id="1",
            raw_job_id="1",
            name="j",
            username="u",
            partition="cur",
            qos="",
            account="",
            reason="Priority",  # queued behind others, NOT short of resources
            submit_time=None,
            start_time_estimate=None,
            priority=100,
            req_cpus=4,
            req_nodes=1,
            req_mem_bytes=0,
            req_gpus=0,
            req_gpu_type="",
            time_limit_seconds=3600,
        )
        v = PendingView()
        v.job = job
        v.config = SlurmwatchConfig()
        # The job's own partition has plenty free, and it is the ONLY partition, so
        # there is no alternative to suggest either.
        v.partitions = [
            PartitionResources(
                "cur",
                True,
                idle_nodes=8,
                cpus_idle=240,
                max_node_cpus=48,
                max_idle_node_cpus=48,
                is_current=True,
            )
        ]
        out = v.render()
        assert "no partition currently has enough free capacity" not in out

    def test_where_tip_still_shown_when_no_current_partition_fits(self) -> None:
        # The complement of the test above: when the job genuinely does not fit its own
        # partition, the explanatory tip must still appear.
        from slurmwatch.tui import PendingView

        job = PendingJob(
            job_id="1",
            raw_job_id="1",
            name="j",
            username="u",
            partition="cur",
            qos="",
            account="",
            reason="Resources",
            submit_time=None,
            start_time_estimate=None,
            priority=100,
            req_cpus=999,  # more cores than any node in the partition has
            req_nodes=1,
            req_mem_bytes=0,
            req_gpus=0,
            req_gpu_type="",
            time_limit_seconds=3600,
        )
        v = PendingView()
        v.job = job
        v.config = SlurmwatchConfig()
        v.partitions = [
            PartitionResources(
                "cur", True, idle_nodes=0, cpus_idle=0, max_node_cpus=48, is_current=True
            )
        ]
        # SW-28: 999 CPUs against a 48-CPU node is PERMANENT, so the tip must not
        # promise a start. Both renderers say the same thing (the cli twin asserts it).
        out = Text.from_markup(v.render()).plain
        assert "can ever hold this request" in out, out
        assert "largest node: 48 CPU" in out
        assert "will not start as submitted" in out
        assert "once resources free up" not in out

        # And the transient case still gets the transient tip.
        job.req_cpus = 16
        out = Text.from_markup(v.render()).plain
        assert "no partition currently has enough free capacity" in out, out
        assert "can ever hold" not in out

    def test_where_header_says_free_nodes_for_gpu_job_with_gpu_detail(self) -> None:
        # Post-4e91d55, available_node_count() counts MIXED nodes with enough free
        # GPUs left for a GPU job when gpu_detail is available — so those aren't
        # "empty", and the header must read "free nodes" like a plain job's, not
        # claim every counted node is fully idle.
        from slurmwatch.tui import PendingView

        job = pending._mock_pending_job("777")
        job.req_gpus = 1
        job.reason = "Priority"  # "Resources" 's own explanation text says "free
        # nodes", which would make the header assertion below pass for the wrong
        # reason.
        v = PendingView()
        v.job = job
        v.config = SlurmwatchConfig()
        v.partitions = [
            PartitionResources(
                "cur",
                True,
                idle_nodes=0,
                mix_nodes=3,
                cpus_idle=8,
                has_gpus=True,
                gpu_types=["a100"],
                free_gpus_per_node=[2, 0, 1],
                gpu_detail=True,
                is_current=True,
            ),
        ]
        plain = Text.from_markup(v.render()).plain
        assert "free nodes" in plain
        assert "empty nodes" not in plain

    def test_where_header_still_says_empty_nodes_for_gpu_job_without_gpu_detail(
        self,
    ) -> None:
        # The conservative fallback (no per-node free-GPU data) still needs a fully
        # idle node, so the header stays "empty nodes" there.
        from slurmwatch.tui import PendingView

        job = pending._mock_pending_job("777")
        job.req_gpus = 1
        job.reason = "Priority"
        v = PendingView()
        v.job = job
        v.config = SlurmwatchConfig()
        v.partitions = [
            PartitionResources(
                "cur",
                True,
                idle_nodes=2,
                cpus_idle=8,
                has_gpus=True,
                gpu_types=["a100"],
                is_current=True,
            ),
        ]
        plain = Text.from_markup(v.render()).plain
        assert "empty nodes" in plain
        assert "free nodes" not in plain

    def test_where_table_truncates_with_a_more_partitions_notice(self) -> None:
        # The cap (_MAX_ROWS) exists so a pathological unfiltered list can't flood
        # the screen — but truncation must say so, never cut the list silently.
        from slurmwatch.tui import PendingView

        job = pending._mock_pending_job("777")
        job.reason = "Priority"
        v = PendingView()
        v.job = job
        v.config = SlurmwatchConfig()
        # 1 current (fits, always kept) + 29 down partitions (none fit) — the fill
        # loop keeps current + 23 of the down ones to reach the cap of 24, dropping 6.
        v.partitions = [
            PartitionResources("cur", True, idle_nodes=8, cpus_idle=64, is_current=True),
        ] + [PartitionResources(f"down{i}", False) for i in range(29)]
        plain = Text.from_markup(v.render()).plain
        assert "and 6 more partition(s)" in plain

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
        # capacity it must never read a self-contradictory "YES/fits now".
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = pending._mock_pending_job("777")
        v.config = SlurmwatchConfig()
        # has_gpus, so the mock GPU job's only blocker is that it is pending HERE —
        # a transient wait, which is what "waiting (current)" is for.
        v.partitions = [
            PartitionResources(
                "mypart",
                True,
                idle_nodes=50,
                cpus_idle=9999,
                has_gpus=True,
                gpu_types=["a100"],
                is_current=True,
            )
        ]
        lines = Text.from_markup(v.render()).plain.splitlines()
        cur = next(ln for ln in lines if "(current)" in ln)
        assert "waiting" in cur and "YES" not in cur

    def test_the_current_partition_names_a_blocker_waiting_cannot_fix(self) -> None:
        """SW-28: "waiting (current)" beside a request no node here can ever hold reads
        as though patience were the answer. `fits` stays forced False either way, so
        this cannot become the self-contradictory "FITS NOW (current)"."""
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = pending._mock_pending_job("777")
        v.job.req_gpus = 0
        v.job.req_cpus = 999  # no node here is that big
        v.config = SlurmwatchConfig()
        v.partitions = [
            PartitionResources(
                "mypart", True, idle_nodes=50, cpus_idle=9999, max_node_cpus=48, is_current=True
            )
        ]
        lines = Text.from_markup(v.render()).plain.splitlines()
        cur = next(ln for ln in lines if "(current)" in ln)
        assert "node too small (current)" in cur, cur
        assert "YES" not in cur and "waiting" not in cur

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
            # The refresh resolves partitions on an executor thread, so poll until it
            # lands rather than assuming a fixed budget: the old ~0.9 s ceiling was
            # exceeded on a loaded machine and the test then asserted against a
            # still-empty view (~1 run in 5 here). Breaks as soon as it is ready.
            view = app.screen.query_one(PendingView)
            for _ in range(200):
                await pilot.pause()
                if view.partitions:
                    break
                await asyncio.sleep(0.03)
            assert view.partitions, "partitions never resolved"
            plain = Text.from_markup(view.render()).plain
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


class TestAssociationGate:
    """SW-2: the WHERE table offered partitions the account cannot submit to.

    Measured on a 28-partition cluster: slurmwatch marked private per-PI partitions
    "YES ▸ can run now" and `sbatch --test-only` answered "Invalid account or
    account/partition combination specified". Their own ACLs read
    `AllowGroups=ALL AllowAccounts=ALL`, so the gate isn't the partition — it's the
    `sacctmgr` association list, which nothing consulted.
    """

    SINFO = (
        "broadwl|up|10|idle|0/280/0/280|(null)|1-00:00:00|64000|28\n"
        "kicpaa|up|4|idle|0/112/0/112|(null)|1-00:00:00|64000|28\n"
        "xenon1t|up|2|idle|0/56/0/56|(null)|1-00:00:00|64000|28\n"
    )
    # Every partition advertises itself as open — which is exactly why capacity plus
    # partition ACLs was not enough.
    SCONTROL = (
        "PartitionName=broadwl AllowGroups=ALL AllowAccounts=ALL State=UP\n"
        "PartitionName=kicpaa AllowGroups=ALL AllowAccounts=ALL State=UP\n"
        "PartitionName=xenon1t AllowGroups=ALL AllowAccounts=ALL State=UP\n"
    )

    def _routed(self, assoc: str | None) -> Callable[..., str]:
        def _run(cmd: list[str], *a: object, **k: object) -> str:
            if cmd[0] == "sacctmgr":
                if assoc is None:
                    raise SlurmCommandError("Slurm binary not found: sacctmgr")
                return assoc
            if cmd[0] == "scontrol":
                return self.SCONTROL
            if cmd[0] == "sinfo" and "-N" in cmd:
                return ""  # no per-node GPU detail needed for a CPU-only fixture
            if cmd[0] == "sinfo":
                return self.SINFO
            return ""

        return _run

    def _names(self, monkeypatch: pytest.MonkeyPatch, assoc: str | None) -> list[str]:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_run_slurm_cmd", self._routed(assoc))
        monkeypatch.setattr(pending, "_user_groups", lambda u: {"users"})
        parts = resolve_cluster_partitions("broadwl", "data-bfi-voter", "youzhi")
        return [p.name for p in parts]

    def test_drops_partitions_the_account_has_no_association_with(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        names = self._names(monkeypatch, "data-bfi-voter|broadwl\n")
        assert "broadwl" in names
        assert "kicpaa" not in names and "xenon1t" not in names, names

    def test_a_blank_partition_field_means_every_partition(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`sacctmgr` writes an association that isn't partition-scoped with an
        empty Partition column; reading that as "no partitions" would hide the
        whole cluster."""
        names = self._names(monkeypatch, "data-bfi-voter|\n")
        assert {"broadwl", "kicpaa", "xenon1t"} <= set(names), names

    def test_another_accounts_rows_do_not_grant_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assoc = "data-bfi-voter|broadwl\npi-someone-else|kicpaa\n"
        names = self._names(monkeypatch, assoc)
        assert "kicpaa" not in names, names

    def test_no_sacctmgr_does_not_filter_anything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Can't-determine must never hide real options — the same rule the
        partition-ACL gate already follows."""
        names = self._names(monkeypatch, None)
        assert {"broadwl", "kicpaa", "xenon1t"} <= set(names), names

    def test_no_row_for_the_job_account_is_treated_as_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the account we were handed matches nothing, our reading is off — that
        is not evidence the user may go nowhere."""
        names = self._names(monkeypatch, "someone-else|kicpaa\n")
        assert {"broadwl", "kicpaa", "xenon1t"} <= set(names), names

    def test_the_current_partition_is_kept_even_without_an_association(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_run_slurm_cmd", self._routed("data-bfi-voter|xenon1t\n"))
        monkeypatch.setattr(pending, "_user_groups", lambda u: {"users"})
        parts = resolve_cluster_partitions("broadwl", "data-bfi-voter", "youzhi")
        assert [p.name for p in parts if p.is_current] == ["broadwl"]

    def test_rows_record_whether_permission_was_checked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_user_groups", lambda u: {"users"})
        monkeypatch.setattr(pending, "_run_slurm_cmd", self._routed("data-bfi-voter|broadwl\n"))
        checked = resolve_cluster_partitions("broadwl", "data-bfi-voter", "youzhi")
        assert all(p.assoc_verified for p in checked)
        monkeypatch.setattr(pending, "_run_slurm_cmd", self._routed(None))
        unchecked = resolve_cluster_partitions("broadwl", "data-bfi-voter", "youzhi")
        assert not any(p.assoc_verified for p in unchecked)

    def _where_table(self, monkeypatch: pytest.MonkeyPatch, assoc: str | None) -> str:
        """The plain-text WHERE table for a pending job, as the CLI prints it."""
        import io

        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_user_groups", lambda u: {"users"})
        monkeypatch.setattr(pending, "_run_slurm_cmd", self._routed(assoc))
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda p: None)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda p, prio: None)
        job = pending._mock_pending_job("12345")
        job.partition = "broadwl"
        job.account = "data-bfi-voter"
        job.username = "youzhi"
        job.req_cpus = 1
        job.req_gpus = 0
        job.req_gpu_type = ""
        job.req_mem_bytes = 1024**3
        job.req_nodes = 1
        buf = io.StringIO()
        cli._print_pending_summary(job, stream=buf)
        return buf.getvalue()

    def test_the_verdict_column_only_claims_can_run_when_it_checked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ "can run now?" asserts capacity AND permission. With no association list
        this column measured room alone, and a partition with room can still reject
        the job — so the header has to stop making the bigger claim (SW-2)."""
        checked = self._where_table(monkeypatch, "data-bfi-voter|\n")
        assert "can run now?" in checked

    def test_the_verdict_column_hedges_when_it_could_not_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unchecked = self._where_table(monkeypatch, None)
        assert "has room now?" in unchecked
        assert "can run now?" not in unchecked

    def test_the_association_query_is_scoped_to_the_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[list[str]] = []

        def _run(cmd: list[str], *a: object, **k: object) -> str:
            seen.append(cmd)
            return self._routed("data-bfi-voter|broadwl\n")(cmd)

        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_user_groups", lambda u: {"users"})
        monkeypatch.setattr(pending, "_run_slurm_cmd", _run)
        resolve_cluster_partitions("broadwl", "data-bfi-voter", "youzhi")
        assoc_cmds = [c for c in seen if c[0] == "sacctmgr"]
        assert assoc_cmds, "the association list is never consulted"
        assert "user=youzhi" in assoc_cmds[0]
        assert "format=Account,Partition" in assoc_cmds[0]


class TestLoadingIsNotAFault:
    """SW-24 / round 38: for the ~2 s while partition data resolves, both panels read
    "unavailable (controller busy)" and "cluster partition info unavailable" — a
    LOADING state described as a FAULT, and the fault it named was a false claim about
    cluster health. Measured at the same moment: `squeue -h -p build` answered in
    0.06 s and `sinfo -h` in 0.20 s, while resolve_cluster_partitions() took 2.0 s on
    76 partitions (a cost that scales with partition count, so bigger sites wait
    longer)."""

    def _view(self, *, resolved: bool) -> object:
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = pending._mock_pending_job("777")
        v.config = SlurmwatchConfig()
        v.resolved = resolved
        # Nothing resolved yet / nothing came back: the two states that looked alike.
        v.partitions = []
        v.queue_running = v.queue_pending = None
        return v

    def _plain(self, view: object) -> str:
        return Text.from_markup(view.render()).plain  # type: ignore[attr-defined]

    def test_before_the_first_pass_it_says_it_is_working(self) -> None:
        out = self._plain(self._view(resolved=False))
        assert "querying partitions" in out, out
        assert "reading the queue" in out
        assert "controller busy" not in out
        assert "unavailable" not in out

    def test_after_a_failed_pass_it_says_what_actually_happened(self) -> None:
        out = self._plain(self._view(resolved=True))
        assert "squeue did not answer" in out, out
        assert "cluster partition info unavailable" in out
        # Still never a fabricated 0 running / 0 pending for a partition that
        # provably holds at least this job.
        assert "0 running" not in out
        # And no claim about why: "controller busy" was a health assertion we
        # cannot make from a failed query.
        assert "controller busy" not in out

    def test_the_loading_state_animates(self) -> None:
        """A static placeholder reads as stuck; the spinner is what says "working",
        and it is the same one the estimate line uses."""
        view = self._view(resolved=False)
        frames = set()
        for f in range(4):
            view.frame = f  # type: ignore[attr-defined]
            line = next(ln for ln in self._plain(view).splitlines() if "querying partitions" in ln)
            frames.add(line.strip()[0])
        assert len(frames) > 1, frames

    def test_data_present_overrides_both(self) -> None:
        view = self._view(resolved=True)
        view.partitions = pending._mock_partitions("build")  # type: ignore[attr-defined]
        view.queue_running, view.queue_pending = 12, 5  # type: ignore[attr-defined]
        out = self._plain(view)
        assert "12" in out and "5" in out
        assert "querying" not in out and "unavailable" not in out

    def test_the_loading_state_is_ascii_clean(self) -> None:
        view = self._view(resolved=False)
        view.config = SlurmwatchConfig(ascii_mode=True)  # type: ignore[attr-defined]
        out = self._plain(view)
        assert out.isascii(), [c for c in out if not c.isascii()]


class TestASlowResolvePassCompletes:
    """Found auditing round 38. The refresh timer fired `run_worker(..., exclusive=
    True)` every 10 s, and Textual's `exclusive` CANCELS the in-flight worker of that
    group — so a pass slower than 10 s was killed and restarted forever and the panels
    never populated. One slow `sinfo` gets there: the partition resolve is ~2 s on 76
    partitions, scales with partition count, and each Slurm command is allowed 15 s.
    The round-38 fix makes that failure a spinner that spins for ever, so the guard
    matters more, not less."""

    def _screen(self) -> Any:
        from slurmwatch.tui import PendingScreen

        return PendingScreen(pending._mock_pending_job("777"), SlurmwatchConfig())

    def test_a_tick_is_skipped_while_a_pass_is_running(self) -> None:
        screen = self._screen()
        started: list[int] = []

        def _worker(coro: Any, *a: object, **k: object) -> None:
            coro.close()  # we never run it; closing keeps the suite warning-free
            started.append(1)

        screen.run_worker = _worker
        screen._refresh_in_flight = True
        screen._kick_refresh()
        assert started == [], "the in-flight pass must not be cancelled"
        screen._refresh_in_flight = False
        screen._kick_refresh()
        assert started == [1], "and a free tick still refreshes"

    def test_the_flag_clears_even_when_the_pass_raises(self) -> None:
        """Otherwise the guard latches and the view freezes for good — strictly worse
        than the cancellation it replaces."""
        import asyncio

        screen = self._screen()

        async def _boom() -> None:
            raise RuntimeError("controller went away")

        screen._refresh_once = _boom
        with pytest.raises(RuntimeError):
            asyncio.run(screen._refresh())
        assert screen._refresh_in_flight is False

    def test_the_flag_clears_on_cancellation(self) -> None:
        """Screen teardown cancels the worker; that must not leave it latched."""
        import asyncio

        screen = self._screen()

        async def _hang() -> None:
            await asyncio.sleep(60)

        screen._refresh_once = _hang

        async def _drive() -> None:
            task = asyncio.ensure_future(screen._refresh())
            await asyncio.sleep(0)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(_drive())
        assert screen._refresh_in_flight is False

    def test_a_real_pass_sets_the_flag_while_it_runs(self) -> None:
        """The guard is only worth anything if an ACTUAL pass raises it — asserting on
        a hand-set flag would pass with the flag never assigned at all."""
        import asyncio

        screen = self._screen()
        started: list[int] = []

        def _worker(coro: Any, *a: object, **k: object) -> None:
            coro.close()  # we never run it; closing keeps the suite warning-free
            started.append(1)

        screen.run_worker = _worker
        gate = asyncio.Event()
        seen: list[bool] = []

        async def _slow() -> None:
            seen.append(screen._refresh_in_flight)
            screen._kick_refresh()
            gate.set()

        screen._refresh_once = _slow
        asyncio.run(screen._refresh())
        assert seen == [True], "the pass itself must raise the flag"
        assert started == [], "and a tick during that pass must start nothing"

    def test_a_finished_screen_never_refreshes(self) -> None:
        screen = self._screen()
        started: list[int] = []

        def _worker(coro: Any, *a: object, **k: object) -> None:
            coro.close()  # we never run it; closing keeps the suite warning-free
            started.append(1)

        screen.run_worker = _worker
        screen._done = True
        screen._kick_refresh()
        assert started == []


class TestASmallMemoryRequestIsNotRenderedAsZero:
    """SW-4's shape, in the three renderers its fix never reached. All three showed a
    Slurm memory request in hardcoded GiB, so `--mem=20M` read "0.0 GiB" — a request
    for no memory at all, on a view whose whole job is explaining what the job asked
    for. The live gauges were fixed for exactly this; these were not."""

    def _job(self, mib: int) -> Any:
        job = pending._mock_pending_job("777")
        job.req_mem_bytes = mib * 1024**2
        return job

    def _pending_view(self, mib: int) -> str:
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = self._job(mib)
        v.config = SlurmwatchConfig()
        v.resolved = True
        v.partitions = pending._mock_partitions(v.job.partition)
        return Text.from_markup(v.render()).plain

    def test_the_tui_request_chip_keeps_its_own_unit(self) -> None:
        out = self._pending_view(20)
        assert "20.0 MiB" in out, out
        assert "0.0 GiB" not in out

    def test_a_large_request_still_reads_in_gib(self) -> None:
        """The familiar rendering for the tens-of-GiB jobs a cluster mostly runs."""
        assert "64.0 GiB" in self._pending_view(65536)

    def test_the_plain_text_report_agrees_with_the_tui(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both renderers, one helper — the SW-4 lesson was that fixing one is half a
        fix, and this report is what a login-node user redirects to a file."""
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: [])
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        cli._print_pending_summary(self._job(20), ascii_mode=True)
        out = capsys.readouterr().out + capsys.readouterr().err
        assert "20.0 MiB" in out, out
        assert "0.0 GiB" not in out

    def test_the_foreign_job_view_allocation_line_too(self) -> None:
        """Another user's job: the same figure, read off scontrol rather than squeue."""
        from slurmwatch.model import JobContext
        from slurmwatch.tui import ForeignJobView

        view = ForeignJobView()
        view.job_ctx = JobContext(
            job_id="1",
            username="someone",
            partition="p",
            nodelist="cn001",
            hostname="login1",
            cpus_allocated=2,
            mem_limit_bytes=20 * 1024**2,
            gpu_count_requested=0,
            gpu_indices=[],
        )
        view.config = SlurmwatchConfig()
        out = Text.from_markup(view.render()).plain
        assert "20.0 MiB" in out, out
        assert "0.0 GiB" not in out


class TestAGpuLessPartitionSaysSoWhateverIsIdle:
    """SW-28's argument, applied to the checks it left behind the aggregate test.

    Measured on a Booth cluster (7 partitions, a 1-GPU job): the three GPU-LESS
    partitions came back as `standard: no GPU`, `highmem: no GPU` and
    `test: no room` — the last only because `test` had no fully idle node that
    minute. Same hardware, same request, two different verdicts, and the transient
    one invites a wait that can never end. `cron` reached "time limit" for the
    mirror-image reason: it happened to have a free node. Every permanent test now
    runs before the transient ones.
    """

    @staticmethod
    def _job(
        req_gpus: int = 1,
        req_gpu_type: str = "",
        req_cpus: int = 4,
        req_nodes: int = 1,
        time_limit_seconds: int = 900,
    ) -> PendingJob:
        return PendingJob(
            job_id="1",
            raw_job_id="1",
            name="j",
            username="u",
            partition="other",
            qos="",
            account="",
            reason="Resources",
            submit_time=None,
            start_time_estimate=None,
            priority=100,
            req_cpus=req_cpus,
            req_nodes=req_nodes,
            req_mem_bytes=0,
            req_gpus=req_gpus,
            req_gpu_type=req_gpu_type,
            time_limit_seconds=time_limit_seconds,
        )

    @pytest.mark.parametrize(("idle_nodes", "cpus_idle"), [(0, 0), (0, 36), (5, 723), (40, 5000)])
    def test_no_gpu_beats_no_room_however_busy_the_partition_is(
        self, idle_nodes: int, cpus_idle: int
    ) -> None:
        part = PartitionResources(
            "cpu-only",
            True,
            idle_nodes=idle_nodes,
            cpus_idle=cpus_idle,
            max_node_cpus=40,
            has_gpus=False,
        )
        assert fit_blocker(self._job(), part) == "no GPU", (idle_nodes, cpus_idle)

    @pytest.mark.parametrize(("idle_nodes", "cpus_idle"), [(0, 0), (5, 779)])
    def test_a_short_time_limit_is_permanent_too(self, idle_nodes: int, cpus_idle: int) -> None:
        """`cron` (MaxTime 5m) must read "time limit" whether or not a node is free."""
        part = PartitionResources(
            "cron",
            True,
            idle_nodes=idle_nodes,
            cpus_idle=cpus_idle,
            max_node_cpus=64,
            timelimit_seconds=300,
        )
        assert fit_blocker(self._job(req_gpus=0), part) == "time limit", idle_nodes

    def test_a_wrong_gpu_type_is_named_not_absorbed_into_no_room(self) -> None:
        part = PartitionResources(
            "l40s",
            True,
            idle_nodes=0,
            cpus_idle=0,
            max_node_cpus=64,
            has_gpus=True,
            gpu_types=["l40s"],
            max_node_gpus=8,
        )
        assert fit_blocker(self._job(req_gpu_type="h100"), part) == "no h100"

    def test_too_few_gpus_per_node_outranks_scarcity(self) -> None:
        part = PartitionResources(
            "small-gpu",
            True,
            idle_nodes=0,
            cpus_idle=0,
            max_node_cpus=64,
            has_gpus=True,
            max_node_gpus=4,
        )
        assert fit_blocker(self._job(req_gpus=8), part) == "too few GPUs"

    def test_gpus_busy_is_transient_so_a_shape_misfit_wins(self) -> None:
        """A node too small for the job must say so, not blame this minute's GPU use."""
        part = PartitionResources(
            "gpu",
            True,
            idle_nodes=0,
            cpus_idle=64,
            max_node_cpus=8,
            has_gpus=True,
            max_node_gpus=8,
            free_gpus_per_node=[0, 0],
            gpu_detail=True,
        )
        assert fit_blocker(self._job(req_gpus=1, req_cpus=64), part) == "node too small"

    def test_gpus_busy_still_wins_over_no_room_when_the_shape_fits(self) -> None:
        """The complement: with nothing permanent wrong, the exact cause is named."""
        part = PartitionResources(
            "gpu",
            True,
            idle_nodes=0,
            cpus_idle=0,
            max_node_cpus=64,
            has_gpus=True,
            max_node_gpus=8,
            free_gpus_per_node=[0, 0],
            gpu_detail=True,
        )
        assert fit_blocker(self._job(req_gpus=1), part) == "GPUs busy"

    def test_a_partition_that_genuinely_fits_still_fits(self) -> None:
        part = PartitionResources(
            "gpu",
            True,
            idle_nodes=2,
            cpus_idle=128,
            max_node_cpus=64,
            has_gpus=True,
            gpu_types=["h100"],
            max_node_gpus=4,
            idle_node_cpus=128,
            max_idle_node_cpus=64,
            timelimit_seconds=172800,
        )
        assert fit_blocker(self._job(req_gpu_type="h100"), part) == ""


class TestAPermanentMisfitIsNotCalledTransient:
    """SW-28: the per-node CPU test sat AFTER the aggregate one, which shadowed it.
    `req_cpus > cpus_avail` is true for any partition with fewer than req_cpus idle
    cores in TOTAL, so a 999-CPU request against a 64-CPU-max cluster was labelled
    "no room" — transient scarcity — everywhere except the three partitions that
    happened to have >999 cores idle at that instant. 20 of 24 verdicts were wrong,
    and the closing tip promised the job "will start once resources free up"."""

    @staticmethod
    def _job(req_cpus: int = 999, req_nodes: int = 1, mem: int = 0) -> PendingJob:
        return PendingJob(
            job_id="1",
            raw_job_id="1",
            name="j",
            username="u",
            partition="other",
            qos="",
            account="",
            reason="PartitionConfig",
            submit_time=None,
            start_time_estimate=None,
            priority=100,
            req_cpus=req_cpus,
            req_nodes=req_nodes,
            req_mem_bytes=mem,
            req_gpus=0,
            req_gpu_type="",
            time_limit_seconds=300,
        )

    @pytest.mark.parametrize("idle_cores", [0, 27, 812, 960, 1015, 1100, 5000])
    def test_the_verdict_does_not_depend_on_how_much_is_idle(self, idle_cores: int) -> None:
        """The reporter's named test: the same hardware and the same request must give
        the same answer whatever happens to be idle at that instant — before this, the
        label flipped between "no room" and "node too small" across exactly these
        values, and the same command an hour later relabelled a partition."""
        part = PartitionResources("p", True, idle_nodes=40, cpus_idle=idle_cores, max_node_cpus=64)
        assert fit_blocker(self._job(), part) == "node too small", idle_cores

    def test_a_request_that_fits_a_node_is_still_transient_when_cores_are_busy(self) -> None:
        """The complement: reordering must not turn ordinary scarcity into a permanent
        verdict."""
        part = PartitionResources("p", True, idle_nodes=0, cpus_idle=4, max_node_cpus=64)
        assert fit_blocker(self._job(req_cpus=16), part) == "no room"

    def test_per_node_memory_is_decided_before_scarcity_too(self) -> None:
        """The same shadowing applied to the memory test sitting beside it."""
        part = PartitionResources(
            "p",
            True,
            idle_nodes=40,
            cpus_idle=0,  # would have short-circuited to "no room"
            max_node_cpus=64,
            max_node_mem_bytes=8 * 1024**3,
        )
        assert fit_blocker(self._job(req_cpus=1, mem=64 * 1024**3), part) == "node too small"

    def test_a_multi_node_request_divides_before_comparing(self) -> None:
        """999 CPUs over 40 nodes is 25 per node, which a 64-CPU node holds — the
        reorder must not make every large aggregate request "node too small"."""
        part = PartitionResources("p", True, idle_nodes=40, cpus_idle=4000, max_node_cpus=64)
        assert fit_blocker(self._job(req_nodes=40), part) == ""

    def test_a_down_partition_still_answers_down_first(self) -> None:
        part = PartitionResources("p", False, idle_nodes=40, cpus_idle=0, max_node_cpus=64)
        assert fit_blocker(self._job(), part) == "down"

    @pytest.mark.parametrize(
        ("blocker", "permanent"),
        [
            ("", False),
            ("no room", False),
            ("GPUs busy", False),
            ("down", False),
            ("node too small", True),
            ("time limit", True),
            ("no GPU", True),
            ("no a100", True),
        ],
    )
    def test_which_blockers_waiting_can_clear(self, blocker: str, permanent: bool) -> None:
        """The tip reads from this: a node's size, a partition's wall-clock ceiling and
        its hardware type do not change because you waited; free cores and busy GPUs
        do."""
        assert blocker_is_permanent(blocker) is permanent


class TestABlockedJobDoesNotAdvertiseCapacity:
    """SW-29: for a held-like job the WHERE table printed 51 "FITS NOW" rows between a
    `Why` line saying it will never start and a `Tip` saying a partition change cannot
    help. Nothing there was false — those partitions do have room — but the screen's
    largest element answered a question the two lines bracketing it call moot, most
    starkly for a BeginTime job deferred 24 hours whose answer to "can run now?" was
    yes, fifty-one times."""

    REASONS = ["DependencyNeverSatisfied", "JobHeldUser", "BeginTime", "ReservationNotAvailable"]

    @staticmethod
    def _parts() -> list[PartitionResources]:
        return [
            PartitionResources(
                "build", True, idle_nodes=1, cpus_idle=28, max_node_cpus=48, is_current=True
            ),
            PartitionResources("broadwl", True, idle_nodes=53, cpus_idle=1101, max_node_cpus=48),
            PartitionResources("econ", True, idle_nodes=1, cpus_idle=28, max_node_cpus=48),
        ]

    @pytest.mark.parametrize("reason", REASONS)
    def test_the_tui_does_not_claim_partitions_can_run_it_now(self, reason: str) -> None:
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = pending._mock_pending_job("777")
        v.job.reason = reason
        v.job.req_gpus = 0
        v.config = SlurmwatchConfig()
        v.resolved = True
        v.partitions = self._parts()
        out = Text.from_markup(v.render()).plain
        assert "FITS NOW" not in out and "YES" not in out, out
        assert "capacity is not the constraint" in out, out

    @pytest.mark.parametrize("reason", REASONS)
    def test_the_text_report_does_not_either(
        self, reason: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Both renderers, since a fix to one is half a fix."""
        job = pending._mock_pending_job("777")
        job.reason = reason
        job.req_gpus = 0
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: self._parts())
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        cli._print_pending_summary(job)
        out = capsys.readouterr().out
        assert "FITS NOW" not in out, out
        assert "capacity is not the constraint" in out

    def test_a_capacity_wait_still_gets_the_table(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The table is correct and valuable for a capacity or priority wait — round 55's
        oversized job got zero FITS NOW rows, and a Priority-blocked job genuinely wants
        to know which partition has room. Only the held-like case is suppressed."""
        job = pending._mock_pending_job("777")
        job.reason = "Priority"
        job.req_gpus = 0
        job.req_cpus = 4
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: self._parts())
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        cli._print_pending_summary(job)
        out = capsys.readouterr().out
        assert "FITS NOW" in out, out
        assert "capacity is not the constraint" not in out

    def test_the_tui_gives_the_same_cap_tip(self) -> None:
        """Both renderers, again — I tested the text twin first and a mutation removing
        this one survived."""
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = pending._mock_pending_job("777")
        # A per-USER cap, deliberately: the per-JOB variants are no longer usage caps
        # (nothing about your usage is in the way — the request itself is too big), so
        # using one here would test the cap tip with an input that must not produce it.
        v.job.reason = "AssocGrpCpuLimit"
        v.job.req_gpus = 0
        v.job.req_cpus = 4
        v.config = SlurmwatchConfig()
        v.resolved = True
        v.partitions = self._parts()
        out = Text.from_markup(v.render()).plain
        assert "a usage limit is capping this job" in out, out
        assert "sacctmgr show assoc" in out
        assert "requeue with" not in out
        # The TUI's verdict wording is "YES"; the text report's is "FITS NOW".
        assert "YES" in out, "a cap is not a hold — the table still belongs here"

    def test_a_usage_capped_job_keeps_the_table_but_gets_the_cap_tip(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A usage cap is not a hold: the job stays priority-ordered, so the table and
        the estimate remain — but the tip must not point at free room, which is not
        what it lacks."""
        job = pending._mock_pending_job("777")
        job.reason = "QOSMaxNodePerUserLimit"
        job.req_gpus = 0
        job.req_cpus = 4
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: self._parts())
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        cli._print_pending_summary(job)
        out = capsys.readouterr().out
        assert "a usage limit is capping this job" in out, out
        assert "sacctmgr show assoc" in out, "name the check that answers it for this site"
        assert "requeue with" not in out and "scontrol update" not in out


class TestTheArrayHalvesAreStatedNotImplied:
    """The no-telemetry schema promises `array_job_id`/`array_task_id`, and the
    foreign-job payload fills them — but the PENDING payload left them None, so a log
    grouped by array silently dropped every queued task even though the id it already
    had (`54222358_1`) says both halves."""

    @staticmethod
    def _job(job_id: str) -> PendingJob:
        job = pending._mock_pending_job("777")
        job.job_id = job_id
        job.raw_job_id = job_id.split("_")[0].split("+")[0]
        return job

    @pytest.mark.parametrize(
        ("job_id", "base", "task"),
        [("54222358_1", "54222358", "1"), ("12345_0", "12345", "0")],
    )
    def test_an_array_task_states_both_halves(self, job_id: str, base: str, task: str) -> None:
        facts = cli._pending_facts(self._job(job_id))
        assert facts["array_job_id"] == base
        assert facts["array_task_id"] == task
        assert facts["job_id"] == job_id, "and the composite id is still there"

    @pytest.mark.parametrize("job_id", ["54364986", "123+1", "54222358_[1-9%3]"])
    def test_anything_that_is_not_one_task_stays_empty(self, job_id: str) -> None:
        """A het component is not an array, and a bracketed RANGE is not one task —
        reporting a task id for either would be inventing it."""
        facts = cli._pending_facts(self._job(job_id))
        assert facts["array_job_id"] is None
        assert facts["array_task_id"] is None

    def test_the_telemetry_payload_states_them_too(self) -> None:
        """All three payloads agree now: the running one carried neither."""
        from slurmwatch.collector import TelemetryCollector
        from slurmwatch.model import JobContext

        ctx = JobContext(
            job_id="12345_3",
            username="u",
            partition="p",
            nodelist="cn1",
            hostname="cn1",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
            array_job_id="12345",
            array_task_id="3",
        )
        snap = TelemetryCollector(ctx)._collect_snapshot_sync()
        assert (snap.array_job_id, snap.array_task_id) == ("12345", "3")
        payload = json.loads(snap.to_json())
        assert payload["array_job_id"] == "12345" and payload["array_task_id"] == "3"
        row = dict(
            zip(
                snap.csv_header(0),
                snap.to_csv_row(0),
                strict=True,
            )
        )
        assert row["array_job_id"] == "12345" and row["array_task_id"] == "3"


class TestSuppressingTheTableKeepsTheTip:
    """SW-29 suppressed the WHERE table for a held-like job — and, because the tip
    ladder was NESTED inside the table branch in both renderers, took the tip with it.
    That tip is the actionable line round 56 called correct: "moving to another
    partition won't start this job — it isn't waiting on free capacity". Found by
    testing a comment which claimed the two renderers mirror each other: they did
    agree, and both were wrong the same way."""

    PARTS = [
        PartitionResources(
            "cur", True, idle_nodes=0, cpus_idle=0, max_node_cpus=48, is_current=True
        ),
        PartitionResources("other", True, idle_nodes=9, cpus_idle=400, max_node_cpus=48),
    ]

    @staticmethod
    def _job(reason: str) -> PendingJob:
        job = pending._mock_pending_job("777")
        job.reason = reason
        job.req_gpus = 0
        job.req_cpus = 4
        return job

    @pytest.mark.parametrize("reason", ["DependencyNeverSatisfied", "JobHeldUser", "BeginTime"])
    def test_the_text_report_keeps_the_tip(
        self, reason: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: self.PARTS)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        cli._print_pending_summary(self._job(reason))
        out = capsys.readouterr().out
        assert "capacity is not the constraint" in out, "the table is still suppressed"
        assert "won't start this job" in out, "but the tip must survive the suppression"
        assert "FITS NOW" not in out

    @pytest.mark.parametrize("reason", ["DependencyNeverSatisfied", "JobHeldUser", "BeginTime"])
    def test_the_tui_keeps_it_too(self, reason: str) -> None:
        from slurmwatch.tui import PendingView

        view = PendingView()
        view.job = self._job(reason)
        view.config = SlurmwatchConfig()
        view.resolved = True
        view.partitions = self.PARTS
        out = Text.from_markup(view.render()).plain
        assert "capacity is not the constraint" in out
        assert "won't start this job" in out, out
        assert "YES" not in out

    def test_a_capacity_wait_is_untouched(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: self.PARTS)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        cli._print_pending_summary(self._job("Priority"))
        out = capsys.readouterr().out
        assert "FITS NOW" in out, "the table belongs here"
        assert "has room for this request" in out, "and so does the requeue suggestion"


class TestPartitionMoveCommandIsComplete:
    """SW-32: the requeue tip printed a command that made the job strictly worse.

    Reported with the proof attached — the command was run verbatim on a pending job:

        $ scontrol update JobId=48850850 Partition=astroplasmas
        $ squeue -h -j 48850850
          48850850 PENDING astroplasmas (InvalidQOS)

    `Resources` clears when a node frees up. `InvalidQOS` never clears. The advice was
    right about the destination (adding `QOS=astroplasmas` starts the job immediately)
    and incomplete about how to get there: a partition move does not move the QOS, and
    on a site where QOS names track partition names — both clusters measured here — the
    destination then rejects the job's inherited QOS. Every partition-move tip on such
    a cluster produced InvalidQOS.

    `fit_blocker`'s docstring already said the estimate "can't see QOS/account limits".
    The defect was that the hedge never reached the screen the imperative was printed on.
    """

    REAL_SHAPES = {
        # measured on midway2 (the reporting cluster): partition-keyed rows
        "partition-keyed": (
            "build|build\nastroplasmas|astroplasmas\nbroadwl|broadwl,broadwl-large,debug"
        ),
        # measured on midway3: ONE cluster-level row, empty partition, 92 QOS names
        "cluster-level": "|aaz,astroplasmas,build,caslake,test",
    }

    @pytest.mark.parametrize("shape", list(REAL_SHAPES))
    def test_both_real_association_layouts_parse(
        self, shape: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A parser that assumed either shape alone would read the other as "no
        associations" and stop offering every partition."""
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda *a, **k: self.REAL_SHAPES[shape])
        table = pending.resolve_user_associations("someone")
        assert table is not None
        assert pending.qos_for_partition("astroplasmas", table) == "astroplasmas"

    def test_partition_move_tip_carries_the_destination_qos(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assoc = {"astroplasmas": ["astroplasmas"]}
        cmd = pending.partition_move_command("48850850", "astroplasmas", assoc)
        assert cmd == "scontrol update JobId=48850850 Partition=astroplasmas QOS=astroplasmas"
        assert pending.partition_move_caveat("astroplasmas", assoc) == ""

    def test_a_qos_named_after_the_partition_wins_over_its_siblings(self) -> None:
        """`broadwl` allows broadwl, broadwl-large and debug. The convention that
        creates this bug is also what resolves it: prefer the one named for the
        partition."""
        assoc = {"broadwl": ["broadwl-large", "broadwl", "debug"]}
        assert pending.qos_for_partition("broadwl", assoc) == "broadwl"

    def test_several_unrelated_qos_names_are_not_guessed_between(self) -> None:
        assoc = {"gpu": ["short", "long"]}
        assert pending.qos_for_partition("gpu", assoc) is None
        cmd = pending.partition_move_command("1", "gpu", assoc)
        assert "QOS=" not in cmd
        assert "does not move with it" in pending.partition_move_caveat("gpu", assoc)

    def test_an_unreadable_table_softens_the_advice_instead_of_guessing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """None means unknown, and unknown must not become a confident QOS clause —
        nor withhold a partition that would have worked."""
        monkeypatch.setattr(pending, "_is_mock", lambda: False)

        def _boom(*_a: object, **_k: object) -> str:
            raise SlurmCommandError("sacctmgr: command not found")

        monkeypatch.setattr(pending, "_run_slurm_cmd", _boom)
        assert pending.resolve_user_associations("u") is None
        assert "QOS=" not in pending.partition_move_command("1", "gpu", None)
        assert "check your QOS" in pending.partition_move_caveat("gpu", None)
        assert pending.partition_allowed_by_assoc("anything", None) is True

    def test_partitions_without_an_association_are_not_suggested(self) -> None:
        """Option 2 of the report: a partition with room the user cannot submit to is
        not an alternative, whatever its idle-core count says."""
        assoc = {"build": ["build"], "astroplasmas": ["astroplasmas"]}
        assert pending.partition_allowed_by_assoc("astroplasmas", assoc) is True
        assert pending.partition_allowed_by_assoc("someone-elses-pi-partition", assoc) is False
        # a cluster-level row applies everywhere, so nothing is withheld there
        assert pending.partition_allowed_by_assoc("anything", {"": ["normal"]}) is True

    PARTS = [
        PartitionResources(
            "cur", True, idle_nodes=0, cpus_idle=0, max_node_cpus=48, is_current=True
        ),
        PartitionResources("astroplasmas", True, idle_nodes=9, cpus_idle=400, max_node_cpus=48),
    ]

    @staticmethod
    def _job() -> PendingJob:
        job = pending._mock_pending_job("48850850")
        job.reason = "Resources"
        job.req_gpus = 0
        job.req_cpus = 4
        return job

    def _wire(self, monkeypatch: pytest.MonkeyPatch, assoc: object) -> None:
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: self.PARTS)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_user_associations", lambda *a, **k: assoc)

    def test_the_text_report_emits_the_complete_command(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._wire(monkeypatch, {"astroplasmas": ["astroplasmas"]})
        cli._print_pending_summary(self._job())
        out = capsys.readouterr().out
        assert "Partition=astroplasmas QOS=astroplasmas" in out, out

    def test_the_dashboard_emits_the_same_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The tip lived in two renderers and was wrong in both, which is why the
        command is built by one shared helper now (the units.py lesson)."""
        from slurmwatch import tui as tui_mod
        from slurmwatch.tui import PendingView

        # BOTH wirings, deliberately: the screen's poll hands the table to the view
        # (`view.assoc`, D23) and the module-level resolver is what it used to call
        # from inside `render()`. Setting both keeps this test a statement about the
        # rendered command rather than about where the table came from.
        monkeypatch.setattr(
            tui_mod, "resolve_user_associations", lambda *a, **k: {"astroplasmas": ["astroplasmas"]}
        )
        view = PendingView()
        view.job = self._job()
        view.config = SlurmwatchConfig()
        view.resolved = True
        view.partitions = self.PARTS
        view.assoc = {"astroplasmas": ["astroplasmas"]}
        out = Text.from_markup(view.render()).plain
        assert "Partition=astroplasmas QOS=astroplasmas" in out, out

    def test_a_partition_the_user_cannot_use_is_dropped_from_the_tip(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._wire(monkeypatch, {"build": ["build"]})  # no astroplasmas association
        cli._print_pending_summary(self._job())
        out = capsys.readouterr().out
        assert "astroplasmas has room" not in out, out


class TestPrintedCommandsCarryTheirJobId:
    """The weaker half of the class SW-32 opened: a command that needs editing.

    `scontrol release <jobid>` is valid syntax — but run verbatim, as the report's
    method demands, it answers `too few arguments for keyword:release`. The explanation
    table is keyed by REASON, so it can only hold a placeholder; every caller, though,
    has the job in hand. Every other command this tool prints (the partition move, the
    srun recovery hint) carries the real id already.
    """

    @pytest.mark.parametrize("reason", ["JobHeldUser", "launch failed requeued held"])
    def test_the_id_is_substituted_when_it_is_known(self, reason: str) -> None:
        out = pending.explain_reason(reason, False, "48850850")
        assert "scontrol release 48850850" in out, out
        assert "<jobid>" not in out

    def test_the_placeholder_survives_when_no_id_is_given(self) -> None:
        """The signature stays backward compatible: a caller without a job (the
        machine-readable reason string) still gets the documented placeholder rather
        than a mangled command."""
        assert "<jobid>" in pending.explain_reason("JobHeldUser")

    def test_the_text_report_and_the_dashboard_both_substitute(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from slurmwatch.tui import PendingView

        job = pending._mock_pending_job("48850850")
        job.reason = "JobHeldUser"
        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: [])
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        cli._print_pending_summary(job)
        assert "scontrol release 48850850" in capsys.readouterr().out

        view = PendingView()
        view.job = job
        view.config = SlurmwatchConfig()
        view.resolved = True
        view.partitions = []
        assert "scontrol release 48850850" in Text.from_markup(view.render()).plain


class TestPendingViewDoesNotQueueIndependentQueries:
    """The pending view's cost is Slurm round-trips, and it was paying for them twice.

    Measured against a real 4600-job queue on midway3, first full frame went from a
    ~3.7 s median (2.0-7.4 s) to ~1.2 s. Neither fix is visible in the rendered
    output, so both are guarded here.
    """

    def test_the_qos_association_table_is_read_once_per_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It is DB configuration, and `PendingView._where` asks for it during RENDER.

        So the un-cached version ran a ~220 ms `sacctmgr` on the event loop every time
        the view redrew, and twice on the first paint (the cli summary asks too).
        """
        calls: list[list[str]] = []

        def _cmd(cmd: list[str], *a: object, **k: object) -> str:
            calls.append(cmd)
            return "|normal,build,test\n"

        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)

        first = pending.resolve_user_associations("youzhi")
        second = pending.resolve_user_associations("youzhi")
        assert first == second
        assert len(calls) == 1, f"asked sacctmgr {len(calls)} times for one user's QOS table"

        # A DIFFERENT user is a different question, and must still be asked.
        pending.resolve_user_associations("someone-else")
        assert len(calls) == 2

    def test_a_transient_failure_is_not_cached_as_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One sacctmgr hiccup must not soften every later answer for the whole session.

        `None` means UNKNOWN and permanently withholds advice, so latching it from a
        failed subprocess would be the SW-32 mistake with a longer blast radius.
        """
        state = {"fail": True}

        def _cmd(cmd: list[str], *a: object, **k: object) -> str:
            if state["fail"]:
                raise SlurmCommandError("slurmdbd is down")
            return "|normal,build\n"

        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_run_slurm_cmd", _cmd)

        assert pending.resolve_user_associations("youzhi") is None
        state["fail"] = False
        assert pending.resolve_user_associations("youzhi") == {"": ["normal", "build"]}

    async def test_partitions_counts_and_rank_are_resolved_concurrently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """They depend on the job and on nothing else, so they must not queue.

        A `threading.Barrier` is the whole test: if the three run concurrently they all
        arrive and it releases immediately. If any awaits another's result first, the
        first one to arrive waits out the timeout alone and trips the barrier broken —
        so a serial implementation fails here instead of merely being slow.
        """
        import threading

        from slurmwatch.tui import PendingApp

        barrier = threading.Barrier(3)
        arrived: list[str] = []
        broke: list[str] = []

        def _sync(name: str) -> None:
            arrived.append(name)
            try:
                barrier.wait(timeout=5.0)
            except threading.BrokenBarrierError:
                broke.append(name)

        job = pending._mock_pending_job("777")
        monkeypatch.setattr("slurmwatch.tui.resolve_pending_job", lambda jid: job)

        def _parts(part: object, acct: object, user: object) -> list[object]:
            _sync("partitions")
            return []

        def _counts(part: object) -> tuple[int, int]:
            _sync("counts")
            return (1, 2)

        def _rank(part: object, prio: object, job_id: object = None) -> tuple[int, int]:
            _sync("rank")
            return (3, 9)

        monkeypatch.setattr("slurmwatch.tui.resolve_cluster_partitions", _parts)
        monkeypatch.setattr("slurmwatch.tui.resolve_queue_counts", _counts)
        monkeypatch.setattr("slurmwatch.tui.resolve_priority_rank", _rank)

        app = PendingApp(job)
        async with app.run_test(size=(110, 40)) as pilot:
            for _ in range(60):
                await pilot.pause()
                await asyncio.sleep(0.03)
                if len(arrived) >= 3 and not barrier.n_waiting:
                    break

        assert sorted(arrived) == ["counts", "partitions", "rank"], arrived
        assert not broke, (
            f"{broke} waited out the barrier alone — the three resolves are running "
            "one after another again, so the view waits for their sum"
        )


class TestNoEstimateIsPromisedToAJobThatCanNeverStart:
    """The "When" line answered "when will it start?" with "calculating… (the
    scheduler estimates a start once the job has waited a few minutes)" for jobs whose
    own REQUEST is the blocker — a promise about an event that cannot happen, printed
    one line under a Why line that already said "waiting won't help".

    Measured on the live queue (Slurm 20.11.8, 87 partitions): every one of the 15
    `QOSMaxWallDurationPerJobLimit` jobs and the single `InvalidAccount` job reports
    `StartTime=N/A`, because the backfill scheduler never plans such a job at all. Job
    47297644 has been PENDING on that reason for 166 days:

        $ squeue -j 47297644 -o "%i|%T|%r|%S"
        47297644|PENDING|QOSMaxWallDurationPerJobLimit|N/A

        Why    QOSMaxWallDurationPerJobLimit - The request exceeds a per-JOB limit
               ... lower the request (--time / --cpus / --nodes); waiting won't help.
        When   calculating... (the scheduler estimates a start once the job has
               waited a few minutes)             <-- after 166 days
        Where  capacity is not the constraint - not shown (see the reason above)

    `capacity_is_irrelevant` already knew, which is how the WHERE table came to be
    suppressed on the same screen; only the line above it kept animating hope. In the
    TUI it was literally animated: `_tick_spinner` ran a spinner at ~8 fps forever,
    against its own comment that "a placeholder that correctly says 'still working'
    must not be able to say it indefinitely".
    """

    # The two families Slurm names apart from a usage cap: the request exceeds a
    # per-JOB ceiling, or the request is invalid as submitted.
    NEVER = [
        "QOSMaxWallDurationPerJobLimit",
        "AssocMaxCpuPerJobLimit",
        "InvalidAccount",
        "InvalidQOS",
        "BadConstraints",
    ]

    PARTS = [
        PartitionResources(
            "cur", True, idle_nodes=2, cpus_idle=96, max_node_cpus=48, is_current=True
        ),
    ]

    @staticmethod
    def _job(reason: str) -> PendingJob:
        job = pending._mock_pending_job("47297644")
        job.reason = reason
        job.req_gpus = 0
        job.req_cpus = 2
        job.start_time_estimate = None  # what Slurm reports for all of these
        return job

    def _report(self, reason: str, monkeypatch: pytest.MonkeyPatch) -> str:
        import io

        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: self.PARTS)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        buf = io.StringIO()
        cli._print_pending_summary(self._job(reason), stream=buf)
        return buf.getvalue()

    @pytest.mark.parametrize("reason", NEVER)
    def test_the_text_report_says_never_instead_of_calculating(
        self, reason: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = self._report(reason, monkeypatch)
        when = next(line for line in out.splitlines() if "When" in line)
        assert "never, as submitted" in when, when
        assert "calculating" not in out, (
            "the scheduler will never estimate a start for this job, so promising one "
            f"contradicts the Why line on the same screen: {when}"
        )

    @pytest.mark.parametrize("reason", NEVER)
    def test_the_tui_says_it_too(self, reason: str) -> None:
        from slurmwatch.tui import PendingView

        view = PendingView()
        view.job = self._job(reason)
        view.config = SlurmwatchConfig()
        view.resolved = True
        view.partitions = self.PARTS
        out = Text.from_markup(view.render()).plain
        assert "never, as submitted" in out, out
        assert "calculating" not in out, out

    def _tui_frames(self, reason: str) -> list[str]:
        """The same view at two spinner frames, rendered against a FROZEN clock.

        The comparison below is "does this line change with the frame index", and
        the line also carries a wall-clock figure -- `waiting 1h 30m so far`, at
        minute granularity. The two renders are microseconds apart, so normally it
        is the same string; under a loaded full-suite run a minute boundary can
        fall between them, `1h 30m` becomes `1h 31m`, and the test reddens for the
        one reason it is not about. Observed exactly once, in a suite run
        concurrent with three other repos' suites, and green 5/5 in isolation and
        alone at full suite.

        Freezing `time.time` removes the variable the assertion does not measure
        and leaves the one it does: the control below still requires the frames to
        DIFFER for a job the scheduler is planning, and that difference comes from
        `frame`, not from the clock.
        """
        from unittest.mock import patch

        from slurmwatch.tui import PendingView

        out = []
        with patch("time.time", return_value=1_788_400_000.0):
            for frame in (0, 3):
                view = PendingView()
                view.job = self._job(reason)
                view.config = SlurmwatchConfig()
                view.resolved = True
                view.partitions = self.PARTS
                view.frame = frame
                out.append(Text.from_markup(view.render()).plain)
        return out

    @pytest.mark.parametrize("reason", NEVER)
    def test_the_spinner_has_nothing_left_to_animate(self, reason: str) -> None:
        # `_tick_spinner` repaints at ~8fps only while an estimate is still coming, and
        # it read that off `is_held_like` alone — so for these reasons it span forever.
        # The static note is frame-independent, which is what "nothing to animate" is.
        a, b = self._tui_frames(reason)
        assert a == b, "the line still changes between frames, i.e. it is still spinning"

    def test_a_minute_rolling_over_between_frames_does_not_look_like_motion(
        self,
    ) -> None:
        """The mechanism behind the flake, pinned deterministically.

        A race cannot be neutered, so what is asserted instead is the CAUSE: a
        clock that advances a whole minute between the two renders must not change
        the line, because `frame` is the only thing this comparison is about. The
        ticking clock is installed OUTSIDE `_tui_frames`, so this passes only
        because the freeze inside it wins -- remove that freeze and this reddens
        with `waiting 1h 30m` against `waiting 1h 31m`.
        """
        from unittest.mock import patch

        ticks = iter([1_788_400_000.0 + 60.0 * i for i in range(64)])
        with patch("time.time", side_effect=lambda: next(ticks)):
            a, b = self._tui_frames("InvalidQOS")
        assert a == b, "a minute boundary between frames reads as the spinner moving"

    def test_the_frozen_clock_is_what_the_line_carries(self) -> None:
        """Vacuity guard: the line has to actually contain a wall-clock figure, or
        the test above would pass against any implementation. If the view stops
        printing the waiting time this should be revisited, not deleted."""
        a, _ = self._tui_frames("InvalidQOS")
        assert "waiting" in a, a

    def test_the_spinner_still_animates_a_real_wait(self) -> None:
        """Control. The spinner is correct for a job the scheduler IS planning, so it
        must keep moving there — the fix stops it only where it was lying."""
        a, b = self._tui_frames("Resources")
        assert a != b, "a genuine capacity wait must still animate"

    # ---- CONTROLS: true both before and after the fix -------------------------

    @pytest.mark.parametrize("reason", ["Resources", "Priority", "", "None"])
    def test_a_capacity_wait_still_gets_the_calculating_line(
        self, reason: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control. A job the scheduler IS working on has an estimate coming, so the
        "calculating…" placeholder is correct there and must survive untouched — the
        fix is about the two families where it is a false promise, not about the line."""
        out = self._report(reason, monkeypatch)
        assert "calculating" in out, out
        assert "never, as submitted" not in out, out

    def test_a_begin_time_job_keeps_its_real_future_estimate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control. A `--begin` job genuinely HAS a start time (live example: job
        56246550, `BeginTime`, StartTime=2026-09-06T00:30:00), so it must keep showing
        it and must not be swept into the "never" branch."""
        import io

        monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: self.PARTS)
        monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
        job = self._job("BeginTime")
        job.start_time_estimate = time.time() + 4 * 86400
        buf = io.StringIO()
        cli._print_pending_summary(job, stream=buf)
        out = buf.getvalue()
        assert "estimated start" in out and "never, as submitted" not in out, out

    def test_the_predicate_split_did_not_move_capacity_is_irrelevant(self) -> None:
        """Control. `capacity_is_irrelevant` was refactored to delegate to the new
        predicate; SW-29's decisions about the WHERE table must be bit-for-bit intact,
        including the deliberate exclusion of a usage cap."""
        for reason in [*self.NEVER, "DependencyNeverSatisfied", "JobHeldUser", "BeginTime"]:
            assert pending.capacity_is_irrelevant(reason) is True, reason
        for reason in ("Resources", "Priority", "", "QOSMaxCpuPerUserLimit"):
            assert pending.capacity_is_irrelevant(reason) is False, reason
        assert pending.is_usage_capped("QOSMaxCpuPerUserLimit") is True
        assert pending.is_usage_capped("QOSMaxWallDurationPerJobLimit") is False
        # A hold is not a "the request must change" case: releasing it starts the job.
        for reason in ("DependencyNeverSatisfied", "JobHeldUser", "BeginTime"):
            assert pending.request_must_change(reason) is False, reason


class TestNodeTooSmallMustBlameHardwareNotThisMinutesOccupancy:
    """A "node too small" verdict is a claim about HARDWARE — `blocker_is_permanent` reports it
    as forever, and the closing tip escalates a clean sweep of it to "no partition on
    this cluster can ever hold this request; it will not start as submitted".

    But it was being decided from `max_node_cpus` / `max_node_mem_bytes`, which are
    summed over schedulable (idle/mix) node lines only. A partition whose large nodes
    are ALLOC therefore reported small ones, and the permanent verdict was really a
    reading of who else is running — exactly the inversion SW-28 removed from the
    aggregate test, still live one paragraph below it.

    Measured on this cluster, `cobey-hm` (`sinfo -a -e -h -o "%R|%a|%D|%t|%C|%G|%l|%m|%c"`):

        cobey-hm|up|1|mix  |17/31/0/48|(null)|infinite|768000 |48
        cobey-hm|up|2|alloc|96/0/0/96 |(null)|infinite|768000 |48
        cobey-hm|up|1|alloc|48/0/0/48 |(null)|infinite|2046270|48
        cobey-hm|up|1|alloc|64/0/0/64 |(null)|infinite|2063994|64   <-- 64 CPU, 1.97 TiB
        cobey-hm|up|1|idle |0/48/0/48 |(null)|infinite|768000 |48

    so max_node_cpus read 48 and a `--cpus-per-task=64` job was told "node too small",
    permanently unrunnable, against a node that exists and is merely busy. `--mem`
    the same, at 750 GiB against a 1.97 TiB machine. Live on three partitions at the
    time of writing (`cobey-hm`, `lgagliardi-ld`, `andrewferguson-gpu`); an hour later
    it is a different three, which is the point.
    """

    SINFO = (
        # %R|%a|%D|%t|%C|%G|%l|%m|%c — cobey-hm as measured, verbatim.
        "cobey-hm|up|1|mix|17/31/0/48|(null)|infinite|768000|48\n"
        "cobey-hm|up|2|alloc|96/0/0/96|(null)|infinite|768000|48\n"
        "cobey-hm|up|1|alloc|48/0/0/48|(null)|infinite|2046270|48\n"
        "cobey-hm|up|1|alloc|64/0/0/64|(null)|infinite|2063994|64\n"
        "cobey-hm|up|1|idle|0/48/0/48|(null)|infinite|768000|48\n"
    )

    @staticmethod
    def _job(**kw: object) -> PendingJob:
        base: dict[str, object] = {
            "job_id": "1",
            "raw_job_id": "1",
            "name": "j",
            "username": "u",
            "partition": "cobey-hm",
            "qos": "",
            "account": "",
            "reason": "Resources",
            "submit_time": None,
            "start_time_estimate": None,
            "priority": None,
            "req_cpus": 1,
            "req_nodes": 1,
            "req_mem_bytes": 0,
            "req_gpus": 0,
            "req_gpu_type": "",
            "time_limit_seconds": None,
        }
        base.update(kw)
        return PendingJob(**base)  # type: ignore[arg-type]

    def _part(self, monkeypatch: pytest.MonkeyPatch) -> PartitionResources:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: self.SINFO)
        return resolve_cluster_partitions("cobey-hm")[0]

    def test_a_busy_big_node_still_counts_as_hardware(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        p = self._part(monkeypatch)
        assert p.max_config_node_cpus == 64, "the ALLOC 64-core node is still 64 cores"
        assert p.max_config_node_mem_bytes == 2063994 * 1024**2

    @pytest.mark.parametrize(
        "kw",
        [
            {"req_cpus": 64},  # fits the ALLOC node, not the free ones
            {"req_mem_bytes": 1500 * 1024**3},  # 1500 GiB: same story
        ],
    )
    def test_too_big_for_what_is_free_is_transient_not_forever(
        self, kw: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        p = self._part(monkeypatch)
        blocker = fit_blocker(self._job(**kw), p)
        assert blocker == "no room", (
            f"a node this size exists in cobey-hm (64 CPU / 1.97 TiB, ALLOC), so "
            f"{blocker!r} blames the hardware for someone else's job"
        )
        assert blocker_is_permanent(blocker) is False

    # ---- CONTROLS: true both before and after the fix -------------------------

    @pytest.mark.parametrize(
        "kw",
        [
            {"req_cpus": 999},  # SW-28's own case: no node anywhere is that wide
            {"req_mem_bytes": 4096 * 1024**3},  # 4 TiB: bigger than any node here
        ],
    )
    def test_too_big_for_the_hardware_is_still_forever(
        self, kw: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control (SW-28). The permanent verdict is the whole value of the screen
        when it is TRUE, so a request no node in the partition could ever hold must
        keep saying so — the fix narrows the label, it must not retire it."""
        p = self._part(monkeypatch)
        blocker = fit_blocker(self._job(**kw), p)
        assert blocker == "node too small", blocker
        assert blocker_is_permanent(blocker) is True

    def test_free_capacity_figures_are_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Control (F2/M5). The new field is read BEFORE the schedulability filter, so
        prove the filter still governs everything it governed: free cores and the
        schedulable per-node maxima must exclude the ALLOC nodes exactly as before, or
        a partition with no placeable node would read "FITS NOW"."""
        p = self._part(monkeypatch)
        assert p.cpus_idle == 31 + 48, "only the mix node's spare cores and the idle node"
        assert (p.idle_nodes, p.mix_nodes) == (1, 1)
        assert p.max_node_cpus == 48, "the ALLOC 64-core node is not free capacity"
        assert p.max_node_mem_bytes == 768000 * 1024**2
        assert p.total_nodes == 6  # totals still count every node
        # And a small job still fits, i.e. nothing above turned into a blocker.
        assert fit_blocker(self._job(req_cpus=8), p) == ""

    def test_an_unknown_config_max_keeps_the_permanent_label(self) -> None:
        """Control. A `PartitionResources` built by a caller (or an `sinfo` that gave
        no %m/%c) leaves the new field at 0, which must mean "unknown" and preserve the
        old verdict — an unreadable field cannot be allowed to soften a real one."""
        p = PartitionResources("p", True, idle_nodes=4, cpus_idle=256, max_node_cpus=8)
        assert p.max_config_node_cpus == 0
        assert fit_blocker(self._job(req_cpus=16), p) == "node too small"
        assert pending._shape_blocker(16, 0) == "node too small"
        assert pending._shape_blocker(16, 15) == "node too small"

    def test_the_label_turns_on_the_configured_maximum_alone(self) -> None:
        # The boundary: a request the biggest node in the partition can hold exactly is
        # a scheduling wait, one core over it is a hardware mismatch.
        assert pending._shape_blocker(16, 17) == "no room"
        assert pending._shape_blocker(16, 16) == "no room"
        assert pending._shape_blocker(17, 16) == "node too small"


class TestTheNeverTipsReasonMustBeAboutWhatBlocks:
    """The closing tip escalates to "no partition on this cluster can ever hold this
    request", and it explained that with ``(largest node: N CPU)`` — the only figure it
    had. For a `time limit`, `no GPU`, `no <type>` or `too few GPUs` blocker a core
    count explains nothing. Measured live: a `--gres=gpu:16 --cpus-per-task=1` job on a
    9-partition account view (every blocker `no GPU` or `too few GPUs`) was told "no
    partition on this cluster can ever hold this request (largest node: 128 CPU)" — a
    reason about cores, at a job that asked for one core.

    This tip is the strongest claim the screen makes, so its reason has to be about the
    thing that actually blocks: name the binding figure, or name none at all.
    """

    @staticmethod
    def _job(**kw: Any) -> PendingJob:
        d: dict[str, Any] = {
            "job_id": "1",
            "raw_job_id": "1",
            "name": "j",
            "username": "u",
            "partition": "cur",
            "qos": "",
            "account": "",
            "reason": "Resources",
            "submit_time": None,
            "start_time_estimate": None,
            "priority": 100,
            "req_cpus": 1,  # one core: nothing about this job is CPU-shaped
            "req_nodes": 1,
            "req_mem_bytes": 0,
            "req_gpus": 0,
            "req_gpu_type": "",
            "time_limit_seconds": 3600,
        }
        d.update(kw)
        return PendingJob(**d)

    @staticmethod
    def _part(**kw: Any) -> PartitionResources:
        # 48 cores / 180 GiB per node — the `caslake` shape on the cluster this was
        # measured on, so a misplaced parenthetical has a real number to quote.
        d: dict[str, Any] = {
            "idle_nodes": 0,
            "cpus_idle": 0,
            "max_node_cpus": 48,
            "max_node_mem_bytes": 180 * 1024**3,
            "is_current": True,
        }
        d.update(kw)
        return PartitionResources("cur", True, **d)

    def _tip(self, job: PendingJob, parts: list[PartitionResources]) -> str:
        """The rendered "can never hold this" tip line, and nothing else — the words
        "CPU" and "cores" appear all over the rest of the report."""
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = job
        v.config = SlurmwatchConfig()
        v.partitions = parts
        out = Text.from_markup(v.render()).plain
        line = next((ln for ln in out.splitlines() if "can ever hold this request" in ln), "")
        assert line, f"the tip under test did not fire:\n{out}"
        assert "will not start as submitted" in line, line
        return line.strip()

    def test_a_per_node_gpu_shortfall_is_not_explained_by_a_core_count(self) -> None:
        job = self._job(req_gpus=16)
        parts = [self._part(has_gpus=True, max_node_gpus=4)]
        assert fit_blocker(job, parts[0]) == "too few GPUs"
        tip = self._tip(job, parts)
        # The point of this test is the core count, and it is still not here. The
        # figure that IS here came later: silence was only ever the second-best answer
        # (see TestTheNeverTipNamesTheGpuAndWallClockFiguresToo), and `max_node_gpus`
        # is both true and the number `fit_blocker` refused the job against.
        assert "largest node: 4 GPU" in tip, tip
        assert "CPU" not in tip, tip

    def test_a_gpu_model_nobody_here_has_is_not_explained_by_a_core_count(self) -> None:
        # h100 against a cluster whose only typed GRES is gpu:a30 (probed with
        # `sinfo -a -e -h -o "%R|%G"`).
        job = self._job(req_gpus=1, req_gpu_type="h100")
        parts = [self._part(has_gpus=True, gpu_types=["a30"], max_node_gpus=4)]
        assert fit_blocker(job, parts[0]) == "no h100"
        tip = self._tip(job, parts)
        assert "largest node" not in tip, tip
        assert "CPU" not in tip, tip

    def test_a_gpu_less_cluster_is_not_explained_by_a_core_count(self) -> None:
        job = self._job(req_gpus=1)
        parts = [self._part()]
        assert fit_blocker(job, parts[0]) == "no GPU"
        tip = self._tip(job, parts)
        assert "largest node" not in tip, tip
        assert "CPU" not in tip, tip

    def test_a_walltime_ceiling_is_not_explained_by_a_core_count(self) -> None:
        job = self._job(time_limit_seconds=8 * 3600)
        parts = [self._part(timelimit_seconds=4 * 3600)]
        assert fit_blocker(job, parts[0]) == "time limit"
        tip = self._tip(job, parts)
        assert "largest node" not in tip, tip
        assert "CPU" not in tip, tip
        # ...and it now names the hour instead of naming nothing.
        assert "max wall-clock: 4:00:00" in tip, tip

    def test_a_core_count_blocker_still_names_the_node_size(self) -> None:
        """CONTROL — and the one that matters most. `(largest node: N CPU)` is the
        tip's only useful explanation when CPUs ARE the constraint, so dropping the
        parenthetical everywhere would satisfy the four tests above and leave this
        screen saying nothing. Passes before and after the fix."""
        job = self._job(req_cpus=999)
        parts = [self._part()]
        assert fit_blocker(job, parts[0]) == "node too small"
        assert "largest node: 48 CPU" in self._tip(job, parts)

    def test_a_memory_blocker_names_the_memory_and_not_the_cores(self) -> None:
        # `node too small` covers both per-node shapes, so a --mem request no node can
        # hold used to be explained by a core count too.
        job = self._job(req_mem_bytes=4 * 1024**4)
        parts = [self._part()]
        assert fit_blocker(job, parts[0]) == "node too small"
        tip = self._tip(job, parts)
        assert "largest node: 180.0 GiB RAM" in tip, tip
        assert "CPU" not in tip, tip

    def test_the_transient_tip_is_untouched(self) -> None:
        """CONTROL. A request this hardware CAN hold still gets told it will start —
        the escalated tip must not spread. Passes before and after the fix."""
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = self._job(req_cpus=16)
        v.config = SlurmwatchConfig()
        v.partitions = [self._part()]
        out = Text.from_markup(v.render()).plain
        assert "no partition currently has enough free capacity" in out, out
        assert "can ever hold" not in out

    def test_the_note_names_only_a_figure_the_request_exceeds(self) -> None:
        # The rule itself, at the helper: a figure is quoted only where the request
        # provably does not fit it.
        note = pending.permanent_blocker_note
        gpu_part = self._part(has_gpus=True, gpu_types=["a30"], max_node_gpus=4)
        assert note(self._job(req_gpus=16), [gpu_part]) == "largest node: 4 GPU"
        # A GPU request that FITS the node width names nothing — the `no <type>` shape
        # has no honest figure (see the same class's `_stays_silent` tests).
        assert note(self._job(req_gpus=1, req_gpu_type="h100"), [gpu_part]) == ""
        assert note(self._job(req_gpus=1), [self._part()]) == ""
        assert (
            note(self._job(time_limit_seconds=8 * 3600), [self._part(timelimit_seconds=1)])
            == "max wall-clock: 0:00:01"
        )
        assert note(self._job(req_cpus=999), [self._part()]) == "largest node: 48 CPU"
        # ...and with nothing measured at all (a caller-built row, or an `sinfo` that
        # gave no %c/%m) there is no figure to name, which is what it said before.
        assert note(self._job(req_cpus=999), [PartitionResources("cur", True)]) == ""
        assert note(self._job(req_cpus=999), []) == ""

    def test_the_node_size_quoted_is_the_one_the_cluster_owns(self) -> None:
        # The permanent verdict is decided against the CONFIGURED node width
        # (`max_config_node_cpus`), because a busy big node is still 64 cores wide —
        # so the tip has to quote that figure and not the schedulable maximum, or it
        # explains a hardware claim with this minute's occupancy.
        parts = [self._part(max_node_cpus=48, max_config_node_cpus=64)]
        assert fit_blocker(self._job(req_cpus=999), parts[0]) == "node too small"
        assert pending.permanent_blocker_note(self._job(req_cpus=999), parts) == (
            "largest node: 64 CPU"
        )
        assert "largest node: 64 CPU" in self._tip(self._job(req_cpus=999), parts)


class TestTheNeverTipNamesTheGpuAndWallClockFiguresToo:
    """The other half of the class above, which stopped one step short on purpose.

    `permanent_blocker_note` was added to stop a CPU count being quoted at a
    `time limit` / `no GPU` / `no <type>` / `too few GPUs` blocker, and it named the
    binding figure for the two shapes it could prove (CPU, RAM) and nothing for the
    rest — the right call while the alternative was a wrong number. But silence is
    still less than the tool knows: for a job refused because it asked for 8 GPUs
    where the widest node owns 4, `(largest node: 4 GPU)` is true, is the very number
    `fit_blocker` refused it against, and is the only thing on that screen that tells
    the user what to change.

    Two shapes stay silent, and the tests below pin that too, because "no figure" has
    to be a decision and not an omission:

    * `no GPU` — the figure would be 0, which is also what an `sinfo` with no %G
      reports, and `has_gpus`/`max_node_gpus` are read from unflagged node lines only
      (live: one `down*` and one `drained*` gpu:4 line), so a partition whose GPU
      nodes are all flagged reads as GPU-less. "Nothing here has a GPU" would then be
      a hardware claim built on this minute's node health.
    * `no <type>` — the honest note would be the models that DO exist, and
      `gpu_types` cannot supply them: of the 20 partitions here that have GPUs, 19
      report the untyped `gpu:2`/`gpu:4` form and only one names a model (`a30`), so
      "available: a30" would hide every other GPU on the cluster.
    """

    @staticmethod
    def _job(**kw: Any) -> PendingJob:
        d: dict[str, Any] = {
            "job_id": "1",
            "raw_job_id": "1",
            "name": "j",
            "username": "u",
            "partition": "gpu",
            "qos": "",
            "account": "",
            "reason": "Resources",
            "submit_time": None,
            "start_time_estimate": None,
            "priority": 100,
            "req_cpus": 1,
            "req_nodes": 1,
            "req_mem_bytes": 0,
            "req_gpus": 0,
            "req_gpu_type": "",
            "time_limit_seconds": 3600,
        }
        d.update(kw)
        return PendingJob(**d)

    @staticmethod
    def _account_view() -> list[PartitionResources]:
        """The live `beagle3-users` account view, as `resolve_cluster_partitions`
        built it: six partitions, one of them the only one with GPUs (4 per node).

        This is the MIXED blocker set — a `--gres=gpu:8 --cpus-per-task=1` job is
        `no GPU` in five of these and `too few GPUs` in the sixth — which is why the
        figure has to be a maximum over the whole list rather than one partition's.
        """
        common: dict[str, Any] = {
            "idle_nodes": 0,
            "cpus_idle": 0,
            "max_node_cpus": 48,
            "max_config_node_cpus": 48,
            "max_node_mem_bytes": 180 * 1024**3,
            "max_config_node_mem_bytes": 180 * 1024**3,
        }
        rows = [
            PartitionResources("caslake", True, is_current=True, **common),
            PartitionResources("amd", True, **common),
            PartitionResources("gpu", True, has_gpus=True, max_node_gpus=4, **common),
            PartitionResources("bigmem", True, **common),
            PartitionResources("build", True, **common),
            PartitionResources("amd-hm", True, **common),
        ]
        return rows

    def _tip(self, job: PendingJob, parts: list[PartitionResources]) -> str:
        from slurmwatch.tui import PendingView

        v = PendingView()
        v.job = job
        v.config = SlurmwatchConfig()
        v.partitions = parts
        out = Text.from_markup(v.render()).plain
        line = next((ln for ln in out.splitlines() if "can ever hold this request" in ln), "")
        assert line, f"the tip under test did not fire:\n{out}"
        return line.strip()

    # ---- FIX 1: the per-node GPU width, including the mixed blocker set ----

    def test_the_gpu_width_is_named_across_a_mixed_no_gpu_and_too_few_gpus_set(self) -> None:
        """FAILS BEFORE (the tip named nothing at all), PASSES AFTER.

        Reproduced live without submitting anything: `resolve_cluster_partitions`
        against the real cluster for account `beagle3-users` returns these six rows,
        and a `--gres=gpu:8 --cpus-per-task=1` job gets `{caslake: no GPU, amd: no
        GPU, gpu: too few GPUs, bigmem: no GPU, build: no GPU, amd-hm: no GPU}` —
        every one permanent. `sinfo -a -h -N -O Gres` says the widest GPU node
        anywhere on the cluster is `gpu:4`, so 4 is the true figure.
        """
        job = self._job(req_gpus=8)
        parts = self._account_view()
        blockers = {p.name: fit_blocker(job, p) for p in parts}
        assert sorted(set(blockers.values())) == ["no GPU", "too few GPUs"], blockers
        assert all(blocker_is_permanent(b) for b in blockers.values()), blockers
        assert pending.permanent_blocker_note(job, parts) == "largest node: 4 GPU"
        tip = self._tip(job, parts)
        assert "(largest node: 4 GPU)" in tip, tip
        # ...and NOT the core count, which is what this whole line of work is about:
        # the job asked for one core.
        assert "CPU" not in tip, tip

    def test_a_gpu_less_partition_in_the_list_cannot_inflate_the_figure(self) -> None:
        """The maximum is over the list, so the five GPU-less rows neither raise the
        figure nor suppress it: they cannot supply 8 GPUs either, which is what makes
        one number a complete reason for a mixed set. A widest node of 2 must read 2.
        """
        parts = self._account_view()
        for p in parts:
            if p.name == "gpu":
                p.max_node_gpus = 2
        assert pending.permanent_blocker_note(self._job(req_gpus=8), parts) == "largest node: 2 GPU"

    def test_the_gpu_figure_is_per_node_not_the_total_request(self) -> None:
        # A 16-GPU request spread over 4 nodes is 4 per node, which the hardware can
        # hold — so `fit_blocker` does not refuse it and the note must not either.
        parts = self._account_view()
        assert fit_blocker(self._job(req_gpus=16, req_nodes=4), parts[2]) != "too few GPUs"
        assert pending.permanent_blocker_note(self._job(req_gpus=16, req_nodes=4), parts) == ""

    # ---- FIX 2: the partition wall-clock ceiling ----

    def test_the_longest_wall_clock_is_named_for_a_time_limit_blocker(self) -> None:
        """FAILS BEFORE (nothing named), PASSES AFTER.

        NO LIVE REPRO EXISTS: all 87 partitions on this cluster are
        `MaxTime=UNLIMITED` (`scontrol show partition | grep -o MaxTime=...` — 87 of
        87), so `timelimit_seconds` is None on every real row and the `time limit`
        blocker cannot be produced from measured data. The ceilings below are set by
        hand on otherwise-real partition rows, and the figure the tip must quote is
        the most generous of them, not the current partition's.
        """
        parts = self._account_view()
        for p, limit in zip(parts, [4 * 3600, 2 * 3600, 3600, 3600, 600, 3600], strict=True):
            p.timelimit_seconds = limit
        job = self._job(time_limit_seconds=8 * 3600)
        assert {fit_blocker(job, p) for p in parts} == {"time limit"}
        assert pending.permanent_blocker_note(job, parts) == "max wall-clock: 4:00:00"
        assert "(max wall-clock: 4:00:00)" in self._tip(job, parts)

    def test_one_unlimited_partition_suppresses_the_wall_clock_figure(self) -> None:
        """CONTROL for fix 2 — passes before AND after.

        `timelimit_seconds is None` is Slurm's UNLIMITED (and also an unreadable
        `sinfo %l`). A single such partition makes "the longest limit here is 4:00:00"
        false however short the others are, so the figure must not be quoted — this is
        the live shape of this cluster, where a `time limit` note is unavailable.
        """
        parts = self._account_view()
        for p in parts:
            p.timelimit_seconds = 600
        parts[3].timelimit_seconds = None
        job = self._job(time_limit_seconds=8 * 3600)
        assert pending.permanent_blocker_note(job, parts) == ""
        # And the real cluster's own shape: no partition reports a ceiling at all.
        for p in parts:
            p.timelimit_seconds = None
        assert pending.permanent_blocker_note(job, parts) == ""

    def test_the_wall_clock_figure_is_written_the_way_slurm_writes_it(self) -> None:
        # It is compared by eye against what the user typed in `--time` and against
        # `sinfo %l`, so it is D-HH:MM:SS / H:MM:SS, not "4h" or "14400".
        assert pending._slurm_duration(4 * 3600) == "4:00:00"
        assert pending._slurm_duration(600) == "0:10:00"
        assert pending._slurm_duration(2 * 86400) == "2-00:00:00"
        assert pending._slurm_duration(86400 + 3661) == "1-01:01:01"
        assert pending._slurm_duration(-5) == "0:00:00"

    # ---- CONTROLS: what must not change ----

    def test_the_cpu_and_ram_figures_are_untouched(self) -> None:
        """CONTROL — passes before AND after. The two shapes that already named their
        binding figure must keep it, and must keep winning over the new branches when
        both bind (the documented order is CPU, RAM, GPU, wall-clock)."""
        parts = self._account_view()
        for p in parts:
            p.timelimit_seconds = 600
        note = pending.permanent_blocker_note
        assert note(self._job(req_cpus=999), parts) == "largest node: 48 CPU"
        assert note(self._job(req_mem_bytes=4 * 1024**4), parts) == "largest node: 180.0 GiB RAM"
        # Both a GPU shortfall and a wall-clock overrun on top of the CPU one: still
        # the CPU figure, because it is exceeded everywhere too and one is enough.
        both = self._job(req_cpus=999, req_gpus=8, time_limit_seconds=8 * 3600)
        assert note(both, parts) == "largest node: 48 CPU"

    def test_a_gpu_less_cluster_stays_silent(self) -> None:
        """CONTROL — passes before AND after. `no GPU` names nothing: the figure would
        be 0, which is indistinguishable from "sinfo reported no %G", and GPU nodes in
        a flagged state (`down*`, `drained*`) are not counted, so 0 can mean "its GPUs
        are unreachable this minute" rather than "there are none"."""
        parts = self._account_view()
        parts[2].has_gpus = False
        parts[2].max_node_gpus = 0
        job = self._job(req_gpus=1)
        assert {fit_blocker(job, p) for p in parts} == {"no GPU"}
        assert pending.permanent_blocker_note(job, parts) == ""
        tip = self._tip(job, parts)
        assert "largest node" not in tip, tip
        assert "GPU" not in tip, tip

    def test_a_gpu_model_nobody_has_stays_silent(self) -> None:
        """CONTROL — passes before AND after. `no <type>` names nothing: the useful
        note is which models DO exist, and `gpu_types` is known-incomplete (19 of the
        20 GPU partitions here report untyped `gpu:N`), so any such list would read as
        exhaustive while hiding most of the cluster's GPUs."""
        parts = self._account_view()
        parts[2].gpu_types = ["a30"]
        job = self._job(req_gpus=1, req_gpu_type="h100")
        assert fit_blocker(job, parts[2]) == "no h100"
        assert all(blocker_is_permanent(fit_blocker(job, p)) for p in parts)
        assert pending.permanent_blocker_note(job, parts) == ""
        tip = self._tip(job, parts)
        assert "largest node" not in tip, tip
        assert "a30" not in tip, tip
