from __future__ import annotations

import os
from pathlib import Path

import pytest

from slurmwatch import slurm
from slurmwatch.exceptions import (
    CgroupNotFoundError,
    CgroupPermissionError,
    JobNotRunningError,
    SlurmCommandError,
)
from slurmwatch.model import short_host
from slurmwatch.slurm import (
    _parse_gpu_count,
    _parse_mem_to_bytes,
    _parse_nodelist,
    _parse_scontrol_field,
    _read_pid_environ,
    _split_cuda_visible,
    detect_cgroup_version,
    resolve_current_jobs,
    resolve_job_context,
)


class TestParseMemToBytes:
    def test_plain_number(self) -> None:
        assert _parse_mem_to_bytes("1024") == 1024

    def test_kilobytes(self) -> None:
        assert _parse_mem_to_bytes("8K") == 8192

    def test_megabytes(self) -> None:
        assert _parse_mem_to_bytes("16M") == 16 * 1024 * 1024

    def test_gigabytes(self) -> None:
        assert _parse_mem_to_bytes("2G") == 2 * 1024 * 1024 * 1024

    def test_terabytes(self) -> None:
        assert _parse_mem_to_bytes("1T") == 1024 * 1024 * 1024 * 1024

    def test_case_insensitive(self) -> None:
        assert _parse_mem_to_bytes("4g") == 4 * 1024 * 1024 * 1024

    def test_invalid_returns_zero(self) -> None:
        assert _parse_mem_to_bytes("") == 0
        assert _parse_mem_to_bytes("abc") == 0

    def test_float_value(self) -> None:
        assert _parse_mem_to_bytes("1.5G") == int(1.5 * 1024**3)


class TestParseNodelist:
    def test_single_node(self) -> None:
        assert _parse_nodelist("cn001") == ["cn001"]

    def test_comma_separated(self) -> None:
        assert _parse_nodelist("cn001,cn002") == ["cn001", "cn002"]

    def test_bracket_range(self) -> None:
        result = _parse_nodelist("cn-[001-003]")
        assert result == ["cn-001", "cn-002", "cn-003"]

    def test_bracket_with_multiple_ranges(self) -> None:
        result = _parse_nodelist("cn-[001-002,005-006]")
        assert result == ["cn-001", "cn-002", "cn-005", "cn-006"]

    def test_bracket_single_value(self) -> None:
        result = _parse_nodelist("cn-[008]")
        assert result == ["cn-008"]

    def test_mixed_nodes(self) -> None:
        result = _parse_nodelist("cn001,cn-[002-003]")
        assert result == ["cn001", "cn-002", "cn-003"]

    def test_null_nodelist(self) -> None:
        assert _parse_nodelist("(null)") == []
        assert _parse_nodelist("") == []


class TestParseGpuCount:
    def test_no_gres(self) -> None:
        assert _parse_gpu_count("") == 0

    def test_single_gpu(self) -> None:
        assert _parse_gpu_count("gpu:1") == 1

    def test_multiple_gpus(self) -> None:
        assert _parse_gpu_count("gpu:4") == 4

    def test_gpu_with_type(self) -> None:
        assert _parse_gpu_count("gpu:a100:2") == 2

    def test_tres_per_node_form(self) -> None:
        # Modern scontrol emits TresPerNode=gres/gpu:2 (or gres/gpu:a100:2).
        assert _parse_gpu_count("gres/gpu:2,gres/ssd:1") == 2
        assert _parse_gpu_count("gres/gpu:a100:2") == 2

    def test_multiple_gpu_entries(self) -> None:
        assert _parse_gpu_count("gpu:a100:2,gpu:h100:2") == 4


class TestParseTresGpus:
    def test_generic_entry(self) -> None:
        assert slurm._parse_tres_gpus("cpu=16,mem=64G,gres/gpu=4") == 4

    def test_gpumem_and_gpuutil_are_not_gpus(self) -> None:
        # gres/gpumem and gres/gpuutil share the gres/gpu prefix but are
        # different TRES; they must be ignored, not crash int() or overwrite
        # the count.
        tres = "billing=1,cpu=1,gres/gpu=1,gres/gpumem=4G,gres/gpuutil=100,mem=4G,node=1"
        assert slurm._parse_tres_gpus(tres) == 1

    def test_generic_preferred_over_typed(self) -> None:
        assert slurm._parse_tres_gpus("gres/gpu=4,gres/gpu:a100=2,gres/gpu:v100=2") == 4

    def test_typed_summed_when_generic_absent(self) -> None:
        assert slurm._parse_tres_gpus("gres/gpu:a100=2,gres/gpu:v100=2") == 4

    def test_no_gpus(self) -> None:
        assert slurm._parse_tres_gpus("cpu=16,mem=64G,node=1") == 0


class TestParseGresIdx:
    def test_typed_with_range(self) -> None:
        record = "JobId=1\n   Nodes=cn001 CPU_IDs=0-15 Mem=64000 GRES=gpu:a100:2(IDX:0-1)\n"
        assert slurm._parse_gres_idx(record, "cn001") == [0, 1]

    def test_untyped_with_list(self) -> None:
        record = "JobId=1\n   Nodes=cn001 CPU_IDs=0-3 Mem=8000 GRES=gpu:4(IDX:0,2)\n"
        assert slurm._parse_gres_idx(record, "cn001") == [0, 2]

    def test_multi_node_picks_this_host(self) -> None:
        record = (
            "JobId=1\n"
            "   Nodes=cn001 CPU_IDs=0-15 Mem=64000 GRES=gpu:2(IDX:0-1)\n"
            "   Nodes=cn002 CPU_IDs=0-15 Mem=64000 GRES=gpu:2(IDX:2-3)\n"
        )
        assert slurm._parse_gres_idx(record, "cn002") == [2, 3]

    def test_multi_node_picks_this_host_when_scontrol_uses_fqdn(self) -> None:
        # #29: FQDN NodeName vs short gethostname must still pick this node's
        # GPUs, else NVML falls back to attaching the first N PCI devices.
        record = (
            "JobId=1\n"
            "   Nodes=cn001.cluster.edu CPU_IDs=0-15 Mem=64000 GRES=gpu:2(IDX:0-1)\n"
            "   Nodes=cn002.cluster.edu CPU_IDs=0-15 Mem=64000 GRES=gpu:2(IDX:2-3)\n"
        )
        assert slurm._parse_gres_idx(record, "cn002") == [2, 3]

    def test_single_line_used_when_host_not_named(self) -> None:
        record = "JobId=1\n   Nodes=cn[001-002] CPU_IDs=0-15 Mem=64000 GRES=gpu:2(IDX:0-1)\n"
        assert slurm._parse_gres_idx(record, "elsewhere") == [0, 1]

    def test_no_detail_lines(self) -> None:
        assert slurm._parse_gres_idx("JobId=1 JobState=RUNNING", "cn001") == []

    def test_expand_idx_list(self) -> None:
        assert slurm._expand_idx_list("0-1,3") == [0, 1, 3]
        assert slurm._expand_idx_list("2") == [2]


class TestParseScontrolField:
    def test_simple_field(self) -> None:
        output = "JobId=12345 JobName=test\n  JobState=RUNNING\n"
        assert _parse_scontrol_field(output, "JobState") == "RUNNING"

    def test_field_with_unicode(self) -> None:
        output = "WorkDir=/home/user/test\n"
        assert _parse_scontrol_field(output, "WorkDir") == "/home/user/test"

    def test_missing_field(self) -> None:
        output = "JobId=12345\n"
        assert _parse_scontrol_field(output, "Partition") is None

    def test_multiple_matches(self) -> None:
        output = "TRES=cpu=4,mem=8G,gres/gpu=2\nAllocTRES=cpu=4,mem=8G,gres/gpu=2\n"
        assert _parse_scontrol_field(output, "AllocTRES") == "cpu=4,mem=8G,gres/gpu=2"


class TestResolveCurrentJobs:
    @pytest.mark.usefixtures("mock_slurm_env")
    def test_mock_mode(self) -> None:
        jobs = resolve_current_jobs("testuser")
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "12345"
        assert jobs[0]["state"] == "R"


class TestDetectCgroupVersion:
    def test_returns_int(self) -> None:
        version = detect_cgroup_version()
        assert version in (1, 2)


class TestParseSlurmDuration:
    def test_hms(self) -> None:
        assert slurm._parse_slurm_duration("03:29:03") == 3 * 3600 + 29 * 60 + 3

    def test_days(self) -> None:
        assert slurm._parse_slurm_duration("2-01:00:00") == 2 * 86400 + 3600

    def test_mmss_fractional(self) -> None:
        assert slurm._parse_slurm_duration("01:30.500") == 90.5

    def test_empty(self) -> None:
        assert slurm._parse_slurm_duration("") == 0.0


class TestIsJobActive:
    """#28: the mid-flight liveness recheck that lets the dashboard notice a job
    ending instead of freezing at its last numbers forever."""

    def test_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "RUNNING\n")
        assert slurm.is_job_active("123") is True

    def test_completing_is_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "COMPLETING\n")
        assert slurm.is_job_active("123") is True

    def test_multi_task_any_running_is_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # squeue widens an array id to every task; any still-running task counts.
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "PENDING\nRUNNING\n")
        assert slurm.is_job_active("123") is True

    def test_empty_output_means_ended(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # squeue lists only active jobs; a completed job is simply absent.
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "\n")
        assert slurm.is_job_active("123") is False

    def test_invalid_job_id_but_squeue_healthy_means_ended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # squeue rejects an id it no longer knows; a healthy ping confirms squeue
        # works, so the job has genuinely ended (not a transient failure).
        def _cmd(cmd: list[str], *a: object, **k: object) -> str:
            if "-j" in cmd:
                raise SlurmCommandError("Invalid job id specified")
            return "999\n"

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _cmd)
        assert slurm.is_job_active("123") is False

    def test_squeue_unreachable_is_unknown_not_ended(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Both the query AND the ping fail: Slurm is unreachable. Must return None
        # so a transient outage never tears down a live dashboard.
        def _cmd(*a: object, **k: object) -> str:
            raise SlurmCommandError("slurmctld down")

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _cmd)
        assert slurm.is_job_active("123") is None

    def test_mock_is_always_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        assert slurm.is_job_active("anything") is True


class TestResolveRemoteUsage:
    def test_aggregates_batch_step_skips_sentinel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # extern step reports a NO_VAL sentinel and no RSS; the batch step has
        # the real sample. Only the batch step must count.
        sstat_out = (
            "51397890.extern||213503982334-14:25:51|0\n51397890.batch|183133764K|03:29:03|1\n"
        )
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: sstat_out)
        usage = slurm.resolve_remote_usage("51397890")
        assert usage.sampled is True
        assert usage.rss_bytes == 183133764 * 1024
        assert usage.cpu_seconds == 3 * 3600 + 29 * 60 + 3

    def test_multi_task_step_multiplies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sstat_out = "99.batch|1000K|00:01:00|4\n"
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: sstat_out)
        usage = slurm.resolve_remote_usage("99")
        assert usage.cpu_seconds == 60 * 4

    def test_not_yet_sampled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "99.extern|||0\n")
        usage = slurm.resolve_remote_usage("99")
        assert usage.sampled is False
        assert usage.rss_bytes == 0

    def test_sstat_failure_is_graceful(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from slurmwatch.exceptions import SlurmCommandError

        def _fail(*a: object, **k: object) -> str:
            raise SlurmCommandError("sstat: no steps")

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _fail)
        usage = slurm.resolve_remote_usage("99")
        assert usage.sampled is False
        assert usage.rss_bytes == 0


class TestReadPidEnviron:
    def test_empty_on_missing_pid(self) -> None:
        result = _read_pid_environ(99999999)
        assert result == {}


class TestScontrolOutputParsing:
    SAMPLE_OUTPUT = """JobId=12345
    JobName=train
    JobState=RUNNING
    Partition=gpu
    NodeList=cn-[001-004]
    NumCPUs=16
    TRES=cpu=16,mem=64G,gres/gpu=4
    AllocTRES=cpu=16,mem=64G,gres/gpu=4
    MinMemoryNode=64G
    StartTime=2024-01-15T10:30:00
    UserId=user(1001)
    WorkDir=/home/user
    """

    def test_parse_all_fields(self) -> None:
        assert _parse_scontrol_field(self.SAMPLE_OUTPUT, "JobState") == "RUNNING"
        assert _parse_scontrol_field(self.SAMPLE_OUTPUT, "Partition") == "gpu"
        assert _parse_scontrol_field(self.SAMPLE_OUTPUT, "NumCPUs") == "16"
        assert _parse_scontrol_field(self.SAMPLE_OUTPUT, "NodeList") == "cn-[001-004]"
        assert _parse_scontrol_field(self.SAMPLE_OUTPUT, "AllocTRES") == "cpu=16,mem=64G,gres/gpu=4"
        assert _parse_scontrol_field(self.SAMPLE_OUTPUT, "TRES") == "cpu=16,mem=64G,gres/gpu=4"


class TestResolveCurrentJobsParsing:
    def test_job_name_with_spaces_and_pipe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The job name (%j) is emitted last, so even a name containing spaces
        # *and* a literal '|' is absorbed by the final field instead of shifting
        # every column after it (B-P10). Field order: id|state|part|D|M|l|R|name.
        squeue = "9001|R|gpu|2|1:23|4:00:00|cn[001-002]|my training|job\n"
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: squeue)
        jobs = resolve_current_jobs("u")
        assert jobs == [
            {
                "job_id": "9001",
                "state": "R",
                "partition": "gpu",
                "nodes": "2",
                "wall_time": "1:23",
                "time_limit": "4:00:00",
                "reason": "cn[001-002]",
                "name": "my training|job",
            }
        ]

    def test_skips_non_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        squeue = "1|PD|gpu|queued|1|0:00|1:00|(Priority)\n2|R|gpu|run|1|0:10|1:00|cn001\n"
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: squeue)
        jobs = resolve_current_jobs("u")
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "2"


class TestSelectJobRecord:
    RECORDS = (
        "JobId=100 ArrayJobId=99 ArrayTaskId=1 JobState=RUNNING NodeList=cn001\n"
        "\n"
        "JobId=101 ArrayJobId=99 ArrayTaskId=2 JobState=RUNNING NodeList=cn002\n"
        "\n"
        "JobId=102 ArrayJobId=99 ArrayTaskId=3 JobState=PENDING NodeList=(null)\n"
    )

    def test_prefers_record_on_this_host(self) -> None:
        record = slurm._select_job_record(self.RECORDS, "cn002")
        assert _parse_scontrol_field(record, "JobId") == "101"

    def test_falls_back_to_first_running(self) -> None:
        record = slurm._select_job_record(self.RECORDS, "elsewhere")
        assert _parse_scontrol_field(record, "JobId") == "100"

    def test_single_record_passthrough(self) -> None:
        record = slurm._select_job_record("JobId=7 JobState=RUNNING\n", "cn001")
        assert _parse_scontrol_field(record, "JobId") == "7"

    FQDN_RECORDS = (
        "JobId=100 ArrayJobId=99 ArrayTaskId=1 JobState=RUNNING NodeList=gpu05.cluster.edu\n"
        "\n"
        "JobId=101 ArrayJobId=99 ArrayTaskId=2 JobState=RUNNING NodeList=gpu06.cluster.edu\n"
    )

    def test_matches_when_scontrol_uses_fqdn_node_names(self) -> None:
        # #29: slurmctld emits FQDN NodeName, gethostname gives the short name.
        record = slurm._select_job_record(self.FQDN_RECORDS, "gpu06")
        assert _parse_scontrol_field(record, "ArrayTaskId") == "2"

    def test_matches_case_insensitively(self) -> None:
        record = slurm._select_job_record(self.RECORDS, "CN002")
        assert _parse_scontrol_field(record, "JobId") == "101"


class TestSplitCudaVisible:
    def test_integers(self) -> None:
        assert _split_cuda_visible("0,1,2") == ([0, 1, 2], [])

    def test_uuids(self) -> None:
        idxs, uuids = _split_cuda_visible("GPU-abc123,MIG-99887766")
        assert idxs == []
        assert uuids == ["GPU-abc123", "MIG-99887766"]

    def test_mixed_and_blank(self) -> None:
        idxs, uuids = _split_cuda_visible("0, ,GPU-abc")
        assert idxs == [0]
        assert uuids == ["GPU-abc"]

    def test_empty(self) -> None:
        assert _split_cuda_visible("") == ([], [])


_SAMPLE_SCONTROL = (
    "JobId=12345 JobState=RUNNING Partition=gpu Account=rcc-staff QOS=normal\n"
    "NodeList=cn-[001-004] NumCPUs=16 NumNodes=1\n"
    "TRES=cpu=16,mem=64G,gres/gpu=4\n"
    "AllocTRES=cpu=16,mem=64G,gres/gpu=4\n"
    "RunTime=01:00:00 TimeLimit=1-00:00:00\n"
    "SubmitTime=2024-01-15T10:29:00 MinMemoryNode=64G StartTime=2024-01-15T10:30:00 "
    "UserId=user(1001)\n"
    "   Command=/home/user/proj/train.py WorkDir=/home/user/proj/runs\n"
    "   Nodes=cn-[001-004] CPU_IDs=0-15 Mem=65536 GRES=gpu:a100:4(IDX:0-3)\n"
)


class TestResolveJobContext:
    @staticmethod
    def _patch_common(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: output)
        monkeypatch.setattr(slurm, "_resolve_uid", lambda u: 1001)
        monkeypatch.setattr(
            slurm,
            "_discover_cgroup_paths",
            lambda *a, **k: {"v2": None, "v1_mem": None, "v1_cpu": None},
        )
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def test_parses_tres_and_nodelist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_common(monkeypatch, _SAMPLE_SCONTROL)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        ctx = resolve_job_context("12345")
        assert ctx.job_id == "12345"
        assert ctx.partition == "gpu"
        assert ctx.cpus_allocated == 16
        assert ctx.mem_limit_bytes == 64 * 1024**3
        assert ctx.gpu_count_requested == 4
        assert ctx.nodelist_resolved == ["cn-001", "cn-002", "cn-003", "cn-004"]
        assert ctx.gpu_uuids == []
        # From the scontrol -d IDX detail, not CUDA_VISIBLE_DEVICES.
        assert ctx.gpu_indices == [0, 1, 2, 3]
        # TimeLimit=1-00:00:00 -> 24h; used to show how long the job can still run.
        assert ctx.time_limit_seconds == 24 * 3600

    def test_parses_job_provenance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The JOB card's provenance comes from the same scontrol record.
        self._patch_common(monkeypatch, _SAMPLE_SCONTROL)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        ctx = resolve_job_context("12345")
        assert ctx.account == "rcc-staff"
        assert ctx.qos == "normal"
        assert ctx.command == "/home/user/proj/train.py"
        assert ctx.work_dir == "/home/user/proj/runs"
        assert ctx.job_state == "RUNNING"
        assert ctx.submit_time is not None and ctx.job_start_time is not None
        # Submitted before it started (queue wait is non-negative).
        assert ctx.submit_time <= ctx.job_start_time

    def test_null_command_is_normalized_to_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # scontrol prints "Command=(null)" for an interactive salloc job; that must
        # become "" so the JOB card omits the line rather than showing "(null)".
        output = _SAMPLE_SCONTROL.replace("Command=/home/user/proj/train.py", "Command=(null)")
        self._patch_common(monkeypatch, output)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        ctx = resolve_job_context("12345")
        assert ctx.command == ""

    def test_unlimited_time_limit_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = _SAMPLE_SCONTROL.replace("TimeLimit=1-00:00:00", "TimeLimit=UNLIMITED")
        self._patch_common(monkeypatch, output)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        ctx = resolve_job_context("12345")
        assert ctx.time_limit_seconds is None

    def test_off_node_falls_back_to_remote(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # From a login node the job's cgroups aren't present; instead of
        # erroring, the context is marked remote (usage comes from sstat).
        self._patch_common(monkeypatch, _SAMPLE_SCONTROL)
        monkeypatch.setattr("socket.gethostname", lambda: "login-01")

        def _raise(*a: object, **k: object) -> dict[str, object]:
            raise CgroupNotFoundError("no cgroup here")

        monkeypatch.setattr(slurm, "_discover_cgroup_paths", _raise)
        ctx = resolve_job_context("12345")
        assert ctx.remote is True
        assert ctx.cgroup_v2_path is None
        # Allocation metadata is still resolved from scontrol.
        assert ctx.mem_limit_bytes == 64 * 1024**3
        assert ctx.gpu_count_requested == 4

    def test_array_task_uses_raw_job_id_for_cgroups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = (
            "JobId=12348 ArrayJobId=12345 ArrayTaskId=3 JobState=RUNNING Partition=gpu\n"
            "NodeList=cn-001 NumCPUs=4 NumNodes=1\n"
            "TRES=cpu=4,mem=8G,node=1\n"
            "UserId=user(1001) StartTime=2024-01-15T10:30:00\n"
        )
        seen: dict[str, object] = {}

        def _capture(job_id: str, *a: object, **k: object) -> dict[str, object]:
            seen["job_id"] = job_id
            return {"v2": None, "v1_mem": None, "v1_cpu": None}

        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: output)
        monkeypatch.setattr(slurm, "_resolve_uid", lambda u: 1001)
        monkeypatch.setattr(slurm, "_discover_cgroup_paths", _capture)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        ctx = resolve_job_context("12345_3")
        # Cgroups are named after the task's raw JobId, not the array form.
        assert seen["job_id"] == "12348"
        # The user-facing id is preserved for display.
        assert ctx.job_id == "12345_3"

    def test_multi_node_scales_to_node_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = (
            "JobId=777 JobState=RUNNING Partition=gpu\n"
            "NodeList=cn-[001-004] NumCPUs=512 NumNodes=4\n"
            "TRES=cpu=512,mem=875G,node=4,billing=512,gres/gpu=16\n"
            "TresPerNode=gres/gpu:4\n"
            "MinMemoryNode=219G UserId=user(1001) StartTime=2024-01-15T10:30:00\n"
        )
        self._patch_common(monkeypatch, output)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-002")
        ctx = resolve_job_context("777")
        # slurmwatch monitors one node: limits must be node-local.
        assert ctx.cpus_allocated == 128
        assert ctx.mem_limit_bytes == 219 * 1024**3
        assert ctx.gpu_count_requested == 4

    def test_multi_node_fqdn_uses_exact_per_node_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #29 end-to-end: on a cluster whose scontrol emits FQDN NodeName, a
        # heterogeneous multi-node job must still read THIS node's exact CPU_IDs/
        # Mem/GPU from the -d detail line — not fall back to job-wide // NumNodes.
        # cn-002 has 16 CPUs / 96000 MB / 2 GPUs; the job-wide average would be
        # 24//2=12 CPUs and 3//2=1 GPU, so a wrong answer is unambiguous.
        output = (
            "JobId=42 JobState=RUNNING Partition=gpu\n"
            "NodeList=cn-[001-002].cluster.edu NumCPUs=24 NumNodes=2\n"
            "TRES=cpu=24,mem=160G,gres/gpu=3 UserId=user(1001) "
            "StartTime=2024-01-15T10:30:00\n"
            "   Nodes=cn-001.cluster.edu CPU_IDs=0-7 Mem=64000 GRES=gpu:1(IDX:0)\n"
            "   Nodes=cn-002.cluster.edu CPU_IDs=0-15 Mem=96000 GRES=gpu:2(IDX:1-2)\n"
        )
        self._patch_common(monkeypatch, output)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-002")  # short name
        ctx = resolve_job_context("42")
        assert ctx.cpus_allocated == 16
        assert ctx.mem_limit_bytes == 96000 * 1024**2
        assert ctx.gpu_indices == [1, 2]

    def test_gpumem_tres_does_not_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = (
            "JobId=888 JobState=RUNNING Partition=gpu\n"
            "NodeList=cn-001 NumCPUs=1 NumNodes=1\n"
            "TRES=billing=1,cpu=1,gres/gpu=1,gres/gpumem=4G,gres/gpuutil=100,mem=4G,node=1\n"
            "UserId=user(1001) StartTime=2024-01-15T10:30:00\n"
        )
        self._patch_common(monkeypatch, output)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        ctx = resolve_job_context("888")
        assert ctx.gpu_count_requested == 1

    def test_pending_job_raises_not_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # B-T7: the JobNotRunningError gate must fire for a non-running record.
        output = (
            "JobId=5 JobState=PENDING Partition=gpu\n"
            "NodeList=(null) NumCPUs=1 NumNodes=1 UserId=user(1001)\n"
        )
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: output)
        monkeypatch.setattr(slurm, "_resolve_uid", lambda u: 1001)
        monkeypatch.setattr("socket.gethostname", lambda: "cn001")
        with pytest.raises(JobNotRunningError):
            resolve_job_context("5")

    def test_multinode_uses_exact_per_node_detail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # B-P4: the per-node CPU_IDs/Mem on the -d detail line are exact; they
        # must win over NumCPUs//NumNodes and MinMemoryNode. cn002 has 20 cores
        # and 96000 MB here, not 32//2=16 cores / 64G.
        output = (
            "JobId=42 JobState=RUNNING Partition=gpu\n"
            "NodeList=cn[001-002] NumCPUs=32 NumNodes=2\n"
            "TRES=cpu=32,mem=128G,node=2,gres/gpu=4\n"
            "MinMemoryNode=64G UserId=user(1001) StartTime=2024-01-15T10:30:00\n"
            "   Nodes=cn001 CPU_IDs=0-11 Mem=48000 GRES=gpu:2(IDX:0-1)\n"
            "   Nodes=cn002 CPU_IDs=0-19 Mem=96000 GRES=gpu:2(IDX:2-3)\n"
        )
        self._patch_common(monkeypatch, output)
        monkeypatch.setattr("socket.gethostname", lambda: "cn002")
        ctx = resolve_job_context("42")
        assert ctx.cpus_allocated == 20  # CPU_IDs=0-19, not 32 // 2
        assert ctx.mem_limit_bytes == 96000 * 1024**2  # Mem=96000 MB, not 64 GiB


class TestParseNodelistMultiDim:
    def test_multi_dimensional_brackets(self) -> None:
        # B-P12: more than one bracket group per element expands cartesian.
        assert _parse_nodelist("rack[1-2]node[3-4]") == [
            "rack1node3",
            "rack1node4",
            "rack2node3",
            "rack2node4",
        ]

    def test_trailing_literal_after_bracket(self) -> None:
        assert _parse_nodelist("gpu[01-02]x") == ["gpu01x", "gpu02x"]


class TestCgroupNameMatch:
    def test_boundary_match(self) -> None:
        # B-P11: job_123 must not match job_1234/job_12345.
        assert slurm._cgroup_name_matches_job("job_123", "123") is True
        assert slurm._cgroup_name_matches_job("job_1234", "123") is False
        assert slurm._cgroup_name_matches_job("job_12345", "123") is False
        assert slurm._cgroup_name_matches_job("job_123.scope", "123") is True
        assert slurm._cgroup_name_matches_job("job_123_0", "123") is True
        assert slurm._cgroup_name_matches_job("unrelated", "123") is False


class TestResolveGpuIndicesUnion:
    def test_cuda_visible_unioned_across_pids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # B-P13: per-task GPU binding gives each rank a different
        # CUDA_VISIBLE_DEVICES; the union is the node's allocation, not rank 0's.
        envs = {
            10: {"CUDA_VISIBLE_DEVICES": "0"},
            11: {"CUDA_VISIBLE_DEVICES": "1"},
            12: {"CUDA_VISIBLE_DEVICES": "2,3"},
        }
        monkeypatch.setattr(slurm, "_read_pid_environ", lambda pid: envs.get(pid, {}))
        idx, uuids = slurm._resolve_gpu_indices("JobId=1 JobState=RUNNING", "cn001", [10, 11, 12])
        assert idx == [0, 1, 2, 3]
        assert uuids == []


class TestReadPidEnvironHappyPath:
    def test_reads_own_environ(self) -> None:
        # B-T9: the NUL-split happy path (previously only the missing-PID case
        # was tested). The test process's own environ is always readable.
        env = _read_pid_environ(os.getpid())
        assert isinstance(env, dict)
        assert "PATH" in env


class TestCgroupDiscovery:
    """B-T1: exercise _discover_cgroup_paths against a real filesystem tree."""

    def test_v2_discovery_prefers_step(
        self, fake_cgroup_v2: Path, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", fake_cgroup_v2)
        paths = slurm._discover_cgroup_paths("12345", uid=1001, step_id="0")
        assert paths["v2"] is not None
        assert paths["v2"].name == "step_0"
        assert "job_12345" in str(paths["v2"])

    def test_v2_discovery_without_step(
        self, fake_cgroup_v2: Path, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", fake_cgroup_v2)
        paths = slurm._discover_cgroup_paths("12345", uid=1001, step_id=None)
        assert paths["v2"] is not None
        assert paths["v2"].name == "job_12345"

    def test_missing_job_raises_not_found(
        self, fake_cgroup_v2: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", fake_cgroup_v2)
        with pytest.raises(CgroupNotFoundError):
            slurm._discover_cgroup_paths("99999", uid=1001, step_id=None)

    def test_iterdir_fallback_respects_numeric_boundary(
        self, fake_cgroup_v2: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # B-P11 at the discovery level: with no exact scope path, the substring
        # fallback must pick job_123, never job_1234.
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", fake_cgroup_v2)
        ss = fake_cgroup_v2 / "system.slice"
        for name in ("job_1234", "job_123"):
            (ss / name).mkdir(parents=True)
            (ss / name / "cgroup.procs").write_text("")
        paths = slurm._discover_cgroup_paths("123", uid=1001, step_id=None)
        assert paths["v2"] is not None
        assert paths["v2"].name == "job_123"

    def test_fallback_descends_into_slurmstepd_scope(
        self, fake_cgroup_v2: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # F3: real Slurm job cgroups live under system.slice/slurmstepd.scope/.
        # When the exact candidate path misses (here a `.scope`-suffixed name),
        # the fallback must descend into slurmstepd.scope to find it — it used to
        # scan only system.slice and silently degrade to remote sstat mode.
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", fake_cgroup_v2)
        scope = fake_cgroup_v2 / "system.slice" / "slurmstepd.scope"
        job_dir = scope / "job_777.scope"  # suffixed -> exact candidate "job_777" misses
        job_dir.mkdir(parents=True)
        (job_dir / "cgroup.procs").write_text("42\n")
        paths = slurm._discover_cgroup_paths("777", uid=None, step_id=None)
        assert paths["v2"] == job_dir

    def test_permission_error_maps_to_cgroup_permission(
        self, fake_cgroup_v2: Path, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", fake_cgroup_v2)

        def _raise(_path: Path) -> None:
            raise PermissionError()

        monkeypatch.setattr(slurm, "_check_cgroup_readable", _raise)
        with pytest.raises(CgroupPermissionError):
            slurm._discover_cgroup_paths("12345", uid=1001, step_id="0")


class TestDetectCgroupVersionReal:
    """B-T4: the tautological ``version in (1, 2)`` check can't catch inversions."""

    def test_v2_when_controllers_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "cgroup.controllers").write_text("cpu memory")
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", tmp_path)
        assert detect_cgroup_version() == 2

    def test_v1_when_controllers_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", tmp_path)
        assert detect_cgroup_version() == 1


class TestMockJobContext:
    """`--demo` / SLURMWATCH_MOCK must hand the dashboard a node it can serve.

    The dashboard reads the node it runs on from the local collector and streams
    every *other* node over srun. A mock nodelist of purely fictional names left
    no local node to select, so `--demo` streamed an unreachable node and showed
    a permanently blank "awaiting telemetry…" dashboard (#27).
    """

    def test_local_host_is_node_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("socket.gethostname", lambda: "midway3-0509.rcc.local")
        ctx = slurm._make_mock_job_context("12345")
        assert ctx.nodelist_resolved[0] == "midway3-0509"
        assert ctx.hostname == "midway3-0509"

    def test_hostname_is_in_the_resolved_nodelist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The invariant the dashboard actually depends on: it can only select a
        # node that is in `nodelist_resolved` (`_set_node` rejects anything else).
        monkeypatch.setattr("socket.gethostname", lambda: "somehost")
        ctx = slurm._make_mock_job_context("12345")
        assert ctx.hostname in ctx.nodelist_resolved

    def test_still_multi_node_so_the_switcher_is_exercised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("socket.gethostname", lambda: "somehost")
        ctx = slurm._make_mock_job_context("12345")
        assert len(ctx.nodelist_resolved) == 4

    def test_nodelist_string_agrees_with_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The JOB card renders ctx.nodelist; it must not advertise nodes that the
        # switcher (driven by nodelist_resolved) doesn't have.
        monkeypatch.setattr("socket.gethostname", lambda: "somehost")
        ctx = slurm._make_mock_job_context("12345")
        assert ctx.nodelist == ",".join(ctx.nodelist_resolved)

    def test_mock_context_is_local_not_remote(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # remote=True would push the demo onto the sstat estimate path.
        monkeypatch.setattr("socket.gethostname", lambda: "somehost")
        assert slurm._make_mock_job_context("12345").remote is False

    @pytest.mark.parametrize("host", ["cn-002", "CN-002", "cn-004.rcc.local", "cn-003"])
    def test_no_duplicate_node_when_host_collides_with_a_filler_name(
        self, host: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A machine actually named cn-002 must not be listed twice in the
        # switcher (the comparison is short_host, so case and domain collide too).
        monkeypatch.setattr("socket.gethostname", lambda: host)
        nodes = slurm._make_mock_job_context("12345").nodelist_resolved
        shorts = [short_host(n) for n in nodes]
        assert len(shorts) == len(set(shorts)) == 4
