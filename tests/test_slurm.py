from __future__ import annotations

import pytest

from slurmwatch import slurm
from slurmwatch.exceptions import CgroupNotFoundError, LoginNodeError
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
    def test_job_name_with_spaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pipe-delimited squeue output keeps fields intact even when the job
        # name contains spaces.
        squeue = "9001|R|gpu|my training job|2|1:23|4:00:00|cn[001-002]\n"
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: squeue)
        jobs = resolve_current_jobs("u")
        assert jobs == [
            {
                "job_id": "9001",
                "state": "R",
                "partition": "gpu",
                "name": "my training job",
                "nodes": "2",
                "wall_time": "1:23",
                "time_limit": "4:00:00",
                "reason": "cn[001-002]",
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
    "JobId=12345 JobState=RUNNING Partition=gpu\n"
    "NodeList=cn-[001-004] NumCPUs=16 NumNodes=1\n"
    "TRES=cpu=16,mem=64G,gres/gpu=4\n"
    "AllocTRES=cpu=16,mem=64G,gres/gpu=4\n"
    "MinMemoryNode=64G StartTime=2024-01-15T10:30:00 UserId=user(1001)\n"
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

    def test_wrong_host_raises_login_node_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: _SAMPLE_SCONTROL)
        monkeypatch.setattr(slurm, "_resolve_uid", lambda u: 1001)
        monkeypatch.setattr("socket.gethostname", lambda: "login-01")

        def _raise(*a: object, **k: object) -> dict[str, object]:
            raise CgroupNotFoundError("no cgroup here")

        monkeypatch.setattr(slurm, "_discover_cgroup_paths", _raise)
        with pytest.raises(LoginNodeError):
            resolve_job_context("12345")

    def test_right_host_propagates_cgroup_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # On a node that IS in the job's nodelist, a cgroup failure must not
        # be rewritten into a misleading 'login node' error.
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: _SAMPLE_SCONTROL)
        monkeypatch.setattr(slurm, "_resolve_uid", lambda u: 1001)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-002")

        def _raise(*a: object, **k: object) -> dict[str, object]:
            raise CgroupNotFoundError("no cgroup here")

        monkeypatch.setattr(slurm, "_discover_cgroup_paths", _raise)
        with pytest.raises(CgroupNotFoundError):
            resolve_job_context("12345")

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
