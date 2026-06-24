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

    def test_mixed_gres(self) -> None:
        assert _parse_gpu_count("gres/gpu:2,gres/ssd:1") == 0

    def test_multiple_gpu_entries(self) -> None:
        assert _parse_gpu_count("gpu:a100:2,gpu:h100:2") == 4


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
    GresDetail=gpu:0:A100-SXM4-80GB:4
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
    "NodeList=cn-[001-004] NumCPUs=16\n"
    "TRES=cpu=16,mem=64G,gres/gpu=4\n"
    "AllocTRES=cpu=16,mem=64G,gres/gpu=4\n"
    "MinMemoryNode=64G StartTime=2024-01-15T10:30:00 UserId=user(1001)\n"
)


class TestResolveJobContext:
    def test_parses_tres_and_nodelist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: _SAMPLE_SCONTROL)
        monkeypatch.setattr(slurm, "_resolve_uid", lambda u: 1001)
        monkeypatch.setattr(slurm, "_find_job_pids", lambda u: [])
        monkeypatch.setattr(
            slurm,
            "_discover_cgroup_paths",
            lambda *a, **k: {"v2": None, "v1_mem": None, "v1_cpu": None},
        )
        ctx = resolve_job_context("12345")
        assert ctx.job_id == "12345"
        assert ctx.partition == "gpu"
        assert ctx.cpus_allocated == 16
        assert ctx.mem_limit_bytes == 64 * 1024**3
        assert ctx.gpu_count_requested == 4
        assert ctx.nodelist_resolved == ["cn-001", "cn-002", "cn-003", "cn-004"]
        assert ctx.gpu_uuids == []

    def test_login_node_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: _SAMPLE_SCONTROL)
        monkeypatch.setattr(slurm, "_resolve_uid", lambda u: 1001)
        monkeypatch.setattr(slurm, "_find_job_pids", lambda u: [])

        def _raise(*a: object, **k: object) -> dict[str, object]:
            raise CgroupNotFoundError("no cgroup here")

        monkeypatch.setattr(slurm, "_discover_cgroup_paths", _raise)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/squeue")
        with pytest.raises(LoginNodeError):
            resolve_job_context("12345")
