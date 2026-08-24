from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from slurmwatch import slurm
from slurmwatch.exceptions import (
    CgroupNotFoundError,
    CgroupPermissionError,
    JobNotFoundError,
    JobNotRunningError,
    SlurmCommandError,
)
from slurmwatch.model import local_node_name, short_host
from slurmwatch.slurm import (
    _parse_gpu_count,
    _parse_mem_to_bytes,
    _parse_nodelist,
    _parse_scontrol_field,
    _read_pid_environ,
    _sacct_final_state,
    _split_cuda_visible,
    detect_cgroup_version,
    resolve_current_jobs,
    resolve_job_context,
)


def test_run_slurm_cmd_normalizes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # cross-cutting F2: a user's SLURM_TIME_FORMAT / locale must not reach the
    # subprocess, or scontrol timestamps stop matching the ISO parser (elapsed 0).
    captured: dict[str, object] = {}

    class _Result:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    def _fake_run(_cmd: list[str], **kwargs: object) -> _Result:
        captured.update(kwargs)
        return _Result()

    monkeypatch.setenv("SLURM_TIME_FORMAT", "relative")
    monkeypatch.setattr("slurmwatch.slurm.subprocess.run", _fake_run)
    assert slurm._run_slurm_cmd(["scontrol", "show", "job"]) == "ok\n"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["SLURM_TIME_FORMAT"] == "standard"
    assert env["LC_ALL"] == "C"
    assert "PATH" in env  # the parent environment is otherwise preserved


class TestOwnerIdentityWhenSlurmPrintsNoUid:
    """`_resolve_uid` had ZERO coverage: on this cluster scontrol always prints
    ``UserId=name(uid)``, so the getpwnam fallback is never taken here — it is taken
    only on sites whose Slurm prints a bare name. That makes it precisely the kind of
    path this exercise is about: identity logic that only runs somewhere else.

    Identity is not cosmetic here. It decides whether slurmwatch shows your own job's
    telemetry or refuses it as another user's, and which ``uid_N`` cgroup it looks in.
    """

    def test_a_bare_name_is_resolved_through_the_name_service(self) -> None:
        import os
        import pwd

        me = pwd.getpwuid(os.getuid()).pw_name
        assert slurm._parse_user_id(me) == (me, os.getuid())

    def test_a_realm_qualified_name_still_resolves(self) -> None:
        """Kerberos/AD sites print `user@REALM`; the realm is not part of the login."""
        import os
        import pwd

        me = pwd.getpwuid(os.getuid()).pw_name
        assert slurm._parse_user_id(f"{me}@EXAMPLE.ORG") == (me, os.getuid())

    def test_an_unknown_name_yields_no_uid_rather_than_a_wrong_one(self) -> None:
        assert slurm._parse_user_id("nosuchuser42") == ("nosuchuser42", None)

    def test_a_placeholder_name_is_never_resolved_to_its_own_uid(self) -> None:
        """SW-1's other half. `nobody` IS a real passwd entry (uid 99/65534), so
        getpwnam succeeds and returns a uid belonging to someone else — which is why
        the name has to be dropped BEFORE any resolution, not after."""
        assert slurm._parse_user_id("nobody") == ("", None)
        assert slurm._parse_user_id("nfsnobody") == ("", None)

    def test_a_placeholder_with_a_real_uid_recovers_the_name(self) -> None:
        """SW-1: a node with a thinner nsswitch prints `UserId=nobody(<real uid>)`.
        The uid in the parentheses is authoritative and the name is re-derived from
        it, so your own job does not read as `nobody`'s."""
        import os
        import pwd

        me = pwd.getpwuid(os.getuid()).pw_name
        assert slurm._parse_user_id(f"nobody({os.getuid()})") == (me, os.getuid())

    def test_whitespace_and_emptiness(self) -> None:
        import os
        import pwd

        me = pwd.getpwuid(os.getuid()).pw_name
        assert slurm._parse_user_id(f"  {me}  ") == (me, os.getuid())
        assert slurm._parse_user_id("") == ("", None)


class TestParseMemToBytes:
    def test_a_bare_integer_is_megabytes_like_sbatch_mem(self) -> None:
        """`sbatch --mem=16` means 16 MiB, and Slurm echoes ReqMem the same way.
        Reading it as 16 BYTES is off by 1,048,576x, and silently — the gauge then
        shows a job "over" a 16-byte limit. SW-12."""
        assert _parse_mem_to_bytes("1024") == 1024 * 1024**2
        assert _parse_mem_to_bytes("16") == 16 * 1024**2

    def test_kilobytes(self) -> None:
        assert _parse_mem_to_bytes("8K") == 8192

    def test_megabytes(self) -> None:
        assert _parse_mem_to_bytes("16M") == 16 * 1024 * 1024

    def test_gigabytes(self) -> None:
        assert _parse_mem_to_bytes("2G") == 2 * 1024 * 1024 * 1024

    def test_terabytes(self) -> None:
        assert _parse_mem_to_bytes("1T") == 1024 * 1024 * 1024 * 1024

    def test_petabytes(self) -> None:
        # Slurm's memory grammar includes P; dropping it silently zeroed a limit.
        assert _parse_mem_to_bytes("1P") == 1024**5
        assert _parse_mem_to_bytes("2PB") == 2 * 1024**5

    def test_case_insensitive(self) -> None:
        assert _parse_mem_to_bytes("4g") == 4 * 1024 * 1024 * 1024

    def test_unreadable_is_None_not_zero(self) -> None:
        """0 means "no limit enforced" downstream, so answering 0 for a spelling we
        can't read promotes a capped job to unlimited. None says "couldn't read
        it", which a caller can act on. SW-12."""
        assert _parse_mem_to_bytes("") is None
        assert _parse_mem_to_bytes("abc") is None
        assert _parse_mem_to_bytes("M") is None
        # A genuine zero is still zero — that one IS "no limit".
        assert _parse_mem_to_bytes("0") == 0

    def test_per_node_and_per_cpu_qualifiers(self) -> None:
        """Slurm <= 20.11 wrote ReqMem as `4Gn` / `500Mc`; the trailing letter says
        whose limit it is, which the caller already knows from the field it read.
        Returning 0 for these made an old cluster's every limit "unlimited". SW-12."""
        assert _parse_mem_to_bytes("4Gn") == 4 * 1024**3
        assert _parse_mem_to_bytes("500Mc") == 500 * 1024**2
        assert _parse_mem_to_bytes("1.5Tn") == int(1.5 * 1024**4)
        assert _parse_mem_to_bytes("2GBn") == 2 * 1024**3
        # A bare integer keeps its MiB meaning through the qualifier.
        assert _parse_mem_to_bytes("800c") == 800 * 1024**2

    def test_float_value(self) -> None:
        assert _parse_mem_to_bytes("1.5G") == int(1.5 * 1024**3)

    def test_two_letter_unit(self) -> None:
        # Tolerate the "64GB"/"512MB" form (trailing B), not silently return 0.
        assert _parse_mem_to_bytes("64GB") == 64 * 1024**3
        assert _parse_mem_to_bytes("512MB") == 512 * 1024**2

    def test_negative_never_passes_through(self) -> None:
        """Not a limit anyone can act on, so it reads as unreadable, not as 0
        (= unlimited)."""
        assert _parse_mem_to_bytes("-5G") is None
        assert _parse_mem_to_bytes("-5") is None


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

    def test_huge_range_is_capped_not_ooming(self) -> None:
        # #audit3-10: a garbage/huge NodeList must be bounded, not expand 100M
        # hosts into memory. Returns quickly with a capped list.
        from slurmwatch.slurm import _MAX_HOSTLIST_NODES

        result = _parse_nodelist("cn[1-100000000]")
        assert len(result) == _MAX_HOSTLIST_NODES

    def test_cartesian_blowup_is_capped(self) -> None:
        from slurmwatch.slurm import _MAX_HOSTLIST_NODES

        result = _parse_nodelist("a[1-10000]b[1-10000]")  # 100M product uncapped
        assert len(result) <= _MAX_HOSTLIST_NODES

    def test_stepped_range(self) -> None:
        # #39: Slurm's 'start-end:step' hostlist syntax used to collapse to one
        # garbage host ('node1-7:2') instead of expanding.
        assert _parse_nodelist("node[1-7:2]") == ["node1", "node3", "node5", "node7"]

    def test_stepped_range_preserves_padding(self) -> None:
        assert _parse_nodelist("gpu[001-007:2]") == ["gpu001", "gpu003", "gpu005", "gpu007"]

    def test_stepped_range_within_multi_group(self) -> None:
        assert _parse_nodelist("cn[1-4:2],cn[10]") == ["cn1", "cn3", "cn10"]

    def test_reversed_range_not_silently_dropped(self) -> None:
        # #39: a reversed range used to vanish (range(3,0) is empty), silently
        # shrinking the node count. Slurm never emits this, but it must not
        # disappear without trace — keep it verbatim.
        assert _parse_nodelist("node[3-1]") == ["node3-1"]

    def test_plain_range_unchanged(self) -> None:
        # Guard: the common non-stepped range must be byte-identical to before.
        assert _parse_nodelist("cn-[001-004]") == ["cn-001", "cn-002", "cn-003", "cn-004"]


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

    def test_expand_idx_list_caps_hostile_range(self) -> None:
        # #audit: a crafted GRES `IDX:0-2000000000` (e.g. via a hostile JobName the
        # raw GRES regex doesn't field-shadow) must not materialize billions of
        # ints and OOM/hang — capped like the NodeList expansion.
        out = slurm._expand_idx_list("0-2000000000")
        assert len(out) <= slurm._MAX_GPU_IDX


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

    def test_value_with_spaces_is_not_truncated(self) -> None:
        # #37: a value containing spaces (a path, args) must survive to the next
        # key= / end-of-line, not stop at the first space. Real scontrol prints
        # each free-text field on its OWN line (#audit3-9), so lay them out that way.
        output = "   Command=/home/me/my run.sh --flag\n   WorkDir=/scratch/proj\n"
        assert _parse_scontrol_field(output, "Command") == "/home/me/my run.sh --flag"
        assert _parse_scontrol_field(output, "WorkDir") == "/scratch/proj"

    def test_free_text_field_does_not_shadow_a_later_field(self) -> None:
        # #audit3-9: a crafted JobName containing "Partition=..." must NOT shadow
        # the real Partition on a later line, nor be truncated itself.
        output = "JobId=12345 JobName=x Partition=[/] evil\n   JobState=PENDING Partition=gpu\n"
        assert _parse_scontrol_field(output, "Partition") == "gpu"
        assert _parse_scontrol_field(output, "JobName") == "x Partition=[/] evil"
        assert _parse_scontrol_field(output, "JobState") == "PENDING"

    def test_value_with_spaces_at_end_of_line(self) -> None:
        assert _parse_scontrol_field("   WorkDir=/scratch/my project\n", "WorkDir") == (
            "/scratch/my project"
        )

    def test_substring_fields_still_isolated(self) -> None:
        # The greedier value capture must not weaken key matching: Mem must not
        # match MinMemoryNode, Nodes must not match NodeList/NumNodes.
        output = "MinMemoryNode=4G NumNodes=4 NodeList=cn[01-04]\n"
        assert _parse_scontrol_field(output, "Mem") is None
        assert _parse_scontrol_field(output, "Nodes") is None
        assert _parse_scontrol_field(output, "MinMemoryNode") == "4G"
        assert _parse_scontrol_field(output, "NumNodes") == "4"

    def test_tres_value_with_embedded_equals_kept_whole(self) -> None:
        # A comma-joined value with '=' inside but no spaces stays intact.
        assert _parse_scontrol_field("TRES=cpu=32,mem=16G,gres/gpu=4\n", "TRES") == (
            "cpu=32,mem=16G,gres/gpu=4"
        )

    def test_partition_not_over_captured_by_colon_key_neighbour(self) -> None:
        # #37 review: real scontrol prints "Partition=gpu AllocNode:Sid=host:pid"
        # on one line. The value boundary must recognise a colon-keyed neighbour,
        # or Partition over-captures the whole tail.
        output = "Partition=gpu AllocNode:Sid=login1:54321 JobState=RUNNING\n"
        assert _parse_scontrol_field(output, "Partition") == "gpu"
        assert _parse_scontrol_field(output, "JobState") == "RUNNING"
        assert _parse_scontrol_field(output, "AllocNode:Sid") == "login1:54321"

    def test_slash_key_neighbour_bounds_value(self) -> None:
        # Slash keys (gres/gpu) are also single tokens, not value boundaries mid-key.
        output = "Foo=bar gres/gpu=4\n"
        assert _parse_scontrol_field(output, "Foo") == "bar"
        assert _parse_scontrol_field(output, "gres/gpu") == "4"

    def test_long_trailing_whitespace_is_not_pathological(self) -> None:
        # The split parser is linear; the previous lazy-regex was O(n^2) on a long
        # trailing-whitespace run. Just assert it returns promptly and correctly.
        assert _parse_scontrol_field("Partition=" + " " * 5000, "Partition") == ""


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

    def test_short_day_forms(self) -> None:
        # After a day component the fields are HH[:MM[:SS]] (left-aligned to hours),
        # so D-HH / D-HH:MM must not read the last field as seconds.
        assert slurm._parse_slurm_duration("2-12") == 2 * 86400 + 12 * 3600
        assert slurm._parse_slurm_duration("2-12:30") == 2 * 86400 + 12 * 3600 + 30 * 60
        assert slurm._parse_slurm_duration("0-05") == 5 * 3600

    def test_leading_minus_is_not_a_day_separator(self) -> None:
        # The day form is `<digits>-HH…`, so the dash must FOLLOW leading digits. A
        # bare "-" test would read a stray "-5" as the day form and return 5 HOURS
        # (18000s) for a value meant to be seconds — a 3600x error. Slurm never emits
        # a negative duration, so this is defensive, but it must not silently inflate.
        assert abs(slurm._parse_slurm_duration("-5")) < 60
        assert slurm._parse_slurm_duration("-5") != 5 * 3600
        # A real day form still parses.
        assert slurm._parse_slurm_duration("5-00:00:00") == 5 * 86400


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

    def test_requeued_or_pending_job_is_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # #audit: a preempted/requeued job (PreemptMode=REQUEUE, `scontrol requeue`,
        # NODE_FAIL+--requeue) goes back to PENDING/REQUEUED but reruns under the
        # same id — it must NOT be read as "ended" and tear the dashboard down.
        for state in ("PENDING", "REQUEUED", "REQUEUE_HOLD"):
            monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, s=state, **k: f"{s}\n")
            assert slurm.is_job_active("123") is True, state

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

    def test_query_timeout_is_unknown_not_ended(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A single-job squeue can take >15s on a loaded login node; a timeout must
        # be "unknown" (None), never read as ended — otherwise a slow controller
        # would tear down a live dashboard. The ping is not even consulted here.
        def _cmd(cmd: list[str], *a: object, **k: object) -> str:
            raise SlurmCommandError("Command squeue -h -j 123 -o %T timed out after 15s")

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

    def test_scopes_to_requested_job_not_whole_array(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # #audit: `sstat -j <ArrayJobId>` widens to EVERY running task (the first
        # task's raw id == the ArrayJobId). Only the requested job's rows may
        # count, else CPU is summed across the whole array (pins the node ~100%).
        sstat_out = (
            "12345.batch|1000K|00:01:00|1\n"
            "12345.0|1000K|00:01:00|1\n"
            "12346.batch|1000K|00:05:00|1\n"  # a DIFFERENT array task -> ignored
            "12347.0|1000K|00:09:00|1\n"  # ditto
        )
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: sstat_out)
        usage = slurm.resolve_remote_usage("12345")
        assert usage.cpu_seconds == 120.0  # only the two 12345.* rows (60s each)

    def test_sstat_failure_is_graceful(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from slurmwatch.exceptions import SlurmCommandError

        def _fail(*a: object, **k: object) -> str:
            raise SlurmCommandError("sstat: no steps")

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _fail)
        usage = slurm.resolve_remote_usage("99")
        assert usage.sampled is False
        assert usage.rss_bytes == 0


class TestSacctFinishedJob:
    """A finished job (purged from scontrol, still in sacct) must report 'finished',
    not 'does not exist' (#audit — no sacct fallback before)."""

    def _wire(self, monkeypatch: pytest.MonkeyPatch, sacct_out: str) -> None:
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)

        def _cmd(cmd: list[str], *a: object, **k: object) -> str:
            if cmd and cmd[0] == "scontrol":
                raise SlurmCommandError("Invalid job id specified")
            if cmd and cmd[0] == "sacct":
                return sacct_out
            return ""

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _cmd)

    def test_finished_job_reports_finished(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._wire(monkeypatch, "COMPLETED|2026-07-13T10:00:00\n")
        with pytest.raises(JobNotRunningError, match="has finished"):
            slurm.resolve_job_context("12345")

    def test_cancelled_reduced_to_state_word(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._wire(monkeypatch, "CANCELLED by 1001|2026-07-13T10:00:00\n")
        with pytest.raises(JobNotRunningError, match="CANCELLED"):
            slurm.resolve_job_context("12345")

    def test_truly_missing_is_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._wire(monkeypatch, "\n")  # sacct empty -> the job never existed
        with pytest.raises(JobNotFoundError):
            slurm.resolve_job_context("99999")


class TestCountHetComponents:
    def test_het_multiple_components(self) -> None:
        out = "JobId=500+0 JobState=RUNNING\n\nJobId=500+1 JobState=RUNNING\n"
        assert slurm._count_het_components(out) == 2

    def test_array_tasks_are_not_het(self) -> None:
        # Array tasks have plain numeric ids (no +component), so this must be 0.
        out = "JobId=500 JobState=RUNNING\n\nJobId=501 JobState=RUNNING\n"
        assert slurm._count_het_components(out) == 0

    def test_het_looking_text_in_a_free_text_field_is_not_counted(self) -> None:
        # A JobName/Comment/Command containing the literal substring
        # "JobId=1+1" must not make an ordinary job look heterogeneous — the
        # real JobId field (500, not het-shaped) is what counts, not raw text
        # matched anywhere in the dump.
        out = "JobId=500 JobName=fake JobId=1+1 het JobState=RUNNING\n"
        assert slurm._count_het_components(out) == 0


class TestLocalNodeName:
    def test_prefers_slurmd_nodename(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # On clusters aliasing NodeName != NodeHostname, $SLURMD_NODENAME is the
        # authoritative match against a job's NodeList (#audit).
        monkeypatch.setenv("SLURMD_NODENAME", "GPU-B-02.cluster")
        assert local_node_name() == "gpu-b-02"  # short + lower-cased

    def test_falls_back_to_hostname(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLURMD_NODENAME", raising=False)
        assert local_node_name()  # non-empty short host off-node


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

    def test_includes_running_and_pending_skips_others(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The picker offers RUNNING and PENDING jobs (a pending pick routes to the
        # why/when/where view); other transient states (completing) are skipped.
        squeue = (
            "1|PD|gpu|2|0:00|1:00:00|(Priority)|queued_job\n"
            "2|R|gpu|1|0:10|1:00:00|cn001|run_job\n"
            "3|CG|gpu|1|0:05|1:00:00|None|finishing\n"
        )
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: squeue)
        jobs = resolve_current_jobs("u")
        assert {str(j["job_id"]): j["state"] for j in jobs} == {"1": "PD", "2": "R"}


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
    # Its own line, last field, exactly as scontrol prints it — JobName is free text,
    # so anything after it on the line would be swallowed into the name.
    "JobName=train-llama-8b\n"
    "NodeList=cn-[001-004] NumCPUs=16 NumNodes=1\n"
    "TRES=cpu=16,mem=64G,gres/gpu=4\n"
    "AllocTRES=cpu=16,mem=64G,gres/gpu=4\n"
    "RunTime=01:00:00 TimeLimit=1-00:00:00\n"
    "SubmitTime=2024-01-15T10:29:00 MinMemoryNode=64G StartTime=2024-01-15T10:30:00 "
    "UserId=user(1001)\n"
    "   Command=/home/user/proj/train.py\n"
    "   WorkDir=/home/user/proj/runs\n"
    "   StdOut=/home/user/proj/runs/slurm-12345.out\n"
    "   StdErr=/home/user/proj/runs/slurm-12345.err\n"
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
        # B3: the compact scontrol NodeList is preserved (not discarded), and it's
        # what displays use — so a wide job shows "cn-[001-004]", not 500 names —
        # while the expanded form stays available for logic.
        assert ctx.nodelist_compact == "cn-[001-004]"
        assert ctx.nodelist == "cn-001,cn-002,cn-003,cn-004"
        assert ctx.nodelist_display == "cn-[001-004]"
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
        # The name the USER gave the job (sbatch -J). It reached the selector and the
        # pending view but never the running-job context, so the dashboard — the one
        # screen you actually sit on — couldn't say which experiment it was watching.
        assert ctx.job_name == "train-llama-8b"
        assert ctx.account == "rcc-staff"
        assert ctx.qos == "normal"
        assert ctx.command == "/home/user/proj/train.py"
        assert ctx.work_dir == "/home/user/proj/runs"
        assert ctx.std_out == "/home/user/proj/runs/slurm-12345.out"
        assert ctx.std_err == "/home/user/proj/runs/slurm-12345.err"
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
        # Array membership is captured for the UI.
        assert ctx.array_job_id == "12345"
        assert ctx.array_task_id == "3"

    def test_array_bare_base_relabels_to_resolved_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `sw 52353625` (the bare array base) resolves to one running task; the
        # context is relabelled to the task actually picked so the header isn't a
        # misleading array-wide id.
        output = (
            "JobId=12348 ArrayJobId=12345 ArrayTaskId=3 JobState=RUNNING Partition=gpu\n"
            "NodeList=cn-001 NumCPUs=4 NumNodes=1\n"
            "TRES=cpu=4,mem=8G,node=1\n"
            "UserId=user(1001) StartTime=2024-01-15T10:30:00\n"
        )
        self._patch_common(monkeypatch, output)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        ctx = resolve_job_context("12345")  # bare base, no "_"
        assert ctx.job_id == "12345_3"
        assert ctx.array_job_id == "12345"
        assert ctx.array_task_id == "3"

    def test_non_array_job_has_empty_array_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_common(monkeypatch, _SAMPLE_SCONTROL)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        ctx = resolve_job_context("12345")
        assert ctx.array_job_id == ""
        assert ctx.array_task_id == ""
        assert ctx.job_id == "12345"  # label untouched for a non-array job

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
        # #33: the per-node GPU *count* also comes from this node's IDX detail (2),
        # not the job-wide 3 // 2 = 1 that used to contradict the rendered rows.
        assert ctx.gpu_count_requested == 2

    _HETERO_2NODE = (
        "JobId=42 JobState=RUNNING Partition=gpu\n"
        "NodeList=cn-[001-002] NumCPUs=24 NumNodes=2\n"
        "TRES=cpu=24,mem=160G,gres/gpu=3 UserId=user(1001) StartTime=2024-01-15T10:30:00\n"
        "   Nodes=cn-001 CPU_IDs=0-7 Mem=64000 GRES=gpu:1(IDX:0)\n"
        "   Nodes=cn-002 CPU_IDs=0-15 Mem=96000 GRES=gpu:2(IDX:1-2)\n"
    )

    def test_per_node_gpu_count_from_idx_detail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # #33: on cn-001 the node holds 1 GPU (IDX:0); the job-wide // NumNodes
        # would also give 1 here, so assert on cn-002 (2 GPUs) where they differ.
        self._patch_common(monkeypatch, self._HETERO_2NODE)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        ctx = resolve_job_context("42")
        assert ctx.cpus_allocated == 8
        assert ctx.mem_limit_bytes == 64000 * 1024**2
        assert ctx.gpu_count_requested == 1

    def test_off_node_scopes_to_first_node_not_job_average(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #31: viewed from a login node (host in no detail line), the per-node
        # limits must be nodelist[0]=cn-001's exact figures (8 CPU / 64000 MB /
        # 1 GPU) — the node the collector will represent — not the job-wide
        # average (24//2=12 CPU, 3//2=1 GPU) that matches no real node.
        def _no_cgroup(*a: object, **k: object) -> dict[str, object]:
            raise CgroupNotFoundError("off-node")

        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: self._HETERO_2NODE)
        monkeypatch.setattr(slurm, "_resolve_uid", lambda u: 1001)
        monkeypatch.setattr(slurm, "_discover_cgroup_paths", _no_cgroup)
        monkeypatch.setattr("socket.gethostname", lambda: "login-01")
        ctx = resolve_job_context("42")
        assert ctx.remote is True
        assert ctx.cpus_allocated == 8  # cn-001, not 12
        assert ctx.mem_limit_bytes == 64000 * 1024**2
        assert ctx.gpu_count_requested == 1

    def test_uneven_division_rounds_up_when_no_detail_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #32: with no -d detail line, dividing 30 CPUs over 4 nodes must round UP
        # to 8 (the largest real node) not truncate to 7, so % isn't inflated
        # against a too-small limit (a false-OOM vector); same for 6 GPUs -> 2.
        output = (
            "JobId=7 JobState=RUNNING Partition=gpu\n"
            "NodeList=cn-[01-04] NumCPUs=30 NumNodes=4\n"
            "TRES=cpu=30,mem=240G,gres/gpu=6 UserId=user(1001) StartTime=2024-01-15T10:30:00\n"
        )
        self._patch_common(monkeypatch, output)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-01")
        ctx = resolve_job_context("7")
        assert ctx.cpus_allocated == 8  # ceil(30/4), not floor 7
        assert ctx.gpu_count_requested == 2  # ceil(6/4)

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

    def test_per_node_gpu_map_is_populated_off_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The job-wide GPU map must survive the OFF-NODE resolution path.

        It is derived from the scontrol record alone, but was only assigned on the
        on-node path (which returns much later), so a login-node --json and the
        cross-node GPU view silently got {} for a multi-node job. Asserts on the
        CONTEXT, not on the parser — the parser passing proves nothing about the
        wire, which is exactly how this regressed unnoticed.
        """
        output = (
            "JobId=42 JobState=RUNNING Partition=gpu\n"
            "NodeList=cn[001-002] NumCPUs=8 NumNodes=2\n"
            "TRES=cpu=8,mem=128G,node=2,gres/gpu=4\n"
            "MinMemoryNode=64G UserId=user(1001) StartTime=2024-01-15T10:30:00\n"
            "   Nodes=cn001 CPU_IDs=0-3 Mem=48000 GRES=gpu:2(IDX:0-1)\n"
            "   Nodes=cn002 CPU_IDs=0-3 Mem=48000 GRES=gpu:2(IDX:2-3)\n"
        )
        self._patch_common(monkeypatch, output)
        monkeypatch.setattr("socket.gethostname", lambda: "login1")
        # THE off-node condition is a missing cgroup, not the hostname: that is what
        # makes resolve_job_context take its early return. Without forcing it the
        # test runs the on-node path, which re-assigns the map further down and would
        # keep passing even with the off-node assignment deleted.
        from slurmwatch.exceptions import CgroupNotFoundError

        def _no_cgroup(*_a: object, **_k: object) -> dict[str, object]:
            raise CgroupNotFoundError("not on the job's node")

        monkeypatch.setattr("slurmwatch.slurm._discover_cgroup_paths", _no_cgroup)
        ctx = resolve_job_context("42")
        assert ctx.remote is True, "test must exercise the off-node path"
        assert ctx.gpu_indices_by_node == {"cn001": [0, 1], "cn002": [2, 3]}
        # And the per-process fields stay empty: nothing is attachable from here.
        assert ctx.gpu_indices == []

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
        # A leading alphanumeric run-on must not match either ("xjob_123" != job 123).
        assert slurm._cgroup_name_matches_job("xjob_123", "123") is False
        # A real path boundary before job_ is fine.
        assert slurm._cgroup_name_matches_job("slurm.slice/job_123.scope", "123") is True


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

    def test_het_component_uses_numeric_base(
        self, fake_cgroup_v2: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # cross-cutting F3: a het component id "12345+0" must resolve to job_12345,
        # not the never-existent job_12345+0.
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", fake_cgroup_v2)
        job = fake_cgroup_v2 / "system.slice" / "slurmstepd.scope" / "job_12345"
        job.mkdir(parents=True, exist_ok=True)
        (job / "cgroup.procs").write_text("")
        paths = slurm._discover_cgroup_paths("12345+0", uid=1001, step_id=None)
        assert paths["v2"] == job

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

    def _v1_daemon_tree(self, base: Path, uid: int, job: str) -> Path:
        """A cgroup-v1 layout with NO per-job cpuacct — only the node-wide daemon
        counter — but a real per-job memory cgroup (the shape of a cluster that
        delegates memory/cpuset per job but not cpuacct)."""
        daemon = base / "cpuacct" / "system.slice" / "slurmd.service"
        daemon.mkdir(parents=True)
        (daemon / "cgroup.procs").write_text("")
        mem_job = base / "memory" / "slurm" / f"uid_{uid}" / f"job_{job}"
        mem_job.mkdir(parents=True)
        (mem_job / "cgroup.procs").write_text("")
        return daemon

    def test_v1_rejects_node_wide_daemon_cpuacct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # P1: on a node with no per-job cpuacct cgroup, our own /proc/self/cgroup
        # names the daemon's NODE-WIDE counter (/system.slice/slurmd.service).
        # Reading that adds every CO-TENANT job's CPU time to this job's usage — a
        # silent over-report the [0, cores] clamp then hides ("100%, perfectly
        # sized"). It must be REJECTED for lack of a job_<id> component so the
        # reader falls through to the job-scoped per-PID /proc sum.
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", tmp_path)
        self._v1_daemon_tree(tmp_path, uid=1001, job="12345")
        monkeypatch.setattr(
            slurm, "_read_self_cgroup", lambda: "4:cpu,cpuacct:/system.slice/slurmd.service\n"
        )
        # The ownership guard ALONE would accept it: every job step rolls up into
        # slurmd.service, and our own SLURM_JOB_ID matches. Only the job-scope
        # requirement rejects it.
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        paths = slurm._discover_cgroup_paths("12345", uid=1001, step_id=None)
        assert paths["v1_cpu"] is None
        # Memory is unaffected — its real per-job cgroup is found by exact path.
        assert paths["v1_mem"] is not None
        assert paths["v1_mem"].name == "job_12345"

    def test_v1_accepts_job_scoped_cpuacct_from_self_cgroup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The other side of P1: a genuine per-job cpuacct cgroup found only via
        # /proc/self/cgroup (a non-standard layout the exact-path candidates miss)
        # DOES carry a job_<id> component, so it must still be accepted — the fix
        # must not blind the fallback it was added to.
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", tmp_path)
        self._v1_daemon_tree(tmp_path, uid=1001, job="12345")
        job_cpu = tmp_path / "cpuacct" / "custom.slice" / "job_12345"
        job_cpu.mkdir(parents=True)
        (job_cpu / "cgroup.procs").write_text("")
        monkeypatch.setattr(
            slurm,
            "_read_self_cgroup",
            lambda: "4:cpu,cpuacct:/custom.slice/job_12345/step_0\n",
        )
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        paths = slurm._discover_cgroup_paths("12345", uid=1001, step_id=None)
        assert paths["v1_cpu"] == job_cpu


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


class TestRunSlurmCmdErrors:
    """_run_slurm_cmd converts subprocess failures into SlurmCommandError so every
    caller's `except SlurmCommandError` handles them, not a raw traceback."""

    def test_file_not_found_becomes_slurm_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*a: object, **k: object) -> None:
            raise FileNotFoundError(2, "No such file or directory")

        monkeypatch.setattr(subprocess, "run", _boom)
        with pytest.raises(SlurmCommandError, match="not found"):
            slurm._run_slurm_cmd(["squeue"])

    def test_fork_eagain_oserror_becomes_slurm_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # N4: at RLIMIT_NPROC on a busy login node, subprocess.run's fork raises
        # BlockingIOError (an OSError, NOT FileNotFoundError). It must be converted,
        # not escape as a raw traceback past every `except SlurmCommandError`.
        def _boom(*a: object, **k: object) -> None:
            raise BlockingIOError(11, "Resource temporarily unavailable")

        monkeypatch.setattr(subprocess, "run", _boom)
        with pytest.raises(SlurmCommandError, match="could not run"):
            slurm._run_slurm_cmd(["squeue"])


class TestSacctFinalStateTerminalOnly:
    """Regression (S2): a still-active sacct row must NOT be reported as 'finished'.

    A transient `scontrol show job -d` failure on a busy controller fell back to
    sacct; when sacct returned a live `RUNNING|Unknown` row, the old code handed it
    back as terminal, so resolve_job_context raised "has finished (State: RUNNING)"
    for a running job — which the TUI surfaced as the bogus "…not PENDING.".
    """

    @pytest.mark.parametrize("state", ["RUNNING", "PENDING", "SUSPENDED", "REQUEUED", "COMPLETING"])
    def test_active_states_are_not_finished(
        self, state: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, s=state, **k: f"{s}|Unknown\n")
        assert _sacct_final_state("123") is None

    def test_terminal_state_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(
            slurm, "_run_slurm_cmd", lambda *a, **k: "COMPLETED|2026-07-16T10:00:00\n"
        )
        assert _sacct_final_state("123") == ("COMPLETED", "2026-07-16T10:00:00")

    def test_cancelled_by_uid_reduced_to_first_word(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(
            slurm, "_run_slurm_cmd", lambda *a, **k: "CANCELLED by 1234|2026-07-16T10:00:00\n"
        )
        assert _sacct_final_state("123") == ("CANCELLED", "2026-07-16T10:00:00")

    def test_no_record_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "\n")
        assert _sacct_final_state("123") is None

    def test_mixed_array_with_any_active_row_is_not_finished(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # M1: a bare array-base id makes `sacct -X` emit one row per task. A terminal
        # task ordered BEFORE a still-RUNNING one must not classify the whole array
        # as finished — any active row means the job is still live.
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        out = "COMPLETED|2026-07-16T10:00:00\nRUNNING|Unknown\nCOMPLETED|2026-07-16T10:05:00\n"
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: out)
        assert _sacct_final_state("123") is None

    def test_active_row_after_terminal_rows_is_not_finished(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Order-independence: the still-active row can appear last and must still win.
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        out = "COMPLETED|2026-07-16T10:00:00\nFAILED|2026-07-16T10:01:00\nPENDING|Unknown\n"
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: out)
        assert _sacct_final_state("123") is None

    def test_all_terminal_array_reports_a_terminal_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When EVERY task is terminal the array really has finished: report a
        # terminal (state, end).
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        out = "COMPLETED|2026-07-16T10:00:00\nFAILED|2026-07-16T10:01:00\n"
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: out)
        assert _sacct_final_state("123") == ("COMPLETED", "2026-07-16T10:00:00")


class TestResolveJobContextTransientFailure:
    """Regression (S2): a transient scontrol timeout on a RUNNING job must not be
    reported as 'finished' or 'not found' — it surfaces as a clear, retryable error."""

    def _fake_run(self, scontrol_exc: str, sacct_out: str) -> Callable[..., str]:
        def _run(cmd: list[str], *a: object, **k: object) -> str:
            if cmd[:3] == ["scontrol", "show", "job"]:
                raise SlurmCommandError(scontrol_exc)
            if cmd and cmd[0] == "sacct":
                return sacct_out
            raise SlurmCommandError(f"unexpected cmd {cmd}")

        return _run

    def test_running_job_on_timeout_is_retryable_not_finished(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(
            slurm,
            "_run_slurm_cmd",
            self._fake_run(
                "Command scontrol show job -d 123 timed out after 15s", "RUNNING|Unknown\n"
            ),
        )
        with pytest.raises(SlurmCommandError) as ei:
            resolve_job_context("123")
        msg = str(ei.value)
        assert "try again" in msg.lower()
        assert "has finished" not in msg  # old misclassification is gone

    def test_finished_job_still_reported_finished(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(
            slurm,
            "_run_slurm_cmd",
            self._fake_run("Invalid job id specified", "COMPLETED|2026-07-16T10:00:00\n"),
        )
        with pytest.raises(JobNotRunningError) as ei:
            resolve_job_context("123")
        assert "has finished" in str(ei.value)

    def test_nonexistent_job_is_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(
            slurm, "_run_slurm_cmd", self._fake_run("Invalid job id specified", "\n")
        )
        with pytest.raises(JobNotFoundError):
            resolve_job_context("123")


class TestResolveArrayTaskCounts:
    """(running, pending) sibling counts for an array via ``squeue -r``."""

    def test_counts_running_and_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            slurm, "_run_slurm_cmd", lambda *a, **k: "RUNNING\nRUNNING\nRUNNING\nPENDING\n"
        )
        assert slurm.resolve_array_task_counts("12345") == (3, 1)

    def test_counts_transient_active_states(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # COMPLETING/SUSPENDED/CONFIGURING/RESIZING are active (not pending, not
        # done), so count them as running rather than dropping them.
        monkeypatch.setattr(
            slurm, "_run_slurm_cmd", lambda *a, **k: "RUNNING\nCOMPLETING\nPENDING\nCONFIGURING\n"
        )
        assert slurm.resolve_array_task_counts("12345") == (3, 1)

    def test_counts_signaling_and_requeued(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # SIGNALING/REQUEUED are transient active states squeue -r can return; count
        # them as running rather than dropping them into neither bucket (A6 sibling).
        monkeypatch.setattr(
            slurm, "_run_slurm_cmd", lambda *a, **k: "RUNNING\nSIGNALING\nREQUEUED\nPENDING\n"
        )
        assert slurm.resolve_array_task_counts("12345") == (3, 1)

    def test_only_transient_states_still_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An array mid-teardown (only COMPLETING tasks) reports them, not None.
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "COMPLETING\nSUSPENDED\n")
        assert slurm.resolve_array_task_counts("12345") == (2, 0)

    def test_empty_result_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Nothing live (all tasks finished) → None, so the caller omits the line
        # rather than printing a fabricated "0 running, 0 pending".
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "\n")
        assert slurm.resolve_array_task_counts("12345") is None

    def test_command_failure_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*a: object, **k: object) -> str:
            raise SlurmCommandError("controller busy")

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _raise)
        assert slurm.resolve_array_task_counts("12345") is None

    def test_no_array_id_is_none(self) -> None:
        assert slurm.resolve_array_task_counts("") is None

    def test_mock_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        assert slurm.resolve_array_task_counts("12345") is None


class TestCgroupContentParsers:
    """Name-agnostic cgroup-path parsing (survives the Slurm 26.05 SLUID rename)."""

    def test_v2_job_dir_job_named(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", tmp_path)
        content = "0::/system.slice/slurmstepd.scope/job_12345/step_0/user/task_0\n"
        assert slurm._v2_job_dir_from_cgroup_content(content) == (
            tmp_path / "system.slice" / "slurmstepd.scope" / "job_12345"
        )

    def test_v2_job_dir_sluid_named(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Slurm 26.05 default: an opaque SLUID instead of job_<id>.
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", tmp_path)
        content = "0::/system.slice/slurmstepd.scope/sEKNKTV3WPV500/step_0/user/task_0\n"
        assert slurm._v2_job_dir_from_cgroup_content(content) == (
            tmp_path / "system.slice" / "slurmstepd.scope" / "sEKNKTV3WPV500"
        )

    def test_v2_job_dir_not_a_slurm_cgroup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", tmp_path)
        assert slurm._v2_job_dir_from_cgroup_content("0::/user.slice/user-1000.slice\n") is None

    def test_v2_job_dir_ignores_v1_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", tmp_path)
        assert slurm._v2_job_dir_from_cgroup_content("5:cpuacct:/slurm/uid_1/job_1\n") is None

    def test_v1_job_dir_per_controller(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", tmp_path)
        content = (
            "9:memory:/slurm/uid_1000/job_777/step_0\n"
            "8:cpu,cpuacct:/slurm/uid_1000/job_777/step_0\n"
        )
        assert slurm._v1_job_dir_from_cgroup_content(content, "memory") == (
            tmp_path / "memory" / "slurm" / "uid_1000" / "job_777"
        )
        assert slurm._v1_job_dir_from_cgroup_content(content, "cpuacct") == (
            tmp_path / "cpuacct" / "slurm" / "uid_1000" / "job_777"
        )


class TestCgroupDiscoverySluid:
    """Slurm 26.05 renames the job cgroup dir to an opaque SLUID; discovery must
    still find it — via /proc/self/cgroup or by process membership — instead of
    silently degrading to the remote sstat view (P1-b of the cluster audit)."""

    @staticmethod
    def _make_sluid_job(fake_cgroup_v2: Path, pid: str = "4242") -> Path:
        scope = fake_cgroup_v2 / "system.slice" / "slurmstepd.scope"
        job = scope / "sEKNKTV3WPV500"
        task = job / "step_0" / "user" / "task_0"
        task.mkdir(parents=True)
        (job / "cgroup.procs").write_text("")
        (task / "cgroup.procs").write_text(f"{pid}\n")
        return job

    def test_found_via_self_cgroup(
        self, fake_cgroup_v2: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", fake_cgroup_v2)
        job = self._make_sluid_job(fake_cgroup_v2)
        monkeypatch.setattr(
            slurm,
            "_read_self_cgroup",
            lambda: "0::/system.slice/slurmstepd.scope/sEKNKTV3WPV500/step_0/user/task_0\n",
        )
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        paths = slurm._discover_cgroup_paths("12345", uid=1001, step_id=None)
        assert paths["v2"] == job

    def test_found_via_membership(
        self, fake_cgroup_v2: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Not inside the cgroup ourselves; identify it by a worker PID's environ.
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", fake_cgroup_v2)
        job = self._make_sluid_job(fake_cgroup_v2)
        monkeypatch.setattr(slurm, "_read_self_cgroup", lambda: "0::/user.slice/user-1.slice\n")
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_JOBID", raising=False)
        monkeypatch.setattr(
            slurm,
            "_read_pid_environ",
            lambda pid: {"SLURM_JOB_ID": "12345"} if pid == 4242 else {},
        )
        paths = slurm._discover_cgroup_paths("12345", uid=1001, step_id=None)
        assert paths["v2"] == job

    def test_self_cgroup_rejected_for_wrong_job(
        self, fake_cgroup_v2: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # We sit in job 999's cgroup but asked for 12345: must not misattribute.
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", fake_cgroup_v2)
        other = fake_cgroup_v2 / "system.slice" / "slurmstepd.scope" / "sOTHER"
        (other / "step_0").mkdir(parents=True)
        (other / "cgroup.procs").write_text("")
        (other / "step_0" / "cgroup.procs").write_text("77\n")
        monkeypatch.setattr(
            slurm,
            "_read_self_cgroup",
            lambda: "0::/system.slice/slurmstepd.scope/sOTHER/step_0/user/task_0\n",
        )
        monkeypatch.setenv("SLURM_JOB_ID", "999")
        monkeypatch.setattr(slurm, "_read_pid_environ", lambda pid: {"SLURM_JOB_ID": "999"})
        with pytest.raises(CgroupNotFoundError):
            slurm._discover_cgroup_paths("12345", uid=1001, step_id=None)

    def test_v1_self_cgroup_rejected_for_wrong_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A2: the v1 /proc/self/cgroup fallback must verify ownership like the v2
        # path — otherwise `sw <other-job>` from inside our own allocation on a
        # cgroup-v1 node silently binds OUR cgroup under the requested job's label.
        base = tmp_path / "cg"
        # No cgroup.controllers file -> detect_cgroup_version() == 1 (pure v1).
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", base)
        for ctl in ("memory", "cpuacct"):
            d = base / ctl / "slurm" / "uid_1001" / "job_999"
            d.mkdir(parents=True)
            (d / "cgroup.procs").write_text("77\n")
        monkeypatch.setattr(
            slurm,
            "_read_self_cgroup",
            lambda: "5:memory:/slurm/uid_1001/job_999\n4:cpuacct:/slurm/uid_1001/job_999\n",
        )
        monkeypatch.setenv("SLURM_JOB_ID", "999")
        monkeypatch.setattr(slurm, "_read_pid_environ", lambda _pid: {"SLURM_JOB_ID": "999"})
        # Asked for 12345 while sitting in 999 -> the fallback must refuse, not
        # bind 999's cgroup under 12345's label.
        with pytest.raises(CgroupNotFoundError):
            slurm._discover_cgroup_paths("12345", uid=None, step_id=None)


class TestParseGresIdxByNode:
    """The whole job's GPU layout from one `scontrol show job -d` record.

    Free of extra Slurm calls and of any srun hop, so a multi-node job's GPU
    allocation is answerable from a login node — which matters because a monitor
    step cannot read GPU utilization at all when the job holds every GPU.
    """

    REAL = (
        "   NumNodes=2 NumCPUs=8 NumTasks=2 CPUs/Task=4\n"
        "   JOB_GRES=gpu:4\n"
        "     Nodes=beagle3-0015 CPU_IDs=1-2,4-5 Mem=53248 GRES=gpu:2(IDX:0,2)\n"
        "     Nodes=beagle3-0020 CPU_IDs=2-5 Mem=53248 GRES=gpu:2(IDX:1-2)\n"
    )

    def test_maps_each_node_to_its_own_indices(self) -> None:
        from slurmwatch.slurm import parse_gres_idx_by_node

        assert parse_gres_idx_by_node(self.REAL) == {
            "beagle3-0015": [0, 2],
            "beagle3-0020": [1, 2],
        }

    def test_expands_a_node_range_line(self) -> None:
        """One line can name many nodes; each holds that same index set."""
        from slurmwatch.slurm import parse_gres_idx_by_node

        rec = "     Nodes=cn[001-003] CPU_IDs=0-15 Mem=64000 GRES=gpu:a100:2(IDX:0-1)\n"
        assert parse_gres_idx_by_node(rec) == {
            "cn001": [0, 1],
            "cn002": [0, 1],
            "cn003": [0, 1],
        }

    def test_cpu_only_job_has_no_entries(self) -> None:
        from slurmwatch.slurm import parse_gres_idx_by_node

        assert parse_gres_idx_by_node("     Nodes=cn001 CPU_IDs=0-3 Mem=8000\n") == {}

    def test_ignores_a_record_without_the_detail_flag(self) -> None:
        """`scontrol show job` (no -d) carries no per-node GRES lines at all."""
        from slurmwatch.slurm import parse_gres_idx_by_node

        assert parse_gres_idx_by_node("JobId=1 JobState=RUNNING\n   TRES=cpu=8,gres/gpu=4\n") == {}

    def test_does_not_union_indices_across_nodes(self) -> None:
        """Each node keeps ITS allocation; merging them would misreport every node.

        beagle3-0015 holds 0,2 and beagle3-0020 holds 1,2 — a union (0,1,2) would
        claim three GPUs on each of a job that has two per node.
        """
        from slurmwatch.slurm import parse_gres_idx_by_node

        out = parse_gres_idx_by_node(self.REAL)
        assert out["beagle3-0015"] == [0, 2]
        assert out["beagle3-0020"] == [1, 2]
        assert all(len(v) == 2 for v in out.values())


class TestUserIdentityFromScontrol:
    """SW-1: take the uid Slurm printed, not the name it printed.

    On a compute node whose passwd lookup can't resolve the running uid, scontrol
    renders the owner as `nobody(<uid>)`. Re-resolving THAT name succeeds — nobody
    is a real account — so the old path substituted uid 99 for the real uid with
    nothing raised, and slurmwatch then treated the user's own job as a stranger's.
    """

    def test_takes_the_numeric_uid_over_the_name(self) -> None:
        name, uid = slurm._parse_user_id("nobody(940740146)")
        assert uid == 940740146, "the uid in the parens is the authoritative one"
        assert name != "nobody", "the placeholder must not be shown as an owner"

    def test_a_resolvable_name_is_kept(self) -> None:
        assert slurm._parse_user_id("youzhi(940740146)") == ("youzhi", 940740146)

    def test_falls_back_to_getpwnam_only_when_slurm_printed_no_uid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_resolve_uid", lambda n: 4242 if n == "alice" else None)
        assert slurm._parse_user_id("alice") == ("alice", 4242)
        assert slurm._parse_user_id("") == ("", None)

    def test_strips_a_realm_suffix(self) -> None:
        assert slurm._parse_user_id("alice@CLUSTER(1001)") == ("alice", 1001)

    def test_your_own_job_is_never_labelled_nobody(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The consequence that mattered: `sw <your own job>` printing "user nobody"
        and "Ask nobody to run slurmwatch" instead of a dashboard."""
        monkeypatch.setattr("os.getuid", lambda: 940740146)
        monkeypatch.setattr("pwd.getpwuid", lambda uid: (_ for _ in ()).throw(KeyError(uid)))
        monkeypatch.setenv("USER", "youzhi")
        name, uid = slurm._parse_user_id("nobody(940740146)")
        assert (name, uid) == ("youzhi", 940740146)

    def test_owner_comparison_uses_the_real_uid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from slurmwatch.model import JobContext

        monkeypatch.setattr("os.getuid", lambda: 940740146)
        _, uid = slurm._parse_user_id("nobody(940740146)")
        ctx = JobContext(
            job_id="1",
            username="nobody",
            partition="build",
            nodelist="midway2-0300",
            hostname="midway2-0300",
            cpus_allocated=2,
            mem_limit_bytes=1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
            uid=uid,
        )
        assert slurm._job_owner_differs(ctx) is False

    def test_context_carries_the_scontrol_uid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record = _SAMPLE_SCONTROL.replace("UserId=user(1001)", "UserId=nobody(940740146)")
        TestResolveJobContext._patch_common(monkeypatch, record)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        monkeypatch.setattr(slurm, "_resolve_uid", lambda n: 99)  # what getpwnam would say
        ctx = resolve_job_context("12345")
        assert ctx.uid == 940740146


class TestCurrentUsername:
    """SW-11: auto-discovery found nothing under cron/systemd/`su`, because the
    chain ended at "" and `squeue -u ""` returns zero rows — reported as "you have
    no jobs" while the job was running."""

    def test_resolves_from_the_uid_when_the_environment_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("USER", raising=False)
        monkeypatch.delenv("LOGNAME", raising=False)
        monkeypatch.setattr("os.getuid", lambda: 940740146)
        monkeypatch.setattr("pwd.getpwuid", lambda uid: type("P", (), {"pw_name": "youzhi"})())
        assert slurm.current_username() == "youzhi"

    def test_prefers_the_uid_over_a_stale_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`su otheruser` (no dash) / `sudo -E` leaves $USER pointing at the old
        login; the uid can't be spoofed by the environment."""
        monkeypatch.setenv("USER", "root")
        monkeypatch.setattr("os.getuid", lambda: 940740146)
        monkeypatch.setattr("pwd.getpwuid", lambda uid: type("P", (), {"pw_name": "youzhi"})())
        assert slurm.current_username() == "youzhi"

    def test_falls_back_to_the_env_when_the_uid_does_not_resolve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("USER", "youzhi")
        monkeypatch.setattr("os.getuid", lambda: 940740146)
        monkeypatch.setattr("pwd.getpwuid", lambda uid: (_ for _ in ()).throw(KeyError(uid)))
        assert slurm.current_username() == "youzhi"

    def test_last_resort_is_the_numeric_uid_never_an_empty_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`squeue -u` documents accepting a numeric uid, and "" is the one answer
        that silently means "everyone's jobs, filtered to none"."""
        monkeypatch.delenv("USER", raising=False)
        monkeypatch.delenv("LOGNAME", raising=False)
        monkeypatch.setattr("os.getuid", lambda: 940740146)
        monkeypatch.setattr("pwd.getpwuid", lambda uid: (_ for _ in ()).throw(KeyError(uid)))
        assert slurm.current_username() == "940740146"

    def test_a_placeholder_name_is_not_taken_as_the_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("USER", "youzhi")
        monkeypatch.setattr("os.getuid", lambda: 940740146)
        monkeypatch.setattr("pwd.getpwuid", lambda uid: type("P", (), {"pw_name": "nobody"})())
        assert slurm.current_username() == "youzhi"

    def test_resolve_current_jobs_asks_squeue_for_a_real_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[list[str]] = []

        def _fake(cmd: list[str], **kw: object) -> str:
            seen.append(cmd)
            return ""

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _fake)
        monkeypatch.delenv("USER", raising=False)
        monkeypatch.delenv("LOGNAME", raising=False)
        monkeypatch.setattr(slurm, "current_username", lambda: "youzhi")
        resolve_current_jobs()
        assert seen and "youzhi" in seen[0]
        assert "" not in seen[0]


class TestMissingJobDetection:
    """SW-7: `scontrol` reports a nonexistent ARRAY TASK on stdout ("Job 1_9 not
    found") and exits 1 with an empty stderr, so a stderr-only message lost the
    detail and the permanent failure was reported as "the controller may be busy —
    try again in a moment"."""

    def _err(self, detail: str) -> SlurmCommandError:
        return SlurmCommandError(f"Command scontrol show job -d 48818945_9 failed (rc=1): {detail}")

    def test_scontrol_stdout_wording_counts_as_missing(self) -> None:
        assert slurm._is_missing_job_error(self._err("Job 48818945_9 not found")) is True

    def test_the_documented_stderr_wording_still_counts(self) -> None:
        exc = self._err("slurm_load_jobs error: Invalid job id specified")
        assert slurm._is_missing_job_error(exc) is True

    def test_a_transient_failure_is_still_transient(self) -> None:
        for detail in (
            "Socket timed out on send/recv operation",
            "Unable to contact slurm controller (connect failure)",
            "slurm_load_jobs error: Zero Bytes were transmitted or received",
        ):
            assert slurm._is_missing_job_error(self._err(detail)) is False, detail

    def test_a_missing_slurm_binary_is_not_a_missing_job(self) -> None:
        """The message says "not found" but is about squeue, not about a job."""
        exc = SlurmCommandError("Slurm binary not found: squeue. Is Slurm installed?")
        assert slurm._is_missing_job_error(exc) is False

    def test_the_failure_detail_falls_back_to_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Result:
            returncode = 1
            stdout = "Job 48818945_9 not found\n"
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        with pytest.raises(SlurmCommandError) as excinfo:
            slurm._run_slurm_cmd(["scontrol", "show", "job", "-d", "48818945_9"])
        assert "not found" in str(excinfo.value)

    def test_stderr_still_wins_when_both_channels_spoke(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Result:
            returncode = 1
            stdout = "some partial output"
            stderr = "slurm_load_jobs error: Socket timed out"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        with pytest.raises(SlurmCommandError) as excinfo:
            slurm._run_slurm_cmd(["scontrol", "show", "job", "1"])
        assert "Socket timed out" in str(excinfo.value)

    def test_a_bad_array_task_raises_JobNotFound_not_a_retry_suggestion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: object, **_k: object) -> str:
            raise SlurmCommandError(
                "Command scontrol show job -d 48818945_9 failed (rc=1): Job 48818945_9 not found"
            )

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _boom)
        with pytest.raises(JobNotFoundError):
            resolve_job_context("48818945_9")


class TestDemoFixturesAreSiteAgnostic:
    """SW-6: the shipped demo data hardcoded a real site's allocation name
    (`rcc-staff`) while every value around it was deliberately generic."""

    def test_no_real_site_account_ships_in_the_mock_job(self) -> None:
        ctx = slurm._make_mock_job_context("12345")
        assert "rcc" not in ctx.account.lower(), ctx.account
        assert ctx.account, "the demo still needs an account to display"

    def test_no_real_site_account_ships_in_the_mock_pending_job(self) -> None:
        from slurmwatch.pending import _mock_pending_job

        job = _mock_pending_job("12345")
        assert "rcc" not in job.account.lower(), job.account
        assert job.account


class TestBareArrayIdAnnouncesItsChoice:
    """SW-9: with tasks 1, 2 and 3 running, `sw <base-id>` monitors _3 every time —
    deterministic, but nothing said which task was on screen or how to ask for
    another, and --help documents only the `12345_3` spelling."""

    ARRAY = _SAMPLE_SCONTROL.replace(
        "JobId=12345 JobState=RUNNING",
        "JobId=12348 ArrayJobId=12345 ArrayTaskId=3 JobState=RUNNING",
    )

    def _resolve(
        self, monkeypatch: pytest.MonkeyPatch, job_id: str, counts: tuple[int, int] | None
    ) -> object:
        TestResolveJobContext._patch_common(monkeypatch, self.ARRAY)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        monkeypatch.setattr(slurm, "resolve_array_task_counts", lambda _a: counts)
        return resolve_job_context(job_id)

    def test_names_the_task_it_picked(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="slurmwatch"):
            ctx = self._resolve(monkeypatch, "12345", (3, 0))
        assert ctx.job_id == "12345_3"  # type: ignore[attr-defined]
        assert "12345_3" in caplog.text
        assert "3 running tasks" in caplog.text

    def test_stays_quiet_when_there_was_nothing_to_choose_between(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One running task is not an ambiguous pick, so don't nag about it."""
        with caplog.at_level("WARNING", logger="slurmwatch"):
            self._resolve(monkeypatch, "12345", (1, 0))
        assert "running tasks" not in caplog.text

    def test_stays_quiet_when_the_task_was_named_explicitly(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="slurmwatch"):
            self._resolve(monkeypatch, "12345_3", (3, 0))
        assert "running tasks" not in caplog.text


class TestUnboundedTimeLimit:
    """Round 2 left the `TimeLimit=UNLIMITED` path untested — no job on the test
    cluster had one — and it feeds the "ran X% of limit" division."""

    def test_unlimited_is_none_not_zero_seconds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for spelling in ("UNLIMITED", "Partition_Limit", "N/A"):
            record = _SAMPLE_SCONTROL.replace("TimeLimit=1-00:00:00", f"TimeLimit={spelling}")
            TestResolveJobContext._patch_common(monkeypatch, record)
            monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
            ctx = resolve_job_context("12345")
            assert ctx.time_limit_seconds is None, spelling


class TestOwnerFieldCannotBeForged:
    """SW-1's own fix added a new poisonable field, which round 5's "do not weaken
    these defences" list demands be checked: a job NAME containing a NEWLINE spills
    onto a line of its own, where a forged `UserId=root(0)` would be read as
    Slurm's — the free-text truncation rule only bounds the line the name starts on.
    """

    REAL = (
        "JobId=12345 JobName=innocent\n"
        "   UserId=youzhi(940740146) GroupId=youzhi(940740146) MCS_label=N/A\n"
    )

    def test_the_plain_record_still_parses(self) -> None:
        assert slurm._owner_from_record(self.REAL) == ("youzhi", 940740146)

    def test_a_newline_in_the_job_name_cannot_forge_the_owner(self) -> None:
        forged = (
            "JobId=12345 JobName=evil\n"
            "UserId=root(0) GroupId=root(0)\n"  # the job name's second line
            "   UserId=youzhi(940740146) GroupId=youzhi(940740146) MCS_label=N/A\n"
        )
        name, uid = slurm._owner_from_record(forged)
        assert uid == 940740146, "took the forged line's uid"
        assert name == "youzhi"

    def test_a_record_without_a_groupid_still_resolves(self) -> None:
        """Not every scontrol version prints them on one line."""
        record = "JobId=12345 JobName=x\nSubmitTime=... UserId=youzhi(940740146)\n"
        assert slurm._owner_from_record(record) == ("youzhi", 940740146)

    def test_an_absent_userid_is_not_invented(self) -> None:
        assert slurm._owner_from_record("JobId=12345 JobState=RUNNING\n") == ("", None)


class TestMemoryUnitDependsOnTheField:
    """SW-12 follow-through: `_parse_mem_to_bytes` serves two vocabularies. A bare
    `--mem`/`ReqMem`/`MinMemory*` counts in MEGABYTES; a bare sstat/sacct `MaxRSS`
    counts in KILOBYTES. One blanket default inflates whichever it isn't by 1024x.
    """

    def test_request_fields_default_to_megabytes(self) -> None:
        assert _parse_mem_to_bytes("16") == 16 * 1024**2

    def test_max_rss_defaults_to_kilobytes(self) -> None:
        assert _parse_mem_to_bytes("523508", "K") == 523508 * 1024

    def test_an_explicit_suffix_always_wins(self) -> None:
        assert _parse_mem_to_bytes("523508K", "K") == 523508 * 1024
        assert _parse_mem_to_bytes("2G", "K") == 2 * 1024**3

    def test_the_sstat_path_reads_a_bare_max_rss_as_kb(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: a site whose sstat omits the suffix must not report 1024x."""
        rows = "12345.batch|523508|00:10:00|1\n"
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: rows)
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        usage = slurm.resolve_remote_usage("12345", 1)
        assert usage.rss_bytes == 523508 * 1024


class TestSqueueRowsSurviveAMultilineJobName:
    """slurmpast's SP-1 root cause, checked in OUR parser: the job NAME is the one
    free-text field in the picker's squeue format, and a name holding a NEWLINE used
    to split its own row so the tail fragments parsed as rows of their own — a
    picker listing jobs that do not exist."""

    def _jobs(self, monkeypatch: pytest.MonkeyPatch, output: str) -> list[dict[str, object]]:
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: output)
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        return resolve_current_jobs("youzhi")

    def test_the_ordinary_case_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = (
            "12345|R|build|1|0:10|1:00:00|None|train\n12346|PD|amd|2|0:00|2:00:00|Resources|eval\n"
        )
        jobs = self._jobs(monkeypatch, out)
        assert [j["job_id"] for j in jobs] == ["12345", "12346"]
        assert jobs[0]["name"] == "train" and jobs[1]["reason"] == "Resources"

    def test_a_newline_in_the_name_does_not_invent_a_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = "12345|R|build|1|0:10|1:00:00|None|line one\nline two\n"
        jobs = self._jobs(monkeypatch, out)
        assert len(jobs) == 1, jobs
        assert jobs[0]["name"] == "line one line two", "the name must survive intact"

    def test_a_fragment_that_imitates_a_state_column_is_not_a_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The old parser only checked field 2 for R/PD, so a fragment whose second
        field happened to be a state became a job."""
        out = "12345|R|build|1|0:10|1:00:00|None|evil\nR|amd|1|0:01|1:00|None|forged\n"
        jobs = self._jobs(monkeypatch, out)
        assert [j["job_id"] for j in jobs] == ["12345"], jobs

    def test_a_crafted_full_row_in_the_name_cannot_hide_the_real_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The residual case: an imitation complete enough to start a record still
        leaves the REAL row correct and resolvable, which is what matters."""
        out = "12345|R|build|1|0:10|1:00:00|None|evil\n99999|R|amd|1|0:01|1:00|None|forged\n"
        jobs = self._jobs(monkeypatch, out)
        assert jobs[0]["job_id"] == "12345"
        assert str(jobs[0]["name"]).startswith("evil")

    def test_only_a_job_id_can_start_a_record(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fragment can carry all 7 delimiters and still not be a row. Without the
        job-id anchor, this name would have added a job called "not-an-id"."""
        out = "12345|R|build|1|0:10|1:00:00|None|evil\nnot-an-id|R|amd|1|0:01|1:00|None|forged\n"
        jobs = self._jobs(monkeypatch, out)
        assert [j["job_id"] for j in jobs] == ["12345"], jobs
        assert "not-an-id" in str(jobs[0]["name"]), "the fragment belongs to the name"

    def test_a_fragment_with_every_delimiter_is_still_not_a_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """And the delimiter count matters too: a numeric-looking fragment with only
        some of the fields is a name continuation, not a truncated job."""
        out = "12345|R|build|1|0:10|1:00:00|None|evil\n67890|R|amd\n"
        jobs = self._jobs(monkeypatch, out)
        assert [j["job_id"] for j in jobs] == ["12345"], jobs

    def test_array_and_het_ids_still_start_records(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = (
            "12345_7|R|build|1|0:10|1:00:00|None|task\n"
            "12346+0|R|build|1|0:10|1:00:00|None|component\n"
        )
        assert [j["job_id"] for j in self._jobs(monkeypatch, out)] == ["12345_7", "12346+0"]


class TestV1RootIsASiteConfigurablePrefix:
    """Round 22: every v1 fixture used the bare `slurm` root, while the test cluster
    emits the node-suffixed form:

        /sys/fs/cgroup/memory/slurm_midway2-0300/uid_940740146/job_48818838

    It appeared to work there only because the /proc/self/cgroup fallback rescued it
    — and that fallback needs slurmwatch to be running INSIDE the job's cgroup. By
    exact path the suffixed root was never tried, so watching another of your own
    jobs on the same node found nothing and degraded to sstat for no reason.
    """

    def _tree(self, base: Path, root: str, uid: int, job: str) -> Path:
        job_dir = base / "memory" / root / f"uid_{uid}" / f"job_{job}"
        job_dir.mkdir(parents=True)
        (job_dir / "cgroup.procs").write_text("")
        cpu_dir = base / "cpuacct" / root / f"uid_{uid}" / f"job_{job}"
        cpu_dir.mkdir(parents=True)
        (cpu_dir / "cgroup.procs").write_text("")
        return job_dir

    def _discover(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: str, step: str | None = None
    ) -> dict[str, Path | None]:
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", tmp_path)
        monkeypatch.setattr(slurm, "detect_cgroup_version", lambda: 1)
        # No /proc/self/cgroup rescue: this is the "on the node, not in the job's
        # cgroup" case, which is exactly where the exact path has to work.
        monkeypatch.setattr(slurm, "_read_self_cgroup", lambda: "")
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        self._tree(tmp_path, root, uid=940740146, job="48818838")
        return slurm._discover_cgroup_paths("48818838", uid=940740146, step_id=step)

    def test_a_node_suffixed_root_is_found_by_exact_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self._discover(tmp_path, monkeypatch, "slurm_midway2-0300")
        assert paths["v1_mem"] is not None, "the suffixed root was not tried"
        assert paths["v1_mem"].name == "job_48818838"
        assert "slurm_midway2-0300" in str(paths["v1_mem"])
        assert paths["v1_cpu"] is not None

    def test_the_bare_root_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = self._discover(tmp_path, monkeypatch, "slurm")
        assert paths["v1_mem"] is not None and paths["v1_mem"].name == "job_48818838"

    def test_a_step_under_a_suffixed_root_is_preferred(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", tmp_path)
        monkeypatch.setattr(slurm, "detect_cgroup_version", lambda: 1)
        monkeypatch.setattr(slurm, "_read_self_cgroup", lambda: "")
        job_dir = self._tree(tmp_path, "slurm_cn-0007", uid=940740146, job="48818838")
        step = job_dir / "step_0"
        step.mkdir()
        (step / "cgroup.procs").write_text("")
        paths = slurm._discover_cgroup_paths("48818838", uid=940740146, step_id="0")
        assert paths["v1_mem"] is not None and paths["v1_mem"].name == "step_0"

    def test_the_bare_root_wins_when_both_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The common layout stays the fast path — no directory listing needed."""
        monkeypatch.setattr(slurm, "_CGROUP_V2_BASE", tmp_path)
        monkeypatch.setattr(slurm, "detect_cgroup_version", lambda: 1)
        monkeypatch.setattr(slurm, "_read_self_cgroup", lambda: "")
        self._tree(tmp_path, "slurm", uid=940740146, job="48818838")
        self._tree(tmp_path, "slurm_cn-0007", uid=940740146, job="48818838")
        paths = slurm._discover_cgroup_paths("48818838", uid=940740146, step_id=None)
        assert paths["v1_mem"] is not None
        assert paths["v1_mem"].parent.parent.name == "slurm"


class TestCgroupDiscoveryGetsTheScontrolUid:
    """SW-1's third consequence, and round 22's point about fixtures: with
    `UserId=nobody(940740146)` the old parser handed cgroup discovery uid 99 (what
    getpwnam("nobody") returns), so it looked under `uid_99` while the job's real
    directory is `uid_940740146`. It only appeared to work because the
    /proc/self/cgroup fallback rescued it — and that needs slurmwatch to be running
    inside the job's cgroup. No fixture with a WRONGLY-resolving name existed, which
    is how 837 tests passed on a cluster where this reproduces."""

    def test_discovery_is_asked_for_the_uid_slurm_printed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = _SAMPLE_SCONTROL.replace("UserId=user(1001)", "UserId=nobody(940740146)")
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: record)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        # What getpwnam("nobody") would answer: a real uid, for the wrong user.
        monkeypatch.setattr(slurm, "_resolve_uid", lambda name: 99)
        asked: list[int | None] = []

        def _spy(
            job_id: str, uid: int | None = None, step_id: str | None = None
        ) -> dict[str, None]:
            asked.append(uid)
            return {"v2": None, "v1_mem": None, "v1_cpu": None}

        monkeypatch.setattr(slurm, "_discover_cgroup_paths", _spy)
        resolve_job_context("12345")
        assert asked == [940740146], f"discovery looked under uid_{asked}"


class TestSlurmFieldSkew:
    """SW-19: field names move between Slurm releases (23.02 renamed `Reserved` to
    `Planned`), and ONE unknown field makes sacct/sstat reject the entire query —
    every column, not just that one. slurmwatch asked for six accounting fields
    blind. All six are valid on 23.02, so this was latent, but the failure mode was
    total AND misreported: "Invalid field requested" fell through to the transient
    branch, telling the user a healthy controller "may be busy — try again"."""

    def test_a_rejected_field_is_not_called_a_busy_controller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Third trigger for that misdiagnosis after SW-7 and rapidu's RD-2, which
        is why the DIAGNOSIS is what got fixed rather than the trigger."""

        def _boom(*_a: object, **_k: object) -> str:
            raise SlurmCommandError(
                "Command sacct -n -P -X -j 12345 --format=State,End failed (rc=1): "
                'sacct: error: Invalid field requested: "Reserved"'
            )

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _boom)
        with pytest.raises(SlurmCommandError) as exc:
            resolve_job_context("12345")
        text = str(exc.value)
        assert "try again" not in text.lower(), text
        assert "field" in text and "slurmwatch bug" in text

    def test_no_slurm_at_all_is_not_called_a_busy_controller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FOURTH trigger of the same misdiagnosis, and the most cluster-agnostic one.

        On a machine with no Slurm client tools — a PBS/Flux/Kubernetes site, a laptop,
        a login node whose module isn't loaded — the exec fails with FileNotFoundError,
        which became "Slurm binary not found: scontrol" and then fell into the generic
        "Couldn't reach the Slurm controller ... it may be busy, try again in a
        moment". Measured by running the real binary with an empty PATH. There is no
        controller in that picture, and retrying is the one thing that cannot help.
        """

        def _boom(*_a: object, **_k: object) -> str:
            raise SlurmCommandError("Slurm binary not found: scontrol. Is Slurm installed?")

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _boom)
        with pytest.raises(SlurmCommandError) as exc:
            resolve_job_context("12345")
        text = str(exc.value)
        assert "try again" not in text.lower(), text
        assert "will not help" in text.lower(), text
        # Actionable in BOTH directions: Slurm here but unloaded, or not Slurm at all.
        assert "PATH" in text and "module load" in text, text
        assert "PBS" in text or "Flux" in text, text
        assert exc.value.kind == "unavailable"

    def test_the_kind_separates_the_three_causes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A consumer acts on the KIND, so it must not be inferred from the prose.

        Three causes, three actions: install/module-load, report a slurmwatch bug, or
        wait. The classification lives on the exception because matching wording is
        the kind of coupling that breaks the moment the wording improves.
        """
        cases = [
            ("Slurm binary not found: scontrol. Is Slurm installed?", "unavailable"),
            ('sacct: error: Invalid field requested: "Reserved"', "unsupported"),
            ("slurm_load_jobs error: Unable to contact slurm controller", "transient"),
        ]
        for message, expected in cases:

            def _boom(*_a: object, _m: str = message, **_k: object) -> str:
                raise SlurmCommandError(_m)

            monkeypatch.setattr(slurm, "_run_slurm_cmd", _boom)
            with pytest.raises(SlurmCommandError) as exc:
                resolve_job_context("12345")
            assert exc.value.kind == expected, (message, exc.value.kind)

    def test_a_default_slurm_error_is_transient(self) -> None:
        """The default has to be the recoverable one: a controller hiccup must not be
        reported as "this cluster cannot run slurmwatch"."""
        assert SlurmCommandError("boom").kind == "transient"

    def test_the_classifier_separates_our_bug_from_a_missing_job(self) -> None:
        bad_field = SlurmCommandError('sacct: error: Invalid field requested: "Reserved"')
        missing = SlurmCommandError("slurm_load_jobs error: Invalid job id specified")
        assert slurm._is_malformed_query_error(bad_field) is True
        assert slurm._is_malformed_query_error(missing) is False
        # ...and a malformed query must NOT be reported as a nonexistent job either.
        assert slurm._is_missing_job_error(bad_field) is False

    def test_unsupported_fields_are_dropped_not_guessed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slurm._HELPFORMAT_CACHE.clear()
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "JobID MaxRSS NTasks\n")
        kept = slurm._drop_unsupported_fields("sstat", ["JobID", "MaxRSS", "AveCPU", "NTasks"])
        assert kept == ["JobID", "MaxRSS", "NTasks"]

    def test_an_unreadable_helpformat_changes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guessing would be worse than trying: with no answer, ask for everything."""
        slurm._HELPFORMAT_CACHE.clear()

        def _boom(*_a: object, **_k: object) -> str:
            raise SlurmCommandError("no such option: --helpformat")

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _boom)
        fields = ["JobID", "MaxRSS"]
        assert slurm._drop_unsupported_fields("sstat", fields) == fields

    def test_sstat_retries_with_the_fields_this_slurm_has(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Losing one column beats losing the query: MaxRSS still arrives on a site
        whose sstat has no AveCPU."""
        slurm._HELPFORMAT_CACHE.clear()
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        calls: list[list[str]] = []

        def _fake(cmd: list[str], **_k: object) -> str:
            calls.append(cmd)
            if "--helpformat" in cmd:
                return "JobID MaxRSS NTasks\n"
            fmt = next(a for a in cmd if a.startswith("--format="))
            if "AveCPU" in fmt:
                raise SlurmCommandError('sstat: error: Invalid field requested: "AveCPU"')
            return "12345.batch|523508K|1\n"

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _fake)
        usage = slurm.resolve_remote_usage("12345", 1)
        assert usage.sampled is True
        assert usage.rss_bytes == 523508 * 1024, "MaxRSS must survive the retry"
        assert usage.cpu_seconds == 0.0, "the dropped column contributes nothing"
        assert any("--helpformat" in c for c in calls), "never asked what it supports"

    def test_the_retry_reads_columns_by_name_not_position(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The trap the retry could have introduced: with AveCPU gone the columns
        shift, so a positional parser would read NTasks as CPU-time — a confidently
        wrong number, worse than the missing data."""
        slurm._HELPFORMAT_CACHE.clear()
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)

        def _fake(cmd: list[str], **_k: object) -> str:
            if "--helpformat" in cmd:
                return "JobID MaxRSS NTasks\n"
            fmt = next(a for a in cmd if a.startswith("--format="))
            if "AveCPU" in fmt:
                raise SlurmCommandError('sstat: error: Invalid field requested: "AveCPU"')
            return "12345.batch|1024K|8\n"  # NTasks=8 must NOT become 8 seconds of CPU

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _fake)
        usage = slurm.resolve_remote_usage("12345", 1)
        assert usage.cpu_seconds == 0.0, "NTasks was read as AveCPU"
        assert usage.rss_bytes == 1024 * 1024 * 8, "NTasks scaling still applies"

    def test_sacct_state_survives_a_rejected_end_column(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slurm._HELPFORMAT_CACHE.clear()
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)

        def _fake(cmd: list[str], **_k: object) -> str:
            if "--helpformat" in cmd:
                return "JobID State\n"
            fmt = next(a for a in cmd if a.startswith("--format="))
            if "End" in fmt:
                raise SlurmCommandError('sacct: error: Invalid field requested: "End"')
            return "COMPLETED\n"

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _fake)
        assert slurm._sacct_final_state("12345") == ("COMPLETED", "")


class TestRequeuedJobIsNotDeclaredFinished:
    """Found by auditing the Slurm state vocabulary, not from a report: which states
    a site can produce depends on its requeue/preemption policy, so a missing one is
    invisible until you run somewhere that uses it.

    `is_job_active` answers False for any state in neither the active nor the
    requeued set, and False means "gone" — the dashboard tears down and prints "job
    ended". For a job that is queued and will run again under the same JobId, that
    is a confidently wrong answer.
    """

    def _active(self, monkeypatch: pytest.MonkeyPatch, state: str) -> bool | None:
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: f"{state}\n")
        return slurm.is_job_active("12345")

    @pytest.mark.parametrize(
        "state",
        ["PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD", "SPECIAL_EXIT"],
    )
    def test_a_queued_job_is_alive(self, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
        assert self._active(monkeypatch, state) is True, state

    @pytest.mark.parametrize(
        "state", ["RUNNING", "COMPLETING", "CONFIGURING", "RESIZING", "SIGNALING", "SUSPENDED"]
    )
    def test_an_allocated_job_is_alive(self, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
        assert self._active(monkeypatch, state) is True, state

    @pytest.mark.parametrize(
        "state", ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED"]
    )
    def test_a_terminal_job_is_not(self, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
        assert self._active(monkeypatch, state) is False, state

    def test_one_live_array_task_keeps_the_whole_array_alive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(
            slurm, "_run_slurm_cmd", lambda *a, **k: "COMPLETED\nSPECIAL_EXIT\nCOMPLETED\n"
        )
        assert slurm.is_job_active("12345") is True

    def test_the_sacct_finished_probe_agrees(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same vocabulary gates `_sacct_final_state`: a requeued-and-held row
        must stop it reporting the job terminal."""
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "SPECIAL_EXIT|Unknown\n")
        assert slurm._sacct_final_state("12345") is None


class TestOwnerCannotBeForgedFromEitherSide:
    """SW-1 feedback (round 32): the reporter defeated the last-wins rule with a
    second vector. `scontrol show job` prints `JobName` BEFORE `UserId=` and
    `Comment` AFTER it, and both accept a newline — so first-wins loses to a forged
    job name and last-wins loses to a forged comment. Neither positional rule is
    safe against both, which is why the uid now comes from squeue's single-field
    output, where a newline cannot forge a row.
    """

    # Their vector 1: the forgery lands BEFORE the real line (job name).
    JOBNAME_VECTOR = (
        "JobId=48819478 JobName=evil\n"
        "   UserId=root(0) GroupId=root(0) MCS_label=N/A\n"
        "   UserId=youzhi(940740146) GroupId=youzhi(940740146) MCS_label=N/A\n"
    )
    # Their vector 2: the forgery lands AFTER it (comment), satisfying both halves
    # of the old rule — numeric uid AND beside a GroupId.
    COMMENT_VECTOR = (
        "JobId=48819479 JobName=benign\n"
        "   UserId=youzhi(940740146) GroupId=youzhi(940740146) MCS_label=N/A\n"
        "   Comment=benign\n"
        "   UserId=victim(12345) GroupId=victim(12345)\n"
    )

    ROW = "940740146|youzhi|RUNNING|build|midway3-0200|2|1\n"

    def test_squeue_is_the_source_and_ignores_both_vectors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All-machine fields in one query: no value in it can contain the delimiter
        or a newline, so nothing here can be forged."""
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: self.ROW)
        facts = slurm._authoritative_job_facts("48819479")
        assert facts["uid"] == "940740146" and facts["username"] == "youzhi"
        assert facts["state"] == "RUNNING" and facts["nodelist"] == "midway3-0200"
        assert facts["cpus"] == "2" and facts["nodes"] == "1"

    def test_the_query_asks_for_no_free_text_field(self) -> None:
        """%j (job name) or %k (comment) in this format would reintroduce the very
        injection it exists to avoid: a newline there forges a whole row."""
        assert "%j" not in slurm._SQUEUE_FACTS_FORMAT
        assert "%k" not in slurm._SQUEUE_FACTS_FORMAT

    def test_an_array_base_id_takes_the_first_usable_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: self.ROW + self.ROW)
        assert slurm._authoritative_job_facts("12345")["uid"] == "940740146"

    def test_squeue_silence_is_not_an_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "")
        assert slurm._authoritative_job_facts("12345") == {}

    def test_a_garbled_row_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "not|a|uid\n")
        assert slurm._authoritative_job_facts("12345") == {}

    @pytest.mark.parametrize("vector", ["JOBNAME_VECTOR", "COMMENT_VECTOR"])
    def test_resolve_prefers_squeue_over_the_poisoned_record(
        self, monkeypatch: pytest.MonkeyPatch, vector: str
    ) -> None:
        """The job belongs to SOMEONE ELSE (4242) and we are a third party (99999),
        so the record fallback cannot rescue this via its own-uid branch — only
        actually asking squeue produces the right owner."""
        record = getattr(self, vector).replace("youzhi(940740146)", "alice(4242)")
        TestResolveJobContext._patch_common(monkeypatch, record)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        monkeypatch.setattr("os.getuid", lambda: 99999)
        monkeypatch.setattr(
            slurm,
            "_authoritative_job_facts",
            lambda jid: {
                "uid": "4242",
                "username": "alice",
                "state": "RUNNING",
                "partition": "gpu",
                "nodelist": "cn-001",
                "cpus": "16",
                "nodes": "1",
            },
        )
        ctx = resolve_job_context("48819479")
        assert ctx.uid == 4242, "the poisoned record won over squeue"
        assert ctx.username == "alice"
        # ...and every other forged field lost too, not only the owner: the record
        # claims attacker-node / 999 CPUs / 7 nodes on its injected line.
        assert ctx.partition == "gpu"
        assert ctx.cpus_allocated == 16
        assert ctx.nodelist_resolved == ["cn-001"]

    def test_the_fallback_refuses_to_guess_when_the_record_disagrees(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With squeue unavailable and two candidate owners, neither position is
        trustworthy — so the uid is left unresolved rather than confidently wrong.
        `_job_owner_differs` then compares NAMES, giving the honest read-only view."""
        monkeypatch.setattr("os.getuid", lambda: 99999)  # neither candidate is us
        name, uid = slurm._owner_from_record(self.COMMENT_VECTOR)
        assert uid is None, "picked a side"
        assert name in ("youzhi", "victim")

    def test_our_own_uid_wins_when_it_is_among_the_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The common case: you are watching your own job. An attacker gains nothing
        by forging the uid of the person reading, so a match is decisive."""
        monkeypatch.setattr("os.getuid", lambda: 940740146)
        for vector in (self.JOBNAME_VECTOR, self.COMMENT_VECTOR):
            assert slurm._owner_from_record(vector) == ("youzhi", 940740146)

    def test_an_unpoisoned_record_still_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("os.getuid", lambda: 12)
        record = "JobId=1 JobName=x\n   UserId=youzhi(940740146) GroupId=youzhi(940740146)\n"
        assert slurm._owner_from_record(record) == ("youzhi", 940740146)


class TestMemPerCpuJobsHaveALimit:
    """A site with DefMemPerCPU set, or a user passing --mem-per-cpu, gets
    MinMemoryCPU on the scontrol record and no MinMemoryNode at all — Slurm prints
    one or the other. The running-job path read only the per-node field, so when the
    TRES mem= was unreadable the limit fell to 0, which the collector treats as "no
    cap" and replaces with the node's whole RAM."""

    RECORD = (
        "JobId=777 JobState=RUNNING Partition=cpu UserId=user(1001)\n"
        "JobName=fit\n"
        "NodeList=cn-001 NumCPUs=6 NumNodes=1\n"
        "TRES=cpu=6,mem=6Zg\n"  # a unit this Slurm spells in a way we cannot read
        "RunTime=00:10:00 TimeLimit=01:00:00\n"
        "SubmitTime=2024-01-15T10:29:00 MinMemoryCPU=4G StartTime=2024-01-15T10:30:00\n"
    )

    def test_mem_per_cpu_is_multiplied_by_the_cpu_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        TestResolveJobContext._patch_common(monkeypatch, self.RECORD)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        ctx = resolve_job_context("777")
        assert ctx.mem_limit_bytes == 4 * 1024**3 * 6, "4G per cpu x 6 cpus"

    def test_it_is_per_node_not_job_wide(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """min_memory_node feeds a PER-NODE limit, so the multiplier is this node's
        cpu count — 24 cpus over 4 nodes is 6 per node, not 24."""
        record = self.RECORD.replace(
            "NodeList=cn-001 NumCPUs=6 NumNodes=1", "NodeList=cn-[001-004] NumCPUs=24 NumNodes=4"
        )
        TestResolveJobContext._patch_common(monkeypatch, record)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        ctx = resolve_job_context("777")
        assert ctx.mem_limit_bytes == 4 * 1024**3 * 6

    def test_min_memory_node_still_wins_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record = self.RECORD.replace("MinMemoryCPU=4G", "MinMemoryNode=10G")
        TestResolveJobContext._patch_common(monkeypatch, record)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        ctx = resolve_job_context("777")
        assert ctx.mem_limit_bytes == 10 * 1024**3

    def test_the_per_node_field_takes_precedence_over_the_per_cpu_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Slurm prints one or the other, so this pins the guard rather than an
        observed record: if both ever appear, the per-node figure already IS what
        min_memory_node means and must not be overwritten by a product."""
        record = self.RECORD.replace("MinMemoryCPU=4G", "MinMemoryNode=10G MinMemoryCPU=4G")
        TestResolveJobContext._patch_common(monkeypatch, record)
        monkeypatch.setattr("socket.gethostname", lambda: "cn-001")
        ctx = resolve_job_context("777")
        assert ctx.mem_limit_bytes == 10 * 1024**3, "not 4G x 6"


class TestAcctGatherDisabledIsNotNotYet:
    """`JobAcctGatherType=jobacct_gather/none` is Slurm's DEFAULT when a site does
    not set it, and with it sstat reports nothing for a running job — ever. Off-node
    slurmwatch said "usage not yet sampled by Slurm (samples ~every 30s) — try again
    shortly", which sends a reader on that cluster round a retry loop forever."""

    def setup_method(self) -> None:
        slurm._ACCT_GATHER_CACHE = None
        slurm._ACCT_GATHER_WARNED = False

    teardown_method = setup_method

    CONFIG = (
        "AccountingStorageType   = accounting_storage/slurmdbd\n"
        "JobAcctGatherFrequency  = 30\n"
        "JobAcctGatherType       = jobacct_gather/{}\n"
    )

    def _patch(self, monkeypatch: pytest.MonkeyPatch, plugin: str) -> None:
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: self.CONFIG.format(plugin))

    def test_detects_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, "none")
        assert slurm.acct_gather_disabled() is True

    def test_a_gathering_cluster_is_not_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, "linux")
        assert slurm.acct_gather_disabled() is False

    def test_a_bare_none_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sites write the plugin path, but the bare type is also accepted."""
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "JobAcctGatherType = none\n")
        assert slurm.acct_gather_disabled() is True

    def test_unreadable_config_claims_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asserting a site's accounting is off on the strength of a failed
        subprocess would be worse than staying quiet."""

        def _boom(*a: object, **k: object) -> str:
            raise SlurmCommandError("no scontrol here")

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _boom)
        assert slurm.acct_gather_disabled() is False

    def test_the_config_is_read_once_per_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[object] = []

        def _count(*a: object, **k: object) -> str:
            calls.append(a)
            return self.CONFIG.format("none")

        monkeypatch.setattr(slurm, "_run_slurm_cmd", _count)
        for _ in range(5):
            slurm.acct_gather_disabled()
        assert len(calls) == 1

    def test_machine_paths_get_the_reason_on_stderr_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """--json/--csv/--log show only numbers: a consumer logging zeros for an hour
        has no other way to learn this Slurm gathers nothing."""
        monkeypatch.setattr(slurm, "_is_mock", lambda: False)
        monkeypatch.setattr(slurm, "acct_gather_disabled", lambda: True)
        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: "")
        with caplog.at_level("WARNING"):
            slurm.resolve_remote_usage("1")
            slurm.resolve_remote_usage("1")
        hits = [r for r in caplog.records if "JobAcctGatherType=none" in r.getMessage()]
        assert len(hits) == 1, "said once, not once per sample"
