from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from slurmwatch import remote
from slurmwatch.model import (
    CpuMetrics,
    GpuMetrics,
    MemoryMetrics,
    TelemetrySnapshot,
)


def _snapshot(host: str = "cn2", node_index: int = 1, node_count: int = 2) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        timestamp=time.time(),
        job_id="123",
        step_id="0",
        hostname=host,
        elapsed_seconds=42,
        cpu=CpuMetrics(cores_allocated=4, usage_ns=10**9, usage_percent=99.0, effective_cores=3.9),
        memory=MemoryMetrics(
            current_bytes=3 * 1024**3,
            limit_bytes=8 * 1024**3,
            peak_bytes=4 * 1024**3,
            usage_percent=37.0,
            oom_guard_warning=False,
            oom_guard_critical=False,
            working_set_bytes=3 * 1024**3,
            cache_bytes=0,
        ),
        gpus=[
            GpuMetrics(
                index=0,
                uuid="G",
                name="A100",
                utilization_percent=90.0,
                memory_used_bytes=1,
                memory_total_bytes=2,
                memory_utilization_percent=50.0,
                power_watts=100.0,
                temperature_celsius=40.0,
                throttling=False,
            )
        ],
        node_count=node_count,
        node_index=node_index,
        gpu_count_requested=1,
        gpu_active_count=1,
    )


class TestSnapshotSerialization:
    def test_from_json_round_trip(self) -> None:
        s = _snapshot()
        assert TelemetrySnapshot.from_json(s.to_json()) == s

    def test_to_json_refuses_a_non_finite_it_cannot_sanitize(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # allow_nan=False is the second layer behind _json_safe, and it is the one that
        # holds the RFC-8259 contract: _json_safe only walks float/dict/list, so a
        # non-finite reaching json.dumps inside a tuple/set (or as a numpy scalar) would
        # be emitted as a bare NaN, which jq and every strict parser reject — silently
        # corrupting a whole --log file. Neutralise the sanitizer to pin the flag itself:
        # a mutation sweep showed dropping allow_nan=False left the suite green.
        from slurmwatch import model as model_mod

        monkeypatch.setattr(model_mod, "_json_safe", lambda o: o)
        s = _snapshot()
        s.cpu.usage_percent = float("nan")
        with pytest.raises(ValueError):
            s.to_json()

    def test_non_finite_round_trips_as_zero_not_a_dropped_frame(self) -> None:
        # to_json maps a non-finite to `null` to stay valid JSON. Feeding that back used
        # to raise TypeError, and parse_snapshot_line swallows any exception as
        # "unparseable" — so the node switcher showed that node as producing NO data at
        # all, with no diagnostic. Losing one metric beats losing the frame.
        s = _snapshot()
        s.cpu.usage_percent = float("nan")
        s.memory.usage_percent = float("inf")
        back = TelemetrySnapshot.from_json(s.to_json())
        assert back.cpu.usage_percent == 0.0
        assert back.memory.usage_percent == 0.0
        assert back.cpu.cores_allocated == s.cpu.cores_allocated  # the rest survives
        assert remote.parse_snapshot_line(s.to_json().encode()) is not None

    def test_csv_neutralizes_a_formula_style_job_name(self) -> None:
        # csv quoting does NOT stop a spreadsheet evaluating a cell that begins = + - @;
        # a job name is arbitrary user text (`sbatch -J '=cmd|"/bin/sh"!A1'`).
        s = _snapshot()
        header = TelemetrySnapshot.csv_header(0)
        name_col = header.index("job_name")
        for hostile in ('=cmd|"/bin/sh"!A1', "+1", "-2", "@SUM(A1)"):
            s.job_name = hostile
            assert s.to_csv_row(0)[name_col].startswith("'")
        s.job_name = "sweep-run17"  # an ordinary name is untouched
        assert s.to_csv_row(0)[name_col] == "sweep-run17"

    def test_to_json_sanitizes_non_finite(self) -> None:
        # allow_nan=False + _json_safe: a stray non-finite metric emits spec-compliant
        # JSON (null), not "NaN"/"Infinity" (which jq rejects), and never crashes.
        s = _snapshot()
        s.cpu.usage_percent = float("nan")
        s.memory.usage_percent = float("inf")
        text = s.to_json()
        assert "NaN" not in text and "Infinity" not in text
        d = json.loads(text)  # parses cleanly under a strict parser
        assert d["cpu"]["usage_percent"] is None
        assert d["memory"]["usage_percent"] is None

    def test_remote_flag_round_trips(self) -> None:
        # #34/#35: the remote tag must survive JSON (the node switcher parses a
        # streamed node's JSON back into a snapshot), and default to False when a
        # snapshot from an older version omits it.
        s = _snapshot()
        s.remote = True
        assert TelemetrySnapshot.from_json(s.to_json()).remote is True
        assert _snapshot().remote is False

    def test_from_dict_ignores_unknown_keys(self) -> None:
        # A small version skew between nodes (an extra field) must not crash.
        d = {
            "timestamp": 1.0,
            "job_id": "1",
            "step_id": None,
            "hostname": "cn1",
            "elapsed_seconds": 1,
            "cpu": {"cores_allocated": 1, "usage_ns": 0, "usage_percent": 0.0, "future_field": 9},
            "memory": {
                "current_bytes": 0,
                "limit_bytes": 0,
                "peak_bytes": 0,
                "usage_percent": 0.0,
                "oom_guard_warning": False,
                "oom_guard_critical": False,
            },
            "gpus": [],
            "node_count": 3,
            "node_index": 2,
            "brand_new_top_level_key": True,
        }
        snap = TelemetrySnapshot.from_dict(d)
        assert snap.node_count == 3 and snap.node_index == 2
        assert snap.cpu.cores_allocated == 1  # parsed despite the unknown cpu field


class TestBuildStreamCommand:
    def test_streams_the_node_via_srun_overlap(self) -> None:
        cmd = remote.build_stream_command("456", "cn007", 1.0, python="/venv/bin/python")
        assert cmd[0] == "srun"
        assert "--jobid=456" in cmd and "--overlap" in cmd
        # Bounded + memory-shared so switching to a node whose GPU is held by the
        # job's own step can't hang the stream (mirrors the login-node hop).
        assert any(a.startswith("--immediate=") for a in cmd)
        assert "--mem=0" in cmd
        assert "--gres=none" not in cmd  # gpu=True (default) -> request the GPU
        # Critical: srun must NOT connect the terminal's stdin to the remote task,
        # or it swallows the user's keystrokes at the live dashboard.
        assert "--input=none" in cmd
        assert cmd[cmd.index("-w") + 1] == "cn007"
        # Runs the same install's headless logger streaming JSONL to stdout.
        assert cmd[-6:-1] == ["456", "--log", "/dev/stdout", "--json", "--interval"]
        assert cmd[-1] == "1"  # interval formatted
        assert cmd[cmd.index("-m") + 1] == "slurmwatch"

    def test_gpu_false_drops_the_gres_request(self) -> None:
        # When the node's GPU is held by the job's own step, the stream must run
        # without requesting a GPU so it still launches (CPU/mem live).
        cmd = remote.build_stream_command("456", "cn007", 1.0, gpu=False)
        assert "--gres=none" in cmd and "--mem=0" in cmd


class TestParseSnapshotLine:
    def test_valid_line(self) -> None:
        snap = remote.parse_snapshot_line(_snapshot(host="cn9", node_index=1).to_json().encode())
        assert snap is not None and snap.hostname == "cn9" and snap.node_index == 1

    def test_garbage_and_empty_are_none(self) -> None:
        assert remote.parse_snapshot_line(b"not json at all\n") is None
        assert remote.parse_snapshot_line(b"   \n") is None
        assert remote.parse_snapshot_line(b"") is None


class _FakeProc:
    """Stands in for a subprocess in open_stream tests; the GPU probe awaits wait()."""

    def __init__(self, rc: int = 0) -> None:
        self._rc = rc
        self.returncode: int | None = None
        self.killed = False

    async def wait(self) -> int:
        self.returncode = self._rc
        return self._rc

    def kill(self) -> None:
        self.killed = True


class _HangingProc:
    """A probe/stream child whose wait() never returns until cancelled — models a
    wedged slurmctld ignoring --immediate."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False

    async def wait(self) -> int:
        await asyncio.Event().wait()  # blocks forever
        return 0  # pragma: no cover

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class TestOpenStream:
    @pytest.mark.asyncio
    async def test_returns_the_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # open_stream now runs a GPU probe (`true`) first, then the stream. The
        # probe returns rc 0 (GPU reachable); the stream returns the real process.
        sentinel = object()

        async def fake_exec(*a: Any, **_k: Any) -> Any:
            return _FakeProc(0) if a and a[-1] == "true" else sentinel

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        assert await remote.open_stream("123", "cn9", 1.0) is sentinel

    @pytest.mark.asyncio
    async def test_detaches_stdin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # stdin must be /dev/null so srun can't read (and steal) the terminal's keys.
        captured: dict[str, Any] = {}

        async def fake_exec(*a: Any, **k: Any) -> Any:
            if a and a[-1] == "true":
                return _FakeProc(0)  # GPU probe
            captured.update(k)
            return object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        await remote.open_stream("123", "cn9", 1.0)
        assert captured["stdin"] == asyncio.subprocess.DEVNULL

    @pytest.mark.asyncio
    async def test_gpu_held_streams_over_ssh_to_get_real_numbers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Probe fails (the job's own step holds the GPUs) -> stream over ssh.

        A ``--gres=none`` step is DENIED /dev/nvidiaN, so it can only ever report
        "GPU unreadable" — useless on a multi-node training job, where an inner
        srun holding every GPU is the normal shape. An adopted ssh session keeps
        the job's cgroups for CPU/memory but sits outside the step's device cgroup,
        so NVML reads real utilization/VRAM/power.
        """
        stream_cmd: list[str] = []

        async def fake_exec(*a: Any, **_k: Any) -> Any:
            if a and a[-1] == "true":
                return _FakeProc(1)  # GPU not reachable by a step
            stream_cmd.extend(a)
            return object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        monkeypatch.delenv("SLURMWATCH_NO_SSH", raising=False)
        assert await remote.open_stream("123", "cn9", 1.0) is not None
        assert stream_cmd[0] == "ssh"
        assert "cn9" in stream_cmd
        assert "--gres=none" not in stream_cmd
        # No remote tty: it would inject control chars into the JSONL stream.
        assert "-t" not in stream_cmd
        remote_cmd = stream_cmd[-1]
        assert "--log /dev/stdout" in remote_cmd and "--json" in remote_cmd

    @pytest.mark.asyncio
    async def test_falls_back_to_a_gres_none_step_when_ssh_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ssh (absent, or the user opted out) -> the blind step, not nothing.

        Live CPU/memory for the other node still beats a stuck switch.
        """
        stream_cmd: list[str] = []

        async def fake_exec(*a: Any, **_k: Any) -> Any:
            if a and a[-1] == "true":
                return _FakeProc(1)
            stream_cmd.extend(a)
            return object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setenv("SLURMWATCH_NO_SSH", "1")
        assert await remote.open_stream("123", "cn9", 1.0) is not None
        assert "--gres=none" in stream_cmd
        assert stream_cmd[0] != "ssh"

    @pytest.mark.asyncio
    async def test_reachable_gpu_still_uses_the_slurm_native_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ssh is for the case a step CAN'T get the GPU — not a replacement.

        When the probe succeeds the step reads the GPUs itself, which costs no ssh
        login (each one leaks threads into the job's .extern stepd).
        """
        stream_cmd: list[str] = []

        async def fake_exec(*a: Any, **_k: Any) -> Any:
            if a and a[-1] == "true":
                return _FakeProc(0)  # GPU reachable by a step
            stream_cmd.extend(a)
            return object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        assert await remote.open_stream("123", "cn9", 1.0) is not None
        assert stream_cmd[0] != "ssh"
        assert "--gres=none" not in stream_cmd

    @pytest.mark.asyncio
    async def test_missing_srun_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(*_a: Any, **_k: Any) -> Any:
            raise FileNotFoundError("srun not found")

        monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
        assert await remote.open_stream("123", "cn9", 1.0) is None

    def test_child_env_strips_slurm_and_disables_hop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURM_STEP_ID", "7")
        monkeypatch.setenv("SLURM_PROCID", "3")
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        monkeypatch.delenv("SLURM_CONF", raising=False)
        env = remote._child_env()
        assert not any(k.startswith("SLURM_") for k in env)  # step context cleared
        assert env["SLURMWATCH_NO_HOP"] == "1"
        assert "SLURMWATCH_MOCK" not in env

    def test_child_env_keeps_slurm_conf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # #51: SLURM_CONF must survive (unlike the rest of SLURM_*) so the nested
        # srun stream can find slurm.conf / reach slurmctld on clusters that export
        # it — matching the login-node hop, which keeps it for the same reason.
        monkeypatch.setenv("SLURM_STEP_ID", "7")
        monkeypatch.setenv("SLURM_CONF", "/etc/slurm/custom.conf")
        env = remote._child_env()
        assert env["SLURM_CONF"] == "/etc/slurm/custom.conf"
        assert "SLURM_STEP_ID" not in env  # the step context is still cleared


class TestStreamSubprocessCleanup:
    """N1: the probe/stream srun must be killed on cancellation (a 25s wait_for
    firing / the user quitting mid-connect), never left running as an orphan."""

    @pytest.mark.asyncio
    async def test_probe_killed_on_cancel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = _HangingProc()

        async def fake_exec(*_a: Any, **_k: Any) -> Any:
            return proc

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        task = asyncio.create_task(remote._stream_can_get_gpu("123", "cn9"))
        await asyncio.sleep(0.05)  # let it spawn and reach proc.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert proc.killed  # reaped in the finally, not orphaned

    @pytest.mark.asyncio
    async def test_probe_killed_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A wedged controller that ignores --immediate: the Python timeout fires and
        # the probe returns False, having killed its child.
        proc = _HangingProc()

        async def fake_exec(*_a: Any, **_k: Any) -> Any:
            return proc

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr(remote, "_GPU_PROBE_SECONDS", -2.95)  # -> timeout ~0.05s
        assert await remote._stream_can_get_gpu("123", "cn9") is False
        assert proc.killed

    @pytest.mark.asyncio
    async def test_open_stream_kills_probe_on_cancel_before_launch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Cancelling the whole open_stream while its GPU probe is still awaiting must
        # reap the probe child (the probe's finally runs), so no orphan is left even
        # when the cancel targets open_stream rather than the probe directly.
        probe = _HangingProc()

        async def fake_exec(*_a: Any, **_k: Any) -> Any:
            return probe

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        task = asyncio.create_task(remote.open_stream("123", "cn9", 1.0))
        await asyncio.sleep(0.05)  # let the probe spawn and reach wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert probe.killed


class TestASiteThatRefusesLoginToComputeSsh:
    """`shutil.which("ssh")` proves the CLIENT exists, not that the site permits it.

    Measured on a Booth cluster (Slurm 25.11): `ssh mcn57` from the login node
    answers `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password).`
    and exits 255. The ssh rung was still PREFERRED there over a `--gres=none` step
    that works, and "permission denied" is what a REFUSED SLURM STEP also says — so
    `stream_error_is_permanent` retired the node for the rest of the session and the
    banner blamed Slurm for ssh's answer. Two bugs from one missing fact: which rung
    produced the text.
    """

    @staticmethod
    async def _open(monkeypatch: pytest.MonkeyPatch, node: str) -> list[str]:
        """open_stream with a GPU a step can't get; returns the argv it launched."""
        cmd: list[str] = []

        async def fake_exec(*a: Any, **_k: Any) -> Any:
            if a and a[-1] == "true":
                return _FakeProc(1)  # the job's own step holds the GPUs
            cmd.extend(a)
            return object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        monkeypatch.delenv("SLURMWATCH_NO_SSH", raising=False)
        assert await remote.open_stream("123", node, 1.0) is not None
        return cmd

    @pytest.mark.asyncio
    async def test_the_transport_actually_used_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert remote.stream_transport("cn9") == ""
        assert (await self._open(monkeypatch, "cn9"))[0] == "ssh"
        assert remote.stream_transport("cn9") == "ssh"

    @pytest.mark.asyncio
    async def test_a_reachable_gpu_records_the_step_rung(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_exec(*a: Any, **_k: Any) -> Any:
            return _FakeProc(0) if a and a[-1] == "true" else object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        assert await remote.open_stream("123", "cn9", 1.0) is not None
        assert remote.stream_transport("cn9") == "step"

    @pytest.mark.asyncio
    async def test_the_next_launch_takes_the_step_after_ssh_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: the step rung is still there, so USE it."""
        assert (await self._open(monkeypatch, "cn9"))[0] == "ssh"
        assert remote.retry_other_stream_transport("cn9") is True
        second = await self._open(monkeypatch, "cn9")
        assert second[0] != "ssh"
        assert "--gres=none" in second
        assert remote.stream_transport("cn9") == "step"

    @pytest.mark.asyncio
    async def test_giving_up_is_still_right_once_the_step_has_failed_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not a retry loop: each rung gets one turn, then the failure is permanent."""
        await self._open(monkeypatch, "cn9")
        assert remote.retry_other_stream_transport("cn9") is True  # ssh -> step
        await self._open(monkeypatch, "cn9")  # now the step
        assert remote.retry_other_stream_transport("cn9") is False

    def test_a_node_never_streamed_has_no_other_rung_to_try(self) -> None:
        """No transport recorded means the launch itself failed — nothing to switch to."""
        assert remote.retry_other_stream_transport("cn-never") is False

    @pytest.mark.asyncio
    async def test_one_node_being_ssh_less_does_not_condemn_another(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sites exist where only some nodes take an adopted login; keep it per node."""
        await self._open(monkeypatch, "cn9")
        assert remote.retry_other_stream_transport("cn9") is True
        assert (await self._open(monkeypatch, "cn10"))[0] == "ssh"

    def test_the_banner_names_ssh_not_slurm(self) -> None:
        text = "youzhi@cn9: Permission denied (publickey,gssapi-keyex,password)."
        # Without the transport this is indistinguishable from a refused step...
        assert "Slurm refused a step" in remote.summarise_stream_error(text, "cn9")
        # ...and with it, the reader is told what actually happened.
        out = remote.summarise_stream_error(text, "cn9", transport="ssh")
        assert "ssh" in out and "cn9" in out
        assert "Slurm refused" not in out

    def test_the_ssh_summary_is_ascii_clean_under_ascii(self) -> None:
        out = remote.summarise_stream_error(
            "cn9: Permission denied (publickey).", "cn9", ascii_mode=True, transport="ssh"
        )
        assert out.isascii(), out

    def test_a_step_failure_is_still_read_as_a_step_failure(self) -> None:
        """The transport must not rewrite a genuine Slurm refusal."""
        out = remote.summarise_stream_error(
            "srun: error: Access/permission denied for job 42", "cn9", transport="step"
        )
        assert "Slurm refused a step" in out


class TestWhyAStreamDiedIsNotDiscarded:
    """The step's stderr was sent to DEVNULL, so the dashboard could only guess.

    Measured on a second cluster: its `/tmp` is node-local, so an install there is
    invisible from the compute node and srun reports
    `execve(): .../python: No such file or directory`. slurmwatch threw that away and
    showed "it may be busy or unreachable - still retrying" — forever, at a failure no
    number of retries can reach. Fifth instance of the misdiagnosis family SW-7, RD-2,
    SW-19 and the missing-Slurm case belong to.
    """

    def test_the_stream_captures_stderr_at_all(self) -> None:
        """A PIPE, not DEVNULL: everything else here depends on it."""
        import asyncio
        import inspect

        src = inspect.getsource(remote.open_stream)
        assert "stderr=asyncio.subprocess.PIPE" in src, src
        assert asyncio  # keep the import meaningful

    @pytest.mark.parametrize(
        ("text", "permanent"),
        [
            ("error: execve(): /tmp/v/bin/python: No such file or directory", True),
            ("srun: error: Access/permission denied for job 42", True),
            ("srun: error: slurm_load_jobs error: Invalid job id specified", True),
            ("srun: error: Invalid user id 4242", True),
            # NOT permanent: this clears when the job's own step releases the CPUs.
            ("srun: error: Unable to create step for job 555: More processors requested", False),
            ("srun: error: Unable to allocate resources: node configuration not available", False),
            ("", False),
        ],
    )
    def test_only_the_unreachable_failures_stop_the_retry(self, text: str, permanent: bool) -> None:
        assert remote.stream_error_is_permanent(text) is permanent

    @pytest.mark.parametrize(
        ("text", "must_contain"),
        [
            ("error: execve(): /x/python: No such file or directory", "compute node can see"),
            ("srun: error: Access/permission denied", "permission denied"),
            ("srun: error: Invalid job id specified", "no longer knows this job"),
            ("srun: error: Unable to create step for job 5", "could not create a step"),
        ],
    )
    def test_the_summary_names_the_cause(self, text: str, must_contain: str) -> None:
        assert must_contain in remote.summarise_stream_error(text, "cn001")

    def test_an_unrecognised_error_is_quoted_not_invented(self) -> None:
        """Better the step's own first line than a guess dressed up as a diagnosis."""
        out = remote.summarise_stream_error("srun: error: something new\nand more", "cn001")
        assert out == "srun: error: something new"

    def test_the_summary_is_ascii_clean_under_ascii(self) -> None:
        out = remote.summarise_stream_error(
            "error: execve(): /x: No such file or directory", "cn001", ascii_mode=True
        )
        assert out.isascii(), out

    @pytest.mark.asyncio
    async def test_reading_the_error_never_blocks_forever(self) -> None:
        """A step that dies without writing anything must not hang the read."""

        class _NoStderr:
            stderr = None

        assert await remote.read_stream_error(_NoStderr()) == ""  # type: ignore[arg-type]
